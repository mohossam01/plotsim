"""plotsim.output — CSV / Parquet / JSONL file writing.

What it does:
    Takes the dict of DataFrames returned by ``plotsim.tables.generate_tables``
    and writes one file per table into the configured output directory, plus a
    YAML copy of the driving config (for round-trippable reproducibility) and
    a human-readable validation report.

    The writer is deliberately the only filesystem-touching module in plotsim.
    Every other module returns DataFrames / reports in memory so callers can
    use the engine programmatically without touching disk.

    ``OutputConfig.format`` selects the per-table encoding. ``"csv"`` is the
    long-standing default; ``"parquet"`` is the M104 addition for columnar
    output (typically 5-10x smaller than the equivalent CSVs, types
    preserved by the format); ``"jsonl"`` (0.6-M16b) writes
    newline-delimited JSON for streaming-ingestion / schema-on-read
    consumers. All branches share the same column-ordering and Int64
    coercion path; only the file extension and the on-disk encoder
    differ. ``config.yaml`` and ``validation_report.txt`` are always
    written as text — they are companions, not table data.

CSV conventions (all tables):
    - encoding: utf-8
    - DataFrame index is NOT written
    - float precision: ``%.4f`` (configurable via ``float_format``)
    - NaN / None renders as the empty string
    - non-numeric fields are quoted (``csv.QUOTE_NONNUMERIC``)
    - integer-typed columns (config ``dtype: int``) render without a ``.0``
      suffix even when pandas has promoted them to float for NaN handling
    - column order follows the table's config: PK(s) first, then FKs in the
      order they appear in config, then remaining columns in config order.
      Any DataFrame columns not declared in the config (for example
      ``stage`` added by ``assign_stages``) are appended last.

Parquet conventions:
    - engine: pyarrow (``plotsim[parquet]`` optional extra). Other engines
      are not supported in V1; ``ImportError`` names the install command.
    - same column ordering and Int64 coercion as CSV
    - DataFrame index is NOT written (``index=False``)
    - compression: snappy (pandas/pyarrow default), explicit for clarity
    - deterministic: same DataFrame + same plotsim/pyarrow versions →
      byte-identical Parquet output across runs

JSONL conventions:
    - one JSON object per line, terminated by ``\\n`` (pinned LF for
      cross-platform byte-identity)
    - writer: ``DataFrame.to_json(orient='records', lines=True,
      date_format='iso', force_ascii=False)``
    - same column ordering and Int64 coercion as CSV; ordered columns
      appear in the same key order inside each JSON object (pandas
      preserves DataFrame column order in ``orient='records'``)
    - NaN / pd.NA / None serialize as JSON ``null``
    - nested ``struct`` cells serialize as native JSON objects;
      ``array`` cells as native JSON arrays — no JSON-string wrapping
    - date columns emit as ISO-8601 strings, not pandas' default
      epoch-ms milliseconds (pinned via ``date_format='iso'``)
    - encoding: utf-8 with ``force_ascii=False`` so unicode characters
      land verbatim rather than as ``\\uXXXX`` escapes

Input:
    ``tables`` (dict[str, pd.DataFrame]), ``PlotsimConfig``, ``ValidationReport``.

Output:
    Side effect: CSV or Parquet files + ``config.yaml`` + ``validation_report.txt``
    on disk. ``write_tables`` returns the output directory ``Path``.
"""

from __future__ import annotations

import csv
import datetime as _dt
import hashlib
import json
from pathlib import Path
from typing import Optional

import pandas as pd

from plotsim.config import (
    FKSource,
    PlotsimConfig,
    PKSource,
    Table,
    dump_config,
    parse_source,
)
from plotsim.denormalize import denormalize_fact_tables
from plotsim.entity_features import (
    ENTITY_FEATURES_BASENAME,
    build_entity_features,
)
from plotsim.log_writer import write_event_logs
from plotsim.holdout import cutoff_period_index, split_fact_tables
from plotsim.manifest import ManifestSchema, write_manifest
from plotsim.quality import apply_issues as _apply_quality_issues
from plotsim.validation import ValidationReport, validate_tables


FLOAT_FORMAT = "%.4f"
NA_REP = ""
CSV_ENCODING = "utf-8"
CONFIG_FILENAME = "config.yaml"
REPORT_FILENAME = "validation_report.txt"

# M104: install hint surfaced when ``output.format == 'parquet'`` but pyarrow
# is missing. Fails fast at the write call so the user sees the issue before
# generation runs all the way through.
_PARQUET_INSTALL_HINT = (
    "Parquet output requires pyarrow. Install it with "
    "`pip install plotsim[parquet]` (or `pip install pyarrow`) and retry."
)


# M121b: sentinel archetype name ``iter_fact_chunks`` yields for
# per_period facts (no entity/archetype axis). The streaming writer
# treats it as a single-row-group write — same on-disk shape as the
# non-streaming path.
_PER_PERIOD_CHUNK_KEY = "__per_period__"


# --- Helpers -----------------------------------------------------------------


def _table_by_name(config: PlotsimConfig, name: str) -> Optional[Table]:
    for tbl in config.tables:
        if tbl.name == name:
            return tbl
    return None


def _ordered_columns(tbl: Table, df_columns: list[str]) -> list[str]:
    """PK → FK (config order) → remaining config columns → extras (e.g. stage).

    Columns declared in config but absent from the DataFrame are dropped (a
    builder chose not to emit them). Columns in the DataFrame but not in
    config go last in DataFrame insertion order — that's where ``stage`` ends
    up.
    """
    pk_cols = [c for c in tbl.primary_key_cols if c in df_columns]
    fk_cols: list[str] = []
    other_cols: list[str] = []
    for col in tbl.columns:
        if col.name in pk_cols:
            continue
        if col.name not in df_columns:
            continue
        parsed = parse_source(col.source)
        if isinstance(parsed, FKSource):
            fk_cols.append(col.name)
        elif isinstance(parsed, PKSource):
            # A column flagged source:"pk" that isn't in primary_key_cols is
            # malformed config — still keep it at the front group.
            pk_cols.append(col.name)
        else:
            other_cols.append(col.name)
    placed = set(pk_cols) | set(fk_cols) | set(other_cols)
    extras = [c for c in df_columns if c not in placed]
    return pk_cols + fk_cols + other_cols + extras


def _coerce_integer_columns(
    tbl: Table,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Ensure columns declared as ``dtype: int`` render without ``.0`` suffix.

    Pandas promotes int columns to float64 when any row is NaN (MCAR noise
    on poisson metrics hits this). ``pd.Int64Dtype`` is the nullable-integer
    equivalent and writes cleanly to CSV: integer cells render as ``5``, null
    cells render as the empty string (via ``na_rep=""``).

    Category B Layer 4: the prior ``df.copy()`` doubled fact-table memory on
    large runs. The Int64 promotion now runs in-place on whatever ``df`` is
    passed — F4 (M102) wraps every external call in a shallow copy at the
    ``write_single_table`` layer so the user's dataframe is not mutated
    (the shallow copy keeps memory flat: one Series-wrapper per column
    instead of duplicating the underlying arrays).
    """
    for col in tbl.columns:
        if col.dtype != "int" or col.name not in df.columns:
            continue
        series = df[col.name]
        if pd.api.types.is_integer_dtype(series.dtype):
            df[col.name] = series.astype("Int64")
            continue
        # Float, object, or bool — round trip through float → Int64, preserving
        # NaN as <NA>.
        coerced = pd.to_numeric(series, errors="coerce")
        df[col.name] = coerced.round().astype("Int64")
    return df


# --- Single-table writer -----------------------------------------------------


def _streaming_parquet_eligible(config: Optional[PlotsimConfig]) -> bool:
    """Whether the streaming Parquet path applies to ``config``'s fact writes.

    Triggers when output format is parquet AND the resolved generation
    mode is vectorized (M121b). Serial mode and CSV output keep the
    pre-mission unified-DataFrame write path. Engine-direct configs
    that never opted into vectorized mode are unaffected because their
    ``generation_mode`` defaults to ``"serial"``.

    0.6-M16a: partitioning wins over streaming. When
    ``output.partition_by`` is set, every fact table routes through
    ``write_single_table``'s partitioned branch (``write_to_dataset``);
    the per-archetype row-group writer would have to compose with
    Hive-style directory layout, which ``pyarrow`` does not support in
    one call. Streaming is an internal optimization; partitioning is
    the user-visible knob — partitioning wins.
    """
    if config is None:
        return False
    if _resolve_output_format(config) != "parquet":
        return False
    if _resolve_partition_by(config) is not None:
        return False
    # Local import: ``plotsim.tables`` imports ``plotsim.output``
    # transitively for non-streaming writes; importing at module
    # scope would create a cycle.
    from plotsim.tables import _resolve_generation_mode

    return _resolve_generation_mode(config) == "vectorized"


def _streaming_fact_table_names(config: PlotsimConfig) -> set[str]:
    """Set of fact-table names the streaming path writes per row group.

    Only ``per_entity_per_period`` facts get the per-archetype row-group
    decomposition; ``per_period`` facts (no entity/archetype axis) are
    written as a single row group via the same ParquetWriter so the
    on-disk layout matches the non-streaming output exactly except for
    row group boundaries.

    0.6-M14c: fact tables with nested (struct / array) columns route
    through the standard ``write_single_table`` path, which builds an
    explicit pyarrow schema from the column config. The streaming
    writer auto-infers schema per chunk and would either lose nesting
    or trip on dtype drift across chunks.
    """
    return {
        tbl.name
        for tbl in config.tables
        if tbl.type == "fact" and not any(c.dtype in ("struct", "array") for c in tbl.columns)
    }


def _write_streaming_parquet_facts(
    config: PlotsimConfig,
    fact_tables: dict[str, pd.DataFrame],
    output_dir: Path,
) -> set[str]:
    """Write fact tables as per-archetype Parquet row groups.

    Uses ``pyarrow.parquet.ParquetWriter`` to stream each archetype
    chunk yielded by ``plotsim.tables.iter_fact_chunks`` as one row
    group. Per_period facts (no entity axis) write as a single row
    group under the sentinel chunk key. Returns the set of fact-table
    names this function wrote so the caller knows to skip them in the
    standard ``write_single_table`` loop.

    Memory contract: peak transient pyarrow buffer is bounded by the
    largest single-archetype chunk size, not the unified DataFrame.
    The unified DataFrame is still resident in memory because
    downstream consumers (``attach_dim_row_id_to_facts``,
    ``assign_stages``, ``build_event_tables``, ``build_bridge_tables``,
    ``entity_features``) all consumed it before the writer ran — so
    the M121b memory win is the *additional* peak from
    ``to_parquet``'s pyarrow conversion of the full DataFrame, which
    the architecture-scalability doc identified as the dominant
    transient overhead.

    The streaming branch only fires for fact tables. Dim/event/bridge
    tables continue through ``write_single_table``'s single-shot
    ``to_parquet`` path.
    """
    _check_parquet_engine_available()
    import pyarrow as pa
    import pyarrow.parquet as pq

    from plotsim.tables import iter_fact_chunks

    streaming_names = _streaming_fact_table_names(config)
    streaming_facts = {name: df for name, df in fact_tables.items() if name in streaming_names}
    if not streaming_facts:
        return set()

    # Pre-process columns once per fact table — column reordering +
    # nullable-Int64 coercion mirror what ``write_single_table`` would
    # do, hoisted up so the schema seen by ParquetWriter matches the
    # non-streaming output exactly.
    prepared: dict[str, pd.DataFrame] = {}
    for name, df in streaming_facts.items():
        tbl = _table_by_name(config, name)
        if tbl is not None:
            ordered = _ordered_columns(tbl, list(df.columns))
            prep = df.loc[:, ordered].copy(deep=False)
            _coerce_integer_columns(tbl, prep)
        else:
            prep = df
        prepared[name] = prep

    # ParquetWriter requires a schema upfront, but empty-DataFrame
    # schema inference returns ``null`` type for object/string columns
    # (no values to type-resolve from). Defer writer creation until
    # the first non-empty chunk for each fact table; infer schema from
    # that first chunk's pyarrow conversion and reuse it for every
    # subsequent row group. Subsequent chunks are slices of the same
    # prepared DataFrame, so dtypes are already consistent — pyarrow
    # accepts them via the ``schema=`` kwarg.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_dir = output_dir.resolve()

    # Pre-resolve and validate paths up front so SEC-02 sandbox
    # rejection happens before any RNG work runs (none here, but
    # the fail-fast contract matches the single-table writer).
    paths: dict[str, Path] = {}
    for name in prepared:
        path = output_dir / f"{name}.parquet"
        if path.resolve().parent != resolved_dir:
            raise ValueError(
                f"_write_streaming_parquet_facts: table name {name!r} "
                f"resolves outside output_dir {str(output_dir)!r}; names "
                f"must be SQL-safe identifiers (no path separators, no ..)"
            )
        paths[name] = path

    writers: dict[str, tuple] = {}
    try:
        # Stream chunks. ``iter_fact_chunks`` yields chunks for the SAME
        # set of fact tables we filtered into ``prepared`` (per_period
        # facts come through under the sentinel chunk key, written as a
        # single row group).
        for _arch_name, chunk in iter_fact_chunks(config, prepared):
            for fact_name, df_chunk in chunk.items():
                if fact_name not in paths:
                    continue  # defensive — shouldn't happen given filter
                pa_table = pa.Table.from_pandas(
                    df_chunk,
                    preserve_index=False,
                )
                if fact_name not in writers:
                    schema = pa_table.schema
                    writer = pq.ParquetWriter(
                        paths[fact_name],
                        schema,
                        compression="snappy",
                    )
                    writers[fact_name] = (writer, schema)
                else:
                    # Reuse the first-chunk schema; cast in case pyarrow
                    # inferred a slightly narrower type for this chunk
                    # (e.g., a chunk with only non-null values where the
                    # first chunk had nulls). ``cast`` is a no-op when
                    # types already match.
                    schema = writers[fact_name][1]
                    if pa_table.schema != schema:
                        pa_table = pa_table.cast(schema)
                writers[fact_name][0].write_table(pa_table)
    finally:
        for writer, _schema in writers.values():
            writer.close()

    return set(prepared.keys())


def _resolve_output_format(config: Optional[PlotsimConfig]) -> str:
    """Return ``'csv'``, ``'parquet'``, or ``'jsonl'`` based on config;
    default to CSV.

    Programmatic callers that pass ``config=None`` or a stub object
    without an ``output`` attribute (e.g. unit tests of
    ``write_single_table`` against an ad-hoc DataFrame) get CSV — the
    long-standing behavior — preserved. The defensive ``getattr`` chain
    keeps that contract intact while the YAML-loaded ``PlotsimConfig``
    surface drives the parquet / jsonl branches.
    """
    if config is None:
        return "csv"
    output_cfg = getattr(config, "output", None)
    if output_cfg is None:
        return "csv"
    return getattr(output_cfg, "format", "csv")


def _resolve_partition_by(config: Optional[PlotsimConfig]) -> Optional[str]:
    """Return ``output.partition_by`` if set, else ``None``.

    Mirrors ``_resolve_output_format``'s defensive ``getattr`` chain so
    stub configs and ``None`` inputs (used by programmatic callers of
    ``write_single_table``) keep the single-file Parquet path.
    """
    if config is None:
        return None
    output_cfg = getattr(config, "output", None)
    if output_cfg is None:
        return None
    return getattr(output_cfg, "partition_by", None)


def _resolve_partition_column_for_table(
    partition_by: str,
    tbl: Optional[Table],
    df: pd.DataFrame,
) -> Optional[str]:
    """0.6-M19 Fix 5: return the actual column name on ``tbl`` to
    partition by, honoring the literal-then-FK-target precedence used
    by ``PlotsimConfig._validate_partition_column``.

    Literal name match wins: if ``partition_by`` is itself a column on
    the DataFrame, return it unchanged. Otherwise look for a column on
    ``tbl`` whose source is ``fk:<dim>.<partition_by>`` and return that
    column's local name — that's how ``partition_by: date_key`` lands
    on a table whose date column is called ``order_date``.

    Returns ``None`` when neither resolution applies; callers fall back
    to single-file Parquet (the "partition where applicable" semantic
    OutputConfig.partition_by already declared).
    """
    if partition_by in df.columns:
        return partition_by
    if tbl is None:
        return None
    for col in tbl.columns:
        parsed = parse_source(col.source)
        if isinstance(parsed, FKSource) and parsed.column == partition_by:
            if col.name in df.columns:
                return col.name
    return None


_FORMAT_EXTENSIONS: dict[str, str] = {
    "csv": "csv",
    "parquet": "parquet",
    "jsonl": "jsonl",
}


def _extension_for_format(output_format: str) -> str:
    """Return the on-disk file extension for ``output_format``.

    Single source of truth so the table writer, the entity-features
    writer, and any future companion writer all agree on the suffix.
    Unknown formats fall back to ``"csv"`` to preserve the legacy
    ``config=None`` contract on ``write_single_table``.
    """
    return _FORMAT_EXTENSIONS.get(output_format, "csv")


def _write_jsonl(df: pd.DataFrame, path: Path) -> None:
    """Write ``df`` as newline-delimited JSON to ``path``.

    Pinned options:

    - ``orient='records'``, ``lines=True`` — one JSON object per row,
      each row self-contained (the streaming-ingestion contract).
    - ``date_format='iso'`` — date / datetime columns emit as ISO-8601
      strings; pandas defaults to epoch-ms for ``orient='records'``,
      which is unfriendly for hand-inspection and most downstream
      JSONL consumers.
    - ``force_ascii=False`` — unicode characters land verbatim; matches
      the utf-8 contract used for CSV.

    NaN / ``pd.NA`` / ``None`` serialise as JSON ``null``. Nested
    ``struct`` / ``array`` cells (which are Python dicts / lists in
    the DataFrame) serialise as native JSON objects / arrays — no
    JSON-string wrapping. The DataFrame's column order is preserved
    in the JSON key order of each row (pandas guarantee for
    ``orient='records'``).
    """
    df.to_json(
        path,
        orient="records",
        lines=True,
        date_format="iso",
        force_ascii=False,
    )


def _check_parquet_engine_available() -> None:
    """Raise ImportError with the install hint if ``pyarrow`` is missing.

    Called at the top of every Parquet write path so the failure surfaces
    before the writer touches disk.
    """
    try:
        import pyarrow  # noqa: F401  (import-only check)
    except ImportError as exc:
        raise ImportError(_PARQUET_INSTALL_HINT) from exc


def write_single_table(
    name: str,
    df: pd.DataFrame,
    output_dir: Path,
    config: Optional[PlotsimConfig] = None,
    float_format: str = FLOAT_FORMAT,
) -> Path:
    """Write one DataFrame to ``<output_dir>/<name>.<csv|parquet>``.

    If ``config`` is provided and declares ``name``, columns are reordered
    PK → FK → others (config order) and ``dtype: int`` columns are coerced
    to nullable integer so the output has no ``.0`` suffixes. Without
    config, the DataFrame is written as-is with the same encoding / quoting
    / NaN conventions.

    File extension and encoder are chosen by ``config.output.format``:
    ``"csv"`` (default) or ``"parquet"``. Parquet writes require
    ``pyarrow``; an explicit ImportError with install hint is raised when
    the engine is missing.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_format = _resolve_output_format(config)
    extension = _extension_for_format(output_format)
    path = output_dir / f"{name}.{extension}"

    # SEC-02 defense-in-depth: ``Table.name`` / ``Column.name`` are regex-
    # validated at config load, but programmatic callers can bypass that by
    # passing ``write_single_table`` a crafted ``name``. A table named
    # ``"../../../etc/shadow"`` would resolve outside ``output_dir`` — reject
    # before the writer touches disk.
    resolved_dir = output_dir.resolve()
    if path.resolve().parent != resolved_dir:
        raise ValueError(
            f"write_single_table: table name {name!r} resolves outside "
            f"output_dir {str(output_dir)!r}; names must be SQL-safe "
            f"identifiers (no path separators, no ..)"
        )

    tbl = _table_by_name(config, name) if config is not None else None
    to_write = df
    if tbl is not None:
        # F4 (M102): take a shallow copy before `_coerce_integer_columns`
        # rewrites integer-column references. Shallow because we only
        # reassign whole columns (no array-data duplication), so the user's
        # `tables[name]` keeps its original Series objects and dtypes —
        # closing the silent in-place-mutation that pre-fix changed
        # `tables[name][int_col]` underneath the caller during write.
        ordered = _ordered_columns(tbl, list(df.columns))
        to_write = df.loc[:, ordered].copy(deep=False)
        _coerce_integer_columns(tbl, to_write)

    if output_format == "parquet":
        _check_parquet_engine_available()
        # 0.6-M16a: when ``output.partition_by`` is set AND a matching
        # column is present in this DataFrame, route through the
        # partitioned writer. Tables that don't carry the partition
        # column fall back to the single-file Parquet path on the next
        # branch — this is the "partition where applicable" semantic
        # declared in ``OutputConfig.partition_by``.
        #
        # 0.6-M19 Fix 5: column resolution honors FK-target fallback.
        # ``partition_by: date_key`` lands on a fact whose date column
        # is named ``order_date`` (FK target ``dim_date.date_key``);
        # the partition directories then use that local column name.
        partition_by = _resolve_partition_by(config)
        if partition_by is not None:
            actual_col = _resolve_partition_column_for_table(partition_by, tbl, to_write)
            if actual_col is not None:
                return _write_partitioned_parquet(to_write, tbl, output_dir, name, actual_col)
        # 0.6-M14c: nested columns (struct / array) need an explicit
        # pyarrow schema so dict / list cells write as native nested
        # types instead of being inferred as opaque object columns.
        # Other columns let pyarrow infer; only nested columns get the
        # explicit treatment.
        if tbl is not None and _table_has_nested_columns(tbl):
            _write_parquet_with_nested_schema(to_write, tbl, path)
        else:
            to_write.to_parquet(
                path,
                engine="pyarrow",
                index=False,
                compression="snappy",
            )
    elif output_format == "jsonl":
        # 0.6-M16b: newline-delimited JSON. Nested struct / array cells
        # land as native JSON objects / arrays (no _serialise_nested_for_csv
        # wrapping); date columns serialise as ISO-8601 strings rather
        # than pandas' default epoch-ms milliseconds for orient=records.
        _write_jsonl(to_write, path)
    else:
        # 0.6-M14c: nested columns serialise via ``json.dumps`` for CSV
        # output. Pandas ``to_csv`` would otherwise call ``str(dict)`` /
        # ``str(list)`` and produce Python literal syntax (single quotes
        # around keys), which round-trips through ``json.loads`` poorly.
        if tbl is not None and _table_has_nested_columns(tbl):
            to_write = _serialise_nested_for_csv(to_write, tbl)
        # Pin LF: pandas defaults to os.linesep, which produces CRLF on
        # Windows and LF on Linux — breaks byte-identical fixture
        # comparisons across platforms.
        to_write.to_csv(
            path,
            index=False,
            encoding=CSV_ENCODING,
            quoting=csv.QUOTE_NONNUMERIC,
            float_format=float_format,
            na_rep=NA_REP,
            lineterminator="\n",
        )
    return path


# --- 0.6-M14c: nested column output helpers ---------------------------------


def _table_has_nested_columns(tbl: Table) -> bool:
    """True when any column on ``tbl`` declares dtype ``struct`` or ``array``."""
    return any(col.dtype in ("struct", "array") for col in tbl.columns)


def _serialise_nested_for_csv(df: pd.DataFrame, tbl: Table) -> pd.DataFrame:
    """Replace nested cells (dict / list) on a copy of ``df`` with JSON strings.

    Only struct/array columns declared on the table config are
    transformed. Other columns pass through untouched. NaN / None
    cells are preserved (CSV writer renders them as the empty
    ``na_rep`` string), so the round-trip is ``json.loads`` on a
    non-empty cell.
    """
    out = df.copy(deep=False)
    for col in tbl.columns:
        if col.dtype not in ("struct", "array") or col.name not in out.columns:
            continue
        out[col.name] = out[col.name].map(
            lambda v: json.dumps(v) if v is not None and not _is_nan_scalar(v) else None
        )
    return out


def _is_nan_scalar(v: object) -> bool:
    """True for float NaN scalars; False for dict / list / other values."""
    try:
        return bool(pd.isna(v)) and not isinstance(v, (dict, list))
    except (TypeError, ValueError):
        return False


def _build_nested_pa_schema(df: pd.DataFrame, tbl: Table):
    """Build a pyarrow schema mapping nested config columns to native
    struct / list types and inferring other columns from pandas dtypes.

    Shared between the single-file Parquet path
    (``_write_parquet_with_nested_schema``) and the partitioned dataset
    path (``_write_partitioned_parquet``) so nested-column round-trip
    behaves identically under both layouts.
    """
    import pyarrow as pa

    fields: list[pa.Field] = []
    for col_name in df.columns:
        tbl_col = next((c for c in tbl.columns if c.name == col_name), None)
        if tbl_col is None or tbl_col.dtype not in ("struct", "array"):
            inferred = pa.array(df[col_name]).type
            fields.append(pa.field(col_name, inferred))
            continue
        if tbl_col.dtype == "struct":
            assert tbl_col.nested_schema is not None
            struct_fields = [
                pa.field(field_name, _pa_primitive(field_type))
                for field_name, field_type in tbl_col.nested_schema.items()
            ]
            fields.append(pa.field(col_name, pa.struct(struct_fields)))
        else:  # array
            assert tbl_col.array_element_type is not None
            fields.append(pa.field(col_name, pa.list_(_pa_primitive(tbl_col.array_element_type))))
    return pa.schema(fields)


def _write_parquet_with_nested_schema(df: pd.DataFrame, tbl: Table, path: Path) -> None:
    """Write ``df`` to Parquet using an explicit pyarrow schema for nested columns.

    The struct schema fields and array element types come from the
    column config (``nested_schema`` / ``array_element_type``).
    Non-nested columns are inferred from the pandas dtype.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = _build_nested_pa_schema(df, tbl)
    table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
    pq.write_table(table, path, compression="snappy")


def _write_partitioned_parquet(
    df: pd.DataFrame,
    tbl: Optional[Table],
    output_dir: Path,
    name: str,
    partition_by: str,
) -> Path:
    """Write ``df`` as a Hive-style partitioned Parquet directory under
    ``<output_dir>/<name>/``.

    Uses ``pyarrow.parquet.write_to_dataset`` with
    ``partition_cols=[partition_by]``; the resulting layout is
    ``<output_dir>/<name>/<partition_by>=<value>/<file>.parquet``.
    File naming inside each partition is pyarrow's default — names are
    not part of the on-disk contract (callers iterate the dataset).

    Nested-column tables (``struct`` / ``array``) reuse
    ``_build_nested_pa_schema`` so the column types survive partitioning
    exactly as they do under the single-file Parquet path.
    """
    _check_parquet_engine_available()
    import pyarrow as pa
    import pyarrow.parquet as pq

    dataset_dir = output_dir / name
    resolved_dir = output_dir.resolve()
    if dataset_dir.resolve().parent != resolved_dir:
        raise ValueError(
            f"_write_partitioned_parquet: table name {name!r} resolves "
            f"outside output_dir {str(output_dir)!r}; names must be "
            f"SQL-safe identifiers (no path separators, no ..)"
        )
    dataset_dir.mkdir(parents=True, exist_ok=True)

    if tbl is not None and _table_has_nested_columns(tbl):
        schema = _build_nested_pa_schema(df, tbl)
        pa_table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
    else:
        pa_table = pa.Table.from_pandas(df, preserve_index=False)

    pq.write_to_dataset(
        pa_table,
        root_path=str(dataset_dir),
        partition_cols=[partition_by],
        compression="snappy",
    )
    return dataset_dir


def _pa_primitive(type_word: str):
    """Map a nested-primitive type word to a pyarrow primitive type."""
    import pyarrow as pa

    return {
        "int": pa.int64(),
        "float": pa.float64(),
        "string": pa.string(),
        "boolean": pa.bool_(),
    }[type_word]


# --- 0.6-M16c: SQL dump writer ---------------------------------------------


SQL_FILENAME = "data.sql"
_SQL_INSERT_BATCH_SIZE = 100


def _sql_quote_identifier(name: str, dialect: str) -> str:
    """Wrap ``name`` in the dialect's identifier quote character.

    PG and SQLite use double quotes; MySQL uses backticks. Embedded
    quote characters in the identifier are doubled (SQL standard) —
    plotsim's column-name regex disallows them in practice, but the
    doubling is cheap and keeps the helper safe for any caller.
    """
    if dialect == "mysql":
        return "`" + name.replace("`", "``") + "`"
    return '"' + name.replace('"', '""') + '"'


def _sql_quote_string(s: str) -> str:
    """SQL string literal: single quotes with embedded ``'`` doubled.

    All three target dialects (PG / MySQL / SQLite) accept the SQL
    standard doubled-single-quote escaping, so the function is
    dialect-agnostic.
    """
    return "'" + s.replace("'", "''") + "'"


def _sql_type_for_dialect(col_dtype: str, dialect: str, *, is_pk_or_fk: bool) -> str:
    """Map a plotsim ``Column.dtype`` to the dialect's SQL type word.

    Type words come from the V1 dtype set (``int`` / ``float`` /
    ``string`` / ``id`` / ``boolean`` / ``date`` / ``struct`` / ``array``).
    Nested types serialise as JSON in TEXT cells — all three dialects
    accept TEXT, though PG has a native JSONB type that operators can
    swap in post-import if they prefer.

    MySQL exception: TEXT / BLOB columns cannot be primary keys
    (or foreign keys) without a key prefix, so when ``is_pk_or_fk`` is
    true and the dtype is string-like, the MySQL mapping switches to
    ``VARCHAR(255)``.
    """
    if dialect == "postgresql":
        mapping = {
            "int": "INTEGER",
            "float": "NUMERIC",
            "string": "TEXT",
            "id": "TEXT",
            "boolean": "BOOLEAN",
            "date": "TIMESTAMP",
            "struct": "TEXT",
            "array": "TEXT",
        }
    elif dialect == "mysql":
        mapping = {
            "int": "INT",
            "float": "DOUBLE",
            "string": "VARCHAR(255)" if is_pk_or_fk else "TEXT",
            "id": "VARCHAR(255)",
            "boolean": "TINYINT(1)",
            "date": "TIMESTAMP",
            "struct": "TEXT",
            "array": "TEXT",
        }
    else:  # sqlite
        mapping = {
            "int": "INTEGER",
            "float": "REAL",
            "string": "TEXT",
            "id": "TEXT",
            "boolean": "INTEGER",
            "date": "TEXT",
            "struct": "TEXT",
            "array": "TEXT",
        }
    return mapping.get(col_dtype, "TEXT")


def _infer_dtype_from_series(series: pd.Series) -> str:
    """Best-effort plotsim-dtype inference for columns without a
    ``Column`` config entry (denormalized wide tables, holdout splits,
    derived columns like ``stage`` / SCD2 audit / CDC audit). Falls
    back to ``string`` for object / mixed cells; the SQL writer treats
    those as TEXT in every dialect.
    """
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_integer_dtype(series):
        return "int"
    if pd.api.types.is_float_dtype(series):
        return "float"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "date"
    return "string"


def _sql_format_value(cell: object, dialect: str) -> str:
    """Format a single DataFrame cell as a SQL literal.

    NaN / pd.NA / None → ``NULL``. Booleans render as ``TRUE`` /
    ``FALSE`` under PG and ``1`` / ``0`` under MySQL + SQLite. Numbers
    render verbatim via ``repr`` (float) / ``str`` (int) — no thousands
    separators, no scientific notation rewrites. Dict / list cells
    (struct / array dtype) serialise via ``json.dumps`` into a quoted
    string. Date / datetime / Timestamp cells emit as ISO-8601 strings
    (``YYYY-MM-DD HH:MM:SS`` for datetimes, ``YYYY-MM-DD`` for dates).
    Everything else falls through to ``str(cell)`` quoted.
    """
    if cell is None or cell is pd.NA:
        return "NULL"
    if isinstance(cell, float) and pd.isna(cell):
        return "NULL"
    if isinstance(cell, bool):
        if dialect == "postgresql":
            return "TRUE" if cell else "FALSE"
        return "1" if cell else "0"
    if isinstance(cell, int):
        return str(cell)
    if isinstance(cell, float):
        return repr(cell)
    if isinstance(cell, (dict, list)):
        return _sql_quote_string(json.dumps(cell, ensure_ascii=False))
    if isinstance(cell, pd.Timestamp):
        if pd.isna(cell):
            return "NULL"
        return _sql_quote_string(cell.isoformat(sep=" "))
    if hasattr(cell, "isoformat"):
        return _sql_quote_string(cell.isoformat())
    # Defensive: pandas Int64 NA arrives here as <NA>.
    try:
        if pd.isna(cell):
            return "NULL"
    except (TypeError, ValueError):
        pass
    return _sql_quote_string(str(cell))


def _sql_table_order(config: PlotsimConfig) -> list[str]:
    """Return table names in dependency-safe write order.

    Star schema: dimensions first (no FK dependencies), then every
    other table type (fact / event / bridge) in config declaration
    order. Within each group, declaration order is preserved so the
    SQL dump replays with stable ordering across runs.
    """
    dims = [t.name for t in config.tables if t.type == "dim"]
    others = [t.name for t in config.tables if t.type != "dim"]
    return dims + others


def _sql_column_defs(
    df: pd.DataFrame,
    tbl: Optional[Table],
    dialect: str,
    *,
    pk_cols: set[str],
    fk_cols: set[str],
) -> list[str]:
    """Build one ``"col" TYPE`` line per DataFrame column.

    SQL types come from the runtime pandas dtype rather than the
    plotsim engine dtype — the engine's ``id`` / ``string`` distinction
    doesn't survive to disk, and the actual cell shape (``int64`` vs
    ``object``) is what the target database needs to accept. The one
    exception is ``struct`` / ``array``: those columns hold dict /
    list cells that ``_sql_format_value`` JSON-serializes, so they
    must be declared as TEXT regardless of pandas dtype.
    """
    config_by_name = {c.name: c for c in tbl.columns} if tbl is not None else {}
    lines: list[str] = []
    for col_name in df.columns:
        is_key = col_name in pk_cols or col_name in fk_cols
        cfg_col = config_by_name.get(col_name)
        dtype: str
        if cfg_col is not None and cfg_col.dtype in ("struct", "array"):
            dtype = cfg_col.dtype
        else:
            dtype = _infer_dtype_from_series(df[col_name])
        type_word = _sql_type_for_dialect(dtype, dialect, is_pk_or_fk=is_key)
        lines.append(f"  {_sql_quote_identifier(col_name, dialect)} {type_word}")
    return lines


def _key_is_unique(df: pd.DataFrame, cols: list[str]) -> bool:
    """True when ``df`` has no duplicate rows under the ``cols`` subset.

    Used to decide whether a ``PRIMARY KEY`` or ``FOREIGN KEY``
    constraint can actually be emitted: SCD2 dims and quality-injected
    duplicates would make a strict UNIQUE constraint fail at
    replay-time, so the SQL writer falls back to bare ``CREATE TABLE``
    (no constraints) when the data doesn't permit one.
    """
    if df.empty or not cols:
        return True
    missing = [c for c in cols if c not in df.columns]
    if missing:
        return False
    return not df.duplicated(subset=cols).any()


def _sql_create_table(
    name: str,
    df: pd.DataFrame,
    tbl: Optional[Table],
    dialect: str,
    *,
    with_constraints: bool,
    all_tables: Optional[dict[str, pd.DataFrame]] = None,
) -> str:
    """Emit a ``CREATE TABLE`` statement for ``df``.

    When ``with_constraints`` is true and ``tbl`` is provided:

    - ``PRIMARY KEY (...)`` is added when the actual data permits.
      SCD2 dims (detected by a ``dim_row_id`` surrogate column) use
      ``dim_row_id`` as the PK. Tables whose natural PK has duplicate
      rows (quality-injected ``duplicate_rows`` / ``volume_anomaly``)
      are emitted without a PK constraint — the dump still replays,
      and the consumer can add constraints after import if needed.
    - ``FOREIGN KEY (col) REFERENCES dim(pk)`` is added for every
      column whose ``source`` parses as ``FKSource`` AND whose target
      dim's referenced column is unique in the actual data. SCD2 dims
      and quality-injected duplicates make the natural FK column
      non-unique, so the constraint is skipped in those cases.

    Wide-table / holdout sidecars pass ``with_constraints=False``
    because their multi-dim shape doesn't fit the FK model.
    """
    pk_cols_actual: list[str] = []
    fk_pairs: list[tuple[str, str, str]] = []

    if with_constraints and tbl is not None:
        # ``dim_row_id`` is the SCD2 surrogate on dim tables. Facts
        # also carry ``dim_row_id`` as a join surrogate when they FK
        # to an SCD2 dim, but the fact's true PK remains the natural
        # composite — only dim tables route to the surrogate PK.
        if tbl.type == "dim" and "dim_row_id" in df.columns:
            pk_cols_actual = ["dim_row_id"]
        elif _key_is_unique(df, list(tbl.primary_key_cols)):
            pk_cols_actual = list(tbl.primary_key_cols)

        for col in tbl.columns:
            if col.name not in df.columns:
                continue
            parsed = parse_source(col.source)
            if not isinstance(parsed, FKSource):
                continue
            target_table = parsed.table
            target_col = parsed.column
            if all_tables is not None and target_table in all_tables:
                target_df = all_tables[target_table]
                if "dim_row_id" in target_df.columns:
                    # SCD2 target — natural FK column is not unique
                    # (multiple versioned rows per natural key).
                    continue
                if not _key_is_unique(target_df, [target_col]):
                    continue
            fk_pairs.append((col.name, target_table, target_col))

    fk_col_set = {c for c, _, _ in fk_pairs}
    body_lines = _sql_column_defs(df, tbl, dialect, pk_cols=set(pk_cols_actual), fk_cols=fk_col_set)

    if pk_cols_actual:
        pk_list = ", ".join(_sql_quote_identifier(c, dialect) for c in pk_cols_actual)
        body_lines.append(f"  PRIMARY KEY ({pk_list})")

    for fk_col, ref_table, ref_col in fk_pairs:
        body_lines.append(
            f"  FOREIGN KEY ({_sql_quote_identifier(fk_col, dialect)}) "
            f"REFERENCES {_sql_quote_identifier(ref_table, dialect)}"
            f"({_sql_quote_identifier(ref_col, dialect)})"
        )

    qname = _sql_quote_identifier(name, dialect)
    body = ",\n".join(body_lines)
    return f"CREATE TABLE {qname} (\n{body}\n);"


def _sql_inserts(
    name: str,
    df: pd.DataFrame,
    dialect: str,
    *,
    batch_size: int = _SQL_INSERT_BATCH_SIZE,
) -> list[str]:
    """Emit batched multi-row ``INSERT`` statements for ``df``.

    Each statement covers up to ``batch_size`` rows (default 100,
    matched to the mission spec). Empty DataFrames produce no
    statements at all — a ``CREATE TABLE`` with zero rows is the
    natural representation.
    """
    if df.empty:
        return []
    qname = _sql_quote_identifier(name, dialect)
    qcols = ", ".join(_sql_quote_identifier(c, dialect) for c in df.columns)
    statements: list[str] = []
    # ``itertuples(index=False)`` preserves dtypes (no Series-wrapping
    # cost per cell) and gives us a stable iteration order matching
    # ``df.columns``.
    rows = list(df.itertuples(index=False, name=None))
    for batch_start in range(0, len(rows), batch_size):
        chunk = rows[batch_start : batch_start + batch_size]
        value_lines = [
            "  (" + ", ".join(_sql_format_value(c, dialect) for c in row) + ")" for row in chunk
        ]
        statements.append(f"INSERT INTO {qname} ({qcols}) VALUES\n" + ",\n".join(value_lines) + ";")
    return statements


def _sql_header(dialect: str, config: PlotsimConfig) -> str:
    """SQL file preamble: dialect label + replay-command hint."""
    from plotsim import __version__ as _plotsim_version

    replay = {
        "postgresql": "psql -d <database> < data.sql",
        "mysql": "mysql <database> < data.sql",
        "sqlite": "sqlite3 <database.sqlite> < data.sql",
    }[dialect]
    table_list = ", ".join(t.name for t in config.tables)
    return (
        f"-- Generated by plotsim {_plotsim_version}\n"
        f"-- Dialect: {dialect}\n"
        f"-- Tables: {table_list}\n"
        f"-- Replay: {replay}\n"
    )


def _write_sql_dump(
    tables_to_write: dict[str, pd.DataFrame],
    config: PlotsimConfig,
    output_dir: Path,
    *,
    wide_tables: Optional[dict[str, pd.DataFrame]] = None,
    holdout_splits: Optional[dict[str, tuple[pd.DataFrame, pd.DataFrame]]] = None,
) -> Path:
    """Write every fact / dim / event / bridge table — plus optional
    denormalized wide and holdout-split sidecars — to a single
    ``data.sql`` file inside ``output_dir``.

    Star schema (dims → others) is emitted first with PK + FK
    constraints. Wide tables and holdout splits follow as trailing
    blocks without FK constraints (operator-stated scope decision at
    M16c kickoff: their multi-dim / partial-fact shape doesn't fit
    the FK model but should still ship in the single-file deliverable).
    """
    dialect = config.output.sql_dialect
    path = output_dir / SQL_FILENAME

    chunks: list[str] = [_sql_header(dialect, config)]
    for name in _sql_table_order(config):
        df = tables_to_write[name]
        tbl = _table_by_name(config, name)
        chunks.append(
            _sql_create_table(
                name,
                df,
                tbl,
                dialect,
                with_constraints=True,
                all_tables=tables_to_write,
            )
        )
        chunks.extend(_sql_inserts(name, df, dialect))

    if wide_tables:
        for wide_name, wide_df in wide_tables.items():
            chunks.append(
                _sql_create_table(wide_name, wide_df, None, dialect, with_constraints=False)
            )
            chunks.extend(_sql_inserts(wide_name, wide_df, dialect))

    if holdout_splits:
        for fact_name, (train_df, holdout_df) in holdout_splits.items():
            for suffix, split_df in (("train", train_df), ("holdout", holdout_df)):
                full_name = f"{fact_name}_{suffix}"
                chunks.append(
                    _sql_create_table(full_name, split_df, None, dialect, with_constraints=False)
                )
                chunks.extend(_sql_inserts(full_name, split_df, dialect))

    path.write_text("\n\n".join(chunks) + "\n", encoding="utf-8")
    return path


# --- Config copy -------------------------------------------------------------


def write_config_copy(
    config: PlotsimConfig,
    output_dir: Path,
) -> Path:
    """Serialize ``config`` back to YAML at ``<output_dir>/config.yaml``.

    Round-trips through ``plotsim.config.dump_config``; the emitted file is
    valid input to ``load_config`` and regenerates the same dataset under the
    same plotsim version.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / CONFIG_FILENAME
    path.write_text(dump_config(config), encoding=CSV_ENCODING)
    return path


# --- Validation report -------------------------------------------------------


def _config_fingerprint(config: PlotsimConfig) -> str:
    """16-char SHA-256 prefix of the JSON-serialized config dump.

    F5 (M102): provides a deterministic identifier for the validation report
    when no wall-clock ``generated_at`` is supplied. ``model_dump(mode='json')``
    emits dates / Decimals / etc. as primitive types; ``sort_keys=True`` and
    ``default=str`` make the serialization order-stable for any field
    pydantic v2 might still leave as a Python object after the JSON-mode
    dump. Same config + same plotsim version → same fingerprint.
    """
    payload = config.model_dump(mode="json")
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _format_report(
    report: ValidationReport,
    generated_at: Optional[_dt.datetime] = None,
    config: Optional[PlotsimConfig] = None,
) -> str:
    errors = report.errors
    warnings = report.warnings
    status = "VALID" if report.ok else "INVALID"
    if generated_at is not None:
        stamp = generated_at.isoformat(timespec="seconds")
    elif config is not None:
        # F5 (M102): deterministic by default — same config + same seed →
        # byte-identical report. CLI passes generated_at to keep the
        # wall-clock timestamp visible to operators (see cli.cmd_run).
        stamp = f"deterministic (config-sha256[:16]={_config_fingerprint(config)})"
    else:
        stamp = "deterministic"
    header = [
        "Plotsim Validation Report",
        "==========================",
        f"Generated: {stamp}",
        f"Errors: {len(errors)} | Warnings: {len(warnings)} | Total: {len(report.issues)}",
        f"Status: {status}",
        "",
    ]
    if not report.issues:
        header.append("All checks passed cleanly.")
        return "\n".join(header) + "\n"

    lines: list[str] = list(header)
    for issue in report.issues:
        tag = "ERROR" if issue.severity == "error" else "WARN "
        table = issue.table or "-"
        lines.append(f"[{tag}] {issue.check} ({table}) — {issue.message}")
        if issue.details:
            for key, value in issue.details.items():
                lines.append(f"        {key}: {value}")
    return "\n".join(lines) + "\n"


def write_validation_report(
    report: ValidationReport,
    output_dir: Path,
    generated_at: Optional[_dt.datetime] = None,
    config: Optional[PlotsimConfig] = None,
) -> Path:
    """Write ``report`` as a human-readable text file.

    Header shows error/warning counts and overall VALID/INVALID status;
    body is one line per issue with the check name, table (or ``-``),
    the message, and a details block.

    F5 (M102): the ``Generated:`` line renders the supplied ``generated_at``
    timestamp, or — when omitted — a deterministic identifier derived from
    ``config`` (a short SHA-256 prefix of the config dump). ``write_tables``
    threads ``config`` through automatically; direct callers that want the
    fingerprint should pass ``config`` explicitly. CLI's ``cmd_run`` passes
    ``generated_at=datetime.now()`` to keep the wall-clock stamp.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / REPORT_FILENAME
    path.write_text(
        _format_report(report, generated_at=generated_at, config=config),
        encoding=CSV_ENCODING,
    )
    return path


# --- 0.6-M9c CDC quality-update flip ----------------------------------------


_CDC_COLUMN_LEVEL_ISSUES = frozenset({"null_injection", "type_mismatch", "schema_drift"})


def _mark_cdc_quality_updates(
    corrupted: dict[str, pd.DataFrame],
    ground_truth: list,
    cdc_facts: set[str],
) -> dict[str, pd.DataFrame]:
    """Set ``_op="U"`` and bump ``_updated_at`` for rows on CDC-enabled
    fact tables that a column-level quality issue mutated.

    Only column-level issues (``null_injection`` / ``type_mismatch`` /
    ``schema_drift``) are marked: their ground-truth row_indices line
    up cleanly with the post-corruption frame because those issues
    don't change row count. Row-level issues (``duplicate_rows`` /
    ``late_arrival``) shift indices on the corrupted frame relative to
    the source, so the helper intentionally skips them — the
    ``_op="I"`` initial state survives, and the discovered limitation
    is documented in the M9c notes.

    ``_updated_at`` is bumped to the LAST period's ``_inserted_at`` on
    the same table — semantic: "the row was inserted at its date_key
    period and touched again at end-of-window when the quality issue
    was applied".
    """
    out = dict(corrupted)
    rows_per_table: dict[str, set[int]] = {}
    for record in ground_truth:
        if record.table not in cdc_facts:
            continue
        if record.issue_type not in _CDC_COLUMN_LEVEL_ISSUES:
            continue
        rows_per_table.setdefault(record.table, set()).update(record.row_indices)

    for table_name, idx_set in rows_per_table.items():
        df = corrupted.get(table_name)
        if df is None or "_op" not in df.columns or "_inserted_at" not in df.columns:
            continue
        bumped = ""
        if len(df) > 0:
            bumped = str(df["_inserted_at"].iloc[-1])
        df = df.copy(deep=True)
        valid_idxs = [i for i in idx_set if 0 <= i < len(df)]
        if valid_idxs:
            df.loc[df.index[valid_idxs], "_op"] = "U"
            df.loc[df.index[valid_idxs], "_updated_at"] = bumped
        out[table_name] = df
    return out


# --- Top-level orchestrator --------------------------------------------------


def write_tables(
    tables: dict[str, pd.DataFrame],
    config: PlotsimConfig,
    report: Optional[ValidationReport] = None,
    output_dir: str | Path | None = None,
    float_format: str = FLOAT_FORMAT,
    base_dir: str | Path | None = None,
    generated_at: Optional[_dt.datetime] = None,
    manifest: Optional[ManifestSchema] = None,
) -> Path:
    """Write every generated table, the config copy, and the validation report.

    If ``output_dir`` is ``None``, uses ``config.output.directory``. The
    directory is created if missing. Existing files at the same paths are
    overwritten (no append; no timestamped subdirs).

    If ``report`` is ``None``, the full validation suite is run on
    ``(config, tables)`` before writing. Callers that already have a
    report (for example, to branch on ``report.ok`` first) should pass
    it through to avoid re-running the checks.

    Generation failures are not masked: if ``report.ok`` is False the CSVs
    are still written so the operator can inspect the broken data. Callers
    that want to block on invalid output should check ``report.ok`` before
    calling this function.

    FIX-08 / SF-2: ``base_dir`` is an optional sandbox root for hosted
    deployments (Streamlit, FastAPI). When set, the resolved target must
    live under ``base_dir`` — absolute-path overrides and ``..`` traversal
    are rejected with :class:`ValueError`. The CLI default (``base_dir=None``)
    preserves full user control over the filesystem.

    M105: when ``manifest`` is supplied AND ``config.manifest.include`` is
    True, ``manifest.json`` is written alongside the table files. The
    caller (CLI's ``cmd_run`` and library users that call
    ``generate_tables_with_state``) is responsible for building the
    manifest because constructing it requires the trajectories used during
    generation, which ``write_tables`` cannot otherwise reach.
    Programmatic callers that pass ``manifest=None`` opt out — useful for
    ad-hoc DataFrames written directly without a generation run.

    Returns the output directory path.
    """
    if report is None:
        report = validate_tables(config, tables)
    raw_target = Path(output_dir) if output_dir is not None else Path(config.output.directory)
    target = _resolve_target(raw_target, base_dir)
    target.mkdir(parents=True, exist_ok=True)

    # M107: post-generation data-quality injection. Clean ``tables`` are
    # already finished (and just consumed by ``validate_tables`` above and
    # by the manifest builder before this call); the quality layer
    # produces a corrupted dict the writer hands to disk while leaving
    # the in-memory clean copy untouched. Manifest's ground-truth
    # ``quality_injections`` field is patched in via
    # ``ManifestSchema.model_copy(update=...)`` so the on-disk
    # manifest.json names every corrupted (table, column, row_indices,
    # clean_values) tuple. Configs without ``quality_issues`` short-
    # circuit and behavior is byte-identical to pre-M107.
    tables_to_write = tables
    manifest_to_write = manifest
    ground_truth: list = []
    if config.quality.quality_issues:
        corrupted, ground_truth = _apply_quality_issues(
            tables,
            config,
            int(config.seed),
        )
        tables_to_write = corrupted
        if manifest_to_write is not None:
            manifest_to_write = manifest_to_write.model_copy(
                update={"quality_injections": ground_truth},
            )

    # 0.6-M9c: flip ``_op`` to ``"U"`` on rows that column-level quality
    # issues mutated. Row-level issues (``duplicate_rows`` /
    # ``late_arrival``) shift indices on the corrupted frame relative to
    # the source, so this pass intentionally only marks the column-level
    # mutations (``null_injection``, ``type_mismatch``, ``schema_drift``)
    # whose ground-truth row_indices align cleanly with the post-
    # corruption frame. ``_updated_at`` is bumped to the LAST period's
    # label — semantic: "the row was inserted at its date_key period and
    # touched again at end-of-window when the quality issue was applied".
    cdc_facts = {t.name for t in config.tables if t.type == "fact" and t.cdc}
    if cdc_facts and ground_truth:
        tables_to_write = _mark_cdc_quality_updates(
            tables_to_write,
            ground_truth,
            cdc_facts,
        )

    # 0.6-M16c: single-file SQL dump path. ``format == "sql"`` bypasses
    # the per-table writer loop, the denormalized sidecar loop, the
    # holdout-split loop, and the streaming-Parquet path — everything
    # the user requested lands in ``data.sql`` instead. Companions
    # (config.yaml, validation_report.txt, manifest.json) + the
    # log-file writer still run below the branch; they are not table
    # data and stay in their canonical formats.
    output_format = _resolve_output_format(config)
    if output_format == "sql":
        sql_wide = (
            denormalize_fact_tables(tables_to_write, config)
            if getattr(config.output, "denormalized", False)
            else None
        )
        sql_holdout = split_fact_tables(config, tables) if config.holdout.enabled else None
        _write_sql_dump(
            tables_to_write,
            config,
            target,
            wide_tables=sql_wide,
            holdout_splits=sql_holdout,
        )
    else:
        # M121b: streaming Parquet path. When format=parquet AND the
        # resolved generation_mode is vectorized, fact tables are written
        # via ``_write_streaming_parquet_facts`` (per-archetype row groups
        # via ParquetWriter) and skipped in the standard loop below. Dim,
        # event, and bridge tables continue through ``write_single_table``.
        # Serial mode and CSV output skip this branch entirely.
        streaming_written: set[str] = set()
        if _streaming_parquet_eligible(config):
            streaming_written = _write_streaming_parquet_facts(
                config,
                tables_to_write,
                target,
            )

        for name, df in tables_to_write.items():
            if name in streaming_written:
                continue
            write_single_table(name, df, target, config=config, float_format=float_format)

        # 0.6-M14a: opt-in wide-table companions. When
        # ``output.denormalized: true``, every fact table is left-joined
        # with its FK'd dims (SCD2 dims filtered to current state) and
        # written as ``<fct_name>_wide.{csv|parquet|jsonl}`` alongside
        # the normalized output. Off by default so pre-M14a output is
        # byte-identical. Consumes ``tables_to_write`` (post-CDC,
        # post-quality) so the wide view matches what landed on disk
        # for the normalized tables. Under format=sql, this loop is
        # bypassed — wide tables land inside ``data.sql`` instead.
        if getattr(config.output, "denormalized", False):
            wide_tables = denormalize_fact_tables(tables_to_write, config)
            for wide_name, wide_df in wide_tables.items():
                # Pass ``config`` so ``_resolve_output_format`` picks up
                # the parquet branch when configured. The wide name
                # (``<fct>_wide``) isn't in ``config.tables``, so
                # ``_table_by_name`` returns None and the column-reorder
                # / Int64-coercion path is skipped — wide frames keep
                # their post-merge column order.
                write_single_table(
                    wide_name,
                    wide_df,
                    target,
                    config=config,
                    float_format=float_format,
                )

    # 0.6-M14b: opt-in log-file companions. When any event table has
    # ``log_format`` set, ``write_event_logs`` emits one ``.log`` file
    # per such event table alongside the regular CSV/Parquet output.
    # No event tables with ``log_format`` configured → no I/O.
    # Consumes ``tables_to_write`` so the log lines reflect the same
    # (post-CDC, post-quality) data that landed in the event CSVs.
    write_event_logs(tables_to_write, config, target)

    write_config_copy(config, target)
    write_validation_report(report, target, generated_at=generated_at, config=config)

    # M109: temporal train/holdout split. Mutually exclusive with
    # quality injection at config load, so ``tables_to_write`` here
    # matches the clean ``tables`` dict whenever ``holdout.enabled`` is
    # true. We still walk ``tables`` (the clean dict) explicitly to
    # make that invariant visible at the call site.
    # 0.6-M16c: under format=sql, holdout splits are emitted inside
    # ``data.sql`` by ``_write_sql_dump`` above — the per-split file
    # writes are skipped here. The manifest's HoldoutInfo stitching
    # below still fires regardless of format.
    holdout_splits: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    if config.holdout.enabled and output_format != "sql":
        holdout_splits = split_fact_tables(config, tables)
        for name, (train_df, holdout_df) in holdout_splits.items():
            write_single_table(
                f"{name}_train",
                train_df,
                target,
                config=config,
                float_format=float_format,
            )
            write_single_table(
                f"{name}_holdout",
                holdout_df,
                target,
                config=config,
                float_format=float_format,
            )

    if manifest_to_write is not None and config.manifest.include:
        # M109: stitch the holdout summary onto the manifest payload
        # right before the file is written so the on-disk
        # manifest.json names ``target_metric`` / ``holdout_periods``
        # / ``cutoff_period_index``. Done here rather than inside
        # ``build_manifest`` so the manifest builder stays free of the
        # writer's I/O ordering — the holdout block is purely a
        # config-derived sidecar, not a function of generation state.
        if config.holdout.enabled:
            from plotsim.manifest import HoldoutInfo

            manifest_to_write = manifest_to_write.model_copy(
                update={
                    "holdout": HoldoutInfo(
                        target_metric=config.holdout.target_metric or "",
                        holdout_periods=int(config.holdout.holdout_periods),
                        cutoff_period_index=cutoff_period_index(config),
                    ),
                },
            )
        write_manifest(manifest_to_write, target)

    # M108: per-entity feature table. The load-time validator
    # (``validate_entity_features_config``) has already ensured
    # ``manifest.include=true`` and ``quality.quality_issues==[]`` are
    # both satisfied when ``entity_features.enabled``, so the only
    # programmatic path that would reach here without a manifest is a
    # caller that constructed ``PlotsimConfig`` in code and passed
    # ``manifest=None``. Surface that as an explicit error rather than
    # silently dropping the file.
    if config.entity_features.enabled:
        if manifest is None:
            raise ValueError(
                "entity_features.enabled=true but no manifest was passed to "
                "write_tables; build the manifest first via "
                "plotsim.manifest.build_manifest and forward it through"
            )
        # Build off the CLEAN ``tables`` dict — quality injection and
        # entity features are mutually exclusive at config load, so
        # ``tables_to_write`` and ``tables`` are identical here, but
        # naming the clean source makes the intent explicit and keeps
        # the entity-feature contract stable if the gate ever loosens.
        # M109: when holdout is enabled, ``build_entity_features``
        # restricts aggregation to training periods and drops the
        # target-metric aggregate columns. The writer hands it the
        # full clean tables dict either way; the holdout-aware
        # filtering happens inside the builder using ``config.holdout``.
        entity_features_df = build_entity_features(config, tables, manifest)
        _write_entity_features(entity_features_df, target, config, float_format)

    return target


def _write_entity_features(
    df: pd.DataFrame,
    output_dir: Path,
    config: PlotsimConfig,
    float_format: str,
) -> Path:
    """Write the M108 per-entity feature DataFrame to disk.

    Filename basename is the module-level constant
    ``ENTITY_FEATURES_BASENAME`` (``_entity_features``); the leading
    underscore signals "derived companion" rather than "table" and
    keeps the file out of any glob that targets ``*.csv``-tables only.
    Format follows ``config.output.format`` so a user opting into
    Parquet (or 0.6-M16b JSONL) for the table set gets the matching
    encoding for the feature file too — no mixed-encoding output dirs.

    Same encoding / quoting / float-format conventions as the regular
    table writers (``CSV_ENCODING``, ``QUOTE_NONNUMERIC``, ``%.4f``)
    on the CSV branch; the JSONL branch uses ``_write_jsonl``'s pinned
    options (orient=records, lines=True, date_format=iso,
    force_ascii=False).
    """
    output_format = _resolve_output_format(config)
    extension = _extension_for_format(output_format)
    path = output_dir / f"{ENTITY_FEATURES_BASENAME}.{extension}"
    if output_format == "parquet":
        _check_parquet_engine_available()
        df.to_parquet(
            path,
            engine="pyarrow",
            index=False,
            compression="snappy",
        )
    elif output_format == "jsonl":
        _write_jsonl(df, path)
    else:
        df.to_csv(
            path,
            index=False,
            encoding=CSV_ENCODING,
            quoting=csv.QUOTE_NONNUMERIC,
            float_format=float_format,
            na_rep=NA_REP,
            lineterminator="\n",
        )
    return path


def _resolve_target(
    raw_target: Path,
    base_dir: str | Path | None,
) -> Path:
    """Resolve ``raw_target`` against an optional sandbox ``base_dir``.

    When ``base_dir`` is ``None``, the target is returned as-is (the CLI
    contract — the user owns their filesystem). When ``base_dir`` is set:

      * Absolute targets are rejected — the caller supplies a relative
        subpath under the sandbox.
      * ``..`` segments that escape ``base_dir`` are rejected after
        normalization.
      * The nested directory is created as needed inside ``base_dir``.

    Paths are compared after ``Path.resolve()`` on ``base_dir`` so symlinks
    in the sandbox root itself resolve consistently; the target is joined
    into the resolved base and re-resolved to catch traversal.
    """
    if base_dir is None:
        return raw_target
    sandbox = Path(base_dir).resolve()
    sandbox.mkdir(parents=True, exist_ok=True)
    if raw_target.is_absolute():
        raise ValueError(
            f"write_tables: output_dir {str(raw_target)!r} is an absolute "
            f"path, which is not allowed when base_dir={str(base_dir)!r} "
            f"is set; pass a relative subpath under base_dir instead"
        )
    resolved = (sandbox / raw_target).resolve()
    try:
        resolved.relative_to(sandbox)
    except ValueError:
        raise ValueError(
            f"write_tables: output_dir {str(raw_target)!r} escapes "
            f"base_dir {str(base_dir)!r}; parent-traversal is not allowed"
        ) from None
    return resolved
