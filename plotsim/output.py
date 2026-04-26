"""plotsim.output — CSV / Parquet file writing.

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
    preserved by the format). Both branches share the same column-ordering
    and Int64 coercion path; only the file extension and the on-disk
    encoder differ. ``config.yaml`` and ``validation_report.txt`` are
    always written as text — they are companions, not table data.

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
    tbl: Table, df: pd.DataFrame,
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


def _resolve_output_format(config: Optional[PlotsimConfig]) -> str:
    """Return ``'csv'`` or ``'parquet'`` based on config; default to CSV.

    Programmatic callers that pass ``config=None`` or a stub object
    without an ``output`` attribute (e.g. unit tests of
    ``write_single_table`` against an ad-hoc DataFrame) get CSV — the
    long-standing behavior — preserved. The defensive ``getattr`` chain
    keeps that contract intact while the YAML-loaded ``PlotsimConfig``
    surface drives the parquet branch.
    """
    if config is None:
        return "csv"
    output_cfg = getattr(config, "output", None)
    if output_cfg is None:
        return "csv"
    return getattr(output_cfg, "format", "csv")


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
    extension = "parquet" if output_format == "parquet" else "csv"
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
        to_write.to_parquet(
            path,
            engine="pyarrow",
            index=False,
            compression="snappy",
        )
    else:
        to_write.to_csv(
            path,
            index=False,
            encoding=CSV_ENCODING,
            quoting=csv.QUOTE_NONNUMERIC,
            float_format=float_format,
            na_rep=NA_REP,
        )
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


# --- Top-level orchestrator --------------------------------------------------


def write_tables(
    tables: dict[str, pd.DataFrame],
    config: PlotsimConfig,
    report: Optional[ValidationReport] = None,
    output_dir: str | Path | None = None,
    float_format: str = FLOAT_FORMAT,
    base_dir: str | Path | None = None,
    generated_at: Optional[_dt.datetime] = None,
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

    Returns the output directory path.
    """
    if report is None:
        report = validate_tables(config, tables)
    raw_target = Path(output_dir) if output_dir is not None else Path(config.output.directory)
    target = _resolve_target(raw_target, base_dir)
    target.mkdir(parents=True, exist_ok=True)

    for name, df in tables.items():
        write_single_table(name, df, target, config=config, float_format=float_format)

    write_config_copy(config, target)
    write_validation_report(report, target, generated_at=generated_at, config=config)
    return target


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
