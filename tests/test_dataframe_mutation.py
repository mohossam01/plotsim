"""F3 + F4 regression — vectorized fact-builder dtype + DataFrame mutation (M102).

This file holds the regression tests for two related Phase 1 fixes:

* **F3 (dtype section).** The vectorized fact-builder used to assign the raw
  float slice from ``metrics_3d`` straight into MetricSource and LagSource
  columns, ignoring the declared ``Column.dtype``. Library callers got
  float64 where they had declared int (or boolean). The CSV path was rescued
  downstream by ``output._coerce_integer_columns`` at write-time, so on-disk
  was correct but in-memory and on-disk diverged. F3 moves the coercion into
  the vectorized path itself via ``_coerce_array_for_dtype``, producing
  ``Int64`` / ``BooleanDtype`` extension arrays that match what the writer
  produces — closing the in-memory-vs-on-disk gap.

* **F4 (mutation section).** ``write_tables`` → ``write_single_table`` used
  to call ``_coerce_integer_columns`` directly on the caller's DataFrame,
  mutating int columns in place (replacing the Series object even when the
  dtype was already correct after F3). F4 wraps every call in a shallow
  copy so the user's dict is never mutated. The mutation tests below assert
  the contract from three angles:

    1. Declared int-column Series objects are *identical* after
       ``write_tables`` — proves the in-place reassignment is gone.
    2. User-added columns (a custom float metric the user attaches between
       ``generate_tables`` and ``write_tables``) round-trip with their dtype
       intact — proves write_tables doesn't reach into anything beyond the
       config-declared columns.
    3. ``pd.read_csv(...)`` (default backend, no nullable hint) recovers
       dtypes that are compatible with the in-memory ones under documented
       CSV equivalences (Int64-with-no-NA ≡ int64; Int64-with-NA ≡ float64;
       BooleanDtype-with-no-NA ≡ bool; BooleanDtype-with-NA ≡ object) —
       proves the most natural user round-trip works without specifying
       ``dtype_backend='numpy_nullable'``.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from plotsim import generate_tables, load_config, write_tables
from plotsim.config import SurrogateKeyWarning


ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = ROOT / "plotsim" / "configs"

# Bundled templates that declare at least one ``dtype: int`` MetricSource
# column on a per-entity-per-period fact table — these are the surface F3
# operates on.
TEMPLATES_WITH_INT_METRICS: dict[str, Path] = {
    "saas":      CONFIGS_DIR / "sample_saas.yaml",       # ticket_count
    "retail":    CONFIGS_DIR / "sample_retail.yaml",     # session_count
    "marketing": CONFIGS_DIR / "sample_marketing.yaml",  # session_count, impressions
}


def _int_metric_columns(cfg) -> list[tuple[str, str]]:
    """Return [(table_name, column_name), ...] for every fact-table column
    with ``dtype: int`` and a MetricSource / LagSource source.
    """
    out = []
    for tbl in cfg.tables:
        if tbl.type != "fact":
            continue
        for col in tbl.columns:
            if col.dtype != "int":
                continue
            if col.source.startswith("metric:") or col.source.startswith("lag:"):
                out.append((tbl.name, col.name))
    return out


@pytest.fixture(scope="module", params=sorted(TEMPLATES_WITH_INT_METRICS))
def template_config(request):
    yaml_path = TEMPLATES_WITH_INT_METRICS[request.param]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SurrogateKeyWarning)
        return request.param, load_config(yaml_path)


def test_vectorized_int_metric_column_is_integer_dtype_in_memory(template_config):
    """F3 — every fact column declared ``dtype: int`` and sourced from a
    metric must come back as an integer-like dtype (Int64 nullable or
    int64) from ``generate_tables``, not float64.

    Pre-fix: the raw ``metrics_3d`` slice landed in the column as
    float64 because the vectorized path skipped the dtype coercion that
    ``_coerce_metric_value`` applied in the scalar fallback. The test
    fails with ``dtype('float64')`` for ticket_count / session_count /
    impressions.
    """
    template, cfg = template_config
    rng = np.random.default_rng(cfg.seed)
    tables = generate_tables(cfg, rng)

    targets = _int_metric_columns(cfg)
    assert targets, f"{template}: no int-dtype metric columns found — test setup wrong"

    for tbl_name, col_name in targets:
        series = tables[tbl_name][col_name]
        assert pd.api.types.is_integer_dtype(series.dtype), (
            f"F3 regression: {template}/{tbl_name}.{col_name} declared "
            f"dtype:int but in-memory dataframe column has "
            f"dtype={series.dtype!r} (expected integer-like). Vectorized "
            f"fact-builder is not applying _coerce_array_for_dtype."
        )


def test_in_memory_dtype_matches_on_disk_round_trip(template_config, tmp_path):
    """F3 — in-memory ``Int64`` round-trips through the CSV writer
    losslessly: every value an in-memory ``Int64`` column carries is
    recoverable as an integer from the on-disk CSV.

    Pre-F3: in-memory was float64 (vectorized path skipped coercion);
    on-disk was Int64 (rescued by output._coerce_integer_columns). They
    didn't agree — the property the operator flagged as actually broken.
    Post-F3 + F4: both ends produce values that survive the round-trip.

    F15-extension (Phase 3 / F16 verification): the prior version of
    this test used ``pd.read_csv(..., dtype_backend='numpy_nullable')``,
    which under ``pytest --cov`` triggers the same numpy-reload
    interaction F15 caught for ``np.polyfit``: pandas' DataFrame
    constructor receives an ``IntegerArray`` whose post-reload type
    identity no longer matches the C extension's isinstance check, and
    raises ``TypeError: Argument 'values' has incorrect type
    (expected numpy.ndarray, got IntegerArray)``. The workaround
    follows F15's pattern — replace the broken-under-cov call site with
    an equivalent that doesn't go through the C ufunc + extension-array
    dispatch.

    The test now (a) reads with the default numpy backend, which yields
    ``int64`` (no nulls in any bundled int-metric column) or ``float64``
    (with nulls) — both are losslessly castable to ``Int64`` — then
    (b) casts the on-disk column to ``Int64`` explicitly and (c)
    asserts the cast values match the in-memory ``Int64`` values.
    Same property, no extension-array path, runs clean under ``--cov``.
    """
    template, cfg = template_config
    rng = np.random.default_rng(cfg.seed)
    tables = generate_tables(cfg, rng)

    targets = _int_metric_columns(cfg)
    assert targets, f"{template}: no int-dtype metric columns found — test setup wrong"

    # Snapshot in-memory dtypes + values BEFORE write_tables. ``write_tables``
    # calls ``output._coerce_integer_columns`` which (pre-F4) mutates the
    # caller's dataframe in place, promoting float64 → Int64 — which would
    # mask the F3 bug by making post-write_tables dtypes already match on-disk.
    in_memory_snapshots = {
        (tbl_name, col_name): tables[tbl_name][col_name].copy()
        for tbl_name, col_name in targets
    }

    write_tables(tables, cfg, output_dir=tmp_path)

    for tbl_name, col_name in targets:
        in_memory_series = in_memory_snapshots[(tbl_name, col_name)]
        in_memory_dtype = in_memory_series.dtype
        # Default backend (no dtype_backend kwarg) — sidesteps the IntegerArray
        # block-form path that interacts with the coverage tracer's numpy reload.
        on_disk_raw = pd.read_csv(tmp_path / f"{tbl_name}.csv")
        on_disk_int64 = on_disk_raw[col_name].astype("Int64")
        assert pd.api.types.is_integer_dtype(in_memory_dtype), (
            f"F3 regression: {template}/{tbl_name}.{col_name} in-memory "
            f"dtype is {in_memory_dtype!r}, expected integer-like."
        )
        # F3+F4: the value the user reads back from the CSV must agree
        # with what the in-memory column held at the moment write_tables
        # was called. NA-aware equality via Int64.
        in_memory_int64 = in_memory_series.astype("Int64").reset_index(drop=True)
        on_disk_int64 = on_disk_int64.reset_index(drop=True)
        assert in_memory_int64.equals(on_disk_int64), (
            f"F3 regression: {template}/{tbl_name}.{col_name} CSV "
            f"round-trip lost values. In-memory ({in_memory_dtype!r}) "
            f"and on-disk-then-Int64 disagree on at least one row — "
            f"vectorized path and write_tables are not agreeing on "
            f"this column's serialized integer values."
        )


# --- F4: write_tables must not mutate the caller's dataframe ----------------


def test_write_tables_does_not_mutate_int_column_series_objects(template_config, tmp_path):
    """F4 — every declared ``dtype:int`` fact-table column must round-trip
    ``write_tables`` with values and dtype intact (no in-place coercion of
    the caller's data).

    Pre-fix: ``_coerce_integer_columns`` runs ``df[col] = series.astype('Int64')``
    on the caller's dataframe, replacing the Series object — and on
    float64 columns, also rewriting the values. The earlier version of
    this test asserted Series object identity (``current is original``),
    which was the cleanest pre-fix observable at the time.

    Post-fix: write_tables operates on a shallow copy; the user's columns
    are untouched. Pandas 3.0 made Copy-on-Write the only mode and
    ``df[col]`` no longer guarantees the same Series wrapper across
    accesses, so the contract is now expressed as value+dtype equality
    on a snapshot taken before the write. That still fails loudly under
    the pre-fix in-place mutation (values would change on float64 inputs;
    dtype would change on coerced columns) without depending on a CoW-
    incompatible identity invariant.
    """
    template, cfg = template_config
    rng = np.random.default_rng(cfg.seed)
    tables = generate_tables(cfg, rng)

    targets = _int_metric_columns(cfg)
    assert targets, f"{template}: no int-dtype metric columns found — test setup wrong"

    # Snapshot value + dtype before write_tables. .copy() is essential
    # under CoW: holding a Series reference is no longer sufficient to
    # observe pre-write state if the writer rebinds the column.
    snapshots = {
        (tbl_name, col_name): tables[tbl_name][col_name].copy()
        for tbl_name, col_name in targets
    }

    write_tables(tables, cfg, output_dir=tmp_path)

    for (tbl_name, col_name), original in snapshots.items():
        current = tables[tbl_name][col_name]
        assert current.dtype == original.dtype, (
            f"F4 regression: {template}/{tbl_name}.{col_name} dtype changed "
            f"from {original.dtype!r} to {current.dtype!r} during write_tables."
        )
        assert current.equals(original), (
            f"F4 regression: {template}/{tbl_name}.{col_name} values changed "
            f"during write_tables — the writer mutated the caller's column."
        )


def test_write_tables_does_not_touch_user_added_columns(template_config, tmp_path):
    """F4 — columns the user attached between generate_tables and
    write_tables must round-trip with their original dtype intact.

    Locks the broader contract beyond declared int columns: write_tables
    is read-only against the caller's dict. Adds a custom float column
    whose name does not collide with any declared column, then asserts it
    is byte-identical post-write.
    """
    template, cfg = template_config
    rng = np.random.default_rng(cfg.seed)
    tables = generate_tables(cfg, rng)

    # Pick the first fact table; attach a custom float column the user
    # might compute themselves (e.g., a derived ratio). Name is chosen to
    # not collide with any declared column.
    fact_name = next(t.name for t in cfg.tables if t.type == "fact")
    fact_df = tables[fact_name]
    custom_values = np.linspace(0.0, 1.0, num=len(fact_df), dtype=np.float64)
    fact_df["plotsim_user_added_ratio"] = custom_values
    snapshot_dtype = fact_df["plotsim_user_added_ratio"].dtype
    snapshot_values = fact_df["plotsim_user_added_ratio"].copy()

    write_tables(tables, cfg, output_dir=tmp_path)

    assert "plotsim_user_added_ratio" in tables[fact_name].columns, (
        f"F4 regression: {template}/{fact_name} dropped a user-added column "
        f"during write_tables — the user's dict is mutated."
    )
    after = tables[fact_name]["plotsim_user_added_ratio"]
    assert after.dtype == snapshot_dtype, (
        f"F4 regression: {template}/{fact_name}.plotsim_user_added_ratio "
        f"dtype changed from {snapshot_dtype!r} to {after.dtype!r} during "
        f"write_tables — write_tables touched a column it doesn't own."
    )
    assert after.equals(snapshot_values), (
        f"F4 regression: {template}/{fact_name}.plotsim_user_added_ratio "
        f"values changed during write_tables."
    )


# --- F4: in-memory ↔ on-disk round-trip under default pd.read_csv backend ---


def _csv_round_trip_kind(series: pd.Series) -> tuple[str, bool]:
    """Normalize a pandas Series dtype to a (kind, has_na) tuple that
    survives CSV serialization. Used to compare in-memory dtypes against
    dtypes inferred by ``pd.read_csv`` (default numpy backend, NOT
    ``dtype_backend='numpy_nullable'``).

    Mapping:
      * any integer (Int64 / int64) → "int"
      * any float (float64 / Float64) → "float"
      * any bool (BooleanDtype / bool / bool_) → "bool"
      * everything else (object / string / date) → "object"
    """
    dt = series.dtype
    has_na = bool(series.isna().any())
    if pd.api.types.is_integer_dtype(dt):
        return "int", has_na
    if pd.api.types.is_float_dtype(dt):
        return "float", has_na
    if pd.api.types.is_bool_dtype(dt):
        return "bool", has_na
    return "object", has_na


def _round_trip_compatible(in_memory: pd.Series, on_disk: pd.Series) -> bool:
    """True iff ``on_disk`` is a CSV-round-trip-compatible representation
    of ``in_memory`` under ``pd.read_csv``'s default dtype inference.

    Known equivalences (lossy at the type-tag level, lossless at the value
    level under the empty-string null convention plotsim writes with):

      * Int64 with no nulls  ≡ int64    (pd.read_csv reads as int64)
      * Int64 with nulls     ≡ float64  (NaN promotes int → float in numpy)
      * Float64              ≡ float64
      * BooleanDtype no NA   ≡ bool
      * BooleanDtype with NA ≡ object   (NaN cells → string "True"/"False" mix)
      * object               ≡ object   (string columns, dates as strings)
    """
    in_kind, in_has_na = _csv_round_trip_kind(in_memory)
    on_kind, _ = _csv_round_trip_kind(on_disk)

    if in_kind == on_kind:
        return True
    if in_kind == "int" and in_has_na and on_kind == "float":
        return True
    if in_kind == "bool" and in_has_na and on_kind == "object":
        return True
    return False


def test_csv_round_trip_default_backend_dtype_compatibility(template_config, tmp_path):
    """F4 / F3 joint contract — every fact-table column's in-memory dtype
    must be CSV-round-trip-compatible with the dtype ``pd.read_csv``
    recovers under default settings (no ``dtype_backend`` hint).

    Default ``pd.read_csv`` is the most common user path. The contract:
    in-memory and on-disk-then-read agree under documented equivalences
    (see ``_round_trip_compatible``). This is the property the operator
    flagged: the two paths must agree from every angle a user can observe,
    not just under the explicit numpy-nullable backend tested above.
    """
    template, cfg = template_config
    rng = np.random.default_rng(cfg.seed)
    tables = generate_tables(cfg, rng)

    # Snapshot every fact-table column's in-memory series BEFORE write_tables
    # — guards against the F4 mutation hiding pre-fix dtype divergence.
    fact_tables = {t.name for t in cfg.tables if t.type == "fact"}
    in_memory: dict[tuple[str, str], pd.Series] = {}
    for name in fact_tables:
        for col_name in tables[name].columns:
            in_memory[(name, col_name)] = tables[name][col_name].copy()

    write_tables(tables, cfg, output_dir=tmp_path)

    for name in fact_tables:
        on_disk_df = pd.read_csv(tmp_path / f"{name}.csv")
        for col_name in on_disk_df.columns:
            in_mem = in_memory.get((name, col_name))
            if in_mem is None:
                # Reordered or extra column the user didn't see in-memory.
                continue
            on_disk = on_disk_df[col_name]
            assert _round_trip_compatible(in_mem, on_disk), (
                f"F4/F3 round-trip mismatch: {template}/{name}.{col_name} "
                f"in-memory dtype={in_mem.dtype!r} (has_na={in_mem.isna().any()}) "
                f"is not CSV-round-trip-compatible with on-disk-then-read "
                f"dtype={on_disk.dtype!r}. The two paths disagree at the "
                f"default-backend level the typical user sees."
            )
