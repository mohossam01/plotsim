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
    "saas":       CONFIGS_DIR / "sample_saas.yaml",        # ticket_count
    "ecommerce":  CONFIGS_DIR / "sample_ecommerce.yaml",   # session_count
    "healthcare": CONFIGS_DIR / "sample_healthcare.yaml",  # visit_count
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
    visit_count.
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
    """F3 — in-memory dtype must match the dtype recovered from the CSV
    after ``write_tables`` → ``pd.read_csv(..., dtype_backend='numpy_nullable')``.

    Pre-fix: in-memory was float64 (vectorized path skipped coercion);
    on-disk was Int64 (rescued by output._coerce_integer_columns). They
    didn't agree — the property the operator flagged as actually broken.
    Post-fix: both ends produce the same ``Int64`` extension dtype.

    The numpy_nullable backend is the explicit pd.read_csv mode that
    preserves nullable integer dtypes across the CSV serialization.
    Without it, ``pd.read_csv`` infers float64 for any integer column
    that contains nulls — which would always pass the comparison
    pre-fix (both float) but mask the actual contract failure.
    """
    template, cfg = template_config
    rng = np.random.default_rng(cfg.seed)
    tables = generate_tables(cfg, rng)

    targets = _int_metric_columns(cfg)
    assert targets, f"{template}: no int-dtype metric columns found — test setup wrong"

    # Snapshot in-memory dtypes BEFORE write_tables. ``write_tables`` calls
    # ``output._coerce_integer_columns`` which (pre-F4) mutates the caller's
    # dataframe in place, promoting float64 → Int64 — which would mask the
    # F3 bug here by making post-write_tables dtypes already match on-disk.
    in_memory_dtypes = {
        (tbl_name, col_name): tables[tbl_name][col_name].dtype
        for tbl_name, col_name in targets
    }

    write_tables(tables, cfg, output_dir=tmp_path)

    for tbl_name, col_name in targets:
        in_memory_dtype = in_memory_dtypes[(tbl_name, col_name)]
        on_disk_df = pd.read_csv(
            tmp_path / f"{tbl_name}.csv",
            dtype_backend="numpy_nullable",
        )
        on_disk_dtype = on_disk_df[col_name].dtype
        assert pd.api.types.is_integer_dtype(in_memory_dtype), (
            f"F3 regression: {template}/{tbl_name}.{col_name} in-memory "
            f"dtype is {in_memory_dtype!r}, expected integer-like."
        )
        assert pd.api.types.is_integer_dtype(on_disk_dtype), (
            f"F3 regression: {template}/{tbl_name}.{col_name} on-disk-then-read "
            f"dtype is {on_disk_dtype!r}, expected integer-like."
        )
        assert in_memory_dtype == on_disk_dtype, (
            f"F3 regression: {template}/{tbl_name}.{col_name} dtype "
            f"diverges across CSV round-trip. In-memory: {in_memory_dtype!r}; "
            f"on-disk-then-read (numpy_nullable backend): {on_disk_dtype!r}. "
            f"This is the property the F3 fix locks down — vectorized path "
            f"and write_tables must agree on the column's serialized dtype."
        )


# --- F4: write_tables must not mutate the caller's dataframe ----------------


def test_write_tables_does_not_mutate_int_column_series_objects(template_config, tmp_path):
    """F4 — every declared ``dtype:int`` fact-table column must reference
    the same Series object after ``write_tables`` as before.

    Pre-fix: ``_coerce_integer_columns`` runs ``df[col] = series.astype('Int64')``
    on the caller's dataframe, replacing the Series object even when the
    dtype is already Int64 (idempotent astype still returns a new wrapper).
    The dtype is unchanged post-F3 but the *object identity* is not.
    Identity is the cleanest pre-fix-failing observable because the
    behavioral change is "your reference becomes stale."

    Post-fix: write_tables operates on a shallow copy; the user's
    Series objects are untouched.
    """
    template, cfg = template_config
    rng = np.random.default_rng(cfg.seed)
    tables = generate_tables(cfg, rng)

    targets = _int_metric_columns(cfg)
    assert targets, f"{template}: no int-dtype metric columns found — test setup wrong"

    # Snapshot Series identity (and dtype) before write_tables.
    snapshots = {
        (tbl_name, col_name): tables[tbl_name][col_name]
        for tbl_name, col_name in targets
    }

    write_tables(tables, cfg, output_dir=tmp_path)

    for (tbl_name, col_name), original in snapshots.items():
        current = tables[tbl_name][col_name]
        assert current is original, (
            f"F4 regression: {template}/{tbl_name}.{col_name} Series object "
            f"was replaced by write_tables (pre-fix _coerce_integer_columns "
            f"runs `df[col] = series.astype('Int64')` on the caller's df). "
            f"User references to the column are silently invalidated."
        )
        assert current.dtype == original.dtype, (
            f"F4 regression: {template}/{tbl_name}.{col_name} dtype changed "
            f"from {original.dtype!r} to {current.dtype!r} during write_tables."
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
