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

* **F4 (mutation section).** ``write_tables``' call to
  ``_coerce_integer_columns`` mutates the caller's DataFrame in place,
  silently changing dtypes after the user has handed over the dict. The
  fix takes a shallow copy of the columns being mutated. (Test still TBD —
  added by the F4 commit.)

Tests below exercise the F3 contract: every fact-table column declared
``dtype: int`` resolves to an integer-like dtype in the in-memory dataframe,
AND that dtype matches what comes back when the CSV is round-tripped via
``pd.read_csv(..., dtype_backend='numpy_nullable')`` — locking the
in-memory-vs-on-disk parity from both sides.
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
