"""F16 (M102) — source-type coverage fixture for per_entity_per_period facts.

The bundled templates exercise ``MetricSource`` heavily on per_entity_per_period
fact tables but leave ``PKSource``, ``GeneratedSource(date_key|period_label|
timestamp)``, ``StaticSource``, ``DerivedSource(period_index|entity_id)``, and
the cross-dim ``FKSource`` branch sparsely covered or uncovered on that grain.
The vectorized dispatch in
:func:`plotsim.tables._vectorized_per_entity_per_period_fact` and the scalar
dispatch in :func:`plotsim.tables._resolve_fact_cell` route on the
``parse_source`` return type, so any new source type lands in those exact
ladders. Adding a new branch where existing branches aren't tested ships two
bugs for one.

This test consumes a synthetic, non-bundled YAML
(``tests/fixtures/source_type_coverage_config.yaml``) that places every
parse_source branch reachable on a per_entity_per_period fact onto two fact
tables: ``fct_vectorized`` (no FakerSource, takes the vectorized path) and
``fct_scalar`` (carries ``generated:faker.sentence``, which forces the scalar
fallback). Together they exercise both halves of the dispatch.

The mission's acceptance gate for F16:

* No errors generating from the synthetic config.
* Every column has the expected dtype.
* Determinism: two runs at the same seed produce byte-identical CSV.
* Coverage on ``plotsim.tables`` strictly increases line and branch
  numbers vs the M101 baseline (line 67% / branch 61%) — measured
  separately and cited in the CHANGELOG, since cross-test coverage
  comparisons are awkward inside a single test run.

The fixture is locked-by-construction (no expected-bytes file). Tightening to
expected-bytes parity is deliberately deferred — the synthetic config's
purpose is reaching the dispatch branches, not pinning their numerical
output, and a saved-bytes assertion would brittle-up against unrelated
metric / curve changes elsewhere in the codebase.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from plotsim import generate_tables, load_config, write_tables
from plotsim.config import (
    DerivedSource,
    FakerSource,
    FKSource,
    GeneratedSource,
    LagSource,
    MetricSource,
    PKSource,
    StaticSource,
    SurrogateKeyWarning,
    TextBucketSource,
    parse_source,
)


ROOT = Path(__file__).resolve().parent.parent
SYNTHETIC_YAML = ROOT / "tests" / "fixtures" / "source_type_coverage_config.yaml"


@pytest.fixture(scope="module")
def synthetic_cfg():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SurrogateKeyWarning)
        return load_config(SYNTHETIC_YAML)


@pytest.fixture(scope="module")
def synthetic_tables(synthetic_cfg):
    rng = np.random.default_rng(synthetic_cfg.seed)
    return generate_tables(synthetic_cfg, rng)


# --- (1) generation succeeds and the schema lands as declared ---------------


def test_synthetic_config_loads_and_every_table_present(synthetic_cfg, synthetic_tables):
    """Every dim + fact table declared in the YAML appears in the output."""
    declared = {t.name for t in synthetic_cfg.tables}
    assert set(synthetic_tables) == declared, (
        f"missing tables: {declared - set(synthetic_tables)!r}; "
        f"extra: {set(synthetic_tables) - declared!r}"
    )


def test_fact_tables_have_expected_row_count(synthetic_cfg, synthetic_tables):
    """3 entities × 12 monthly periods = 36 rows on each per_entity_per_period fact."""
    n_entities = len(synthetic_cfg.entities)
    n_periods = synthetic_cfg.time_window.period_count()
    expected_rows = n_entities * n_periods
    for tbl_name in ("fct_vectorized", "fct_scalar"):
        df = synthetic_tables[tbl_name]
        assert len(df) == expected_rows, (
            f"{tbl_name}: got {len(df)} rows, expected {expected_rows} "
            f"({n_entities} entities × {n_periods} periods)"
        )


# --- (2) every column lands with the expected dtype kind --------------------


_VECTORIZED_DTYPE_CHECKS: dict[str, str] = {
    "row_id": "object",  # PKSource → "row_id-NNNN-<entity_pk>"
    "widget_id": "object",  # FKSource (local entity)
    "date_key": "integer",  # FKSource (local date) — int from dim_date
    "plan_id": "object",  # FKSource (cross-dim) — string PK
    "engagement_score": "float",  # MetricSource (float)
    "support_ticket_count": "integer",  # MetricSource (int) — nullable Int64
    "engagement_lag1": "float",  # LagSource (float)
    "support_lag2": "integer",  # LagSource (int) — nullable Int64
    "event_ts": "datetime",  # GeneratedSource(timestamp)
    "period_dk": "integer",  # GeneratedSource(date_key)
    "period_lbl": "object",  # GeneratedSource(period_label)
    "region": "object",  # StaticSource
    "p_idx": "integer",  # DerivedSource(period_index)
    "entity_label": "object",  # DerivedSource(entity_id)
    "sentiment": "object",  # M105: TextBucketSource (vectorized)
}

_SCALAR_DTYPE_CHECKS: dict[str, str] = {
    "row_id": "object",  # PKSource (scalar)
    "widget_id": "object",  # FKSource (local entity, scalar)
    "date_key": "integer",  # FKSource (local date, scalar)
    "plan_id": "object",  # FKSource (cross-dim, scalar)
    "engagement_score": "float",  # MetricSource (scalar)
    "engagement_lag1": "float",  # LagSource (scalar)
    "note": "object",  # FakerSource — forces the scalar path
    "region": "object",  # StaticSource (scalar)
    "event_ts": "datetime",  # GeneratedSource(timestamp, scalar)
    "period_dk": "integer",  # GeneratedSource(date_key, scalar)
    "period_lbl": "object",  # GeneratedSource(period_label, scalar)
    "p_idx": "integer",  # DerivedSource(period_index, scalar)
    "entity_label": "object",  # DerivedSource(entity_id, scalar)
    "sentiment": "object",  # M105: TextBucketSource (scalar)
}


def _dtype_kind(series: pd.Series) -> str:
    if pd.api.types.is_integer_dtype(series.dtype):
        return "integer"
    if pd.api.types.is_float_dtype(series.dtype):
        return "float"
    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        return "datetime"
    if pd.api.types.is_bool_dtype(series.dtype):
        return "boolean"
    return "object"


@pytest.mark.parametrize("col_name,expected_kind", sorted(_VECTORIZED_DTYPE_CHECKS.items()))
def test_vectorized_column_dtype_matches_expected(synthetic_tables, col_name, expected_kind):
    df = synthetic_tables["fct_vectorized"]
    assert col_name in df.columns, f"fct_vectorized missing column {col_name!r}"
    actual = _dtype_kind(df[col_name])
    assert actual == expected_kind, (
        f"fct_vectorized.{col_name}: dtype {df[col_name].dtype!r} maps to "
        f"{actual!r}, expected {expected_kind!r}"
    )


@pytest.mark.parametrize("col_name,expected_kind", sorted(_SCALAR_DTYPE_CHECKS.items()))
def test_scalar_column_dtype_matches_expected(synthetic_tables, col_name, expected_kind):
    df = synthetic_tables["fct_scalar"]
    assert col_name in df.columns, f"fct_scalar missing column {col_name!r}"
    actual = _dtype_kind(df[col_name])
    assert actual == expected_kind, (
        f"fct_scalar.{col_name}: dtype {df[col_name].dtype!r} maps to "
        f"{actual!r}, expected {expected_kind!r}"
    )


# --- (3) every parse_source branch is genuinely exercised -------------------


def test_vectorized_table_exercises_every_reachable_source_branch(synthetic_cfg):
    """Confirm the YAML places at least one column per parse_source branch
    on the vectorized fact table. If a future edit drops one of these
    columns, the test fails — protecting the dispatch coverage F16 buys.
    """
    tbl = next(t for t in synthetic_cfg.tables if t.name == "fct_vectorized")
    parsed_types: list[type] = []
    generated_providers: set[str] = set()
    derived_fields: set[str] = set()
    for col in tbl.columns:
        parsed = parse_source(col.source)
        parsed_types.append(type(parsed))
        if isinstance(parsed, GeneratedSource):
            generated_providers.add(parsed.provider)
        if isinstance(parsed, DerivedSource):
            derived_fields.add(parsed.field)
    assert PKSource in parsed_types
    assert FKSource in parsed_types
    assert MetricSource in parsed_types
    assert LagSource in parsed_types
    assert GeneratedSource in parsed_types
    assert StaticSource in parsed_types
    assert DerivedSource in parsed_types
    # M105: TextBucketSource lives on the vectorized fact dispatch.
    assert TextBucketSource in parsed_types
    assert generated_providers == {"timestamp", "date_key", "period_label"}
    assert derived_fields == {"period_index", "entity_id"}
    # Vectorized path should NOT carry FakerSource — that forces scalar fallback.
    assert FakerSource not in parsed_types


def test_scalar_table_carries_faker_to_force_scalar_path(synthetic_cfg):
    """fct_scalar must include at least one FakerSource column so
    _build_per_entity_per_period_fact routes it through the scalar
    fallback. Without that column the table would be vectorized and
    _resolve_fact_cell coverage would not increase.
    """
    tbl = next(t for t in synthetic_cfg.tables if t.name == "fct_scalar")
    parsed_types = [type(parse_source(col.source)) for col in tbl.columns]
    assert FakerSource in parsed_types, (
        "fct_scalar lost its FakerSource column — fct_scalar is now "
        "vectorized and the scalar dispatch is no longer covered."
    )
    # M105: TextBucketSource must also land on the scalar table so the
    # _resolve_fact_cell branch is exercised (the vectorized branch is
    # covered by fct_vectorized).
    assert TextBucketSource in parsed_types, (
        "fct_scalar lost its TextBucketSource column — the scalar "
        "_resolve_fact_cell branch for text:bucket is no longer covered."
    )


# --- (4) determinism — byte-identical CSVs across two runs ------------------


def _run_and_write(cfg, dst: Path) -> dict[str, bytes]:
    rng = np.random.default_rng(cfg.seed)
    tables = generate_tables(cfg, rng)
    write_tables(tables, cfg, output_dir=dst)
    csvs: dict[str, bytes] = {}
    for tbl in cfg.tables:
        csv_path = dst / f"{tbl.name}.csv"
        if csv_path.exists():
            csvs[tbl.name] = csv_path.read_bytes()
    return csvs


def test_synthetic_config_produces_byte_identical_csvs_across_runs(synthetic_cfg, tmp_path):
    """Two runs at the same seed produce byte-identical CSVs. Locks
    determinism end-to-end across every parse_source branch — the
    invariant the bundled-template parity tests guarantee on a narrower
    surface.

    Note: ``write_tables`` also writes ``validation_report.txt``. F5
    made that file deterministic by default (config-fingerprint instead
    of wall-clock timestamp), so this test compares all written files
    including the report.
    """
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    csvs_a = _run_and_write(synthetic_cfg, run_a)
    csvs_b = _run_and_write(synthetic_cfg, run_b)
    assert sorted(csvs_a) == sorted(csvs_b)
    for name in csvs_a:
        assert csvs_a[name] == csvs_b[name], (
            f"{name}.csv differs between two runs at seed "
            f"{synthetic_cfg.seed} — determinism contract broken on "
            f"the source-type coverage fixture."
        )
    report_a = (run_a / "validation_report.txt").read_bytes()
    report_b = (run_b / "validation_report.txt").read_bytes()
    assert report_a == report_b, (
        "validation_report.txt differs across runs — F5's deterministic " "default has regressed."
    )


# --- (5) source-type semantics — values match the dispatch contract ---------


def test_pk_source_values_are_unique_and_period_keyed(synthetic_tables):
    """PKSource builds f'{col.name}-{period:04d}-{entity_pk}' on the
    vectorized path and the same on the scalar path. Values must be
    unique per (entity, period) row, and the encoded period must match
    the row's period_index.
    """
    df = synthetic_tables["fct_vectorized"].copy()
    assert df["row_id"].is_unique
    # Each row_id encodes the period. Pull it back out and assert it
    # matches p_idx (DerivedSource(period_index)).
    parsed_periods = df["row_id"].str.extract(r"row_id-(\d{4})-")[0].astype(int)
    assert (parsed_periods.to_numpy() == df["p_idx"].to_numpy()).all(), (
        "PKSource period-encoding diverges from DerivedSource(period_index) — "
        "vectorized fact-builder is mis-aligning the two."
    )


def test_derived_entity_id_matches_local_entity_fk(synthetic_tables):
    """DerivedSource(entity_id) must broadcast the same entity PK as
    the local-entity FKSource on the same row. Vectorized path shares
    ``entity_pk_repeated`` between the two branches; scalar path passes
    ``entity_pk_value`` to both. Drift here would mean the dispatch is
    reading entity identity from two different sources.
    """
    for tbl_name in ("fct_vectorized", "fct_scalar"):
        df = synthetic_tables[tbl_name]
        assert (
            df["entity_label"] == df["widget_id"]
        ).all(), f"{tbl_name}: derived:entity_id != fk:dim_widget.widget_id"


def test_generated_date_key_matches_local_date_fk(synthetic_tables):
    """GeneratedSource(date_key) must produce the same value as the
    local-date FKSource — both read from dim_date.date_key for the
    row's period. Drift would mean GeneratedSource is indexing dim_date
    independently from the FK broadcast.
    """
    for tbl_name in ("fct_vectorized", "fct_scalar"):
        df = synthetic_tables[tbl_name]
        # Both should be int (date_key dtype). Compare element-wise.
        assert (
            df["period_dk"].to_numpy() == df["date_key"].to_numpy()
        ).all(), f"{tbl_name}: generated:date_key != fk:dim_date.date_key"


def test_static_source_values_are_constant(synthetic_tables):
    """StaticSource must broadcast the configured value across every
    row of the fact table. Vectorized path uses ``np.full(...)``; scalar
    path returns ``parsed.value`` per cell.
    """
    vec = synthetic_tables["fct_vectorized"]
    assert (
        vec["region"] == "emea"
    ).all(), "fct_vectorized.region not uniformly 'emea' — StaticSource broken"
    sca = synthetic_tables["fct_scalar"]
    assert (
        sca["region"] == "scalar_region"
    ).all(), "fct_scalar.region not uniformly 'scalar_region' — StaticSource broken"


def test_lag_source_falls_back_to_current_period_for_short_history(synthetic_tables):
    """LagSource at periods=1 means period 0 has no history → falls
    back to the current period's value. periods=2 means periods 0 and
    1 fall back. This invariant is shared by the vectorized
    (``np.where(target_idx < 0, base, target_idx)``) and scalar
    (``if target_idx < 0: target_idx = period_idx``) paths.
    """
    df = synthetic_tables["fct_vectorized"].sort_values(["widget_id", "p_idx"])
    # For each entity, period-0 engagement_lag1 should equal period-0
    # engagement_score (fall-back). Same for support_lag2 across periods 0–1.
    for entity_pk, group in df.groupby("widget_id", sort=False):
        period0 = group.iloc[0]
        assert (
            period0["engagement_lag1"] == period0["engagement_score"]
        ), f"{entity_pk}: lag=1 fall-back failed at period 0"
        for k in (0, 1):
            period_k = group.iloc[k]
            assert (
                period_k["support_lag2"] == period_k["support_ticket_count"]
            ), f"{entity_pk}: lag=2 fall-back failed at period {k}"


def test_faker_source_values_are_non_empty_strings(synthetic_tables):
    """The FakerSource ``generated:faker.sentence`` column on
    fct_scalar must hold non-empty strings — confirms the scalar
    fallback is invoking Faker, not silently filling None.
    """
    notes = synthetic_tables["fct_scalar"]["note"]
    assert notes.notna().all(), "fct_scalar.note has nulls — FakerSource silent-failed"
    assert (notes.str.len() > 0).all(), "fct_scalar.note has empty strings"
