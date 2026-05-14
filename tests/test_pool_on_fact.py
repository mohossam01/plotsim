"""PoolSource on per_entity_per_period fact tables.

Widens the M114 / M19-fix-1 pool-source surface to include the most
common fact grain: one row per (entity, period). Before this change,
``validate_value_pool_coverage`` rejected the combination at load and
no dispatch handler existed in ``COLUMN_DISPATCH`` for
``BuilderKind.PER_ENTITY_PER_PERIOD_FACT_{SCALAR,VECTORIZED}`` × ``PoolSource``.

Coverage:

* End-to-end: a fact column with ``pool:`` source emits a value drawn
  from the row's entity's own pool — entities never cross-contaminate.
* Determinism: same ``(config, seed)`` → identical column.
* Forced-scalar path: when a Faker column is on the same fact, the
  builder switches to the scalar path; the pool column must still
  resolve correctly there.
* Per_period fact (the dim_date-style grain) is still rejected — the
  widening doesn't accidentally cover grains without an entity binding.
"""

from __future__ import annotations

import warnings

import pytest

import plotsim
from plotsim.config import (
    Archetype,
    Column,
    CurveSegment,
    Domain,
    Entity,
    Metric,
    OutputConfig,
    PlotsimConfig,
    SurrogateKeyWarning,
    Table,
    TimeWindow,
)
from plotsim.tables import generate_tables


def _flat_archetype() -> Archetype:
    return Archetype(
        name="flat",
        label="flat",
        description="constant 0.5 plateau",
        curve_segments=[
            CurveSegment(
                curve="plateau",
                params={"level": 0.5},
                start_pct=0.0,
                end_pct=1.0,
            ),
        ],
    )


def _config_with_pool_fact_column(
    *,
    seed: int = 42,
    extra_fact_cols: list[Column] | None = None,
) -> PlotsimConfig:
    """Build a minimal config: 2 entities × 4 periods, fact has
    metric + pool column (and optionally any caller-supplied extras)."""
    entities = [
        Entity(name="cohort_a", archetype="flat", size=1),
        Entity(name="cohort_b", archetype="flat", size=1),
    ]
    pool_col = Column(
        name="payment_type",
        dtype="string",
        source="pool:payment_type",
        value_pool={
            "cohort_a": ["card", "cash"],
            "cohort_b": ["online", "wallet"],
        },
    )
    fact_cols = [
        Column(name="date_key", dtype="id", source="fk:dim_date.date_key"),
        Column(name="entity_id", dtype="id", source="fk:dim_entity.entity_id"),
        Column(name="m", dtype="float", source="metric:m"),
        pool_col,
    ]
    if extra_fact_cols:
        fact_cols.extend(extra_fact_cols)
    fct = Table(
        name="fct_m",
        type="fact",
        grain="per_entity_per_period",
        primary_key=["date_key", "entity_id"],
        foreign_keys=["dim_date.date_key", "dim_entity.entity_id"],
        columns=fact_cols,
    )
    dim_date = Table(
        name="dim_date",
        type="dim",
        grain="per_period",
        primary_key="date_key",
        columns=[
            Column(name="date_key", dtype="id", source="pk"),
            Column(name="date", dtype="date", source="generated:date_key"),
        ],
    )
    dim_entity = Table(
        name="dim_entity",
        type="dim",
        grain="per_entity",
        primary_key="entity_id",
        columns=[Column(name="entity_id", dtype="id", source="pk")],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SurrogateKeyWarning)
        return PlotsimConfig(
            domain=Domain(
                name="t",
                description="t",
                entity_type="entity",
                entity_label="Entities",
            ),
            time_window=TimeWindow(
                start="2024-01",
                end="2024-04",
                granularity="monthly",
            ),
            seed=seed,
            metrics=[
                Metric(
                    name="m",
                    label="m",
                    distribution="normal",
                    params={"mu": 1.0, "sigma": 0.1},
                    polarity="positive",
                ),
            ],
            archetypes=[_flat_archetype()],
            entities=entities,
            tables=[dim_date, dim_entity, fct],
            output=OutputConfig(format="csv", directory="out/test_pool_fact"),
        )


def _assert_pools_honored(fact_df, dim_entity_df):
    """Every row's payment_type ∈ that row's entity's declared pool."""
    expected = {
        "cohort_a": {"card", "cash"},
        "cohort_b": {"online", "wallet"},
    }
    # dim_entity rows are in config.entities order.
    pk_to_name = {
        dim_entity_df.iloc[i]["entity_id"]: ["cohort_a", "cohort_b"][i]
        for i in range(len(dim_entity_df))
    }
    for _, row in fact_df.iterrows():
        entity_name = pk_to_name[row["entity_id"]]
        assert row["payment_type"] in expected[entity_name], (
            f"row entity_id={row['entity_id']} (cohort {entity_name}) drew "
            f"payment_type={row['payment_type']!r}, not in pool "
            f"{expected[entity_name]}"
        )


def test_pool_on_per_entity_per_period_fact_loads_and_dispatches():
    """Build the config (was rejected pre-fix) and generate without error."""
    cfg = _config_with_pool_fact_column()
    tables = generate_tables(cfg)
    fact = tables["fct_m"]
    assert len(fact) == 8  # 2 entities × 4 periods
    assert "payment_type" in fact.columns


def test_pool_values_match_per_row_entity_pool_vec_path():
    """Vectorized path: no Faker column on the fact, so ``forces_scalar``
    is False and the vec dispatcher runs ``_fact_vec_pool``."""
    cfg = _config_with_pool_fact_column()
    tables = generate_tables(cfg)
    _assert_pools_honored(tables["fct_m"], tables["dim_entity"])


def test_pool_values_match_per_row_entity_pool_scalar_path():
    """Forced-scalar path: a Faker column flips ``forces_scalar=True``
    so every column (including pool) routes through the scalar
    dispatcher and ``_fact_scalar_pool``."""
    cfg = _config_with_pool_fact_column(
        extra_fact_cols=[
            Column(name="note", dtype="string", source="generated:faker.word"),
        ],
    )
    tables = generate_tables(cfg)
    _assert_pools_honored(tables["fct_m"], tables["dim_entity"])


def test_pool_on_fact_is_deterministic_under_seed():
    """Same seed → byte-identical column across two independent builds."""
    cfg_a = _config_with_pool_fact_column(seed=2026)
    cfg_b = _config_with_pool_fact_column(seed=2026)
    fa = generate_tables(cfg_a)["fct_m"]["payment_type"].tolist()
    fb = generate_tables(cfg_b)["fct_m"]["payment_type"].tolist()
    assert fa == fb


def test_pool_on_fact_seed_change_changes_draws():
    """Sanity: changing seed shifts the draws (rules out a constant-
    return bug that would also pass the determinism test)."""
    fa = generate_tables(_config_with_pool_fact_column(seed=1))["fct_m"]["payment_type"].tolist()
    fb = generate_tables(_config_with_pool_fact_column(seed=2))["fct_m"]["payment_type"].tolist()
    assert fa != fb


def test_pool_still_rejected_on_per_period_fact():
    """The widening adds per_entity_per_period — per_period (the
    dim_date-style grain) still has no per-row entity binding, so
    the validator must keep rejecting it."""
    entity = Entity(name="e1", archetype="flat", size=1)
    pool_col = Column(
        name="payment_type",
        dtype="string",
        source="pool:payment_type",
        value_pool={"e1": ["card"]},
    )
    bad_per_period_fact = Table(
        name="fct_period",
        type="fact",
        grain="per_period",
        primary_key="date_key",
        foreign_keys=["dim_date.date_key"],
        columns=[
            Column(name="date_key", dtype="id", source="fk:dim_date.date_key"),
            pool_col,
        ],
    )
    dim_date = Table(
        name="dim_date",
        type="dim",
        grain="per_period",
        primary_key="date_key",
        columns=[
            Column(name="date_key", dtype="id", source="pk"),
            Column(name="date", dtype="date", source="generated:date_key"),
        ],
    )
    dim_entity = Table(
        name="dim_entity",
        type="dim",
        grain="per_entity",
        primary_key="entity_id",
        columns=[Column(name="entity_id", dtype="id", source="pk")],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SurrogateKeyWarning)
        with pytest.raises(ValueError, match="per_entity_per_period fact"):
            PlotsimConfig(
                domain=Domain(
                    name="t",
                    description="t",
                    entity_type="entity",
                    entity_label="Entities",
                ),
                time_window=TimeWindow(
                    start="2024-01",
                    end="2024-04",
                    granularity="monthly",
                ),
                seed=0,
                metrics=[
                    Metric(
                        name="m",
                        label="m",
                        distribution="normal",
                        params={"mu": 1.0, "sigma": 0.1},
                        polarity="positive",
                    ),
                ],
                archetypes=[_flat_archetype()],
                entities=[entity],
                tables=[dim_date, dim_entity, bad_per_period_fact],
                output=OutputConfig(
                    format="csv",
                    directory="out/test_pool_per_period",
                ),
            )


def test_pool_source_reexport_still_works():
    """Smoke test the top-level re-export — same as the M114 surface."""
    assert plotsim.PoolSource is not None
