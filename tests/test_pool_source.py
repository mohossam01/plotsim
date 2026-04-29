"""M114 — PoolSource per-entity value pools.

Covers the full surface added by mission-114 for the new
``pool:<name>`` column source and its paired ``Column.value_pool``
field:

* ``parse_source('pool:industry')`` returns a ``PoolSource`` and rejects
  malformed grammar (empty name, embedded colons).
* ``Column._pool_pairing`` enforces that ``pool:`` source and
  ``value_pool`` are paired or both absent — same discipline as
  ``_scd_pairing``.
* ``Column._pool_pairing`` rejects empty entity-name keys, empty value
  lists, and non-string / empty-string values.
* ``PlotsimConfig._value_pool_gates`` (which delegates to
  ``validate_value_pool_coverage``) rejects:
    - PoolSource on a non-per_entity dim,
    - value_pool missing keys for entities producing rows in the dim,
    - value_pool with extra keys for unknown entities.
* End-to-end: ``build_dim_entity`` honours per-entity pools — each
  entity's row contains only values from its own pool.
* Determinism: same ``(config, seed)`` → identical pool selections
  across two runs (same RNG draws → same indices).
* Re-export: ``PoolSource`` is reachable from ``plotsim`` top-level.
"""
from __future__ import annotations

import warnings

import numpy as np
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
    PoolSource,
    SurrogateKeyWarning,
    Table,
    TimeWindow,
    parse_source,
)
from plotsim.tables import generate_tables


def _make_config(
    entities: list[Entity],
    *,
    dim_entity_columns: list[Column],
) -> PlotsimConfig:
    """Build a minimal PlotsimConfig with caller-supplied entities and
    a configurable per_entity dim. The fact and dim_date layers are
    fixed minimums so the test can focus on the pool surface.
    """
    metric = Metric(
        name="m", label="m",
        distribution="normal", params={"mu": 1.0, "sigma": 0.1},
        polarity="positive",
    )
    arch = Archetype(
        name="flat", label="flat",
        description="constant 0.5 plateau",
        curve_segments=[
            CurveSegment(
                curve="plateau", params={"level": 0.5},
                start_pct=0.0, end_pct=1.0,
            ),
        ],
    )
    fct = Table(
        name="fct_m", type="fact", grain="per_entity_per_period",
        primary_key=["date_key", "entity_id"],
        foreign_keys=["dim_date.date_key", "dim_entity.entity_id"],
        columns=[
            Column(name="date_key", dtype="id", source="fk:dim_date.date_key"),
            Column(name="entity_id", dtype="id", source="fk:dim_entity.entity_id"),
            Column(name="m", dtype="float", source="metric:m"),
        ],
    )
    dim_date = Table(
        name="dim_date", type="dim", grain="per_period",
        primary_key="date_key",
        columns=[
            Column(name="date_key", dtype="id", source="pk"),
            Column(name="date", dtype="date", source="generated:date_key"),
        ],
    )
    dim_entity = Table(
        name="dim_entity", type="dim", grain="per_entity",
        primary_key="entity_id",
        columns=dim_entity_columns,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SurrogateKeyWarning)
        return PlotsimConfig(
            domain=Domain(
                name="t", description="t",
                entity_type="entity", entity_label="Entities",
            ),
            time_window=TimeWindow(
                start="2024-01", end="2024-04", granularity="monthly",
            ),
            seed=42,
            metrics=[metric],
            archetypes=[arch],
            entities=entities,
            tables=[dim_date, dim_entity, fct],
            output=OutputConfig(format="csv", directory="out/m114_pool"),
        )


# --- parse_source grammar ---------------------------------------------------


def test_parse_source_pool_returns_pool_source():
    parsed = parse_source("pool:industry")
    assert isinstance(parsed, PoolSource)
    assert parsed.name == "industry"


def test_parse_source_pool_rejects_empty_name():
    with pytest.raises(ValueError, match="requires a name"):
        parse_source("pool:")


def test_parse_source_pool_rejects_embedded_colons():
    with pytest.raises(ValueError, match="no extra colons"):
        parse_source("pool:industry:extra")


def test_parse_source_pool_rejects_non_identifier_name():
    with pytest.raises(ValueError):
        parse_source("pool:has space")


# --- Column-level pairing validation ----------------------------------------


def _entity_and_id_columns() -> list[Column]:
    return [
        Column(name="entity_id", dtype="id", source="pk"),
    ]


def test_column_pool_source_with_value_pool_loads():
    col = Column(
        name="industry",
        dtype="string",
        source="pool:industry",
        value_pool={"e1": ["saas", "fintech"]},
    )
    assert isinstance(parse_source(col.source), PoolSource)
    assert col.value_pool == {"e1": ["saas", "fintech"]}


def test_column_pool_source_without_value_pool_rejects():
    with pytest.raises(ValueError, match="no 'value_pool' block"):
        Column(name="industry", dtype="string", source="pool:industry")


def test_column_value_pool_without_pool_source_rejects():
    with pytest.raises(ValueError, match="value_pool block but source"):
        Column(
            name="industry",
            dtype="string",
            source="static:foo",
            value_pool={"e1": ["a"]},
        )


def test_column_value_pool_empty_list_rejects():
    with pytest.raises(ValueError, match="is empty"):
        Column(
            name="industry",
            dtype="string",
            source="pool:industry",
            value_pool={"e1": []},
        )


def test_column_value_pool_empty_string_value_rejects():
    with pytest.raises(ValueError, match="non-empty strings"):
        Column(
            name="industry",
            dtype="string",
            source="pool:industry",
            value_pool={"e1": [""]},
        )


# --- Cross-model coverage validation ---------------------------------------


def test_value_pool_missing_entity_key_rejects():
    e1 = Entity(name="e1", archetype="flat", size=1)
    e2 = Entity(name="e2", archetype="flat", size=1)
    pool_col = Column(
        name="industry", dtype="string", source="pool:industry",
        value_pool={"e1": ["saas"]},  # missing e2
    )
    with pytest.raises(ValueError, match=r"missing entries for entities \['e2'\]"):
        _make_config(
            [e1, e2],
            dim_entity_columns=_entity_and_id_columns() + [pool_col],
        )


def test_value_pool_extra_entity_key_rejects():
    e1 = Entity(name="e1", archetype="flat", size=1)
    pool_col = Column(
        name="industry", dtype="string", source="pool:industry",
        value_pool={"e1": ["saas"], "ghost": ["fintech"]},
    )
    with pytest.raises(ValueError, match=r"unknown entities \['ghost'\]"):
        _make_config(
            [e1],
            dim_entity_columns=_entity_and_id_columns() + [pool_col],
        )


def test_pool_source_on_non_per_entity_dim_rejects():
    # Place a PoolSource on a per_reference dim — coverage gate refuses
    # because there's no per-entity 1:1 binding to look up against.
    e1 = Entity(name="e1", archetype="flat", size=1)
    metric = Metric(
        name="m", label="m",
        distribution="normal", params={"mu": 1.0, "sigma": 0.1},
        polarity="positive",
    )
    arch = Archetype(
        name="flat", label="flat",
        description="constant 0.5 plateau",
        curve_segments=[
            CurveSegment(
                curve="plateau", params={"level": 0.5},
                start_pct=0.0, end_pct=1.0,
            ),
        ],
    )
    dim_date = Table(
        name="dim_date", type="dim", grain="per_period",
        primary_key="date_key",
        columns=[
            Column(name="date_key", dtype="id", source="pk"),
            Column(name="date", dtype="date", source="generated:date_key"),
        ],
    )
    dim_entity = Table(
        name="dim_entity", type="dim", grain="per_entity",
        primary_key="entity_id",
        columns=[Column(name="entity_id", dtype="id", source="pk")],
    )
    dim_ref = Table(
        name="dim_plan", type="dim", grain="per_reference",
        primary_key="plan_id",
        columns=[
            Column(name="plan_id", dtype="id", source="pk"),
            Column(
                name="plan_kind", dtype="string",
                source="pool:plan_kind",
                value_pool={"e1": ["starter"]},
            ),
        ],
    )
    fct = Table(
        name="fct_m", type="fact", grain="per_entity_per_period",
        primary_key=["date_key", "entity_id"],
        foreign_keys=["dim_date.date_key", "dim_entity.entity_id"],
        columns=[
            Column(name="date_key", dtype="id", source="fk:dim_date.date_key"),
            Column(name="entity_id", dtype="id", source="fk:dim_entity.entity_id"),
            Column(name="m", dtype="float", source="metric:m"),
        ],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SurrogateKeyWarning)
        with pytest.raises(ValueError, match="not a per_entity dim"):
            PlotsimConfig(
                domain=Domain(
                    name="t", description="t",
                    entity_type="entity", entity_label="Entities",
                ),
                time_window=TimeWindow(
                    start="2024-01", end="2024-04", granularity="monthly",
                ),
                seed=0,
                metrics=[metric],
                archetypes=[arch],
                entities=[e1],
                tables=[dim_date, dim_entity, dim_ref, fct],
                output=OutputConfig(format="csv", directory="out/m114_bad_pool"),
            )


# --- End-to-end: pool sampling honors per-entity bindings -------------------


def test_each_entity_draws_only_from_its_own_pool():
    """Two cohorts, disjoint pools. Every row in each cohort's slice of
    dim_entity must contain a value from that cohort's pool — never the
    other cohort's pool. Per_entity grain emits one row per entity, so
    we have two rows total."""
    e1 = Entity(name="e1", archetype="flat", size=1)
    e2 = Entity(name="e2", archetype="flat", size=1)
    pool_col = Column(
        name="industry", dtype="string", source="pool:industry",
        value_pool={
            "e1": ["saas", "fintech"],
            "e2": ["retail", "b2c"],
        },
    )
    cfg = _make_config(
        [e1, e2],
        dim_entity_columns=_entity_and_id_columns() + [pool_col],
    )
    tables = generate_tables(cfg, np.random.default_rng(cfg.seed))
    dim_entity = tables["dim_entity"]
    assert len(dim_entity) == 2
    # entity_id is a sequential id, but the iteration order matches
    # config.entities order, so row 0 → e1, row 1 → e2.
    assert dim_entity.iloc[0]["industry"] in {"saas", "fintech"}
    assert dim_entity.iloc[1]["industry"] in {"retail", "b2c"}


def test_pool_sampling_is_deterministic_across_runs():
    """Same (config, seed) → identical industry selections across two
    independent runs. The single-RNG contract makes this tight."""
    e1 = Entity(name="e1", archetype="flat", size=1)
    e2 = Entity(name="e2", archetype="flat", size=1)
    e3 = Entity(name="e3", archetype="flat", size=1)
    pool_col = Column(
        name="industry", dtype="string", source="pool:industry",
        value_pool={
            "e1": ["a1", "a2", "a3", "a4"],
            "e2": ["b1", "b2", "b3", "b4"],
            "e3": ["c1", "c2", "c3", "c4"],
        },
    )
    cfg = _make_config(
        [e1, e2, e3],
        dim_entity_columns=_entity_and_id_columns() + [pool_col],
    )
    tables_a = generate_tables(cfg, np.random.default_rng(cfg.seed))
    tables_b = generate_tables(cfg, np.random.default_rng(cfg.seed))
    np.testing.assert_array_equal(
        tables_a["dim_entity"]["industry"].to_numpy(),
        tables_b["dim_entity"]["industry"].to_numpy(),
    )


def test_pool_source_re_exported_from_plotsim():
    """``PoolSource`` is in the public ``plotsim`` namespace, mirroring
    the ``TextBucketSource`` precedent (M105)."""
    assert plotsim.PoolSource is PoolSource
    assert "PoolSource" in plotsim.__all__
