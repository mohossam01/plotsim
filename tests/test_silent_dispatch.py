"""F14 regression — silent-dispatch audit (M102).

Mission 100 named ``_build_per_period_fact`` (``tables.py:644``) as a
silent-dispatch site: an isinstance ladder over the parsed-source
Union with an ``else: row[col.name] = None`` fall-through. An
unhandled source type produced a column of ``None`` values with no
signal to the user — same bug class as F1's ``_resolve_event_row``
fallback to ``candidates.iloc[0]``.

F14 sweep audited every isinstance ladder over the parse_source
Union in ``plotsim/`` and confirmed the existing raises in
``_resolve_fact_cell`` (``tables.py:574``) and the per-entity /
sub-entity / reference dim builders in ``plotsim/dimensions.py``.
Two silent-fallback sites were converted to explicit raises in
``tables.py``:

* ``_build_per_period_fact`` outer ``else`` (line 698 pre-fix) →
  raises ``TypeError`` naming the column, source, and source-class.
* ``_build_proportional_event`` deterministic-dispatch outer
  ``else`` (line 995 pre-fix) → raises ``TypeError``. The inner
  ``DerivedSource.field`` unhandled branch (line 993 pre-fix) was
  also converted; it now raises ``ValueError`` naming the bad
  field.

Tests:

* ``test_per_period_fact_raises_on_unhandled_source_type`` —
  per_period fact column with ``source: "derived:nonsense"``
  (DerivedSource is not in the per_period_fact dispatch ladder).
  Post-fix raises ``TypeError`` with the column name.
* ``test_event_builder_raises_on_unhandled_source_type`` —
  proportional-event column with ``source: "metric:m1"`` (
  MetricSource is not in the deterministic-event dispatch and is
  not classified as stochastic). Post-fix raises ``TypeError``.
* ``test_event_builder_raises_on_unrecognized_derived_field`` —
  event column with ``source: "derived:nonsense"`` (
  DerivedSource only handles ``entity_id`` / ``date_key`` on event
  tables). Post-fix raises ``ValueError`` naming the bad field.
* ``test_resolve_fact_cell_existing_raise_unaffected`` —
  ``_resolve_fact_cell`` already raised on unhandled cases pre-F14;
  confirm the F14 sweep didn't regress it.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from plotsim import generate_tables
from plotsim.config import (
    Archetype,
    Column,
    CurveSegment,
    Domain,
    Entity,
    Metric,
    OutputConfig,
    PlotsimConfig,
    ProportionalSource,
    SurrogateKeyWarning,
    Table,
    TimeWindow,
)


def _base_config(
    *,
    extra_per_period_columns: list[Column] | None = None,
    extra_event_columns: list[Column] | None = None,
    event_row_count_source: str | None = None,
) -> PlotsimConfig:
    """Build a minimal config with a per-period fact and an optional
    event table. The caller decides which columns to add to test
    silent-dispatch behavior."""
    metrics = [
        Metric(
            name="m1", label="m1",
            distribution="normal", params={"mu": 1.0, "sigma": 0.1},
            polarity="positive",
        ),
    ]
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
        columns=[
            Column(name="entity_id", dtype="id", source="pk"),
        ],
    )
    # Per-entity-per-period fact (the trajectory-driven backbone).
    fct_eep = Table(
        name="fct_eep", type="fact", grain="per_entity_per_period",
        primary_key=["date_key", "entity_id"],
        foreign_keys=["dim_date.date_key", "dim_entity.entity_id"],
        columns=[
            Column(name="date_key", dtype="id", source="fk:dim_date.date_key"),
            Column(name="entity_id", dtype="id", source="fk:dim_entity.entity_id"),
            Column(name="m1", dtype="float", source="metric:m1"),
        ],
    )
    # Per-period fact (no entity axis) — the F14 silent-dispatch target.
    fct_per_period_columns = [
        Column(name="date_key", dtype="id", source="fk:dim_date.date_key"),
        Column(name="m1_avg", dtype="float", source="metric:m1"),
    ]
    if extra_per_period_columns:
        fct_per_period_columns.extend(extra_per_period_columns)
    fct_per_period = Table(
        name="fct_per_period", type="fact", grain="per_period",
        primary_key="date_key",
        foreign_keys=["dim_date.date_key"],
        columns=fct_per_period_columns,
    )
    tables = [dim_date, dim_entity, fct_eep, fct_per_period]

    if extra_event_columns or event_row_count_source:
        evt_columns = [
            Column(name="event_id", dtype="id", source="pk"),
            Column(name="date_key", dtype="id", source="fk:dim_date.date_key"),
            Column(name="entity_id", dtype="id", source="fk:dim_entity.entity_id"),
        ]
        if extra_event_columns:
            evt_columns.extend(extra_event_columns)
        evt = Table(
            name="evt_test", type="event", grain="variable",
            primary_key="event_id",
            foreign_keys=["dim_date.date_key", "dim_entity.entity_id"],
            columns=evt_columns,
            row_count_source=event_row_count_source,
        )
        tables.append(evt)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SurrogateKeyWarning)
        return PlotsimConfig(
            domain=Domain(
                name="t", description="t",
                entity_type="entity", entity_label="Entities",
            ),
            time_window=TimeWindow(
                start="2024-01", end="2024-06", granularity="monthly",
            ),
            seed=0,
            metrics=metrics,
            archetypes=[arch],
            entities=[Entity(name="e1", archetype="flat", size=2)],
            tables=tables,
            output=OutputConfig(format="csv", directory="out/f14"),
        )


# --- Per-period fact silent-dispatch ----------------------------------------


def test_per_period_fact_raises_on_unhandled_source_type():
    """A DerivedSource on a per_period fact column hits the F14
    raise — pre-F14 it silently filled the column with None."""
    cfg = _base_config(
        extra_per_period_columns=[
            Column(
                name="bogus_derived",
                dtype="string",
                source="derived:nonsense",
            ),
        ],
    )
    with pytest.raises(TypeError) as exc_info:
        generate_tables(cfg, np.random.default_rng(0))
    msg = str(exc_info.value)
    assert "bogus_derived" in msg
    assert "derived:nonsense" in msg
    assert "DerivedSource" in msg


# --- Event-builder outer-else dispatch --------------------------------------


def test_event_builder_raises_on_unhandled_source_type():
    """A MetricSource on a proportional event column hits the F14
    outer-else raise. ``_is_stochastic`` does not classify
    MetricSource as stochastic, so it falls into the deterministic
    dispatch and triggers the silent fallback pre-F14."""
    cfg = _base_config(
        extra_event_columns=[
            Column(name="m1_snap", dtype="float", source="metric:m1"),
        ],
        event_row_count_source="proportional:m1:scale:5",
    )
    with pytest.raises(TypeError) as exc_info:
        generate_tables(cfg, np.random.default_rng(0))
    msg = str(exc_info.value)
    assert "m1_snap" in msg
    assert "MetricSource" in msg


# --- Event-builder DerivedSource.field unhandled ----------------------------


def test_event_builder_raises_on_unrecognized_derived_field():
    """``DerivedSource.field`` is only ``entity_id`` / ``date_key`` on
    event tables. Anything else used to silently produce a column
    of None; F14 raises."""
    cfg = _base_config(
        extra_event_columns=[
            Column(
                name="bogus_derived",
                dtype="string",
                source="derived:nonsense",
            ),
        ],
        event_row_count_source="proportional:m1:scale:5",
    )
    with pytest.raises(ValueError) as exc_info:
        generate_tables(cfg, np.random.default_rng(0))
    msg = str(exc_info.value)
    assert "bogus_derived" in msg
    assert "nonsense" in msg


# --- Pre-existing raise paths unaffected ------------------------------------


def test_resolve_fact_cell_existing_raise_unaffected():
    """``_resolve_fact_cell`` already raised on unhandled sources
    pre-F14; confirm F14 didn't regress it. Construction: a
    per_entity_per_period fact column with a ``derived:bogus``
    source still raises a clear ValueError."""
    metrics = [
        Metric(
            name="m1", label="m1",
            distribution="normal", params={"mu": 1.0, "sigma": 0.1},
            polarity="positive",
        ),
    ]
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
        columns=[
            Column(name="entity_id", dtype="id", source="pk"),
        ],
    )
    fct = Table(
        name="fct_m1", type="fact", grain="per_entity_per_period",
        primary_key=["date_key", "entity_id"],
        foreign_keys=["dim_date.date_key", "dim_entity.entity_id"],
        columns=[
            Column(name="date_key", dtype="id", source="fk:dim_date.date_key"),
            Column(name="entity_id", dtype="id", source="fk:dim_entity.entity_id"),
            Column(name="m1", dtype="float", source="metric:m1"),
            Column(
                name="bogus", dtype="string", source="derived:nonsense",
            ),
        ],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SurrogateKeyWarning)
        cfg = PlotsimConfig(
            domain=Domain(
                name="t", description="t",
                entity_type="entity", entity_label="Entities",
            ),
            time_window=TimeWindow(
                start="2024-01", end="2024-03", granularity="monthly",
            ),
            seed=0,
            metrics=metrics,
            archetypes=[arch],
            entities=[Entity(name="e1", archetype="flat", size=1)],
            tables=[dim_date, dim_entity, fct],
            output=OutputConfig(format="csv", directory="out/f14"),
        )
    with pytest.raises(ValueError) as exc_info:
        generate_tables(cfg, np.random.default_rng(0))
    assert "nonsense" in str(exc_info.value)
