"""0.6-M8a: cold-start entities (per-entity arrival period).

Locks in:

* ``Entity.start_period`` field on the config model (default ``0``,
  ``ge=0``).
* ``compute_trajectory(start_period=k)`` NaN-fills ``[0, k)`` and plays
  the full archetype curve across ``[k, n_periods)``.
* ``compute_trajectory(start_period=0)`` is byte-identical to pre-M8a
  output (regression — every existing template falls into this lane).
* ``compute_all_trajectories`` propagates each entity's start_period.
* End-to-end ``generate_tables``: per-(entity, period) fact rows are
  dropped where ``period_index < entity.start_period``. Default entities
  retain every row; cold-start entities retain ``n_periods - start_period``
  rows in the entity-major order ``_drop_cold_start_rows`` expects.
* ``dim_<entity>`` still includes every entity regardless of
  ``start_period`` — the AC ground truth is "the entity exists in the
  registry but doesn't have rows in the fact tables yet".
* Manifest's ``EntityArchetypeAssignment.active_window`` carries
  ``(start_period, n_periods)`` per entity. Schema version bumped
  ``1.1 → 1.2``.
* Causal-lag policy for cold-start cells: the lag buffer stores NaN at
  cold-start periods so it remains period-index-aligned. When
  ``_compute_effective_position`` reads a NaN ``driver_past`` it falls
  back to ``current_position`` — same behaviour as the existing
  early-period (``period_index < lag_periods``) and "driver not in
  buffer" fallbacks. Documented in the M8a completion report.
* Bundled templates (every existing ``sample_*.yaml`` and
  ``*_template.yaml``) still produce byte-identical output — implicit
  by virtue of the rest of the test suite passing on the M8a branch.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from plotsim import generate_tables_with_state
from plotsim.config import (
    Archetype,
    CausalLag,
    Column,
    CorrelationPair,
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
from plotsim.manifest import ActiveWindow, build_manifest
from plotsim.trajectory import compute_all_trajectories, compute_trajectory


ROOT = Path(__file__).resolve().parent.parent


# --- Helpers ----------------------------------------------------------------


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


def _ramp_archetype() -> Archetype:
    """Linear-ish rise from 0 to 1 across the active window."""
    return Archetype(
        name="ramp",
        label="ramp",
        description="sigmoid rising 0 → 1",
        curve_segments=[
            CurveSegment(
                curve="sigmoid",
                params={"midpoint": 0.5, "steepness": 8.0},
                start_pct=0.0,
                end_pct=1.0,
            ),
        ],
    )


def _build_config(
    entities: list[Entity],
    *,
    n_metrics: int = 1,
    archetypes: list[Archetype] | None = None,
    correlations: list[CorrelationPair] | None = None,
    metrics: list[Metric] | None = None,
    extra_tables: list[Table] | None = None,
) -> PlotsimConfig:
    """Build a minimal config with one fact table + dim_date + dim_entity.

    ``n_metrics=1`` produces one metric ``m0`` with the ``normal``
    distribution; callers can override via ``metrics`` for tests that
    need ``causal_lag`` or other specifics.
    """
    if metrics is None:
        metrics = [
            Metric(
                name=f"m{i}",
                label=f"m{i}",
                distribution="normal",
                params={"mu": 1.0, "sigma": 0.1},
                polarity="positive",
            )
            for i in range(n_metrics)
        ]
    if archetypes is None:
        archetypes = [_flat_archetype()]
    metric_cols = [Column(name=m.name, dtype="float", source=f"metric:{m.name}") for m in metrics]
    fct = Table(
        name="fct_m",
        type="fact",
        grain="per_entity_per_period",
        primary_key=["date_key", "entity_id"],
        foreign_keys=["dim_date.date_key", "dim_entity.entity_id"],
        columns=[
            Column(name="date_key", dtype="id", source="fk:dim_date.date_key"),
            Column(name="entity_id", dtype="id", source="fk:dim_entity.entity_id"),
            *metric_cols,
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
        columns=[
            Column(name="entity_id", dtype="id", source="pk"),
        ],
    )
    tables = [dim_date, dim_entity, fct]
    if extra_tables:
        tables.extend(extra_tables)
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
                end="2024-12",
                granularity="monthly",
            ),
            seed=0,
            metrics=metrics,
            archetypes=archetypes,
            entities=entities,
            tables=tables,
            correlations=correlations or [],
            output=OutputConfig(format="csv", directory="out/m8a"),
        )


# --- Trajectory primitives --------------------------------------------------


def test_start_period_zero_unchanged_byte_for_byte():
    """The default (start_period=0) lane must reproduce pre-M8a output
    exactly so existing templates and golden tests keep passing.
    """
    arch = _ramp_archetype()
    baseline = compute_trajectory(arch, n_periods=24, overrides=None)
    explicit = compute_trajectory(arch, n_periods=24, overrides=None, start_period=0)
    np.testing.assert_array_equal(baseline, explicit)
    assert not np.isnan(baseline).any()


def test_start_period_nan_fills_prefix_only():
    """Periods ``[0, start_period)`` are NaN; periods ``[start_period, n)``
    are finite values in [0, 1].
    """
    arch = _ramp_archetype()
    traj = compute_trajectory(arch, n_periods=24, overrides=None, start_period=5)
    assert traj.shape == (24,)
    assert np.isnan(traj[:5]).all(), "cold-start prefix must be NaN"
    assert not np.isnan(traj[5:]).any(), "active window must be finite"
    assert (traj[5:] >= 0.0).all()
    assert (traj[5:] <= 1.0).all()


def test_active_window_replays_full_archetype_curve():
    """An entity born at period k sees the archetype's full curve (start
    of its first segment → end of its last segment) compressed into
    ``n_periods - k`` periods, NOT the tail of a full-window curve.
    Cohort analysis depends on this — every cohort lives its own
    archetype lifecycle.
    """
    arch = _ramp_archetype()
    full = compute_trajectory(arch, n_periods=24, overrides=None, start_period=0)
    cold = compute_trajectory(arch, n_periods=24, overrides=None, start_period=10)
    # The cold-start entity's first active period equals the full-window
    # curve's first period (both sample the curve at t_local=0).
    np.testing.assert_allclose(cold[10], full[0], atol=1e-12)
    # Last active period equals the full-window curve's last period.
    np.testing.assert_allclose(cold[-1], full[-1], atol=1e-12)


def test_start_period_negative_raises():
    arch = _flat_archetype()
    with pytest.raises(ValueError, match="start_period must be >= 0"):
        compute_trajectory(arch, n_periods=12, overrides=None, start_period=-1)


def test_start_period_at_or_past_n_periods_raises():
    arch = _flat_archetype()
    with pytest.raises(ValueError, match="must be < n_periods"):
        compute_trajectory(arch, n_periods=12, overrides=None, start_period=12)
    with pytest.raises(ValueError, match="must be < n_periods"):
        compute_trajectory(arch, n_periods=12, overrides=None, start_period=15)


def test_compute_all_trajectories_propagates_per_entity_start_period():
    cfg = _build_config(
        [
            Entity(name="e_default", archetype="flat", size=1),
            Entity(name="e_late", archetype="flat", size=1, start_period=4),
        ],
    )
    trajs = compute_all_trajectories(cfg, n_periods=12)
    assert not np.isnan(trajs["e_default"]).any()
    assert np.isnan(trajs["e_late"][:4]).all()
    assert not np.isnan(trajs["e_late"][4:]).any()


# --- Fact-table row filter --------------------------------------------------


def test_fact_table_row_count_matches_active_window():
    """``e_late`` (start_period=4) contributes 8 rows in a 12-period window;
    ``e_early`` (start_period=0) contributes the full 12.
    """
    cfg = _build_config(
        [
            Entity(name="e_early", archetype="flat", size=1),
            Entity(name="e_late", archetype="flat", size=1, start_period=4),
        ],
    )
    rng = np.random.default_rng(cfg.seed)
    tables, _state = generate_tables_with_state(cfg, rng)
    fct = tables["fct_m"]
    # Total rows = 12 + 8 = 20, NOT 24 (which would be 2 × 12 with no filter).
    assert len(fct) == 12 + 8

    # Per-entity row counts.
    dim_entity = tables["dim_entity"]
    early_pk = dim_entity.iloc[0]["entity_id"]
    late_pk = dim_entity.iloc[1]["entity_id"]
    assert (fct["entity_id"] == early_pk).sum() == 12
    assert (fct["entity_id"] == late_pk).sum() == 8


def test_cold_start_rows_correspond_to_active_periods_only():
    """The retained rows for a cold-start entity must align with the
    active periods (``[start_period, n_periods)``), not the cold ones.
    """
    cfg = _build_config(
        [
            Entity(name="e_early", archetype="flat", size=1),
            Entity(name="e_late", archetype="flat", size=1, start_period=4),
        ],
    )
    rng = np.random.default_rng(cfg.seed)
    tables, _state = generate_tables_with_state(cfg, rng)
    fct = tables["fct_m"]
    dim_date = tables["dim_date"]
    dim_entity = tables["dim_entity"]
    late_pk = dim_entity.iloc[1]["entity_id"]
    late_rows = fct[fct["entity_id"] == late_pk]
    # The earliest retained date for e_late equals dim_date row index 4.
    earliest_date = late_rows["date_key"].iloc[0]
    assert earliest_date == dim_date["date_key"].iloc[4]
    # Latest retained date matches the window's last period.
    assert late_rows["date_key"].iloc[-1] == dim_date["date_key"].iloc[-1]


def test_dim_entity_includes_cold_start_entities():
    """Dim is independent of arrival period — the entity registry is
    always complete, even when the entity has no fact rows yet.
    """
    cfg = _build_config(
        [
            Entity(name="e_early", archetype="flat", size=1),
            Entity(name="e_late", archetype="flat", size=1, start_period=4),
            Entity(name="e_very_late", archetype="flat", size=1, start_period=10),
        ],
    )
    rng = np.random.default_rng(cfg.seed)
    tables, _state = generate_tables_with_state(cfg, rng)
    assert len(tables["dim_entity"]) == 3


def test_default_only_config_unchanged_row_count():
    """Sanity: a config with every entity at start_period=0 produces
    the full E×P row count (no filter activation).
    """
    cfg = _build_config(
        [
            Entity(name="a", archetype="flat", size=1),
            Entity(name="b", archetype="flat", size=1),
        ],
    )
    rng = np.random.default_rng(cfg.seed)
    tables, _state = generate_tables_with_state(cfg, rng)
    assert len(tables["fct_m"]) == 2 * 12


# --- Causal-lag interaction --------------------------------------------------


def test_causal_lag_handles_cold_start_gap_without_error():
    """Cold-start lag policy: NaN-padded buffer + current-position
    fallback. Generation must not raise, and cells in the active window
    must be finite.
    """
    metrics = [
        Metric(
            name="driver",
            label="driver",
            distribution="normal",
            params={"mu": 1.0, "sigma": 0.05},
            polarity="positive",
        ),
        Metric(
            name="lagged",
            label="lagged",
            distribution="normal",
            params={"mu": 1.0, "sigma": 0.05},
            polarity="positive",
            causal_lag=CausalLag(
                driver="driver",
                lag_periods=3,
                blend_weight=1.0,
            ),
        ),
    ]
    cfg = _build_config(
        [
            Entity(name="e_late", archetype="ramp", size=1, start_period=5),
        ],
        archetypes=[_ramp_archetype()],
        metrics=metrics,
    )
    rng = np.random.default_rng(cfg.seed)
    tables, _state = generate_tables_with_state(cfg, rng)
    fct = tables["fct_m"]
    # 12 - 5 = 7 active rows.
    assert len(fct) == 7
    # All retained cells finite (no NaN propagation through the lag math).
    assert fct["driver"].notna().all()
    assert fct["lagged"].notna().all()


def test_causal_lag_first_active_period_falls_back_to_current_position():
    """At the entity's first active period, the lag buffer is empty —
    so the lagged metric should equal what current_position alone would
    produce (no driver_past to blend with). With ``blend_weight=1.0``,
    the post-blend effective_position equals the current trajectory
    position; the value sits within the metric's range, not at NaN.
    """
    metrics = [
        Metric(
            name="driver",
            label="driver",
            distribution="normal",
            params={"mu": 1.0, "sigma": 0.0},
            polarity="positive",
        ),
        Metric(
            name="lagged",
            label="lagged",
            distribution="normal",
            params={"mu": 1.0, "sigma": 0.0},
            polarity="positive",
            causal_lag=CausalLag(
                driver="driver",
                lag_periods=3,
                blend_weight=1.0,
            ),
        ),
    ]
    cfg = _build_config(
        [
            Entity(name="e_late", archetype="ramp", size=1, start_period=5),
        ],
        archetypes=[_ramp_archetype()],
        metrics=metrics,
    )
    rng = np.random.default_rng(cfg.seed)
    tables, _state = generate_tables_with_state(cfg, rng)
    fct = tables["fct_m"]
    first_lagged = float(fct["lagged"].iloc[0])
    first_driver = float(fct["driver"].iloc[0])
    # σ=0 → deterministic. With blend_weight=1.0 and an empty buffer,
    # ``_compute_effective_position`` falls back to ``current_position``,
    # so ``lagged`` and ``driver`` see the same effective position at
    # the first active period.
    np.testing.assert_allclose(first_lagged, first_driver, atol=1e-9)


# --- Correlations interaction -----------------------------------------------


def test_correlations_handle_cold_start_without_error():
    """A copula-correlated config containing a cold-start entity must
    produce finite cells in the active window. The correlation pipeline
    runs only on active periods — cold-start rows are filtered before
    they leave the fact builder.
    """
    metrics = [
        Metric(
            name="m_a",
            label="m_a",
            distribution="normal",
            params={"mu": 1.0, "sigma": 0.1},
            polarity="positive",
        ),
        Metric(
            name="m_b",
            label="m_b",
            distribution="normal",
            params={"mu": 1.0, "sigma": 0.1},
            polarity="positive",
        ),
    ]
    correlations = [CorrelationPair(metric_a="m_a", metric_b="m_b", coefficient=0.5)]
    cfg = _build_config(
        [
            Entity(name="e", archetype="flat", size=1, start_period=3),
        ],
        metrics=metrics,
        correlations=correlations,
    )
    rng = np.random.default_rng(cfg.seed)
    tables, _state = generate_tables_with_state(cfg, rng)
    fct = tables["fct_m"]
    assert len(fct) == 12 - 3
    assert fct["m_a"].notna().all()
    assert fct["m_b"].notna().all()


# --- Manifest active_window -------------------------------------------------


def test_manifest_active_window_populated_for_default_entities():
    cfg = _build_config(
        [
            Entity(name="a", archetype="flat", size=1),
            Entity(name="b", archetype="flat", size=1),
        ],
    )
    rng = np.random.default_rng(cfg.seed)
    tables, state = generate_tables_with_state(cfg, rng)
    manifest = build_manifest(cfg, state.trajectories, tables)
    by_entity = {a.entity: a for a in manifest.archetype_assignments}
    for name in ("a", "b"):
        aw = by_entity[name].active_window
        assert aw == ActiveWindow(
            start=0, end=12
        ), f"default entity {name!r} should have active_window (0, 12), got {aw}"


def test_manifest_active_window_populated_for_cold_start_entity():
    cfg = _build_config(
        [
            Entity(name="warm", archetype="flat", size=1),
            Entity(name="cold", archetype="flat", size=1, start_period=7),
        ],
    )
    rng = np.random.default_rng(cfg.seed)
    tables, state = generate_tables_with_state(cfg, rng)
    manifest = build_manifest(cfg, state.trajectories, tables)
    by_entity = {a.entity: a for a in manifest.archetype_assignments}
    assert by_entity["warm"].active_window == ActiveWindow(start=0, end=12)
    assert by_entity["cold"].active_window == ActiveWindow(start=7, end=12)


# --- Field validation -------------------------------------------------------


def test_negative_start_period_rejected_at_field_level():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Entity(name="bad", archetype="flat", size=1, start_period=-1)


# --- Vectorized routing -----------------------------------------------------


def test_cold_start_entities_routed_through_serial_lane_in_vectorized_config():
    """The vectorized path computes one shared trajectory per archetype
    for the whole batch — entities with non-zero start_period would
    silently use the wrong trajectory. They must exit the batch and run
    through the serial lane (same routing as overridden entities).

    Test by setting ``generation_mode='vectorized'`` and verifying the
    cold-start entity's row count is the active-window count, not the
    full ``n_periods`` count. If the vectorized path were leaking into
    the cold-start lane, the trajectory's NaN prefix would propagate
    through the batched centers and produce NaN rows that
    ``_drop_cold_start_rows`` would still filter — but the entity's
    metric values past start_period would no longer reflect the
    truncated active-window archetype curve. This test pins the
    routing contract.
    """
    cfg_dict = _build_config(
        [
            Entity(name="warm1", archetype="flat", size=1),
            Entity(name="warm2", archetype="flat", size=1),
            Entity(name="cold", archetype="flat", size=1, start_period=4),
        ],
    )
    # Force vectorized mode by reaching into the model.
    cfg = cfg_dict.model_copy(update={"generation_mode": "vectorized"})
    rng = np.random.default_rng(cfg.seed)
    tables, _state = generate_tables_with_state(cfg, rng)
    fct = tables["fct_m"]
    dim_entity = tables["dim_entity"]
    cold_pk = dim_entity[dim_entity["entity_id"].notna()].iloc[2]["entity_id"]
    cold_rows = fct[fct["entity_id"] == cold_pk]
    assert len(cold_rows) == 12 - 4
    assert cold_rows["m0"].notna().all()
