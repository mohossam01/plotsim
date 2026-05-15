"""0.6-M11 — Time-varying correlations (phase-keyed Cholesky).

Locks in:

* ``CorrelationPhase`` engine model + ``PlotsimConfig.correlation_phases``
  field with overlap rejection, within-window check, and baseline-required
  validator (``correlation_phases`` non-empty → ``correlations`` non-empty).
* ``PlotsimConfig.resolve_period_to_phase`` pre-computes the period →
  phase index resolver the engine threads through the Cholesky lookup.
* End-to-end engine: single-phase configs (no ``correlation_phases``)
  produce output byte-identical to pre-M11 — implicit by the rest of the
  test suite passing on the M11 branch, plus an explicit regression
  check here against an empty-phases vs absent-phases construction.
* End-to-end engine: phase-keyed correlations measurably differ across
  configured boundaries. With ``compensate_correlations=True`` and a flat
  trajectory, the realized table-wide Pearson per phase window matches
  the configured target to within sampling tolerance, and a sign flip
  between phases is reproducible at the cell level.
* Per-phase PSD projection: each phase's matrix runs through Higham
  independently; a non-PD phase produces ``CorrelationAdjustment``
  records on the manifest with ``phase_index`` set.
* Per-phase manifest entries: ``correlation_phases`` summary list,
  per-phase ``CorrelationEntry`` rows, per-phase ``CorrelationAdjustment``
  rows. ``MANIFEST_SCHEMA_VERSION`` bumped ``1.3 → 1.4``.
* Cold-start interaction: an entity born inside a phase window sees
  that phase's factor from its first active period.
* Determinism: same ``(config, seed)`` → byte-identical fact tables.
* Builder translation: ``ConnectionPhase`` → ``CorrelationPhase`` round-trip
  including relationship-word vocabulary; ``connection_phases=[]``
  default produces an empty engine ``correlation_phases``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from plotsim.builder import create as builder_create
from plotsim.config import (
    Archetype,
    Column,
    CorrelationPair,
    CorrelationPhase,
    CurveSegment,
    Domain,
    Entity,
    Metric,
    NoiseConfig,
    OutputConfig,
    PlotsimConfig,
    Table,
    TimeWindow,
)
from plotsim.manifest import (
    MANIFEST_SCHEMA_VERSION,
    CorrelationPhaseInfo,
    build_manifest,
)
from plotsim.tables import generate_tables, generate_tables_with_state
from plotsim.validation import (
    project_phase_correlation_or_issue,
    validate_correlation_psd,
)


# --- Helpers ----------------------------------------------------------------


def _month_str(year: int, n_months: int) -> str:
    """Compute the end-of-window month string for an ``n_months`` window starting Jan ``year``."""
    end_year = year + (n_months - 1) // 12
    end_month = ((n_months - 1) % 12) + 1
    return f"{end_year:04d}-{end_month:02d}"


def _two_metric_config(
    *,
    baseline_pairs: list[CorrelationPair],
    phases: list[CorrelationPhase] | None = None,
    n_entities: int = 30,
    n_months: int = 24,
    compensate: bool = True,
    seed: int = 7,
) -> PlotsimConfig:
    """Flat-trajectory 2-metric config for correlation-regime experiments."""
    kwargs: dict = dict(
        domain=Domain(name="r", description="regime", entity_type="cohort", entity_label="C"),
        time_window=TimeWindow(
            start="2024-01",
            end=_month_str(2024, n_months),
            granularity="monthly",
        ),
        seed=seed,
        metrics=[
            Metric(
                name="x",
                label="X",
                distribution="normal",
                params={"mu": 0.0, "sigma": 1.0},
                polarity="positive",
            ),
            Metric(
                name="y",
                label="Y",
                distribution="normal",
                params={"mu": 0.0, "sigma": 1.0},
                polarity="positive",
            ),
        ],
        archetypes=[
            Archetype(
                name="flat",
                label="Flat",
                description="constant 0.5 plateau",
                curve_segments=[
                    CurveSegment(
                        curve="plateau",
                        params={"level": 0.5},
                        start_pct=0.0,
                        end_pct=1.0,
                    ),
                ],
            ),
        ],
        entities=[Entity(name=f"e{i:03d}", archetype="flat", size=1) for i in range(n_entities)],
        tables=[
            Table(
                name="dim_date",
                type="dim",
                grain="per_period",
                columns=[Column(name="date_key", dtype="id", source="pk")],
                primary_key="date_key",
            ),
            Table(
                name="dim_entity",
                type="dim",
                grain="per_entity",
                columns=[Column(name="entity_id", dtype="id", source="pk")],
                primary_key="entity_id",
            ),
            Table(
                name="fct_metrics",
                type="fact",
                grain="per_entity_per_period",
                primary_key=["date_key", "entity_id"],
                foreign_keys=["dim_date.date_key", "dim_entity.entity_id"],
                columns=[
                    Column(name="date_key", dtype="id", source="fk:dim_date.date_key"),
                    Column(name="entity_id", dtype="id", source="fk:dim_entity.entity_id"),
                    Column(name="x", dtype="float", source="metric:x"),
                    Column(name="y", dtype="float", source="metric:y"),
                ],
            ),
        ],
        correlations=baseline_pairs,
        correlation_phases=phases or [],
        noise=NoiseConfig(),
        output=OutputConfig(format="csv", directory="out"),
        compensate_correlations=compensate,
    )
    return PlotsimConfig(**kwargs)


def _pearson_in_window(
    df,
    metric_a: str,
    metric_b: str,
    *,
    period_indices: list[int],
    period_label_col: str = "date_key",
) -> float:
    """Pearson r between two columns restricted to a set of period indices.

    Rows are entity-major; period index of a row is the rank of its
    ``date_key`` among sorted unique date_keys. Returns NaN if either
    selected column is constant.
    """
    unique_dates = sorted(df[period_label_col].unique())
    date_to_period = {d: i for i, d in enumerate(unique_dates)}
    period_of_row = np.array([date_to_period[d] for d in df[period_label_col].tolist()])
    mask = np.isin(period_of_row, period_indices)
    a = df[metric_a].to_numpy()[mask].astype(float)
    b = df[metric_b].to_numpy()[mask].astype(float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _three_metric_config_with_phase(
    *,
    baseline_pairs: list[CorrelationPair],
    phase_pairs: list[CorrelationPair],
) -> PlotsimConfig:
    """3-metric config with one phase covering the first half of the window."""
    return PlotsimConfig(
        domain=Domain(name="r", description="-", entity_type="c", entity_label="C"),
        time_window=TimeWindow(start="2024-01", end="2024-12", granularity="monthly"),
        seed=3,
        metrics=[
            Metric(
                name="a",
                label="A",
                distribution="normal",
                params={"mu": 0.0, "sigma": 1.0},
                polarity="positive",
            ),
            Metric(
                name="b",
                label="B",
                distribution="normal",
                params={"mu": 0.0, "sigma": 1.0},
                polarity="positive",
            ),
            Metric(
                name="c",
                label="C",
                distribution="normal",
                params={"mu": 0.0, "sigma": 1.0},
                polarity="positive",
            ),
        ],
        archetypes=[
            Archetype(
                name="flat",
                label="Flat",
                description="-",
                curve_segments=[
                    CurveSegment(
                        curve="plateau",
                        params={"level": 0.5},
                        start_pct=0.0,
                        end_pct=1.0,
                    ),
                ],
            ),
        ],
        entities=[Entity(name=f"e{i:02d}", archetype="flat", size=1) for i in range(10)],
        tables=[
            Table(
                name="dim_date",
                type="dim",
                grain="per_period",
                columns=[Column(name="date_key", dtype="id", source="pk")],
                primary_key="date_key",
            ),
            Table(
                name="dim_entity",
                type="dim",
                grain="per_entity",
                columns=[Column(name="entity_id", dtype="id", source="pk")],
                primary_key="entity_id",
            ),
            Table(
                name="fct_metrics",
                type="fact",
                grain="per_entity_per_period",
                primary_key=["date_key", "entity_id"],
                foreign_keys=["dim_date.date_key", "dim_entity.entity_id"],
                columns=[
                    Column(name="date_key", dtype="id", source="fk:dim_date.date_key"),
                    Column(name="entity_id", dtype="id", source="fk:dim_entity.entity_id"),
                    Column(name="a", dtype="float", source="metric:a"),
                    Column(name="b", dtype="float", source="metric:b"),
                    Column(name="c", dtype="float", source="metric:c"),
                ],
            ),
        ],
        correlations=baseline_pairs,
        correlation_phases=[
            CorrelationPhase(
                start_period=0,
                end_period=5,
                correlations=phase_pairs,
            ),
        ],
        noise=NoiseConfig(),
        output=OutputConfig(format="csv", directory="out"),
    )


# --- CorrelationPhase model -------------------------------------------------


class TestCorrelationPhaseModel:
    """Field-level validation on the phase model itself."""

    def test_end_after_start_rejected(self):
        with pytest.raises(ValueError, match="end_period.*<.*start_period"):
            CorrelationPhase(start_period=5, end_period=3, correlations=[])

    def test_equal_start_and_end_allowed(self):
        phase = CorrelationPhase(start_period=4, end_period=4, correlations=[])
        assert phase.start_period == phase.end_period == 4

    def test_negative_start_period_rejected(self):
        with pytest.raises(ValueError, match="greater than or equal to 0"):
            CorrelationPhase(start_period=-1, end_period=5, correlations=[])


# --- PlotsimConfig validators -----------------------------------------------


class TestConfigValidators:
    """Engine-level cross-field validators for phases."""

    def test_phases_without_baseline_rejected(self):
        with pytest.raises(ValueError, match="baseline"):
            _two_metric_config(
                baseline_pairs=[],
                phases=[
                    CorrelationPhase(
                        start_period=0,
                        end_period=11,
                        correlations=[
                            CorrelationPair(metric_a="x", metric_b="y", coefficient=0.5),
                        ],
                    ),
                ],
            )

    def test_phase_outside_time_window_rejected(self):
        with pytest.raises(ValueError, match="end_period=.*but time_window has only"):
            _two_metric_config(
                baseline_pairs=[CorrelationPair(metric_a="x", metric_b="y", coefficient=0.5)],
                phases=[
                    CorrelationPhase(
                        start_period=0,
                        end_period=99,
                        correlations=[
                            CorrelationPair(metric_a="x", metric_b="y", coefficient=0.5),
                        ],
                    ),
                ],
            )

    def test_overlapping_phases_rejected(self):
        with pytest.raises(ValueError, match="overlap"):
            _two_metric_config(
                baseline_pairs=[CorrelationPair(metric_a="x", metric_b="y", coefficient=0.5)],
                phases=[
                    CorrelationPhase(
                        start_period=0,
                        end_period=11,
                        correlations=[
                            CorrelationPair(metric_a="x", metric_b="y", coefficient=0.7),
                        ],
                    ),
                    CorrelationPhase(
                        start_period=10,
                        end_period=20,
                        correlations=[
                            CorrelationPair(metric_a="x", metric_b="y", coefficient=-0.7),
                        ],
                    ),
                ],
            )

    def test_adjacent_phases_allowed(self):
        cfg = _two_metric_config(
            baseline_pairs=[CorrelationPair(metric_a="x", metric_b="y", coefficient=0.5)],
            phases=[
                CorrelationPhase(
                    start_period=0,
                    end_period=11,
                    correlations=[
                        CorrelationPair(metric_a="x", metric_b="y", coefficient=0.7),
                    ],
                ),
                CorrelationPhase(
                    start_period=12,
                    end_period=23,
                    correlations=[
                        CorrelationPair(metric_a="x", metric_b="y", coefficient=-0.7),
                    ],
                ),
            ],
        )
        assert len(cfg.correlation_phases) == 2

    def test_out_of_order_phases_allowed(self):
        cfg = _two_metric_config(
            baseline_pairs=[CorrelationPair(metric_a="x", metric_b="y", coefficient=0.5)],
            phases=[
                CorrelationPhase(
                    start_period=12,
                    end_period=23,
                    correlations=[
                        CorrelationPair(metric_a="x", metric_b="y", coefficient=-0.7),
                    ],
                ),
                CorrelationPhase(
                    start_period=0,
                    end_period=11,
                    correlations=[
                        CorrelationPair(metric_a="x", metric_b="y", coefficient=0.7),
                    ],
                ),
            ],
        )
        assert len(cfg.correlation_phases) == 2


# --- resolve_period_to_phase ------------------------------------------------


class TestPeriodToPhaseResolver:
    """The pre-computed per-period phase index table."""

    def test_no_phases_returns_all_none(self):
        cfg = _two_metric_config(
            baseline_pairs=[CorrelationPair(metric_a="x", metric_b="y", coefficient=0.5)],
            phases=None,
            n_months=12,
        )
        resolved = cfg.resolve_period_to_phase()
        assert len(resolved) == 12
        assert all(p is None for p in resolved)

    def test_full_window_phase_covers_every_period(self):
        cfg = _two_metric_config(
            baseline_pairs=[CorrelationPair(metric_a="x", metric_b="y", coefficient=0.5)],
            phases=[
                CorrelationPhase(
                    start_period=0,
                    end_period=11,
                    correlations=[
                        CorrelationPair(metric_a="x", metric_b="y", coefficient=0.7),
                    ],
                ),
            ],
            n_months=12,
        )
        resolved = cfg.resolve_period_to_phase()
        assert resolved == [0] * 12

    def test_partial_coverage_falls_back_to_baseline(self):
        cfg = _two_metric_config(
            baseline_pairs=[CorrelationPair(metric_a="x", metric_b="y", coefficient=0.5)],
            phases=[
                CorrelationPhase(
                    start_period=4,
                    end_period=7,
                    correlations=[
                        CorrelationPair(metric_a="x", metric_b="y", coefficient=0.7),
                    ],
                ),
            ],
            n_months=12,
        )
        resolved = cfg.resolve_period_to_phase()
        assert resolved == [None, None, None, None, 0, 0, 0, 0, None, None, None, None]

    def test_two_phases_resolve_to_their_indices(self):
        cfg = _two_metric_config(
            baseline_pairs=[CorrelationPair(metric_a="x", metric_b="y", coefficient=0.5)],
            phases=[
                CorrelationPhase(
                    start_period=0,
                    end_period=3,
                    correlations=[
                        CorrelationPair(metric_a="x", metric_b="y", coefficient=0.7),
                    ],
                ),
                CorrelationPhase(
                    start_period=8,
                    end_period=11,
                    correlations=[
                        CorrelationPair(metric_a="x", metric_b="y", coefficient=-0.7),
                    ],
                ),
            ],
            n_months=12,
        )
        resolved = cfg.resolve_period_to_phase()
        assert resolved == [0, 0, 0, 0, None, None, None, None, 1, 1, 1, 1]


# --- Single-phase regression -----------------------------------------------


class TestSinglePhaseRegression:
    """Configs without ``correlation_phases`` produce byte-identical output."""

    def test_empty_phases_list_matches_absent_phases(self):
        baseline = [CorrelationPair(metric_a="x", metric_b="y", coefficient=0.5)]
        cfg_no_phases = _two_metric_config(baseline_pairs=baseline, phases=None)
        cfg_empty_phases = _two_metric_config(baseline_pairs=baseline, phases=[])
        tables_no = generate_tables(cfg_no_phases)
        tables_empty = generate_tables(cfg_empty_phases)
        for name in tables_no:
            np.testing.assert_array_equal(
                tables_no[name].to_numpy(),
                tables_empty[name].to_numpy(),
                err_msg=f"single-phase regression on table {name!r}",
            )


# --- Phase boundary measurable change --------------------------------------


class TestPhaseBoundaryCorrelationChange:
    """A regime change across phase windows produces a measurable Pearson flip."""

    def test_positive_then_negative_phases_flip_realized_correlation(self):
        baseline = [CorrelationPair(metric_a="x", metric_b="y", coefficient=0.5)]
        phases = [
            CorrelationPhase(
                start_period=0,
                end_period=11,
                correlations=[
                    CorrelationPair(metric_a="x", metric_b="y", coefficient=0.85),
                ],
            ),
            CorrelationPhase(
                start_period=12,
                end_period=23,
                correlations=[
                    CorrelationPair(metric_a="x", metric_b="y", coefficient=-0.85),
                ],
            ),
        ]
        cfg = _two_metric_config(
            baseline_pairs=baseline,
            phases=phases,
            n_entities=80,
            n_months=24,
            compensate=True,
        )
        tables = generate_tables(cfg)
        fct = tables["fct_metrics"]
        r_phase1 = _pearson_in_window(fct, "x", "y", period_indices=list(range(0, 12)))
        r_phase2 = _pearson_in_window(fct, "x", "y", period_indices=list(range(12, 24)))
        assert r_phase1 > 0.5, f"phase 1 expected r ≈ +0.85, got {r_phase1:.3f}"
        assert r_phase2 < -0.5, f"phase 2 expected r ≈ -0.85, got {r_phase2:.3f}"
        assert r_phase1 * r_phase2 < 0

    def test_baseline_applies_in_uncovered_periods(self):
        baseline = [CorrelationPair(metric_a="x", metric_b="y", coefficient=0.8)]
        phases = [
            CorrelationPhase(
                start_period=12,
                end_period=23,
                correlations=[
                    CorrelationPair(metric_a="x", metric_b="y", coefficient=-0.8),
                ],
            ),
        ]
        cfg = _two_metric_config(
            baseline_pairs=baseline,
            phases=phases,
            n_entities=80,
            n_months=24,
            compensate=True,
        )
        tables = generate_tables(cfg)
        fct = tables["fct_metrics"]
        r_baseline = _pearson_in_window(fct, "x", "y", period_indices=list(range(0, 12)))
        r_phase = _pearson_in_window(fct, "x", "y", period_indices=list(range(12, 24)))
        assert r_baseline > 0.5
        assert r_phase < -0.5

    def test_empty_phase_correlations_falls_through_to_baseline(self):
        baseline = [CorrelationPair(metric_a="x", metric_b="y", coefficient=0.7)]
        cfg_with_empty_phase = _two_metric_config(
            baseline_pairs=baseline,
            phases=[
                CorrelationPhase(
                    start_period=0,
                    end_period=11,
                    correlations=[],
                ),
            ],
            n_months=24,
        )
        cfg_no_phase = _two_metric_config(baseline_pairs=baseline, n_months=24)
        tables_phase = generate_tables(cfg_with_empty_phase)
        tables_no = generate_tables(cfg_no_phase)
        for name in tables_phase:
            np.testing.assert_array_equal(
                tables_phase[name].to_numpy(),
                tables_no[name].to_numpy(),
                err_msg=f"empty-phase no-op on table {name!r}",
            )


# --- Per-phase PSD projection ----------------------------------------------


class TestPerPhasePSD:
    """Non-PD phase matrix triggers Higham projection independently of baseline."""

    def test_non_pd_phase_emits_warning_with_phase_label(self):
        non_pd_pairs = [
            CorrelationPair(metric_a="a", metric_b="b", coefficient=0.9),
            CorrelationPair(metric_a="b", metric_b="c", coefficient=0.9),
            CorrelationPair(metric_a="a", metric_b="c", coefficient=-0.9),
        ]
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            cfg = _three_metric_config_with_phase(
                baseline_pairs=[
                    CorrelationPair(metric_a="a", metric_b="b", coefficient=0.3),
                ],
                phase_pairs=non_pd_pairs,
            )
        phase_warns = [w for w in captured if "correlation_phases[0]" in str(w.message)]
        assert (
            phase_warns
        ), f"expected phase-tagged warning, got {[str(w.message) for w in captured]}"
        assert cfg._phase_correlation_adjustments is not None
        assert 0 in cfg._phase_correlation_adjustments

    def test_pd_baseline_and_non_pd_phase_both_recorded(self):
        non_pd_pairs = [
            CorrelationPair(metric_a="a", metric_b="b", coefficient=0.9),
            CorrelationPair(metric_a="b", metric_b="c", coefficient=0.9),
            CorrelationPair(metric_a="a", metric_b="c", coefficient=-0.9),
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cfg = _three_metric_config_with_phase(
                baseline_pairs=[
                    CorrelationPair(metric_a="a", metric_b="b", coefficient=0.3),
                ],
                phase_pairs=non_pd_pairs,
            )
        assert cfg._correlation_adjustments is None
        assert cfg._phase_correlation_adjustments is not None
        records = cfg._phase_correlation_adjustments[0]
        assert len(records) == 3

    def test_validate_correlation_psd_reports_clean_after_projection(self):
        non_pd_pairs = [
            CorrelationPair(metric_a="a", metric_b="b", coefficient=0.9),
            CorrelationPair(metric_a="b", metric_b="c", coefficient=0.9),
            CorrelationPair(metric_a="a", metric_b="c", coefficient=-0.9),
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cfg = _three_metric_config_with_phase(
                baseline_pairs=[
                    CorrelationPair(metric_a="a", metric_b="b", coefficient=0.3),
                ],
                phase_pairs=non_pd_pairs,
            )
        issues = validate_correlation_psd(cfg)
        assert issues == []

    def test_project_phase_correlation_or_issue_on_pd_phase(self):
        cfg = _two_metric_config(
            baseline_pairs=[CorrelationPair(metric_a="x", metric_b="y", coefficient=0.5)],
            phases=[
                CorrelationPhase(
                    start_period=0,
                    end_period=11,
                    correlations=[
                        CorrelationPair(metric_a="x", metric_b="y", coefficient=0.3),
                    ],
                ),
            ],
        )
        issues, adjustments, _projected = project_phase_correlation_or_issue(cfg, 0)
        assert issues == []
        assert adjustments is None

    def test_project_phase_correlation_or_issue_invalid_index(self):
        cfg = _two_metric_config(
            baseline_pairs=[CorrelationPair(metric_a="x", metric_b="y", coefficient=0.5)],
        )
        with pytest.raises(IndexError):
            project_phase_correlation_or_issue(cfg, 5)


# --- Manifest integration ---------------------------------------------------


class TestManifestIntegration:
    """Manifest carries per-phase entries and the new top-level summary."""

    def test_schema_version_is_1_9(self):
        # 0.6-M13 bumped 1.4 → 1.5 for ``source_entity_mappings``; 0.6-M18
        # bumped 1.5 → 1.6 for ``parent_child_relations``; 0.6-M22 bumped
        # 1.6 → 1.7 for the optional ``noise_config`` field; 0.6-M23
        # bumped 1.7 → 1.8 for ``noise_family`` / ``degrees_of_freedom``
        # on ``NoiseConfigInfo`` and broadened its emission criterion;
        # 0.6-M24 bumped 1.8 → 1.9 for the additive ``target_metric``
        # field on ``TreatmentAssignment`` / ``TreatmentCohort``.
        assert MANIFEST_SCHEMA_VERSION == "1.9"

    def test_no_phases_yields_empty_correlation_phases_list(self):
        cfg = _two_metric_config(
            baseline_pairs=[CorrelationPair(metric_a="x", metric_b="y", coefficient=0.5)],
        )
        tables, state = generate_tables_with_state(cfg)
        manifest = build_manifest(config=cfg, trajectories=state.trajectories, tables=tables)
        assert manifest.correlation_phases == []

    def test_correlation_phases_info_populated(self):
        phases = [
            CorrelationPhase(
                start_period=0,
                end_period=11,
                correlations=[
                    CorrelationPair(metric_a="x", metric_b="y", coefficient=0.7),
                ],
            ),
            CorrelationPhase(
                start_period=12,
                end_period=23,
                correlations=[
                    CorrelationPair(metric_a="x", metric_b="y", coefficient=-0.7),
                ],
            ),
        ]
        cfg = _two_metric_config(
            baseline_pairs=[CorrelationPair(metric_a="x", metric_b="y", coefficient=0.5)],
            phases=phases,
            n_months=24,
        )
        tables, state = generate_tables_with_state(cfg)
        manifest = build_manifest(config=cfg, trajectories=state.trajectories, tables=tables)
        assert manifest.correlation_phases == [
            CorrelationPhaseInfo(phase_index=0, start_period=0, end_period=11, n_pairs=1),
            CorrelationPhaseInfo(phase_index=1, start_period=12, end_period=23, n_pairs=1),
        ]

    def test_per_phase_correlation_entries(self):
        phases = [
            CorrelationPhase(
                start_period=0,
                end_period=11,
                correlations=[
                    CorrelationPair(metric_a="x", metric_b="y", coefficient=0.7),
                ],
            ),
            CorrelationPhase(
                start_period=12,
                end_period=23,
                correlations=[
                    CorrelationPair(metric_a="x", metric_b="y", coefficient=-0.7),
                ],
            ),
        ]
        cfg = _two_metric_config(
            baseline_pairs=[CorrelationPair(metric_a="x", metric_b="y", coefficient=0.5)],
            phases=phases,
            n_months=24,
        )
        tables, state = generate_tables_with_state(cfg)
        manifest = build_manifest(config=cfg, trajectories=state.trajectories, tables=tables)
        by_phase = [e.phase_index for e in manifest.correlations]
        assert by_phase == [None, 0, 1]
        baseline_entry = manifest.correlations[0]
        phase0_entry = manifest.correlations[1]
        phase1_entry = manifest.correlations[2]
        assert baseline_entry.requested == 0.5
        assert phase0_entry.requested == 0.7
        assert phase1_entry.requested == -0.7
        assert phase0_entry.projected > 0
        assert phase1_entry.projected < 0

    def test_per_phase_adjustment_carries_phase_index(self):
        non_pd_pairs = [
            CorrelationPair(metric_a="a", metric_b="b", coefficient=0.9),
            CorrelationPair(metric_a="b", metric_b="c", coefficient=0.9),
            CorrelationPair(metric_a="a", metric_b="c", coefficient=-0.9),
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cfg = _three_metric_config_with_phase(
                baseline_pairs=[
                    CorrelationPair(metric_a="a", metric_b="b", coefficient=0.3),
                ],
                phase_pairs=non_pd_pairs,
            )
            tables, state = generate_tables_with_state(cfg)
        manifest = build_manifest(config=cfg, trajectories=state.trajectories, tables=tables)
        assert manifest.correlation_adjustments is not None
        for adj in manifest.correlation_adjustments:
            assert adj.phase_index == 0


# --- Cold-start interaction ------------------------------------------------


class TestColdStartInteraction:
    """Entities born inside a phase window pick up the phase's factor."""

    def test_cold_start_entity_inside_phase_window_sees_phase_correlation(self):
        baseline = [CorrelationPair(metric_a="x", metric_b="y", coefficient=0.8)]
        phases = [
            CorrelationPhase(
                start_period=4,
                end_period=15,
                correlations=[
                    CorrelationPair(metric_a="x", metric_b="y", coefficient=-0.8),
                ],
            ),
        ]
        cfg = PlotsimConfig(
            domain=Domain(name="r", description="cold", entity_type="cohort", entity_label="C"),
            time_window=TimeWindow(start="2024-01", end="2025-04", granularity="monthly"),
            seed=11,
            metrics=[
                Metric(
                    name="x",
                    label="X",
                    distribution="normal",
                    params={"mu": 0.0, "sigma": 1.0},
                    polarity="positive",
                ),
                Metric(
                    name="y",
                    label="Y",
                    distribution="normal",
                    params={"mu": 0.0, "sigma": 1.0},
                    polarity="positive",
                ),
            ],
            archetypes=[
                Archetype(
                    name="flat",
                    label="Flat",
                    description="-",
                    curve_segments=[
                        CurveSegment(
                            curve="plateau",
                            params={"level": 0.5},
                            start_pct=0.0,
                            end_pct=1.0,
                        ),
                    ],
                ),
            ],
            entities=[
                Entity(name=f"e{i:03d}", archetype="flat", size=1, start_period=6)
                for i in range(60)
            ],
            tables=[
                Table(
                    name="dim_date",
                    type="dim",
                    grain="per_period",
                    columns=[Column(name="date_key", dtype="id", source="pk")],
                    primary_key="date_key",
                ),
                Table(
                    name="dim_entity",
                    type="dim",
                    grain="per_entity",
                    columns=[Column(name="entity_id", dtype="id", source="pk")],
                    primary_key="entity_id",
                ),
                Table(
                    name="fct_metrics",
                    type="fact",
                    grain="per_entity_per_period",
                    primary_key=["date_key", "entity_id"],
                    foreign_keys=["dim_date.date_key", "dim_entity.entity_id"],
                    columns=[
                        Column(name="date_key", dtype="id", source="fk:dim_date.date_key"),
                        Column(name="entity_id", dtype="id", source="fk:dim_entity.entity_id"),
                        Column(name="x", dtype="float", source="metric:x"),
                        Column(name="y", dtype="float", source="metric:y"),
                    ],
                ),
            ],
            correlations=baseline,
            correlation_phases=phases,
            noise=NoiseConfig(),
            output=OutputConfig(format="csv", directory="out"),
            compensate_correlations=True,
        )
        tables = generate_tables(cfg)
        fct = tables["fct_metrics"]
        r_phase = _pearson_in_window(fct, "x", "y", period_indices=list(range(6, 16)))
        assert r_phase < -0.3, f"cold-start rows inside phase expected r < -0.3, got {r_phase:.3f}"


# --- Determinism ------------------------------------------------------------


class TestDeterminism:
    """Same config + seed → byte-identical fact tables across runs."""

    def test_phase_keyed_output_deterministic(self):
        phases = [
            CorrelationPhase(
                start_period=0,
                end_period=11,
                correlations=[
                    CorrelationPair(metric_a="x", metric_b="y", coefficient=0.7),
                ],
            ),
            CorrelationPhase(
                start_period=12,
                end_period=23,
                correlations=[
                    CorrelationPair(metric_a="x", metric_b="y", coefficient=-0.7),
                ],
            ),
        ]

        def _run() -> dict:
            cfg = _two_metric_config(
                baseline_pairs=[CorrelationPair(metric_a="x", metric_b="y", coefficient=0.5)],
                phases=phases,
                n_months=24,
                seed=42,
            )
            return generate_tables(cfg)

        a = _run()
        b = _run()
        for name in a:
            np.testing.assert_array_equal(
                a[name].to_numpy(),
                b[name].to_numpy(),
                err_msg=f"determinism violated on table {name!r}",
            )


# --- Builder translation ----------------------------------------------------


class TestBuilderTranslation:
    """Builder ``connection_phases`` → engine ``correlation_phases``."""

    def test_empty_connection_phases_yields_empty_engine_phases(self):
        cfg = builder_create(
            about="regime",
            unit="cohort",
            window=("2024-01", "2024-06"),
            metrics=[
                {"name": "x", "label": "X", "type": "score", "polarity": "positive"},
                {"name": "y", "label": "Y", "type": "score", "polarity": "positive"},
            ],
            segments=[
                {"name": "seg", "count": 5, "archetype": "growth"},
            ],
            connections=["x driven_by y"],
            seed=1,
        )
        assert cfg.correlation_phases == []

    def test_builder_connection_phases_translate_to_engine_phases(self):
        cfg = builder_create(
            about="regime",
            unit="cohort",
            window=("2024-01", "2025-12"),
            metrics=[
                {"name": "x", "label": "X", "type": "score", "polarity": "positive"},
                {"name": "y", "label": "Y", "type": "score", "polarity": "positive"},
            ],
            segments=[
                {"name": "seg", "count": 5, "archetype": "growth"},
            ],
            connections=["x driven_by y"],
            connection_phases=[
                {
                    "start_period": 0,
                    "end_period": 11,
                    "connections": ["x driven_by y"],
                },
                {
                    "start_period": 12,
                    "end_period": 23,
                    "connections": ["x opposes y"],
                },
            ],
            seed=1,
        )
        assert len(cfg.correlation_phases) == 2
        ph0_pairs = cfg.correlation_phases[0].correlations
        ph1_pairs = cfg.correlation_phases[1].correlations
        assert len(ph0_pairs) == 1
        assert len(ph1_pairs) == 1
        assert ph0_pairs[0].coefficient > 0
        assert ph1_pairs[0].coefficient < 0
