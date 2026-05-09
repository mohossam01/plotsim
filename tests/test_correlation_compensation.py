"""M120 — Trajectory-aware correlation pre-compensation.

Covers the end-to-end flow that subtracts the trajectory's structural
covariance from the user's correlation target before the Cholesky factor
hits the per-cell copula:

  * Pure helpers (``estimate_trajectory_covariance``,
    ``compensate_correlation_matrix``) — shape, ordering, identity-noop,
    seasonal modulation, infeasibility clamping.
  * End-to-end: with compensation ON, a builder-style config with 2
    archetypes (growth + decline) and an "opposes" connection produces a
    fact table whose ``df.corr()`` shows the configured sign — with OFF,
    the trajectory covariance dominates and the realized correlation is
    the wrong sign. The mission's "make connections visible" promise.
  * Backwards compatibility: engine-direct configs default to
    ``compensate_correlations=False`` and remain byte-identical to
    pre-M120 baselines (delegated to the existing
    ``test_layer4_reference_fixtures_match`` regression).
  * Manifest: declared pairs surface as ``CorrelationCompensation``
    records; engine-direct runs leave ``correlation_compensations=None``.
  * Inspect: ``trace_metric_cell`` replays against the compensated
    Cholesky factor — the coefficient driving each cell is the
    compensated value, not the raw user target.
  * Determinism: same (config, seed) → byte-identical CSVs with
    compensation enabled.
  * Metric cap: M > ``_MAX_METRICS_FOR_COMPENSATION`` warns and falls
    through to the legacy direct-copula path.
"""

from __future__ import annotations

import warnings
from typing import Any, Optional

import numpy as np
import pandas as pd
import pytest

from plotsim.config import (
    Archetype,
    Column,
    CorrelationPair,
    CurveSegment,
    Domain,
    Entity,
    Metric,
    NoiseConfig,
    OutputConfig,
    PlotsimConfig,
    SeasonalEffect,
    Table,
    TimeWindow,
)
from plotsim.manifest import (
    CorrelationCompensation,
    build_manifest,
)
from plotsim.metrics import (
    _MAX_METRICS_FOR_COMPENSATION,
    compensate_correlation_matrix,
    estimate_trajectory_covariance,
)
from plotsim.tables import generate_tables, generate_tables_with_state


# --- Fixture builders --------------------------------------------------------


def _growth_archetype(name: str = "growth") -> Archetype:
    return Archetype(
        name=name,
        label=name,
        description="rising sigmoid",
        curve_segments=[
            CurveSegment(
                curve="sigmoid",
                params={"midpoint": 0.5, "steepness": 8.0},
                start_pct=0.0,
                end_pct=1.0,
            ),
        ],
    )


def _decline_archetype(name: str = "decline") -> Archetype:
    return Archetype(
        name=name,
        label=name,
        description="falling sigmoid",
        curve_segments=[
            CurveSegment(
                curve="sigmoid",
                params={"midpoint": 0.5, "steepness": -8.0},
                start_pct=0.0,
                end_pct=1.0,
            ),
        ],
    )


def _flat_archetype(name: str = "flat", level: float = 0.5) -> Archetype:
    return Archetype(
        name=name,
        label=name,
        description="constant plateau",
        curve_segments=[
            CurveSegment(curve="plateau", params={"level": level}, start_pct=0.0, end_pct=1.0),
        ],
    )


def _lognorm_metric(
    name: str,
    *,
    polarity: str = "positive",
    s: float = 0.55,
) -> Metric:
    """Lognorm with positive ``loc`` floor so center never hits the bypass.

    Default ``s=0.55`` is the sweet-spot empirically found for the
    end-to-end fixtures: at this noise level the copula's variance
    contribution (s²≈0.30) is comparable to the trajectory's
    center-variance (var(log(c))≈0.25 for loc=1, scale=5 on a sigmoid
    trajectory), so realized table-wide Pearson is sensitive to the
    pre-compensation step. With less noise (e.g. ``s=0.2``) the
    trajectory dominates and compensation can't shift the realized
    correlation regardless of the configured target; with more noise
    the copula dominates and OFF already lands close to the user
    target. The mission's "make-connections-visible" AC is checked at
    this s.
    """
    return Metric(
        name=name,
        label=name,
        distribution="lognorm",
        params={"s": s, "loc": 1.0, "scale": 5.0},
        polarity=polarity,
    )


def _make_config(
    *,
    metrics: list[Metric],
    archetypes: list[Archetype],
    entity_per_archetype: int = 100,
    correlations: Optional[list[CorrelationPair]] = None,
    compensate: bool = True,
    seed: int = 42,
    seasonal_effects: Optional[list[SeasonalEffect]] = None,
    noise: Optional[NoiseConfig] = None,
    granularity: str = "monthly",
    start: str = "2023-01",
    end: str = "2024-12",
) -> PlotsimConfig:
    entities: list[Entity] = []
    for arch in archetypes:
        for i in range(entity_per_archetype):
            entities.append(
                Entity(
                    name=f"{arch.name}_{i:03d}",
                    archetype=arch.name,
                    size=1,
                )
            )
    fact_cols: list[Column] = [
        Column(name="date_key", dtype="id", source="fk:dim_date.date_key"),
        Column(name="entity_id", dtype="id", source="fk:dim_entity.entity_id"),
    ]
    for m in metrics:
        fact_cols.append(Column(name=m.name, dtype="float", source=f"metric:{m.name}"))
    return PlotsimConfig(
        domain=Domain(
            name="m120 harness", description="-", entity_type="unit", entity_label="Unit"
        ),
        time_window=TimeWindow(start=start, end=end, granularity=granularity),
        seed=seed,
        metrics=metrics,
        archetypes=archetypes,
        entities=entities,
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
                columns=[
                    Column(name="entity_id", dtype="id", source="pk"),
                    Column(name="entity_name", dtype="string", source="derived:name"),
                ],
                primary_key="entity_id",
            ),
            Table(
                name="fct_metrics",
                type="fact",
                grain="per_entity_per_period",
                columns=fact_cols,
                primary_key=["date_key", "entity_id"],
            ),
        ],
        correlations=correlations or [],
        noise=noise if noise is not None else NoiseConfig(),
        seasonal_effects=seasonal_effects or [],
        compensate_correlations=compensate,
        output=OutputConfig(format="csv", directory="out"),
    )


def _table_corr(df: pd.DataFrame, a: str, b: str) -> float:
    return float(df[[a, b]].corr().iloc[0, 1])


# --- TestEstimateTrajectoryCovariance ---------------------------------------


class TestEstimateTrajectoryCovariance:
    """Pure-function checks on the trajectory covariance estimator."""

    def test_shape_matches_metric_count(self):
        cfg = _make_config(
            metrics=[_lognorm_metric(f"m{i}") for i in range(4)],
            archetypes=[_growth_archetype(), _decline_archetype()],
            entity_per_archetype=10,
        )
        cov = estimate_trajectory_covariance(cfg)
        assert cov.shape == (4, 4)

    def test_diagonal_is_unit(self):
        cfg = _make_config(
            metrics=[_lognorm_metric("a"), _lognorm_metric("b")],
            archetypes=[_growth_archetype(), _decline_archetype()],
            entity_per_archetype=5,
        )
        cov = estimate_trajectory_covariance(cfg)
        np.testing.assert_allclose(np.diag(cov), 1.0)

    def test_within_archetype_same_polarity_metrics_correlated_positive(self):
        # Both metrics positive polarity, both growth + decline mix → centers
        # within each archetype move together. Within-archetype Pearson ≈ +1
        # for either archetype; weighted mean stays near +1.
        cfg = _make_config(
            metrics=[_lognorm_metric("a"), _lognorm_metric("b")],
            archetypes=[_growth_archetype(), _decline_archetype()],
            entity_per_archetype=10,
        )
        cov = estimate_trajectory_covariance(cfg)
        assert cov[0, 1] > 0.9

    def test_opposite_polarity_within_archetype_anti_correlated(self):
        # ``a`` positive (rises with traj), ``b`` negative (falls with traj).
        # Within archetype, centers move opposite → Pearson ≈ -1.
        cfg = _make_config(
            metrics=[
                _lognorm_metric("a", polarity="positive"),
                _lognorm_metric("b", polarity="negative"),
            ],
            archetypes=[_growth_archetype()],
            entity_per_archetype=10,
        )
        cov = estimate_trajectory_covariance(cfg)
        assert cov[0, 1] < -0.9

    def test_flat_archetype_returns_zero_off_diagonal(self):
        # Plateau trajectory → constant centers → std=0 → safe corrcoef
        # produces zero off-diagonal (no info to correlate).
        cfg = _make_config(
            metrics=[_lognorm_metric("a"), _lognorm_metric("b")],
            archetypes=[_flat_archetype()],
            entity_per_archetype=5,
        )
        cov = estimate_trajectory_covariance(cfg)
        assert abs(cov[0, 1]) < 1e-12

    def test_seasonal_modulation_changes_estimate(self):
        # With distinct per-metric seasonal_sensitivity values, adding a
        # seasonal effect must shift the off-diagonal covariance entries —
        # the two metrics' centers no longer move strictly proportionally
        # across periods. Same-sensitivity metrics scale identically and
        # leave the Pearson untouched (it's invariant to per-column
        # multiplicative scaling), which is why this test uses asymmetric
        # sensitivities.
        metric_a = Metric(
            name="a",
            label="A",
            distribution="lognorm",
            params={"s": 0.2, "loc": 1.0, "scale": 5.0},
            polarity="positive",
            seasonal_sensitivity=0.0,
        )
        metric_b = Metric(
            name="b",
            label="B",
            distribution="lognorm",
            params={"s": 0.2, "loc": 1.0, "scale": 5.0},
            polarity="positive",
            seasonal_sensitivity=2.0,
        )
        archetypes = [_growth_archetype()]
        seasonal = [SeasonalEffect(months=(12, 1, 2), strength=0.8)]
        cfg_off = _make_config(
            metrics=[metric_a, metric_b],
            archetypes=archetypes,
            entity_per_archetype=5,
        )
        cfg_on = _make_config(
            metrics=[metric_a, metric_b],
            archetypes=archetypes,
            entity_per_archetype=5,
            seasonal_effects=seasonal,
        )
        cov_off = estimate_trajectory_covariance(cfg_off)
        cov_on = estimate_trajectory_covariance(cfg_on)
        assert not np.allclose(cov_off, cov_on)


# --- TestCompensateCorrelationMatrix ----------------------------------------


class TestCompensateCorrelationMatrix:
    """Pure-function checks on the compensation matrix builder."""

    def test_zero_correlations_is_noop(self):
        # Zero declared pairs ⇒ early return with the original matrix.
        # Mirrors the AC: "Pre-compensation produces identical results to
        # current behavior when all correlations are zero."
        user_mat = np.eye(3)
        traj_cov = np.array([[1.0, 0.6, 0.5], [0.6, 1.0, 0.4], [0.5, 0.4, 1.0]])
        metrics = [
            Metric(
                name=f"m{i}",
                label=f"m{i}",
                distribution="normal",
                params={"mu": 1.0, "sigma": 0.1},
                polarity="positive",
            )
            for i in range(3)
        ]
        compensated, records = compensate_correlation_matrix(
            user_mat,
            traj_cov,
            metrics,
            [],
        )
        assert records == []
        np.testing.assert_array_equal(compensated, user_mat)

    def test_subtracts_trajectory_contribution_off_diagonal(self):
        metrics = [
            Metric(
                name="a",
                label="A",
                distribution="normal",
                params={"mu": 1.0, "sigma": 0.1},
                polarity="positive",
            ),
            Metric(
                name="b",
                label="B",
                distribution="normal",
                params={"mu": 1.0, "sigma": 0.1},
                polarity="positive",
            ),
        ]
        user_mat = np.array([[1.0, -0.4], [-0.4, 1.0]])
        traj_cov = np.array([[1.0, 0.3], [0.3, 1.0]])
        pairs = [CorrelationPair(metric_a="a", metric_b="b", coefficient=-0.4)]
        compensated, records = compensate_correlation_matrix(
            user_mat,
            traj_cov,
            metrics,
            pairs,
        )
        assert compensated[0, 1] == pytest.approx(-0.7)
        assert compensated[1, 0] == pytest.approx(-0.7)
        np.testing.assert_allclose(np.diag(compensated), 1.0)
        assert len(records) == 1
        rec = records[0]
        assert rec["metric_a"] == "a"
        assert rec["metric_b"] == "b"
        assert rec["user_target"] == pytest.approx(-0.4)
        assert rec["trajectory_contribution"] == pytest.approx(0.3)
        assert rec["compensated_target"] == pytest.approx(-0.7)
        assert rec["achievable"] == pytest.approx(-0.7)
        assert rec["infeasible"] is False

    def test_clamps_infeasible_to_unit_interval_and_marks(self):
        metrics = [
            Metric(
                name="a",
                label="A",
                distribution="normal",
                params={"mu": 1.0, "sigma": 0.1},
                polarity="positive",
            ),
            Metric(
                name="b",
                label="B",
                distribution="normal",
                params={"mu": 1.0, "sigma": 0.1},
                polarity="positive",
            ),
        ]
        # Trajectory contributes +0.85, user wants -0.5 → -0.5 - 0.85 = -1.35.
        user_mat = np.array([[1.0, -0.5], [-0.5, 1.0]])
        traj_cov = np.array([[1.0, 0.85], [0.85, 1.0]])
        pairs = [CorrelationPair(metric_a="a", metric_b="b", coefficient=-0.5)]
        compensated, records = compensate_correlation_matrix(
            user_mat,
            traj_cov,
            metrics,
            pairs,
        )
        assert compensated[0, 1] == pytest.approx(-1.0)
        assert records[0]["compensated_target"] == pytest.approx(-1.35)
        assert records[0]["achievable"] == pytest.approx(-1.0)
        assert records[0]["infeasible"] is True

    def test_records_sorted_for_determinism(self):
        metrics = [
            Metric(
                name=n,
                label=n,
                distribution="normal",
                params={"mu": 1.0, "sigma": 0.1},
                polarity="positive",
            )
            for n in ("c", "a", "b")
        ]
        n = 3
        user_mat = np.eye(n)
        user_mat[0, 1] = user_mat[1, 0] = 0.5  # c↔a
        user_mat[0, 2] = user_mat[2, 0] = 0.4  # c↔b
        user_mat[1, 2] = user_mat[2, 1] = 0.3  # a↔b
        traj_cov = np.eye(n) * 0.0
        np.fill_diagonal(traj_cov, 1.0)
        pairs = [
            CorrelationPair(metric_a="c", metric_b="a", coefficient=0.5),
            CorrelationPair(metric_a="c", metric_b="b", coefficient=0.4),
            CorrelationPair(metric_a="a", metric_b="b", coefficient=0.3),
        ]
        _, records = compensate_correlation_matrix(
            user_mat,
            traj_cov,
            metrics,
            pairs,
        )
        # Sorted by (metric_a, metric_b) regardless of input order.
        assert [(r["metric_a"], r["metric_b"]) for r in records] == [
            ("a", "b"),
            ("c", "a"),
            ("c", "b"),
        ]


# --- TestEndToEndCompensation -----------------------------------------------


class TestEndToEndCompensation:
    """The mission's headline promise — connections become visible."""

    @pytest.fixture
    def opposes_pair_cfg(self) -> dict[str, Any]:
        """Two archetypes (growth/decline) × four positive-polarity metrics
        × one ``opposes`` connection between m0 and m1.

        ``entity_per_archetype=200`` sets sample size at 9600 cells so
        Pearson stderr lands ~0.01, well below the realized magnitudes
        the AC checks for. Default lognorm ``s=0.55`` puts trajectory
        and copula variances in the same ballpark — the regime where
        the mission's compensation is observable.
        """
        metrics = [_lognorm_metric(f"m{i}") for i in range(4)]
        archetypes = [_growth_archetype(), _decline_archetype()]
        correlations = [
            CorrelationPair(metric_a="m0", metric_b="m1", coefficient=-0.6),
        ]
        return dict(
            metrics=metrics,
            archetypes=archetypes,
            correlations=correlations,
            entity_per_archetype=200,
        )

    def test_compensation_on_flips_realized_sign_negative(
        self,
        opposes_pair_cfg,
    ):
        cfg = _make_config(compensate=True, **opposes_pair_cfg)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tables = generate_tables(cfg)
        observed = _table_corr(tables["fct_metrics"], "m0", "m1")
        # Sign match is the headline AC. Magnitude is bounded by the
        # variance ratio between trajectory and copula — the mission's
        # within-archetype formula doesn't aim for exact recovery, only
        # for delivering the configured sign.
        assert observed < 0.0, (
            f"compensation ON: expected negative table-wide corr(m0, m1), " f"got {observed:+.4f}"
        )

    def test_compensation_off_lets_trajectory_show_through(self, opposes_pair_cfg):
        cfg = _make_config(compensate=False, **opposes_pair_cfg)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tables = generate_tables(cfg)
        observed = _table_corr(tables["fct_metrics"], "m0", "m1")
        # AC: without compensation, the trajectory's structural
        # contribution leaks into the table-wide Pearson and produces a
        # positive realized correlation despite the user's negative
        # target. The mirror archetype mix in the harness diffuses pure
        # trajectory dominance — what's tested is that the residual
        # trajectory signal still flips the realized sign vs. what the
        # user wrote.
        assert observed > 0.0, (
            f"compensation OFF: expected trajectory contribution to drive "
            f"corr(m0, m1) above zero (user wrote -0.6); got {observed:+.4f}"
        )

    def test_compensation_swings_table_corr_toward_configured_sign(
        self,
        opposes_pair_cfg,
    ):
        cfg_off = _make_config(compensate=False, **opposes_pair_cfg)
        cfg_on = _make_config(compensate=True, **opposes_pair_cfg)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_off = generate_tables(cfg_off)["fct_metrics"]
            df_on = generate_tables(cfg_on)["fct_metrics"]
        c_off = _table_corr(df_off, "m0", "m1")
        c_on = _table_corr(df_on, "m0", "m1")
        # Compensation must move realized corr toward the configured
        # negative target. ``c_on < c_off`` is the operational test.
        assert c_on < c_off, (
            f"compensation must shift realized corr(m0, m1) toward the "
            f"configured -0.6 target; observed off={c_off:+.4f} "
            f"on={c_on:+.4f} (no shift in the right direction)"
        )

    def test_zero_correlations_compensation_on_byte_identical_to_off(self):
        # AC: pre-compensation with zero declared correlations is a no-op.
        metrics = [_lognorm_metric(f"m{i}") for i in range(3)]
        archetypes = [_growth_archetype(), _decline_archetype()]
        cfg_off = _make_config(
            metrics=metrics,
            archetypes=archetypes,
            entity_per_archetype=20,
            compensate=False,
        )
        cfg_on = _make_config(
            metrics=metrics,
            archetypes=archetypes,
            entity_per_archetype=20,
            compensate=True,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tables_off = generate_tables(cfg_off)
            tables_on = generate_tables(cfg_on)
        df_off = tables_off["fct_metrics"][["m0", "m1", "m2"]]
        df_on = tables_on["fct_metrics"][["m0", "m1", "m2"]]
        pd.testing.assert_frame_equal(df_off, df_on)


# --- TestSeasonalInteraction ------------------------------------------------


class TestSeasonalInteraction:
    """M119 + M120: seasonal modulation enters the compensation estimate."""

    def test_traj_covariance_uses_seasonal_factors(self):
        # Asymmetric per-metric seasonal_sensitivity is required: with
        # equal sensitivities both metric centers are scaled by the same
        # multiplier per period, which leaves Pearson invariant. Mismatched
        # sensitivities break the proportional scaling so seasonal
        # modulation actually moves the estimated correlation.
        metric_a = Metric(
            name="a",
            label="A",
            distribution="lognorm",
            params={"s": 0.55, "loc": 1.0, "scale": 5.0},
            polarity="positive",
            seasonal_sensitivity=0.0,
        )
        metric_b = Metric(
            name="b",
            label="B",
            distribution="lognorm",
            params={"s": 0.55, "loc": 1.0, "scale": 5.0},
            polarity="positive",
            seasonal_sensitivity=2.0,
        )
        archetypes = [_growth_archetype(), _decline_archetype()]
        cfg_off = _make_config(
            metrics=[metric_a, metric_b],
            archetypes=archetypes,
            entity_per_archetype=10,
        )
        cfg_on = _make_config(
            metrics=[metric_a, metric_b],
            archetypes=archetypes,
            entity_per_archetype=10,
            seasonal_effects=[SeasonalEffect(months=(11, 12, 1), strength=0.6)],
        )
        cov_off = estimate_trajectory_covariance(cfg_off)
        cov_on = estimate_trajectory_covariance(cfg_on)
        assert not np.allclose(cov_off, cov_on)


# --- TestBackwardCompatibility ----------------------------------------------


class TestBackwardCompatibility:
    """Engine-direct configs default to ``compensate_correlations=False``."""

    def test_default_value_on_plotsimconfig_is_false(self):
        # AC: engine-direct configs default to off. Check the field's model
        # default directly — no fixture needed.
        assert PlotsimConfig.model_fields["compensate_correlations"].default is False

    def test_engine_direct_template_has_compensation_disabled(self):
        # Per AC: engine-direct configs default to OFF for byte-identical
        # output. The five bundled engine templates load with the model
        # default, never opt in. ``test_layer4_reference_fixtures_match``
        # is the actual byte-identical regression — this test guards
        # only the field's value.
        from plotsim.config import load_config

        cfg = load_config("plotsim/configs/sample_saas.yaml")
        assert cfg.compensate_correlations is False


# --- TestBuilderIntegration -------------------------------------------------


class TestBuilderIntegration:
    """The builder layer flips the contract: connections always compensated."""

    def test_builder_sets_compensate_correlations_true(self, tmp_path):
        from plotsim.builder import create_from_yaml

        yaml_path = tmp_path / "builder_input.yaml"
        yaml_path.write_text(
            """\
about: SaaS subscription churn
unit: company
window:
  start: 2023-01
  end: 2023-12
  every: monthly
metrics:
  - name: nps
    label: NPS score
    type: score
    polarity: positive
  - name: support_tickets
    label: Tickets
    type: count
    polarity: negative
connections:
  - nps opposes support_tickets
segments:
  - name: champions
    archetype: growth
    count: 10
  - name: at_risk
    archetype: decline
    count: 10
""",
            encoding="utf-8",
        )
        cfg = create_from_yaml(yaml_path)
        assert cfg.compensate_correlations is True


# --- TestManifest -----------------------------------------------------------


class TestManifest:
    """Compensation records appear on the manifest's
    ``correlation_compensations`` list under a builder-style run; engine-
    direct runs leave it ``None``."""

    def test_compensation_run_emits_records(self):
        cfg = _make_config(
            metrics=[_lognorm_metric("a"), _lognorm_metric("b")],
            archetypes=[_growth_archetype(), _decline_archetype()],
            correlations=[
                CorrelationPair(metric_a="a", metric_b="b", coefficient=-0.4),
            ],
            entity_per_archetype=10,
            compensate=True,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tables, state = generate_tables_with_state(cfg)
        manifest = build_manifest(cfg, state.trajectories, tables)
        assert manifest.correlation_compensations is not None
        assert len(manifest.correlation_compensations) == 1
        rec = manifest.correlation_compensations[0]
        assert isinstance(rec, CorrelationCompensation)
        assert (rec.metric_a, rec.metric_b) == ("a", "b")
        assert rec.user_target == pytest.approx(-0.4)
        # trajectory contributes positively for two same-polarity metrics
        # with mirror-image archetypes ⇒ compensated_target < user_target.
        assert rec.compensated_target < rec.user_target

    def test_engine_direct_run_emits_no_compensation_records(self):
        cfg = _make_config(
            metrics=[_lognorm_metric("a"), _lognorm_metric("b")],
            archetypes=[_growth_archetype(), _decline_archetype()],
            correlations=[
                CorrelationPair(metric_a="a", metric_b="b", coefficient=-0.4),
            ],
            entity_per_archetype=5,
            compensate=False,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tables, state = generate_tables_with_state(cfg)
        manifest = build_manifest(cfg, state.trajectories, tables)
        assert manifest.correlation_compensations is None


# --- TestInspectTrace -------------------------------------------------------


class TestInspectTrace:
    """``trace_metric_cell`` must replay against the COMPENSATED Cholesky
    factor, so the coefficient driving each cell matches what the engine
    actually used."""

    def test_inspect_uses_compensated_cholesky(self):
        from plotsim.inspect import _hoist_cholesky
        from plotsim.metrics import (
            _build_correlation_matrix,
            _toposort_metrics,
            project_correlation_matrix,
        )

        cfg_kwargs = dict(
            metrics=[_lognorm_metric("a"), _lognorm_metric("b")],
            archetypes=[_growth_archetype(), _decline_archetype()],
            correlations=[
                CorrelationPair(metric_a="a", metric_b="b", coefficient=-0.5),
            ],
            entity_per_archetype=8,
        )
        cfg_off = _make_config(compensate=False, **cfg_kwargs)
        cfg_on = _make_config(compensate=True, **cfg_kwargs)
        L_off = _hoist_cholesky(cfg_off)
        L_on = _hoist_cholesky(cfg_on)
        # The OFF Cholesky reproduces the user matrix; the ON Cholesky
        # absorbs the compensation. They must differ off-diagonally.
        assert not np.allclose(L_off, L_on)

        # ``L_on`` should be the Cholesky of the COMPENSATED matrix, not
        # the raw user matrix.
        sorted_metrics = _toposort_metrics(list(cfg_on.metrics))
        raw = _build_correlation_matrix(sorted_metrics, list(cfg_on.correlations))
        traj_cov = estimate_trajectory_covariance(cfg_on, metric_order=sorted_metrics)
        compensated, _ = compensate_correlation_matrix(
            raw,
            traj_cov,
            sorted_metrics,
            list(cfg_on.correlations),
        )
        projected, _, _ = project_correlation_matrix(compensated)
        np.testing.assert_allclose(L_on, np.linalg.cholesky(projected))


# --- TestDeterminism --------------------------------------------------------


class TestDeterminism:
    """Same (config, seed) → byte-identical CSVs even with compensation on."""

    def test_compensation_on_determinism(self):
        cfg_kwargs = dict(
            metrics=[_lognorm_metric("a"), _lognorm_metric("b")],
            archetypes=[_growth_archetype(), _decline_archetype()],
            correlations=[
                CorrelationPair(metric_a="a", metric_b="b", coefficient=-0.4),
            ],
            entity_per_archetype=12,
            compensate=True,
            seed=2026,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cfg_a = _make_config(**cfg_kwargs)
            df_a = generate_tables(cfg_a)["fct_metrics"]
            cfg_b = _make_config(**cfg_kwargs)
            df_b = generate_tables(cfg_b)["fct_metrics"]
        pd.testing.assert_frame_equal(df_a, df_b)


# --- TestMetricCap ----------------------------------------------------------


class TestMetricCap:
    """M > ``_MAX_METRICS_FOR_COMPENSATION`` warns and falls through."""

    def test_over_cap_warns_and_falls_back(self):
        # Build a config that crosses the cap. Use few entities and a short
        # time window to keep the cell count under the engine's load-time
        # combined-scale gate.
        n = _MAX_METRICS_FOR_COMPENSATION + 1
        metrics = [_lognorm_metric(f"m{i:02d}") for i in range(n)]
        archetypes = [_growth_archetype()]
        cfg = _make_config(
            metrics=metrics,
            archetypes=archetypes,
            entity_per_archetype=2,
            correlations=[
                CorrelationPair(metric_a="m00", metric_b="m01", coefficient=-0.3),
            ],
            compensate=True,
            start="2024-01",
            end="2024-03",
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            generate_tables(cfg)
        cap_warnings = [
            w for w in caught if "compensate_correlations=true but config has" in str(w.message)
        ]
        assert cap_warnings, (
            "expected metric-cap warning when M exceeds "
            f"_MAX_METRICS_FOR_COMPENSATION ({_MAX_METRICS_FOR_COMPENSATION})"
        )

    def test_exactly_at_cap_runs_compensation(self):
        # Cap is inclusive — `<=` in tables.py and inspect.py.
        n = _MAX_METRICS_FOR_COMPENSATION
        metrics = [_lognorm_metric(f"m{i:02d}") for i in range(n)]
        archetypes = [_growth_archetype()]
        cfg = _make_config(
            metrics=metrics,
            archetypes=archetypes,
            entity_per_archetype=2,
            correlations=[
                CorrelationPair(metric_a="m00", metric_b="m01", coefficient=-0.3),
            ],
            compensate=True,
            start="2024-01",
            end="2024-03",
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            tables, state = generate_tables_with_state(cfg)
        cap_warnings = [
            w for w in caught if "compensate_correlations=true but config has" in str(w.message)
        ]
        assert not cap_warnings
        manifest = build_manifest(cfg, state.trajectories, tables)
        assert manifest.correlation_compensations is not None


# --- TestInfeasibility ------------------------------------------------------


class TestInfeasibility:
    """Trajectory contribution exceeding user target → infeasible record +
    warning, but the engine still produces clean CSV (no NaN, no Cholesky
    failure)."""

    def test_infeasible_pair_warns_and_produces_clean_output(self):
        # Same-polarity metrics on a single growth archetype have within-
        # archetype Pearson ≈ +1.0, so any negative user target produces
        # an infeasible compensated_target (≤ -1.0).
        cfg = _make_config(
            metrics=[_lognorm_metric("a"), _lognorm_metric("b")],
            archetypes=[_growth_archetype()],
            correlations=[
                CorrelationPair(metric_a="a", metric_b="b", coefficient=-0.5),
            ],
            entity_per_archetype=10,
            compensate=True,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            tables, state = generate_tables_with_state(cfg)
        warned = [w for w in caught if "infeasible" in str(w.message)]
        assert warned, "expected infeasibility warning for same-polarity opposes"

        df = tables["fct_metrics"][["a", "b"]]
        assert not df.isna().any().any(), "compensated copula must not yield NaN"

        manifest = build_manifest(cfg, state.trajectories, tables)
        assert manifest.correlation_compensations is not None
        infeasible_records = [r for r in manifest.correlation_compensations if r.infeasible]
        assert len(infeasible_records) == 1
        rec = infeasible_records[0]
        assert rec.compensated_target < -1.0
        assert rec.achievable == pytest.approx(-1.0)
