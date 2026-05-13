"""Internal verification tests — mission: plotsim-mission-internal-verification-tests.

White-box tests that import internal plotsim modules directly and verify each
pipeline stage produces mathematically correct intermediate results. Companion
to ``test_output_fidelity.py``: black-box tests catch *what* is wrong; these
white-box tests pinpoint *where* it is wrong.

Eight categories, one per pipeline stage or cross-stage interaction:
    1. Trajectory engine correctness        (curves → trajectory)
    2. Position → center mapping            (trajectory → metric centers)
    3. Gaussian copula round-trip           (apply_correlations)
    4. Causal lag mechanics                 (_compute_effective_position + toposort)
    5. Noise / outlier / MCAR ordering      (apply_noise, per-period pipeline)
    6. Dimension table construction         (build_all_dimensions, FK sampling)
    7. Validation layer correctness         (PSD / FK / date / causal checks)
    8. Scipy distribution mapping           (_get_scipy_dist ↔ sample_single_metric)
"""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sp_stats
from scipy.stats import norm as sp_norm

from plotsim.config import (
    Archetype,
    CausalLag,
    Column,
    CorrelationPair,
    CurveSegment,
    Domain,
    Entity,
    Metric,
    NoiseConfig,
    OutputConfig,
    PlotsimConfig,
    Table,
    TimeWindow,
    ValueRange,
)
from plotsim.curves import CURVE_REGISTRY
from plotsim.dimensions import (
    build_all_dimensions,
    build_dim_reference,
)
from plotsim.metrics import (
    _compute_effective_position,
    _get_scipy_dist,
    _toposort_metrics,
    apply_correlations,
    apply_noise,
    generate_entity_metrics,
    generate_metrics_for_period,
    position_to_center,
    sample_single_metric,
)
from plotsim.trajectory import compute_trajectory
from plotsim.validation import (
    _lag_alignment_better_for_entity,
    validate_correlation_psd,
    validate_date_spine,
    validate_fk_integrity,
)


# --- Helpers ----------------------------------------------------------------


def _rng(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)


def _metric(
    name: str = "m",
    *,
    distribution: str = "normal",
    params: dict | None = None,
    polarity: str = "positive",
    value_range: ValueRange | None = None,
    causal_lag: CausalLag | None = None,
) -> Metric:
    """Minimal Metric constructor used across classes."""
    if params is None:
        params = {"mu": 10.0, "sigma": 1.0}
    return Metric(
        name=name,
        label=name,
        distribution=distribution,  # type: ignore[arg-type]
        params=params,
        polarity=polarity,  # type: ignore[arg-type]
        value_range=value_range,
        causal_lag=causal_lag,
    )


def _single_segment_archetype(curve: str, params: dict | None = None) -> Archetype:
    return Archetype(
        name=f"{curve}_arch",
        label=curve,
        description="white-box test archetype",
        curve_segments=[
            CurveSegment(
                curve=curve,  # type: ignore[arg-type]
                params=params or {},
                start_pct=0.0,
                end_pct=1.0,
            ),
        ],
    )


# ============================================================================
# Category 1 — Trajectory engine correctness
# ============================================================================


class TestTrajectoryEngine:
    def test_1a_curve_values_at_known_positions(self):
        """Curve implementations match their closed-form math at endpoints.

        plotsim's sigmoid / compound / logistic min-max-normalize so
        endpoints always land at 0/1; we verify those invariants plus
        closed-form values for curves that don't renormalize (decay,
        plateau, oscillating, step, sawtooth).
        """
        # plateau — constant
        traj = compute_trajectory(_single_segment_archetype("plateau", {"level": 0.42}), 50)
        np.testing.assert_allclose(traj, 0.42, atol=1e-12)

        # exp_decay — closed form exp(-rate * t_local)
        rate = 3.0
        traj = compute_trajectory(_single_segment_archetype("exp_decay", {"rate": rate}), 100)
        t_local = np.linspace(0.0, 1.0, 100)
        np.testing.assert_allclose(traj, np.exp(-rate * t_local), atol=1e-12)

        # sigmoid rising — min-max normalization forces endpoints to {0, 1}
        traj = compute_trajectory(
            _single_segment_archetype("sigmoid", {"rising": True, "steepness": 10.0}),
            100,
        )
        assert traj[0] == pytest.approx(0.0, abs=1e-12)
        assert traj[-1] == pytest.approx(1.0, abs=1e-12)

        # step — threshold=0.5, before=1.0, after=0.0
        traj = compute_trajectory(
            _single_segment_archetype("step", {"threshold": 0.5, "before": 1.0, "after": 0.0}),
            100,
        )
        t_local = np.linspace(0.0, 1.0, 100)
        expected = np.where(t_local < 0.5, 1.0, 0.0)
        np.testing.assert_allclose(traj, expected, atol=1e-12)

        # oscillating — amplitude-bounded sinusoid
        amp = 0.2
        center = 0.5
        traj = compute_trajectory(
            _single_segment_archetype(
                "oscillating",
                {"period": 2.0, "amplitude": amp, "center": center},
            ),
            200,
        )
        assert float(traj.min()) >= center - amp - 1e-12
        assert float(traj.max()) <= center + amp + 1e-12

    def test_1b_trajectory_bounds(self):
        """Every registered curve produces values in [0, 1]."""
        # Sensible default params per curve type.
        defaults: dict[str, dict] = {
            "sigmoid": {"rising": True, "steepness": 8.0, "midpoint": 0.5},
            "exp_decay": {"rate": 2.0},
            "step": {"threshold": 0.3, "before": 1.0, "after": 0.0},
            "logistic": {"k": 8.0, "midpoint": 0.5, "ceiling": 1.0},
            "plateau": {"level": 0.6},
            "oscillating": {"period": 3.0, "amplitude": 0.3, "center": 0.5},
            "compound": {"base_rate": 0.05, "acceleration": 0.02},
            "sawtooth": {"period": 2.0, "amplitude": 0.8, "base": 0.1},
        }
        for curve in CURVE_REGISTRY:
            traj = compute_trajectory(
                _single_segment_archetype(curve, defaults[curve]),
                150,
            )
            assert traj.min() >= 0.0 - 1e-12, curve
            assert traj.max() <= 1.0 + 1e-12, curve

    def test_1c_trajectory_monotonicity_where_expected(self):
        """Rising sigmoid is non-decreasing; exp_decay is non-increasing."""
        traj = compute_trajectory(
            _single_segment_archetype("sigmoid", {"rising": True, "steepness": 8.0}),
            100,
        )
        diffs = np.diff(traj)
        assert (diffs >= -1e-12).all()

        traj = compute_trajectory(_single_segment_archetype("exp_decay", {"rate": 2.5}), 100)
        diffs = np.diff(traj)
        assert (diffs <= 1e-12).all()


# ============================================================================
# Category 2 — Position → center mapping
# ============================================================================


class TestPositionToCenter:
    def test_2a_known_mappings_per_distribution(self):
        """Verify the closed-form center formula for each distribution family."""
        # lognorm: center = loc + scale * p
        m = _metric(distribution="lognorm", params={"s": 0.5, "loc": 10.0, "scale": 50.0})
        assert position_to_center(0.0, m) == pytest.approx(10.0)
        assert position_to_center(0.5, m) == pytest.approx(35.0)
        assert position_to_center(1.0, m) == pytest.approx(60.0)

        # gamma: shape * scale * p
        m = _metric(distribution="gamma", params={"shape": 2.0, "scale": 4.0})
        assert position_to_center(0.0, m) == pytest.approx(0.0)
        assert position_to_center(0.5, m) == pytest.approx(4.0)
        assert position_to_center(1.0, m) == pytest.approx(8.0)

        # poisson: lambda * p
        m = _metric(distribution="poisson", params={"lambda": 10.0})
        assert position_to_center(0.0, m) == pytest.approx(0.0)
        assert position_to_center(1.0, m) == pytest.approx(10.0)

        # beta with value_range: rescaled linearly
        m = _metric(
            distribution="beta",
            params={"alpha": 2.0, "beta": 5.0},
            value_range=ValueRange(min=0.0, max=10.0),
        )
        assert position_to_center(0.0, m) == pytest.approx(0.0)
        assert position_to_center(0.5, m) == pytest.approx(5.0)
        assert position_to_center(1.0, m) == pytest.approx(10.0)

        # normal: mu * p
        m = _metric(distribution="normal", params={"mu": 30.0, "sigma": 3.0})
        assert position_to_center(0.5, m) == pytest.approx(15.0)

        # weibull: scale * p
        m = _metric(distribution="weibull", params={"shape": 1.5, "scale": 20.0})
        assert position_to_center(0.25, m) == pytest.approx(5.0)

    def test_2b_polarity_inversion(self):
        """Negative polarity flips position: position_to_center(p, neg) == position_to_center(1-p, pos)."""
        params = {"loc": 0.0, "scale": 100.0, "s": 0.5}
        m_pos = _metric(distribution="lognorm", params=params, polarity="positive")
        m_neg = _metric(distribution="lognorm", params=params, polarity="negative")
        for p in (0.0, 0.2, 0.5, 0.8, 1.0):
            assert position_to_center(p, m_neg) == pytest.approx(position_to_center(1.0 - p, m_pos))


# ============================================================================
# Category 3 — Gaussian copula round-trip
# ============================================================================


class TestGaussianCopula:
    def _two_metrics(self, dist_a: str = "lognorm", dist_b: str = "beta") -> tuple[Metric, Metric]:
        ma = _metric("a", distribution=dist_a, params={"s": 0.3, "loc": 0.0, "scale": 50.0})
        mb = _metric(
            "b",
            distribution=dist_b,
            params={"alpha": 2.0, "beta": 5.0},
            value_range=ValueRange(min=0.0, max=10.0),
        )
        return ma, mb

    def test_3a_identity_correlation_preserves_marginal(self):
        """M127b: ``apply_correlations`` with cholesky_L=I draws independent
        marginals from its own ``rng.standard_normal`` rather than passing
        through caller-supplied ``independent`` values. The contract under
        the new flip is that the OUTPUT marginal still matches each metric's
        distribution (KS test) — not that the cells equal the caller's
        independent draws.
        """
        from scipy import stats as sp_stats

        ma, mb = self._two_metrics()
        ca, cb = 50.0, 5.0
        # Correlation coefficient=0 ⇒ identity off-diagonal ⇒ Cholesky = I.
        corrs = [CorrelationPair(metric_a="a", metric_b="b", coefficient=0.0)]
        rng = _rng(0)

        n = 1000
        out_a, out_b = [], []
        for _ in range(n):
            ia = sample_single_metric(ca, ma, rng)
            ib = sample_single_metric(cb, mb, rng)
            out = apply_correlations(
                {"a": ia, "b": ib},
                {"a": ca, "b": cb},
                corrs,
                [ma, mb],
                cholesky_L=np.eye(2),
                rng=rng,
            )
            out_a.append(out["a"])
            out_b.append(out["b"])

        # KS test: the output marginals must still match each metric's
        # distribution. Identity correlation means no cross-coupling, so
        # the two columns should each look like an independent draw from
        # their own marginal. M127b's Gaussian-space tail clamp causes
        # ~0.005 KS statistic on bounded-support beta marginals at
        # n=1000; threshold 0.001 keeps the test as a regression guard
        # while accepting the new-copula's documented tail-clip behavior.
        from plotsim.metrics import _get_scipy_dist

        dist_a = _get_scipy_dist(ma, ca)
        dist_b = _get_scipy_dist(mb, cb)
        ks_a = sp_stats.kstest(out_a, dist_a.cdf)
        ks_b = sp_stats.kstest(out_b, dist_b.cdf)
        assert ks_a.pvalue > 0.001, f"identity-copula marginal A drifted: {ks_a}"
        assert ks_b.pvalue > 0.001, f"identity-copula marginal B drifted: {ks_b}"

    def test_3b_cdf_round_trip_per_distribution(self):
        """ppf(cdf(x)) == x within tolerance for every continuous family."""
        cases = [
            (
                _metric(distribution="lognorm", params={"s": 0.5, "loc": 0.0, "scale": 50.0}),
                50.0,
                30.0,
            ),
            (_metric(distribution="gamma", params={"shape": 2.0, "scale": 1.0}), 4.0, 3.5),
            (
                _metric(
                    distribution="beta",
                    params={"alpha": 2.0, "beta": 5.0},
                    value_range=ValueRange(min=0.0, max=10.0),
                ),
                3.0,
                2.5,
            ),
            (_metric(distribution="normal", params={"mu": 10.0, "sigma": 2.0}), 5.0, 4.3),
            (_metric(distribution="weibull", params={"shape": 1.5, "scale": 1.0}), 5.0, 3.2),
        ]
        for m, center, x in cases:
            dist = _get_scipy_dist(m, center)
            assert dist is not None, m.distribution
            u = float(dist.cdf(x))
            assert 0.0 < u < 1.0, (m.distribution, u)
            x_back = float(dist.ppf(u))
            assert x_back == pytest.approx(x, abs=1e-8), m.distribution

        # Poisson is discrete: ppf(cdf(k)) recovers k exactly on integers.
        m_poisson = _metric(distribution="poisson", params={"lambda": 5.0})
        dist = _get_scipy_dist(m_poisson, 5.0)
        assert dist is not None
        for k in (0, 1, 5, 10):
            u = float(dist.cdf(k))
            assert int(dist.ppf(u)) == k

    def test_3c_marginal_preservation_under_correlation(self):
        """KS test: correlated output still matches each metric's own distribution."""
        ma, mb = self._two_metrics()
        ca, cb = 50.0, 5.0
        corrs = [CorrelationPair(metric_a="a", metric_b="b", coefficient=0.8)]
        rng = _rng(7)

        n = 2000
        out_a, out_b = [], []
        for _ in range(n):
            ia = sample_single_metric(ca, ma, rng)
            ib = sample_single_metric(cb, mb, rng)
            adj = apply_correlations(
                {"a": ia, "b": ib},
                {"a": ca, "b": cb},
                corrs,
                [ma, mb],
                rng=rng,
            )
            out_a.append(adj["a"])
            out_b.append(adj["b"])

        dist_a = _get_scipy_dist(ma, ca)
        dist_b = _get_scipy_dist(mb, cb)
        ks_a = sp_stats.kstest(out_a, dist_a.cdf)
        ks_b = sp_stats.kstest(out_b, dist_b.cdf)
        assert ks_a.pvalue > 0.01, f"marginal A drifted: {ks_a}"
        assert ks_b.pvalue > 0.01, f"marginal B drifted: {ks_b}"

    def test_3d_correlation_delivery_in_gaussian_space(self):
        """Pearson of cdf→ppf back-mapped values recovers the configured ρ."""
        ma, mb = self._two_metrics()
        ca, cb = 50.0, 5.0
        target = 0.7
        corrs = [CorrelationPair(metric_a="a", metric_b="b", coefficient=target)]
        rng = _rng(99)

        n = 2000
        out_a, out_b = [], []
        for _ in range(n):
            ia = sample_single_metric(ca, ma, rng)
            ib = sample_single_metric(cb, mb, rng)
            adj = apply_correlations(
                {"a": ia, "b": ib},
                {"a": ca, "b": cb},
                corrs,
                [ma, mb],
                rng=rng,
            )
            out_a.append(adj["a"])
            out_b.append(adj["b"])

        dist_a = _get_scipy_dist(ma, ca)
        dist_b = _get_scipy_dist(mb, cb)
        u_a = np.clip(dist_a.cdf(out_a), 1e-10, 1 - 1e-10)
        u_b = np.clip(dist_b.cdf(out_b), 1e-10, 1 - 1e-10)
        z_a = sp_norm.ppf(u_a)
        z_b = sp_norm.ppf(u_b)
        r = float(np.corrcoef(z_a, z_b)[0, 1])
        assert abs(r - target) < 0.05, f"gaussian-space r={r:.3f}, expected ≈ {target}"

    def test_3e_boundary_clamping(self):
        """Extreme-center samples don't propagate NaN/±inf through the copula."""
        ma, mb = self._two_metrics(dist_a="beta", dist_b="beta")
        # Very tight-tail beta at center ≈ 0 — more likely to produce cdf≈0.
        ma = _metric(
            "a",
            distribution="beta",
            params={"alpha": 2.0, "beta": 5.0},
            value_range=ValueRange(min=0.0, max=10.0),
        )
        corrs = [CorrelationPair(metric_a="a", metric_b="b", coefficient=0.9)]
        rng = _rng(1)

        # Use center ≈ 0.05 so beta draws land near the lower bound.
        ca = 0.05
        cb = 0.05
        for _ in range(500):
            ia = sample_single_metric(ca, ma, rng)
            ib = sample_single_metric(cb, mb, rng)
            out = apply_correlations(
                {"a": ia, "b": ib},
                {"a": ca, "b": cb},
                corrs,
                [ma, mb],
                rng=rng,
            )
            va, vb = out["a"], out["b"]
            # Copula must not propagate NaN / ±inf out of the CDF clamp —
            # value_range clamping itself happens later in _clamp_and_round.
            assert np.isfinite(va) and np.isfinite(vb)


# ============================================================================
# Category 4 — Causal lag mechanics
# ============================================================================


class TestCausalLagMechanics:
    def test_4a_toposort_correctness(self):
        """C→B→A chain gets emitted as [A, B, C]."""
        a = _metric("a")
        b = _metric("b", causal_lag=CausalLag(driver="a", lag_periods=1))
        c = _metric("c", causal_lag=CausalLag(driver="b", lag_periods=1))
        ordered = _toposort_metrics([c, a, b])
        names = [m.name for m in ordered]
        assert names.index("a") < names.index("b") < names.index("c")

    def test_4b_toposort_with_no_lags_preserves_declaration_order(self):
        """No-lag metrics round-trip with all names intact (stable order)."""
        metrics = [_metric(n) for n in ("x", "y", "z")]
        ordered = _toposort_metrics(metrics)
        assert {m.name for m in ordered} == {"x", "y", "z"}
        # graphlib.TopologicalSorter emits ready-layer nodes in insertion order.
        assert [m.name for m in ordered] == ["x", "y", "z"]

    def test_4c_lag_buffer_holds_effective_positions(self):
        """Lagged metric's effective position at t reads the driver's
        effective position at t-lag, falling back to own trajectory while
        history is short.
        """
        # Driver "a" has no lag; target "b" lags a by 2, full override.
        a = _metric("a")
        b = _metric("b", causal_lag=CausalLag(driver="a", lag_periods=2, blend_weight=1.0))
        trajectory = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

        # Replay the effective-position logic exactly as
        # ``generate_metrics_for_period`` does it — driver first, target
        # second, buffer populated inline.
        buf: dict[str, list[float]] = {"a": [], "b": []}
        eff_a_series, eff_b_series = [], []
        for t, pos in enumerate(trajectory):
            ea = _compute_effective_position(pos, a, buf, t)
            buf["a"].append(ea)
            eff_a_series.append(ea)
            eb = _compute_effective_position(pos, b, buf, t)
            buf["b"].append(eb)
            eff_b_series.append(eb)

        assert eff_a_series == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        # Periods 0-1: history too short, effective = own trajectory.
        # Period t (>=2): reads buf["a"][t-2] — which is trajectory[t-2].
        assert eff_b_series == pytest.approx([0.1, 0.2, 0.1, 0.2, 0.3, 0.4])

    def test_4d_chain_composition_values(self):
        """A→B(lag 2)→C(lag 1) composes to C(t) reading A(t-3)."""
        a = _metric("a")
        b = _metric("b", causal_lag=CausalLag(driver="a", lag_periods=2, blend_weight=1.0))
        c = _metric("c", causal_lag=CausalLag(driver="b", lag_periods=1, blend_weight=1.0))
        trajectory = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

        buf: dict[str, list[float]] = {"a": [], "b": [], "c": []}
        for t, pos in enumerate(trajectory):
            for m in (a, b, c):  # driver→target order
                e = _compute_effective_position(pos, m, buf, t)
                buf[m.name].append(e)

        # At period 5: eff_c reads buf["b"][4], which was eff_b at t=4
        # reading buf["a"][2] = trajectory[2] = 0.2.
        assert buf["a"][5] == pytest.approx(trajectory[5])
        assert buf["b"][5] == pytest.approx(trajectory[3])
        assert buf["c"][5] == pytest.approx(trajectory[2])

    def test_4e_blend_weight_interpolation(self):
        """Non-unit blend interpolates between current and driver's past."""
        a = _metric("a")
        b = _metric("b", causal_lag=CausalLag(driver="a", lag_periods=3, blend_weight=0.6))
        trajectory = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

        buf: dict[str, list[float]] = {"a": [], "b": []}
        for t, pos in enumerate(trajectory):
            ea = _compute_effective_position(pos, a, buf, t)
            buf["a"].append(ea)
            eb = _compute_effective_position(pos, b, buf, t)
            buf["b"].append(eb)

        # period 5: 0.4 * trajectory[5] + 0.6 * buf["a"][2]
        #         = 0.4 * 0.5 + 0.6 * 0.2 = 0.32
        assert buf["b"][5] == pytest.approx(0.4 * 0.5 + 0.6 * 0.2)

    def test_4f_blend_weight_zero_is_pure_current(self):
        """blend_weight=0 disables the lag entirely."""
        a = _metric("a")
        b = _metric("b", causal_lag=CausalLag(driver="a", lag_periods=3, blend_weight=0.0))
        trajectory = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

        buf: dict[str, list[float]] = {"a": [], "b": []}
        for t, pos in enumerate(trajectory):
            buf["a"].append(_compute_effective_position(pos, a, buf, t))
            buf["b"].append(_compute_effective_position(pos, b, buf, t))
        assert buf["b"] == pytest.approx(trajectory)


# ============================================================================
# Category 5 — Noise / outlier / MCAR ordering
# ============================================================================


class TestNoiseOutlierMCAROrdering:
    def _two_correlated_metrics(self) -> tuple[list[Metric], list[CorrelationPair]]:
        ma = _metric("a", distribution="normal", params={"mu": 50.0, "sigma": 0.0})
        mb = _metric("b", distribution="normal", params={"mu": 50.0, "sigma": 0.0})
        return [ma, mb], [CorrelationPair(metric_a="a", metric_b="b", coefficient=0.9)]

    def test_5a_noise_is_additive_post_correlation(self):
        """Same seed + same trajectory + gaussian-only noise → noise deltas
        between metrics are uncorrelated, proving noise is applied AFTER
        the Cholesky mixes sample residuals.
        """
        metrics, corrs = self._two_correlated_metrics()
        # Use a plateau trajectory so every period shares the same center ⇒
        # sampling RNG is in the identical state across the two runs, and
        # pre-noise correlated values match byte-for-byte between runs.
        traj = np.full(400, 0.7)

        rng_clean = _rng(0)
        vals_clean = generate_entity_metrics(
            traj,
            metrics,
            corrs,
            None,
            rng_clean,
        )
        rng_noisy = _rng(0)
        noise = NoiseConfig(gaussian_sigma=0.1, outlier_rate=0.0, mcar_rate=0.0)
        vals_noisy = generate_entity_metrics(
            traj,
            metrics,
            corrs,
            noise,
            rng_noisy,
        )

        delta_a = np.asarray(vals_noisy["a"], dtype=float) - np.asarray(
            vals_clean["a"], dtype=float
        )
        delta_b = np.asarray(vals_noisy["b"], dtype=float) - np.asarray(
            vals_clean["b"], dtype=float
        )
        # Deltas are the independent per-metric Gaussian draws; their
        # cross-correlation must be near zero (not the 0.9 of the samples).
        r = float(np.corrcoef(delta_a, delta_b)[0, 1])
        assert abs(r) < 0.15, f"noise deltas correlated at r={r:.3f}, expected ≈ 0"

    def test_5b_outlier_replacement_magnitude(self):
        """Forced-outlier replacements land in sign(v)·U(3|v|, 10|v|)."""
        noise = NoiseConfig(gaussian_sigma=0.0, outlier_rate=1.0, mcar_rate=0.0)
        rng = _rng(2024)
        v = 50.0
        for _ in range(500):
            out = apply_noise(v, noise, rng)
            assert out is not None
            assert 3.0 * v <= out <= 10.0 * v, out

        # Negative values stay negative and fall in [-10|v|, -3|v|].
        v = -20.0
        for _ in range(500):
            out = apply_noise(v, noise, rng)
            assert out is not None
            assert -10.0 * abs(v) <= out <= -3.0 * abs(v), out

    def test_5c_mcar_produces_none_when_forced(self):
        """mcar_rate=1.0 nulls every metric in the period."""
        metrics, corrs = self._two_correlated_metrics()
        noise = NoiseConfig(gaussian_sigma=0.0, outlier_rate=0.0, mcar_rate=1.0)
        rng = _rng(5)
        out = generate_metrics_for_period(
            0.5,
            metrics,
            corrs,
            noise,
            None,
            0,
            rng,
        )
        assert out["a"] is None
        assert out["b"] is None

    def test_5d_mcar_does_not_bias_remaining_correlation(self):
        """Values that survive MCAR still carry the configured correlation."""
        # Distributions must have real variance for a meaningful Pearson on the
        # survivors — a plateau + sigma=0 normal collapses everyone to one
        # point and the correlation would be undefined (zero variance).
        ma = _metric(
            "a",
            distribution="lognorm",
            params={"s": 0.3, "loc": 0.0, "scale": 50.0},
        )
        mb = _metric(
            "b",
            distribution="lognorm",
            params={"s": 0.3, "loc": 0.0, "scale": 50.0},
        )
        corrs = [CorrelationPair(metric_a="a", metric_b="b", coefficient=0.8)]
        noise = NoiseConfig(gaussian_sigma=0.0, outlier_rate=0.0, mcar_rate=0.3)
        traj = np.full(3000, 0.7)
        rng = _rng(11)
        out = generate_entity_metrics(traj, [ma, mb], corrs, noise, rng)

        a = np.asarray(out["a"], dtype=object)
        b = np.asarray(out["b"], dtype=object)
        mask = np.array([x is not None and y is not None for x, y in zip(a, b)])
        # MCAR removes ~30% per metric independently; joint survival ~49%.
        assert mask.sum() > 500
        aa = np.array([float(x) for x in a[mask]])
        bb = np.array([float(x) for x in b[mask]])
        r = float(np.corrcoef(aa, bb)[0, 1])
        assert abs(r - 0.8) < 0.1, f"remaining r={r:.3f} (expected ≈ 0.8)"


# ============================================================================
# Category 6 — Dimension table construction
# ============================================================================


def _dim_reference_table(name: str = "dim_plan", values: list[str] | None = None) -> Table:
    values = values or ["free", "pro", "team", "business", "enterprise"]
    return Table(
        name=name,
        type="dim",
        grain="per_reference",
        primary_key="plan_id",
        columns=[
            Column(name="plan_id", dtype="id", source="pk"),
            Column(name="plan_name", dtype="string", source=f"static:{','.join(values)}"),
        ],
    )


def _dim_entity_table_with_fk(name: str = "dim_company", fk_col: str = "plan_id") -> Table:
    return Table(
        name=name,
        type="dim",
        grain="per_entity",
        primary_key="company_id",
        columns=[
            Column(name="company_id", dtype="id", source="pk"),
            Column(name="company_name", dtype="string", source="generated:faker.company"),
            Column(name=fk_col, dtype="id", source="fk:dim_plan.plan_id"),
        ],
    )


class TestDimensionConstruction:
    def test_6a_fk_distribution_is_non_degenerate(self):
        """Multi-row reference dim + ≥100 entities ⇒ FK column has multi-value
        distribution (regression guard on FIX-04 row-0 collapse).
        """
        plan_tbl = _dim_reference_table()
        company_tbl = _dim_entity_table_with_fk()
        entities = [
            Entity(name=f"cohort_{i}", archetype="plateau_arch", size=1) for i in range(100)
        ]
        cfg = PlotsimConfig(
            domain=Domain(name="t", description="t", entity_type="e", entity_label="E"),
            time_window=TimeWindow(start="2024-01", end="2024-03", granularity="monthly"),
            seed=7,
            metrics=[_metric("m", distribution="normal", params={"mu": 1.0, "sigma": 0.1})],
            archetypes=[_single_segment_archetype("plateau", {"level": 0.5})],
            entities=entities,
            tables=[plan_tbl, company_tbl],
            output=OutputConfig(format="csv", directory="out"),
        )
        dims = build_all_dimensions(cfg, _rng(7))
        assert "dim_plan" in dims
        assert "dim_company" in dims
        assert len(dims["dim_company"]) == 100
        unique_plans = dims["dim_company"]["plan_id"].nunique()
        assert unique_plans >= 3, (
            f"only {unique_plans} distinct plans across 100 entities — "
            "FK distribution regressed to single-value collapse"
        )

    def test_6b_dimension_dates_within_configured_window(self):
        """Parameterized faker.date_between values land inside time_window."""
        tbl = Table(
            name="dim_company",
            type="dim",
            grain="per_entity",
            primary_key="company_id",
            columns=[
                Column(name="company_id", dtype="id", source="pk"),
                Column(
                    name="founded_date",
                    dtype="date",
                    source=(
                        "generated:faker.date_between:start_date:2022-01-01:end_date:2024-12-31"
                    ),
                ),
            ],
        )
        entities = [Entity(name=f"cohort_{i}", archetype="plateau_arch", size=1) for i in range(50)]
        cfg = PlotsimConfig(
            domain=Domain(name="t", description="t", entity_type="e", entity_label="E"),
            time_window=TimeWindow(start="2022-01", end="2024-12", granularity="monthly"),
            seed=3,
            metrics=[_metric("m", distribution="normal", params={"mu": 1.0, "sigma": 0.1})],
            archetypes=[_single_segment_archetype("plateau", {"level": 0.5})],
            entities=entities,
            tables=[tbl],
            output=OutputConfig(format="csv", directory="out"),
        )
        dims = build_all_dimensions(cfg, _rng(3))
        df = dims["dim_company"]
        lo = date(2022, 1, 1)
        hi = date(2024, 12, 31)
        for d in df["founded_date"].tolist():
            assert isinstance(d, date)
            assert lo <= d <= hi, f"{d} outside [{lo}, {hi}]"

    def test_6c_reference_dim_row_count_matches_static_values(self):
        """``build_dim_reference`` row count tracks the longest static-CSV column."""
        tbl = _dim_reference_table(values=["a", "b", "c", "d", "e"])
        df = build_dim_reference(tbl, _rng(0))
        assert len(df) == 5
        assert df["plan_name"].tolist() == ["a", "b", "c", "d", "e"]


# ============================================================================
# Category 7 — Validation layer correctness
# ============================================================================


def _bare_psd_config(pairs: list[CorrelationPair]) -> SimpleNamespace:
    """Minimal duck-typed config carrying only what ``validate_correlation_psd``
    reads. PlotsimConfig itself gates non-PSD correlations at load time
    (``_cross_reference_integrity`` via F-04), so we can't build the defective
    matrix through the real model — but the validator is designed to run
    against the same shape, which is all we need here.
    """
    names_used: set[str] = set()
    for p in pairs:
        names_used.update([p.metric_a, p.metric_b])
    metrics = [
        _metric(n, distribution="normal", params={"mu": 1.0, "sigma": 1.0})
        for n in sorted(names_used) or ["m1"]
    ]
    return SimpleNamespace(metrics=metrics, correlations=pairs)


class TestValidationLayer:
    def test_7a_psd_validator_returns_no_issues_after_projection(self):
        """M111: triangle-inequality-violating correlations are auto-corrected
        via Higham nearest-PD projection, not reported as errors.

        Pre-M111 (FIX-F04) the validator returned one error issue for any
        non-PD matrix. Under M111, ``validate_correlation_psd`` runs the
        projection and returns ``[]`` for any matrix that successfully
        projects (the warning is emitted by the load-time pydantic
        validator on ``PlotsimConfig``, not by this re-check). An issue
        is only returned if Higham AND the eigenvalue-clipping fallback
        BOTH fail to produce a Cholesky-able matrix — which is
        impossible for any symmetric input.
        """
        pairs = [
            CorrelationPair(metric_a="a", metric_b="b", coefficient=0.9),
            CorrelationPair(metric_a="a", metric_b="c", coefficient=0.9),
            CorrelationPair(metric_a="b", metric_b="c", coefficient=-0.9),
        ]
        cfg = _bare_psd_config(pairs)
        issues = validate_correlation_psd(cfg)
        assert issues == []

    def test_7b_psd_validator_passes_valid_matrix(self):
        """Identity and mild-correlation matrices pass cleanly."""
        # Empty correlations ⇒ no matrix assembly, no issues.
        cfg_empty = _bare_psd_config([])
        assert validate_correlation_psd(cfg_empty) == []

        # Single moderate pair ⇒ PD 2×2 matrix.
        cfg_mild = _bare_psd_config(
            [
                CorrelationPair(metric_a="a", metric_b="b", coefficient=0.5),
            ]
        )
        assert validate_correlation_psd(cfg_mild) == []

    def test_7c_fk_validator_catches_orphans(self):
        """An FK value absent from the parent PK set is flagged as error."""
        parent_tbl = _dim_reference_table(values=["free", "pro", "team"])
        child_tbl = Table(
            name="fct_usage",
            type="fact",
            grain="per_entity_per_period",
            primary_key=["plan_id"],
            columns=[
                Column(name="plan_id", dtype="id", source="fk:dim_plan.plan_id"),
            ],
        )
        cfg = PlotsimConfig(
            domain=Domain(name="t", description="t", entity_type="e", entity_label="E"),
            time_window=TimeWindow(start="2024-01", end="2024-02", granularity="monthly"),
            seed=1,
            metrics=[_metric("m", distribution="normal", params={"mu": 1.0, "sigma": 1.0})],
            archetypes=[_single_segment_archetype("plateau", {"level": 0.5})],
            entities=[Entity(name="x", archetype="plateau_arch", size=1)],
            tables=[parent_tbl, child_tbl],
            output=OutputConfig(format="csv", directory="out"),
        )
        tables = {
            "dim_plan": pd.DataFrame({"plan_id": ["p-001", "p-002", "p-003"]}),
            "fct_usage": pd.DataFrame({"plan_id": ["p-001", "p-002", "p-999"]}),
        }
        issues = validate_fk_integrity(cfg, tables)
        errors = [i for i in issues if i.severity == "error"]
        assert len(errors) == 1
        assert errors[0].details["orphan_count"] == 1
        assert "p-999" in errors[0].details["orphans_sample"]

    def test_7d_fk_validator_passes_clean_data(self):
        """All FK values resolve ⇒ no errors."""
        parent_tbl = _dim_reference_table(values=["free", "pro"])
        child_tbl = Table(
            name="fct_usage",
            type="fact",
            grain="per_entity_per_period",
            primary_key=["plan_id"],
            columns=[
                Column(name="plan_id", dtype="id", source="fk:dim_plan.plan_id"),
            ],
        )
        cfg = PlotsimConfig(
            domain=Domain(name="t", description="t", entity_type="e", entity_label="E"),
            time_window=TimeWindow(start="2024-01", end="2024-02", granularity="monthly"),
            seed=1,
            metrics=[_metric("m", distribution="normal", params={"mu": 1.0, "sigma": 1.0})],
            archetypes=[_single_segment_archetype("plateau", {"level": 0.5})],
            entities=[Entity(name="x", archetype="plateau_arch", size=1)],
            tables=[parent_tbl, child_tbl],
            output=OutputConfig(format="csv", directory="out"),
        )
        tables = {
            "dim_plan": pd.DataFrame({"plan_id": ["p-001", "p-002"]}),
            "fct_usage": pd.DataFrame({"plan_id": ["p-001", "p-001", "p-002"]}),
        }
        issues = validate_fk_integrity(cfg, tables)
        errors = [i for i in issues if i.severity == "error"]
        assert errors == []

    def test_7e_date_spine_catches_gaps(self):
        """A missing month in a monthly series is reported."""
        cfg = PlotsimConfig(
            domain=Domain(name="t", description="t", entity_type="e", entity_label="E"),
            time_window=TimeWindow(start="2024-01", end="2024-04", granularity="monthly"),
            seed=1,
            metrics=[_metric("m", distribution="normal", params={"mu": 1.0, "sigma": 1.0})],
            archetypes=[_single_segment_archetype("plateau", {"level": 0.5})],
            entities=[Entity(name="x", archetype="plateau_arch", size=1)],
            tables=[],
            output=OutputConfig(format="csv", directory="out"),
        )
        # 2024-02 missing — gap between Jan and Mar.
        dim_date = pd.DataFrame(
            {
                "date_key": [20240101, 20240301, 20240401],
                "date": [date(2024, 1, 1), date(2024, 3, 1), date(2024, 4, 1)],
            }
        )
        issues = validate_date_spine(cfg, {"dim_date": dim_date})
        assert any("gap" in i.message for i in issues)

    def test_7f_causal_coherence_50_percent_ratio_threshold(self):
        """_lag_alignment_better_for_entity applies the 50 % |lagged|/|unlagged| rule."""
        # Perfect lag: shifting the driver by 2 lines up identically with metric.
        driver = np.arange(20, dtype=float)
        metric_perfect_lag = np.concatenate([np.full(2, np.nan), driver[:-2]])
        # Under masked Pearson the lagged correlation should be strong.
        result = _lag_alignment_better_for_entity(metric_perfect_lag, driver, lag=2)
        assert result is True

        # Undefined-correlation branch: a constant metric has zero std, so
        # _pearson returns None and the alignment check bails out early.
        flat_metric = np.zeros(20, dtype=float)
        result = _lag_alignment_better_for_entity(flat_metric, driver, lag=2)
        assert result is None


# ============================================================================
# Category 8 — Scipy distribution mapping
# ============================================================================


class TestScipyDistMapping:
    def test_8a_round_trip_consistency_per_distribution(self):
        """KS test: samples from sample_single_metric match _get_scipy_dist's
        parameterization.
        """
        cases = [
            (_metric(distribution="lognorm", params={"s": 0.5, "loc": 0.0, "scale": 50.0}), 50.0),
            (_metric(distribution="gamma", params={"shape": 3.0, "scale": 1.0}), 4.5),
            (_metric(distribution="normal", params={"mu": 10.0, "sigma": 2.0}), 5.0),
            (_metric(distribution="weibull", params={"shape": 1.5, "scale": 1.0}), 5.0),
            (
                _metric(
                    distribution="beta",
                    params={"alpha": 2.0, "beta": 5.0},
                    value_range=ValueRange(min=0.0, max=10.0),
                ),
                3.0,
            ),
        ]
        for m, center in cases:
            rng = _rng(123)
            samples = np.array([sample_single_metric(center, m, rng) for _ in range(5000)])
            dist = _get_scipy_dist(m, center)
            assert dist is not None
            ks = sp_stats.kstest(samples, dist.cdf)
            assert ks.pvalue > 0.01, f"{m.distribution}: p={ks.pvalue:.4f}"

    def test_8b_lognormal_parameterization_alignment(self):
        """numpy rng.lognormal ↔ scipy stats.lognorm share the same (s, center)."""
        center = 50.0
        s = 0.85
        rng = np.random.default_rng(42)
        samples = rng.lognormal(mean=float(np.log(center)), sigma=s, size=10000)
        ks = sp_stats.kstest(samples, sp_stats.lognorm(s=s, scale=center).cdf)
        assert ks.pvalue > 0.01, f"lognorm numpy↔scipy mismatch: p={ks.pvalue:.4f}"

    def test_8c_poisson_cdf_step_behavior(self):
        """Discrete Poisson CDF's step semantics survive the frozen-dist wrapper."""
        m = _metric(distribution="poisson", params={"lambda": 5.0})
        dist = _get_scipy_dist(m, 5.0)
        assert dist is not None
        # Exact CDF match at integer points
        assert float(dist.cdf(5)) == pytest.approx(float(sp_stats.poisson(mu=5).cdf(5)))
        # Round-trip on integer: ppf(cdf(k)) == k for every k in the support.
        for k in (0, 3, 5, 8):
            assert int(dist.ppf(float(dist.cdf(k)))) == k
        # Step boundary: just past cdf(4), ppf jumps from 4 → 5.
        # scipy.stats.poisson.ppf returns the smallest k with cdf(k) >= p, so
        # the step lives between cdf(4) and cdf(5), not at cdf(5).
        cdf4 = float(dist.cdf(4))
        assert int(dist.ppf(cdf4)) == 4
        assert int(dist.ppf(cdf4 + 1e-6)) == 5
