"""Tests for plotsim.metrics — Mission 004 acceptance criteria.

Covers: position → center per distribution; polarity inversion; per-distribution
sampling validity and monotonicity; value_range clamping and poisson int
output; Cholesky correlated noise (positive, negative, multi-metric);
causal-lag inflection shift, fallback, and no-leak across entities; noise
injection (gaussian, outlier, MCAR); determinism; and end-to-end generation
against both sample YAML configs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from plotsim import load_config
from plotsim.config import (
    CausalLag,
    CorrelationPair,
    Metric,
    NoiseConfig,
    ValueRange,
)
from plotsim.metrics import (
    _toposort_metrics,
    apply_correlations,
    apply_noise,
    generate_entity_metrics,
    generate_metrics_for_period,
    position_to_center,
    sample_single_metric,
)

ROOT = Path(__file__).resolve().parent.parent
SAAS_YAML = ROOT / "plotsim" / "configs" / "sample_saas.yaml"
HR_YAML = ROOT / "plotsim" / "configs" / "sample_hr.yaml"


# --- Fixtures ----------------------------------------------------------------


def _metric(
    name: str = "m",
    *,
    distribution: str = "normal",
    params: dict | None = None,
    polarity: str = "positive",
    value_range: ValueRange | None = None,
    causal_lag: CausalLag | None = None,
) -> Metric:
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


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


# --- position_to_center ------------------------------------------------------


def test_position_to_center_lognorm_linear_in_position():
    m = _metric(distribution="lognorm", params={"s": 0.5, "loc": 0.0, "scale": 100.0})
    assert position_to_center(0.0, m) == pytest.approx(0.0)
    assert position_to_center(0.5, m) == pytest.approx(50.0)
    assert position_to_center(1.0, m) == pytest.approx(100.0)


def test_position_to_center_gamma_shape_times_scale_times_position():
    m = _metric(distribution="gamma", params={"shape": 2.0, "scale": 3.0})
    assert position_to_center(1.0, m) == pytest.approx(6.0)
    assert position_to_center(0.25, m) == pytest.approx(1.5)


def test_position_to_center_poisson():
    m = _metric(distribution="poisson", params={"lambda": 5.0})
    assert position_to_center(1.0, m) == pytest.approx(5.0)
    assert position_to_center(0.1, m) == pytest.approx(0.5)


def test_position_to_center_beta_uses_value_range():
    m = _metric(
        distribution="beta",
        params={"alpha": 2.0, "beta": 5.0},
        value_range=ValueRange(min=0.0, max=10.0),
    )
    assert position_to_center(0.0, m) == pytest.approx(0.0)
    assert position_to_center(0.5, m) == pytest.approx(5.0)
    assert position_to_center(1.0, m) == pytest.approx(10.0)


def test_position_to_center_beta_no_value_range_identity():
    m = _metric(distribution="beta", params={"alpha": 2.0, "beta": 5.0})
    assert position_to_center(0.3, m) == pytest.approx(0.3)


def test_position_to_center_normal():
    m = _metric(distribution="normal", params={"mu": 30.0, "sigma": 15.0})
    assert position_to_center(0.5, m) == pytest.approx(15.0)


def test_position_to_center_weibull():
    m = _metric(distribution="weibull", params={"shape": 1.5, "scale": 10.0})
    assert position_to_center(0.5, m) == pytest.approx(5.0)


def test_position_to_center_polarity_inverts():
    m_pos = _metric(distribution="normal", params={"mu": 10.0, "sigma": 1.0},
                    polarity="positive")
    m_neg = _metric(distribution="normal", params={"mu": 10.0, "sigma": 1.0},
                    polarity="negative")
    assert position_to_center(0.8, m_pos) == pytest.approx(position_to_center(0.2, m_neg))


def test_position_to_center_unsupported_raises():
    # Bypass Pydantic by mutating a dict; build a fake metric via model_construct.
    m = Metric.model_construct(
        name="x", label="x", distribution="bogus", params={},
        polarity="positive",
        value_range=None, causal_lag=None,
    )
    with pytest.raises(ValueError, match="unsupported distribution"):
        position_to_center(0.5, m)


# --- sample_single_metric ----------------------------------------------------


@pytest.mark.parametrize("dist,params,center", [
    ("lognorm", {"s": 0.5, "loc": 0.0, "scale": 100.0}, 50.0),
    ("gamma", {"shape": 2.0, "scale": 3.0}, 3.0),
    ("poisson", {"lambda": 5.0}, 3.0),
    ("beta", {"alpha": 2.0, "beta": 5.0}, 0.5),
    ("normal", {"mu": 30.0, "sigma": 5.0}, 15.0),
    ("weibull", {"shape": 1.5, "scale": 10.0}, 5.0),
])
def test_sample_single_metric_no_nan_or_inf(dist, params, center):
    m = _metric(distribution=dist, params=params)
    rng = _rng(42)
    samples = [sample_single_metric(center, m, rng) for _ in range(200)]
    arr = np.array(samples, dtype=float)
    assert not np.isnan(arr).any()
    assert not np.isinf(arr).any()


def test_sample_gamma_zero_center_returns_zero():
    m = _metric(distribution="gamma", params={"shape": 2.0, "scale": 3.0})
    assert sample_single_metric(0.0, m, _rng(0)) == 0.0


def test_sample_unsupported_raises():
    m = Metric.model_construct(
        name="x", label="x", distribution="bogus", params={},
        polarity="positive",
        value_range=None, causal_lag=None,
    )
    with pytest.raises(ValueError, match="unsupported distribution"):
        sample_single_metric(1.0, m, _rng(0))


# --- Polarity / monotonicity -------------------------------------------------


def _mean_value_at_position(
    metric: Metric, position: float, n: int = 400, seed: int = 0,
) -> float:
    rng = _rng(seed)
    center = position_to_center(position, metric)
    vals = [sample_single_metric(center, metric, rng) for _ in range(n)]
    return float(np.mean(vals))


@pytest.mark.parametrize("dist,params", [
    ("lognorm", {"s": 0.5, "loc": 0.0, "scale": 100.0}),
    ("gamma", {"shape": 2.0, "scale": 3.0}),
    ("poisson", {"lambda": 20.0}),
    ("beta", {"alpha": 2.0, "beta": 5.0}),
    ("normal", {"mu": 30.0, "sigma": 5.0}),
    ("weibull", {"shape": 1.5, "scale": 10.0}),
])
def test_positive_polarity_increases_mean_with_position(dist, params):
    vr = ValueRange(min=0.0, max=1.0) if dist == "beta" else None
    m = _metric(distribution=dist, params=params, polarity="positive",
                value_range=vr)
    low = _mean_value_at_position(m, 0.1, seed=0)
    high = _mean_value_at_position(m, 0.9, seed=0)
    assert high > low, f"{dist}: expected high-pos mean > low-pos mean (got {high} vs {low})"


@pytest.mark.parametrize("dist,params", [
    ("lognorm", {"s": 0.5, "loc": 0.0, "scale": 100.0}),
    ("gamma", {"shape": 2.0, "scale": 3.0}),
    ("poisson", {"lambda": 20.0}),
    ("beta", {"alpha": 2.0, "beta": 5.0}),
    ("normal", {"mu": 30.0, "sigma": 5.0}),
])
def test_negative_polarity_decreases_mean_with_position(dist, params):
    vr = ValueRange(min=0.0, max=1.0) if dist == "beta" else None
    m = _metric(distribution=dist, params=params, polarity="negative",
                value_range=vr)
    low = _mean_value_at_position(m, 0.1, seed=1)
    high = _mean_value_at_position(m, 0.9, seed=1)
    assert low > high, f"{dist}: expected low-pos mean > high-pos mean (got {low} vs {high})"


# --- Clamping + rounding -----------------------------------------------------


def test_value_range_clamps_min_and_max():
    m = _metric(distribution="normal", params={"mu": 100.0, "sigma": 50.0},
                value_range=ValueRange(min=20.0, max=40.0))
    out = generate_entity_metrics(
        trajectory=np.full(500, 0.5),
        metrics=[m],
        correlations=None,
        noise=None,
        rng=_rng(0),
    )
    arr = out["m"].astype(float)
    assert (arr >= 20.0).all()
    assert (arr <= 40.0).all()


def test_poisson_returns_int_array():
    m = _metric(distribution="poisson", params={"lambda": 5.0})
    out = generate_entity_metrics(
        trajectory=np.linspace(0.1, 0.9, 20),
        metrics=[m],
        correlations=None,
        noise=None,
        rng=_rng(0),
    )
    assert out["m"].dtype.kind in ("i", "u")
    # And every value is a non-negative integer
    assert (out["m"] >= 0).all()


# --- Correlated noise via Cholesky ------------------------------------------


def _two_metric_correlated_series(
    coeff: float, n_periods: int = 1200, seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Drive two normal metrics with a flat trajectory so only residuals co-move."""
    m_a = _metric("a", distribution="normal", params={"mu": 100.0, "sigma": 10.0})
    m_b = _metric("b", distribution="normal", params={"mu": 100.0, "sigma": 10.0})
    pair = CorrelationPair(metric_a="a", metric_b="b", coefficient=coeff)
    out = generate_entity_metrics(
        trajectory=np.full(n_periods, 0.5),
        metrics=[m_a, m_b],
        correlations=[pair],
        noise=None,
        rng=_rng(seed),
    )
    return out["a"].astype(float), out["b"].astype(float)


def test_correlation_positive_high_coefficient():
    a, b = _two_metric_correlated_series(0.8)
    r = float(np.corrcoef(a, b)[0, 1])
    assert 0.65 <= r <= 0.92, f"expected ~0.8, got {r}"


def test_correlation_negative_high_coefficient():
    a, b = _two_metric_correlated_series(-0.8)
    r = float(np.corrcoef(a, b)[0, 1])
    assert -0.92 <= r <= -0.65, f"expected ~-0.8, got {r}"


def test_no_correlations_config_independent():
    m_a = _metric("a", distribution="normal", params={"mu": 100.0, "sigma": 10.0})
    m_b = _metric("b", distribution="normal", params={"mu": 100.0, "sigma": 10.0})
    out = generate_entity_metrics(
        trajectory=np.full(1200, 0.5),
        metrics=[m_a, m_b],
        correlations=None,
        noise=None,
        rng=_rng(7),
    )
    r = float(np.corrcoef(out["a"].astype(float), out["b"].astype(float))[0, 1])
    assert abs(r) < 0.2, f"expected near-zero correlation, got {r}"


@pytest.mark.parametrize("k", [2, 4, 6])
def test_cholesky_handles_multi_metric_sizes(k):
    """Identity matrix (no pairs) is trivially PSD; just check it runs."""
    metrics = [
        _metric(f"m{i}", distribution="normal",
                params={"mu": 10.0 + i, "sigma": 1.0})
        for i in range(k)
    ]
    # One chained correlation pair to exercise Cholesky on a k x k matrix.
    correlations = [
        CorrelationPair(metric_a=f"m{i}", metric_b=f"m{i+1}", coefficient=0.3)
        for i in range(k - 1)
    ]
    out = generate_entity_metrics(
        trajectory=np.full(50, 0.5),
        metrics=metrics,
        correlations=correlations,
        noise=None,
        rng=_rng(0),
    )
    assert len(out) == k
    for m in metrics:
        arr = out[m.name].astype(float)
        assert not np.isnan(arr).any()


def test_non_psd_correlation_matrix_raises():
    # 3 metrics with every off-diagonal at -1.0 → not positive semi-definite.
    # Per FIX-01, apply_correlations now raises ValueError instead of silently
    # falling back to independent samples — silent fallback hid configuration
    # defects from every downstream statistical check.
    metrics = [_metric(f"m{i}", distribution="normal",
                       params={"mu": 10.0, "sigma": 1.0}) for i in range(3)]
    correlations = [
        CorrelationPair(metric_a="m0", metric_b="m1", coefficient=-1.0),
        CorrelationPair(metric_a="m0", metric_b="m2", coefficient=-1.0),
        CorrelationPair(metric_a="m1", metric_b="m2", coefficient=-1.0),
    ]
    with pytest.raises(ValueError, match="positive semi-definite"):
        generate_entity_metrics(
            trajectory=np.full(20, 0.5),
            metrics=metrics,
            correlations=correlations,
            noise=None,
            rng=_rng(0),
        )


def test_apply_correlations_empty_list_returns_input():
    m_a = _metric("a", distribution="normal", params={"mu": 10.0, "sigma": 1.0})
    m_b = _metric("b", distribution="normal", params={"mu": 10.0, "sigma": 1.0})
    indep = {"a": 11.0, "b": 9.0}
    centers = {"a": 10.0, "b": 10.0}
    out = apply_correlations(indep, centers, [], [m_a, m_b])
    assert out == indep


# --- Causal lag --------------------------------------------------------------


def test_causal_lag_no_lag_matches_trajectory():
    """Without causal_lag, generate_entity_metrics uses the trajectory unmodified."""
    driver = _metric("driver", distribution="normal",
                     params={"mu": 100.0, "sigma": 0.01})
    rng = _rng(0)
    # A step trajectory: low → high.
    traj = np.concatenate([np.full(10, 0.2), np.full(10, 0.8)])
    out = generate_entity_metrics(
        trajectory=traj, metrics=[driver], correlations=None, noise=None, rng=rng,
    )
    arr = out["driver"].astype(float)
    # Step in trajectory at t=10 → step in mean at t=10.
    assert arr[:10].mean() < arr[10:].mean() - 30


def test_causal_lag_shifts_inflection():
    """A lagged metric's inflection appears ~lag periods after the driver's."""
    # Driver: positive polarity, tracks trajectory directly.
    driver = _metric("driver", distribution="normal",
                     params={"mu": 100.0, "sigma": 0.01})
    # Lagged: positive polarity, same distribution, lag_periods=3.
    lagged = _metric(
        "lagged", distribution="normal",
        params={"mu": 100.0, "sigma": 0.01},
        causal_lag=CausalLag(driver="driver", lag_periods=3),
    )
    traj = np.concatenate([np.full(15, 0.1), np.full(15, 0.9)])
    out = generate_entity_metrics(
        trajectory=traj, metrics=[driver, lagged],
        correlations=None, noise=None, rng=_rng(0),
    )
    driver_arr = out["driver"].astype(float)
    lagged_arr = out["lagged"].astype(float)
    # 0.4.0 default blend_weight=1.0 collapses the blend to driver_past, so
    # the lagged series is a 3-period shift of the driver series. Driver
    # jumps between t=14→15; lagged reads lag_buffer[driver][t-3], which
    # transitions between t=17 (reads index 14, still 0.1) and t=18 (reads
    # index 15, now 0.9). Weak assertion below survives any blend_weight
    # in (0, 1] since the lagged inflection is always strictly later than
    # the driver's.
    driver_inflection = int(np.argmax(np.diff(driver_arr)))
    lagged_inflection = int(np.argmax(np.diff(lagged_arr)))
    assert lagged_inflection > driver_inflection


def test_causal_lag_fallback_when_insufficient_history():
    """For period < lag_periods, effective position = current position."""
    driver = _metric("driver", distribution="normal",
                     params={"mu": 100.0, "sigma": 0.01})
    # lag=5, but trajectory starts high — first 5 periods should match "high".
    lagged = _metric(
        "lagged", distribution="normal",
        params={"mu": 100.0, "sigma": 0.01},
        causal_lag=CausalLag(driver="driver", lag_periods=5),
    )
    traj = np.full(10, 0.9)
    out = generate_entity_metrics(
        trajectory=traj, metrics=[driver, lagged],
        correlations=None, noise=None, rng=_rng(0),
    )
    lagged_arr = out["lagged"].astype(float)
    # All periods should produce ~90 (mu * 0.9) since position is flat.
    assert lagged_arr.mean() == pytest.approx(90.0, abs=1.0)


def test_causal_lag_no_entity_leak():
    """Two entities generated sequentially don't share lag history."""
    driver = _metric("driver", distribution="normal",
                     params={"mu": 100.0, "sigma": 0.01})
    lagged = _metric(
        "lagged", distribution="normal",
        params={"mu": 100.0, "sigma": 0.01},
        causal_lag=CausalLag(driver="driver", lag_periods=2),
    )
    rng = _rng(0)
    # Entity 1: high trajectory.
    out1 = generate_entity_metrics(
        trajectory=np.full(5, 0.9), metrics=[driver, lagged],
        correlations=None, noise=None, rng=rng,
    )
    # Entity 2: low trajectory. If lag_buffer leaked, entity 2's first 2 periods
    # would blend against entity 1's 0.9 driver history.
    out2 = generate_entity_metrics(
        trajectory=np.full(5, 0.1), metrics=[driver, lagged],
        correlations=None, noise=None, rng=rng,
    )
    arr2 = out2["lagged"].astype(float)
    # Should all be around mu*0.1 = 10.
    assert arr2.mean() == pytest.approx(10.0, abs=1.0)
    # Sanity: entity 1 was around 90.
    arr1 = out1["lagged"].astype(float)
    assert arr1.mean() == pytest.approx(90.0, abs=1.0)


def test_causal_lag_blend_weight_default_is_full_override():
    """0.4.0: CausalLag.blend_weight defaults to 1.0 — metric-at-T equals
    driver-at-(T-N), xcorr peaks at exactly lag_periods."""
    lag = CausalLag(driver="driver", lag_periods=2)
    assert lag.blend_weight == 1.0


# --- F-02 / 0.4.0: chain composition, blend weight, topological order -------


def test_causal_lag_full_override_equals_shifted_driver():
    """R-11: blend_weight=1.0 makes lagged[t] pointwise equal driver[t-N].

    With sigma=0 the normal distribution collapses to a delta on ``center``,
    so driver[t] = mu * traj[t] and lagged[t] = mu * traj[t-N] for t >= N.
    """
    driver = _metric("driver", distribution="normal",
                     params={"mu": 100.0, "sigma": 0.0})
    lagged = _metric(
        "lagged", distribution="normal",
        params={"mu": 100.0, "sigma": 0.0},
        causal_lag=CausalLag(driver="driver", lag_periods=3, blend_weight=1.0),
    )
    traj = np.linspace(0.1, 0.9, 30)
    out = generate_entity_metrics(
        trajectory=traj, metrics=[driver, lagged],
        correlations=None, noise=None, rng=_rng(0),
    )
    d = out["driver"].astype(float)
    l = out["lagged"].astype(float)
    np.testing.assert_allclose(l[3:], d[:-3], atol=1e-9)


def test_causal_lag_blend_weight_06_recovers_prior_behavior():
    """R-12: blend_weight=0.6 reproduces the pre-0.4.0 hardcoded blend.

    lagged[t] = mu * (0.4*traj[t] + 0.6*traj[t-N]) for t >= N. Users who want
    the softer behavior set blend_weight explicitly rather than getting it
    silently as a default.
    """
    driver = _metric("driver", distribution="normal",
                     params={"mu": 100.0, "sigma": 0.0})
    lagged = _metric(
        "lagged", distribution="normal",
        params={"mu": 100.0, "sigma": 0.0},
        causal_lag=CausalLag(driver="driver", lag_periods=3, blend_weight=0.6),
    )
    traj = np.linspace(0.1, 0.9, 30)
    out = generate_entity_metrics(
        trajectory=traj, metrics=[driver, lagged],
        correlations=None, noise=None, rng=_rng(0),
    )
    l = out["lagged"].astype(float)
    mu = 100.0
    expected = mu * (0.4 * traj[3:] + 0.6 * traj[:-3])
    np.testing.assert_allclose(l[3:], expected, atol=1e-9)


def test_causal_lag_chain_composes():
    """R-13: A → B(lag=2) → C(lag=3), all blend_weight=1.0.

    With effective-position buffering + topological processing order, C at
    period t reads B's effective position at t-3, which was itself buffered
    as A at (t-3)-2 = t-5. So C[t] = A[t-5] for t >= 5.
    """
    a = _metric("a", distribution="normal",
                params={"mu": 100.0, "sigma": 0.0})
    b = _metric(
        "b", distribution="normal",
        params={"mu": 100.0, "sigma": 0.0},
        causal_lag=CausalLag(driver="a", lag_periods=2, blend_weight=1.0),
    )
    c = _metric(
        "c", distribution="normal",
        params={"mu": 100.0, "sigma": 0.0},
        causal_lag=CausalLag(driver="b", lag_periods=3, blend_weight=1.0),
    )
    traj = np.linspace(0.1, 0.9, 30)
    out = generate_entity_metrics(
        trajectory=traj, metrics=[a, b, c],
        correlations=None, noise=None, rng=_rng(0),
    )
    a_arr = out["a"].astype(float)
    c_arr = out["c"].astype(float)
    np.testing.assert_allclose(c_arr[5:], a_arr[:-5], atol=1e-9)


def test_causal_lag_driver_field_is_meaningful():
    """R-14: naming a lagged metric as driver vs a non-lagged one changes output.

    Pre-0.4.0, buffer stored raw trajectory for every key, so the driver
    field was a no-op (all targets with the same lag_periods produced
    identical series). After the fix, driver's own lag propagates through
    the buffer and the target reads the chained value.

    Setup:
      base            — no lag (effective = trajectory)
      intermediate    — lag=5 on base (effective = trajectory shifted by 5)
      target_base     — lag=2 on base          → target[t] = traj[t-2]
      target_intermed — lag=2 on intermediate  → target[t] = traj[t-7] for t >= 7
    """
    base = _metric("base", distribution="normal",
                   params={"mu": 100.0, "sigma": 0.0})
    intermediate = _metric(
        "intermediate", distribution="normal",
        params={"mu": 100.0, "sigma": 0.0},
        causal_lag=CausalLag(driver="base", lag_periods=5, blend_weight=1.0),
    )
    target_base = _metric(
        "target_base", distribution="normal",
        params={"mu": 100.0, "sigma": 0.0},
        causal_lag=CausalLag(driver="base", lag_periods=2, blend_weight=1.0),
    )
    target_intermed = _metric(
        "target_intermed", distribution="normal",
        params={"mu": 100.0, "sigma": 0.0},
        causal_lag=CausalLag(driver="intermediate", lag_periods=2, blend_weight=1.0),
    )
    traj = np.linspace(0.1, 0.9, 30)
    out = generate_entity_metrics(
        trajectory=traj,
        metrics=[base, intermediate, target_base, target_intermed],
        correlations=None, noise=None, rng=_rng(0),
    )
    t_base = out["target_base"].astype(float)
    t_inter = out["target_intermed"].astype(float)
    # Post-resolution window: target_intermed reads through the 5-period
    # upstream lag plus its own 2-period, so t >= 7 is where both are fully
    # resolved to a shifted trajectory read.
    np.testing.assert_allclose(t_base[7:], 100.0 * traj[5:-2], atol=1e-9)
    np.testing.assert_allclose(t_inter[7:], 100.0 * traj[:-7], atol=1e-9)
    # Sanity: the two targets disagree (pre-fix they would be identical).
    assert np.max(np.abs(t_base[7:] - t_inter[7:])) > 1.0


def test_toposort_metrics_orders_driver_before_target():
    """R-15: _toposort_metrics yields driver-before-target order even when
    metrics are declared in reverse."""
    a = _metric("a")
    b = _metric(
        "b",
        causal_lag=CausalLag(driver="a", lag_periods=1, blend_weight=1.0),
    )
    c = _metric(
        "c",
        causal_lag=CausalLag(driver="b", lag_periods=1, blend_weight=1.0),
    )
    # Input order [c, b, a]; expected output has a < b < c.
    names = [m.name for m in _toposort_metrics([c, b, a])]
    assert names.index("a") < names.index("b") < names.index("c")


def test_toposort_metrics_preserves_declaration_order_without_chains():
    """Independent metrics must land in insertion order, so configs without
    causal_lag keep byte-identical RNG consumption vs pre-0.4.0."""
    names_in = ["alpha", "bravo", "charlie", "delta", "echo"]
    metrics_in = [_metric(n) for n in names_in]
    out = [m.name for m in _toposort_metrics(metrics_in)]
    assert out == names_in


def test_causal_lag_saas_template_determinism():
    """R-16: SaaS template (has a causal lag) is byte-identical across two
    runs with the same seed, including the lagged metric's array."""
    cfg = load_config(SAAS_YAML)
    traj = np.linspace(0.05, 0.95, 24)
    out1 = generate_entity_metrics(
        trajectory=traj, metrics=list(cfg.metrics),
        correlations=list(cfg.correlations), noise=cfg.noise,
        rng=_rng(cfg.seed),
    )
    out2 = generate_entity_metrics(
        trajectory=traj, metrics=list(cfg.metrics),
        correlations=list(cfg.correlations), noise=cfg.noise,
        rng=_rng(cfg.seed),
    )
    assert set(out1.keys()) == set(out2.keys())
    for name in out1:
        np.testing.assert_array_equal(
            np.asarray(out1[name]), np.asarray(out2[name]),
        )


# --- Noise injection ---------------------------------------------------------


def test_noise_gaussian_sigma_zero_is_identity():
    m = _metric(distribution="normal", params={"mu": 50.0, "sigma": 0.01})
    noise = NoiseConfig(gaussian_sigma=0.0, outlier_rate=0.0, mcar_rate=0.0)
    rng1 = _rng(0)
    clean_out = generate_entity_metrics(
        trajectory=np.full(50, 0.5), metrics=[m],
        correlations=None, noise=None, rng=rng1,
    )
    rng2 = _rng(0)
    noised_out = generate_entity_metrics(
        trajectory=np.full(50, 0.5), metrics=[m],
        correlations=None, noise=noise, rng=rng2,
    )
    np.testing.assert_array_equal(clean_out["m"], noised_out["m"])


def test_noise_outlier_rate_approximately_matches():
    m = _metric(distribution="normal", params={"mu": 100.0, "sigma": 5.0})
    noise = NoiseConfig(gaussian_sigma=0.0, outlier_rate=0.05, mcar_rate=0.0)
    out = generate_entity_metrics(
        trajectory=np.full(10000, 0.5), metrics=[m],
        correlations=None, noise=noise, rng=_rng(0),
    )
    arr = out["m"].astype(float)
    # Baseline sample is ~N(50, 5). Outlier = uniform(3x, 10x) = 150–500.
    # Count values beyond what clean sampling could reasonably produce.
    outlier_count = int((arr > 130.0).sum())
    rate = outlier_count / len(arr)
    assert 0.03 < rate < 0.07, f"expected ~5% outliers, got {rate:.3f}"


def test_noise_mcar_rate_approximately_matches():
    m = _metric(distribution="normal", params={"mu": 100.0, "sigma": 5.0})
    noise = NoiseConfig(gaussian_sigma=0.0, outlier_rate=0.0, mcar_rate=0.02)
    out = generate_entity_metrics(
        trajectory=np.full(10000, 0.5), metrics=[m],
        correlations=None, noise=noise, rng=_rng(0),
    )
    none_count = sum(1 for v in out["m"] if v is None)
    rate = none_count / len(out["m"])
    assert 0.01 < rate < 0.03, f"expected ~2% MCAR nulls, got {rate:.3f}"


def test_apply_noise_all_zeros_passthrough():
    noise = NoiseConfig()
    rng = _rng(0)
    # None of the branches should fire; rng should be untouched.
    state_before = rng.bit_generator.state
    v = apply_noise(42.5, noise, rng)
    assert v == 42.5
    assert rng.bit_generator.state == state_before


# --- Determinism -------------------------------------------------------------


def test_same_seed_identical_output():
    m = _metric(distribution="lognorm",
                params={"s": 0.5, "loc": 0.0, "scale": 100.0})
    traj = np.linspace(0.1, 0.9, 24)
    a = generate_entity_metrics(traj, [m], None, None, _rng(123))
    b = generate_entity_metrics(traj, [m], None, None, _rng(123))
    np.testing.assert_array_equal(a["m"], b["m"])


def test_different_seed_different_values_same_shape():
    m = _metric(distribution="lognorm",
                params={"s": 0.5, "loc": 0.0, "scale": 100.0})
    traj = np.linspace(0.1, 0.9, 200)
    a = generate_entity_metrics(traj, [m], None, None, _rng(1))["m"].astype(float)
    b = generate_entity_metrics(traj, [m], None, None, _rng(2))["m"].astype(float)
    # Same shape.
    assert a.shape == b.shape
    # Not identical.
    assert not np.array_equal(a, b)
    # Similar mean (same distributional shape).
    assert abs(a.mean() - b.mean()) / a.mean() < 0.25


# --- generate_metrics_for_period signature ----------------------------------


def test_generate_metrics_for_period_returns_per_metric_dict():
    metrics = [
        _metric("a", distribution="normal", params={"mu": 10.0, "sigma": 1.0}),
        _metric("b", distribution="poisson", params={"lambda": 5.0}),
    ]
    out = generate_metrics_for_period(
        trajectory_position=0.5,
        metrics=metrics,
        correlations=None,
        noise=None,
        lag_buffer=None,
        period_index=0,
        rng=_rng(0),
    )
    assert set(out.keys()) == {"a", "b"}
    assert isinstance(out["a"], float)
    # Poisson values are rounded to ints but stored as floats (np will cast).
    assert out["b"] == int(out["b"])


# --- End-to-end with sample YAMLs -------------------------------------------


def test_sample_saas_every_metric_generates():
    cfg = load_config(SAAS_YAML)
    rng = _rng(cfg.seed)
    # Use a non-trivial trajectory to exercise lag buffers + correlations.
    traj = np.linspace(0.05, 0.95, 24)
    out = generate_entity_metrics(
        trajectory=traj,
        metrics=list(cfg.metrics),
        correlations=list(cfg.correlations),
        noise=cfg.noise,
        rng=rng,
    )
    assert set(out.keys()) == {m.name for m in cfg.metrics}
    for m in cfg.metrics:
        arr = out[m.name]
        assert len(arr) == 24, f"{m.name} wrong length"
        # No NaN/inf among non-null values.
        numeric = np.array([v for v in arr if v is not None], dtype=float)
        assert not np.isnan(numeric).any(), f"{m.name} has NaN"
        assert not np.isinf(numeric).any(), f"{m.name} has inf"


def test_sample_hr_every_metric_generates():
    cfg = load_config(HR_YAML)
    rng = _rng(cfg.seed)
    traj = np.linspace(0.05, 0.95, 36)
    out = generate_entity_metrics(
        trajectory=traj,
        metrics=list(cfg.metrics),
        correlations=list(cfg.correlations),
        noise=cfg.noise,
        rng=rng,
    )
    assert set(out.keys()) == {m.name for m in cfg.metrics}
    for m in cfg.metrics:
        arr = out[m.name]
        assert len(arr) == 36
        numeric = np.array([v for v in arr if v is not None], dtype=float)
        assert not np.isnan(numeric).any()
        assert not np.isinf(numeric).any()


# --- Category B Layer 3: Cholesky hoist (SEC-08) -----------------------------


def test_build_correlation_matrix_structure():
    """The helper returns a symmetric matrix with 1s on the diagonal, the
    configured coefficients at (i, j) / (j, i), and 0s elsewhere. Metric
    order determines index; pairs referencing unknown metrics are skipped
    (matches the pre-hoist behavior).
    """
    from plotsim.metrics import _build_correlation_matrix
    metrics = [
        Metric(name="a", label="A", distribution="normal",
               params={"sigma": 1.0}, polarity="positive"),
        Metric(name="b", label="B", distribution="normal",
               params={"sigma": 1.0}, polarity="positive"),
        Metric(name="c", label="C", distribution="normal",
               params={"sigma": 1.0}, polarity="positive"),
    ]
    correlations = [
        CorrelationPair(metric_a="a", metric_b="b", coefficient=0.5),
        CorrelationPair(metric_a="a", metric_b="c", coefficient=-0.3),
    ]
    mat = _build_correlation_matrix(metrics, correlations)
    assert mat.shape == (3, 3)
    for i in range(3):
        assert mat[i, i] == 1.0
    assert mat[0, 1] == 0.5
    assert mat[1, 0] == 0.5
    assert mat[0, 2] == -0.3
    assert mat[2, 0] == -0.3
    assert mat[1, 2] == 0.0


def test_apply_correlations_cholesky_L_matches_unhoisted():
    """apply_correlations must return identical output whether it computes
    Cholesky internally or receives a pre-computed L."""
    from plotsim.metrics import _build_correlation_matrix, apply_correlations
    metrics = [
        Metric(name="a", label="A", distribution="normal",
               params={"sigma": 1.0}, polarity="positive"),
        Metric(name="b", label="B", distribution="normal",
               params={"sigma": 1.0}, polarity="positive"),
    ]
    correlations = [
        CorrelationPair(metric_a="a", metric_b="b", coefficient=0.7),
    ]
    centers = {"a": 10.0, "b": 5.0}
    independent = {"a": 11.0, "b": 4.5}

    mat = _build_correlation_matrix(metrics, correlations)
    L = np.linalg.cholesky(mat)

    out_internal = apply_correlations(independent, centers, correlations, metrics)
    out_hoisted = apply_correlations(
        independent, centers, correlations, metrics, cholesky_L=L,
    )
    assert set(out_internal.keys()) == set(out_hoisted.keys())
    for k in out_internal:
        assert out_internal[k] == pytest.approx(out_hoisted[k], rel=1e-12)


def test_apply_correlations_empty_list_ignores_cholesky_L():
    """When correlations is empty, apply_correlations short-circuits and never
    touches cholesky_L — even a deliberately bogus L must not affect output.
    """
    from plotsim.metrics import apply_correlations
    metrics = [
        Metric(name="a", label="A", distribution="normal",
               params={"sigma": 1.0}, polarity="positive"),
    ]
    bogus_L = np.array([[999.0]])
    out = apply_correlations({"a": 1.0}, {"a": 1.0}, [], metrics, cholesky_L=bogus_L)
    assert out == {"a": 1.0}


def test_generate_tables_byte_identical_across_runs():
    """Same (config, seed) must produce frame-equal output end-to-end after
    the hoist. The hoist is semantically null: same RNG consumption order,
    same numerical path.
    """
    from plotsim.tables import generate_tables
    import pandas as pd
    cfg = load_config(SAAS_YAML)
    a = generate_tables(cfg, np.random.default_rng(cfg.seed))
    b = generate_tables(cfg, np.random.default_rng(cfg.seed))
    assert set(a.keys()) == set(b.keys())
    for name in a:
        pd.testing.assert_frame_equal(
            a[name].reset_index(drop=True),
            b[name].reset_index(drop=True),
        )


def test_generate_tables_no_correlations_flows_none_L_through():
    """A config with no correlations must flow cholesky_L=None through the
    chain without attempting any matrix construction."""
    from plotsim.config import PlotsimConfig
    from plotsim.tables import generate_tables
    raw = {
        "domain": {"name": "t", "description": "t",
                   "entity_type": "x", "entity_label": "x"},
        "time_window": {"start": "2024-01", "end": "2024-06",
                         "granularity": "monthly"},
        "seed": 1,
        "metrics": [{
            "name": "m", "label": "M", "distribution": "lognorm",
            "params": {"s": 0.5, "scale": 1.0}, "polarity": "positive",
        }],
        "archetypes": [{
            "name": "a", "label": "A", "description": "-",
            "curve_segments": [{
                "curve": "plateau", "params": {"level": 0.5},
                "start_pct": 0.0, "end_pct": 1.0,
            }],
        }],
        "entities": [{"name": "e", "archetype": "a", "size": 3}],
        "tables": [
            {"name": "dim_date", "type": "dim", "grain": "per_period",
             "columns": [{"name": "date_key", "dtype": "id", "source": "pk"}],
             "primary_key": "date_key"},
            {"name": "dim_x", "type": "dim", "grain": "per_entity",
             "columns": [{"name": "x_id", "dtype": "id", "source": "pk"}],
             "primary_key": "x_id"},
            {"name": "fct_m", "type": "fact", "grain": "per_entity_per_period",
             "columns": [
                 {"name": "date_key", "dtype": "id",
                  "source": "fk:dim_date.date_key"},
                 {"name": "x_id", "dtype": "id",
                  "source": "fk:dim_x.x_id"},
                 {"name": "m", "dtype": "float", "source": "metric:m"},
             ],
             "primary_key": ["date_key", "x_id"],
             "foreign_keys": ["dim_date.date_key", "dim_x.x_id"]},
        ],
        # correlations deliberately omitted — default factory gives empty list.
        "output": {"format": "csv", "directory": "out"},
    }
    cfg = PlotsimConfig(**raw)
    tables = generate_tables(cfg, np.random.default_rng(0))
    assert "fct_m" in tables
    assert len(tables["fct_m"]) > 0


# --- F-01 / 0.4.0: Gaussian copula correlation recovery ---------------------
#
# R-01..R-10 from plotsim-mission-f01-correlation-fix.md. These tests MUST
# fail against the pre-0.4.0 center-normalized residual transform — they are
# the behavioral guarantee of the copula rewrite.


def _simulate_correlated_pair(
    metric_a: Metric, metric_b: Metric, coeff: float,
    n_entities: int = 100, n_periods: int = 36, seed: int = 0,
) -> float:
    """Run N entities through plateau trajectory and return observed Pearson.

    Plateau isolation: a flat trajectory means the configured coefficient is
    the only thing driving co-movement between the two metrics' output
    columns. Anything else (co-variation via shared trajectory shape, lag
    blending, archetype overrides) is held constant.
    """
    pair = CorrelationPair(metric_a=metric_a.name, metric_b=metric_b.name,
                           coefficient=coeff)
    rng = _rng(seed)
    all_a: list[float] = []
    all_b: list[float] = []
    for _ in range(n_entities):
        out = generate_entity_metrics(
            trajectory=np.full(n_periods, 0.5),
            metrics=[metric_a, metric_b],
            correlations=[pair],
            noise=None,
            rng=rng,
        )
        all_a.extend(float(x) for x in out[metric_a.name])
        all_b.extend(float(x) for x in out[metric_b.name])
    return float(np.corrcoef(np.array(all_a), np.array(all_b))[0, 1])


@pytest.mark.parametrize(
    "dist_a,params_a,dist_b,params_b,vr_b,coeff",
    [
        # Three SaaS template pairs from the mission context.
        # engagement (beta) × mrr (lognormal) at 0.72.
        ("beta", {"alpha": 2.0, "beta": 5.0}, "lognorm",
         {"s": 0.4, "scale": 1.0}, None, 0.72),
        # engagement (beta) × churn_risk (beta) at -0.55.
        ("beta", {"alpha": 2.0, "beta": 5.0}, "beta",
         {"alpha": 2.0, "beta": 5.0}, ValueRange(min=0.0, max=1.0), -0.55),
        # support_tickets (poisson) × churn_risk (beta) at 0.55 — discrete
        # pair; mission tolerance ±0.15 vs continuous ±0.08.
        ("poisson", {"lambda": 3.0}, "beta",
         {"alpha": 2.0, "beta": 5.0}, ValueRange(min=0.0, max=1.0), 0.55),
    ],
)
def test_r01_configured_correlation_matches_observed_plateau(
    dist_a, params_a, dist_b, params_b, vr_b, coeff,
):
    """R-01 / F-01: observed Pearson must be within ±0.10 (±0.15 for poisson)
    of configured under plateau isolation. The pre-0.4.0 center-normalized
    residual transform attenuated these to roughly 0.3× to 0.9× of configured
    depending on distribution pairing.
    """
    m_a = _metric("a", distribution=dist_a, params=params_a)
    m_b = _metric("b", distribution=dist_b, params=params_b,
                  value_range=vr_b)
    observed = _simulate_correlated_pair(m_a, m_b, coeff,
                                         n_entities=100, n_periods=36, seed=11)
    tol = 0.15 if "poisson" in (dist_a, dist_b) else 0.10
    assert abs(observed - coeff) <= tol, (
        f"{dist_a}×{dist_b}: expected {coeff} ± {tol}, got {observed:.4f}"
    )


@pytest.mark.parametrize(
    "dist_a,params_a,dist_b,params_b,vr_b",
    [
        ("normal", {"mu": 100.0, "sigma": 10.0}, "normal",
         {"mu": 50.0, "sigma": 5.0}, None),
        ("normal", {"mu": 100.0, "sigma": 10.0}, "lognorm",
         {"s": 0.5, "scale": 1.0}, None),
        ("lognorm", {"s": 0.5, "scale": 1.0}, "gamma",
         {"shape": 2.0, "scale": 1.0}, None),
        ("gamma", {"shape": 2.0, "scale": 1.0}, "beta",
         {"alpha": 2.0, "beta": 5.0}, ValueRange(min=0.0, max=1.0)),
        ("beta", {"alpha": 2.0, "beta": 5.0}, "weibull",
         {"shape": 1.5, "scale": 1.0}, None),
        ("weibull", {"shape": 1.5, "scale": 1.0}, "normal",
         {"mu": 20.0, "sigma": 3.0}, None),
    ],
)
def test_r01b_multiple_distribution_pairings(
    dist_a, params_a, dist_b, params_b, vr_b,
):
    """R-01b: every continuous-continuous pair recovers a configured 0.6
    correlation within ±0.10 under plateau isolation and 200 entities × 22
    periods ≈ 4,400 samples — enough for the ±0.10 envelope to exclude
    sampling noise on correctly-implemented copulas.
    """
    vr_a = ValueRange(min=0.0, max=1.0) if dist_a == "beta" else None
    m_a = _metric("a", distribution=dist_a, params=params_a, value_range=vr_a)
    m_b = _metric("b", distribution=dist_b, params=params_b, value_range=vr_b)
    observed = _simulate_correlated_pair(m_a, m_b, 0.6,
                                         n_entities=200, n_periods=22, seed=23)
    assert abs(observed - 0.6) <= 0.10, (
        f"{dist_a}×{dist_b}: expected 0.6 ± 0.10, got {observed:.4f}"
    )


def test_r08_lognormal_stays_positive_under_extreme_correlation():
    """R-08: the copula's inverse-CDF round trip lands every correlated draw
    inside the metric's support. For lognormal that means strictly positive —
    the pre-0.4.0 ``c * (1 + corr_r)`` path could go negative when
    ``corr_r < -1``, which happened under high variance + strong correlation.
    """
    m_lognorm = _metric("a", distribution="lognorm",
                        params={"s": 1.5, "scale": 1.0})
    m_high_var = _metric("b", distribution="normal",
                         params={"mu": 100.0, "sigma": 80.0})
    pair = CorrelationPair(metric_a="a", metric_b="b", coefficient=0.9)
    rng = _rng(7)
    all_a: list[float] = []
    for _ in range(500):
        out = generate_entity_metrics(
            trajectory=np.full(8, 0.5),
            metrics=[m_lognorm, m_high_var],
            correlations=[pair],
            noise=None,
            rng=rng,
        )
        all_a.extend(float(x) for x in out["a"])
    arr = np.array(all_a)
    assert (arr > 0.0).all(), (
        f"lognormal produced {int((arr <= 0.0).sum())} non-positive values "
        f"under the copula — the inverse-CDF round trip is broken"
    )


def test_r10_determinism_under_copula():
    """R-10: same config + same seed still produces byte-identical output
    after the copula rewrite. The determinism guarantee is preserved; only
    the output *values* change relative to pre-0.4.0.
    """
    from plotsim.tables import generate_tables
    import pandas as pd
    cfg = load_config(SAAS_YAML)
    a = generate_tables(cfg, np.random.default_rng(cfg.seed))
    b = generate_tables(cfg, np.random.default_rng(cfg.seed))
    assert set(a.keys()) == set(b.keys())
    for name in a:
        pd.testing.assert_frame_equal(
            a[name].reset_index(drop=True),
            b[name].reset_index(drop=True),
        )
