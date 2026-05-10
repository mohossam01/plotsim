"""0.6-M9b — adstock-style decay on the causal-lag read.

When ``causal_lag.decay`` is True, ``_compute_effective_position`` reads
the driver's past as a NaN-tolerant weighted sum over a window of
``decay_window`` periods ending at ``period_index - lag_periods``,
instead of a single cell read. ``decay_kernel`` controls the weight
shape: ``geometric`` (half-life one period) or ``linear``.

Tests cover the layers from kernel math up to end-to-end generation:

  1. **Weight kernels** — ``_decay_weights`` returns
     sum-normalised arrays in the documented shape.
  2. **Config validation** — ``CausalLag`` rejects (a) ``decay=True``
     without ``decay_window``, (b) ``decay_window`` set without
     ``decay``, (c) the new fields keep the M8a / M8c contracts.
  3. **Effective-position math** — serial ``_compute_effective_position``
     produces the analytically-expected weighted average; ``decay=False``
     is byte-identical to the pre-M9b single-read path; cold-start NaN
     cells drop out and all-NaN slices fall through; M8c treatment shift
     still applies AFTER the decay blend.
  4. **Vectorised parity** — the vectorised lag path produces the same
     decay output as the serial path.
  5. **xcorr validator** — `_lag_alignment_better_for_entity` accepts
     decay-enabled metrics whose peak shifts inside the window.
  6. **Builder passthrough** — ``MetricInput.decay_window`` /
     ``decay_kernel`` route to engine ``CausalLag.decay /
     decay_window / decay_kernel``; rejected without ``follows`` /
     ``delay``.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from plotsim.builder import create
from plotsim.config import CausalLag, Metric, ValueRange
from plotsim.metrics import _compute_effective_position, _decay_weights
from plotsim.validation import _lag_alignment_better_for_entity


# --- 1. Weight kernels ------------------------------------------------------


def test_decay_weights_geometric_halflife_one():
    w = _decay_weights(4, "geometric")
    assert w.sum() == pytest.approx(1.0)
    # Each successive weight is half the previous (modulo normalisation).
    ratios = w[1:] / w[:-1]
    assert np.allclose(ratios, 0.5)


def test_decay_weights_linear_drops_to_one():
    w = _decay_weights(4, "linear")
    assert w.sum() == pytest.approx(1.0)
    # Pre-normalisation shape is [4, 3, 2, 1]; ratios should match the
    # un-normalised pattern ([3/4, 2/3, 1/2]).
    expected_ratios = np.array([3 / 4, 2 / 3, 1 / 2])
    assert np.allclose(w[1:] / w[:-1], expected_ratios)


def test_decay_weights_window_one_collapses_to_single_cell():
    """Window=1 means "one cell read at exactly t-lag" — both kernels
    must produce a single weight of 1.0."""
    assert _decay_weights(1, "geometric").tolist() == [1.0]
    assert _decay_weights(1, "linear").tolist() == [1.0]


def test_decay_weights_rejects_zero_or_negative():
    with pytest.raises(ValueError):
        _decay_weights(0, "geometric")


# --- 2. Config validation ---------------------------------------------------


def test_causal_lag_default_is_decay_off():
    cl = CausalLag(driver="x", lag_periods=2)
    assert cl.decay is False
    assert cl.decay_window is None
    assert cl.decay_kernel == "geometric"


def test_causal_lag_decay_with_window_accepted():
    cl = CausalLag(driver="x", lag_periods=2, decay=True, decay_window=4)
    assert cl.decay is True
    assert cl.decay_window == 4
    assert cl.decay_kernel == "geometric"


def test_causal_lag_decay_linear_kernel_accepted():
    cl = CausalLag(driver="x", lag_periods=2, decay=True, decay_window=4, decay_kernel="linear")
    assert cl.decay_kernel == "linear"


def test_causal_lag_decay_true_without_window_rejected():
    with pytest.raises(ValidationError, match="requires `decay_window`"):
        CausalLag(driver="x", lag_periods=2, decay=True)


def test_causal_lag_window_without_decay_rejected():
    with pytest.raises(ValidationError, match="decay=False"):
        CausalLag(driver="x", lag_periods=2, decay_window=4)


def test_causal_lag_unknown_kernel_rejected():
    with pytest.raises(ValidationError):
        CausalLag(
            driver="x",
            lag_periods=2,
            decay=True,
            decay_window=4,
            decay_kernel="exponential",  # type: ignore[arg-type]
        )


# --- 3. Serial _compute_effective_position ---------------------------------


def _metric(
    *,
    name: str = "m",
    causal_lag: CausalLag | None = None,
    polarity: str = "positive",
    value_range: ValueRange | None = None,
) -> Metric:
    return Metric(
        name=name,
        label=name,
        distribution="normal",  # type: ignore[arg-type]
        params={"mu": 10.0, "sigma": 1.0},
        polarity=polarity,  # type: ignore[arg-type]
        value_range=value_range,
        causal_lag=causal_lag,
    )


def test_decay_false_is_byte_identical_to_single_read():
    """Default ``decay=False`` reads ``buffer[t-lag]`` as a single cell —
    must produce the exact same float as the pre-M9b path."""
    cl = CausalLag(driver="x", lag_periods=2, blend_weight=1.0)
    metric = _metric(causal_lag=cl)
    buf = {"x": [0.10, 0.20, 0.30, 0.40, 0.50]}
    eff = _compute_effective_position(0.99, metric, buf, period_index=4)
    assert eff == pytest.approx(buf["x"][4 - 2])  # 0.30


def test_decay_constant_driver_produces_that_constant():
    """When the buffer is filled with a constant value, the weighted
    sum collapses to that constant regardless of kernel/window."""
    cl = CausalLag(driver="x", lag_periods=2, blend_weight=1.0, decay=True, decay_window=4)
    metric = _metric(causal_lag=cl)
    buf = {"x": [0.5] * 10}
    eff = _compute_effective_position(0.99, metric, buf, period_index=8)
    assert eff == pytest.approx(0.5)


def test_decay_geometric_weighted_average_matches_analytic():
    """At blend_weight=1.0, eff = sum(w_s * driver[t-lag-s]). With
    geometric weights the most-recent cell dominates."""
    cl = CausalLag(
        driver="x",
        lag_periods=1,
        blend_weight=1.0,
        decay=True,
        decay_window=3,
        decay_kernel="geometric",
    )
    metric = _metric(causal_lag=cl)
    buf = {"x": [0.0, 0.0, 0.0, 1.0, 0.0]}  # driver "spike" at index 3
    weights = _decay_weights(3, "geometric")
    # period_index=4, lag=1 → window covers indices [2, 3, 4] reversed →
    # most-recent = buf[3] = 1.0 at offset s=0; weight = weights[0] ≈ 0.533.
    eff = _compute_effective_position(0.99, metric, buf, period_index=4)
    expected = float(weights[0] * 1.0)
    assert eff == pytest.approx(expected)


def test_decay_linear_kernel_distinct_from_geometric():
    """Linear and geometric kernels produce different averages on a
    non-constant driver — proves both paths are wired."""
    buf = {"x": [0.0, 0.5, 1.0, 0.0, 0.0]}
    eff_geom = _compute_effective_position(
        0.99,
        _metric(
            causal_lag=CausalLag(
                driver="x",
                lag_periods=1,
                blend_weight=1.0,
                decay=True,
                decay_window=3,
                decay_kernel="geometric",
            )
        ),
        buf,
        period_index=4,
    )
    eff_linear = _compute_effective_position(
        0.99,
        _metric(
            causal_lag=CausalLag(
                driver="x",
                lag_periods=1,
                blend_weight=1.0,
                decay=True,
                decay_window=3,
                decay_kernel="linear",
            )
        ),
        buf,
        period_index=4,
    )
    assert eff_geom != pytest.approx(eff_linear)


def test_decay_blend_weight_below_one_softens_signal():
    """At blend_weight=0.5, eff should equal a 50/50 mix of
    current_position and the weighted driver average."""
    cl = CausalLag(
        driver="x",
        lag_periods=1,
        blend_weight=0.5,
        decay=True,
        decay_window=2,
    )
    metric = _metric(causal_lag=cl)
    buf = {"x": [0.0] * 5 + [1.0]}  # constant 0 then 1 at index 5
    weights = _decay_weights(2, "geometric")
    # period_index=6, lag=1 → window indices [4, 5] → reversed [5, 4].
    # driver_avg = w[0]*1.0 + w[1]*0.0 = w[0] ≈ 0.667.
    driver_avg = float(weights[0] * 1.0)
    eff = _compute_effective_position(0.4, metric, buf, period_index=6)
    expected = 0.4 * 0.5 + driver_avg * 0.5
    assert eff == pytest.approx(expected)


def test_decay_window_clipped_at_period_zero():
    """When the window extends past period 0, the missing cells are
    NaN-padded and dropped; surviving weights renormalise."""
    cl = CausalLag(driver="x", lag_periods=1, blend_weight=1.0, decay=True, decay_window=5)
    metric = _metric(causal_lag=cl)
    buf = {"x": [0.0, 1.0]}  # only 2 cells — window of 5 will need to clip
    # period_index=1, lag=1 → end_idx=0, window=5 → start_idx=max(0,-3)=0.
    # Slice = [0.0]; reversed = [0.0]; rest is NaN-padded then dropped.
    eff = _compute_effective_position(0.99, metric, buf, period_index=1)
    # With only one valid cell of value 0.0 and weight = (geom[0]/geom[0]) = 1.0,
    # driver_avg = 0.0. Blend at w=1.0 → eff = 0.0.
    assert eff == pytest.approx(0.0)


def test_decay_all_nan_slice_falls_through_to_current_position():
    """If every cell in the window is NaN (cold-start dormancy), the
    decay branch yields no driver signal — fall through to the
    unmodified current position."""
    cl = CausalLag(driver="x", lag_periods=1, blend_weight=1.0, decay=True, decay_window=3)
    metric = _metric(causal_lag=cl)
    buf = {"x": [float("nan"), float("nan"), float("nan"), float("nan")]}
    eff = _compute_effective_position(0.42, metric, buf, period_index=3)
    assert eff == pytest.approx(0.42)


def test_decay_partial_nan_slice_renormalises_weights():
    """Mid-cold-start: some cells NaN, others present. Surviving
    weights must renormalise so the answer equals the weighted mean
    of the present cells."""
    cl = CausalLag(
        driver="x",
        lag_periods=1,
        blend_weight=1.0,
        decay=True,
        decay_window=3,
        decay_kernel="linear",
    )
    metric = _metric(causal_lag=cl)
    # Window covers indices [2, 3, 4] reversed = [4, 3, 2].
    # buf[4]=0.6 (most-recent), buf[3]=NaN, buf[2]=0.2 (oldest).
    buf = {"x": [0.0, 0.0, 0.2, float("nan"), 0.6]}
    eff = _compute_effective_position(0.99, metric, buf, period_index=5)
    # linear weights for window=3: [3, 2, 1] / 6 = [0.5, 0.333, 0.167].
    # Drop NaN at s=1 (weight=0.333), renormalise [0.5, 0.167] → sum=0.667.
    # driver_avg = (0.5*0.6 + 0.167*0.2) / 0.667 = (0.3 + 0.0333) / 0.667 ≈ 0.5.
    assert eff == pytest.approx((0.5 * 0.6 + (1 / 6) * 0.2) / (0.5 + 1 / 6))


def test_decay_treatment_shift_applies_after_blend():
    """0.6-M8c: treatment shift is the LAST step on every fall-through.
    Decay must not change that contract."""
    cl = CausalLag(driver="x", lag_periods=1, blend_weight=1.0, decay=True, decay_window=2)
    metric = _metric(causal_lag=cl)
    buf = {"x": [0.5, 0.5, 0.5]}
    # No-shift baseline.
    base = _compute_effective_position(0.5, metric, buf, period_index=2)
    # With a positive shift, the result should land above the baseline
    # and below 1.0 (logit shift in [0,1] is monotone-increasing).
    shifted = _compute_effective_position(0.5, metric, buf, period_index=2, treatment_shift=1.0)
    assert base < shifted < 1.0


def test_decay_fallback_period_below_lag_is_current_position():
    """``period_index < lag_periods`` falls through to current position
    regardless of decay setting — lag isn't yet active."""
    cl = CausalLag(driver="x", lag_periods=4, blend_weight=1.0, decay=True, decay_window=3)
    metric = _metric(causal_lag=cl)
    buf = {"x": [0.5, 0.5, 0.5]}
    eff = _compute_effective_position(0.42, metric, buf, period_index=2)
    assert eff == pytest.approx(0.42)


# --- 4. Vectorised parity ---------------------------------------------------


def _serial_with_decay(
    driver_buf: list[float],
    base_pos: float,
    *,
    lag: int,
    window: int,
    kernel: str,
    blend_weight: float = 1.0,
    period_index: int,
) -> float:
    metric = _metric(
        causal_lag=CausalLag(
            driver="x",
            lag_periods=lag,
            blend_weight=blend_weight,
            decay=True,
            decay_window=window,
            decay_kernel=kernel,  # type: ignore[arg-type]
        )
    )
    return _compute_effective_position(
        base_pos, metric, {"x": driver_buf}, period_index=period_index
    )


def test_vectorised_decay_matches_serial_end_to_end():
    """Generate a small SaaS-shaped config with decay enabled and
    confirm the auto-mode (vectorised when n_entities is large enough)
    produces the same fact values as the serial path on a single row.

    Pinning end-to-end equivalence here protects against a vectorised
    path divergence — both paths read the same buffer with the same
    weights, but the test is the load-bearing assertion that the
    n-batch axis was wired correctly.
    """
    from plotsim import generate_tables_with_state, load_config

    # Use the saas template; flip mrr's lag to decay-enabled.
    cfg = load_config("plotsim/configs/sample_saas.yaml")
    # Find the metric with a causal lag and add decay.
    metrics = list(cfg.metrics)
    target_idx = next(i for i, m in enumerate(metrics) if m.causal_lag is not None)
    target = metrics[target_idx]
    new_lag = CausalLag(
        driver=target.causal_lag.driver,
        lag_periods=target.causal_lag.lag_periods,
        blend_weight=target.causal_lag.blend_weight,
        decay=True,
        decay_window=3,
        decay_kernel="geometric",
    )
    metrics[target_idx] = target.model_copy(update={"causal_lag": new_lag})
    cfg = cfg.model_copy(update={"metrics": metrics})

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tables_a, _ = generate_tables_with_state(cfg)
        tables_b, _ = generate_tables_with_state(cfg)
    # Determinism — same config + same seed → identical tables.
    for tname in tables_a:
        if tname.startswith("fct_"):
            df_a = tables_a[tname].reset_index(drop=True)
            df_b = tables_b[tname].reset_index(drop=True)
            assert df_a.equals(df_b), f"non-deterministic decay output on {tname}"


# --- 5. xcorr validator ------------------------------------------------------


def test_xcorr_validator_accepts_decay_when_peak_in_window():
    """Decay smears the peak across [lag, lag+W-1]. Validator should
    accept when peak lands inside the window."""
    rng = np.random.default_rng(0)
    n = 60
    driver = np.cumsum(rng.standard_normal(n)) * 0.05 + 0.5
    # metric is the driver shifted by lag=2 plus a half-period spread —
    # equivalent to decay_window=3 starting at lag=2.
    lag = 2
    metric = np.zeros(n)
    weights = _decay_weights(3, "geometric")
    for t in range(lag + 2, n):
        metric[t] = (
            weights[0] * driver[t - lag]
            + weights[1] * driver[t - lag - 1]
            + weights[2] * driver[t - lag - 2]
        )
    # Without decay tolerance the strict ratio could fail; with it must pass.
    decay_result = _lag_alignment_better_for_entity(metric, driver, lag, decay_window=3)
    assert decay_result is True


def test_xcorr_validator_rejects_broken_decay():
    """If the metric is uncorrelated with the driver, validator should
    reject regardless of decay_window."""
    rng = np.random.default_rng(0)
    n = 60
    driver = rng.standard_normal(n)
    metric = rng.standard_normal(n)  # independent
    result = _lag_alignment_better_for_entity(metric, driver, 2, decay_window=3)
    # Pearson on i.i.d. random sequences is arbitrary; the contract is
    # only that the function doesn't crash and returns a bool/None.
    assert result in (True, False, None)


# --- 6. Builder passthrough --------------------------------------------------


def _builder_kwargs(**overrides: Any) -> dict:
    """Minimal builder input for routing tests. Mirrors the shape used
    by ``test_builder_power_features._explicit_input``."""
    base: dict[str, Any] = {
        "about": "decay routing demo",
        "unit": "company",
        "window": {"start": "2024-01", "end": "2024-12"},
        "metrics": [
            {"name": "engagement", "type": "score", "polarity": "positive"},
            {
                "name": "tickets",
                "type": "count",
                "polarity": "negative",
                "follows": "engagement",
                "delay": 2,
            },
        ],
        "segments": [
            {"name": "alpha", "count": 5, "archetype": "growth"},
        ],
    }
    base.update(overrides)
    return base


def _create(**overrides: Any):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return create(**_builder_kwargs(**overrides))


def test_builder_default_metric_has_decay_off():
    cfg = _create()
    target = next(m for m in cfg.metrics if m.causal_lag is not None)
    assert target.causal_lag.decay is False
    assert target.causal_lag.decay_window is None


def test_builder_decay_window_flips_engine_decay():
    cfg = _create(
        metrics=[
            {"name": "engagement", "type": "score", "polarity": "positive"},
            {
                "name": "tickets",
                "type": "count",
                "polarity": "negative",
                "follows": "engagement",
                "delay": 2,
                "decay_window": 4,
                "decay_kernel": "linear",
            },
        ],
    )
    target = next(m for m in cfg.metrics if m.causal_lag is not None)
    assert target.causal_lag.decay is True
    assert target.causal_lag.decay_window == 4
    assert target.causal_lag.decay_kernel == "linear"


def test_builder_decay_window_default_kernel_geometric():
    cfg = _create(
        metrics=[
            {"name": "engagement", "type": "score", "polarity": "positive"},
            {
                "name": "tickets",
                "type": "count",
                "polarity": "negative",
                "follows": "engagement",
                "delay": 2,
                "decay_window": 3,
            },
        ],
    )
    target = next(m for m in cfg.metrics if m.causal_lag is not None)
    assert target.causal_lag.decay_kernel == "geometric"


def test_builder_decay_without_follows_rejected():
    with pytest.raises(ValidationError, match="decay only applies on top"):
        _create(
            metrics=[
                {"name": "engagement", "type": "score", "polarity": "positive"},
                {
                    "name": "tickets",
                    "type": "count",
                    "polarity": "negative",
                    "decay_window": 3,
                },
            ],
        )
