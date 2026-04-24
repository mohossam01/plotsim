"""plotsim.metrics — trajectory positions → metric values.

What it does:
    Turns a trajectory position in [0, 1] into an actual numeric metric value.
    For each entity at each period, all metrics read from the SAME trajectory
    position (the trajectory-first invariant), which is then mapped per-metric
    to a distribution center, sampled, correlated, noised, clamped, and
    optionally nulled for MCAR.

    This is where randomness enters the system. Every sample flows through a
    caller-supplied ``numpy.random.Generator``, so identical seed + identical
    inputs → identical output.

Input:
    - Trajectory position (scalar) or full trajectory array (ndarray, [0,1]).
    - list[Metric] — distribution, params, polarity, value_range, causal_lag.
    - list[CorrelationPair] | None — pairwise correlation coefficients.
    - NoiseConfig | None — gaussian σ, outlier rate, MCAR rate.
    - np.random.Generator — seeded by caller.

Output:
    - generate_metrics_for_period → dict[metric_name → float | int | None].
    - generate_entity_metrics     → dict[metric_name → np.ndarray].

Mission-spec deviations (flagged in M004 completion report):
    1. ``generate_metrics_for_period`` takes an explicit ``noise`` param; the
       spec signature omitted it, but step 5 (apply noise) needs access.
    2. ``lag_buffer`` holds trajectory positions, not generated values. The
       blend formula ``current*(1-w) + driver_past*w`` is only dimensionally
       coherent on [0,1] positions — blending a raw MRR value against a
       position is not meaningful.
    3. For beta, the spec's ``rng.beta(α,β)*scale`` ignores center, which
       breaks the "higher position → higher mean" acceptance criterion. We
       shift-to-center instead: the beta shape's variance is preserved but
       its expected value lands on ``center``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from plotsim.config import (
    Archetype,
    CorrelationPair,
    Metric,
    NoiseConfig,
)


# 60% driver-past + 40% current. V1 constant — future work may expose this
# per-metric. Chosen so lag has a visible effect in tests without overwhelming
# the current-period signal entirely.
LAG_BLEND_WEIGHT = 0.6

# Below this absolute value, a center is treated as zero for correlation/
# residual purposes. Avoids amplifying floating-point noise by 1/epsilon
# during Cholesky reconstruction.
_CENTER_EPS = 1e-9


# --- Polarity and position → center ------------------------------------------

def _apply_polarity(position: float, metric: Metric) -> float:
    """Flip position if metric has negative polarity."""
    if metric.polarity == "negative":
        return 1.0 - position
    return position


def position_to_center(position: float, metric: Metric) -> float:
    """Map a trajectory position in [0,1] to a distribution center.

    Polarity is applied first (negative → 1-position), then each distribution
    uses its own location/scale params. For beta, which is intrinsically
    [0,1], the center is rescaled to ``value_range`` span when one is set so
    the sampler can land values inside the configured range.
    """
    p = _apply_polarity(position, metric)
    dist = metric.distribution
    params = metric.params

    if dist == "lognorm":
        loc = params.get("loc", 0.0)
        scale = params.get("scale", 1.0)
        return loc + scale * p
    if dist == "gamma":
        shape = params["shape"]
        scale = params.get("scale", 1.0)
        return shape * scale * p
    if dist == "poisson":
        lam = params.get("lambda", 1.0)
        return lam * p
    if dist == "beta":
        vr = metric.value_range
        if vr is not None and vr.min is not None and vr.max is not None:
            return vr.min + p * (vr.max - vr.min)
        return p
    if dist == "normal":
        mu = params.get("mu", 0.0)
        return mu * p
    if dist == "weibull":
        scale = params.get("scale", 1.0)
        return scale * p
    raise ValueError(f"unsupported distribution {dist!r}")


# --- Sampling ----------------------------------------------------------------

def sample_single_metric(
    center: float,
    metric: Metric,
    rng: np.random.Generator,
) -> float:
    """Draw one sample from the metric's distribution, centered on ``center``.

    Value-range clamping and poisson integer-rounding happen in the caller
    (``_clamp_and_round``), AFTER noise injection, so this function returns
    the raw distributional draw.
    """
    dist = metric.distribution
    params = metric.params

    if dist == "lognorm":
        s = params["s"]
        safe_center = max(center, _CENTER_EPS)
        return float(rng.lognormal(mean=float(np.log(safe_center)), sigma=s))

    if dist == "gamma":
        shape = params["shape"]
        if shape <= 0.0 or center <= 0.0:
            return 0.0
        return float(rng.gamma(shape=shape, scale=center / shape))

    if dist == "poisson":
        lam = max(center, 0.0)
        return float(rng.poisson(lam=lam))

    if dist == "beta":
        alpha = params["alpha"]
        beta = params["beta"]
        raw = float(rng.beta(a=alpha, b=beta))
        base_mean = alpha / (alpha + beta)
        vr = metric.value_range
        if vr is not None and vr.min is not None and vr.max is not None:
            span = vr.max - vr.min
            return center + (raw - base_mean) * span
        scale = params.get("scale", 1.0)
        return (raw - base_mean + center) * scale

    if dist == "normal":
        sigma = params["sigma"]
        return float(rng.normal(loc=center, scale=sigma))

    if dist == "weibull":
        shape = params["shape"]
        return float(rng.weibull(a=shape)) * center

    raise ValueError(f"unsupported distribution {dist!r}")


def _clamp_and_round(value: float, metric: Metric) -> float:
    vr = metric.value_range
    if vr is not None:
        if vr.min is not None and value < vr.min:
            value = vr.min
        if vr.max is not None and value > vr.max:
            value = vr.max
    if metric.distribution == "poisson":
        return float(int(round(value)))
    return value


# --- Correlated noise via Cholesky ------------------------------------------

def apply_correlations(
    independent: dict[str, float],
    centers: dict[str, float],
    correlations: list[CorrelationPair],
    metrics: list[Metric],
) -> dict[str, float]:
    """Adjust independent samples so pairwise correlations match config.

    Residuals are normalized by center so metrics on different scales are
    comparable, then transformed through the Cholesky factor of the target
    correlation matrix and reconstructed as ``c * (1 + corr_r)``. When the
    matrix is non-PSD (bad user config), we fall back to independent samples
    rather than crash.

    Metrics whose center is near zero bypass the transform — their residual
    would be undefined and the reconstruction would collapse to 0.
    """
    if not correlations:
        return dict(independent)

    names = [m.name for m in metrics]
    idx = {n: i for i, n in enumerate(names)}
    n = len(names)

    r = np.zeros(n, dtype=float)
    bypass = [False] * n
    for i, name in enumerate(names):
        c = centers[name]
        v = independent[name]
        if v is None or abs(c) < _CENTER_EPS:
            bypass[i] = True
            r[i] = 0.0
        else:
            r[i] = (v - c) / c

    mat = np.eye(n)
    for pair in correlations:
        if pair.metric_a in idx and pair.metric_b in idx:
            i = idx[pair.metric_a]
            j = idx[pair.metric_b]
            mat[i, j] = pair.coefficient
            mat[j, i] = pair.coefficient

    try:
        L = np.linalg.cholesky(mat)
    except np.linalg.LinAlgError:
        return dict(independent)

    corr_r = L @ r

    out = dict(independent)
    for i, name in enumerate(names):
        if bypass[i]:
            continue
        c = centers[name]
        out[name] = c * (1.0 + corr_r[i])
    return out


# --- Causal lag --------------------------------------------------------------

def _compute_effective_position(
    current_position: float,
    metric: Metric,
    lag_buffer: Optional[dict[str, list[float]]],
    period_index: int,
) -> float:
    """Blend the current trajectory position with the driver's past position.

    Operates on pre-polarity positions in [0,1] so both operands share the
    same semantic axis ("how well is this entity doing"). The metric's own
    polarity is applied afterwards in ``position_to_center``.

    Falls back to the unmodified current position when: no causal_lag is
    configured, insufficient history exists, or the driver isn't in the
    buffer.
    """
    if metric.causal_lag is None:
        return current_position
    lag = metric.causal_lag.lag_periods
    if period_index < lag:
        return current_position
    if lag_buffer is None:
        return current_position
    driver = metric.causal_lag.driver
    driver_history = lag_buffer.get(driver)
    if driver_history is None or len(driver_history) < period_index - lag + 1:
        return current_position
    driver_past = driver_history[period_index - lag]
    w = LAG_BLEND_WEIGHT
    return current_position * (1.0 - w) + driver_past * w


# --- Noise -------------------------------------------------------------------

def apply_noise(
    value: float,
    noise: NoiseConfig,
    rng: np.random.Generator,
) -> Optional[float]:
    """Inject gaussian jitter, outliers, and MCAR nulls.

    Ordering: gaussian → outlier → MCAR. An outlier can still be nullified by
    MCAR on the same step. Each branch skips its rng call when the governing
    rate/sigma is 0, so a fully zero NoiseConfig is a pure pass-through and
    consumes no randomness.

    value_range clamping happens in the caller, AFTER noise, so outliers that
    escape the range get clipped appropriately.
    """
    v = value
    if noise.gaussian_sigma > 0.0:
        # Multiplicative jitter. For |v|=0 we fall back to absolute sigma so
        # a metric that legitimately sits at 0 still receives some noise.
        mag = abs(v) if v != 0.0 else 1.0
        v = v + float(rng.normal(loc=0.0, scale=noise.gaussian_sigma * mag))

    if noise.outlier_rate > 0.0 and rng.random() < noise.outlier_rate:
        sign = 1.0 if v >= 0.0 else -1.0
        mag = abs(v) if v != 0.0 else 1.0
        v = sign * float(rng.uniform(mag * 3.0, mag * 10.0))

    if noise.mcar_rate > 0.0 and rng.random() < noise.mcar_rate:
        return None

    return v


# --- Per-period and per-entity orchestration --------------------------------

def _apply_archetype_overrides(
    metric: Metric, archetype: Optional[Archetype],
) -> Metric:
    """Return `metric` with distribution/params substituted when the archetype
    declares an override for it. Polarity, value_range, and causal_lag are
    never overridable — only the distribution family and its shape params.
    """
    if archetype is None:
        return metric
    override = archetype.metric_overrides.get(metric.name)
    if override is None:
        return metric
    updates: dict = {}
    if override.distribution is not None:
        updates["distribution"] = override.distribution
    if override.params is not None:
        updates["params"] = override.params
    return metric.model_copy(update=updates) if updates else metric


def generate_metrics_for_period(
    trajectory_position: float,
    metrics: list[Metric],
    correlations: Optional[list[CorrelationPair]],
    noise: Optional[NoiseConfig],
    lag_buffer: Optional[dict[str, list[float]]],
    period_index: int,
    rng: np.random.Generator,
    archetype: Optional[Archetype] = None,
) -> dict[str, Optional[float]]:
    """Generate every metric for one entity at one time step.

    Pipeline per metric:
        1. resolve archetype override (distribution/params) if any
        2. current position → optional lag blend with driver's past position
        3. (polarity + distribution-specific) position → center
        4. sample independent value from the distribution
    Then once across all metrics:
        5. apply Cholesky correlation on residuals (if correlations given)
        6. apply noise (if noise config given): gaussian → outlier → MCAR
        7. clamp to value_range, round poisson to int
    """
    effective = [_apply_archetype_overrides(m, archetype) for m in metrics]
    centers: dict[str, float] = {}
    independent: dict[str, Optional[float]] = {}

    for em in effective:
        eff_pos = _compute_effective_position(
            trajectory_position, em, lag_buffer, period_index,
        )
        center = position_to_center(eff_pos, em)
        centers[em.name] = center
        independent[em.name] = sample_single_metric(center, em, rng)

    if correlations:
        correlated = apply_correlations(independent, centers, correlations, effective)
    else:
        correlated = dict(independent)

    out: dict[str, Optional[float]] = {}
    for em in effective:
        v = correlated[em.name]
        if v is None:
            out[em.name] = None
            continue
        if noise is not None:
            maybe_v = apply_noise(v, noise, rng)
            if maybe_v is None:
                out[em.name] = None
                continue
            v = maybe_v
        out[em.name] = _clamp_and_round(v, em)
    return out


def generate_entity_metrics(
    trajectory: np.ndarray,
    metrics: list[Metric],
    correlations: Optional[list[CorrelationPair]],
    noise: Optional[NoiseConfig],
    rng: np.random.Generator,
    archetype: Optional[Archetype] = None,
) -> dict[str, np.ndarray]:
    """Generate every metric's full time series for one entity.

    Walks forward through the trajectory, feeding the trajectory position at
    each step into a fresh lag buffer so later periods can look backwards
    through ``causal_lag``. The buffer is local to this call — generating
    another entity starts with an empty buffer, so lag history cannot leak
    across entities.

    When ``archetype`` is provided, any ``metric_overrides`` it declares are
    applied per period (distribution family and shape params only).

    Return arrays are ``int`` for poisson, ``float`` otherwise. Whether a
    metric's series is poisson is decided on the (possibly overridden)
    effective distribution, so an override from ``poisson`` → ``normal``
    (or vice versa) correctly changes the output dtype. If any MCAR null
    appears in a metric's series, that array becomes ``dtype=object`` to
    carry ``None`` alongside numbers.
    """
    effective = [_apply_archetype_overrides(m, archetype) for m in metrics]
    n_periods = len(trajectory)
    lag_buffer: dict[str, list[float]] = {m.name: [] for m in metrics}
    collected: dict[str, list] = {m.name: [] for m in metrics}

    for t in range(n_periods):
        pos = float(trajectory[t])
        period_out = generate_metrics_for_period(
            pos, metrics, correlations, noise, lag_buffer, t, rng,
            archetype=archetype,
        )
        for m in metrics:
            collected[m.name].append(period_out[m.name])
            lag_buffer[m.name].append(pos)

    result: dict[str, np.ndarray] = {}
    for em in effective:
        values = collected[em.name]
        if any(v is None for v in values):
            result[em.name] = np.array(values, dtype=object)
        elif em.distribution == "poisson":
            result[em.name] = np.array(values, dtype=int)
        else:
            result[em.name] = np.array(values, dtype=float)
    return result
