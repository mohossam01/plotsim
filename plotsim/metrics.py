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
    2. ``lag_buffer`` holds effective positions (trajectory position
       optionally blended with the driver's past effective position via
       ``CausalLag.blend_weight``). Effective positions stay in [0,1] so
       the blend formula remains dimensionally coherent. Populated inside
       ``generate_metrics_for_period`` in topological order (drivers
       before targets), which makes multi-hop chains A→B→C compose
       truthfully. 0.4.0 behavior; pre-0.4.0 stored raw trajectory
       positions.
    3. For beta, the spec's ``rng.beta(α,β)*scale`` ignores center, which
       breaks the "higher position → higher mean" acceptance criterion. We
       shift-to-center instead: the beta shape's variance is preserved but
       its expected value lands on ``center``.
"""

from __future__ import annotations

import warnings
from graphlib import CycleError, TopologicalSorter
from typing import Optional

import numpy as np
from scipy import stats as sp_stats
from scipy.stats import norm as sp_norm

from plotsim.config import (
    Archetype,
    CorrelationPair,
    Metric,
    NoiseConfig,
)


# Below this absolute value, a center is treated as zero for correlation
# purposes. A degenerate distribution (poisson λ≈0, lognorm scale≈0) has no
# meaningful CDF transform; the independent draw is preserved unchanged.
_CENTER_EPS = 1e-9

# F-01 / 0.4.0: clamp uniform values before norm.ppf. The exact 0.0 / 1.0
# endpoints of a CDF map to ±inf under the inverse normal, which propagates
# NaN through the Cholesky and back. 1e-10 is tight enough that the clipped
# Gaussian value stays above |z| ≈ 6.36 — well beyond any configured
# correlation's effective range — without introducing visible bias.
_CDF_CLAMP = 1e-10


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


def _get_scipy_dist(metric: Metric, center: float):
    """Return a scipy.stats frozen distribution matching ``sample_single_metric``.

    F-01 / 0.4.0. The copula transform in ``apply_correlations`` needs the
    CDF/PPF of whatever distribution ``sample_single_metric`` drew from, with
    the same parameters centered on the trajectory-derived ``center``. Each
    branch below mirrors one in ``sample_single_metric`` — if that function
    grows a new distribution family, this dispatch has to grow too.

    Returns ``None`` when the distribution is degenerate (e.g. gamma with
    ``shape <= 0`` or ``center <= 0``, lognorm with ``center`` below the
    sample floor). Callers treat ``None`` as a bypass: the independent draw
    passes through unchanged, same as the pre-0.4.0 near-zero-center branch.
    """
    dist = metric.distribution
    params = metric.params

    if dist == "lognorm":
        s = float(params["s"])
        if center <= _CENTER_EPS:
            return None
        return sp_stats.lognorm(s=s, scale=float(center))

    if dist == "gamma":
        shape = float(params["shape"])
        if shape <= 0.0 or center <= 0.0:
            return None
        return sp_stats.gamma(a=shape, scale=float(center) / shape)

    if dist == "poisson":
        lam = max(float(center), 0.0)
        if lam <= _CENTER_EPS:
            return None
        return sp_stats.poisson(mu=lam)

    if dist == "beta":
        alpha = float(params["alpha"])
        beta = float(params["beta"])
        base_mean = alpha / (alpha + beta)
        vr = metric.value_range
        # ``sample_single_metric`` produces ``c + (raw - base_mean) * span``
        # when value_range is set, else ``(raw - base_mean + c) * scale``.
        # Both are affine re-parameterizations of Beta(a, b); expressing
        # them via scipy's (loc, scale) gives an exact CDF/PPF match.
        if vr is not None and vr.min is not None and vr.max is not None:
            span = float(vr.max - vr.min)
            return sp_stats.beta(
                a=alpha, b=beta,
                loc=float(center) - base_mean * span,
                scale=span,
            )
        scale = float(params.get("scale", 1.0))
        return sp_stats.beta(
            a=alpha, b=beta,
            loc=scale * (float(center) - base_mean),
            scale=scale,
        )

    if dist == "normal":
        sigma = float(params["sigma"])
        if sigma <= 0.0:
            return None
        return sp_stats.norm(loc=float(center), scale=sigma)

    if dist == "weibull":
        shape = float(params["shape"])
        if shape <= 0.0 or center <= 0.0:
            return None
        # ``rng.weibull(a=shape)`` is standard-Weibull; the sampler then
        # multiplies by ``center``. In scipy, that's scale=center.
        return sp_stats.weibull_min(c=shape, scale=float(center))

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

def _build_correlation_matrix(
    metrics: list[Metric],
    correlations: list[CorrelationPair],
) -> np.ndarray:
    """Assemble the n×n correlation matrix from metrics + pairs.

    The matrix depends only on metric name ORDER and the correlation pairs —
    not on trajectory, archetype, cohort, lag state, or any runtime value.
    Category B Layer 3 hoists the Cholesky factor to the top of
    ``generate_tables`` using this helper; fallback callers inside
    ``apply_correlations`` use the same helper so the math stays in one place.
    """
    names = [m.name for m in metrics]
    idx = {n: i for i, n in enumerate(names)}
    n = len(names)
    mat = np.eye(n)
    for pair in correlations:
        if pair.metric_a in idx and pair.metric_b in idx:
            i = idx[pair.metric_a]
            j = idx[pair.metric_b]
            mat[i, j] = pair.coefficient
            mat[j, i] = pair.coefficient
    return mat


# --- Higham nearest-PD projection (M111) ------------------------------------

# Threshold below which a per-pair adjustment is treated as numerical noise
# rather than a real change. Higham at convergence can drift cells by ~1e-15
# off Frobenius-optimal even when the input was already PD; without this
# guard, byte-identical-output regressions would surface for PD configs.
_ADJUSTMENT_NOISE_FLOOR = 1e-12


def _higham_nearest_pd(
    A: np.ndarray, max_iter: int = 200, tol: float = 1e-10,
) -> tuple[np.ndarray, bool]:
    """Higham 2002 alternating projections — nearest correlation matrix.

    Iterates between two projections with Dykstra correction:

      * ``P_S``: project onto the symmetric positive-semi-definite cone
        (eigendecompose, clamp negative eigenvalues to zero, recompose).
      * ``P_U``: project onto the unit-diagonal symmetric set (set
        diagonal to 1.0; off-diagonal stays as ``P_S`` left it).

    Dykstra's correction term ``DS`` is carried between iterations so
    the sequence converges to the Frobenius-nearest valid correlation
    matrix — a naive single-projection clip lands at *some* PSD matrix
    but not the optimal one.

    Returns ``(projected, converged)``. Convergence here means the
    iterate stabilized below ``tol`` (relative Frobenius change between
    successive iterates) — purely an iterate-stability test, not a
    minimum-eigenvalue test. Higham's optimal projection can land on
    the PSD boundary (min eigenvalue exactly 0) for inputs whose
    nearest correlation matrix is rank-deficient; lifting that to a
    strict-PD margin is ``_ensure_pd_margin``'s job. ``False`` means
    the loop hit ``max_iter`` first.
    """
    A_sym = (A + A.T) / 2.0
    Y = A_sym.copy()
    DS = np.zeros_like(A_sym)
    Y_prev: Optional[np.ndarray] = None
    for _ in range(max_iter):
        R = Y - DS
        eigvals, eigvecs = np.linalg.eigh(R)
        eigvals_pos = np.maximum(eigvals, 0.0)
        X = (eigvecs * eigvals_pos) @ eigvecs.T
        X = (X + X.T) / 2.0
        DS = X - R
        Y_new = X.copy()
        np.fill_diagonal(Y_new, 1.0)
        if Y_prev is not None:
            base = float(np.linalg.norm(Y_new, ord="fro"))
            diff = float(np.linalg.norm(Y_new - Y_prev, ord="fro"))
            if base > 0 and diff / base < tol:
                return Y_new, True
        Y_prev = Y_new
        Y = Y_new
    return Y, False


def _ensure_pd_margin(Y: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """Lift min eigenvalue to ``>= tol`` if Higham landed on the boundary.

    Higham's Frobenius-optimal projection of a non-PD matrix can land
    on the PSD boundary (min eigenvalue = 0 — PSD but not PD). Cholesky
    needs strict PD; this helper applies the smallest eigenvalue floor
    that pushes the result above ``tol``, then renormalizes to unit
    diagonal. When ``Y`` already has margin, returns it unchanged.

    The internal floor uses ``2 * tol`` to absorb the float-precision
    drift introduced by the subsequent diagonal renormalization step
    (``D^(-1/2) X D^(-1/2)`` can drop eigenvalues by a few ULPs). With
    a 2× cushion the result reliably clears ``tol`` even on 50×50
    adversarial inputs where the unconstrained minimum sat at exactly
    the boundary.

    Not a "fallback" — Higham is doing the heavy lifting; this is a
    margin-only post-process. The fallback proper (``_eigvalue_clip_to_pd``)
    fires only when Higham itself fails to converge.
    """
    A_sym = (Y + Y.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(A_sym)
    if float(eigvals.min()) > tol:
        return Y
    floor = 2.0 * tol
    eigvals_clipped = np.maximum(eigvals, floor)
    X = (eigvecs * eigvals_clipped) @ eigvecs.T
    X = (X + X.T) / 2.0
    diag = np.sqrt(np.maximum(np.diag(X), floor))
    X = X / np.outer(diag, diag)
    np.fill_diagonal(X, 1.0)
    return X


def _eigvalue_clip_to_pd(A: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """Eigenvalue-clipping fallback for non-converging Higham.

    Clamps eigenvalues to ``>= tol`` then renormalizes to unit diagonal
    via ``D^(-1/2) X D^(-1/2)``. Result is a valid PD correlation matrix
    with minimum eigenvalue ``>= tol`` but is NOT Frobenius-optimal —
    Higham is the optimal projection; this is a safety net for the
    50×50-all-0.95 adversarial worst case where Higham's iterate
    refuses to stabilize within ``max_iter``.
    """
    A_sym = (A + A.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(A_sym)
    eigvals_clipped = np.maximum(eigvals, tol)
    X = (eigvecs * eigvals_clipped) @ eigvecs.T
    X = (X + X.T) / 2.0
    diag = np.sqrt(np.maximum(np.diag(X), tol))
    X = X / np.outer(diag, diag)
    np.fill_diagonal(X, 1.0)
    return X


def project_correlation_matrix(
    mat: np.ndarray, max_iter: int = 200, tol: float = 1e-10,
) -> tuple[np.ndarray, bool, bool]:
    """Project ``mat`` onto the nearest PD correlation matrix if needed.

    Returns ``(projected, projection_used, used_fallback)``.

      * ``mat`` already PD (Cholesky succeeds) → ``(mat.copy(), False, False)``.
        Byte-identical pass-through preserves pre-M111 behavior on every
        config whose user-specified matrix is already PD.
      * Higham iterate stabilizes → margin nudge if needed → returns
        ``(projected, True, False)``. The margin nudge is silent — it's
        part of the projection, not a separate fallback.
      * Higham hits ``max_iter`` → eigenvalue-clipping fallback runs and
        emits a ``UserWarning`` noting the fallback. Returns
        ``(clipped, True, True)``.

    Raises ``RuntimeError`` if both Higham and eigenvalue-clipping fail
    to produce a Cholesky-able matrix — should be impossible for any
    symmetric input. Replaces the pre-M111 silent identity fallback;
    the architectural constraint is "raise explicitly rather than
    silently return identity."
    """
    try:
        np.linalg.cholesky(mat)
        return mat.copy(), False, False
    except np.linalg.LinAlgError:
        pass

    projected, converged = _higham_nearest_pd(mat, max_iter=max_iter, tol=tol)
    if converged:
        result = _ensure_pd_margin(projected, tol=tol)
        try:
            np.linalg.cholesky(result)
            return result, True, False
        except np.linalg.LinAlgError:
            # Numerical pathology — Higham converged but the margin nudge
            # somehow didn't yield a Cholesky-able matrix. Fall through
            # to the dedicated fallback rather than raising mid-way.
            pass

    clipped = _eigvalue_clip_to_pd(mat, tol=tol)
    try:
        np.linalg.cholesky(clipped)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError(
            "Higham nearest-PD projection and eigenvalue-clipping fallback "
            "both failed to produce a positive-definite correlation matrix. "
            f"Input shape: {mat.shape}. Min eigenvalue of input: "
            f"{float(np.linalg.eigvalsh((mat + mat.T) / 2.0).min()):.6e}."
        ) from exc
    warnings.warn(
        f"Higham nearest-PD projection did not converge in {max_iter} "
        "iterations; falling back to eigenvalue-clipping. The result is a "
        "valid positive-definite correlation matrix but is not "
        "Frobenius-optimal.",
        UserWarning,
        stacklevel=2,
    )
    return clipped, True, True


def _correlation_adjustment_records(
    original: np.ndarray,
    projected: np.ndarray,
    metrics: list[Metric],
    correlations: list[CorrelationPair],
) -> list[dict]:
    """Per-pair before/after records for warnings + manifest.

    Walks ``correlations`` (so the records reflect what the user asked
    for, not all matrix off-diagonals), dedupes by unordered (a, b),
    drops entries below the numerical noise floor, sorts by
    (metric_a, metric_b) for deterministic warning text and manifest
    ordering.
    """
    names = [m.name for m in metrics]
    idx = {n: i for i, n in enumerate(names)}
    seen: set[tuple[str, str]] = set()
    records: list[dict] = []
    for pair in correlations:
        if pair.metric_a not in idx or pair.metric_b not in idx:
            continue
        key = tuple(sorted((pair.metric_a, pair.metric_b)))
        if key in seen:
            continue
        seen.add(key)
        i, j = idx[pair.metric_a], idx[pair.metric_b]
        req = float(original[i, j])
        ach = float(projected[i, j])
        diff = abs(req - ach)
        if diff <= _ADJUSTMENT_NOISE_FLOOR:
            continue
        records.append({
            "metric_a": pair.metric_a,
            "metric_b": pair.metric_b,
            "requested": req,
            "achieved": ach,
            "adjustment": diff,
        })
    records.sort(key=lambda r: (r["metric_a"], r["metric_b"]))
    return records


def _format_correlation_adjustment_warning(records: list[dict]) -> str:
    """Render the per-pair adjustment warning text.

    Format pinned by the M111 mission spec so downstream tooling that
    parses the warning string stays stable. ``stacklevel=2`` is the
    caller's responsibility, not this function's.
    """
    parts = [
        f"{r['metric_a']} ↔ {r['metric_b']}: "
        f"{r['requested']:.4f} → {r['achieved']:.4f} "
        f"(Δ{r['adjustment']:.4f})"
        for r in records
    ]
    return (
        "Correlation matrix was not positive definite. "
        f"Adjusted {len(records)} pairs to nearest valid values: "
        + ", ".join(parts)
    )


def apply_correlations(
    independent: dict[str, float],
    centers: dict[str, float],
    correlations: list[CorrelationPair],
    metrics: list[Metric],
    cholesky_L: Optional[np.ndarray] = None,
) -> dict[str, float]:
    """Adjust independent samples so pairwise correlations match config.

    F-01 / 0.4.0 — Gaussian copula. Each independent draw is pushed through
    its own distribution's CDF to produce a uniform [0,1] value, then through
    the standard normal inverse CDF to get a standard Gaussian. The Cholesky
    factor ``L`` is applied in Gaussian space (where every residual has unit
    variance by construction), then the transformed Gaussian is pushed back
    through the standard normal CDF and the metric's inverse CDF to recover
    a value in the original distribution's support.

    Why this replaces the pre-0.4.0 center-normalized-residual transform:
    center-normalization made metrics scale-comparable but not variance-
    comparable, so configured coefficients were attenuated by distribution-
    pair-dependent factors (~0.29× to ~0.91× of configured on the shipped
    templates). In Gaussian space the attenuation vanishes — the Cholesky
    delivers exactly the configured correlation, and the inverse CDF round
    trip preserves each metric's marginal distribution exactly. No sign
    flips either: a lognormal cannot return negative after ``ppf``.

    Bypass: if a metric's center is near zero (degenerate distribution —
    gamma shape≤0, lognorm scale≈0, poisson λ≈0), ``_get_scipy_dist``
    returns ``None`` and the independent draw passes through unchanged.
    This preserves the pre-0.4.0 near-zero-center behaviour so identities
    like ``position=0 → value≈0`` still hold.

    Poisson note: scipy's ``poisson.cdf`` is a step function; ``ppf`` maps
    a continuous uniform onto integer values. The copula still drives the
    Gaussian-space correlation to the configured value, but the observed
    Pearson on the resulting integers will be slightly below configured —
    inherent to correlating discrete distributions, and still dramatically
    closer than the pre-0.4.0 attenuation.

    M111: a non-PD correlation matrix is auto-corrected via Higham
    nearest-PD projection rather than raising. The pre-M111 raise
    documented "config defect, not runtime condition"; under M111 the
    load-time validator on ``PlotsimConfig`` already projected and
    warned the user, so by the time this cold-path is reached the
    projection is redundant in the production flow but kept as a safety
    net for callers that bypass the validator (programmatic config
    construction, ad-hoc tests). The cold path projects silently — the
    load-time validator owns the user-facing warning.

    Category B Layer 3 (SEC-08): when the caller has already computed the
    Cholesky factor at the top of ``generate_tables``, they pass it as
    ``cholesky_L`` and this function skips the redundant matrix assembly +
    ``np.linalg.cholesky`` call. The matrix is config-invariant across the
    per-(cohort, period) loop. Direct external callers that omit
    ``cholesky_L`` still work; the in-function path builds the matrix on
    demand.
    """
    if not correlations:
        return dict(independent)

    names = [m.name for m in metrics]
    metric_by_name = {m.name: m for m in metrics}
    n = len(names)

    # Build the Gaussian-space residual vector. Bypass metrics contribute 0
    # (the identity element under L @ · — they don't pull the other metrics'
    # correlated draws off target), and their independent value is preserved
    # unchanged at the end.
    z = np.zeros(n, dtype=float)
    bypass = [False] * n
    frozen_dists: list = [None] * n
    for i, name in enumerate(names):
        c = centers[name]
        v = independent[name]
        if v is None:
            bypass[i] = True
            continue
        dist_obj = _get_scipy_dist(metric_by_name[name], c)
        if dist_obj is None:
            bypass[i] = True
            continue
        u = float(dist_obj.cdf(v))
        if not np.isfinite(u):
            bypass[i] = True
            continue
        u = min(max(u, _CDF_CLAMP), 1.0 - _CDF_CLAMP)
        z[i] = float(sp_norm.ppf(u))
        frozen_dists[i] = dist_obj

    # F2 (M102): every metric bypassed → no correlation transform applies
    # and the independent draws are already the answer. Short-circuits the
    # Cholesky (or the matrix build for the cold path) on dead periods.
    if all(bypass):
        return dict(independent)

    if cholesky_L is not None:
        L = cholesky_L
    else:
        mat = _build_correlation_matrix(metrics, correlations)
        # M111: project to nearest PD if the matrix isn't already. PD
        # inputs pass through unchanged; non-PD inputs (only reachable
        # when the load-time PlotsimConfig validator was bypassed) get
        # auto-corrected. A RuntimeError from projection — both Higham
        # and the eigenvalue-clipping fallback failing — is genuinely
        # unrecoverable and propagates.
        projected, _used, _fallback = project_correlation_matrix(mat)
        L = np.linalg.cholesky(projected)

    if any(bypass):
        # F2 (M102): bypass-aware correlation. The pre-fix `L @ z` mixed the
        # forced z[i]=0 from bypass slots into every active metric's correlated
        # Gaussian, attenuating non-bypass-pair correlations during bypass
        # periods. Slice the original correlation matrix down to the active
        # rows/cols (recovered as L @ L.T — exact for the full-rank Cholesky
        # the load-time PSD gate guarantees), Cholesky-factor the principal
        # submatrix (PSD inherits to principal submatrices), and apply L_sub
        # only to the active z-values.
        active_idx = np.where(np.logical_not(bypass))[0]
        mat_full = L @ L.T
        L_sub = np.linalg.cholesky(mat_full[np.ix_(active_idx, active_idx)])
        corr_z = np.zeros(n, dtype=float)
        # Bypass slots stay at 0 here; the output loop below skips them via
        # `if bypass[i]: continue`, so the zeros never leak into `out`.
        corr_z[active_idx] = L_sub @ z[active_idx]
    else:
        corr_z = L @ z

    out = dict(independent)
    for i, name in enumerate(names):
        if bypass[i]:
            continue
        u_corr = float(sp_norm.cdf(corr_z[i]))
        u_corr = min(max(u_corr, _CDF_CLAMP), 1.0 - _CDF_CLAMP)
        out[name] = float(frozen_dists[i].ppf(u_corr))
    return out


# --- Causal lag --------------------------------------------------------------

def _compute_effective_position(
    current_position: float,
    metric: Metric,
    lag_buffer: Optional[dict[str, list[float]]],
    period_index: int,
) -> float:
    """Blend the current trajectory position with the driver's past
    effective position.

    Operates on pre-polarity positions in [0,1] so both operands share the
    same semantic axis ("how well is this entity doing"). The metric's own
    polarity is applied afterwards in ``position_to_center``.

    Blend weight comes from ``metric.causal_lag.blend_weight`` (0.4.0
    default 1.0). At ``w=1`` the blend collapses to ``driver_past`` —
    metric at ``T`` equals driver at ``T-N`` and cross-correlation peaks
    at exactly ``lag_periods``. Values below 1.0 soften the lag.

    Falls back to the unmodified current position when: no causal_lag is
    configured, ``period_index < lag_periods``, the lag buffer is
    ``None``, or the driver isn't in the buffer / has too short a history.
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
    w = metric.causal_lag.blend_weight
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

def _toposort_metrics(metrics: list[Metric]) -> list[Metric]:
    """Reorder metrics so each causal_lag driver comes before its target.

    The buffer-write-inside-the-per-metric-loop design in
    ``generate_metrics_for_period`` only composes chains truthfully when
    drivers are computed before targets. This helper produces that order.

    Uses ``graphlib.TopologicalSorter`` (stdlib, Python 3.9+). Nodes
    within a ready-layer are emitted in insertion order, so a config
    with no causal_lag entries returns its metrics in declaration order
    — preserving RNG consumption order for configs that don't use the
    lag feature.

    A driver that isn't in the input metric list (possible via
    programmatic ``Metric`` construction that bypasses
    ``PlotsimConfig._cross_reference_integrity``) is treated as absent:
    the target has no ordering constraint, matching
    ``_compute_effective_position``'s "driver not in buffer" fallback.

    Config-time cycle detection at
    ``PlotsimConfig._cross_reference_integrity`` (config.py) catches
    cycles before they reach here. The ``CycleError`` re-raise is
    defensive for direct library callers who construct metrics outside
    a validated config.
    """
    by_name = {m.name: m for m in metrics}
    ts: TopologicalSorter[str] = TopologicalSorter()
    for m in metrics:
        if m.causal_lag is not None and m.causal_lag.driver in by_name:
            ts.add(m.name, m.causal_lag.driver)
        else:
            ts.add(m.name)
    try:
        ordered = list(ts.static_order())
    except CycleError as exc:
        raise ValueError(
            f"causal_lag metrics form a cycle: {exc.args[1]!r}"
        ) from exc
    return [by_name[name] for name in ordered]


def _apply_archetype_overrides(
    metric: Metric, archetype: Optional[Archetype],
) -> Metric:
    """Return `metric` with overridable fields substituted from the archetype.

    Distribution, distribution params, and (M114) ``value_range`` may be
    overridden per-archetype. Polarity and causal_lag are never overridable —
    polarity flips would silently invert the archetype's directional intent,
    and lag chains are global structural objects.

    The ``value_range`` substitution propagates through the entire downstream
    pipeline because every center/sampler/clamper helper reads
    ``metric.value_range`` from the (possibly overridden) effective Metric:

      * ``position_to_center`` (``beta`` branch) shifts the center into the
        override span;
      * ``sample_single_metric`` (``beta`` branch) draws against the
        override span;
      * ``_get_scipy_dist`` (``beta`` branch) parameterizes the copula CDF
        consistently with the override span;
      * ``_clamp_and_round`` clamps every distribution to the override
        bounds AFTER noise.

    Subset semantics (``override.value_range`` ⊆ ``metric.value_range``) are
    enforced at config load in
    ``PlotsimConfig._cross_reference_integrity``; this helper trusts the
    pre-validated config.
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
    if override.value_range is not None:
        updates["value_range"] = override.value_range
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
    cholesky_L: Optional[np.ndarray] = None,
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
        if lag_buffer is not None:
            # Append this metric's effective position BEFORE moving on to
            # the next metric, so later metrics in the same period that
            # depend on this one (multi-hop chains) see a fully resolved
            # value at index period_index-lag. Caller must iterate
            # metrics in topological driver→target order for chains to
            # compose correctly; ``generate_entity_metrics`` does this.
            lag_buffer[em.name].append(eff_pos)
        center = position_to_center(eff_pos, em)
        centers[em.name] = center
        independent[em.name] = sample_single_metric(center, em, rng)

    if correlations:
        correlated = apply_correlations(
            independent, centers, correlations, effective,
            cholesky_L=cholesky_L,
        )
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
    cholesky_L: Optional[np.ndarray] = None,
) -> dict[str, np.ndarray]:
    """Generate every metric's full time series for one entity.

    Walks forward through the trajectory, feeding each period's effective
    position (post-lag-blend) into a fresh lag buffer so later periods
    can look backwards through ``causal_lag``. The buffer is local to
    this call — generating another entity starts with an empty buffer,
    so lag history cannot leak across entities.

    Metrics are processed in topological driver→target order
    (``_toposort_metrics``). For configs without causal_lag chains this
    is a stable permutation of declaration order, so RNG consumption
    order is preserved. For configs with chains, drivers are guaranteed
    to have their effective position buffered before any target reads
    it, which is what makes multi-hop lags compose.

    When ``archetype`` is provided, any ``metric_overrides`` it declares
    are applied per period (distribution family and shape params only).

    Return arrays are ``int`` for poisson, ``float`` otherwise. Whether
    a metric's series is poisson is decided on the (possibly
    overridden) effective distribution, so an override from ``poisson``
    → ``normal`` (or vice versa) correctly changes the output dtype. If
    any MCAR null appears in a metric's series, that array becomes
    ``dtype=object`` to carry ``None`` alongside numbers.
    """
    sorted_metrics = _toposort_metrics(list(metrics))
    effective = [_apply_archetype_overrides(m, archetype) for m in sorted_metrics]
    n_periods = len(trajectory)
    lag_buffer: dict[str, list[float]] = {m.name: [] for m in sorted_metrics}
    collected: dict[str, list] = {m.name: [] for m in sorted_metrics}

    for t in range(n_periods):
        pos = float(trajectory[t])
        # lag_buffer is now populated inline inside generate_metrics_for_period
        # — no outer-loop append. Effective positions (not raw trajectory) land
        # in the buffer, so chains A→B→C compose.
        period_out = generate_metrics_for_period(
            pos, sorted_metrics, correlations, noise, lag_buffer, t, rng,
            archetype=archetype, cholesky_L=cholesky_L,
        )
        for m in sorted_metrics:
            collected[m.name].append(period_out[m.name])

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
