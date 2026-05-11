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

Notes on the public surface:
    1. ``generate_metrics_for_period`` takes an explicit ``noise`` param so
       step 5 (apply noise) has access.
    2. ``lag_buffer`` holds effective positions (trajectory position
       optionally blended with the driver's past effective position via
       ``CausalLag.blend_weight``). Effective positions stay in [0,1] so
       the blend formula remains dimensionally coherent. Populated inside
       ``generate_metrics_for_period`` in topological order (drivers
       before targets), which makes multi-hop chains A→B→C compose
       truthfully.
    3. For beta, ``rng.beta(α,β)*scale`` would ignore center, which breaks
       the "higher position → higher mean" acceptance criterion. We
       shift-to-center instead: the beta shape's variance is preserved
       but its expected value lands on ``center``.
"""

from __future__ import annotations

import warnings
from graphlib import CycleError, TopologicalSorter
from typing import TYPE_CHECKING, Optional, cast

import numpy as np
from scipy.stats import norm as sp_norm

from plotsim._distribution_registry import (
    DISTRIBUTION_REGISTRY as DISTRIBUTION_REGISTRY,
    get_family,
)
from plotsim.config import (
    Archetype,
    CorrelationPair,
    Metric,
    NoiseConfig,
)

if TYPE_CHECKING:
    from plotsim.config import PlotsimConfig


# Below this absolute value, a center is treated as zero for the
# distribution-family policies that care (lognorm clamps the underlying mu,
# poisson treats lambda as 0). The registry's per-family math reads this
# constant via ``plotsim._distribution_registry``; this module re-exports
# it for backwards-compat with callers that imported the old name.
_CENTER_EPS = 1e-9

# Clamp Gaussian residuals before pushing them through Φ in the copula's
# ppf path. ``Φ(corr_z)`` near 0 or 1 propagates ±∞ into ``dist.ppf`` for
# bounded-support families (beta/poisson). 1e-10 in *uniform* space is the
# previous M127a value; the same tightness is preserved in Gaussian space
# by clipping ``corr_z`` to ``[Φ⁻¹(_CDF_CLAMP), Φ⁻¹(1 - _CDF_CLAMP)]``.
_CDF_CLAMP = 1e-10
_GAUSSIAN_CLAMP_LO = float(sp_norm.ppf(_CDF_CLAMP))
_GAUSSIAN_CLAMP_HI = float(sp_norm.ppf(1.0 - _CDF_CLAMP))


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

    Dispatches to ``DISTRIBUTION_REGISTRY[<family>].sample_scalar``. Value-
    range clamping and poisson integer-rounding happen in the caller
    (``_clamp_and_round``), AFTER noise injection, so this function returns
    the raw distributional draw.
    """
    family = get_family(metric.distribution)
    return family.sample_scalar(center, metric.params, metric.value_range, rng)


def _get_scipy_dist(metric: Metric, center: float):
    """Return a scipy.stats frozen distribution matching ``sample_single_metric``.

    M127b: kept as a backward-compatibility shim so verification tests
    (``tests/test_internal_verification.py``) can run KS / round-trip
    checks against the marginal CDF/PPF without coupling to the new
    distribution registry. Production code paths no longer depend on
    this helper — the new copula draws Gaussians and pushes them through
    the registry's ``direct_transform`` or ``ppf_batch`` directly.

    Returns ``None`` for degenerate centers, mirroring the pre-M127b
    contract (``lognorm`` ``center <= _CENTER_EPS``, ``gamma`` shape ≤ 0
    or center ≤ 0, ``poisson`` λ ≤ ``_CENTER_EPS``, ``normal`` σ ≤ 0,
    ``weibull`` shape ≤ 0 or center ≤ 0).
    """
    from scipy import stats as sp_stats

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
        if vr is not None and vr.min is not None and vr.max is not None:
            span = float(vr.max - vr.min)
            return sp_stats.beta(
                a=alpha,
                b=beta,
                loc=float(center) - base_mean * span,
                scale=span,
            )
        scale = float(params.get("scale", 1.0))
        return sp_stats.beta(
            a=alpha,
            b=beta,
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
    A: np.ndarray,
    max_iter: int = 200,
    tol: float = 1e-10,
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
    return cast(np.ndarray, X)


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
    return cast(np.ndarray, X)


def project_correlation_matrix(
    mat: np.ndarray,
    max_iter: int = 200,
    tol: float = 1e-10,
) -> tuple[np.ndarray, bool, bool]:
    """Project ``mat`` onto the nearest PD correlation matrix if needed.

    Returns ``(projected, projection_used, used_fallback)``.

      * ``mat`` already PD (Cholesky succeeds) → ``(mat.copy(), False, False)``.
        Byte-identical pass-through for every config whose user-specified
        matrix is already PD.
      * Higham iterate stabilizes → margin nudge if needed → returns
        ``(projected, True, False)``. The margin nudge is silent — it's
        part of the projection, not a separate fallback.
      * Higham hits ``max_iter`` → eigenvalue-clipping fallback runs and
        emits a ``UserWarning`` noting the fallback. Returns
        ``(clipped, True, True)``.

    Raises ``RuntimeError`` if both Higham and eigenvalue-clipping fail
    to produce a Cholesky-able matrix — should be impossible for any
    symmetric input. The architectural constraint is "raise explicitly
    rather than silently return identity."
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
        a, b = sorted((pair.metric_a, pair.metric_b))
        key: tuple[str, str] = (a, b)
        if key in seen:
            continue
        seen.add(key)
        i, j = idx[pair.metric_a], idx[pair.metric_b]
        req = float(original[i, j])
        ach = float(projected[i, j])
        diff = abs(req - ach)
        if diff <= _ADJUSTMENT_NOISE_FLOOR:
            continue
        records.append(
            {
                "metric_a": pair.metric_a,
                "metric_b": pair.metric_b,
                "requested": req,
                "achieved": ach,
                "adjustment": diff,
            }
        )
    records.sort(key=lambda r: (r["metric_a"], r["metric_b"]))
    return records


def _format_correlation_adjustment_warning(records: list[dict]) -> str:
    """Render the per-pair adjustment warning text.

    Format is part of the public contract — downstream tooling parses
    this string. ``stacklevel=2`` is the caller's responsibility, not
    this function's.
    """
    parts = [
        f"{r['metric_a']} ↔ {r['metric_b']}: "
        f"{r['requested']:.4f} → {r['achieved']:.4f} "
        f"(Δ{r['adjustment']:.4f})"
        for r in records
    ]
    return (
        "Correlation matrix was not positive definite. "
        f"Adjusted {len(records)} pairs to nearest valid values: " + ", ".join(parts)
    )


def apply_correlations(
    independent: dict[str, Optional[float]],
    centers: dict[str, float],
    correlations: list[CorrelationPair],
    metrics: list[Metric],
    cholesky_L: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> dict[str, Optional[float]]:
    """Generate copula-correlated marginal draws for every metric at one cell.

    M127b copula reformulation. The pipeline is now:

        1. ``z = rng.standard_normal(M)`` — one fresh Gaussian per metric.
        2. ``corr_z = L @ z`` — push through the Cholesky factor of the
           (already trajectory-compensated and Higham-projected)
           correlation matrix to apply pairwise correlations.
        3. ``corr_z = clip(corr_z, ...)`` — clamp away from the tails so
           the bounded-support family ppfs never see ``Φ⁻¹(0)`` / ``Φ⁻¹(1)``.
        4. Per family: if the family registers a ``direct_transform``
           (lognorm, normal), map ``corr_z → value`` in closed form.
           Otherwise (beta, poisson, gamma, weibull) push through
           ``Φ`` and call ``family.ppf_batch`` for the marginal value.

    The previous ``draw → CDF → Φ⁻¹ → L @ → Φ → ppf`` round-trip is gone;
    so are the bypass mechanism (independent-draw passthrough at degenerate
    centers), the per-row scalar fallback in the batch path, and the
    submatrix-Cholesky path that handled bypass slots. Degenerate centers
    now simply produce whatever the family-specific marginal yields at
    that center (lognorm with center ≈ 0 returns values near 0; poisson
    with λ ≈ 0 returns 0). The independent ``rng.standard_normal`` draw
    feeds the same correlation transform regardless of center.

    The ``independent`` argument is preserved for backward-compat with
    callers (notably ``plotsim.inspect``) that already drew per-metric
    samples and want to see the copula-correlated companion. **It is no
    longer consumed inside the correlated path** — the new copula draws
    its own Gaussian residuals from ``rng``. Callers passing ``None``
    cells (MCAR doesn't surface here, but defensive) get those slots
    propagated unchanged so ``apply_noise`` downstream sees consistent
    output keys. The empty-correlations short-circuit still returns
    ``dict(independent)`` byte-identically.

    ``cholesky_L`` is optional only for direct callers; the orchestrator
    always pre-computes it once at the top of ``generate_tables`` (the
    matrix is config-invariant across the per-(cohort, period) loop).
    When omitted, the function rebuilds the matrix and projects to the
    nearest PD via ``project_correlation_matrix`` (silent — the load-
    time validator on ``PlotsimConfig`` owns the user-facing warning).
    """
    if not correlations:
        return dict(independent)

    if rng is None:
        raise ValueError(
            "apply_correlations requires `rng` after the M127b copula flip; "
            "the new pipeline draws standard Gaussians from `rng` rather than "
            "round-tripping the caller-supplied `independent` values"
        )

    names = [m.name for m in metrics]
    n = len(names)
    if n == 0:
        return dict(independent)

    if cholesky_L is not None:
        L = cholesky_L
    else:
        mat = _build_correlation_matrix(metrics, correlations)
        # M111: project to nearest PD if the matrix isn't already. PD inputs
        # pass through unchanged; non-PD inputs (only reachable when the
        # load-time PlotsimConfig validator was bypassed) get auto-corrected.
        projected, _used, _fallback = project_correlation_matrix(mat)
        L = np.linalg.cholesky(projected)

    z = rng.standard_normal(n)
    corr_z = L @ z
    # Clip in Gaussian space so the bounded-ppf families never see ``Φ⁻¹(0)``
    # or ``Φ⁻¹(1)`` after the round-trip. ``_GAUSSIAN_CLAMP_LO/HI`` mirror
    # the previous uniform-space clamp at ``_CDF_CLAMP`` 1e-10.
    corr_z = np.clip(corr_z, _GAUSSIAN_CLAMP_LO, _GAUSSIAN_CLAMP_HI)

    out: dict[str, Optional[float]] = dict(independent)
    # Group metrics by family so each family's transform runs once on a
    # contiguous index slice. ``direct_transform`` families (lognorm,
    # normal) skip the Φ + ppf round trip entirely; the others push the
    # correlated Gaussian through Φ and then ``family.ppf_batch``.
    families = [get_family(m.distribution) for m in metrics]
    centers_arr = np.asarray([centers[name] for name in names], dtype=np.float64)

    # Direct-transform families: closed-form Z → value.
    direct_idx = [i for i, fam in enumerate(families) if fam.direct_transform is not None]
    for i in direct_idx:
        m = metrics[i]
        fam = families[i]
        val = fam.direct_transform(  # type: ignore[misc]
            np.asarray([corr_z[i]], dtype=np.float64),
            np.asarray([centers_arr[i]], dtype=np.float64),
            m.params,
            m.value_range,
        )
        out[names[i]] = float(val[0])

    # PPF families: Φ(corr_z) → ppf. Group by family name so each scipy
    # frozen-dist call covers all metrics in that family at once.
    ppf_groups: dict[str, list[int]] = {}
    for i, fam in enumerate(families):
        if fam.direct_transform is None:
            ppf_groups.setdefault(fam.name, []).append(i)
    for fam_name, idx_list in ppf_groups.items():
        idx_arr = np.asarray(idx_list, dtype=np.int64)
        u = sp_norm.cdf(corr_z[idx_arr])
        # Defense in depth — Φ on a clamped Gaussian stays away from the
        # exact endpoints, but the bounded-ppf families (beta) still read
        # their tails better with a uniform-space clamp.
        u = np.clip(u, _CDF_CLAMP, 1.0 - _CDF_CLAMP)
        # ppf_batch dispatches by family; one scipy call per family per cell.
        # All metrics in this group share the same family but may have
        # distinct params/value_range, so call ppf_batch one metric at a
        # time. This still collapses N scalar scipy calls into one batched
        # one when the same family appears multiple times across the cell.
        for i in idx_list:
            m = metrics[i]
            fam = families[i]
            local_u = np.asarray(
                [u[idx_list.index(i)]],
                dtype=np.float64,
            )
            local_centers = np.asarray(
                [centers_arr[i]],
                dtype=np.float64,
            )
            val = fam.ppf_batch(local_u, local_centers, m.params, m.value_range)
            out[names[i]] = float(val[0])

    return out


# --- M120: trajectory-aware correlation pre-compensation --------------------

# Above this metric count the additive trajectory + copula decomposition gets
# noisy enough that the realized table-wide Pearson signs flip on enough pairs
# to fall below the mission's 80% sign-match floor. Configs with more metrics
# than this skip pre-compensation and emit a warning rather than degrade
# silently. Mirrors `MAX_METRICS_FOR_COMPENSATION` in the mission spec.
_MAX_METRICS_FOR_COMPENSATION = 20


def _archetype_seasonal_sensitivity(
    archetype_name: str,
    config: "PlotsimConfig",
) -> float:
    """Size-weighted mean of ``entity.seasonal_sensitivity`` for this archetype.

    Entity sensitivities scale the global seasonal modulation per entity.
    The trajectory-covariance estimator runs once per archetype, so it needs a
    representative scalar — size-weighted mean keeps a 10-entity-cohort with
    sensitivity 2.0 from being equally weighted with a single-entity cohort
    at 0.5 inside the same archetype. Returns 1.0 if the archetype has no
    entities (defensive — config validators reject this earlier).
    """
    total_size = 0
    weighted = 0.0
    for ent in config.entities:
        if ent.archetype != archetype_name:
            continue
        size = max(1, int(ent.size))
        total_size += size
        weighted += size * float(ent.seasonal_sensitivity)
    if total_size == 0:
        return 1.0
    return weighted / total_size


def _archetype_centers(
    archetype: Archetype,
    metrics: list[Metric],
    n_periods: int,
    seasonal_factors: Optional[np.ndarray],
    entity_seasonal_sensitivity: float,
) -> np.ndarray:
    """Compute the ``(n_periods, n_metrics)`` center matrix for this archetype.

    Per-cell value is ``position_to_center(traj[t], effective_metric)`` with
    archetype overrides applied to the metric, then multiplied by the
    seasonal modulation factor when ``seasonal_factors`` is non-None.
    Causal-lag blending is intentionally NOT applied here — the trajectory
    covariance estimate is a population-level model of "what the engine
    expects to draw before the copula touches it," and the lag blend
    introduces per-entity history that doesn't apply to a one-shot estimate.
    Documented as a known limitation in the docstring of
    ``estimate_trajectory_covariance``.
    """
    from plotsim.trajectory import compute_trajectory

    traj = compute_trajectory(archetype, n_periods, overrides=None)
    effective = [_apply_archetype_overrides(m, archetype) for m in metrics]
    centers = np.zeros((n_periods, len(metrics)), dtype=np.float64)
    for j, em in enumerate(effective):
        for t in range(n_periods):
            center = position_to_center(float(traj[t]), em)
            if seasonal_factors is not None:
                strength = (
                    float(seasonal_factors[t])
                    * em.seasonal_sensitivity
                    * entity_seasonal_sensitivity
                )
                if strength != 0.0:
                    center = center * (1.0 + strength)
                    vr = em.value_range
                    if vr is not None:
                        if vr.min is not None and center < vr.min:
                            center = vr.min
                        if vr.max is not None and center > vr.max:
                            center = vr.max
            centers[t, j] = center
    return centers


def _safe_corrcoef(centers: np.ndarray) -> np.ndarray:
    """Pearson correlation across columns of ``centers`` (period axis = rows).

    Constant-column metrics (std == 0) make ``np.corrcoef`` return NaN for
    the affected pairs. Treat those as "no trajectory contribution" — sub a
    zero off-diagonal so the compensation step doesn't propagate NaN into
    the user matrix. Diagonal is forced to 1.0 regardless.
    """
    n_metrics = centers.shape[1]
    if n_metrics == 0:
        return np.zeros((0, 0), dtype=np.float64)
    if centers.shape[0] < 2:
        # One period → no variance, no correlation. Returns an identity-shaped
        # matrix so callers can subtract it without dimension errors.
        out = np.eye(n_metrics, dtype=np.float64)
        return out
    stds = centers.std(axis=0, ddof=0)
    n = n_metrics
    out = np.eye(n, dtype=np.float64)
    nonconst = np.where(stds > 0.0)[0]
    if len(nonconst) >= 2:
        sub = np.corrcoef(centers[:, nonconst], rowvar=False)
        # ``np.corrcoef`` on a degenerate input may still emit NaNs; scrub.
        sub = np.nan_to_num(sub, nan=0.0, posinf=0.0, neginf=0.0)
        out[np.ix_(nonconst, nonconst)] = sub
    np.fill_diagonal(out, 1.0)
    return out


def estimate_trajectory_covariance(
    config: "PlotsimConfig",
    metric_order: Optional[list[Metric]] = None,
) -> np.ndarray:
    """Expected within-archetype trajectory correlation, weighted by entity count.

    For each archetype:

      1. Build the base trajectory ``compute_trajectory(archetype, n_periods)``
         (no per-entity ``inflection_month`` overrides — this is a
         population-level estimate).
      2. Apply ``_apply_archetype_overrides`` to every metric so distribution-
         family swaps and ``value_range`` overrides feed into the centers.
      3. For each period, compute ``position_to_center`` per metric and
         multiply by the seasonal modulation factor (per-entity sensitivity
         is the size-weighted mean across entities of this archetype).
         Causal-lag blending is **not** applied — see ``_archetype_centers``
         for the rationale.
      4. Pearson-correlate metric columns across the period axis to get
         ``r_traj_a``, with NaN-safe handling for constant centers.

    The archetype-level matrices are then averaged with weights
    ``w_a = sum(entity.size for entity in archetype) / total_entity_size``
    to produce the table-wide trajectory contribution.

    The output is in the same order as ``metric_order`` (defaults to
    ``_toposort_metrics(config.metrics)`` so the returned matrix lines up
    with the Cholesky factor ``tables.py`` builds in toposort order).

    Returns a ``(n_metrics, n_metrics)`` matrix with diagonal 1.0 and
    off-diagonals in [-1, 1]. Empty correlations or zero archetypes →
    identity (compensation against identity is a no-op).
    """
    if metric_order is None:
        metric_order = _toposort_metrics(list(config.metrics))
    n = len(metric_order)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)

    # Local import: ``plotsim.tables`` already imports from ``plotsim.metrics``,
    # so a top-level import here would create a cycle.
    from plotsim.tables import _build_seasonal_factors

    n_periods = config.time_window.period_count()
    seasonal_factors = _build_seasonal_factors(config, n_periods)

    arch_by_name = {a.name: a for a in config.archetypes}
    archetype_sizes: dict[str, int] = {}
    for ent in config.entities:
        archetype_sizes[ent.archetype] = archetype_sizes.get(ent.archetype, 0) + max(
            1, int(ent.size)
        )
    total_size = sum(archetype_sizes.values())
    if total_size == 0:
        return np.eye(n, dtype=np.float64)

    accumulator = np.zeros((n, n), dtype=np.float64)
    for arch_name, size in archetype_sizes.items():
        archetype = arch_by_name.get(arch_name)
        if archetype is None:
            continue
        sensitivity = _archetype_seasonal_sensitivity(arch_name, config)
        centers = _archetype_centers(
            archetype,
            metric_order,
            n_periods,
            seasonal_factors,
            sensitivity,
        )
        r_a = _safe_corrcoef(centers)
        accumulator += (size / total_size) * r_a

    np.fill_diagonal(accumulator, 1.0)
    # Numerical hygiene — Pearson within ±1 by definition, but float-mean
    # accumulation can drift by an ULP.
    accumulator = cast(np.ndarray, np.clip(accumulator, -1.0, 1.0))
    np.fill_diagonal(accumulator, 1.0)
    return accumulator


def compensate_correlation_matrix(
    user_matrix: np.ndarray,
    traj_covariance: np.ndarray,
    metrics: list[Metric],
    correlations: list[CorrelationPair],
) -> tuple[np.ndarray, list[dict]]:
    """Subtract the trajectory contribution from declared pairs only.

    Off-diagonal cells of ``user_matrix`` correspond to the operator's
    declared correlation pairs (their explicit contract). Subtracting
    ``traj_covariance`` from those targets gives the copula a goal whose
    realized output, after the trajectory contribution recombines
    additively, lands close to what the user wrote.

    Crucially, **undeclared** pairs (auto-zero off-diagonals) are NOT
    compensated. Compensating them would treat ``0`` as a structural
    user contract and force ``r_copula = -r_traj`` everywhere, which
    pushes the matrix into a near-rank-1 / near-degenerate region that
    Higham heavily distorts; the Higham distortion then leaks back into
    the declared pairs and undoes the compensation we wanted there.
    Leaving undeclared pairs at zero — "pairs the user didn't mention
    follow whatever the trajectory does" — keeps Higham's downstream
    projection close to identity.

    Records are emitted only for the declared ``correlations`` pairs
    (the user wrote them; they're the contract the manifest reports
    against). Each record includes:

      * ``user_target`` — coefficient as written in the YAML.
      * ``trajectory_contribution`` — within-archetype-weighted Pearson the
        trajectory itself induces between this pair.
      * ``compensated_target`` — pre-clamp ``user_target - trajectory_contribution``.
      * ``achievable`` — ``compensated_target`` clamped to ``[-1, 1]``.
      * ``infeasible`` — True when ``compensated_target`` fell outside
        ``[-1, 1]`` (sign of the user target was preserved through the clamp
        but magnitude is bounded by what the copula can produce).
      * ``adjustment`` — ``abs(user_target - achievable)`` for sort/filter.

    Diagonal is forced to 1.0 so the result is a valid correlation
    matrix. The returned matrix is **not** Higham-projected — callers
    are expected to feed it through ``project_correlation_matrix``
    before ``np.linalg.cholesky``.
    """
    n = user_matrix.shape[0]
    if n == 0 or not correlations:
        return user_matrix.copy(), []

    names = [m.name for m in metrics]
    idx = {name: pos for pos, name in enumerate(names)}

    compensated = user_matrix.copy()
    np.fill_diagonal(compensated, 1.0)

    seen: set[tuple[str, str]] = set()
    records: list[dict] = []
    for pair in correlations:
        if pair.metric_a not in idx or pair.metric_b not in idx:
            continue
        a, b = sorted((pair.metric_a, pair.metric_b))
        key: tuple[str, str] = (a, b)
        if key in seen:
            continue
        seen.add(key)
        i, j = idx[pair.metric_a], idx[pair.metric_b]
        target = float(user_matrix[i, j])
        traj = float(traj_covariance[i, j])
        raw = target - traj
        achievable = max(-1.0, min(1.0, raw))
        infeasible = raw != achievable
        compensated[i, j] = achievable
        compensated[j, i] = achievable
        records.append(
            {
                "metric_a": pair.metric_a,
                "metric_b": pair.metric_b,
                "user_target": target,
                "trajectory_contribution": traj,
                "compensated_target": raw,
                "achievable": achievable,
                "infeasible": infeasible,
                "adjustment": abs(target - achievable),
            }
        )
    records.sort(key=lambda r: (r["metric_a"], r["metric_b"]))
    return compensated, records


def _format_correlation_compensation_warning(records: list[dict]) -> str:
    """Render the per-pair compensation warning text for infeasible pairs.

    Only infeasible records (the copula target fell outside ``[-1, 1]`` after
    the trajectory subtraction) get a user-visible line — feasible ones are
    invisible to the operator because the compensated copula simply delivers
    what they asked for. Empty input → empty string (caller short-circuits).
    """
    infeasible = [r for r in records if r["infeasible"]]
    if not infeasible:
        return ""
    parts = [
        f"{r['metric_a']} ↔ {r['metric_b']}: configured "
        f"{r['user_target']:.4f}, trajectory contributes "
        f"{r['trajectory_contribution']:+.4f}, copula target "
        f"{r['compensated_target']:.4f} (infeasible), achievable "
        f"≈ {r['achievable']:.4f}"
        for r in infeasible
    ]
    return (
        f"Trajectory-aware correlation pre-compensation: {len(infeasible)} "
        "configured correlation(s) are not jointly achievable given the "
        "archetype mix's trajectory covariance — copula targets clamped to "
        "the achievable range. " + "; ".join(parts)
    )


# --- Causal lag --------------------------------------------------------------


def _apply_logit_shift(p: float, log_odds_shift: float) -> float:
    """0.6-M8c: shift a position ``p`` in [0, 1] by ``log_odds_shift`` units
    in logit space.

    Mathematically: ``sigmoid(logit(p) + log_odds_shift)``. A positive
    shift pushes ``p`` toward 1 (raises the trajectory's effective
    position — e.g. boosts engagement); a negative shift pushes toward 0.
    Working in log-odds space gives the right "diminishing returns"
    behaviour for an A/B lift: a shift of +0.5 moves p=0.5 to ~0.62, but
    only moves p=0.9 to ~0.94 — the same lift produces less absolute
    movement near the boundaries.

    Numerical guards:
      * ``shift == 0.0`` short-circuits to ``p`` (preserves byte-identity
        for zero-shift entities — the control-arm contract).
      * ``p`` is clamped to ``[1e-12, 1 - 1e-12]`` before logit so
        boundary positions don't blow up. An entity that's flatlined at
        exactly 0 or 1 stays flatlined post-treatment; the lift can't
        move a position the trajectory has already pinned.
      * The sigmoid is computed via the ``z >= 0`` / ``z < 0`` split so
        ``np.exp`` overflow can't fire on extreme shifts.
    """
    if log_odds_shift == 0.0:
        return p
    eps = 1e-12
    p_clamped = max(eps, min(1.0 - eps, p))
    z = float(np.log(p_clamped / (1.0 - p_clamped))) + log_odds_shift
    if z >= 0.0:
        return float(1.0 / (1.0 + np.exp(-z)))
    e = float(np.exp(z))
    return e / (1.0 + e)


def _decay_weights(window: int, kernel: str) -> np.ndarray:
    """Adstock weights for ``window`` periods, indexed s=0 (most recent)
    to s=window-1 (oldest), normalised to sum to 1.

    ``geometric`` — weights ∝ ``0.5**s``. Half-life of one period; the
    most recent cell carries the largest share. Sum-normalised so the
    blend-weight semantic on top is unchanged from the discrete case.
    ``linear`` — weights ∝ ``window - s``, dropping linearly from
    ``window`` at s=0 to ``1`` at s=window-1, then sum-normalised.
    """
    if window < 1:
        raise ValueError(f"_decay_weights: window must be >= 1, got {window}")
    s = np.arange(window, dtype=np.float64)
    raw: np.ndarray
    if kernel == "geometric":
        raw = np.power(0.5, s)
    elif kernel == "linear":
        raw = (window - s).astype(np.float64)
    else:  # pragma: no cover — validator rejects unknown kernels
        raise ValueError(f"_decay_weights: unknown kernel {kernel!r}")
    normalised: np.ndarray = raw / float(raw.sum())
    return normalised


def _compute_effective_position(
    current_position: float,
    metric: Metric,
    lag_buffer: Optional[dict[str, list[float]]],
    period_index: int,
    treatment_shift: float = 0.0,
) -> float:
    """Blend the current trajectory position with the driver's past
    effective position, then apply the (optional) treatment lift.

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

    0.6-M8c: ``treatment_shift`` is applied LAST — after the lag blend,
    just before return, on every fall-through path. Default ``0.0`` is
    the control-arm and pre-treatment contract: byte-identical pre-M8c
    output. The caller (``generate_metrics_for_period``) decides whether
    treatment is active for the (entity, period) pair and passes either
    ``entity.treatment_lift_log_odds`` (when active) or ``0.0``.

    0.6-M9b: when ``causal_lag.decay`` is True, ``driver_past`` becomes a
    NaN-tolerant weighted sum over the buffer slice
    ``[period_index - lag - decay_window + 1, period_index - lag]`` with
    weights from ``_decay_weights(decay_window, decay_kernel)``. NaN
    cells (cold-start fallback) drop out and the surviving weights are
    renormalised; an all-NaN slice falls through to the unmodified
    current position, matching the single-read fallback. ``decay=False``
    is byte-identical to the pre-M9b single-read path.
    """
    eff = current_position
    if metric.causal_lag is not None:
        lag = metric.causal_lag.lag_periods
        if period_index >= lag and lag_buffer is not None:
            driver = metric.causal_lag.driver
            driver_history = lag_buffer.get(driver)
            if driver_history is not None and len(driver_history) >= period_index - lag + 1:
                cl = metric.causal_lag
                if cl.decay and cl.decay_window is not None:
                    # 0.6-M9b: NaN-tolerant weighted sum over the buffer
                    # slice. ``s=0`` is the most-recent cell at
                    # ``period_index - lag``; ``s=window-1`` is the oldest
                    # at ``period_index - lag - window + 1`` (clipped at 0).
                    window = cl.decay_window
                    end_idx = period_index - lag
                    start_idx = max(0, end_idx - window + 1)
                    raw_slice = driver_history[start_idx : end_idx + 1]
                    raw_arr = np.asarray(raw_slice, dtype=np.float64)
                    # Right-pad with NaN if start was clipped at 0 (window
                    # extends past period 0). The pad slot is "oldest",
                    # i.e. high s.
                    pad_count = window - len(raw_arr)
                    if pad_count > 0:
                        raw_arr = np.concatenate(
                            [raw_arr, np.full(pad_count, np.nan, dtype=np.float64)]
                        )
                    # raw_arr is ordered start_idx..end_idx (oldest first).
                    # Reverse so index 0 = most recent (matches _decay_weights).
                    raw_arr = raw_arr[::-1]
                    weights = _decay_weights(window, cl.decay_kernel)
                    valid = ~np.isnan(raw_arr)
                    if valid.any():
                        w_valid = weights[valid]
                        w_sum = w_valid.sum()
                        if w_sum > 0.0:
                            driver_past_val = float((raw_arr[valid] * w_valid).sum() / w_sum)
                            w = cl.blend_weight
                            eff = current_position * (1.0 - w) + driver_past_val * w
                else:
                    driver_past = driver_history[period_index - lag]
                    # 0.6-M8a: cold-start periods append NaN to the buffer (so the
                    # buffer stays period-index-aligned for lag lookups); a NaN
                    # driver_past means the driver wasn't yet active at
                    # ``period_index - lag``. Same fallback as "driver not in
                    # buffer" — current position only.
                    if not (isinstance(driver_past, float) and np.isnan(driver_past)):
                        w = cl.blend_weight
                        eff = current_position * (1.0 - w) + driver_past * w
    # 0.6-M8c: treatment shift in logit space. The early-return paths
    # above all collapsed into a single ``eff`` so the shift now applies
    # uniformly regardless of which lag fallback fired.
    if treatment_shift != 0.0:
        eff = _apply_logit_shift(eff, treatment_shift)
    return eff


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
        raise ValueError(f"causal_lag metrics form a cycle: {exc.args[1]!r}") from exc
    return [by_name[name] for name in ordered]


def _apply_archetype_overrides(
    metric: Metric,
    archetype: Optional[Archetype],
) -> Metric:
    """Return `metric` with overridable fields substituted from the archetype.

    Distribution, distribution params, and ``value_range`` may be overridden
    per-archetype. Polarity and causal_lag are never overridable — polarity
    flips would silently invert the archetype's directional intent, and lag
    chains are global structural objects.

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
    enforced at config load by ``PlotsimConfig`` cross-reference validators;
    this helper trusts the pre-validated config.
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
    seasonal_global: float = 0.0,
    entity_seasonal_sensitivity: float = 1.0,
    treatment_shift: float = 0.0,
) -> dict[str, Optional[float]]:
    """Generate every metric for one entity at one time step.

    Pipeline per metric:
        1. resolve archetype override (distribution/params) if any
        2. current position → optional lag blend with driver's past position
        3. (polarity + distribution-specific) position → center
        4. seasonal modulation — multiply center by
           ``(1 + seasonal_global × metric.seasonal_sensitivity ×
              entity_seasonal_sensitivity)``, then clamp to ``value_range``
           BEFORE the distributional draw. Skipped when
           ``seasonal_global == 0.0``.
        5. sample independent value from the distribution
    Then once across all metrics:
        6. apply Cholesky correlation on residuals (if correlations given)
        7. apply noise (if noise config given): gaussian → outlier → MCAR
        8. clamp to value_range, round poisson to int
    """
    effective = [_apply_archetype_overrides(m, archetype) for m in metrics]
    centers: dict[str, float] = {}
    independent: dict[str, Optional[float]] = {}
    correlations_active = bool(correlations)

    for em in effective:
        eff_pos = _compute_effective_position(
            trajectory_position,
            em,
            lag_buffer,
            period_index,
            treatment_shift=treatment_shift,
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
        if seasonal_global != 0.0:
            effective_strength = (
                seasonal_global * em.seasonal_sensitivity * entity_seasonal_sensitivity
            )
            if effective_strength != 0.0:
                center = center * (1.0 + effective_strength)
                vr = em.value_range
                if vr is not None:
                    if vr.min is not None and center < vr.min:
                        center = vr.min
                    if vr.max is not None and center > vr.max:
                        center = vr.max
        centers[em.name] = center
        # M127b: per-metric ``sample_single_metric`` only runs on the
        # no-correlations path. With correlations, the new copula draws
        # one batched ``rng.standard_normal(M)`` inside ``apply_correlations``
        # and pushes it through the family transforms — calling the per-
        # metric sampler too would double-count RNG draws and the indep
        # values get discarded.
        if not correlations_active:
            independent[em.name] = sample_single_metric(center, em, rng)
        else:
            independent[em.name] = None  # placeholder; populated below

    if correlations_active:
        assert correlations is not None  # correlations_active = bool(correlations)
        correlated = apply_correlations(
            independent,
            centers,
            correlations,
            effective,
            cholesky_L=cholesky_L,
            rng=rng,
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
    cholesky_by_period: Optional[list[np.ndarray]] = None,
    seasonal_factors: Optional[np.ndarray] = None,
    entity_seasonal_sensitivity: float = 1.0,
    treatment_lift_log_odds: Optional[float] = None,
    treatment_start_period: int = 0,
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

    # 0.6-M8c: pre-resolve the per-period treatment shift. None or 0.0
    # produces a flat-zero array — every per-period call sees
    # ``treatment_shift=0.0`` and ``_compute_effective_position``'s
    # short-circuit fires (byte-identical pre-M8c output for entities
    # with no treatment fields set). When set, the shift only applies
    # for periods ``>= treatment_start_period``; pre-treatment periods
    # see 0.0 so the AC "pre-treatment baseline is identical across
    # groups" holds at the trajectory level.
    treatment_shift_t = (
        treatment_lift_log_odds
        if treatment_lift_log_odds is not None and treatment_lift_log_odds != 0.0
        else None
    )

    for t in range(n_periods):
        pos = float(trajectory[t])
        # 0.6-M8a: NaN trajectory position = cold-start period (entity not
        # yet active). Skip metric generation entirely — emit ``None`` for
        # every metric. We still append NaN to ``lag_buffer`` for every
        # metric so the buffer stays period-index-aligned (downstream lag
        # lookups index by ``period_index - lag``, not by buffer position).
        # ``_compute_effective_position`` recognises a NaN ``driver_past``
        # and falls back to ``current_position`` — the same behaviour as
        # "driver not in buffer" / "history too short". No RNG draws happen
        # at NaN periods, so a given entity's RNG consumption shrinks by
        # exactly the size of its cold-start prefix; entities with
        # ``start_period=0`` consume RNG identically to pre-M8a.
        if np.isnan(pos):
            for m in sorted_metrics:
                collected[m.name].append(None)
                lag_buffer[m.name].append(float("nan"))
            continue
        # lag_buffer is now populated inline inside generate_metrics_for_period
        # — no outer-loop append. Effective positions (not raw trajectory) land
        # in the buffer, so chains A→B→C compose.
        seasonal_global_t = float(seasonal_factors[t]) if seasonal_factors is not None else 0.0
        # 0.6-M8c: only apply the lift on or after ``treatment_start_period``.
        # Pre-treatment periods see ``0.0`` so the trajectory's effective
        # position is identical across treatment / control arms — the
        # "pre-treatment baseline is identical" AC holds at this level.
        shift_t = (
            float(treatment_shift_t)
            if treatment_shift_t is not None and t >= treatment_start_period
            else 0.0
        )
        # 0.6-M11: per-period Cholesky factor selection. When the caller
        # supplies ``cholesky_by_period`` (the M11 orchestrator path),
        # index by period to pick the factor active at this phase. The
        # legacy ``cholesky_L`` parameter remains the fallback for direct
        # callers (notably ``plotsim.inspect``); when both are None the
        # downstream copula short-circuits via ``correlations`` empty.
        cholesky_L_t = cholesky_by_period[t] if cholesky_by_period is not None else cholesky_L
        period_out = generate_metrics_for_period(
            pos,
            sorted_metrics,
            correlations,
            noise,
            lag_buffer,
            t,
            rng,
            archetype=archetype,
            cholesky_L=cholesky_L_t,
            seasonal_global=seasonal_global_t,
            entity_seasonal_sensitivity=entity_seasonal_sensitivity,
            treatment_shift=shift_t,
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


# --- M121: vectorized (archetype-batched) generation -----------------------
#
# The per-(entity, period) Python loop in ``generate_entity_metrics`` walks
# each entity through every period one cell at a time. M121 adds a parallel
# path that groups entities by archetype, batches the metric draws + copula
# across the entity axis at each period, and produces the same dict-shaped
# result the orchestrator already consumes from the serial path. Selection
# is per-config via ``PlotsimConfig.generation_mode``.
#
# Design constraints (from `docs/engine-internals.md` §2.4a):
#   * Per-archetype Cholesky is *not* recomputed — the caller passes the
#     same ``cholesky_L`` that the serial path consumes (M120 compensation
#     and M111 Higham projection both happen at orchestrator level).
#   * Entities with per-entity overrides (``inflection_month`` etc.) are
#     excluded from the batch by the caller and processed via the serial
#     path. A batch only contains the "no-override" subset of entities
#     sharing one archetype.
#   * Vectorized and serial RNG consumption orders differ. Same seed +
#     same mode → byte-identical output; same seed across modes →
#     statistically equivalent but distinct cell values. Documented as
#     part of the dual-path determinism contract.
#   * Causal-lag chains use a (n_batch, n_periods) per-metric buffer so
#     each batch row carries its own lag history — no cross-entity
#     leakage.

# Threshold below which ``PlotsimConfig.generation_mode == "auto"`` selects
# the serial path. Vectorization wins above this; below it the constant
# overhead (archetype grouping, full-axis ndarray allocation) eats the
# small-config savings.
#
# M121b basis: ``analysis/perf/m121_vectorized.py`` measured the
# `saas_template.yaml` builder config (95 entities across 6 segments,
# 6 metrics, 24 monthly periods, 1 connection) and the stress config
# (1,020 entities across 3 segments, 20 metrics, 24 periods, 4
# connections). Speedups (serial / vectorized wall-clock):
#
#   * baseline (95 entities, 6 segments) → 3.75×
#   * stress (1,020 entities, 3 segments) → 69.6×
#
# The baseline lands above 2× even at 95 entities split across 6
# archetypes (largest archetype-batch ~25 entities). 50 stays the right
# threshold: the smallest archetype batch in the baseline is ~10
# entities and the run still beats serial by a factor of 4. Re-tuning
# down to ~30 would not change behavior on any bundled template (all
# already cross 50); re-tuning up would lose the baseline-config win.
#
# The threshold keys on ``max(archetype_group_size)`` (per-archetype
# batch size, not total entity count). This catches the thin-archetype
# case — 60 entities × 12 archetypes has avg group size 5 and would
# pay vectorized setup cost with no per-batch amortization win. See
# ``plotsim.tables._resolve_generation_mode`` for the resolution.
# Manifest's ``vectorized_threshold_used`` field carries the constant
# value at generation time so downstream tooling can detect drift if
# the heuristic refines further.
_VECTORIZED_AUTO_THRESHOLD = 50


def sample_single_metric_batch(
    centers: np.ndarray,
    metric: Metric,
    rng: np.random.Generator,
) -> np.ndarray:
    """Batched draw of one metric across ``n`` entities at one period.

    Dispatches to ``DISTRIBUTION_REGISTRY[<family>].sample_batch``.
    ``centers`` is shape ``(n,)``; the return is shape ``(n,)``. Value-
    range clamping and poisson rounding still happen later in the caller
    (``_clamp_and_round_batch``), AFTER noise.
    """
    centers = centers.astype(np.float64, copy=False)
    family = get_family(metric.distribution)
    return family.sample_batch(centers, metric.params, metric.value_range, rng)


def _draw_correlated_gaussians_batch(
    cholesky_L: np.ndarray,
    n: int,
    M: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw ``(n, M)`` correlated standard Gaussians via ``Z @ L^T``.

    M127b's batched copula entry: one ``rng.standard_normal((n, M))`` call
    fills a unit-Gaussian matrix; right-multiplying by ``L^T`` yields rows
    whose pairwise correlations equal ``L @ L^T`` — the configured
    correlation matrix (post-compensation, post-Higham). Subsequent
    family-grouped transforms use these correlated Gaussians as input.
    """
    z = rng.standard_normal((n, M))
    return cast(np.ndarray, z @ cholesky_L.T)


def _apply_correlations_batch(
    independent: Optional[np.ndarray],
    centers: np.ndarray,
    metrics: list[Metric],
    correlations: list[CorrelationPair],
    cholesky_L: Optional[np.ndarray],
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, int]:
    """Vectorized M127b copula across the batch axis at one period.

    ``centers`` is shape ``(n, M)``. ``cholesky_L`` is the Cholesky factor
    of the (already trajectory-compensated and Higham-projected)
    correlation matrix. Pipeline:

        1. ``Z = rng.standard_normal((n, M))``.
        2. ``corr_Z = Z @ L^T`` — columns now correlated according to L L^T.
        3. ``corr_Z = clip(corr_Z, ...)`` — keep bounded-ppf families away
           from the Gaussian tails.
        4. Per family, in declaration order: if ``direct_transform`` exists
           (lognorm, normal), map ``corr_Z[:, j]`` to the marginal in closed
           form. Otherwise, push through ``Φ`` and call
           ``family.ppf_batch(u, centers[:, j], params, value_range)``.

    The legacy ``draw → CDF → ndtri → L @ → ndtr → ppf`` round trip and
    its per-row scalar bypass fallback are deleted. ``independent`` is
    accepted but ignored on the correlated path — kept in the signature
    so existing callers don't have to be rewritten in lockstep.

    Returns ``(correlated, bypass_cell_count)``. The bypass count is now
    structurally zero (no fallback exists); the second tuple element is
    retained so the orchestrator's wiring stays compatible with the
    pre-M127b manifest field ``bypass_fallback_counts`` (which is now
    always an empty dict — see release notes).
    """
    del independent  # unused in M127b — see docstring
    n, M = centers.shape
    if not correlations or M == 0 or cholesky_L is None:
        # No copula step — caller already has the per-metric independent
        # draws. Return zeros-shape and let the caller fold them in.
        return np.zeros((n, M), dtype=np.float64), 0

    if rng is None:
        raise ValueError(
            "_apply_correlations_batch requires `rng` after the M127b "
            "copula flip; the new pipeline draws standard Gaussians "
            "from `rng` rather than round-tripping prior independent draws"
        )

    corr_z = _draw_correlated_gaussians_batch(cholesky_L, n, M, rng)
    corr_z = np.clip(corr_z, _GAUSSIAN_CLAMP_LO, _GAUSSIAN_CLAMP_HI)

    out = np.empty((n, M), dtype=np.float64)
    families = [get_family(m.distribution) for m in metrics]
    for j, (m, fam) in enumerate(zip(metrics, families)):
        col_centers = centers[:, j].astype(np.float64, copy=False)
        if fam.direct_transform is not None:
            out[:, j] = fam.direct_transform(
                corr_z[:, j],
                col_centers,
                m.params,
                m.value_range,
            )
        else:
            u = sp_norm.cdf(corr_z[:, j])
            u = np.clip(u, _CDF_CLAMP, 1.0 - _CDF_CLAMP)
            out[:, j] = fam.ppf_batch(u, col_centers, m.params, m.value_range)
    return out, 0


def _apply_noise_batch(
    values: np.ndarray,
    noise: NoiseConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Vectorized gaussian → outlier → MCAR pipeline across a 1-D batch.

    Mirrors ``apply_noise`` but operates on shape ``(n,)``. MCAR cells
    become ``np.nan`` (caller decides the dtype based on whether NaNs
    appeared in the full series). Each branch consumes one batched RNG
    call so the inner loop is O(M·P) RNG calls instead of O(n·M·P).

    Ordering matches the scalar version: gaussian first, then outlier
    replacement (independent of gaussian), then MCAR null. Outlier-
    replaced values can still be MCAR-nulled in the same step. Each
    branch consumes RNG only when its rate/sigma is non-zero, so a
    fully-zero ``NoiseConfig`` is a pass-through.
    """
    n = values.shape[0]
    v = values.astype(np.float64, copy=True)

    if noise.gaussian_sigma > 0.0:
        # Multiplicative jitter. Where v==0, fall back to absolute sigma.
        mag = np.where(v != 0.0, np.abs(v), 1.0)
        v = v + rng.normal(loc=0.0, scale=noise.gaussian_sigma * mag, size=n)

    if noise.outlier_rate > 0.0:
        coin = rng.random(size=n)
        triggered = coin < noise.outlier_rate
        sign = np.where(v >= 0.0, 1.0, -1.0)
        mag = np.where(v != 0.0, np.abs(v), 1.0)
        # Unconditional ``rng.uniform`` keeps RNG order deterministic
        # within the vectorized path even when ``triggered`` is empty —
        # the cross-mode RNG-divergence contract already covers any
        # difference vs the scalar path's per-cell branching.
        outlier = sign * rng.uniform(low=mag * 3.0, high=mag * 10.0, size=n)
        v = np.where(triggered, outlier, v)

    if noise.mcar_rate > 0.0:
        coin = rng.random(size=n)
        v = np.where(coin < noise.mcar_rate, np.nan, v)

    return v


def _clamp_and_round_batch(
    values: np.ndarray,
    metric: Metric,
) -> np.ndarray:
    """Vectorized ``_clamp_and_round`` — preserves NaN cells."""
    vr = metric.value_range
    nan_mask = np.isnan(values)
    out = values.astype(np.float64, copy=True)
    if vr is not None:
        if vr.min is not None:
            out = np.where(out < vr.min, vr.min, out)
        if vr.max is not None:
            out = np.where(out > vr.max, vr.max, out)
    if metric.distribution == "poisson":
        rounded = np.rint(out)
        out = rounded
    if nan_mask.any():
        out = np.where(nan_mask, np.nan, out)
    return out


def generate_archetype_batch(
    archetype: Archetype,
    batch_entities: list,
    metrics: list[Metric],
    correlations: Optional[list[CorrelationPair]],
    noise: Optional[NoiseConfig],
    n_periods: int,
    rng: np.random.Generator,
    cholesky_L: Optional[np.ndarray] = None,
    cholesky_by_period: Optional[list[np.ndarray]] = None,
    seasonal_factors: Optional[np.ndarray] = None,
    bypass_counter: Optional[dict[str, int]] = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Generate every metric series for ``len(batch_entities)`` entities.

    All entities in the batch share one archetype (so one trajectory)
    and have NO per-entity ``overrides``. They may differ in
    ``seasonal_sensitivity``, which scales the seasonal modulation per row.

    Pipeline at each period (vectorized along the batch axis):

      1. Resolve archetype overrides on every metric → effective metrics.
      2. Compute the (n_batch, M) center matrix:
           ``position_to_center(traj[t], em)`` per metric, then
           ``× (1 + seasonal[t] × em.sens × entity.sens[i])``,
           then clamp to ``value_range``.
      3. Apply causal-lag blending in topological driver→target order;
         maintains a (n_batch, n_periods) buffer per metric so chains
         A→B→C compose batch-wise without cross-entity leakage.
      4. Sample independent values via ``sample_single_metric_batch``.
      5. Apply the batched copula via ``_apply_correlations_batch``.
      6. Apply noise via ``_apply_noise_batch``.
      7. Clamp + round via ``_clamp_and_round_batch``.

    Returns ``dict[entity.name → dict[metric.name → np.ndarray]]`` —
    same shape as the serial path's return so the orchestrator can
    union vectorized + serial entity_metrics into one dict before
    handing it to ``build_fact_tables`` and ``build_bridge_tables``.

    Determinism: same ``(archetype, batch_entities, metrics, ..., rng_state)``
    → byte-identical output. RNG order differs from the serial path
    (per-batch ``size=n`` calls vs serial's per-cell scalar calls), so
    cross-mode cell values are not byte-identical even at the same seed.
    Documented as part of the dual-path determinism contract.

    ``bypass_counter`` is accepted for backward compat with M121b's
    pre-M127b orchestrator; it is no longer populated. The bypass
    machinery (``_bypass_mask_batch``, per-row scalar fallback, submatrix
    Cholesky) was removed in M127b — every cell of the new copula
    pipeline produces a finite value, so there is nothing to count.
    Callers passing a counter dict get it back unchanged.
    """
    from plotsim.trajectory import compute_trajectory

    n_batch = len(batch_entities)
    if n_batch == 0:
        return {}
    del bypass_counter  # unused in M127b — see docstring

    sorted_metrics = _toposort_metrics(list(metrics))
    effective = [_apply_archetype_overrides(m, archetype) for m in sorted_metrics]
    M = len(effective)

    # Trajectory is shared across the batch (no per-entity overrides in
    # the batch by construction; overridden entities run the serial path).
    traj = compute_trajectory(archetype, n_periods, overrides=None)

    # Per-entity seasonal sensitivity vector — broadcast across the
    # period axis when modulating centers.
    ent_sens = np.array(
        [float(e.seasonal_sensitivity) for e in batch_entities],
        dtype=np.float64,
    )  # shape (n_batch,)

    # Pre-compute base centers per (period, metric) — these are the
    # archetype-common values BEFORE seasonal modulation. Shape (P, M).
    base_centers = np.zeros((n_periods, M), dtype=np.float64)
    for j, em in enumerate(effective):
        for t in range(n_periods):
            base_centers[t, j] = position_to_center(float(traj[t]), em)

    # Per-metric seasonal sensitivity scalar — applied to the global
    # per-period strength alongside the entity sensitivity vector.
    em_sens = np.array(
        [float(em.seasonal_sensitivity) for em in effective],
        dtype=np.float64,
    )  # shape (M,)

    # Lag buffers — one (n_batch, n_periods) array per metric, holding
    # the *effective position* (post-lag-blend) at each period. Mirrors
    # the scalar ``lag_buffer`` dict-of-lists; shape switches to a 2D
    # ndarray so per-batch lookups stay vectorized.
    lag_buffer = {
        em.name: np.full((n_batch, n_periods), np.nan, dtype=np.float64) for em in effective
    }

    # Output: per-entity dict-of-arrays accumulator. Each metric's
    # series is built up period-by-period so we can decide dtype at the
    # end (object if any NaN appeared, int for poisson, else float).
    series = {
        e.name: {em.name: np.full(n_periods, np.nan, dtype=np.float64) for em in effective}
        for e in batch_entities
    }

    for t in range(n_periods):
        # 1. Effective positions per (batch, metric) at this period.
        #    Drivers walk the batch in declaration order; metrics walk
        #    in topo order so a chain A→B→C resolves A's effective
        #    position before B reads it.
        eff_pos = np.empty((n_batch, M), dtype=np.float64)
        for j, em in enumerate(effective):
            base_pos = float(traj[t])
            if em.causal_lag is None:
                eff_pos[:, j] = base_pos
                lag_buffer[em.name][:, t] = base_pos
                continue
            lag = em.causal_lag.lag_periods
            if t < lag:
                eff_pos[:, j] = base_pos
                lag_buffer[em.name][:, t] = base_pos
                continue
            driver = em.causal_lag.driver
            driver_buf = lag_buffer.get(driver)
            if driver_buf is None:
                eff_pos[:, j] = base_pos
                lag_buffer[em.name][:, t] = base_pos
                continue
            cl = em.causal_lag
            if cl.decay and cl.decay_window is not None:
                # 0.6-M9b: NaN-tolerant weighted sum over the buffer
                # slice. Mirrors the serial path. Indexing convention:
                # ``s=0`` is the most-recent cell at column ``t-lag``,
                # ``s=window-1`` is the oldest at ``t-lag-window+1`` (or
                # NaN-padded when the window extends past period 0).
                window = cl.decay_window
                end_col = t - lag
                start_col = max(0, end_col - window + 1)
                slice_arr = driver_buf[:, start_col : end_col + 1]  # (n_batch, taken)
                taken = slice_arr.shape[1]
                if taken < window:
                    pad = np.full((slice_arr.shape[0], window - taken), np.nan, dtype=np.float64)
                    slice_arr = np.concatenate([slice_arr, pad], axis=1)
                # Slice currently runs oldest→newest; flip so axis=1
                # index 0 == most-recent (matches _decay_weights).
                slice_arr = slice_arr[:, ::-1]
                weights = _decay_weights(window, cl.decay_kernel)
                valid = ~np.isnan(slice_arr)
                w_row = np.where(valid, weights[None, :], 0.0)
                w_sum = w_row.sum(axis=1)
                vals = np.where(valid, slice_arr, 0.0)
                weighted_sum = (vals * w_row).sum(axis=1)
                driver_past = np.where(
                    w_sum > 0.0, weighted_sum / np.where(w_sum > 0.0, w_sum, 1.0), base_pos
                )
                w = float(cl.blend_weight)
                blended = base_pos * (1.0 - w) + driver_past * w
                eff_pos[:, j] = blended
                lag_buffer[em.name][:, t] = blended
            else:
                driver_past = driver_buf[:, t - lag]  # shape (n_batch,)
                # If driver_past has NaN (history not yet populated for that
                # row — shouldn't happen since topo order resolves drivers
                # first), fall back to the base position.
                driver_past = np.where(np.isnan(driver_past), base_pos, driver_past)
                w = float(cl.blend_weight)
                blended = base_pos * (1.0 - w) + driver_past * w
                eff_pos[:, j] = blended
                lag_buffer[em.name][:, t] = blended

        # 2. Centers per (batch, metric) at this period — apply seasonal
        #    modulation row-wise + clamp to value_range. When seasonal
        #    factor is exactly 0 (M119 default) the multiplier is 1.0 and
        #    the result equals the base centers (matches scalar's branch
        #    elision so cross-mode equivalence holds at the center level).
        if seasonal_factors is not None and float(seasonal_factors[t]) != 0.0:
            # Recompute centers from effective positions (lag may have
            # shifted them off the trajectory's t-th position).
            centers = np.empty((n_batch, M), dtype=np.float64)
            for j, em in enumerate(effective):
                # Vectorized position_to_center along the batch axis.
                # The scalar call uses metric-specific arithmetic; for
                # batch we reuse ``_position_to_center_batch`` (defined
                # below) which dispatches by distribution.
                centers[:, j] = _position_to_center_batch(eff_pos[:, j], em)
            seasonal_t = float(seasonal_factors[t])
            mult = 1.0 + seasonal_t * em_sens[None, :] * ent_sens[:, None]
            centers = cast(np.ndarray, centers * mult)
            # Clamp to value_range column-by-column (each metric may
            # have different bounds).
            for j, em in enumerate(effective):
                vr = em.value_range
                if vr is not None:
                    if vr.min is not None:
                        centers[:, j] = np.where(centers[:, j] < vr.min, vr.min, centers[:, j])
                    if vr.max is not None:
                        centers[:, j] = np.where(centers[:, j] > vr.max, vr.max, centers[:, j])
        else:
            centers = np.empty((n_batch, M), dtype=np.float64)
            for j, em in enumerate(effective):
                centers[:, j] = _position_to_center_batch(eff_pos[:, j], em)

        # 3+4. M127b copula flip. With correlations, the new pipeline draws
        #    one ``rng.standard_normal((n, M))`` inside
        #    ``_apply_correlations_batch`` and pushes it through L^T plus
        #    family-grouped transforms — no per-metric independent draw is
        #    needed (it would just get discarded). Without correlations,
        #    fall back to the per-metric ``sample_single_metric_batch`` so
        #    each marginal is drawn from its own distribution as before.
        #
        # 0.6-M11: pick the per-period Cholesky factor at this phase. The
        # batched copula already operates per-period (entities batched on
        # axis 0 at one period at a time), so resolving the factor by
        # index ``[t]`` here covers phase-keyed correlations without any
        # within-batch axis split.
        cholesky_L_t = cholesky_by_period[t] if cholesky_by_period is not None else cholesky_L
        if correlations and cholesky_L_t is not None:
            correlated, _period_bypass = _apply_correlations_batch(
                None,
                centers,
                effective,
                correlations,
                cholesky_L_t,
                rng=rng,
            )
        else:
            correlated = np.empty((n_batch, M), dtype=np.float64)
            for j, em in enumerate(effective):
                correlated[:, j] = sample_single_metric_batch(
                    centers[:, j],
                    em,
                    rng,
                )

        # 5+6+7. Noise + clamp/round per metric column.
        for j, em in enumerate(effective):
            col = correlated[:, j]
            if noise is not None:
                col = _apply_noise_batch(col, noise, rng)
            col = _clamp_and_round_batch(col, em)
            for i, ent in enumerate(batch_entities):
                series[ent.name][em.name][t] = col[i]

    # Coerce dtypes per entity per metric — matches scalar's contract.
    result: dict[str, dict[str, np.ndarray]] = {}
    for ent in batch_entities:
        per_metric: dict[str, np.ndarray] = {}
        for em in effective:
            arr = series[ent.name][em.name]
            if np.isnan(arr).any():
                # Carry NaN as Python ``None`` (object dtype) so the
                # downstream ``_build_metrics_3d`` scrub matches the
                # serial path's MCAR contract.
                obj = np.empty(n_periods, dtype=object)
                for k in range(n_periods):
                    if np.isnan(arr[k]):
                        obj[k] = None
                    elif em.distribution == "poisson":
                        obj[k] = int(round(float(arr[k])))
                    else:
                        obj[k] = float(arr[k])
                per_metric[em.name] = obj
            elif em.distribution == "poisson":
                per_metric[em.name] = arr.astype(int)
            else:
                per_metric[em.name] = arr.astype(float)
        result[ent.name] = per_metric

    # M127b: bypass machinery deleted; nothing to surface to the counter.
    # The ``bypass_counter`` parameter remains in the signature for
    # backward compat — see docstring.

    return result


def _position_to_center_batch(
    positions: np.ndarray,
    metric: Metric,
) -> np.ndarray:
    """Vectorized ``position_to_center`` along the batch axis.

    Mirrors the scalar branch by branch. Polarity is applied on the
    array first; per-distribution math follows. Returns shape ``(n,)``.
    """
    p = np.where(metric.polarity == "negative", 1.0 - positions, positions)
    dist = metric.distribution
    params = metric.params
    if dist == "lognorm":
        loc = float(params.get("loc", 0.0))
        scale = float(params.get("scale", 1.0))
        return loc + scale * p
    if dist == "gamma":
        shape = float(params["shape"])
        scale = float(params.get("scale", 1.0))
        return shape * scale * p
    if dist == "poisson":
        lam = float(params.get("lambda", 1.0))
        return lam * p
    if dist == "beta":
        vr = metric.value_range
        if vr is not None and vr.min is not None and vr.max is not None:
            return float(vr.min) + p * float(vr.max - vr.min)
        return p
    if dist == "normal":
        mu = float(params.get("mu", 0.0))
        return mu * p
    if dist == "weibull":
        scale = float(params.get("scale", 1.0))
        return scale * p
    raise ValueError(f"unsupported distribution {dist!r}")
