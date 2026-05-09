"""plotsim._distribution_registry — distribution family dispatch.

Single source of truth for the per-family math the metric sampler, copula,
and batch counterparts all need to share. Replaces the 5-6 identical
distribution ladders that lived in ``plotsim.metrics`` (one in
``sample_single_metric``, one in ``_get_scipy_dist``, one in
``sample_single_metric_batch``, one in ``_column_cdf_batch`` / ``_column_ppf_batch``,
plus the bypass ladder).

Each family registers four primitives:

  * ``sample_scalar(center, params, value_range, rng)`` — one independent draw
    centered on ``center``. Returns float.
  * ``sample_batch(centers, params, value_range, rng)`` — vectorized draw
    centered on a per-row ``centers`` array. Returns shape ``(n,)`` float64.
  * ``ppf_batch(uniform, centers, params, value_range)`` — batched inverse-CDF
    pulling correlated uniforms back into the marginal. ``uniform`` is shape
    ``(n,)`` in ``(0, 1)``; ``centers`` is shape ``(n,)``. Returns shape ``(n,)``.
  * ``direct_transform(gaussian, centers, params, value_range)`` — pull
    correlated Gaussians directly into the marginal without going through
    Φ + ppf. Returns ``None`` for families that genuinely need the ppf path
    (beta, poisson, gamma, weibull). Returns shape ``(n,)`` for families
    that admit a closed-form transform from a unit-Gaussian
    (lognorm, normal).

The transform contract is what powers M127b's copula flip:

  * ``direct_transform`` is the post-Cholesky path. Normal and lognorm have
    closed-form maps from a standard Gaussian to the metric's marginal, so
    the copula skips the Φ → ppf round trip entirely for them.
  * ``ppf_batch`` is the fallback for families that do not admit a closed-form
    Gaussian transform — beta, poisson, gamma, weibull go through
    ``Φ(corr_z) → ppf``.

Adding a new distribution family touches exactly one place: register a new
``DistributionFamily`` instance below.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
from scipy import stats as sp_stats


# Mirrors the constant in ``plotsim.metrics``. Centers below this are treated
# as zero for the lognorm/poisson degenerate-center policy: lognorm clamps
# the underlying mu, poisson treats lambda as 0.
_CENTER_EPS = 1e-9


@dataclass(frozen=True)
class DistributionFamily:
    """One row of the registry — collects a family's sampling/transform math."""

    name: str
    sample_scalar: Callable[[float, dict, Optional[Any], np.random.Generator], float]
    sample_batch: Callable[[np.ndarray, dict, Optional[Any], np.random.Generator], np.ndarray]
    ppf_batch: Callable[[np.ndarray, np.ndarray, dict, Optional[Any]], np.ndarray]
    # ``None`` when the family must round-trip via Φ + ppf for the copula.
    # Otherwise: ``(gaussian, centers, params, value_range) → values``.
    direct_transform: Optional[Callable[[np.ndarray, np.ndarray, dict, Optional[Any]], np.ndarray]]


# --- lognorm -----------------------------------------------------------------
#
# scipy's lognorm(s, scale=center) is the marginal we sample and invert.
# A unit Gaussian maps directly to this marginal via:
#     X = scale * exp(s * Z)
# where ``Z ~ N(0, 1)``. That's the closed form M127b uses to skip the
# Φ + ppf round trip for normal/lognorm families.


def _lognorm_sample_scalar(center, params, vr, rng):
    s = params["s"]
    safe_center = max(center, _CENTER_EPS)
    return float(rng.lognormal(mean=float(np.log(safe_center)), sigma=s))


def _lognorm_sample_batch(centers, params, vr, rng):
    s = float(params["s"])
    safe = np.maximum(centers, _CENTER_EPS)
    n = centers.shape[0]
    return rng.lognormal(mean=np.log(safe), sigma=s, size=n).astype(np.float64)


def _lognorm_ppf_batch(uniform, centers, params, vr):
    s = float(params["s"])
    safe = np.maximum(centers, _CENTER_EPS)
    return sp_stats.lognorm(s=s, scale=safe).ppf(uniform)


def _lognorm_direct_transform(gaussian, centers, params, vr):
    """Closed-form: ``X = scale * exp(s * Z)`` with ``scale = max(center, eps)``.

    Equivalent to ``lognorm(s, scale).ppf(Φ(z))`` but skips two scipy calls.
    """
    s = float(params["s"])
    safe = np.maximum(centers, _CENTER_EPS)
    return safe * np.exp(s * gaussian)


# --- gamma -------------------------------------------------------------------
#
# scipy's gamma(shape, scale=center/shape) has no closed-form unit-Gaussian
# transform; the copula keeps the Φ + ppf path for this family.


def _gamma_sample_scalar(center, params, vr, rng):
    shape = params["shape"]
    if shape <= 0.0 or center <= 0.0:
        return 0.0
    return float(rng.gamma(shape=shape, scale=center / shape))


def _gamma_sample_batch(centers, params, vr, rng):
    shape = float(params["shape"])
    n = centers.shape[0]
    if shape <= 0.0:
        return np.zeros(n, dtype=np.float64)
    active = centers > 0.0
    scales = np.where(active, centers / shape, 1.0)
    draws = rng.gamma(shape=shape, scale=scales, size=n)
    return np.where(active, draws, 0.0)


def _gamma_ppf_batch(uniform, centers, params, vr):
    shape = float(params["shape"])
    safe_centers = np.where(centers > 0.0, centers, 1.0)
    return sp_stats.gamma(a=shape, scale=safe_centers / shape).ppf(uniform)


# --- poisson -----------------------------------------------------------------
#
# Discrete distribution; ppf is a step function so the copula keeps the
# Φ + ppf path. Degenerate λ ≤ eps produces a deterministic-zero column
# (matches scalar ``sample_single_metric``'s ``max(center, 0.0)`` guard).


def _poisson_sample_scalar(center, params, vr, rng):
    lam = max(center, 0.0)
    return float(rng.poisson(lam=lam))


def _poisson_sample_batch(centers, params, vr, rng):
    lam = np.maximum(centers, 0.0)
    n = centers.shape[0]
    return rng.poisson(lam=lam, size=n).astype(np.float64)


def _poisson_ppf_batch(uniform, centers, params, vr):
    lam = np.maximum(centers, 0.0)
    return sp_stats.poisson(mu=lam).ppf(uniform)


# --- beta --------------------------------------------------------------------
#
# scipy.beta with affine reparameterization to land the mean on ``center``.
# Two paths depending on whether ``value_range`` is set; both produce the
# same shape modulo the affine transform. No closed-form unit-Gaussian map.


def _beta_sample_scalar(center, params, vr, rng):
    alpha = params["alpha"]
    beta = params["beta"]
    raw = float(rng.beta(a=alpha, b=beta))
    base_mean = alpha / (alpha + beta)
    if vr is not None and vr.min is not None and vr.max is not None:
        span = vr.max - vr.min
        return center + (raw - base_mean) * span
    scale = params.get("scale", 1.0)
    return (raw - base_mean + center) * scale


def _beta_sample_batch(centers, params, vr, rng):
    alpha = float(params["alpha"])
    beta = float(params["beta"])
    n = centers.shape[0]
    raw = rng.beta(a=alpha, b=beta, size=n)
    base_mean = alpha / (alpha + beta)
    if vr is not None and vr.min is not None and vr.max is not None:
        span = float(vr.max - vr.min)
        return centers + (raw - base_mean) * span
    scale = float(params.get("scale", 1.0))
    return (raw - base_mean + centers) * scale


def _beta_ppf_batch(uniform, centers, params, vr):
    alpha = float(params["alpha"])
    beta = float(params["beta"])
    base_mean = alpha / (alpha + beta)
    if vr is not None and vr.min is not None and vr.max is not None:
        span = float(vr.max - vr.min)
        return sp_stats.beta(
            a=alpha,
            b=beta,
            loc=centers - base_mean * span,
            scale=span,
        ).ppf(uniform)
    scale = float(params.get("scale", 1.0))
    return sp_stats.beta(
        a=alpha,
        b=beta,
        loc=scale * (centers - base_mean),
        scale=scale,
    ).ppf(uniform)


# --- normal ------------------------------------------------------------------
#
# Closed-form direct transform: ``X = center + sigma * Z``. Sigma > 0 is
# enforced at config-load by the marginal validator.


def _normal_sample_scalar(center, params, vr, rng):
    sigma = params["sigma"]
    return float(rng.normal(loc=center, scale=sigma))


def _normal_sample_batch(centers, params, vr, rng):
    sigma = float(params["sigma"])
    n = centers.shape[0]
    return rng.normal(loc=centers, scale=sigma, size=n).astype(np.float64)


def _normal_ppf_batch(uniform, centers, params, vr):
    sigma = float(params["sigma"])
    return sp_stats.norm(loc=centers, scale=sigma).ppf(uniform)


def _normal_direct_transform(gaussian, centers, params, vr):
    """Closed-form: ``X = center + sigma * Z``."""
    sigma = float(params["sigma"])
    return centers + sigma * gaussian


# --- weibull -----------------------------------------------------------------
#
# weibull_min(c=shape, scale=center). No closed-form Gaussian transform.


def _weibull_sample_scalar(center, params, vr, rng):
    shape = params["shape"]
    return float(rng.weibull(a=shape)) * center


def _weibull_sample_batch(centers, params, vr, rng):
    shape = float(params["shape"])
    n = centers.shape[0]
    if shape <= 0.0:
        return np.zeros(n, dtype=np.float64)
    return rng.weibull(a=shape, size=n).astype(np.float64) * centers


def _weibull_ppf_batch(uniform, centers, params, vr):
    shape = float(params["shape"])
    safe_centers = np.where(centers > 0.0, centers, 1.0)
    return sp_stats.weibull_min(c=shape, scale=safe_centers).ppf(uniform)


# --- Registry ----------------------------------------------------------------

DISTRIBUTION_REGISTRY: dict[str, DistributionFamily] = {
    "lognorm": DistributionFamily(
        name="lognorm",
        sample_scalar=_lognorm_sample_scalar,
        sample_batch=_lognorm_sample_batch,
        ppf_batch=_lognorm_ppf_batch,
        direct_transform=_lognorm_direct_transform,
    ),
    "gamma": DistributionFamily(
        name="gamma",
        sample_scalar=_gamma_sample_scalar,
        sample_batch=_gamma_sample_batch,
        ppf_batch=_gamma_ppf_batch,
        direct_transform=None,
    ),
    "poisson": DistributionFamily(
        name="poisson",
        sample_scalar=_poisson_sample_scalar,
        sample_batch=_poisson_sample_batch,
        ppf_batch=_poisson_ppf_batch,
        direct_transform=None,
    ),
    "beta": DistributionFamily(
        name="beta",
        sample_scalar=_beta_sample_scalar,
        sample_batch=_beta_sample_batch,
        ppf_batch=_beta_ppf_batch,
        direct_transform=None,
    ),
    "normal": DistributionFamily(
        name="normal",
        sample_scalar=_normal_sample_scalar,
        sample_batch=_normal_sample_batch,
        ppf_batch=_normal_ppf_batch,
        direct_transform=_normal_direct_transform,
    ),
    "weibull": DistributionFamily(
        name="weibull",
        sample_scalar=_weibull_sample_scalar,
        sample_batch=_weibull_sample_batch,
        ppf_batch=_weibull_ppf_batch,
        direct_transform=None,
    ),
}


def get_family(distribution: str) -> DistributionFamily:
    """Return the registered family for ``distribution`` or raise ValueError."""
    fam = DISTRIBUTION_REGISTRY.get(distribution)
    if fam is None:
        raise ValueError(f"unsupported distribution {distribution!r}")
    return fam
