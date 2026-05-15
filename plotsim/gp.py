"""plotsim.gp — RBF Gaussian-process kernel fit for trajectory characterization.

Mission 026: emits a per-archetype (or per-entity for override-bearing
entities) RBF kernel fit into the manifest so downstream consumers can
characterize trajectory smoothness (length scale, signal variance) without
re-fitting their own GP against the realized metric values.

This module is analytical-only and stateless:

  * No filesystem I/O.
  * No engine state — callers translate trajectory arrays into ``(x, y)``
    training data and pass them in.
  * No RNG — same input → same fit result.

The kernel is RBF (squared exponential) plus a homoscedastic noise term:

    k(t, t') = signal_variance * exp(-0.5 * (t - t')² / length_scale²)
             + noise_variance * I

Hyperparameters are optimized in log-space (``L-BFGS-B``) for numerical
stability — positivity is guaranteed by the parametrization. Degenerate
inputs (constant trajectories, ``< 3`` finite training points,
zero-range support, Cholesky failures, optimizer non-convergence) are
caught and surfaced as ``RBFFitResult(converged=False, hyperparameters=None,
log_marginal_likelihood=None)`` — never as exceptions. This is the
contract the manifest builder relies on: a failed fit must not abort
manifest emission.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import minimize

_LOG_TWO_PI = float(np.log(2.0 * np.pi))
# Trajectories whose variance is below this floor are treated as flat —
# the RBF likelihood surface is degenerate (signal_variance → 0 floor)
# and fitting has no informational content.
_FLAT_VARIANCE_FLOOR = 1e-12
# Log-space bounds for each hyperparameter. Keeps the optimizer away from
# numerically unstable regions where Cholesky would fail.
_LOG_HYPER_MIN = -8.0
_LOG_HYPER_MAX = 8.0
# Jitter added to the kernel diagonal for numerical stability beyond the
# noise term itself — keeps near-singular matrices factorizable.
_JITTER = 1e-10


@dataclass(frozen=True)
class RBFFitResult:
    """Outcome of one RBF kernel fit.

    ``converged=True`` means the optimizer reported success AND produced
    finite hyperparameters. ``converged=False`` is the catch-all for
    every degenerate path; consumers should gate downstream usage on the
    flag rather than inspecting ``hyperparameters`` directly.
    """

    converged: bool
    hyperparameters: Optional[dict[str, float]]
    log_marginal_likelihood: Optional[float]
    n_train: int


def _neg_log_marginal_likelihood(
    log_theta: np.ndarray,
    y_centered: np.ndarray,
    d2: np.ndarray,
) -> float:
    """Negative log marginal likelihood for the RBF + noise kernel.

    ``d2`` is the pairwise squared-distance matrix of the training inputs;
    ``y_centered`` is the training targets with the empirical mean
    subtracted (so the GP prior mean of zero is appropriate). Returns
    ``1e10`` as a soft penalty for Cholesky failures so the optimizer
    backs off the bad region instead of crashing.
    """
    log_length_scale, log_signal_var, log_noise_var = log_theta
    length_scale = float(np.exp(log_length_scale))
    signal_var = float(np.exp(log_signal_var))
    noise_var = float(np.exp(log_noise_var))
    n = y_centered.shape[0]
    k = signal_var * np.exp(-0.5 * d2 / (length_scale * length_scale))
    k = k + (noise_var + _JITTER) * np.eye(n)
    try:
        chol = np.linalg.cholesky(k)
    except np.linalg.LinAlgError:
        return 1e10
    alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, y_centered))
    log_det = 2.0 * float(np.sum(np.log(np.diag(chol))))
    nll = 0.5 * float(y_centered @ alpha) + 0.5 * log_det + 0.5 * float(n) * _LOG_TWO_PI
    if not np.isfinite(nll):
        return 1e10
    return float(nll)


def fit_rbf(x: np.ndarray, y: np.ndarray) -> RBFFitResult:
    """Fit an RBF kernel + homoscedastic noise to ``(x, y)``.

    Args:
        x: 1-D training inputs (typically period indices as float64).
        y: 1-D training targets (typically trajectory positions in [0, 1]).

    Returns:
        ``RBFFitResult``. Non-finite cells in either ``x`` or ``y`` are
        masked out before fitting. Returns ``converged=False`` (and null
        hyperparameters / likelihood) when:

          * fewer than 3 finite training points remain after masking;
          * the masked ``y`` has variance below ``_FLAT_VARIANCE_FLOOR``;
          * the masked ``x`` has zero range (all inputs identical);
          * the optimizer fails to converge OR returns a non-finite NLL.

        On success ``hyperparameters`` carries the three keys
        ``length_scale``, ``signal_variance``, ``noise_variance`` (in the
        original unstandardized scale) and ``log_marginal_likelihood``
        carries the maximized value (positive sign — the function
        minimizes the *negative* log likelihood and the result is
        negated before reporting).
    """
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.shape != y_arr.shape or x_arr.ndim != 1:
        return RBFFitResult(
            converged=False,
            hyperparameters=None,
            log_marginal_likelihood=None,
            n_train=0,
        )
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    xf = x_arr[mask]
    yf = y_arr[mask]
    n = int(xf.size)
    if n < 3:
        return RBFFitResult(
            converged=False,
            hyperparameters=None,
            log_marginal_likelihood=None,
            n_train=n,
        )
    y_var = float(np.var(yf))
    if y_var < _FLAT_VARIANCE_FLOOR:
        return RBFFitResult(
            converged=False,
            hyperparameters=None,
            log_marginal_likelihood=None,
            n_train=n,
        )
    x_range = float(xf.max() - xf.min())
    if x_range <= 0.0:
        return RBFFitResult(
            converged=False,
            hyperparameters=None,
            log_marginal_likelihood=None,
            n_train=n,
        )

    yf_centered = yf - float(np.mean(yf))
    diff = xf[:, None] - xf[None, :]
    d2 = diff * diff

    init_length_scale = max(x_range / 4.0, 1.0)
    init_signal_var = max(y_var, 1e-6)
    init_noise_var = max(y_var * 1e-3, 1e-8)
    log_theta0 = np.array(
        [
            float(np.log(init_length_scale)),
            float(np.log(init_signal_var)),
            float(np.log(init_noise_var)),
        ],
        dtype=np.float64,
    )
    bounds = [
        (_LOG_HYPER_MIN, _LOG_HYPER_MAX),
        (_LOG_HYPER_MIN, _LOG_HYPER_MAX),
        (_LOG_HYPER_MIN, _LOG_HYPER_MAX),
    ]
    try:
        result = minimize(
            _neg_log_marginal_likelihood,
            log_theta0,
            args=(yf_centered, d2),
            method="L-BFGS-B",
            bounds=bounds,
        )
    except (ValueError, np.linalg.LinAlgError):
        return RBFFitResult(
            converged=False,
            hyperparameters=None,
            log_marginal_likelihood=None,
            n_train=n,
        )
    if not bool(result.success) or not np.isfinite(result.fun):
        return RBFFitResult(
            converged=False,
            hyperparameters=None,
            log_marginal_likelihood=None,
            n_train=n,
        )
    log_length_scale, log_signal_var, log_noise_var = result.x
    return RBFFitResult(
        converged=True,
        hyperparameters={
            "length_scale": float(np.exp(log_length_scale)),
            "signal_variance": float(np.exp(log_signal_var)),
            "noise_variance": float(np.exp(log_noise_var)),
        },
        log_marginal_likelihood=float(-result.fun),
        n_train=n,
    )


__all__ = ["RBFFitResult", "fit_rbf"]
