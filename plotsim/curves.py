"""plotsim.curves — composable mathematical curves producing [0,1] trajectories.

What it does:
    Defines the curve library (sigmoid, exp_decay, step, logistic, plateau,
    oscillating, compound, sawtooth). Each curve takes a normalised time array
    plus curve-specific parameters and returns trajectory values in [0,1].
    The module also exposes CURVE_REGISTRY and evaluate_segment — the single
    entry point used by the trajectory engine (Mission 003).

Input:
    t (np.ndarray, values in [0,1], monotonically increasing within a segment)
    plus curve-specific parameters.

Output:
    np.ndarray of the same shape as t, clamped to [0,1].
"""

from __future__ import annotations

from typing import Any, Callable, cast

import numpy as np


def _empty_float() -> np.ndarray:
    return np.empty(0, dtype=float)


def _minmax_normalize(arr: np.ndarray, target_max: float = 1.0) -> np.ndarray:
    """Rescale arr so min→0 and max→target_max. Constant/empty inputs are clipped."""
    if arr.size == 0:
        return arr
    lo = float(arr.min())
    hi = float(arr.max())
    if hi - lo < 1e-12:
        return np.clip(arr, 0.0, target_max)
    return (arr - lo) / (hi - lo) * target_max


def sigmoid(
    t: np.ndarray,
    midpoint: float = 0.5,
    steepness: float = 10.0,
    rising: bool = True,
) -> np.ndarray:
    if t.size == 0:
        return _empty_float()
    raw = 1.0 / (1.0 + np.exp(-steepness * (t - midpoint)))
    normed = _minmax_normalize(raw)
    if not rising:
        normed = 1.0 - normed
    return np.clip(normed, 0.0, 1.0)


def exp_decay(t: np.ndarray, rate: float = 3.0) -> np.ndarray:
    if t.size == 0:
        return _empty_float()
    return cast(np.ndarray, np.clip(np.exp(-rate * t), 0.0, 1.0))


def step(
    t: np.ndarray,
    threshold: float = 0.5,
    before: float = 1.0,
    after: float = 0.0,
) -> np.ndarray:
    if t.size == 0:
        return _empty_float()
    out = np.where(t < threshold, before, after).astype(float)
    return np.clip(out, 0.0, 1.0)


def logistic(
    t: np.ndarray,
    k: float = 10.0,
    midpoint: float = 0.5,
    ceiling: float = 1.0,
) -> np.ndarray:
    if t.size == 0:
        return _empty_float()
    raw = ceiling / (1.0 + np.exp(-k * (t - midpoint)))
    normed = _minmax_normalize(raw, target_max=ceiling)
    return np.clip(normed, 0.0, 1.0)


def plateau(t: np.ndarray, level: float = 0.5) -> np.ndarray:
    if t.size == 0:
        return _empty_float()
    return np.clip(np.full(t.shape, level, dtype=float), 0.0, 1.0)


def oscillating(
    t: np.ndarray,
    period: float = 2.0,
    amplitude: float = 0.2,
    center: float = 0.5,
) -> np.ndarray:
    if t.size == 0:
        return _empty_float()
    raw = center + amplitude * np.sin(2.0 * np.pi * period * t)
    return cast(np.ndarray, np.clip(raw, 0.0, 1.0))


def compound(
    t: np.ndarray,
    base_rate: float = 0.05,
    acceleration: float = 0.02,
) -> np.ndarray:
    if t.size == 0:
        return _empty_float()
    rate = base_rate + acceleration * t
    raw = np.cumsum(rate)
    normed = _minmax_normalize(raw)
    return np.clip(normed, 0.0, 1.0)


def sawtooth(
    t: np.ndarray,
    period: float = 3.0,
    amplitude: float = 0.8,
    base: float = 0.1,
) -> np.ndarray:
    if t.size == 0:
        return _empty_float()
    raw = base + amplitude * ((t * period) % 1.0)
    return np.clip(raw, 0.0, 1.0)


CURVE_REGISTRY: dict[str, Callable[..., np.ndarray]] = {
    "sigmoid": sigmoid,
    "exp_decay": exp_decay,
    "step": step,
    "logistic": logistic,
    "plateau": plateau,
    "oscillating": oscillating,
    "compound": compound,
    "sawtooth": sawtooth,
}


def evaluate_segment(
    t_segment: np.ndarray,
    curve_type: str,
    params: dict[str, Any] | None,
) -> np.ndarray:
    """Dispatch to the named curve function, return output clamped to [0,1].

    Raises:
        ValueError: if curve_type is not registered.
    """
    fn = CURVE_REGISTRY.get(curve_type)
    if fn is None:
        raise ValueError(
            f"unknown curve type {curve_type!r}; " f"registered: {sorted(CURVE_REGISTRY)}"
        )
    out = fn(t_segment, **(params or {}))
    return np.clip(out, 0.0, 1.0)
