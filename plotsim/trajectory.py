"""plotsim.trajectory — stitch archetype curve segments into master trajectories.

What it does:
    Given an Archetype (ordered list of CurveSegments covering [0, 1] with no
    gaps) and a period count, produces a length-n_periods array in [0, 1] by
    slicing the period axis into segment-sized chunks and evaluating each
    segment's curve on its local [0, 1] axis. Discontinuities at segment
    boundaries are preserved — the engine does not smooth them.

    Zero randomness lives here. This module is a pure function of its inputs.
    Metric generators (Mission 004) read trajectory[period] and shape values
    around that position. The trajectory is the single source of truth for
    "where is this entity at time t" across all metrics — that is the
    trajectory-first invariant.

Input:
    Archetype (curve_segments), n_periods, optional entity overrides dict.
    Or a full PlotsimConfig for all-entity batch computation.

Output:
    np.ndarray of shape (n_periods,), values in [0.0, 1.0]. For
    compute_all_trajectories: dict mapping entity.name → trajectory array.
"""

from __future__ import annotations

import calendar
from datetime import date, timedelta
from typing import Any, cast

import numpy as np

from plotsim.config import Archetype, PlotsimConfig, TimeWindow
from plotsim.curves import evaluate_segment


def _parse_yyyy_mm(s: str) -> tuple[int, int]:
    year, month = s.split("-", 1)
    return int(year), int(month)


def _month_range_inclusive(
    start_ym: tuple[int, int], end_ym: tuple[int, int]
) -> list[tuple[int, int]]:
    sy, sm = start_ym
    ey, em = end_ym
    total = (ey - sy) * 12 + (em - sm) + 1
    out: list[tuple[int, int]] = []
    for k in range(total):
        offset = sm - 1 + k
        out.append((sy + offset // 12, offset % 12 + 1))
    return out


def _window_date_range(time_window: TimeWindow) -> tuple[date, date]:
    sy, sm = _parse_yyyy_mm(time_window.start)
    ey, em = _parse_yyyy_mm(time_window.end)
    start = date(sy, sm, 1)
    end = date(ey, em, calendar.monthrange(ey, em)[1])
    return start, end


def compute_time_steps(time_window: TimeWindow) -> np.ndarray:
    """Generate ordered period labels for the configured granularity.

    monthly: ``YYYY-MM`` — one label per calendar month in [start, end].
    weekly:  ``YYYY-Www`` (ISO) — each ISO week that overlaps the window.
    daily:   ``YYYY-MM-DD`` — every day from the 1st of start month through
             the last day of end month.

    Returns:
        np.ndarray of dtype=object holding string labels.
    """
    start_ym = _parse_yyyy_mm(time_window.start)
    end_ym = _parse_yyyy_mm(time_window.end)

    if time_window.granularity == "monthly":
        labels = [f"{y:04d}-{m:02d}" for y, m in _month_range_inclusive(start_ym, end_ym)]
    elif time_window.granularity == "weekly":
        start_d, end_d = _window_date_range(time_window)
        labels = []
        seen: set[str] = set()
        d = start_d
        one_day = timedelta(days=1)
        while d <= end_d:
            iso_year, iso_week, _ = d.isocalendar()
            label = f"{iso_year:04d}-W{iso_week:02d}"
            if label not in seen:
                seen.add(label)
                labels.append(label)
            d += one_day
    elif time_window.granularity == "daily":
        start_d, end_d = _window_date_range(time_window)
        labels = []
        d = start_d
        one_day = timedelta(days=1)
        while d <= end_d:
            labels.append(d.isoformat())
            d += one_day
    else:  # pragma: no cover — Pydantic Literal blocks other values at load
        raise ValueError(f"unknown granularity {time_window.granularity!r}")

    return np.array(labels, dtype=object)


def _segment_boundaries(
    archetype: Archetype,
    n_periods: int,
    shift: float = 0.0,
) -> list[int]:
    """Return ``len(segments) + 1`` monotonic period indices covering [0, n_periods].

    Segment i occupies periods [boundaries[i], boundaries[i+1]). Intermediate
    boundaries are ``floor((end_pct + shift) * n_periods)``, clamped to
    [0, n_periods] and enforced monotonically non-decreasing so a shift that
    pushes one boundary past another collapses a segment to length 0 rather
    than producing negative-length slices. First boundary is always 0; last
    is always n_periods so the final segment absorbs any rounding remainder.
    """
    segs = archetype.curve_segments
    boundaries: list[int] = [0]
    for seg in segs[:-1]:
        shifted = seg.end_pct + shift
        if shifted < 0.0:
            shifted = 0.0
        elif shifted > 1.0:
            shifted = 1.0
        idx = int(np.floor(shifted * n_periods))
        if idx < boundaries[-1]:
            idx = boundaries[-1]
        if idx > n_periods:
            idx = n_periods
        boundaries.append(idx)
    boundaries.append(n_periods)
    return boundaries


def _resolve_shift(
    archetype: Archetype,
    n_periods: int,
    overrides: dict[str, Any] | None,
) -> float:
    if not overrides:
        return 0.0
    inflection = overrides.get("inflection_month")
    if inflection is None:
        return 0.0
    default_inflection_pct = archetype.curve_segments[0].end_pct
    return float(inflection) / n_periods - default_inflection_pct


def compute_trajectory(
    archetype: Archetype,
    n_periods: int,
    overrides: dict[str, Any] | None = None,
    start_period: int = 0,
) -> np.ndarray:
    """Stitch archetype.curve_segments into one length-n_periods array.

    Args:
        archetype: has ordered curve_segments covering [0, 1].
        n_periods: total number of time steps (>= 1).
        overrides: optional Entity.overrides dict. Recognised keys:
            ``inflection_month`` (int) — shifts segment boundaries so that the
            first-segment end (the archetype's canonical inflection) lands on
            the specified period index. Shifts are clamped to [0, 1]. For
            cold-start entities (``start_period > 0``) the inflection is
            interpreted relative to the entity's own active window.
        start_period: first period at which the entity is active (0 = present
            for the full window). Periods ``[0, start_period)`` are NaN-filled;
            the archetype's full curve plays out across the active window
            ``[start_period, n_periods)``. Default ``0`` preserves pre-M8a
            behaviour byte-for-byte.

    Returns:
        np.ndarray of shape (n_periods,). Active periods are in [0.0, 1.0];
        cold-start prefix periods are ``np.nan``.
    """
    if n_periods < 1:
        raise ValueError(f"n_periods must be >= 1, got {n_periods}")
    if start_period < 0:
        raise ValueError(f"start_period must be >= 0, got {start_period}")
    if start_period >= n_periods:
        raise ValueError(
            f"start_period ({start_period}) must be < n_periods ({n_periods}); "
            f"a fully-cold entity has no active periods"
        )

    active_n = n_periods - start_period
    shift = _resolve_shift(archetype, active_n, overrides)
    boundaries = _segment_boundaries(archetype, active_n, shift)
    active = np.zeros(active_n, dtype=float)

    for i, seg in enumerate(archetype.curve_segments):
        start_idx = boundaries[i]
        end_idx = boundaries[i + 1]
        length = end_idx - start_idx
        if length <= 0:
            continue
        # linspace(0, 1, 1) == [0.0]; single-period segments thus sample the
        # curve's start. Acceptable — segment-length-1 is degenerate anyway.
        t_local = np.linspace(0.0, 1.0, length)
        values = evaluate_segment(t_local, seg.curve, dict(seg.params))
        active[start_idx:end_idx] = values

    # evaluate_segment already clamps, but a belt-and-braces clip keeps the
    # output contract explicit at the module boundary.
    active = np.clip(active, 0.0, 1.0)

    if start_period == 0:
        return cast(np.ndarray, active)
    out = np.full(n_periods, np.nan, dtype=float)
    out[start_period:] = active
    return cast(np.ndarray, out)


def compute_all_trajectories(
    config: PlotsimConfig,
    n_periods: int,
) -> dict[str, np.ndarray]:
    """Compute one trajectory per entity in config, keyed by entity.name.

    Looks up each entity's archetype in config.archetypes, applies entity
    overrides, and returns the resulting arrays. Raises KeyError if an
    entity references an archetype not in config — this should never happen
    because PlotsimConfig cross-reference validation catches it, but we
    guard explicitly so the failure mode is locatable.
    """
    arch_by_name = {a.name: a for a in config.archetypes}
    out: dict[str, np.ndarray] = {}
    for entity in config.entities:
        archetype = arch_by_name.get(entity.archetype)
        if archetype is None:
            raise KeyError(
                f"entity {entity.name!r} references unknown archetype "
                f"{entity.archetype!r}; config validation should have caught this"
            )
        # F9 / 0.5: Entity.overrides is now Optional[EntityOverrides]
        # rather than a permissive dict. compute_trajectory's interface
        # stays dict-keyed for direct test callers — convert here.
        overrides_dict = entity.overrides.model_dump() if entity.overrides is not None else None
        out[entity.name] = compute_trajectory(
            archetype,
            n_periods,
            overrides_dict,
            start_period=entity.start_period,
        )
    return out
