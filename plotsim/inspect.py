"""plotsim.inspect — single-cell trace of the metric pipeline.

What it does:
    Reconstructs the full pipeline path for one ``(entity, period, metric)``
    triple — from the entity's trajectory position at that period through to
    the realized fact-table cell value. Captures every documented intermediate:
    polarity-flipped position, distribution center, independent draw,
    correlated draw (post-Gaussian-copula), noise, clamp/round, fact-table cell.

    This is the only sanctioned external consumer of ``plotsim.metrics``
    private internals. Notebooks and tests use ``trace_metric_cell`` as the
    ground-truth verifier of the trajectory-first invariant: every realized
    metric value must be reproducible from the trajectory position via the
    documented pipeline.

How it works:
    Re-executes generation deterministically. Given ``config`` and ``seed``,
    runs the engine once to capture the realized fact tables and the
    ``GenerationState`` (trajectories, SCD, bridges). Then runs a second
    replay with a fresh RNG seeded identically: mirrors the engine's setup
    (build_all_dimensions → compute_all_trajectories → expand_scd_dims →
    Cholesky hoist), consumes RNG draws for entities ``[0..target_idx)`` via
    ``generate_entity_metrics`` (matches the entity-major outer loop in
    ``tables._compute_entity_metrics``), and for the target entity walks the
    period loop manually — calling the engine's per-period helpers and
    capturing intermediate values at the target period.

    Replay must NOT reorder entities or skip periods. Doing either would
    desynchronize from the engine's RNG consumption, breaking the §7
    traceback assertion (``result.realized_cell == fct.<col> at row``).

Out of scope (v1):
    Trajectory-only traces (``trace_trajectory``), correlation-only traces,
    manifest-field traces, per-feature traces for SCD/bridges/events/stages/
    quality/holdout. ``trace_metric_cell`` is the entire v1 surface.

Future enhancement (deferred):
    Caching intermediates in ``GenerationState`` during generation so
    ``trace_metric_cell`` becomes O(1) lookup instead of re-execution.
    Acceptable v1 cost: re-execution is bounded by entity count and runs in
    seconds for the largest bundled template.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd

from plotsim.config import (
    MetricSource,
    PlotsimConfig,
    parse_source,
)
from plotsim.dimensions import build_all_dimensions
from plotsim.metrics import (
    _apply_archetype_overrides,
    _build_correlation_matrix,
    _clamp_and_round,
    _compute_effective_position,
    _get_scipy_dist,
    _toposort_metrics,
    apply_correlations,
    apply_noise,
    generate_entity_metrics,
    generate_metrics_for_period,
    position_to_center,
    project_correlation_matrix,
    sample_single_metric,
)
from plotsim.tables import (
    expand_scd_dims,
    generate_tables_with_state,
)
from plotsim.trajectory import compute_all_trajectories


__all__ = [
    "TraceResult",
    "trace_metric_cell",
    "EntityNotFound",
    "PeriodOutOfRange",
    "MetricNotFound",
]


class EntityNotFound(KeyError):
    """Raised when ``entity_name`` is not in ``config.entities``."""


class PeriodOutOfRange(IndexError):
    """Raised when ``period_index`` is outside ``[0, n_periods)``."""


class MetricNotFound(KeyError):
    """Raised when ``metric_name`` is not in ``config.metrics``."""


@dataclass(frozen=True)
class TraceResult:
    """Full pipeline trace for one ``(entity, period, metric)`` cell.

    Every field is the value at the corresponding pipeline stage:

        trajectory_position
            → effective_position    (after causal_lag blend, if configured)
            → distribution_center   (after polarity flip + dist-specific map)
            → independent_draw      (raw distributional sample)
            → correlated_draw       (after Gaussian-copula transform)
            → noised_value          (after gaussian/outlier/MCAR)
            → clamped_value         (after value_range clamp + poisson round)
            → realized_cell         (as found in the generated fact table)

    A ``realized_cell == fct.<metric> at (entity, period) row`` equality check
    is the load-bearing assertion this dataclass exists to support.
    """

    entity_name: str
    archetype_name: str
    period_index: int
    metric_name: str
    trajectory_position: float
    polarity: Literal["positive", "negative"]
    effective_position: float
    causal_lag_driver: Optional[str]
    causal_lag_blend_weight: Optional[float]
    distribution_family: str
    distribution_center: float
    independent_draw: float
    correlated_draw: float
    bypass_in_copula: bool
    noised_value: Optional[float]
    mcar_fired: bool
    outlier_fired: bool
    clamped_value: Optional[float]
    realized_cell: Optional[float]


def trace_metric_cell(
    config: PlotsimConfig,
    entity_name: str,
    period_index: int,
    metric_name: str,
    seed: Optional[int] = None,
) -> TraceResult:
    """Reconstruct the full pipeline path for one ``(entity, period, metric)``.

    Parameters
    ----------
    config
        Loaded ``PlotsimConfig``. Must declare the named entity and metric.
    entity_name
        Name of an entity in ``config.entities``.
    period_index
        Zero-based period index. Must be in ``[0, n_periods)``.
    metric_name
        Name of a metric in ``config.metrics``.
    seed
        Optional override for ``config.seed``. ``None`` reuses the config seed,
        which is what the engine uses for ``generate_tables_with_state``.

    Returns
    -------
    TraceResult
        Frozen dataclass populated with every documented intermediate.

    Raises
    ------
    EntityNotFound
    PeriodOutOfRange
    MetricNotFound

    Example
    -------
    >>> from plotsim import load_config
    >>> from plotsim.inspect import trace_metric_cell
    >>> cfg = load_config("plotsim/configs/sample_saas.yaml")
    >>> result = trace_metric_cell(cfg, "acme_corp_cohort", 12, "mrr")
    >>> 0.0 <= result.trajectory_position <= 1.0
    True
    >>> result.metric_name
    'mrr'
    """
    entity_idx = _resolve_entity_index(config, entity_name)
    target_entity = config.entities[entity_idx]

    metric_in_config = next(
        (m for m in config.metrics if m.name == metric_name), None,
    )
    if metric_in_config is None:
        raise MetricNotFound(
            f"metric {metric_name!r} not in config.metrics; available: "
            f"{[m.name for m in config.metrics]}"
        )

    effective_seed = config.seed if seed is None else seed

    # ---- Pass 1: run engine to get realized tables + ground-truth state. ----
    tables, state = generate_tables_with_state(
        config, np.random.default_rng(effective_seed),
    )
    n_periods = len(state.trajectories[entity_name])
    if period_index < 0 or period_index >= n_periods:
        raise PeriodOutOfRange(
            f"period_index {period_index} outside [0, {n_periods}); the engine "
            f"computed {n_periods} periods for this config"
        )

    trajectory_position = float(state.trajectories[entity_name][period_index])

    # Resolve the entity's archetype (post-override metric source).
    arch_by_name = {a.name: a for a in config.archetypes}
    archetype = arch_by_name.get(target_entity.archetype)
    archetype_name = target_entity.archetype

    # The engine toposorts metrics inside generate_entity_metrics so chains
    # compose; replay must use the same order or the lag_buffer / RNG order
    # diverge from the engine.
    sorted_metrics = _toposort_metrics(list(config.metrics))
    effective_metrics = [
        _apply_archetype_overrides(m, archetype) for m in sorted_metrics
    ]
    target_eff_metric = next(
        em for em in effective_metrics if em.name == metric_name
    )

    polarity = target_eff_metric.polarity
    distribution_family = target_eff_metric.distribution
    causal_lag = target_eff_metric.causal_lag
    causal_lag_driver = causal_lag.driver if causal_lag is not None else None
    causal_lag_blend_weight = (
        causal_lag.blend_weight if causal_lag is not None else None
    )

    # ---- Pass 2: replay with a fresh RNG; capture intermediates. ----
    replay_rng = np.random.default_rng(effective_seed)

    # Mirror generate_tables_with_state's setup steps that consume RNG before
    # the entity-metric loop. Skip the PSD validation duplicate (Pass 1
    # already passed it; the same config will pass here too).
    dim_tables_replay = build_all_dimensions(config, replay_rng)
    trajectories_replay = compute_all_trajectories(config, n_periods)
    dim_tables_replay, _scd_state_replay = expand_scd_dims(
        config, dim_tables_replay, trajectories_replay,
    )

    cholesky_L = _hoist_cholesky(config)

    # Consume RNG for entities [0..target_entity_idx) by running their full
    # entity-metric pipelines. Output is discarded — the goal is to advance
    # the RNG to the same state the engine reaches when it starts the target
    # entity.
    for prior in config.entities[:entity_idx]:
        prior_traj = trajectories_replay[prior.name]
        generate_entity_metrics(
            prior_traj,
            list(config.metrics),
            list(config.correlations),
            config.noise,
            replay_rng,
            archetype=arch_by_name.get(prior.archetype),
            cholesky_L=cholesky_L,
        )

    # Walk the target entity's period loop manually so we can capture the
    # intermediates at ``period_index``. Periods [0..period_index) are
    # consumed normally to keep RNG state and ``lag_buffer`` aligned with
    # the engine's path.
    target_traj = trajectories_replay[entity_name]
    lag_buffer: dict[str, list[float]] = {m.name: [] for m in sorted_metrics}
    correlations_arg = (
        list(config.correlations) if config.correlations else None
    )

    for t in range(period_index):
        pos = float(target_traj[t])
        generate_metrics_for_period(
            pos,
            sorted_metrics,
            correlations_arg,
            config.noise,
            lag_buffer,
            t,
            replay_rng,
            archetype=archetype,
            cholesky_L=cholesky_L,
        )

    # At ``period_index``, mirror generate_metrics_for_period's body manually
    # so we can read out the intermediates for the target metric without
    # touching engine internals beyond the public/private helpers we're
    # already authorized to call.
    pos_target = float(target_traj[period_index])
    centers: dict[str, float] = {}
    independent: dict[str, Optional[float]] = {}
    target_effective_position: Optional[float] = None
    for em in effective_metrics:
        eff_pos = _compute_effective_position(
            pos_target, em, lag_buffer, period_index,
        )
        # Buffer write must happen inline before moving to the next metric —
        # mirrors generate_metrics_for_period, so multi-hop causal_lag chains
        # see the freshly-resolved driver value.
        lag_buffer[em.name].append(eff_pos)
        center = position_to_center(eff_pos, em)
        centers[em.name] = center
        independent[em.name] = sample_single_metric(center, em, replay_rng)
        if em.name == metric_name:
            target_effective_position = eff_pos

    assert target_effective_position is not None  # toposort includes target
    target_distribution_center = centers[metric_name]
    target_independent_draw = independent[metric_name]

    # Apply correlations (deterministic — no RNG draws).
    if config.correlations:
        correlated = apply_correlations(
            independent,
            centers,
            list(config.correlations),
            effective_metrics,
            cholesky_L=cholesky_L,
        )
    else:
        correlated = dict(independent)
    target_correlated_draw = correlated[metric_name]

    # Bypass-in-copula: degenerate distribution at this center, OR no
    # correlation matrix configured at all (no copula step applies, so the
    # bypass concept doesn't either).
    target_dist_obj = _get_scipy_dist(
        target_eff_metric, target_distribution_center,
    )
    bypass_in_copula = bool(config.correlations) and (
        target_dist_obj is None or target_independent_draw is None
    )

    # Noise + clamp: walk all effective metrics in toposort order, mirroring
    # generate_metrics_for_period. Capture intermediates for the target
    # metric only; advance RNG for the other metrics so the replay's RNG
    # state stays in lockstep with the engine. apply_noise's public surface
    # doesn't expose which branch fired (gaussian / outlier / MCAR), so we
    # snapshot the RNG before each apply_noise call and replay it in a side
    # generator to introspect — see _detect_noise_branches.
    noised_value: Optional[float] = None
    clamped_value: Optional[float] = None
    mcar_fired = False
    outlier_fired = False

    for em in effective_metrics:
        v = correlated[em.name]
        if v is None:
            if em.name == metric_name:
                noised_value = None
                clamped_value = None
            continue
        if config.noise is None:
            v_after = float(v)
            if em.name == metric_name:
                noised_value = v_after
                clamped_value = float(_clamp_and_round(v_after, em))
            continue
        rng_state_snapshot = replay_rng.bit_generator.state
        noised = apply_noise(float(v), config.noise, replay_rng)
        if em.name == metric_name:
            outlier_fired, mcar_fired = _detect_noise_branches(
                float(v), config.noise, rng_state_snapshot,
            )
            noised_value = None if noised is None else float(noised)
            if noised is None:
                clamped_value = None
            else:
                clamped_value = float(
                    _clamp_and_round(float(noised), em)
                )

    # Realized cell — the load-bearing field. Must equal the value in the
    # generated fact table at (entity, period) row, modulo MCAR-induced NaN.
    realized_cell = _resolve_realized_cell(
        config, tables, n_periods, entity_idx, period_index, metric_name,
    )

    return TraceResult(
        entity_name=entity_name,
        archetype_name=archetype_name,
        period_index=period_index,
        metric_name=metric_name,
        trajectory_position=trajectory_position,
        polarity=polarity,
        effective_position=float(target_effective_position),
        causal_lag_driver=causal_lag_driver,
        causal_lag_blend_weight=(
            float(causal_lag_blend_weight)
            if causal_lag_blend_weight is not None
            else None
        ),
        distribution_family=distribution_family,
        distribution_center=float(target_distribution_center),
        independent_draw=float(target_independent_draw),
        correlated_draw=float(target_correlated_draw),
        bypass_in_copula=bypass_in_copula,
        noised_value=noised_value,
        mcar_fired=mcar_fired,
        outlier_fired=outlier_fired,
        clamped_value=clamped_value,
        realized_cell=realized_cell,
    )


# --- Helpers ---------------------------------------------------------------

def _resolve_entity_index(config: PlotsimConfig, entity_name: str) -> int:
    for i, e in enumerate(config.entities):
        if e.name == entity_name:
            return i
    raise EntityNotFound(
        f"entity {entity_name!r} not in config.entities; available: "
        f"{[e.name for e in config.entities]}"
    )


def _hoist_cholesky(config: PlotsimConfig) -> Optional[np.ndarray]:
    """Re-derive the engine's Cholesky factor for replay.

    Mirrors ``generate_tables_with_state``'s hoist exactly: assemble the
    correlation matrix in toposorted-metric order, project to nearest-PD if
    needed, factor.
    """
    if not config.correlations:
        return None
    sorted_metrics = _toposort_metrics(list(config.metrics))
    mat = _build_correlation_matrix(sorted_metrics, list(config.correlations))
    projected_mat, _used, _fallback = project_correlation_matrix(mat)
    return np.linalg.cholesky(projected_mat)


def _detect_noise_branches(
    value: float,
    noise,
    rng_state_snapshot: dict,
) -> tuple[bool, bool]:
    """Replay apply_noise's RNG draws to detect which branches fired.

    Returns ``(outlier_fired, mcar_fired)``. Uses a side ``np.random.Generator``
    initialized from a snapshot of the engine's RNG state taken just before
    the real ``apply_noise`` call. Mirrors apply_noise's RNG consumption
    order: gaussian → outlier check → optional uniform → mcar check.
    """
    side = np.random.default_rng()
    side.bit_generator.state = rng_state_snapshot
    v = value
    if noise.gaussian_sigma > 0.0:
        mag = abs(v) if v != 0.0 else 1.0
        v = v + float(side.normal(loc=0.0, scale=noise.gaussian_sigma * mag))
    outlier_fired = False
    if noise.outlier_rate > 0.0:
        if side.random() < noise.outlier_rate:
            outlier_fired = True
            sign = 1.0 if v >= 0.0 else -1.0
            mag = abs(v) if v != 0.0 else 1.0
            v = sign * float(side.uniform(mag * 3.0, mag * 10.0))
    mcar_fired = False
    if noise.mcar_rate > 0.0:
        if side.random() < noise.mcar_rate:
            mcar_fired = True
    return outlier_fired, mcar_fired


def _resolve_realized_cell(
    config: PlotsimConfig,
    tables: dict,
    n_periods: int,
    entity_idx: int,
    period_index: int,
    metric_name: str,
) -> Optional[float]:
    """Find the fact-table cell for ``(entity, period, metric)``.

    Walks ``config.tables`` looking for the first fact table that has a column
    sourced from ``metric:<metric_name>``. Row order in per_entity_per_period
    fact tables is entity-major, period-minor (asserted in §7 of the
    acceptance notebook), so the flat row index is
    ``entity_idx * n_periods + period_index``.

    Returns ``None`` when the cell is NaN (MCAR-nullified) or no fact table
    sources this metric.
    """
    for tbl in config.tables:
        if tbl.type != "fact":
            continue
        col_for_metric = None
        for col in tbl.columns:
            parsed = parse_source(col.source)
            if isinstance(parsed, MetricSource) and parsed.metric == metric_name:
                col_for_metric = col.name
                break
        if col_for_metric is None:
            continue

        df = tables.get(tbl.name)
        if df is None:
            continue

        if tbl.grain == "per_entity_per_period":
            flat_idx = entity_idx * n_periods + period_index
            if flat_idx >= len(df):
                return None
            cell = df.iloc[flat_idx][col_for_metric]
        elif tbl.grain == "per_period":
            # No entity axis; one row per period. Cell is uniform across
            # entities by construction.
            if period_index >= len(df):
                return None
            cell = df.iloc[period_index][col_for_metric]
        else:
            return None

        if cell is None:
            return None
        try:
            cell_f = float(cell)
        except (TypeError, ValueError):
            return None
        if pd.isna(cell_f):
            return None
        return cell_f

    return None
