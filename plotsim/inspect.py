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
from collections.abc import Mapping
from typing import Any, Literal, Optional

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
    _build_seasonal_factors,
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
            → modulated_center      (M119: after seasonal modulation + clamp)
            → independent_draw      (raw distributional sample)
            → correlated_draw       (after Gaussian-copula transform)
            → noised_value          (after gaussian/outlier/MCAR)
            → clamped_value         (after value_range clamp + poisson round)
            → realized_cell         (as found in the generated fact table)

    A ``realized_cell == fct.<metric> at (entity, period) row`` equality check
    is the load-bearing assertion this dataclass exists to support.

    M119 fields:
      * ``seasonal_factor`` — effective multiplier at this cell:
        ``seasonal_global × metric.seasonal_sensitivity ×
        entity.seasonal_sensitivity``. ``0.0`` when no
        ``seasonal_effects`` are configured.
      * ``modulated_center`` — ``distribution_center × (1 + seasonal_factor)``,
        clamped to ``value_range``. Equals ``distribution_center`` when
        ``seasonal_factor == 0.0``. This is the value the distribution is
        actually sampled around.
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
    seasonal_factor: float
    modulated_center: float
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
        (m for m in config.metrics if m.name == metric_name),
        None,
    )
    if metric_in_config is None:
        raise MetricNotFound(
            f"metric {metric_name!r} not in config.metrics; available: "
            f"{[m.name for m in config.metrics]}"
        )

    effective_seed = config.seed if seed is None else seed

    # ---- Pass 1: run engine to get realized tables + ground-truth state. ----
    tables, state = generate_tables_with_state(
        config,
        np.random.default_rng(effective_seed),
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
    effective_metrics = [_apply_archetype_overrides(m, archetype) for m in sorted_metrics]
    target_eff_metric = next(em for em in effective_metrics if em.name == metric_name)

    polarity = target_eff_metric.polarity
    distribution_family = target_eff_metric.distribution
    causal_lag = target_eff_metric.causal_lag
    causal_lag_driver = causal_lag.driver if causal_lag is not None else None
    causal_lag_blend_weight = causal_lag.blend_weight if causal_lag is not None else None

    # ---- Pass 2: replay with a fresh RNG; capture intermediates. ----
    replay_rng = np.random.default_rng(effective_seed)

    # Mirror generate_tables_with_state's setup steps that consume RNG before
    # the entity-metric loop. Skip the PSD validation duplicate (Pass 1
    # already passed it; the same config will pass here too).
    dim_tables_replay = build_all_dimensions(config, replay_rng)
    trajectories_replay = compute_all_trajectories(config, n_periods)
    dim_tables_replay, _scd_state_replay = expand_scd_dims(
        config,
        dim_tables_replay,
        trajectories_replay,
    )

    # 0.6-M11: build the per-period Cholesky list so cells inside
    # ``correlation_phases`` windows replay against the right factor.
    # ``cholesky_L`` (baseline) is retained for diagnostic display only;
    # the actual per-period replay below indexes ``cholesky_by_period``.
    cholesky_by_period = _hoist_cholesky_by_period(config, n_periods)
    cholesky_L = cholesky_by_period[period_index] if cholesky_by_period else None

    # M119: pre-compute the global per-period seasonal-strength array. The
    # replay must thread the same factors that ``tables._compute_entity_metrics``
    # uses, otherwise the RNG-consuming sample around a (modulated) center
    # diverges from the engine and the §7 ``realized_cell`` equality breaks.
    seasonal_factors = _build_seasonal_factors(config, n_periods)

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
            cholesky_by_period=cholesky_by_period,
            seasonal_factors=seasonal_factors,
            entity_seasonal_sensitivity=prior.seasonal_sensitivity,
        )

    # Walk the target entity's period loop manually so we can capture the
    # intermediates at ``period_index``. Periods [0..period_index) are
    # consumed normally to keep RNG state and ``lag_buffer`` aligned with
    # the engine's path.
    target_traj = trajectories_replay[entity_name]
    lag_buffer: dict[str, list[float]] = {m.name: [] for m in sorted_metrics}
    correlations_arg = list(config.correlations) if config.correlations else None

    for t in range(period_index):
        pos = float(target_traj[t])
        seasonal_global_t = float(seasonal_factors[t]) if seasonal_factors is not None else 0.0
        # 0.6-M11: per-period factor selection; matches the engine's
        # phase-keyed dispatch so pre-target periods consume RNG against
        # the same Cholesky factor the engine used.
        cholesky_L_t = cholesky_by_period[t] if cholesky_by_period is not None else None
        generate_metrics_for_period(
            pos,
            sorted_metrics,
            correlations_arg,
            config.noise,
            lag_buffer,
            t,
            replay_rng,
            archetype=archetype,
            cholesky_L=cholesky_L_t,
            seasonal_global=seasonal_global_t,
            entity_seasonal_sensitivity=target_entity.seasonal_sensitivity,
        )

    # At ``period_index``, mirror generate_metrics_for_period's body manually
    # so we can read out the intermediates for the target metric without
    # touching engine internals beyond the public/private helpers we're
    # already authorized to call.
    pos_target = float(target_traj[period_index])
    seasonal_global_target = (
        float(seasonal_factors[period_index]) if seasonal_factors is not None else 0.0
    )
    entity_seasonal_sens = target_entity.seasonal_sensitivity
    centers: dict[str, float] = {}
    pre_modulation_centers: dict[str, float] = {}
    independent: dict[str, Optional[float]] = {}
    target_effective_position: Optional[float] = None
    target_seasonal_factor = 0.0
    correlations_active = bool(config.correlations)
    for em in effective_metrics:
        eff_pos = _compute_effective_position(
            pos_target,
            em,
            lag_buffer,
            period_index,
        )
        # Buffer write must happen inline before moving to the next metric —
        # mirrors generate_metrics_for_period, so multi-hop causal_lag chains
        # see the freshly-resolved driver value.
        lag_buffer[em.name].append(eff_pos)
        base_center = position_to_center(eff_pos, em)
        pre_modulation_centers[em.name] = base_center
        center = base_center
        # M119: mirror generate_metrics_for_period's seasonal modulation. We
        # apply it for every effective metric (not just the target) so the
        # sample_single_metric RNG draws happen around the same center the
        # engine sees, keeping replay RNG state aligned with generation.
        if seasonal_global_target != 0.0:
            effective_strength = (
                seasonal_global_target * em.seasonal_sensitivity * entity_seasonal_sens
            )
            if effective_strength != 0.0:
                center = base_center * (1.0 + effective_strength)
                vr = em.value_range
                if vr is not None:
                    if vr.min is not None and center < vr.min:
                        center = vr.min
                    if vr.max is not None and center > vr.max:
                        center = vr.max
        if em.name == metric_name:
            target_seasonal_factor = (
                seasonal_global_target * em.seasonal_sensitivity * entity_seasonal_sens
            )
        centers[em.name] = center
        # M127b: only the no-correlations path draws independent marginals
        # per metric. With correlations the new copula draws one batched
        # ``standard_normal(M)`` inside ``apply_correlations``; calling
        # ``sample_single_metric`` here would consume RNG twice and the
        # values would get discarded.
        if not correlations_active:
            independent[em.name] = sample_single_metric(center, em, replay_rng)
        else:
            independent[em.name] = None
        if em.name == metric_name:
            target_effective_position = eff_pos

    assert target_effective_position is not None  # toposort includes target
    target_distribution_center = pre_modulation_centers[metric_name]
    target_modulated_center = centers[metric_name]

    # Apply correlations. With correlations active the copula consumes RNG
    # (one ``standard_normal(M)`` draw); without correlations the per-metric
    # ``sample_single_metric`` draws above are the answer.
    if correlations_active:
        correlated = apply_correlations(
            independent,
            centers,
            list(config.correlations),
            effective_metrics,
            cholesky_L=cholesky_L,
            rng=replay_rng,
        )
        # M127b: the new copula draws its own Gaussians; there is no
        # separate per-metric "independent" sample distinct from the
        # correlated value. For dataclass-shape stability we surface the
        # same value as both ``independent_draw`` and ``correlated_draw``
        # — the value IS what the new pipeline produced for this cell's
        # marginal. Tests that previously distinguished the two on the
        # correlation-active path were updated for the version boundary.
        target_independent_draw = correlated[metric_name]
    else:
        correlated = dict(independent)
        target_independent_draw = independent[metric_name]
    target_correlated_draw = correlated[metric_name]

    # M127b: bypass machinery deleted. The copula now produces a finite
    # value for every (metric, center) pair, so the bypass-in-copula
    # concept no longer applies. The ``bypass_in_copula`` field is kept
    # on the dataclass for backward compat with code that reads it but
    # is now structurally False.
    bypass_in_copula = False

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
        noised = apply_noise(
            float(v),
            config.noise,
            replay_rng,
            trajectory_position=pos_target,
        )
        if em.name == metric_name:
            outlier_fired, mcar_fired = _detect_noise_branches(
                float(v),
                config.noise,
                rng_state_snapshot,
                trajectory_position=pos_target,
            )
            noised_value = None if noised is None else float(noised)
            if noised is None:
                clamped_value = None
            else:
                clamped_value = float(_clamp_and_round(float(noised), em))

    # Realized cell — the load-bearing field. Must equal the value in the
    # generated fact table at (entity, period) row, modulo MCAR-induced NaN.
    realized_cell = _resolve_realized_cell(
        config,
        tables,
        n_periods,
        entity_idx,
        period_index,
        metric_name,
    )

    assert target_independent_draw is not None  # set above for the target metric
    assert target_correlated_draw is not None
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
            float(causal_lag_blend_weight) if causal_lag_blend_weight is not None else None
        ),
        distribution_family=distribution_family,
        distribution_center=float(target_distribution_center),
        seasonal_factor=float(target_seasonal_factor),
        modulated_center=float(target_modulated_center),
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
    """Re-derive the engine's baseline Cholesky factor for replay.

    Mirrors ``generate_tables_with_state``'s baseline hoist exactly:
    assemble the correlation matrix in toposorted-metric order, optionally
    apply the M120 trajectory-aware pre-compensation (matching the
    engine's ``compensate_correlations`` flag and
    ``_MAX_METRICS_FOR_COMPENSATION`` cap), project to nearest-PD if
    needed, factor. The mirror is required so ``trace_metric_cell``
    reports the compensated coefficient the engine actually drove the
    cell against rather than the raw user target.

    0.6-M11: this helper returns the BASELINE factor only. For configs
    with ``correlation_phases`` set, use ``_hoist_cholesky_by_period``
    to get the per-period list; ``trace_metric_cell`` indexes that list
    by the traced period so traces against cells inside phase windows
    see the active phase's factor.
    """
    return _build_one_cholesky(config, list(config.correlations))


def _build_one_cholesky(
    config: PlotsimConfig,
    pairs: list,
) -> Optional[np.ndarray]:
    """Shared per-pair-list Cholesky build used by baseline + each phase."""
    if not pairs:
        return None
    sorted_metrics = _toposort_metrics(list(config.metrics))
    mat = _build_correlation_matrix(sorted_metrics, pairs)
    if config.compensate_correlations:
        from plotsim.metrics import (
            _MAX_METRICS_FOR_COMPENSATION,
            compensate_correlation_matrix,
            estimate_trajectory_covariance,
        )

        if len(sorted_metrics) <= _MAX_METRICS_FOR_COMPENSATION:
            traj_cov = estimate_trajectory_covariance(
                config,
                metric_order=sorted_metrics,
            )
            mat, _records = compensate_correlation_matrix(
                mat,
                traj_cov,
                sorted_metrics,
                pairs,
            )
    projected_mat, _used, _fallback = project_correlation_matrix(mat)
    return np.linalg.cholesky(projected_mat)


def _hoist_cholesky_by_period(
    config: PlotsimConfig,
    n_periods: int,
) -> Optional[list[np.ndarray]]:
    """0.6-M11: per-period Cholesky list mirroring the orchestrator.

    Returns ``None`` when the config has no correlations (matches
    ``generate_tables_with_state``'s ``cholesky_by_period = None``
    short-circuit). Otherwise builds one factor for the baseline plus
    one per declared phase, then expands to a length-``n_periods`` list
    via ``config.resolve_period_to_phase``. Same structure the engine
    threads through the generation pipeline; the trace replay reads
    ``cholesky_by_period[period_index]`` to recover the factor active
    at the traced cell.
    """
    if not config.correlations:
        return None
    L_baseline = _build_one_cholesky(config, list(config.correlations))
    assert L_baseline is not None  # non-empty correlations → non-None
    if not config.correlation_phases:
        return [L_baseline] * n_periods
    phase_factors: dict[int, np.ndarray] = {}
    for phase_idx, phase in enumerate(config.correlation_phases):
        if not phase.correlations:
            phase_factors[phase_idx] = L_baseline
            continue
        L_phase = _build_one_cholesky(config, list(phase.correlations))
        assert L_phase is not None  # non-empty phase.correlations → non-None
        phase_factors[phase_idx] = L_phase
    period_to_phase = config.resolve_period_to_phase()
    return [
        phase_factors[ph_idx] if ph_idx is not None else L_baseline for ph_idx in period_to_phase
    ]


def _detect_noise_branches(
    value: float,
    noise,
    rng_state_snapshot: Mapping[str, Any],
    trajectory_position: Optional[float] = None,
) -> tuple[bool, bool]:
    """Replay apply_noise's RNG draws to detect which branches fired.

    Returns ``(outlier_fired, mcar_fired)``. Uses a side ``np.random.Generator``
    initialized from a snapshot of the engine's RNG state taken just before
    the real ``apply_noise`` call. Mirrors apply_noise's RNG consumption
    order: gaussian → outlier check → optional uniform → mcar check.

    0.6-M22: when the engine ran with ``noise.scale_with_trajectory=True``,
    the gaussian branch must replay the same trajectory-scaled scale to keep
    the side RNG in lockstep — same number of bytes consumed, same value
    drawn. Callers must pass the same ``trajectory_position`` the engine
    saw at this cell.

    0.6-M23: when ``noise.noise_family`` is non-default, the replay must
    invoke the same family on the side generator so the post-jitter RNG
    state matches the engine's. Otherwise the subsequent ``random()`` calls
    for outlier and MCAR checks would read from a different byte position,
    yielding garbage outlier-injection records in the manifest.
    """
    side = np.random.default_rng()
    side.bit_generator.state = rng_state_snapshot
    v = value
    if noise.gaussian_sigma > 0.0:
        if getattr(noise, "scale_with_trajectory", False) and trajectory_position is not None:
            scale = noise.gaussian_sigma * float(trajectory_position)
        else:
            mag = abs(v) if v != 0.0 else 1.0
            scale = noise.gaussian_sigma * mag
        family = getattr(noise, "noise_family", "gaussian")
        if family == "gaussian":
            v = v + float(side.normal(loc=0.0, scale=scale))
        elif family == "student_t":
            df = float(noise.degrees_of_freedom)
            v = v + float(side.standard_t(df)) * scale
        else:  # "laplace"
            v = v + float(side.laplace(loc=0.0, scale=scale))
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
