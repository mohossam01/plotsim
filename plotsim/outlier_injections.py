"""plotsim.outlier_injections — per-cell outlier ground truth.

What it does:
    Detects which (entity, period, metric) cells had ``noise.outlier_rate``
    fire during generation, so the manifest can record outlier ground truth
    for downstream anomaly-detection scoring without the consumer having
    to re-derive it from the noisy fact tables.

How it works:
    The detector re-runs ``generate_tables_with_state`` once with
    ``plotsim.metrics.apply_noise`` monkey-patched to a recording wrapper.
    The wrapper snapshots the engine RNG state immediately before each
    real ``apply_noise`` call, replays the snapshot in a side generator
    via ``plotsim.inspect._detect_noise_branches``, and emits an
    ``OutlierInjection`` record when the outlier branch fires.

    The wrapper relies on the deterministic call ordering of serial-mode
    generation: ``apply_noise`` is invoked exactly once per
    ``(entity, period, metric_in_toposort)`` cell, walked in
    entity-major / period-middle / metric-minor order. A monotonic
    counter on the wrapper maps each call back to its coordinates.

When it skips:
    Returns ``None`` (manifest field stays ``null``) in three cases:

      1. ``config.noise.outlier_rate == 0.0`` — the noise pipeline never
         consults the outlier branch, so re-running the engine just to
         observe zero firings is wasted work.
      2. The run resolves to vectorized generation. ``_apply_noise_batch``
         consumes RNG in a different order than per-cell ``apply_noise``,
         so a serial replay would record outliers at cells that don't
         match the actual fact table. Recording vectorized outliers
         requires either a parallel batch detector or invasive
         instrumentation; neither is in scope for 0.6-M5.
      3. Cell count exceeds ``OUTLIER_DETECTION_CELL_BUDGET`` (1M). The
         detector replays the full metric pipeline, so wall time scales
         with the cell count. Above the budget the cost-benefit tilts
         the wrong way for what's effectively a debug aid.

    Returns ``[]`` (empty list) when the detector ran and observed no
    outlier firings. Returns a populated list when at least one cell
    fired. The three states ``None`` / ``[]`` / non-empty are distinct
    on the wire.

Why monkey-patching:
    The maturity-doc note for this feature constrains the implementation
    to "non-invasive — no engine logic change". Adding a recorder
    callback to ``apply_noise`` would be cleaner code-wise but it changes
    a public engine surface. Monkey-patching at the module level
    achieves the same observation without touching engine source. The
    patch is strictly local to the detector's ``try/finally`` block, so
    other callers of ``apply_noise`` (most notably
    ``inspect.trace_metric_cell``) see no behavior change.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from plotsim.config import PlotsimConfig
from plotsim.inspect import _detect_noise_branches
from plotsim.manifest import OutlierInjection
from plotsim.metrics import _toposort_metrics


OUTLIER_DETECTION_CELL_BUDGET = 1_000_000


def detect_outlier_injections(config: PlotsimConfig) -> Optional[list[OutlierInjection]]:
    """Detect per-cell outlier firings for a config.

    Returns ``None`` when detection is skipped (zero outlier rate,
    vectorized mode, or cell budget exceeded), ``[]`` when detection
    ran and saw no firings, or a list of ``OutlierInjection`` records
    when at least one cell drew an outlier multiplier.

    See module docstring for the skip rules and the replay design.
    """
    if config.noise.outlier_rate <= 0.0:
        return None

    # Local imports keep this module cheap to import for the common
    # case (manifest builder calls us, but most runs never reach the
    # heavy work past the noise-rate gate above).
    from plotsim.tables import _resolve_generation_mode, generate_tables_with_state
    import plotsim.metrics as _metrics_mod

    if _resolve_generation_mode(config) != "serial":
        return None

    if not config.entities or not config.metrics:
        return None

    n_periods = _resolve_n_periods(config)
    cell_count = len(config.entities) * n_periods * len(config.metrics)
    if cell_count > OUTLIER_DETECTION_CELL_BUDGET:
        return None

    sorted_metrics = _toposort_metrics(list(config.metrics))
    metric_names = [m.name for m in sorted_metrics]
    entity_names = [e.name for e in config.entities]
    n_metrics = len(metric_names)
    cells_per_entity = n_periods * n_metrics

    records: list[OutlierInjection] = []
    call_counter = 0
    original_apply_noise = _metrics_mod.apply_noise

    def recording_apply_noise(
        value: float,
        noise: Any,
        rng: np.random.Generator,
        trajectory_position: Optional[float] = None,
    ) -> Optional[float]:
        nonlocal call_counter
        # Snapshot BEFORE the real call — _detect_noise_branches replays
        # the same RNG draws gaussian → outlier → MCAR using a side
        # generator seeded from this exact state, so capturing post-call
        # would advance the engine RNG and the snapshot would no longer
        # match what the noise pipeline saw.
        snapshot = rng.bit_generator.state
        # 0.6-M22: forward ``trajectory_position`` so the wrapper preserves
        # the engine's heteroscedastic-noise contract end-to-end. The
        # detection replay needs the same position to reproduce the
        # gaussian draw byte-for-byte and keep the side RNG aligned.
        result = original_apply_noise(value, noise, rng, trajectory_position=trajectory_position)
        if noise is not None and noise.outlier_rate > 0.0:
            outlier_fired, _ = _detect_noise_branches(
                value,
                noise,
                snapshot,
                trajectory_position=trajectory_position,
            )
            if outlier_fired:
                idx = call_counter
                entity_idx, rem = divmod(idx, cells_per_entity)
                period_idx, metric_idx = divmod(rem, n_metrics)
                if entity_idx < len(entity_names) and metric_idx < n_metrics:
                    records.append(
                        OutlierInjection(
                            entity=entity_names[entity_idx],
                            period_index=period_idx,
                            metric=metric_names[metric_idx],
                        )
                    )
        call_counter += 1
        return result

    # Force the replay to serial mode regardless of the original
    # ``generation_mode`` setting. The early-return above already
    # guaranteed the original resolves to serial, but ``"auto"`` is
    # archetype-batch-size-dependent — pinning to ``"serial"`` here
    # makes the replay's mode independent of any future change to
    # ``_resolve_generation_mode``'s threshold rule.
    config_for_replay = config.model_copy(update={"generation_mode": "serial"})
    rng = np.random.default_rng(int(config.seed))

    _metrics_mod.apply_noise = recording_apply_noise  # type: ignore[assignment]
    try:
        generate_tables_with_state(config_for_replay, rng)
    finally:
        _metrics_mod.apply_noise = original_apply_noise  # type: ignore[assignment]

    records.sort(key=lambda r: (r.entity, r.period_index, r.metric))
    return records


def _resolve_n_periods(config: PlotsimConfig) -> int:
    """Compute the run's period count without building dim_date.

    Mirrors ``compute_all_trajectories``'s n_periods derivation but
    avoids the dim-table build cost. The cell-count gate needs the
    period count BEFORE the replay runs — building dim_date here just
    to count rows would defeat the budget's purpose.
    """
    from plotsim.trajectory import compute_time_steps

    return int(len(compute_time_steps(config.time_window)))


__all__ = [
    "OUTLIER_DETECTION_CELL_BUDGET",
    "detect_outlier_injections",
]
