"""Interpreter — translate ``UserInput`` into a generation-ready ``PlotsimConfig``.

The interpreter performs ten translation steps; the order matters because
schema translation in step 7 depends on metric/distribution decisions in
step 3 and the lookup tables built across steps 1–5.

  1. about + unit          → Domain
  2. window                → TimeWindow
  3. metrics               → list[Metric]   (recipes + range-conditional picks)
  4. segments              → list[Archetype] + list[Entity]
  5. connections           → list[CorrelationPair]
  6. lifecycle (optional)  → StageSequence (enforce_order=False)
  7. dimensions/facts/events → list[Table]; when empty, auto-generate
                                dim_date + dim_{unit} + fct_{unit}
  8. sub-entity dims (count > 1)        → ``count × Σ segment.count`` rows via
                                           generated:row_index source
  9. seed (secrets) + OutputConfig csv → PlotsimConfig wrapping the above
 10. PlotsimConfig validation runs as a safety net — if it raises, the
     interpreter has produced an inconsistent config and the bug is here,
     not in the user's input.

Errors raised here are interpreter bugs. User-facing validation already
ran inside ``UserInput.model_validate``.
"""

from __future__ import annotations

import secrets
from typing import Any, Optional

import numpy as np

from plotsim._types import is_dim_date_dtype
from plotsim.config import (
    Archetype,
    BridgeCardinality,
    BridgeMetric,
    BridgeTableConfig,
    CausalLag,
    Column,
    CorrelationPair,
    CorrelationPhase,
    Distribution,
    Domain,
    Dtype,
    Entity,
    EntityFeaturesConfig,
    Grain,
    HoldoutConfig,
    Metric,
    MetricOverride,
    MultiSourceConfig,
    NarrativeConfig,
    NoiseConfig,
    OutputConfig,
    PlotsimConfig,
    QualityConfig,
    QualityIssue,
    SCDType2Config,
    SeasonalEffect,
    SourceDeclaration,
    StageDefinition,
    StageSequence,
    Table,
    TimeWindow,
    ValueRange,
)

from .input import (
    BridgeColumnInput,
    ColumnInput,
    DimInput,
    EventInput,
    ExplicitArrival,
    FactInput,
    LifecycleInput,
    LinearArrival,
    MetricInput,
    SegmentInput,
    SourceInput,
    StepArrival,
    UniformArrival,
    UserInput,
)
from .parser import parse_archetype
from .recipes import (
    AMOUNT_BETA_PARAMS,
    AMOUNT_LOGNORM_LOC,
    AMOUNT_LOGNORM_RATIO_THRESHOLD,
    AMOUNT_LOGNORM_S,
    BASELINE_RECIPES,
    INDEX_DISTRIBUTION,
    INDEX_SIGMA_FRACTION,
    METRIC_RECIPES,
    RELATIONSHIP_RECIPES,
)


# Friction-item #5: auto-generated dim_{unit} faker mapping. Operator decision:
# known unit words pick a sensible faker; everything else falls back to
# faker.company. Users can override by declaring the schema explicitly.
UNIT_FAKER_MAP: dict[str, str] = {
    "company": "faker.company",
    "employee": "faker.name",
    "customer": "faker.name",
}


# ── Public entry point ──────────────────────────────────────────────────────


def interpret(user_input: UserInput) -> PlotsimConfig:
    """Translate ``UserInput`` into a ready-to-generate ``PlotsimConfig``.

    The PlotsimConfig is validated by Pydantic on construction; if that
    raises, the interpreter has produced an inconsistent config and the
    bug is here.
    """
    domain = _build_domain(user_input)
    time_window = _build_time_window(user_input)
    n_periods = time_window.period_count()

    metrics = _build_metrics(user_input)
    metric_by_name = {m.name: m for m in metrics}

    # 0.6-M8b: resolve the seed BEFORE entity expansion so segment
    # arrival distributions can draw deterministic per-entity
    # ``start_period`` values from a seed-derived RNG. The same integer
    # is then stored on ``PlotsimConfig.seed`` (line ~167 below) and
    # consumed by the engine at generation time — both phases see the
    # same input but instantiate independent RNGs, so the start_period
    # draws don't pollute the engine's RNG stream.
    seed = user_input.seed if user_input.seed is not None else secrets.randbelow(2**32)

    archetypes, entities = _build_archetypes_and_entities(
        user_input, n_periods, metric_by_name, seed
    )

    correlations = _build_correlations(user_input)
    correlation_phases = _build_correlation_phases(user_input)
    stages = _build_stages(user_input)

    # M117: ``segment.count`` columns translate to a PoolSource on the dim
    # table; the value_pool maps each expanded entity name to the original
    # cohort population value. After segment expansion the per-entity
    # ``Entity.size`` is always 1 in the builder path, so a previously
    # ``derived:size`` source would have emitted 1 for every row — losing
    # the original cohort size the column is meant to carry.
    segment_count_value_pool = _build_segment_count_value_pool(user_input)

    # M122: ``pool.{attr}`` columns map each expanded entity to the list of
    # values declared on its segment's ``attributes[attr]``. Built once and
    # threaded down through ``_build_tables`` → ``_translate_column``.
    attribute_value_pools = _build_attribute_value_pools(user_input)

    tables = _build_tables(
        user_input,
        metric_by_name,
        segment_count_value_pool,
        attribute_value_pools,
    )

    bridges = _translate_bridges(user_input, metric_by_name)
    quality = _translate_quality(user_input)
    holdout = _translate_holdout(user_input)
    entity_features = _translate_entity_features(user_input)

    seasonal_effects = _build_seasonal_effects(user_input)
    multi_source = _build_multi_source(user_input)

    # M124 / 0.6-M8b: ``seed`` was resolved above (before entity expansion)
    # so segment arrival distributions could draw deterministic
    # ``start_period`` values. Same integer flows into the engine config
    # below.

    # Output / noise / locale — three engine knobs the builder now
    # surfaces directly. None defaults preserve historical builder
    # behaviour byte-for-byte (csv to ``output/``, no noise, en_US faker).
    if user_input.output is not None:
        output_cfg = OutputConfig(
            format=user_input.output.format,
            directory=user_input.output.directory,
            cell_budget=user_input.output.cell_budget,
            denormalized=user_input.output.denormalized,
            partition_by=user_input.output.partition_by,
            sql_dialect=user_input.output.sql_dialect,
        )
    else:
        output_cfg = OutputConfig(format="csv", directory="output")

    if user_input.noise is not None:
        noise_cfg = NoiseConfig(
            gaussian_sigma=user_input.noise.gaussian_sigma,
            outlier_rate=user_input.noise.outlier_rate,
            mcar_rate=user_input.noise.mcar_rate,
            scale_with_trajectory=user_input.noise.scale_with_trajectory,
            noise_family=user_input.noise.noise_family,
            degrees_of_freedom=user_input.noise.degrees_of_freedom,
        )
    else:
        noise_cfg = NoiseConfig()

    return PlotsimConfig(
        domain=domain,
        time_window=time_window,
        seed=seed,
        metrics=metrics,
        archetypes=archetypes,
        entities=entities,
        tables=tables,
        bridges=bridges,
        quality=quality,
        holdout=holdout,
        entity_features=entity_features,
        correlations=correlations,
        correlation_phases=correlation_phases,
        stages=stages,
        seasonal_effects=seasonal_effects,
        noise=noise_cfg,
        locale=user_input.locale,
        # M120: pre-compensation is always on for builder-produced configs.
        # User-declared ``connections`` are table-wide intent ("satisfaction
        # opposes support_tickets"), and the trajectory's structural
        # covariance otherwise washes out the copula's signal at mixed-
        # archetype scale. Engine-direct configs default to ``False`` for
        # backwards compatibility; the builder layer flips that contract
        # because the ``connections`` vocabulary explicitly promises
        # table-wide visibility.
        compensate_correlations=True,
        # M121: builder-produced configs target the dual-path "auto"
        # selector — the per-segment expansion (M117) easily produces
        # multi-hundred-entity configs where vectorization is a clear
        # speedup. Below the auto threshold (50 entities) the resolver
        # falls back to ``serial`` so small interactive previews keep
        # the byte-identical-to-pre-M121 baseline. Engine-direct configs
        # stay on ``serial`` by default so bundled templates round-trip
        # byte-identically on disk.
        generation_mode="auto",
        output=output_cfg,
        multi_source=multi_source,
    )


def _build_multi_source(user_input: UserInput) -> Optional[MultiSourceConfig]:
    """0.6-M13: translate builder ``sources`` → engine ``MultiSourceConfig``.

    Empty ``UserInput.sources`` → ``None`` (the engine default — dim
    builder skips the per-source pass). Non-empty list → one
    ``SourceDeclaration`` per ``SourceInput`` with the four fields passed
    through 1:1. The engine validators on ``MultiSourceConfig`` and
    ``PlotsimConfig._multi_source_requires_per_entity_dim`` run on the
    constructed config and surface any remaining issues (e.g. emission
    collisions against existing dim table names).
    """
    if not user_input.sources:
        return None
    return MultiSourceConfig(
        sources=[_translate_source(s) for s in user_input.sources],
    )


def _translate_source(s: SourceInput) -> SourceDeclaration:
    """1:1 builder ``SourceInput`` → engine ``SourceDeclaration``.

    Field shapes match exactly; the split exists so the builder layer can
    raise its own input-vocabulary errors before the engine sees the
    translated config.
    """
    return SourceDeclaration(
        name=s.name,
        key_scheme=s.key_scheme,
        name_drift_rate=s.name_drift_rate,
        attribute_drift_rate=s.attribute_drift_rate,
    )


def _build_seasonal_effects(user_input: UserInput) -> list[SeasonalEffect]:
    """Translate ``UserInput.seasonality`` into engine ``SeasonalEffect`` list.

    1:1 translation — months and strength pass through unchanged. The
    builder's ``SeasonalEffectInput`` and the engine's ``SeasonalEffect``
    have identical shapes; the split exists so the builder layer can
    enforce its own validation messages without import-cycling on the
    engine layer's typed model.
    """
    return [
        SeasonalEffect(months=tuple(eff.months), strength=eff.strength)
        for eff in user_input.seasonality
    ]


# ── Step 1 + 2: domain, time window ─────────────────────────────────────────


def _build_domain(user_input: UserInput) -> Domain:
    return Domain(
        name=user_input.about,
        description=user_input.about,
        entity_type=user_input.unit,
        entity_label=_pluralise(user_input.unit).title(),
    )


def _pluralise(word: str) -> str:
    """Naive English pluralisation good enough for default labels."""
    if word.endswith("y") and not word.endswith(("ay", "ey", "iy", "oy", "uy")):
        return word[:-1] + "ies"
    if word.endswith(("s", "x", "z", "ch", "sh")):
        return word + "es"
    return word + "s"


def _build_time_window(user_input: UserInput) -> TimeWindow:
    # WindowInput.start/end are already YYYY-MM strings post-validation.
    return TimeWindow(
        start=user_input.window.start,
        end=user_input.window.end,
        granularity=user_input.window.every,
    )


# ── Step 3: metrics ─────────────────────────────────────────────────────────


def _build_metrics(user_input: UserInput) -> list[Metric]:
    return [_metric_from_input(m) for m in user_input.metrics]


def _metric_from_input(m: MetricInput) -> Metric:
    distribution, params = _pick_distribution(m)
    value_range = _value_range_from_input(m)

    causal_lag: Optional[CausalLag] = None
    if m.follows is not None and m.delay is not None:
        # 0.6-M9b: presence of ``decay_window`` flips engine ``decay``
        # to True. ``decay_kernel`` is always passed through; the engine
        # ignores it when decay is False.
        causal_lag = CausalLag(
            driver=m.follows,
            lag_periods=m.delay,
            decay=m.decay_window is not None,
            decay_window=m.decay_window,
            decay_kernel=m.decay_kernel,
        )

    return Metric(
        name=m.name,
        label=m.label or m.name.replace("_", " ").title(),
        distribution=distribution,
        params=params,
        polarity=m.polarity,
        value_range=value_range,
        causal_lag=causal_lag,
        seasonal_sensitivity=m.seasonal_sensitivity,
    )


def _pick_distribution(m: MetricInput) -> tuple[Distribution, dict[str, float]]:
    """Pick distribution + params for a metric.

    Precedence (mission 0.6-M6):

      1. **Explicit** — ``MetricInput.distribution`` + ``distribution_params``.
         When set, bypass auto-pick entirely. Per-family param validation
         already ran on ``MetricInput`` construction.
      2. **Range-inferred** — ``amount`` picks lognorm vs beta from the
         declared range; ``index`` picks normal centered on the midpoint.
      3. **Type-default** — ``METRIC_RECIPES`` for ``score`` and ``count``.
    """
    if m.distribution is not None:
        params = dict(m.distribution_params) if m.distribution_params else {}
        return m.distribution, params

    if m.type in METRIC_RECIPES:
        recipe = METRIC_RECIPES[m.type]
        # Cast to ``dict[str, float]`` — Metric.params is typed that way and
        # poisson's ``lambda`` is already a float in our recipes.
        return recipe["distribution"], dict(recipe["params"])

    if m.type == "index":
        assert m.range is not None  # MetricInput validator already enforced
        lo, hi = m.range
        mu = (lo + hi) / 2.0
        sigma = (hi - lo) * INDEX_SIGMA_FRACTION
        return INDEX_DISTRIBUTION, {"mu": mu, "sigma": sigma}

    if m.type == "amount":
        assert m.range is not None
        lo, hi = m.range
        if lo == 0.0 or (lo > 0.0 and (hi / lo) >= AMOUNT_LOGNORM_RATIO_THRESHOLD):
            scale = (lo + hi) / 2.0 if lo > 0.0 else hi / 2.0
            return "lognorm", {
                "s": AMOUNT_LOGNORM_S,
                "loc": AMOUNT_LOGNORM_LOC,
                "scale": scale,
            }
        return "beta", dict(AMOUNT_BETA_PARAMS)

    raise AssertionError(f"unhandled metric type {m.type!r}")  # MetricInput validator gates


def _value_range_from_input(m: MetricInput) -> Optional[ValueRange]:
    """Build a ValueRange from the metric input.

    ``score`` defaults to [0, 1]. ``count`` has no value_range. ``amount``
    and ``index`` use the user-declared range verbatim.
    """
    if m.type == "count":
        return None
    if m.range is not None:
        lo, hi = m.range
        return ValueRange(min=float(lo), max=float(hi))
    if m.type == "score":
        return ValueRange(min=0.0, max=1.0)
    return None


# ── Step 4: archetypes + entities ───────────────────────────────────────────


def _build_archetypes_and_entities(
    user_input: UserInput,
    n_periods: int,
    metric_by_name: dict[str, Metric],
    seed: int,
) -> tuple[list[Archetype], list[Entity]]:
    """One archetype per segment, ``segment.count`` entities per segment.

    Each segment with ``count: N`` produces N individual ``Entity(size=1)``
    objects named ``{segment_name}_{i:04d}``, all sharing the segment's
    archetype. ``Entity.size`` remains the engine-direct sub-entity dim
    multiplier (untouched by this expansion); ``Table.count`` carries that
    role in the builder path.

    Baselines on the segment translate into per-archetype
    ``MetricOverride.value_range`` — applied once per archetype, shared
    by all expanded entities of that archetype.

    0.6-M8b: when a segment declares ``arrival: <distribution>``, every
    expanded entity gets a per-entity ``start_period`` drawn from that
    distribution. Draws are deterministic — a single ``np.random.default_rng(seed)``
    walks segments in declaration order, so the same ``UserInput`` plus
    the same seed produces the same per-entity arrival schedule. Step
    and explicit shapes are deterministic by construction (no RNG draw).
    Segments without ``arrival`` keep ``start_period=0`` for every
    entity, preserving pre-M8b output.
    """
    archetypes: list[Archetype] = []
    entities: list[Entity] = []

    arrival_rng = np.random.default_rng(seed)
    # 0.6-M8c: independent RNG for treatment assignments. The salt
    # decouples treatment_rng from arrival_rng so changing a segment's
    # ``arrival`` shape (which consumes RNG state on arrival_rng) does
    # NOT shift which entities land in the treatment arm. Same seed +
    # same treatment config + arbitrary arrival changes → identical
    # treatment assignments. Pinned by
    # ``test_treatment_assignments_independent_of_arrival_shape`` in
    # ``tests/test_treatment_control.py``.
    treatment_rng = np.random.default_rng(seed ^ TREATMENT_SALT)

    for s in user_input.segments:
        curve_segments = parse_archetype(s.archetype, n_periods=n_periods)
        metric_overrides = _baseline_to_overrides(s, metric_by_name)
        archetypes.append(
            Archetype(
                name=s.name,
                label=s.label or s.name.replace("_", " ").title(),
                description=s.label or s.archetype,
                curve_segments=curve_segments,
                metric_overrides=metric_overrides,
            )
        )
        start_periods = _draw_segment_arrivals(s, n_periods, arrival_rng)
        treatment_assignments = _draw_treatment_assignments(s, treatment_rng)
        for i in range(s.count):
            ta = treatment_assignments[i]
            entities.append(
                Entity(
                    name=f"{s.name}_{i:04d}",
                    archetype=s.name,
                    size=1,
                    seasonal_sensitivity=s.seasonal_sensitivity,
                    start_period=start_periods[i],
                    treatment_group=ta[0] if ta is not None else None,
                    treatment_lift_log_odds=ta[1] if ta is not None else None,
                    treatment_start_period=(
                        s.treatment.start_period if s.treatment is not None else 0
                    ),
                    treatment_target_metric=(
                        s.treatment.target_metric if s.treatment is not None else None
                    ),
                )
            )

    return archetypes, entities


# 0.6-M8c: salt for the treatment-assignment RNG stream. Decouples
# treatment draws from arrival draws so changing one feature's shape
# doesn't shift the other's outputs. The exact value is arbitrary —
# what matters is that it's distinct from 0 (so ``seed ^ SALT != seed``)
# and stable across releases (changing the salt shifts every
# treatment assignment, which would break determinism for users on
# pinned seeds). Hex digits of "M8C" → 0x4D3843.
TREATMENT_SALT = 0x4D3843


def _draw_treatment_assignments(
    s: SegmentInput,
    rng: np.random.Generator,
) -> list[Optional[tuple[str, Optional[float]]]]:
    """Return ``s.count`` per-entity ``(label, lift)`` tuples.

    Each entry is either:

      * ``None`` — the segment has no ``treatment`` config; every entity
        in this segment is treatment-free (pre-M8c behaviour).
      * ``(treatment_label, lift_log_odds)`` — the entity is in the
        treatment arm. ``lift_log_odds`` is the segment's configured lift.
      * ``(control_label, None)`` — the entity is in the control arm.
        Lift is None (the engine reads this as "no shift").

    Deterministic: ``rng.choice(s.count, size=n_treatment, replace=False)``
    gives a fixed permutation under the same seed. RNG consumption is
    one batched draw per segment with ``treatment`` set; segments
    without ``treatment`` consume zero — so toggling treatment on /
    off for one segment doesn't shift treatment draws on subsequent
    segments.
    """
    if s.treatment is None:
        return [None] * s.count

    n_treatment = int(round(s.treatment.fraction * s.count))
    # Edge cases: fraction=0.0 → empty treated set, every entity is
    # control. fraction=1.0 → entire segment is treatment. Both flow
    # through ``rng.choice(N, size=k, replace=False)`` cleanly because
    # ``size=0`` returns an empty array and ``size=N`` is a full
    # permutation. We still draw to keep RNG consumption uniform across
    # the fraction range — guards against a future user passing a
    # fraction-driven sweep where every cell expects identical RNG
    # state.
    if n_treatment == 0:
        # ``size=0`` would short-circuit; force the draw for RNG state
        # uniformity across fraction values.
        treated_indices: set[int] = set()
        rng.choice(s.count, size=0, replace=False)
    else:
        treated_indices = set(int(x) for x in rng.choice(s.count, size=n_treatment, replace=False))

    out: list[Optional[tuple[str, Optional[float]]]] = []
    for i in range(s.count):
        if i in treated_indices:
            out.append((s.treatment.treatment_label, s.treatment.lift_log_odds))
        else:
            out.append((s.treatment.control_label, None))
    return out


def _draw_segment_arrivals(
    s: SegmentInput,
    n_periods: int,
    rng: np.random.Generator,
) -> list[int]:
    """Return ``s.count`` per-entity ``start_period`` values for one segment.

    Dispatches on ``s.arrival.kind``:

      * ``None`` — every entity at period 0 (pre-M8b behaviour, no RNG draw).
      * ``uniform`` — ``rng.integers(start, end, size=count)``.
      * ``linear`` — triangular CDF inversion via ``rng.random(size=count)``;
        ``increasing`` direction back-loads, ``decreasing`` front-loads.
      * ``step`` — deterministic block expansion with rounding remainder
        absorbed by the last block; no RNG draw.
      * ``explicit`` — pass-through of the user-supplied list; no RNG draw.

    ``end=None`` defaults to ``n_periods - MIN_ACTIVE_PERIODS`` so every
    drawn entity has at least ``MIN_ACTIVE_PERIODS`` active periods. The
    config-load validator (``validate_cold_start_active_periods``) is
    the durable contract — this default just makes the common case
    "spread arrivals across the window" work without the user computing
    bounds manually.

    RNG consumption: ``uniform`` and ``linear`` each consume exactly
    one batched ``size=count`` draw per segment. ``step`` and
    ``explicit`` consume zero. So a config that adds an ``explicit``
    segment between two ``uniform`` segments doesn't shift the RNG
    stream the ``uniform`` segments observe.
    """
    arrival = s.arrival
    if arrival is None:
        return [0] * s.count

    # MIN_ACTIVE_PERIODS lives on validation.py — local import to dodge
    # the circular reference (validation.py imports from config which
    # imports from builder). Same workaround the rest of this module
    # uses for plotsim.validation.
    from plotsim.validation import MIN_ACTIVE_PERIODS

    default_end = max(1, n_periods - MIN_ACTIVE_PERIODS)

    if isinstance(arrival, UniformArrival):
        end = arrival.end if arrival.end is not None else default_end
        if end <= arrival.start:
            raise ValueError(
                f"segment {s.name!r}: uniform arrival end ({end}) must be "
                f"> start ({arrival.start}); the validator catches the "
                f"degenerate case where every drawn start_period would "
                f"violate MIN_ACTIVE_PERIODS"
            )
        draws = rng.integers(low=arrival.start, high=end, size=s.count)
        return [int(x) for x in draws]

    if isinstance(arrival, LinearArrival):
        end = arrival.end if arrival.end is not None else default_end
        if end <= arrival.start:
            raise ValueError(
                f"segment {s.name!r}: linear arrival end ({end}) must be > start ({arrival.start})"
            )
        u = rng.random(size=s.count)
        span = end - arrival.start
        if arrival.direction == "increasing":
            # Back-loaded ramp: density rises with period. CDF^-1 of the
            # triangular distribution on [start, end) with peak at end is
            # ``start + span * sqrt(u)``.
            offsets = np.floor(span * np.sqrt(u)).astype(np.int64)
        else:
            # Front-loaded ramp: density falls with period. CDF^-1 with
            # peak at start is ``start + span * (1 - sqrt(1 - u))``.
            offsets = np.floor(span * (1.0 - np.sqrt(1.0 - u))).astype(np.int64)
        # Clamp the rare ``floor`` result of ``span * 1.0`` (when
        # ``u`` rounds to exactly 1.0 in float64) back into ``[start, end)``.
        offsets = np.clip(offsets, 0, span - 1)
        return [int(arrival.start + o) for o in offsets]

    if isinstance(arrival, StepArrival):
        # Deterministic by construction. Allocate counts via
        # ``round(fraction * count)`` per block; the last block absorbs
        # rounding remainder so the total exactly equals ``count``.
        counts: list[int] = []
        running = 0
        for j, block in enumerate(arrival.blocks):
            if j == len(arrival.blocks) - 1:
                counts.append(s.count - running)
            else:
                c = int(round(block.fraction * s.count))
                counts.append(c)
                running += c
        out: list[int] = []
        for block, c in zip(arrival.blocks, counts):
            out.extend([block.period] * c)
        return out

    if isinstance(arrival, ExplicitArrival):
        # Length-vs-count match was enforced by the SegmentInput
        # validator; return a defensive copy so a mutating consumer
        # can't reach back into the immutable model.
        return list(arrival.start_periods)

    # mypy reachability: the discriminated union covers all four kinds.
    raise ValueError(f"segment {s.name!r}: unknown arrival kind {arrival!r}")


def _build_segment_count_value_pool(user_input: UserInput) -> dict[str, list[str]]:
    """Map every expanded entity name to its segment's original ``count``.

    ``segment.count`` column types resolve to a PoolSource on the dim.
    PoolSource value_pool keys must cover every entity that produces
    a row in the per_entity dim (``validate_value_pool_coverage``); after
    expansion that means one key per individual entity in
    ``config.entities``. The pool list is a single-element list per entity
    because the value is constant within a segment — every expanded entity
    from the same segment shares the original ``count``. Pool sampling
    consumes one RNG draw per row regardless of pool length, so the
    one-element pool keeps the resolution deterministic without polluting
    the RNG stream beyond what the existing PoolSource path already does.
    """
    pool: dict[str, list[str]] = {}
    for s in user_input.segments:
        value = [str(s.count)]
        for i in range(s.count):
            pool[f"{s.name}_{i:04d}"] = value
    return pool


def _baseline_to_overrides(
    s: SegmentInput,
    metric_by_name: dict[str, Metric],
) -> dict[str, MetricOverride]:
    """Translate a segment's baseline labels into per-metric value_range overrides.

    A segment with ``baseline: {mrr: high}`` restricts mrr's value_range
    for entities in this archetype to the upper third of the metric's
    global range. Count metrics (no value_range on the parent) are
    skipped — there's no meaningful sub-range to restrict.
    """
    overrides: dict[str, MetricOverride] = {}
    for metric_name, baseline_word in s.baseline.items():
        metric = metric_by_name.get(metric_name)
        if metric is None or metric.value_range is None:
            continue  # UserInput validator caught the orphan; count metrics skipped
        vmin = metric.value_range.min
        vmax = metric.value_range.max
        if vmin is None or vmax is None:
            continue
        lo_frac, hi_frac = BASELINE_RECIPES[baseline_word]
        span = vmax - vmin
        overrides[metric_name] = MetricOverride(
            value_range=ValueRange(
                min=vmin + lo_frac * span,
                max=vmin + hi_frac * span,
            )
        )
    return overrides


# ── Step 5: correlations ────────────────────────────────────────────────────


def _build_correlations(user_input: UserInput) -> list[CorrelationPair]:
    """Skip ``independent`` / ``0.0`` entries — the engine warns on
    explicit-zero pairs (RedundantCorrelationWarning) and unlisted pairs
    already get zero off-diagonal.

    Each connection carries either a relationship word (looked up in
    ``RELATIONSHIP_RECIPES``) or an explicit ``coefficient`` in
    ``[-1.0, 1.0]``; the input model enforces that exactly one is set.
    """
    return _connections_to_pairs(user_input.connections)


def _connections_to_pairs(connections: list) -> list[CorrelationPair]:
    """Shared translation: ``ConnectionInput`` list → ``CorrelationPair`` list.

    Used by both baseline ``_build_correlations`` and per-phase
    ``_build_correlation_phases``. Skips zero-coefficient entries (the
    engine already treats unlisted pairs as zero off-diagonal, so an
    explicit zero would just emit a ``RedundantCorrelationWarning``).
    """
    pairs: list[CorrelationPair] = []
    for c in connections:
        if c.coefficient is not None:
            coef = c.coefficient
        else:
            assert c.relationship is not None  # input model: exactly one of coef/relationship
            coef = RELATIONSHIP_RECIPES[c.relationship]
        if coef == 0.0:
            continue
        pairs.append(
            CorrelationPair(
                metric_a=c.metric_a,
                metric_b=c.metric_b,
                coefficient=coef,
            )
        )
    return pairs


def _build_correlation_phases(user_input: UserInput) -> list[CorrelationPhase]:
    """0.6-M11: translate builder ``connection_phases`` → engine ``correlation_phases``.

    Each ``ConnectionPhase`` carries its own ``connections`` list using
    the same relationship-word / explicit-coefficient vocabulary as the
    top-level ``connections`` field. Per-phase entries flow through
    ``_connections_to_pairs`` so the zero-skip rule applies uniformly.

    An empty ``connection_phases`` list translates to an empty engine
    ``correlation_phases`` list — the engine falls back to the
    single-Cholesky path, byte-identical to pre-M11.
    """
    return [
        CorrelationPhase(
            start_period=phase.start_period,
            end_period=phase.end_period,
            correlations=_connections_to_pairs(phase.connections),
        )
        for phase in user_input.connection_phases
    ]


# ── Step 6: lifecycle → StageSequence ───────────────────────────────────────


def _build_stages(user_input: UserInput) -> Optional[StageSequence]:
    if user_input.lifecycle is None:
        return None
    return _stage_sequence_from_lifecycle(user_input.lifecycle)


def _stage_sequence_from_lifecycle(lc: LifecycleInput) -> StageSequence:
    """Build a legacy-mode StageSequence (threshold_exit > threshold_enter).

    Each stage's ``threshold_exit`` is the next stage's ``threshold_enter``;
    the terminal stage's exit is None. ``enforce_order`` defaults to False
    (free-mode stages); irreversible lifecycle transitions are SCD Type 2's
    job, so stages reflect *current* lifecycle state. Set
    ``LifecycleInput.enforce_order = True`` to opt into the engine's
    monotonic stage walk; ``downgrade_delay`` then enables the
    hysteresis demote path.
    """
    sequence: list[StageDefinition] = []
    n = len(lc.stages)
    for i, stage in enumerate(lc.stages):
        if i == n - 1:
            exit_threshold: Optional[float] = None
        else:
            exit_threshold = lc.stages[i + 1].threshold
        sequence.append(
            StageDefinition(
                name=stage.name,
                threshold_enter=stage.threshold,
                threshold_exit=exit_threshold,
            )
        )
    return StageSequence(
        field=lc.track,
        sequence=sequence,
        enforce_order=lc.enforce_order,
        downgrade_delay=lc.downgrade_delay,
    )


# ── Step 7 + 8: schema (dimensions, facts, events) ──────────────────────────


def _build_tables(
    user_input: UserInput,
    metric_by_name: dict[str, Metric],
    segment_count_value_pool: dict[str, list[str]],
    attribute_value_pools: dict[str, dict[str, list[str]]],
) -> list[Table]:
    """Translate the schema section, or auto-generate when none provided.

    When dimensions/facts/events are all empty, generate a minimal viable
    schema: dim_date (per_period), dim_{unit} (per_entity), fct_{unit}
    (per_entity_per_period) carrying every metric.

    ``segment_count_value_pool`` threads the cohort-population pool
    through to ``_translate_column`` so ``segment.count`` columns can
    resolve to a PoolSource keyed by expanded entity names.

    ``attribute_value_pools`` is the per-attribute value_pool keyed by
    attribute name; ``pool.{attr}`` columns resolve to a PoolSource
    using the value list at that key.
    """
    if not user_input.dimensions and not user_input.facts and not user_input.events:
        return _auto_generate_schema(
            user_input,
            metric_by_name,
            attribute_value_pools,
        )

    # Build a dim PK lookup for FK resolution. dim_date is special-cased
    # because it might not appear in user_input.dimensions explicitly.
    # 0.6-M18: fact-table PKs are also seeded here so ``ref.fct_<parent>``
    # on a per_parent_row child resolves cleanly. Fact PK convention
    # mirrors ``_translate_fact``'s child / variable branches: the
    # first ``id``-typed column, falling back to the first column.
    dim_pk: dict[str, str] = {"dim_date": "date_key"}
    for d in user_input.dimensions:
        dim_pk[d.name] = _dim_primary_key(d)
    for f in user_input.facts:
        # Only variable / per_parent_row facts expose a single-column
        # PK that an FK can reference. The per_entity_per_period
        # default uses a composite (date_key, entity_id) PK that
        # children can't FK to, so we skip it. Per-parent-row children
        # don't get referenced themselves (M18 caps at one level), but
        # registering their PK is harmless and keeps the dict uniform.
        if f.row_count_driver is not None or f.parent_table is not None:
            pk_candidates = [c.name for c in f.columns if c.type == "id"]
            dim_pk[f.name] = pk_candidates[0] if pk_candidates else f.columns[0].name

    # Reference dims (``reference: true``) are static lookups; their FK
    # columns on a fact table do not contribute to the fact's PK because
    # they don't expand the per-entity-per-period grain.
    reference_dims: set[str] = {d.name for d in user_input.dimensions if d.reference}

    # Build a "metric → fact_table" map for SCD trigger_metric resolution.
    metric_to_fact: dict[str, str] = {}
    for f in user_input.facts:
        for col in f.columns:
            if col.type.startswith("metric."):
                metric_name = col.type.split(".", 1)[1]
                metric_to_fact.setdefault(metric_name, f.name)

    tables: list[Table] = []
    for d in user_input.dimensions:
        tables.append(
            _translate_dim(
                d,
                dim_pk,
                metric_to_fact,
                segment_count_value_pool,
                attribute_value_pools,
            )
        )
    for f in user_input.facts:
        tables.append(
            _translate_fact(
                f,
                dim_pk,
                metric_by_name,
                reference_dims,
                attribute_value_pools,
            )
        )
    for e in user_input.events:
        tables.append(_translate_event(e, dim_pk, attribute_value_pools))

    # M124: auto-fill ``dim_date`` when the user declared an explicit schema
    # but omitted it. The fact/event builders unconditionally key on the
    # ``dim_date.date_key`` PK; without this the engine raises
    # ``KeyError: 'dim_date'`` deep inside dim resolution. Auto-generation
    # only fires when ALL of dimensions/facts/events are empty, so explicit-
    # schema users used to have to remember to declare dim_date themselves
    # for every config — even though the date_key column is fully derivable
    # from the time window.
    table_names = {t.name for t in tables}
    if "dim_date" not in table_names:
        tables.insert(0, _make_default_dim_date())
        table_names.add("dim_date")

    # M124: bridges may reference ``dim_{unit}`` even when the user didn't
    # declare it as a dimension. The builder's
    # ``_bridge_references_resolve`` validator already accepts this name as
    # always-available, but the engine ``PlotsimConfig`` validator rejects
    # it because the table isn't actually present. Auto-prepend a minimal
    # ``dim_{unit}`` so the engine sees it.
    bridge_targets = {side for b in user_input.bridges for side in (b.left, b.right)}
    auto_unit_dim = f"dim_{user_input.unit}"
    if auto_unit_dim in bridge_targets and auto_unit_dim not in table_names:
        tables.append(
            _make_default_dim_unit(
                user_input,
                attribute_value_pools,
            )
        )
        table_names.add(auto_unit_dim)

    return tables


def _dim_primary_key(d: DimInput) -> str:
    """Return the column name that is the PK of this dim.

    Convention: the column with ``type: id`` is the PK.
    """
    for col in d.columns:
        if col.type == "id":
            return col.name
    # Fallback: first column. Engine will surface a clearer error if
    # neither convention holds.
    return d.columns[0].name


def _translate_dim(
    d: DimInput,
    dim_pk: dict[str, str],
    metric_to_fact: dict[str, str],
    segment_count_value_pool: dict[str, list[str]],
    attribute_value_pools: dict[str, dict[str, list[str]]],
) -> Table:
    columns: list[Column] = []
    for col in d.columns:
        columns.append(
            _translate_column(
                col,
                owning_table=d.name,
                dim_pk=dim_pk,
                metric_to_fact=metric_to_fact,
                is_dim_date=(d.name == "dim_date"),
                segment_count_value_pool=segment_count_value_pool,
                attribute_value_pools=attribute_value_pools,
            )
        )
    pk = _dim_primary_key(d)
    foreign_keys = _foreign_keys(d.columns, dim_pk)
    grain = _dim_grain(d)
    # M117: ``DimInput.count`` is the sub-entity dim row multiplier from the
    # builder surface; on engine variable-grain dims this becomes
    # ``Table.count`` and composes with ``Entity.size`` in the dim builder.
    # On non-variable dims ``Table.count > 1`` would raise at engine load,
    # so the engine model rejects the misconfiguration upstream.
    return Table(
        name=d.name,
        type="dim",
        grain=grain,
        columns=columns,
        primary_key=pk,
        foreign_keys=foreign_keys,
        count=d.count,
    )


def _dim_grain(d: DimInput) -> Grain:
    if d.reference:
        return "per_reference"
    if d.per == "period":
        return "per_period"
    # M115 sub-entity dim (declared with per: unit but having an FK to
    # another dim) is detected by the presence of a ref.X column to a
    # non-date dim. For now: the ``count`` field forces variable grain.
    if any(col.type.startswith("ref.") and col.type != "ref.dim_date" for col in d.columns):
        return "variable"
    return "per_entity"


def _foreign_keys(
    columns: list[ColumnInput],
    dim_pk: dict[str, str],
) -> list[str]:
    fks: list[str] = []
    for col in columns:
        if col.type.startswith("ref."):
            target_dim = col.type.split(".", 1)[1]
            target_pk = dim_pk.get(target_dim)
            if target_pk is None:
                # Unknown ref target — let engine validation report it.
                continue
            fks.append(f"{target_dim}.{target_pk}")
    return fks


def _translate_fact(
    f: FactInput,
    dim_pk: dict[str, str],
    metric_by_name: dict[str, Metric],
    reference_dims: set[str],
    attribute_value_pools: dict[str, dict[str, list[str]]],
) -> Table:
    metric_to_fact: dict[str, str] = {f.name: f.name}  # not used for SCD on facts
    columns = [
        _translate_column(
            col,
            owning_table=f.name,
            dim_pk=dim_pk,
            metric_to_fact=metric_to_fact,
            metric_by_name=metric_by_name,
            # ``segment.count`` stays dim-only (cohort population is a
            # per-entity-dim concept); pass an empty pool so the
            # segment.count branch keeps its existing rejection on
            # facts.
            segment_count_value_pool={},
            # 0.6-M19 Fix 1: pool.{attr} columns are now valid on
            # facts. Thread the per-attribute pools and flip the gate
            # so ``pool.tier`` / ``pool.region`` on a fact resolves
            # the same way it already does on dims.
            attribute_value_pools=attribute_value_pools,
            pool_columns_allowed=True,
        )
        for col in f.columns
    ]
    foreign_keys = _foreign_keys(f.columns, dim_pk)

    # 0.6-M18: grain dispatch. The builder supports three fact shapes:
    #   * per_entity_per_period (default — one row per (entity, period))
    #   * variable (parent of a per_parent_row child; row count derived
    #     from row_count_driver × row_count_scale)
    #   * per_parent_row (child of a parent fact; row count = parent
    #     rows × uniform(children_per_row))
    # Both new shapes route through dedicated config fields rather
    # than overloading existing ones, so a fact that sets neither pair
    # stays byte-identical with pre-M18.
    grain: Grain
    if f.parent_table is not None:
        grain = "per_parent_row"
        row_count_source = None
        parent_table = f.parent_table
        children_per_row = f.children_per_row
        # Children have an FK to their parent fact, not a (date_key,
        # entity_id) composite — the inherited entity / period columns
        # are non-PK on the child. The PK convention defaults to the
        # ``id``-typed column or first column (mirroring events).
        pk_candidates = [col.name for col in f.columns if col.type == "id"]
        pk: str | list[str] = pk_candidates[0] if pk_candidates else f.columns[0].name
    elif f.row_count_driver is not None:
        grain = "variable"
        row_count_source = f"proportional:{f.row_count_driver}:scale:{f.row_count_scale}"
        parent_table = None
        children_per_row = None
        pk_candidates = [col.name for col in f.columns if col.type == "id"]
        pk = pk_candidates[0] if pk_candidates else f.columns[0].name
    else:
        grain = "per_entity_per_period"
        row_count_source = None
        parent_table = None
        children_per_row = None
        pk = _composite_pk_from_refs(f.columns, reference_dims)

    return Table(
        name=f.name,
        type="fact",
        grain=grain,
        columns=columns,
        primary_key=pk,
        foreign_keys=foreign_keys,
        row_count_source=row_count_source,
        cdc=f.cdc,
        parent_table=parent_table,
        children_per_row=children_per_row,
    )


def _composite_pk_from_refs(
    columns: list[ColumnInput],
    reference_dims: set[str],
) -> list[str]:
    """Fact PK convention: ref.X columns where X is NOT a reference dim.

    Reference dims (``reference: true``) are static lookups — they do
    not expand the per-entity-per-period grain, so their FK on a fact
    table is documentary, not part of the natural key. Mirrors the
    sample_saas template (fct_revenue PK = [date_key, company_id]
    even though plan_id is also an FK).
    """
    pk: list[str] = []
    for col in columns:
        if not col.type.startswith("ref."):
            continue
        target_dim = col.type.split(".", 1)[1]
        if target_dim in reference_dims:
            continue
        pk.append(col.name)
    return pk


def _translate_event(
    e: EventInput,
    dim_pk: dict[str, str],
    attribute_value_pools: dict[str, dict[str, list[str]]],
) -> Table:
    columns: list[Column] = []
    for col in e.columns:
        columns.append(_translate_event_column(col, e, dim_pk, attribute_value_pools))
    foreign_keys = _foreign_keys(e.columns, dim_pk)

    # Event PK convention: column with type=id. Falls back to first col.
    pk: str | list[str]
    pk_candidates = [col.name for col in e.columns if col.type == "id"]
    pk = pk_candidates[0] if pk_candidates else e.columns[0].name

    row_count_source = None
    if e.trigger == "proportional":
        row_count_source = f"proportional:{e.driver}:scale:{e.scale}"

    return Table(
        name=e.name,
        type="event",
        grain="variable",
        columns=columns,
        primary_key=pk,
        foreign_keys=foreign_keys,
        row_count_source=row_count_source,
        cdc=False,
        log_format=e.log_format,
        log_filename=e.log_filename,
    )


def _translate_event_column(
    col: ColumnInput,
    e: EventInput,
    dim_pk: dict[str, str],
    attribute_value_pools: dict[str, dict[str, list[str]]],
) -> Column:
    """Event columns add the ``flag`` type for threshold trigger columns."""
    if col.type == "flag":
        if e.trigger != "threshold":
            raise ValueError(
                f"event {e.name!r} column {col.name!r}: type 'flag' is "
                f"only valid in threshold-triggered events"
            )
        direction = "above" if e.above is not None else "below"
        threshold = e.above if e.above is not None else e.below
        for_periods = e.for_periods if e.for_periods is not None else 1
        source = f"threshold:{e.metric}:{direction}:{threshold}:for:{for_periods}"
        return Column(name=col.name, dtype="boolean", source=source)
    return _translate_column(
        col,
        owning_table=e.name,
        dim_pk=dim_pk,
        metric_to_fact={},  # events don't host SCDs
        # ``segment.count`` stays dim-only — empty pool keeps that
        # rejection path on event columns.
        segment_count_value_pool={},
        # 0.6-M19 Fix 1: pool.{attr} is now valid on event columns
        # for the same reason it's valid on facts — every row resolves
        # to a single entity via the per_entity FK chain, so
        # ``value_pool[entity_name]`` is indexable.
        attribute_value_pools=attribute_value_pools,
        pool_columns_allowed=True,
    )


# ── Column-type vocabulary translator ───────────────────────────────────────


def _translate_column(
    col: ColumnInput,
    *,
    owning_table: str,
    dim_pk: dict[str, str],
    metric_to_fact: dict[str, str],
    segment_count_value_pool: dict[str, list[str]],
    attribute_value_pools: dict[str, dict[str, list[str]]],
    pool_columns_allowed: bool = True,
    metric_by_name: Optional[dict[str, Metric]] = None,
    is_dim_date: bool = False,
) -> Column:
    """Translate a builder ColumnInput into an engine Column.

    Vocabulary handled here:

      * ``id``                     → dtype=id,      source=pk
      * ``ref.{dim}``              → dtype=id,      source=fk:{dim}.{pk}
      * ``metric.{name}``          → dtype=int|float, source=metric:{name}
      * ``faker.{kind}``           → dtype=string|int, source=generated:faker.{kind}
      * ``static.{value}``         → dtype=float|string, source=static:{value}
      * ``segment.count``          → dtype=int,     source=pool:cohort_size + value_pool
      * ``timestamp``              → dtype=date,    source=generated:timestamp
      * ``date``/``int``/``string``/``float`` (dim_date dtype words)
                                   → dtype=<word>,  source=generated:date_key
      * ``bucket``                 → dtype=string,  source=text:bucket:[labels]
      * ``scd``                    → dtype=string,  source=scd_type2 + SCDType2Config

    ``segment.count`` translates to a PoolSource. After segment expansion
    every ``Entity.size`` is 1 in the builder path, so ``derived:size``
    would emit 1 for every row instead of the cohort population. The
    PoolSource's value_pool maps each expanded entity name to
    ``[str(original_count)]``.
    """
    t = col.type

    # ─── id column → pk ──────────────────────────────────────────────────
    if t == "id":
        return Column(name=col.name, dtype="id", source="pk")

    # ─── ref.<dim> ──────────────────────────────────────────────────────
    if t.startswith("ref."):
        target_dim = t.split(".", 1)[1]
        target_pk = dim_pk.get(target_dim)
        if target_pk is None:
            # Defer the error to engine validation; it has cross-reference context.
            target_pk = f"{target_dim.removeprefix('dim_')}_id"
        return Column(
            name=col.name,
            dtype="id",
            source=f"fk:{target_dim}.{target_pk}",
        )

    # ─── metric.<name> ──────────────────────────────────────────────────
    if t.startswith("metric."):
        metric_name = t.split(".", 1)[1]
        dtype: Dtype = "int"
        if metric_by_name is not None:
            metric = metric_by_name.get(metric_name)
            if metric is not None:
                dtype = "int" if metric.distribution == "poisson" else "float"
        else:
            dtype = "float"
        return Column(name=col.name, dtype=dtype, source=f"metric:{metric_name}")

    # ─── faker.<kind> ───────────────────────────────────────────────────
    if t.startswith("faker."):
        kind = t.split(".", 1)[1]
        faker_dtype: Dtype = "int" if kind == "year" else "string"
        return Column(
            name=col.name,
            dtype=faker_dtype,
            source=f"generated:faker.{kind}",
        )

    # ─── geo.<field> ────────────────────────────────────────────────────
    # Row-coherent geo bundle. All ``geo.<field>`` columns on the same dim
    # row resolve from a single bundle entry drawn from
    # ``plotsim.data.GEO_LOCATIONS``, so a (country, city) pair in the
    # output is always a real city in the stated country. ``latitude`` and
    # ``longitude`` are floats; the rest are strings.
    if t.startswith("geo."):
        from plotsim.data import GEO_BUNDLE_FIELDS

        geo_field = t.split(".", 1)[1]
        if geo_field not in GEO_BUNDLE_FIELDS:
            raise ValueError(
                f"column {col.name!r} in {owning_table!r}: unknown geo "
                f"field {geo_field!r}. Valid fields: "
                f"{sorted(GEO_BUNDLE_FIELDS)}"
            )
        geo_dtype: Dtype = "float" if geo_field in {"latitude", "longitude"} else "string"
        return Column(
            name=col.name,
            dtype=geo_dtype,
            source=f"generated:geo.{geo_field}",
        )

    # ─── static.<value> ─────────────────────────────────────────────────
    if t.startswith("static."):
        value = t.split(".", 1)[1]
        # Numeric static value → float; everything else → string.
        static_dtype: Dtype
        try:
            float(value)
            static_dtype = "float"
        except ValueError:
            static_dtype = "string"
        return Column(name=col.name, dtype=static_dtype, source=f"static:{value}")

    # ─── pool.{attr} → pool:{attr} + value_pool keyed by entity ─────────
    if t.startswith("pool."):
        attr_name = t.split(".", 1)[1]
        if not attr_name:
            raise ValueError(
                f"column {col.name!r} in {owning_table!r}: type 'pool.' "
                f"requires an attribute name (e.g. 'pool.industry')"
            )
        if not pool_columns_allowed:
            raise ValueError(
                f"column {col.name!r} in {owning_table!r}: type "
                f"'pool.{attr_name}' is only valid on per_entity dim "
                f"columns (segment attributes are not exposed on fact / "
                f"event tables)"
            )
        pool = attribute_value_pools.get(attr_name)
        if pool is None:
            available = sorted(attribute_value_pools)
            raise ValueError(
                f"column {col.name!r} in {owning_table!r}: type "
                f"'pool.{attr_name}' references attribute {attr_name!r} "
                f"which is not declared on every segment. Available "
                f"attributes (declared on every segment): "
                f"{available if available else '<none>'}"
            )
        return Column(
            name=col.name,
            dtype="string",
            source=f"pool:{attr_name}",
            value_pool=dict(pool),
        )

    # ─── segment.count → pool:cohort_size + value_pool ──────────────────
    if t == "segment.count":
        if not segment_count_value_pool:
            raise ValueError(
                f"column {col.name!r} in {owning_table!r}: type "
                f"'segment.count' is only valid on per_entity dim columns "
                f"(an unreachable branch under normal builder flow — fact "
                f"and event paths supply an empty pool to surface this)"
            )
        return Column(
            name=col.name,
            dtype="int",
            source="pool:cohort_size",
            value_pool=dict(segment_count_value_pool),
        )

    # ─── timestamp ──────────────────────────────────────────────────────
    if t == "timestamp":
        return Column(name=col.name, dtype="date", source="generated:timestamp")

    # ─── dim_date dtype words ───────────────────────────────────────────
    if is_dim_date_dtype(t):
        if not is_dim_date:
            raise ValueError(
                f"column {col.name!r} in {owning_table!r}: dtype-only type "
                f"{t!r} is supported on dim_date columns only. Other tables "
                f"must use a source-bearing type (metric.X, faker.X, "
                f"static.X, ref.X, etc.)"
            )
        return Column(
            name=col.name,
            dtype=t,
            source="generated:date_key",
        )

    # ─── bucket ─────────────────────────────────────────────────────────
    if t == "bucket":
        if not col.labels:
            raise ValueError(
                f"column {col.name!r}: type 'bucket' requires a non-empty `labels` list"
            )
        labels_str = ", ".join(col.labels)
        return Column(
            name=col.name,
            dtype="string",
            source=f"text:bucket:[{labels_str}]",
        )

    # ─── range → range:<min>:<max> ──────────────────────────────────────
    # 0.6-M19 Fix 2: explicit numeric range with engine-side per-row
    # uniform draw. The column's ``dtype`` (defaulted to ``int`` here
    # when omitted by the user — matches Faker's ``random_int`` shape
    # for the common case) decides whether the engine emits integers
    # or floats. Author intent comes from ``range: [min, max]`` on
    # ColumnInput; the source string carries the bounds.
    if t == "range":
        if col.range is None or len(col.range) != 2:
            raise ValueError(
                f"column {col.name!r}: type 'range' requires a `range: "
                f"[min, max]` two-element list"
            )
        lo, hi = col.range
        if hi < lo:
            raise ValueError(f"column {col.name!r}: range [{lo}, {hi}] has max < min")
        # Both bounds in {int, float}. Use int dtype when both bounds
        # are integers (preserves author intent); otherwise float.
        both_int = isinstance(lo, int) and isinstance(hi, int) and not isinstance(lo, bool)
        range_dtype: Dtype = "int" if both_int else "float"
        return Column(
            name=col.name,
            dtype=range_dtype,
            source=f"range:{lo}:{hi}",
        )

    # ─── scd ────────────────────────────────────────────────────────────
    if t == "scd":
        if not col.tracks or not col.tiers or not col.at:
            raise ValueError(
                f"column {col.name!r}: type 'scd' requires `tracks`, `tiers`, and `at` sub-fields"
            )
        target_fact = metric_to_fact.get(col.tracks)
        if target_fact is None:
            raise ValueError(
                f"column {col.name!r}: scd tracks metric {col.tracks!r}, "
                f"but no fact table emits that metric"
            )
        return Column(
            name=col.name,
            dtype="string",
            source="scd_type2",
            scd_type2=SCDType2Config(
                trigger_metric=f"{target_fact}.{col.tracks}",
                thresholds=tuple(col.at),
                labels=tuple(col.tiers),
            ),
        )

    # ─── narrative ──────────────────────────────────────────────────────
    if t == "narrative":
        if col.template is None or col.lexicons is None:
            raise ValueError(
                f"column {col.name!r}: type 'narrative' requires "
                f"`template` and `lexicons` sub-fields (got "
                f"template={col.template!r}, "
                f"lexicons={'set' if col.lexicons is not None else 'None'})"
            )
        # The source key defaults to the column name — narrative columns are
        # uniquely identified within a table by name, so reusing it produces
        # a self-documenting source string and avoids a separate ``key``
        # input field. ``NarrativeConfig`` runs its own structural
        # validation (template placeholders, archetype/slot/band coverage)
        # at construction; cross-config archetype-name checks happen in
        # ``PlotsimConfig._narrative_gates``.
        narrative_kwargs: dict[str, Any] = {
            "template": col.template,
            "lexicons": col.lexicons,
        }
        if col.bands is not None:
            narrative_kwargs["bands"] = tuple(col.bands)
        return Column(
            name=col.name,
            dtype="string",
            source=f"narrative:{col.name}",
            narrative=NarrativeConfig(**narrative_kwargs),
        )

    # ─── struct (0.6-M14c) ──────────────────────────────────────────────
    if t == "struct":
        if not col.nested_schema:
            raise ValueError(
                f"column {col.name!r}: type 'struct' requires "
                f"`nested_schema: {{<field>: <int|float|string|boolean>, ...}}`"
            )
        return Column(
            name=col.name,
            dtype="struct",
            source="nested",
            nested_schema=dict(col.nested_schema),
        )

    # ─── array (0.6-M14c) ───────────────────────────────────────────────
    if t == "array":
        if col.array_element_type is None:
            raise ValueError(
                f"column {col.name!r}: type 'array' requires "
                f"`array_element_type: <int|float|string|boolean>`"
            )
        return Column(
            name=col.name,
            dtype="array",
            source="nested",
            array_element_type=col.array_element_type,
            array_length=col.array_length,
        )

    raise ValueError(
        f"column {col.name!r} in {owning_table!r}: unknown type {t!r}. "
        f"Valid types: id, ref.X, metric.X, faker.X, geo.X, static.X, "
        f"segment.count, pool.X, timestamp, date, int, string, float, "
        f"bucket, scd, narrative, struct, array"
    )


# ── Auto-generated schema ───────────────────────────────────────────────────


def _make_default_dim_date() -> Table:
    """The minimal ``dim_date`` shape used by both the auto-schema branch
    and the explicit-schema fallback. Five columns: PK + date / year /
    month / quarter, all derivable from the time window.
    """
    return Table(
        name="dim_date",
        type="dim",
        grain="per_period",
        columns=[
            Column(name="date_key", dtype="id", source="pk"),
            Column(name="date", dtype="date", source="generated:date_key"),
            Column(name="year", dtype="int", source="generated:date_key"),
            Column(name="month", dtype="int", source="generated:date_key"),
            Column(name="quarter", dtype="int", source="generated:date_key"),
        ],
        primary_key="date_key",
    )


def _make_default_dim_unit(
    user_input: UserInput,
    attribute_value_pools: dict[str, dict[str, list[str]]],
) -> Table:
    """The minimal ``dim_{unit}`` shape — PK + faker-name column, plus one
    ``pool:{attr}`` column per attribute declared on every segment.

    Used by the auto-schema branch and the explicit-schema fallback when
    a bridge references ``dim_{unit}`` but the user didn't declare it.
    """
    unit = user_input.unit
    pk_col = f"{unit}_id"
    name_col = f"{unit}_name"
    faker_kind = UNIT_FAKER_MAP.get(unit, "faker.company")

    columns: list[Column] = [
        Column(name=pk_col, dtype="id", source="pk"),
        Column(name=name_col, dtype="string", source=f"generated:{faker_kind}"),
    ]
    for attr_name in sorted(attribute_value_pools):
        columns.append(
            Column(
                name=attr_name,
                dtype="string",
                source=f"pool:{attr_name}",
                value_pool=dict(attribute_value_pools[attr_name]),
            )
        )
    return Table(
        name=f"dim_{unit}",
        type="dim",
        grain="per_entity",
        columns=columns,
        primary_key=pk_col,
    )


def _auto_generate_schema(
    user_input: UserInput,
    metric_by_name: dict[str, Metric],
    attribute_value_pools: dict[str, dict[str, list[str]]],
) -> list[Table]:
    """Minimal default schema: dim_date + dim_{unit} + fct_{unit}.

    Used when the user provides no schema. Carries every metric on the
    fact table; sub-entity dims, multi-fact splits, and events are
    out of scope for the auto path — users who need those declare an
    explicit schema.

    When ``attribute_value_pools`` is non-empty, ``dim_{unit}`` gains
    one ``pool:{attr}`` column per attribute, alphabetically ordered.
    Auto-schema users who declare segment attributes get those attributes
    surfaced on the dim without writing the schema by hand.
    """
    unit = user_input.unit
    unit_dim = f"dim_{unit}"
    fact_table = f"fct_{unit}"
    pk_col = f"{unit}_id"

    dim_date = _make_default_dim_date()
    dim_unit = _make_default_dim_unit(user_input, attribute_value_pools)

    fact_columns: list[Column] = [
        Column(name="date_key", dtype="id", source="fk:dim_date.date_key"),
        Column(name=pk_col, dtype="id", source=f"fk:{unit_dim}.{pk_col}"),
    ]
    for m in user_input.metrics:
        engine_metric = metric_by_name[m.name]
        m_dtype: Dtype = "int" if engine_metric.distribution == "poisson" else "float"
        fact_columns.append(
            Column(
                name=m.name,
                dtype=m_dtype,
                source=f"metric:{m.name}",
            )
        )
    fact = Table(
        name=fact_table,
        type="fact",
        grain="per_entity_per_period",
        columns=fact_columns,
        primary_key=["date_key", pk_col],
        foreign_keys=["dim_date.date_key", f"{unit_dim}.{pk_col}"],
    )

    return [dim_date, dim_unit, fact]


# ── M122: pool.{attr} value-pool builder ────────────────────────────────────


def _build_attribute_value_pools(
    user_input: UserInput,
) -> dict[str, dict[str, list[str]]]:
    """Build per-attribute value pools keyed by expanded entity name.

    Returns ``{attr_name: {entity_name: [values]}}`` for every attribute
    declared on EVERY segment. Attributes declared on only some segments
    are omitted from the auto-schema (they would leave entities in other
    segments with no pool entry, which the engine's
    ``validate_value_pool_coverage`` rejects). Explicit ``pool.{attr}``
    columns referencing partial attributes raise at column-translate
    time with a clear message.

    Scalar attribute values (``tier: enterprise``) wrap into a
    single-element list — PoolSource value lists are always ``list[str]``
    and a single value is just a one-element list. Numeric / bool
    attribute values are stringified because PoolSource columns are
    ``dtype=string``; round-tripping ints via the pool machinery is
    intentional (the engine writes string cells).
    """
    if not user_input.segments:
        return {}

    # Find attributes declared on every segment. ``user_input.segments`` is
    # non-empty here (early return above), so ``per_segment_keys`` is too.
    per_segment_keys = [set(s.attributes.keys()) for s in user_input.segments]
    common_keys = set.intersection(*per_segment_keys)

    pools: dict[str, dict[str, list[str]]] = {}
    for attr in common_keys:
        per_entity: dict[str, list[str]] = {}
        for s in user_input.segments:
            raw = s.attributes[attr]
            if isinstance(raw, (list, tuple)):
                values = [str(v) for v in raw]
            else:
                values = [str(raw)]
            for i in range(s.count):
                per_entity[f"{s.name}_{i:04d}"] = values
        pools[attr] = per_entity
    return pools


# ── M122: bridges / quality / holdout / entity_features translators ─────────


def _translate_bridges(
    user_input: UserInput,
    metric_by_name: dict[str, Metric],
) -> list[BridgeTableConfig]:
    """Translate ``UserInput.bridges`` into engine ``BridgeTableConfig``.

    1:1 by construction:
      * ``left`` / ``right`` → ``connects=[left, right]``
      * ``cardinality=(min, max)`` → ``BridgeCardinality(min, max)``
      * ``driver`` non-null → ``trajectory_driven=True`` (engine default
        is also True; the field is documentary on the builder side —
        engine bridge generation reads the entity's trajectory directly,
        not a specific metric)
      * ``columns[*]`` → ``BridgeMetric`` via the same column-type
        vocabulary, restricted to ``metric.X``, ``static.X``, and
        ``faker.X`` (the only sources the engine ``BridgeMetric``
        validator allows)
    """
    bridges: list[BridgeTableConfig] = []
    for b in user_input.bridges:
        bridge_metrics = [
            _translate_bridge_column(col, metric_by_name, b.name) for col in b.columns
        ]
        bridges.append(
            BridgeTableConfig(
                name=b.name,
                connects=[b.left, b.right],
                cardinality=BridgeCardinality(min=b.cardinality[0], max=b.cardinality[1]),
                trajectory_driven=True,
                metrics=bridge_metrics,
            )
        )
    return bridges


def _translate_bridge_column(
    col: BridgeColumnInput,
    metric_by_name: dict[str, Metric],
    bridge_name: str,
) -> BridgeMetric:
    """Translate one bridge column to a BridgeMetric.

    Bridge metrics support ``metric:X`` / ``static:X`` /
    ``generated:faker.X`` only — bridges are static rows with no period
    axis, so period-anchored sources (timestamps, threshold-firing,
    proportional row counts, lag, refs) are rejected by the engine.
    The builder layer raises a clearer message before the engine sees
    it.
    """
    t = col.type
    if t.startswith("metric."):
        metric_name = t.split(".", 1)[1]
        metric = metric_by_name.get(metric_name)
        metric_dtype: Dtype = "float"
        if metric is not None and metric.distribution == "poisson":
            metric_dtype = "int"
        return BridgeMetric(
            name=col.name,
            dtype=metric_dtype,
            source=f"metric:{metric_name}",
        )
    if t.startswith("static."):
        value = t.split(".", 1)[1]
        static_dtype: Dtype
        try:
            float(value)
            static_dtype = "float"
        except ValueError:
            static_dtype = "string"
        return BridgeMetric(
            name=col.name,
            dtype=static_dtype,
            source=f"static:{value}",
        )
    if t.startswith("faker."):
        kind = t.split(".", 1)[1]
        faker_dtype: Dtype = "int" if kind == "year" else "string"
        return BridgeMetric(
            name=col.name,
            dtype=faker_dtype,
            source=f"generated:faker.{kind}",
        )
    raise ValueError(
        f"bridge {bridge_name!r} column {col.name!r}: type {t!r} is not "
        f"supported on bridge rows. Bridge metrics accept metric.X, "
        f"static.X, and faker.X only — bridges have no period axis to "
        f"anchor period-derived sources against."
    )


def _translate_quality(user_input: UserInput) -> QualityConfig:
    """Translate ``UserInput.quality`` into engine ``QualityConfig``.

    Each input issue maps to one ``QualityIssue``:
      * ``column`` set → ``target_columns=[column]``.
      * ``column`` omitted (only valid for ``duplicate_rows``,
        ``late_arrival``, and ``volume_anomaly``) → ``target_columns=["*"]``
        — the engine's sentinel for "every eligible column on the
        resolved table".
      * ``volume_anomaly`` additionally carries ``mode`` plus one of
        ``period`` / ``periods`` — those route onto the engine's
        ``mode`` / ``target_period`` / ``target_periods`` fields.
    """
    issues: list[QualityIssue] = []
    for q in user_input.quality:
        target_columns = [q.column] if q.column else ["*"]
        issues.append(
            QualityIssue(
                type=q.issue,
                target_table=q.table,
                target_columns=target_columns,
                rate=q.rate,
                seed_offset=q.seed_offset,
                mode=q.mode,
                target_period=q.period,
                target_periods=q.periods,
            )
        )
    return QualityConfig(quality_issues=issues)


def _translate_holdout(user_input: UserInput) -> HoldoutConfig:
    """Translate ``UserInput.holdout`` into engine ``HoldoutConfig``.

    No holdout declared → disabled config (the PlotsimConfig default).
    Engine-side gates (target metric resolves to a numeric fact column,
    train_periods >= min_training_periods, no overlap with quality
    issues) raise at PlotsimConfig load.
    """
    h = user_input.holdout
    if h is None:
        return HoldoutConfig()
    return HoldoutConfig(
        enabled=True,
        target_metric=h.target,
        holdout_periods=h.periods,
        min_training_periods=h.min_training_periods,
    )


def _translate_entity_features(user_input: UserInput) -> EntityFeaturesConfig:
    """Translate ``UserInput.entity_features`` into ``EntityFeaturesConfig``.

    No declaration → disabled (default). The boolean shorthand
    ``entity_features: true`` is normalised to an empty
    ``EntityFeaturesInput`` upstream, so reaching here with a non-None
    value always means "enabled, with these settings."
    """
    ef = user_input.entity_features
    if ef is None:
        return EntityFeaturesConfig()
    return EntityFeaturesConfig(
        enabled=True,
        metrics=list(ef.metrics),
        include_labels=ef.include_labels,
    )
