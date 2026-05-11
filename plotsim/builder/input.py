"""UserInput model — plain-language input for the builder.

The builder accepts the same shape from two surfaces:

- ``create_from_yaml(path)`` — YAML file conforming to the template at
  ``plotsim/configs/templates/saas_template.yaml``.
- ``create(**kwargs)`` — Python keyword arguments mirroring that template.

Both surfaces normalise into a ``UserInput`` instance, which the interpreter
(``plotsim.builder.interpreter``) translates into a ``PlotsimConfig``.

Validation philosophy
---------------------
Structural problems (duplicate names, orphan references, causal cycles,
unknown vocabulary, malformed archetype DSL) raise at ``UserInput``
construction time. Semantic concerns (a 12-period seasonal pattern, a
single-segment cohort, a maximally-correlated config) emit ``UserWarning``
and let the build proceed — they are choices the user can defend, not bugs.

Errors here name both sides of the problem so the user can act without
re-reading the spec.
"""

from __future__ import annotations

import warnings
from typing import Annotated, Any, Literal, Optional, Union

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from plotsim._types import Distribution

from .parser import ArchetypeParseError, parse_archetype
from .recipes import (
    VALID_BASELINE_WORDS,
    VALID_RELATIONSHIP_WORDS,
)


# Per-family params spec for the explicit-distribution path on
# ``MetricInput`` (mission 0.6-M6). Keep in sync with
# ``plotsim._distribution_registry`` — adding a family there means
# adding its required/optional keys here so the builder layer can
# validate the params dict before it reaches the engine.
#
# ``poisson`` has no parameters; the metric's center IS lambda.
# ``beta`` accepts an optional ``scale`` (used when no value_range is
# pinned, see ``_distribution_registry._beta_sample_*``).
_DISTRIBUTION_REQUIRED_KEYS: dict[str, set[str]] = {
    "lognorm": {"s"},
    "gamma": {"shape"},
    "poisson": set(),
    "beta": {"alpha", "beta"},
    "normal": {"sigma"},
    "weibull": {"shape"},
}
_DISTRIBUTION_OPTIONAL_KEYS: dict[str, set[str]] = {
    "beta": {"scale"},
}


# ── Window ──────────────────────────────────────────────────────────────────


class WindowInput(BaseModel):
    """Time window declaration.

    Accepts either keyword form (``{"start": "2023-01", "end": "2024-12"}``)
    or a 2- or 3-tuple ``("2023-01", "2024-12", "monthly")``. The tuple form
    is normalised by ``UserInput._coerce_window_tuple`` before this model
    sees the data.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    start: str
    end: str
    every: Literal["daily", "weekly", "monthly"] = "monthly"


# ── Metric ──────────────────────────────────────────────────────────────────


class MetricInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    type: Literal["score", "amount", "count", "index"]
    polarity: Literal["positive", "negative"]
    label: Optional[str] = None
    range: Optional[tuple[float, float]] = None
    follows: Optional[str] = None
    delay: Optional[int] = None
    # M119: per-metric seasonal sensitivity. Default ``1.0`` (follow the
    # global ``seasonality`` strength at face value). ``-0.5`` halves and
    # inverts; ``0.0`` makes the metric immune. Translated unchanged onto
    # ``Metric.seasonal_sensitivity`` by the interpreter.
    seasonal_sensitivity: float = 1.0
    # 0.6-M6: explicit per-metric distribution choice. When set, bypasses
    # the interpreter's auto-pick (type/range → distribution) entirely.
    # ``Distribution`` is a Literal of the six implemented families, so
    # pydantic surfaces the valid name list on a typo. ``distribution_params``
    # is validated below against ``_DISTRIBUTION_REQUIRED_KEYS`` /
    # ``_DISTRIBUTION_OPTIONAL_KEYS`` to ensure each family gets exactly
    # the params it can use.
    distribution: Optional[Distribution] = None
    distribution_params: Optional[dict[str, float]] = None
    # 0.6-M9b: opt-in adstock-style decay on the causal lag. Setting
    # ``decay_window`` (>= 1) flips ``CausalLag.decay`` to True; the
    # presence of the field IS the opt-in (no separate boolean — keeps
    # the builder surface flat). ``decay_kernel`` defaults to
    # ``"geometric"`` (half-life of one period) and is ignored when
    # ``decay_window`` is None. Both fields require ``follows`` /
    # ``delay`` to be set — decay without a base lag is meaningless.
    decay_window: Optional[int] = Field(default=None, ge=1, le=10_000)
    decay_kernel: Literal["geometric", "linear"] = "geometric"

    @field_validator("name")
    @classmethod
    def _name_is_simple(cls, v: str) -> str:
        if not v.replace("_", "").isalnum():
            raise ValueError(f"metric name {v!r} must be alphanumeric or underscores only")
        return v

    @model_validator(mode="after")
    def _range_required_for_amount_and_index(self) -> "MetricInput":
        if self.type in ("amount", "index") and self.range is None:
            raise ValueError(
                f"metric {self.name!r} of type {self.type!r} requires a "
                f"`range: [min, max]` (only `score` and `count` may omit it)"
            )
        if self.type == "count" and self.range is not None:
            raise ValueError(
                f"metric {self.name!r} of type 'count' must not declare a "
                f"range — counts are unbounded integers"
            )
        if self.range is not None:
            lo, hi = self.range
            if lo >= hi:
                raise ValueError(
                    f"metric {self.name!r} range [{lo}, {hi}] is invalid: "
                    f"min must be strictly less than max"
                )
        return self

    @model_validator(mode="after")
    def _distribution_params_match_family(self) -> "MetricInput":
        """0.6-M6: validate ``distribution_params`` against the chosen family.

        - ``distribution_params`` set without ``distribution`` is rejected
          (orphan params have no family to belong to).
        - When ``distribution`` is set, every key in
          ``_DISTRIBUTION_REQUIRED_KEYS[distribution]`` must be present;
          extra keys (not required, not optional) are rejected with a
          message naming both required and optional keys for the family.
        - ``poisson`` requires no params; ``distribution_params`` may be
          omitted or an empty dict. A non-empty dict is rejected.
        """
        if self.distribution is None:
            if self.distribution_params is not None:
                raise ValueError(
                    f"metric {self.name!r}: `distribution_params` is set "
                    f"but `distribution` is not — pick a family or remove both"
                )
            return self
        required = _DISTRIBUTION_REQUIRED_KEYS[self.distribution]
        optional = _DISTRIBUTION_OPTIONAL_KEYS.get(self.distribution, set())
        params = self.distribution_params or {}
        missing = required - params.keys()
        if missing:
            raise ValueError(
                f"metric {self.name!r}: distribution {self.distribution!r} "
                f"requires params {sorted(required)}, missing {sorted(missing)}"
            )
        extra = params.keys() - (required | optional)
        if extra:
            allowed = sorted(required | optional) or ["<none>"]
            raise ValueError(
                f"metric {self.name!r}: distribution {self.distribution!r} "
                f"does not accept params {sorted(extra)}. Accepted: {allowed}"
            )
        return self

    @model_validator(mode="after")
    def _follows_and_delay_paired(self) -> "MetricInput":
        if (self.follows is None) != (self.delay is None):
            raise ValueError(
                f"metric {self.name!r}: `follows` and `delay` must be "
                f"declared together (got follows={self.follows!r}, "
                f"delay={self.delay!r}). To remove the lag, omit both."
            )
        if self.delay is not None and self.delay < 1:
            raise ValueError(f"metric {self.name!r}: `delay` must be >= 1, got {self.delay}")
        if self.follows is not None and self.follows == self.name:
            raise ValueError(f"metric {self.name!r} cannot follow itself")
        return self

    @model_validator(mode="after")
    def _decay_requires_follows(self) -> "MetricInput":
        """0.6-M9b: ``decay_window`` requires ``follows`` / ``delay``.

        Decay spreads the driver's effect over a window of past periods,
        which is only meaningful when there's a base lag to spread. Set
        without ``follows``/``delay``, the field is nonsense — fail fast.
        """
        if self.decay_window is not None and self.follows is None:
            raise ValueError(
                f"metric {self.name!r}: `decay_window` is set but "
                f"`follows`/`delay` are not — decay only applies on top "
                f"of a base lag. Add `follows: <driver>` and `delay: <N>`."
            )
        return self


# ── Segment arrival distributions (0.6-M8b) ─────────────────────────────────
#
# Builder-internal models. These describe how the interpreter spreads a
# segment's `count` entities across the time window via per-entity
# `Entity.start_period` draws (the M8a cold-start surface). They live
# here, not in ``plotsim.config``, because:
#
#   * ``SegmentInput`` itself is builder-only — engine-direct YAML users
#     define ``entities`` directly with their own ``Entity.start_period``
#     values and never touch segments.
#   * The arrival distribution is *transient input*: the interpreter
#     consumes it, draws per-entity start_periods, and produces the
#     fully-expanded ``Entity`` list. Nothing downstream of the
#     interpreter knows the distribution existed.
#   * Pattern match: ``ConnectionInput``, ``MetricInput``, ``SegmentInput``
#     all live here for the same reason. The engine surface stays narrow.
#
# Discriminated union by ``kind`` so a downstream reader (and IDE
# autocomplete) can dispatch the four shapes without dict-typing.
# Each model is ``frozen=True`` + ``extra="forbid"`` to match the rest
# of the builder input surface.


class _ArrivalBase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class UniformArrival(_ArrivalBase):
    """Entities arrive evenly distributed across ``[start, end)``.

    ``start`` defaults to 0 (entities can arrive from period 0). ``end``
    defaults to ``None`` — interpreted as ``n_periods - MIN_ACTIVE_PERIODS``
    by the interpreter, so every entity has at least ``MIN_ACTIVE_PERIODS``
    active periods. Set ``end`` explicitly to compress arrivals into a
    sub-window.

    Implementation: ``rng.integers(start, end, size=count)``. Same seed +
    same segment ordering → identical draws.
    """

    kind: Literal["uniform"]
    start: int = Field(default=0, ge=0)
    end: Optional[int] = Field(default=None, ge=1)


class LinearArrival(_ArrivalBase):
    """Entities arrive at a linearly varying rate across ``[start, end)``.

    ``direction='increasing'`` back-loads arrivals (more entities arrive
    in later periods — typical of an organic growth cohort).
    ``direction='decreasing'`` front-loads (more arrivals early — a
    promotional spike).

    Implementation: triangular CDF inversion. For ``increasing``,
    ``period = start + floor((end - start) * sqrt(u))`` where
    ``u ~ Uniform[0, 1)``; for ``decreasing``, ``1 - sqrt(1 - u)``.
    Same seed + same segment ordering → identical draws.
    """

    kind: Literal["linear"]
    start: int = Field(default=0, ge=0)
    end: Optional[int] = Field(default=None, ge=1)
    direction: Literal["increasing", "decreasing"] = "increasing"


class StepArrivalBlock(_ArrivalBase):
    """One block in a ``StepArrival`` schedule.

    ``period`` is the arrival period for entities in this block.
    ``fraction`` is the share of the segment's ``count`` that arrives at
    this period. Fractions across all blocks must sum to ``1.0`` within
    a small tolerance (validated on the parent ``StepArrival``).
    """

    period: int = Field(ge=0)
    fraction: float = Field(gt=0.0, le=1.0)


class StepArrival(_ArrivalBase):
    """Entities arrive in discrete blocks at specified periods.

    Models cohort cuts: e.g. ``[(0, 0.5), (6, 0.3), (12, 0.2)]`` means
    50 % of the segment arrives at period 0, 30 % at period 6, 20 % at
    period 12. Counts are derived by ``round(fraction * count)`` with
    the last block absorbing rounding remainder so total entities equal
    ``count`` exactly.

    Same input → same per-entity assignment, no RNG involved (the
    schedule is deterministic by construction).
    """

    kind: Literal["step"]
    blocks: list[StepArrivalBlock] = Field(min_length=1)

    @model_validator(mode="after")
    def _fractions_sum_to_one(self) -> "StepArrival":
        total = sum(b.fraction for b in self.blocks)
        if not (0.999 <= total <= 1.001):
            raise ValueError(
                f"step arrival blocks: fractions must sum to 1.0 "
                f"(±0.001 tolerance for floating-point); got {total}"
            )
        return self


class ExplicitArrival(_ArrivalBase):
    """Per-entity ``start_period`` values, supplied directly.

    ``start_periods`` length must equal the segment's ``count`` (the
    parent ``SegmentInput`` validator enforces). Useful for
    reproducibility tests, golden fixtures, or research configs where
    cohort timing is the experimental variable.

    No RNG draw — the assignment is deterministic by construction.
    """

    kind: Literal["explicit"]
    start_periods: list[int] = Field(min_length=1)

    @field_validator("start_periods")
    @classmethod
    def _all_non_negative(cls, v: list[int]) -> list[int]:
        for i, p in enumerate(v):
            if p < 0:
                raise ValueError(
                    f"explicit arrival start_periods[{i}] = {p} "
                    f"is negative; start_period must be >= 0"
                )
        return v


# Discriminated union — a downstream reader can dispatch on the ``kind``
# tag without try/except on each model. Pydantic's discriminator routing
# also produces locatable error messages ("kind=step requires blocks")
# when an input dict is malformed, vs the smart-union fallback's noisier
# multi-error tree.
ArrivalDistribution = Annotated[
    Union[UniformArrival, LinearArrival, StepArrival, ExplicitArrival],
    Field(discriminator="kind"),
]


# ── Segment treatment / control config (0.6-M8c) ────────────────────────────


class TreatmentConfig(BaseModel):
    """0.6-M8c: per-segment A/B test split.

    Builder-internal; the interpreter consumes this to assign per-entity
    ``treatment_group`` / ``treatment_lift_log_odds`` / ``treatment_start_period``
    on the expanded ``Entity`` objects (the M8c engine surface).

    Fields:

      * ``fraction`` — share of the segment that lands in the treatment
        arm. ``0.5`` = half the entities get the lift. The remainder go
        to the control arm with ``treatment_lift_log_odds=None`` (no
        shift) but the same ``control_label`` for cohort grouping.
      * ``lift_log_odds`` — known effect size in log-odds units. Applied
        to the treatment arm only. The control arm sees ``None``.
      * ``start_period`` — absolute period index at which treatment
        begins. ``0`` (default) = treatment from period 0; cold-start
        entities still see the lift only after their own
        ``Entity.start_period`` (the engine respects the
        per-(entity, period) gate). Values ``> 0`` carve out a baseline
        window where treatment and control share identical trajectory
        positions — the AC for "pre-treatment baseline is identical".
      * ``treatment_label`` / ``control_label`` — cohort labels for the
        manifest. Defaults match the conventional A/B labelling.

    RNG isolation: the interpreter draws treatment assignments from a
    distinct ``np.random.default_rng(seed ^ TREATMENT_SALT)`` stream,
    so adding / changing the segment's ``arrival`` distribution does
    not shift which entities land in the treatment arm. M8b's
    arrival_rng and M8c's treatment_rng are seed-coupled but
    draw-independent.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    fraction: float = Field(ge=0.0, le=1.0)
    lift_log_odds: float
    start_period: int = Field(default=0, ge=0)
    treatment_label: str = Field(default="treatment", min_length=1)
    control_label: str = Field(default="control", min_length=1)

    @field_validator("lift_log_odds")
    @classmethod
    def _lift_finite(cls, v: float) -> float:
        if not (-1e6 < v < 1e6):
            raise ValueError(
                f"treatment lift_log_odds={v} is non-finite or extreme; "
                f"sensible A/B test lifts are typically in [-2.0, 2.0]"
            )
        return v


# ── Segment ─────────────────────────────────────────────────────────────────


class SegmentInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    count: int = Field(ge=3, le=5_000)
    archetype: str = Field(min_length=1)
    label: Optional[str] = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    baseline: dict[str, str] = Field(default_factory=dict)

    @field_validator("attributes")
    @classmethod
    def _attributes_are_str_or_str_list(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Each attribute value must be a string or a list of strings.

        M122: ``attributes`` doubles as the source for ``pool.{attr}``
        column types — every value must be coerce-able to a list of
        strings (scalars wrap into a single-element list at the
        interpreter step). Numeric attribute values are stringified
        because PoolSource is dtype=string only.
        """
        for key, value in v.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"segment attribute key {key!r} must be a non-empty string")
            if isinstance(value, (list, tuple)):
                if not value:
                    raise ValueError(f"attribute {key!r}: list must be non-empty")
                for item in value:
                    if not isinstance(item, (str, int, float, bool)):
                        raise ValueError(
                            f"attribute {key!r}: list entries must be "
                            f"scalars (str/int/float/bool), got "
                            f"{type(item).__name__}"
                        )
            elif not isinstance(value, (str, int, float, bool)):
                raise ValueError(
                    f"attribute {key!r}: value must be a scalar "
                    f"(str/int/float/bool) or a list of scalars, got "
                    f"{type(value).__name__}"
                )
        return v

    # M119: per-segment seasonal sensitivity. Default ``1.0`` (follow
    # the global ``seasonality`` strength at face value). The interpreter
    # copies this value onto every expanded ``Entity.seasonal_sensitivity``
    # within the segment, so two segments with the same archetype but
    # different sensitivities show different seasonal amplitudes while
    # sharing the underlying trajectory shape.
    seasonal_sensitivity: float = 1.0

    # 0.6-M8b: optional cohort-mix-evolution shape. ``None`` (default)
    # means every entity in the segment starts at period 0 — preserves
    # pre-M8b output byte-for-byte for existing templates. When set, the
    # interpreter draws per-entity ``Entity.start_period`` values from
    # the chosen distribution using a seed-derived RNG; the segment-level
    # field never reaches the engine config, only the per-entity values
    # do (via the M8a ``Entity.start_period`` surface).
    arrival: Optional[ArrivalDistribution] = None
    # 0.6-M8c: optional A/B test split. ``None`` (default) means no
    # treatment / control assignment for this segment. When set, the
    # interpreter draws a deterministic per-entity treatment assignment
    # (using the salted treatment_rng — independent of arrival_rng so
    # changing arrival shape doesn't shift treatment assignments) and
    # populates each ``Entity``'s ``treatment_group`` /
    # ``treatment_lift_log_odds`` / ``treatment_start_period`` fields.
    treatment: Optional[TreatmentConfig] = None

    @field_validator("name")
    @classmethod
    def _name_is_simple(cls, v: str) -> str:
        if not v.replace("_", "").isalnum():
            raise ValueError(f"segment name {v!r} must be alphanumeric or underscores only")
        return v

    @field_validator("baseline")
    @classmethod
    def _baseline_words_in_vocabulary(cls, v: dict[str, str]) -> dict[str, str]:
        unknown = {m: word for m, word in v.items() if word not in VALID_BASELINE_WORDS}
        if unknown:
            raise ValueError(
                f"baseline values {unknown} are not in the baseline "
                f"vocabulary {sorted(VALID_BASELINE_WORDS)}"
            )
        return v

    @model_validator(mode="after")
    def _arrival_explicit_length_matches_count(self) -> "SegmentInput":
        """0.6-M8b: ``ExplicitArrival.start_periods`` length must match
        the segment's ``count``. The check sits on ``SegmentInput`` (not
        on ``ExplicitArrival``) because count is a sibling field — the
        sub-model can't see it during its own validation.
        """
        if isinstance(self.arrival, ExplicitArrival):
            if len(self.arrival.start_periods) != self.count:
                raise ValueError(
                    f"segment {self.name!r}: explicit arrival has "
                    f"{len(self.arrival.start_periods)} start_periods but "
                    f"segment count is {self.count}; lengths must match"
                )
        return self


# ── Connection ──────────────────────────────────────────────────────────────


class ConnectionInput(BaseModel):
    """A correlation-pair connection between two metrics.

    Accepted shapes:
        - "engagement driven_by mrr"                (3-token string, word)
        - "engagement 0.42 mrr"                     (3-token string, number)
        - ("engagement", "driven_by", "mrr")         (3-tuple, word)
        - ("engagement", 0.42, "mrr")                (3-tuple, number)
        - {"a": ..., "relationship": ..., "b": ...} (dict, word)
        - {"a": ..., "coefficient": 0.42, "b": ...} (dict, number)

    The relationship word maps to a fixed coefficient via
    ``RELATIONSHIP_RECIPES``; numeric form lets the user pin any
    coefficient in ``[-1.0, 1.0]``. Exactly one of ``relationship`` /
    ``coefficient`` must be set on the canonical model.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    metric_a: str
    metric_b: str
    relationship: Optional[str] = None
    coefficient: Optional[float] = Field(default=None, ge=-1.0, le=1.0)

    @model_validator(mode="after")
    def _exactly_one_of_relationship_or_coefficient(self) -> "ConnectionInput":
        if self.relationship is None and self.coefficient is None:
            raise ValueError(
                f"connection {self.metric_a!r} ↔ {self.metric_b!r}: "
                f"set either `relationship` (vocabulary word) or "
                f"`coefficient` (numeric in [-1, 1])"
            )
        if self.relationship is not None and self.coefficient is not None:
            raise ValueError(
                f"connection {self.metric_a!r} ↔ {self.metric_b!r}: "
                f"set `relationship` OR `coefficient`, not both. The word "
                f"already implies a coefficient — see "
                f"RELATIONSHIP_RECIPES for the table"
            )
        if self.relationship is not None and self.relationship not in VALID_RELATIONSHIP_WORDS:
            raise ValueError(
                f"unknown relationship word {self.relationship!r}. Valid: "
                f"{sorted(VALID_RELATIONSHIP_WORDS)}"
            )
        return self

    @model_validator(mode="after")
    def _endpoints_distinct(self) -> "ConnectionInput":
        if self.metric_a == self.metric_b:
            label = self.relationship if self.relationship is not None else self.coefficient
            raise ValueError(
                f"connection endpoints must be distinct, got "
                f"{self.metric_a!r} {label} {self.metric_b!r}"
            )
        return self


# ── Lifecycle (settled-decision #20: `lifecycle` is the input keyword,
#    `stages` is the inner list of named thresholds; both are also
#    accepted as the outer block name for resilience) ────────────────────────


class LifecycleStageInput(BaseModel):
    """Single named threshold within a lifecycle ladder."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    threshold: float = Field(ge=0.0, le=1.0)


class LifecycleInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    track: str = Field(min_length=1)
    stages: list[LifecycleStageInput] = Field(min_length=2)
    # Default ``False`` is the engine-side legacy mode and the historical
    # builder behaviour: stage assignment is stateless free-mode (each
    # period independently picks the highest threshold the realised value
    # satisfies). Set ``True`` for a monotonic stage walk where the
    # cursor advances only and an entity can't jump back to an earlier
    # stage on a transient dip — this maps onto the engine's
    # ``StageSequence.enforce_order`` flag verbatim.
    enforce_order: bool = False
    # Hysteresis — under ``enforce_order=True``, allow the cursor to
    # step backward once the entity has sat below the demote threshold
    # for ``downgrade_delay`` consecutive periods. ``None`` (default)
    # preserves strict monotonicity. Ignored when
    # ``enforce_order=False``.
    downgrade_delay: Optional[int] = Field(default=None, ge=1, le=120)

    @model_validator(mode="before")
    @classmethod
    def _normalise_stage_shapes(cls, data: Any) -> Any:
        # Accept stage shapes:
        #   {"onboarding": 0.0}                  (single-key dict)
        #   ("onboarding", 0.0)                  (tuple)
        #   {"name": "onboarding", "threshold": 0.0} (canonical)
        if not isinstance(data, dict):
            return data
        stages = data.get("stages")
        if not isinstance(stages, list):
            return data
        normalised = []
        for stage in stages:
            if isinstance(stage, dict) and len(stage) == 1 and "name" not in stage:
                ((k, v),) = stage.items()
                normalised.append({"name": k, "threshold": v})
            elif isinstance(stage, (list, tuple)) and len(stage) == 2:
                normalised.append({"name": stage[0], "threshold": stage[1]})
            else:
                normalised.append(stage)
        return {**data, "stages": normalised}

    @model_validator(mode="after")
    def _stages_strictly_ascending_unique_names(self) -> "LifecycleInput":
        names = [s.name for s in self.stages]
        if len(set(names)) != len(names):
            raise ValueError(f"lifecycle stage names must be unique, got {names}")
        thresholds = [s.threshold for s in self.stages]
        for prev, curr in zip(thresholds, thresholds[1:]):
            if curr <= prev:
                raise ValueError(
                    f"lifecycle stage thresholds must be strictly " f"ascending, got {thresholds}"
                )
        return self


# ── Schema (dimensions / facts / events) ────────────────────────────────────


class ColumnInput(BaseModel):
    """A single column declaration.

    The ``type`` field uses the builder's plain-language vocabulary
    (``id``, ``ref.{table}``, ``metric.{name}``, ``faker.{which}``,
    ``geo.{field}``, ``static.{value}``, ``segment.count``,
    ``pool.{attr}``, ``timestamp``, ``flag``, ``date``, ``int``,
    ``string``, ``float``, ``bucket``, ``scd``, ``narrative``).
    Sub-fields are present only for the shape they target:
    ``tracks``/``tiers``/``at`` for SCD columns, ``labels`` for buckets,
    ``template`` / ``lexicons`` / ``bands`` for narratives.

    The interpreter (Phase 3) translates ``type`` into the engine's
    ``Column.dtype`` + ``Column.source`` pair. Per friction-item #3
    operator decision: ``date`` and ``int`` are dtype declarations and
    pass through to ``Column.dtype``; the source is generated from
    ``date_key`` for dim_date columns.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    type: str = Field(min_length=1)
    # SCD sub-fields
    tracks: Optional[str] = None
    tiers: Optional[list[str]] = None
    at: Optional[list[float]] = None
    # Bucket sub-field
    labels: Optional[list[str]] = None
    # Narrative sub-fields. ``template`` is a sentence with ``{slot}``
    # placeholders; ``lexicons`` is an ``{archetype: {slot: {band: [phrase, ...]}}}``
    # nested dict; ``bands`` defaults to ``("low", "mid", "high")`` when omitted.
    # All three are interpreted into ``Column.narrative: NarrativeConfig`` —
    # see ``plotsim.config.NarrativeConfig`` for the full validation rules.
    #
    # Lexicon-key gotcha: the archetype keys must match the engine-level
    # archetype names, which in the builder API are the **segment names**
    # (e.g. ``"risers"``, ``"fallers"``) — NOT the recipe keywords passed
    # in each segment's ``archetype:`` field (``"growth"``, ``"decline"``).
    # The builder picks the recipe via ``archetype:`` then names the
    # resulting archetype after the segment, so per-segment lexicons are
    # the right granularity (two segments using the same recipe can still
    # speak with different vocabulary).
    template: Optional[str] = None
    lexicons: Optional[dict[str, dict[str, dict[str, list[str]]]]] = None
    bands: Optional[list[str]] = None


class DimInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    columns: list[ColumnInput] = Field(min_length=1)
    per: Optional[Literal["period", "unit"]] = None
    reference: bool = False
    # M115: sub-entity dims (e.g. dim_user). count × segment.count rows.
    count: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def _per_or_reference_not_both(self) -> "DimInput":
        if self.reference and self.per is not None:
            raise ValueError(
                f"dimension {self.name!r}: `reference: true` and `per` are " f"mutually exclusive"
            )
        return self


class FactInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    columns: list[ColumnInput] = Field(min_length=1)
    metrics: list[str] = Field(default_factory=list)
    # 0.6-M9c: opt-in fact-side CDC. Routes to engine ``Table.cdc``;
    # adds ``_inserted_at`` / ``_updated_at`` / ``_op`` audit columns
    # to the fact at generation time. Default False preserves pre-M9c
    # output byte-for-byte.
    cdc: bool = False


class EventInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    columns: list[ColumnInput] = Field(min_length=1)
    trigger: Literal["proportional", "threshold"]
    # proportional-only fields
    driver: Optional[str] = None
    scale: Optional[float] = Field(default=None, ge=0.0)
    # threshold-only fields
    metric: Optional[str] = None
    above: Optional[float] = None
    below: Optional[float] = None
    # Friction-item #2: accept both `for` (YAML) and `for_periods` (Python).
    for_periods: Optional[int] = Field(
        default=None,
        ge=1,
        validation_alias=AliasChoices("for_periods", "for"),
    )

    @model_validator(mode="after")
    def _trigger_specific_fields(self) -> "EventInput":
        if self.trigger == "proportional":
            if self.driver is None:
                raise ValueError(
                    f"event {self.name!r}: trigger 'proportional' requires " f"a `driver` metric"
                )
            if self.scale is None:
                raise ValueError(
                    f"event {self.name!r}: trigger 'proportional' requires " f"a numeric `scale`"
                )
        else:  # threshold
            if self.metric is None:
                raise ValueError(
                    f"event {self.name!r}: trigger 'threshold' requires a " f"`metric` to watch"
                )
            if self.above is None and self.below is None:
                raise ValueError(
                    f"event {self.name!r}: trigger 'threshold' requires " f"`above` or `below`"
                )
            if self.above is not None and self.below is not None:
                raise ValueError(
                    f"event {self.name!r}: pick one of `above` / `below`, " f"not both"
                )
        return self


# ── Seasonality (M119) ─────────────────────────────────────────────────────


class SeasonalEffectInput(BaseModel):
    """One global seasonal effect spanning a set of calendar months.

    Multiple effects may overlap — strengths sum at each period before
    per-metric and per-segment ``seasonal_sensitivity`` multipliers apply.
    Months are 1..12; uniqueness within a single effect is enforced. The
    interpreter translates this 1:1 to ``plotsim.config.SeasonalEffect``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    months: tuple[int, ...] = Field(min_length=1, max_length=12)
    strength: float

    @field_validator("months")
    @classmethod
    def _months_in_range_unique(cls, v: tuple[int, ...]) -> tuple[int, ...]:
        for m in v:
            if not 1 <= int(m) <= 12:
                raise ValueError(f"seasonality month {m} out of range; valid: 1..12")
        if len(set(v)) != len(v):
            raise ValueError(
                f"seasonality months must be unique within one effect, " f"got {list(v)}"
            )
        return v


# ── Quality / Holdout / EntityFeatures / Bridges (M122) ────────────────────


class QualityIssueInput(BaseModel):
    """One post-generation data-quality corruption.

    Six issue types map 1:1 to the engine's ``QualityIssue.type``. The
    builder accepts a single ``column`` name (or omits it for
    ``duplicate_rows`` / ``late_arrival`` / ``volume_anomaly``); the
    interpreter expands to ``target_columns=[column]`` or the ``"*"``
    sentinel when omitted.

    ``volume_anomaly`` carries three extra fields: ``mode`` (``"spike"``
    or ``"drop"``) plus exactly one of ``period`` (single int) or
    ``periods`` (list of ints) naming the target period(s). The
    interpreter routes them onto the engine's ``mode`` /
    ``target_period`` / ``target_periods`` fields. They are rejected
    when set on any other issue type.

    Engine-level cross-references (table exists, column exists on that
    table, rate honored against table size) are validated by
    ``PlotsimConfig._quality_gates`` at interpreter exit.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    table: str = Field(min_length=1)
    issue: Literal[
        "null_injection",
        "duplicate_rows",
        "type_mismatch",
        "late_arrival",
        "schema_drift",
        "volume_anomaly",
    ]
    rate: float = Field(ge=0.0, le=1.0)
    column: Optional[str] = None
    seed_offset: int = Field(default=0, ge=0)
    # 0.6-M9a: volume_anomaly extras. ``mode`` picks spike vs drop;
    # exactly one of ``period`` / ``periods`` names the 0-based period
    # index/indices. All three are required only for volume_anomaly
    # and rejected on any other issue.
    mode: Optional[Literal["spike", "drop"]] = None
    period: Optional[int] = Field(default=None, ge=0)
    periods: Optional[list[int]] = Field(default=None, max_length=10_000)

    @model_validator(mode="after")
    def _column_required_for_column_typed_issues(self) -> "QualityIssueInput":
        if self.issue in ("null_injection", "type_mismatch", "schema_drift"):
            if not self.column:
                raise ValueError(
                    f"quality issue {self.issue!r} on table "
                    f"{self.table!r} requires a `column` name (the column "
                    f"to corrupt). `column` is only optional for "
                    f"`duplicate_rows`, `late_arrival`, and "
                    f"`volume_anomaly`."
                )
        return self

    @model_validator(mode="after")
    def _volume_anomaly_required_fields(self) -> "QualityIssueInput":
        is_va = self.issue == "volume_anomaly"
        has_mode = self.mode is not None
        has_period = self.period is not None
        has_periods = self.periods is not None
        if is_va:
            if self.column is not None:
                raise ValueError(
                    f"quality issue 'volume_anomaly' on table "
                    f"{self.table!r} is row-level and does not accept "
                    f"`column` (got {self.column!r}); use `period` or "
                    f"`periods` to name the target period(s)"
                )
            if not has_mode:
                raise ValueError(
                    f"quality issue 'volume_anomaly' on table "
                    f"{self.table!r} requires `mode` set to 'spike' or "
                    f"'drop'"
                )
            if has_period == has_periods:
                if has_period and has_periods:
                    raise ValueError(
                        f"quality issue 'volume_anomaly' on table "
                        f"{self.table!r} accepts exactly one of `period` "
                        f"or `periods`, not both"
                    )
                raise ValueError(
                    f"quality issue 'volume_anomaly' on table "
                    f"{self.table!r} requires `period` (single int) or "
                    f"`periods` (list of ints) — neither was set"
                )
            if has_periods and self.periods is not None:
                if len(self.periods) == 0:
                    raise ValueError(
                        f"quality issue 'volume_anomaly' on table "
                        f"{self.table!r} `periods` must be a non-empty "
                        f"list when set"
                    )
                for entry in self.periods:
                    if not isinstance(entry, int) or entry < 0:
                        raise ValueError(
                            f"quality issue 'volume_anomaly' on table "
                            f"{self.table!r} `periods` entries must be "
                            f"non-negative integers (got {entry!r})"
                        )
        else:
            extras = []
            if has_mode:
                extras.append("mode")
            if has_period:
                extras.append("period")
            if has_periods:
                extras.append("periods")
            if extras:
                raise ValueError(
                    f"quality issue {self.issue!r} on table "
                    f"{self.table!r} does not accept fields {extras!r}; "
                    f"those are only valid when issue='volume_anomaly'"
                )
        return self


class HoldoutInput(BaseModel):
    """Temporal train/holdout split for ML target workflows.

    Maps to ``HoldoutConfig(enabled=True, target_metric=target,
    holdout_periods=periods)`` — see engine-side ``HoldoutConfig``
    docstring for cutoff math and gate rules. The engine raises at
    ``PlotsimConfig`` load if ``periods`` exceeds ``n_periods -
    min_training_periods``; we surface a builder-side message for the
    most common error (target not declared) and let the engine catch
    the rest.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    target: str = Field(min_length=1)
    periods: int = Field(ge=1, le=10_000)
    min_training_periods: int = Field(default=3, ge=1, le=10_000)


class EntityFeaturesInput(BaseModel):
    """Per-entity flat feature table emission settings.

    The bool shorthand ``entity_features: true`` translates to this
    model with all defaults (every numeric metric, labels on). The
    dict form lets callers narrow the metric set or strip labels.

    Engine-side gates (manifest required, no quality_issues, metric
    references resolve to numeric fact columns) raise at PlotsimConfig
    load.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    metrics: list[str] = Field(default_factory=list, max_length=50)
    include_labels: bool = True


class BridgeColumnInput(BaseModel):
    """A single bridge-row column. Same shorthand as fact/dim columns.

    Bridge rows are static per (entity_a, entity_b) pair — no period
    axis — so only ``metric.{name}``, ``static.{value}``, and
    ``faker.{kind}`` types are valid here. The interpreter rejects
    anything else with a context-rich error.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    type: str = Field(min_length=1)


class BridgeInput(BaseModel):
    """Many-to-many bridge between two dim tables.

    Both ``left`` and ``right`` reference dim table names. ``cardinality``
    is an inclusive ``[min, max]`` pair: each ``left`` entity associates
    with ``min..max`` ``right`` entities (sampled uniformly when no
    ``driver`` is set; biased by trajectory position when ``driver``
    is set).

    ``driver`` (optional) is a metric name. Non-null values flip the
    engine's ``trajectory_driven`` to True; null leaves it at the
    engine default (which is also True). The driver name is mostly
    documentary at the builder layer — engine bridge generation
    queries the entity's trajectory directly, not a specific metric.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    left: str = Field(min_length=1)
    right: str = Field(min_length=1)
    cardinality: tuple[int, int]
    driver: Optional[str] = None
    columns: list[BridgeColumnInput] = Field(default_factory=list, max_length=20)

    @field_validator("name")
    @classmethod
    def _name_is_simple(cls, v: str) -> str:
        if not v.replace("_", "").isalnum():
            raise ValueError(f"bridge name {v!r} must be alphanumeric or underscores only")
        return v

    @field_validator("cardinality")
    @classmethod
    def _cardinality_bounds(cls, v: tuple[int, int]) -> tuple[int, int]:
        lo, hi = v
        if lo < 0:
            raise ValueError(f"bridge cardinality min ({lo}) must be >= 0")
        if hi < 1:
            raise ValueError(f"bridge cardinality max ({hi}) must be >= 1")
        if lo > hi:
            raise ValueError(
                f"bridge cardinality [min, max] = [{lo}, {hi}]: " f"min must be <= max"
            )
        return v

    @model_validator(mode="after")
    def _left_and_right_distinct(self) -> "BridgeInput":
        if self.left == self.right:
            raise ValueError(
                f"bridge {self.name!r}: left and right must be distinct "
                f"dim tables (got both as {self.left!r}; self-join "
                f"bridges are not supported)"
            )
        return self


# ── Noise / Output (top-level shorthand-bearing models) ────────────────────


class NoiseInput(BaseModel):
    """Distributional noise applied during generation.

    1:1 mirror of the engine's ``NoiseConfig``. The interpreter passes
    these values through unchanged. The ``noise:`` field on
    ``UserInput`` also accepts a string preset name
    (``"clean"`` / ``"slightly_messy"`` / ``"realistic"`` / ``"dirty"``)
    which is resolved to one of these ``NoiseInput`` values by the
    ``_coerce_noise`` helper before validation.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    gaussian_sigma: float = Field(default=0.0, ge=0.0, le=5.0)
    outlier_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    mcar_rate: float = Field(default=0.0, ge=0.0, le=1.0)


class OutputInput(BaseModel):
    """Output-format selector and target directory.

    1:1 mirror of the engine's ``OutputConfig``. The string shorthand
    ``output: parquet`` / ``output: csv`` resolves to the
    ``OutputInput(format=<word>, directory="output")`` default by
    ``_coerce_output``; pass the dict form to override the directory.

    ``cell_budget`` (M7) mirrors ``OutputConfig.cell_budget`` — the
    soft cap consumed by the load-time scale estimator. ``None``
    falls through to ``PLOTSIM_CELL_BUDGET`` env var, then the
    2,000,000-cell default; ``0`` disables the soft gate; positive
    integers raise (or lower) the cap to that value. Promoting the
    field through the builder lets ``create(output={'cell_budget':
    N})`` override the cap without env vars.

    ``denormalized`` (0.6-M14a) mirrors ``OutputConfig.denormalized``
    — opt-in wide-table companion writer. ``False`` (default) keeps
    output byte-identical to pre-M14a. ``True`` emits
    ``<fct_name>_wide.{csv|parquet}`` alongside each normalized fact
    table, with FK'd dims left-joined onto the fact (SCD2 dims
    filtered to current state).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    format: Literal["csv", "parquet"] = "csv"
    directory: str = "output"
    cell_budget: Optional[int] = Field(default=None, ge=0)
    denormalized: bool = False


class SourceInput(BaseModel):
    """0.6-M13: one upstream system in the multi-source / overlap layout.

    1:1 mirror of the engine's
    :class:`~plotsim.config.SourceDeclaration`. The builder layer
    surfaces the same vocabulary directly — the three drift rates and
    the key-scheme enum have no friendlier "preset" form to translate
    through; users either want the canonical names or they want explicit
    rates.

    Two name-related drift kinds + one key-scheme drift + one
    attribute-conflict drift; rates default to ``0.0`` so a user can
    enable just one flavor on a source if that's what their teaching
    scenario calls for. The engine validator caps the source list to 5;
    the builder caps lower (``max_length=5``) here so a builder-side
    typo doesn't reach the engine.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    key_scheme: Literal["prefix_padded", "numeric", "uuid_short"] = "prefix_padded"
    name_drift_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    attribute_drift_rate: float = Field(default=0.0, ge=0.0, le=1.0)


# Mapping from short, lower-cased preset names to the engine's
# canonical ``NoiseConfig`` parameter triples. Friendly aliases (``clean``,
# ``messy``, ``very_messy``) map onto the same four engine presets so
# users can pick whichever vocabulary reads naturally. The engine's own
# ``NOISE_PRESETS`` dict in ``plotsim.config`` keys on title-case names
# (``"Perfectly clean"``, ``"Slightly messy"``, ...); the builder layer
# uses lower-snake-case so YAML scalars without quotes round-trip cleanly.
NOISE_PRESET_PARAMS: dict[str, dict[str, float]] = {
    "perfectly_clean": {"gaussian_sigma": 0.00, "outlier_rate": 0.00, "mcar_rate": 0.000},
    "slightly_messy": {"gaussian_sigma": 0.03, "outlier_rate": 0.01, "mcar_rate": 0.005},
    "realistic": {"gaussian_sigma": 0.05, "outlier_rate": 0.02, "mcar_rate": 0.010},
    "dirty": {"gaussian_sigma": 0.10, "outlier_rate": 0.05, "mcar_rate": 0.030},
    # Friendly aliases (user prompt vocabulary)
    "clean": {"gaussian_sigma": 0.00, "outlier_rate": 0.00, "mcar_rate": 0.000},
    "messy": {"gaussian_sigma": 0.05, "outlier_rate": 0.02, "mcar_rate": 0.010},
    "very_messy": {"gaussian_sigma": 0.10, "outlier_rate": 0.05, "mcar_rate": 0.030},
}


def _coerce_noise(value: Any) -> Any:
    """Accept preset string shorthand on ``UserInput.noise``.

    Strings resolve via ``NOISE_PRESET_PARAMS`` to the corresponding
    ``NoiseInput`` parameter dict. Dicts pass through unchanged
    (Pydantic constructs ``NoiseInput`` from them). ``None`` and
    ``NoiseInput`` instances also pass through untouched.
    """
    if isinstance(value, str):
        key = value.strip().lower().replace(" ", "_")
        if key not in NOISE_PRESET_PARAMS:
            raise ValueError(
                f"unknown noise preset {value!r}. Valid presets: "
                f"{sorted(NOISE_PRESET_PARAMS)} (or pass a dict with "
                f"`gaussian_sigma`, `outlier_rate`, `mcar_rate`)"
            )
        return dict(NOISE_PRESET_PARAMS[key])
    return value


def _coerce_output(value: Any) -> Any:
    """Accept format-string shorthand on ``UserInput.output``.

    ``"csv"`` / ``"parquet"`` resolve to the corresponding
    ``OutputInput(format=<word>, directory="output")`` default. Dicts
    pass through unchanged.
    """
    if isinstance(value, str):
        word = value.strip().lower()
        if word not in ("csv", "parquet"):
            raise ValueError(
                f"unknown output format {value!r}. Valid: 'csv' or "
                f"'parquet' (or pass a dict `{{format: ..., directory: ...}}`)"
            )
        return {"format": word, "directory": "output"}
    return value


# ── Coercion helpers for shorthand forms ────────────────────────────────────


def _coerce_entity_features(value: Any) -> Any:
    """Accept ``True`` / ``False`` shorthand on the ``entity_features`` field.

    ``True`` → empty ``EntityFeaturesInput()`` (defaults). ``False`` →
    ``None`` (no entity-features section). Dicts pass through unchanged.
    """
    if value is True:
        return {}
    if value is False:
        return None
    return value


# ── UserInput root ──────────────────────────────────────────────────────────


# Window window may arrive as a tuple or a dict; normalise to dict.
def _coerce_window_tuple(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        if len(value) == 2:
            return {"start": value[0], "end": value[1]}
        if len(value) == 3:
            return {"start": value[0], "end": value[1], "every": value[2]}
        raise ValueError(
            f"window tuple must have 2 or 3 elements (start, end, [every]), " f"got {len(value)}"
        )
    return value


# Connection may arrive as a 3-token string, a 3-tuple, or a dict. The
# middle slot may be a relationship word (``"driven_by"``, ``"opposes"``,
# ...) or a numeric coefficient in ``[-1.0, 1.0]``; we route to the
# matching ``ConnectionInput`` field so downstream validation can flag
# missing/duplicate slots without re-parsing the same string.
def _coerce_connection(value: Any) -> Any:
    if isinstance(value, str):
        tokens = value.split()
        if len(tokens) != 3:
            raise ValueError(
                f"connection string {value!r} must have exactly three "
                f"whitespace-separated tokens: '<metric_a> <relationship "
                f"or coefficient> <metric_b>'"
            )
        a, mid, b = tokens
        try:
            coef = float(mid)
        except ValueError:
            return {"metric_a": a, "relationship": mid, "metric_b": b}
        return {"metric_a": a, "coefficient": coef, "metric_b": b}
    if isinstance(value, (list, tuple)):
        if len(value) != 3:
            raise ValueError(
                f"connection tuple {value!r} must have three elements: "
                f"(metric_a, relationship_or_coefficient, metric_b)"
            )
        a, mid, b = value
        if isinstance(mid, bool):
            # bool is a subclass of int in Python — be explicit and reject
            # so the caller doesn't silently get coefficient=1.0 from `True`.
            raise ValueError(
                f"connection middle slot must be a string (relationship "
                f"word) or a number (coefficient), not bool: {value!r}"
            )
        if isinstance(mid, (int, float)):
            return {"metric_a": a, "coefficient": float(mid), "metric_b": b}
        return {"metric_a": a, "relationship": mid, "metric_b": b}
    return value


WindowLike = Union[WindowInput, dict[str, Any], tuple, list]
ConnectionLike = Union[ConnectionInput, str, tuple, list, dict[str, Any]]


# ── Connection phase (0.6-M11) ──────────────────────────────────────────────
#
# A builder-side phase window over the time axis with its own connection
# list. Mirrors the engine's ``CorrelationPhase`` but uses the
# relationship-word vocabulary (or explicit coefficients) the builder
# already accepts on ``ConnectionInput``. Translated to
# ``CorrelationPhase`` by ``plotsim.builder.interpreter._build_correlation_phases``.


class ConnectionPhase(BaseModel):
    """0.6-M11: a builder-side phase window with its own ``connections`` list.

    ``start_period`` / ``end_period`` are inclusive bounds matching the
    engine's ``CorrelationPhase`` contract. ``connections`` accepts the
    same shorthand forms (3-token string, 3-tuple, dict) as the
    top-level ``UserInput.connections`` field via the
    ``_coerce_connection`` pre-normalizer.

    Periods not covered by any phase fall back to the baseline
    ``UserInput.connections`` list. The engine validator
    ``_correlation_phases_require_baseline`` rejects configs where
    ``connection_phases`` is non-empty but the baseline is empty.
    """

    start_period: int = Field(ge=0)
    end_period: int = Field(ge=0)
    connections: list[ConnectionInput] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _normalise_shorthand_connections(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        normalised = dict(data)
        if "connections" in normalised and isinstance(normalised["connections"], list):
            normalised["connections"] = [_coerce_connection(c) for c in normalised["connections"]]
        return normalised

    @model_validator(mode="after")
    def _end_after_start(self) -> "ConnectionPhase":
        if self.end_period < self.start_period:
            raise ValueError(
                f"connection_phases entry has end_period={self.end_period} "
                f"< start_period={self.start_period}; phases must satisfy "
                f"start_period <= end_period (both inclusive)"
            )
        return self


class UserInput(BaseModel):
    """Root user-facing input model.

    ``unit`` and the segment archetype DSL are validated structurally
    (no engine concepts leak in here). Cross-reference checks (orphan
    references, causal-lag cycles, archetype DSL validity) run after
    nested models are constructed.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        # Accept canonical and alias names side-by-side.
        populate_by_name=True,
    )

    about: str = Field(min_length=1)
    unit: str = Field(min_length=1)
    window: WindowInput
    metrics: list[MetricInput] = Field(min_length=1)
    segments: list[SegmentInput] = Field(min_length=1)
    connections: list[ConnectionInput] = Field(default_factory=list)
    # 0.6-M11: time-varying connections. Empty default → engine receives
    # ``correlation_phases=[]`` and runs the single-Cholesky path
    # (byte-identical to pre-M11). When non-empty, the baseline
    # ``connections`` list must also be non-empty — the engine validator
    # enforces this on the translated ``correlation_phases`` / ``correlations``
    # pair. ``max_length=64`` matches the engine cap.
    connection_phases: list[ConnectionPhase] = Field(default_factory=list, max_length=64)
    # Friction-item #1: settled on `lifecycle` as input keyword.
    # `stages` is also accepted as an alias since the mission spec text
    # used that name and some users will follow the spec literally.
    lifecycle: Optional[LifecycleInput] = Field(
        default=None,
        validation_alias=AliasChoices("lifecycle", "stages"),
    )
    dimensions: list[DimInput] = Field(default_factory=list)
    facts: list[FactInput] = Field(default_factory=list)
    events: list[EventInput] = Field(default_factory=list)
    # M119: optional global seasonality. Empty list (default) → no
    # modulation is configured and engine output is byte-identical to
    # pre-M119 baselines.
    seasonality: list[SeasonalEffectInput] = Field(default_factory=list)
    # M122: power-user features — bridges, quality injection, holdout
    # split, and entity-features emission. All four are optional; their
    # default empty / disabled forms produce engine output identical to
    # pre-M122 baselines.
    bridges: list[BridgeInput] = Field(default_factory=list, max_length=20)
    quality: list[QualityIssueInput] = Field(default_factory=list, max_length=50)
    holdout: Optional[HoldoutInput] = None
    entity_features: Optional[EntityFeaturesInput] = None
    # 0.6-M13: multi-source / overlap mode. Empty default → engine receives
    # ``multi_source=None`` and the dim builder skips the per-source pass
    # (byte-identical to pre-M13). When non-empty, at least 2 sources are
    # required — a 1-source declaration has no overlap to resolve. The
    # engine validator caps at 5 sources (teaching range); the builder
    # mirrors that cap so user typos surface at the builder layer with the
    # user's own input vocabulary in the error message.
    sources: list[SourceInput] = Field(default_factory=list, max_length=5)
    # M124: optional explicit seed. When ``None`` the interpreter draws one
    # from ``secrets.randbelow(2**32)`` (preserves prior non-determinism for
    # callers that don't pin a seed). When set, the interpreter threads the
    # value onto ``PlotsimConfig.seed`` verbatim — same seed in, same
    # output out, every run.
    seed: Optional[int] = Field(default=None, ge=0, le=2**32 - 1)
    # Noise / output / locale — three engine knobs surfaced at the
    # builder layer. ``None`` defaults preserve historical behaviour
    # byte-for-byte (no noise, csv to ``output/``, en_US faker locale).
    noise: Optional[NoiseInput] = None
    output: Optional[OutputInput] = None
    locale: Union[str, list[str]] = "en_US"

    # ── Pre-normalisation: accept tuple/string shorthand on inputs ─────────

    @model_validator(mode="before")
    @classmethod
    def _normalise_shorthands(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        normalised = dict(data)
        if "window" in normalised:
            normalised["window"] = _coerce_window_tuple(normalised["window"])
        if "connections" in normalised and isinstance(normalised["connections"], list):
            normalised["connections"] = [_coerce_connection(c) for c in normalised["connections"]]
        if "entity_features" in normalised:
            normalised["entity_features"] = _coerce_entity_features(normalised["entity_features"])
        if "noise" in normalised and normalised["noise"] is not None:
            normalised["noise"] = _coerce_noise(normalised["noise"])
        if "output" in normalised and normalised["output"] is not None:
            normalised["output"] = _coerce_output(normalised["output"])
        return normalised

    # ── Structural cross-reference validators ──────────────────────────────

    @model_validator(mode="after")
    def _no_duplicate_metric_names(self) -> "UserInput":
        names = [m.name for m in self.metrics]
        seen = set()
        dups = sorted({n for n in names if n in seen or seen.add(n)})  # type: ignore[func-returns-value]
        if dups:
            raise ValueError(f"duplicate metric name(s): {dups}")
        return self

    @model_validator(mode="after")
    def _sources_valid(self) -> "UserInput":
        """0.6-M13: validate the multi-source declaration block.

        Empty list → feature off, no checks. Non-empty list must declare
        at least 2 distinct source names. The engine catches both rules
        on the translated ``MultiSourceConfig``, but raising here surfaces
        the error against the user's own ``sources:`` vocabulary instead
        of the engine's translated field path.
        """
        if not self.sources:
            return self
        if len(self.sources) < 2:
            raise ValueError(
                f"sources: at least 2 distinct source declarations are "
                f"required when the multi-source block is set (got "
                f"{len(self.sources)}). A single-source declaration has "
                f"no overlap to resolve against — either declare a second "
                f"source or remove the block entirely"
            )
        names = [s.name for s in self.sources]
        if len(set(names)) != len(names):
            raise ValueError(
                f"sources: duplicate source name(s) in {names!r}. Each "
                f"declared source must be unique because the name doubles "
                f"as the per-source dim suffix"
            )
        return self

    @model_validator(mode="after")
    def _no_duplicate_segment_names(self) -> "UserInput":
        names = [s.name for s in self.segments]
        seen = set()
        dups = sorted({n for n in names if n in seen or seen.add(n)})  # type: ignore[func-returns-value]
        if dups:
            raise ValueError(f"duplicate segment name(s): {dups}")
        return self

    @model_validator(mode="after")
    def _follows_targets_exist(self) -> "UserInput":
        metric_names = {m.name for m in self.metrics}
        for m in self.metrics:
            if m.follows is not None and m.follows not in metric_names:
                raise ValueError(
                    f"metric {m.name!r}: `follows: {m.follows!r}` references "
                    f"an unknown metric. Available metrics: "
                    f"{sorted(metric_names)}"
                )
        return self

    @model_validator(mode="after")
    def _no_causal_lag_cycles(self) -> "UserInput":
        # Build follows graph and check for cycles.
        graph: dict[str, str] = {m.name: m.follows for m in self.metrics if m.follows is not None}
        for start in graph:
            seen = {start}
            node: str | None = graph[start]
            while node is not None:
                if node in seen:
                    raise ValueError(
                        f"causal-lag cycle detected starting from metric "
                        f"{start!r}: {' → '.join(sorted(seen))}"
                    )
                seen.add(node)
                node = graph.get(node)
        return self

    @model_validator(mode="after")
    def _connection_endpoints_exist(self) -> "UserInput":
        metric_names = {m.name for m in self.metrics}
        for c in self.connections:
            for endpoint in (c.metric_a, c.metric_b):
                if endpoint not in metric_names:
                    raise ValueError(
                        f"connection {c.metric_a!r} {c.relationship} "
                        f"{c.metric_b!r}: endpoint {endpoint!r} is not a "
                        f"declared metric. Available: {sorted(metric_names)}"
                    )
        return self

    @model_validator(mode="after")
    def _baseline_targets_exist(self) -> "UserInput":
        metric_names = {m.name for m in self.metrics}
        for s in self.segments:
            for metric in s.baseline:
                if metric not in metric_names:
                    raise ValueError(
                        f"segment {s.name!r}: baseline references unknown "
                        f"metric {metric!r}. Available: {sorted(metric_names)}"
                    )
        return self

    @model_validator(mode="after")
    def _lifecycle_track_exists(self) -> "UserInput":
        if self.lifecycle is None:
            return self
        metric_names = {m.name for m in self.metrics}
        if self.lifecycle.track not in metric_names:
            raise ValueError(
                f"lifecycle.track={self.lifecycle.track!r} is not a "
                f"declared metric. Available: {sorted(metric_names)}"
            )
        return self

    @model_validator(mode="after")
    def _bridge_references_resolve(self) -> "UserInput":
        """M122: bridge.left/right must reference a declared dim or the
        auto-generated ``dim_{unit}`` / ``dim_date``; bridge.driver must
        reference a declared metric.

        Auto-generated dims are always available — when the user omits
        the schema entirely, ``dim_date`` and ``dim_{unit}`` are still
        emitted by the interpreter. Explicit-schema users may also bridge
        between user-declared dims.
        """
        if not self.bridges:
            return self
        declared_dims: set[str] = {d.name for d in self.dimensions}
        declared_dims.add("dim_date")
        declared_dims.add(f"dim_{self.unit}")
        metric_names = {m.name for m in self.metrics}
        for b in self.bridges:
            for side, dim in (("left", b.left), ("right", b.right)):
                if dim not in declared_dims:
                    raise ValueError(
                        f"bridge {b.name!r} {side}={dim!r}: not a declared "
                        f"dimension. Available: {sorted(declared_dims)}"
                    )
            if b.driver is not None and b.driver not in metric_names:
                raise ValueError(
                    f"bridge {b.name!r} driver={b.driver!r}: not a declared "
                    f"metric. Available: {sorted(metric_names)}"
                )
        return self

    @model_validator(mode="after")
    def _entity_features_metrics_exist(self) -> "UserInput":
        """M122: entity_features.metrics must reference declared metrics."""
        if self.entity_features is None or not self.entity_features.metrics:
            return self
        metric_names = {m.name for m in self.metrics}
        for name in self.entity_features.metrics:
            if name not in metric_names:
                raise ValueError(
                    f"entity_features.metrics: {name!r} is not a declared "
                    f"metric. Available: {sorted(metric_names)}"
                )
        return self

    @model_validator(mode="after")
    def _archetype_specs_parse(self) -> "UserInput":
        n_periods = self._compute_n_periods()
        for s in self.segments:
            try:
                parse_archetype(s.archetype, n_periods=n_periods)
            except ArchetypeParseError as err:
                raise ValueError(f"segment {s.name!r} archetype {s.archetype!r}: {err}") from err
        return self

    # ── Semantic warnings (do not block construction) ──────────────────────

    @model_validator(mode="after")
    def _semantic_warnings(self) -> "UserInput":
        n_periods = self._compute_n_periods()

        # Short window + seasonal: < 24 monthly periods (or equivalent).
        # We compare the period count, not the calendar window — daily/weekly
        # configs need proportional density to capture two cycles.
        seasonal_used = any("seasonal" in s.archetype for s in self.segments)
        if seasonal_used and n_periods < 24:
            warnings.warn(
                f"seasonal pattern declared but the {n_periods}-period "
                f"window may be too short to recover two clean cycles "
                f"(rule of thumb: >= 24 periods)",
                UserWarning,
                stacklevel=2,
            )

        if len(self.segments) == 1:
            warnings.warn(
                "only one segment declared — variation across the dataset "
                "will reflect distribution noise, not archetype mix",
                UserWarning,
                stacklevel=2,
            )

        big_correlation_words = {"mirrors", "inverts"}
        n_metrics = len(self.metrics)
        if n_metrics >= 8 and any(
            c.relationship in big_correlation_words for c in self.connections
        ):
            warnings.warn(
                f"using mirrors/inverts (|0.75|) with {n_metrics} metrics "
                f"can over-constrain the correlation matrix and force the "
                f"engine's PSD projection to make large adjustments",
                UserWarning,
                stacklevel=2,
            )

        return self

    # ── Helpers ────────────────────────────────────────────────────────────

    def _compute_n_periods(self) -> int:
        """Approximate window length in periods for parser/warning checks.

        We only need this for archetype DSL validation and the
        seasonal-window heuristic. The exact engine count comes from
        ``TimeWindow.period_count`` after interpretation.
        """
        # Parse YYYY-MM[-DD] start/end and compute a coarse delta.
        from datetime import date as _date

        def parse_partial(s: str) -> _date:
            parts = s.split("-")
            if len(parts) == 2:
                return _date(int(parts[0]), int(parts[1]), 1)
            if len(parts) == 3:
                return _date(int(parts[0]), int(parts[1]), int(parts[2]))
            raise ValueError(f"window date {s!r} not in YYYY-MM[-DD] form")

        try:
            start = parse_partial(self.window.start)
            end = parse_partial(self.window.end)
        except (ValueError, TypeError):
            # Fallback — a coarse default that won't raise spurious warnings.
            return 24

        days = (end - start).days
        if self.window.every == "daily":
            return max(days + 1, 1)
        if self.window.every == "weekly":
            return max(days // 7 + 1, 1)
        # monthly: rough estimate (12 months/year) good enough for warnings
        months = (end.year - start.year) * 12 + (end.month - start.month) + 1
        return max(months, 1)
