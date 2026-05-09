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
from typing import Any, Literal, Optional, Union

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from .parser import ArchetypeParseError, parse_archetype
from .recipes import (
    VALID_BASELINE_WORDS,
    VALID_RELATIONSHIP_WORDS,
)


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
    ``static.{value}``, ``segment.count``, ``pool.{attr}``, ``timestamp``,
    ``flag``, ``date``, ``int``, ``string``, ``float``, ``bucket``,
    ``scd``). Sub-fields are present only for the shape they target:
    ``tracks``/``tiers``/``at`` for SCD columns, ``labels`` for buckets.

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

    Five issue types map 1:1 to the engine's ``QualityIssue.type``. The
    builder accepts a single ``column`` name (or omits it for
    ``duplicate_rows`` / ``late_arrival``); the interpreter expands to
    ``target_columns=[column]`` or the ``"*"`` sentinel when omitted.

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
    ]
    rate: float = Field(ge=0.0, le=1.0)
    column: Optional[str] = None
    seed_offset: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _column_required_for_column_typed_issues(self) -> "QualityIssueInput":
        if self.issue in ("null_injection", "type_mismatch", "schema_drift"):
            if not self.column:
                raise ValueError(
                    f"quality issue {self.issue!r} on table "
                    f"{self.table!r} requires a `column` name (the column "
                    f"to corrupt). `column` is only optional for "
                    f"`duplicate_rows` and `late_arrival`."
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
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    format: Literal["csv", "parquet"] = "csv"
    directory: str = "output"


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
