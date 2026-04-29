"""UserInput model — plain-language input for the builder.

The builder accepts the same shape from two surfaces:

- ``create_from_yaml(path)`` — YAML file conforming to the template at
  ``plotsim/configs/new/saas_template.yaml``.
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
    VALID_METRIC_TYPES,
    VALID_POLARITIES,
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

    @field_validator("name")
    @classmethod
    def _name_is_simple(cls, v: str) -> str:
        if not v.replace("_", "").isalnum():
            raise ValueError(
                f"metric name {v!r} must be alphanumeric or underscores only"
            )
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
            raise ValueError(
                f"metric {self.name!r}: `delay` must be >= 1, got {self.delay}"
            )
        if self.follows is not None and self.follows == self.name:
            raise ValueError(
                f"metric {self.name!r} cannot follow itself"
            )
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

    @field_validator("name")
    @classmethod
    def _name_is_simple(cls, v: str) -> str:
        if not v.replace("_", "").isalnum():
            raise ValueError(
                f"segment name {v!r} must be alphanumeric or underscores only"
            )
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
        - "engagement driven_by mrr"               (3-token string)
        - ("engagement", "driven_by", "mrr")        (3-tuple)
        - {"a": ..., "relationship": ..., "b": ...} (dict)

    All three normalise to (metric_a, relationship, metric_b).
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    metric_a: str
    relationship: str
    metric_b: str

    @field_validator("relationship")
    @classmethod
    def _relationship_in_vocabulary(cls, v: str) -> str:
        if v not in VALID_RELATIONSHIP_WORDS:
            raise ValueError(
                f"unknown relationship word {v!r}. Valid: "
                f"{sorted(VALID_RELATIONSHIP_WORDS)}"
            )
        return v

    @model_validator(mode="after")
    def _endpoints_distinct(self) -> "ConnectionInput":
        if self.metric_a == self.metric_b:
            raise ValueError(
                f"connection endpoints must be distinct, got "
                f"{self.metric_a!r} {self.relationship} {self.metric_b!r}"
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
            raise ValueError(
                f"lifecycle stage names must be unique, got {names}"
            )
        thresholds = [s.threshold for s in self.stages]
        for prev, curr in zip(thresholds, thresholds[1:]):
            if curr <= prev:
                raise ValueError(
                    f"lifecycle stage thresholds must be strictly "
                    f"ascending, got {thresholds}"
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
                f"dimension {self.name!r}: `reference: true` and `per` are "
                f"mutually exclusive"
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
                    f"event {self.name!r}: trigger 'proportional' requires "
                    f"a `driver` metric"
                )
            if self.scale is None:
                raise ValueError(
                    f"event {self.name!r}: trigger 'proportional' requires "
                    f"a numeric `scale`"
                )
        else:  # threshold
            if self.metric is None:
                raise ValueError(
                    f"event {self.name!r}: trigger 'threshold' requires a "
                    f"`metric` to watch"
                )
            if self.above is None and self.below is None:
                raise ValueError(
                    f"event {self.name!r}: trigger 'threshold' requires "
                    f"`above` or `below`"
                )
            if self.above is not None and self.below is not None:
                raise ValueError(
                    f"event {self.name!r}: pick one of `above` / `below`, "
                    f"not both"
                )
        return self


# ── UserInput root ──────────────────────────────────────────────────────────


# Window window may arrive as a tuple or a dict; normalise to dict.
def _coerce_window_tuple(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        if len(value) == 2:
            return {"start": value[0], "end": value[1]}
        if len(value) == 3:
            return {"start": value[0], "end": value[1], "every": value[2]}
        raise ValueError(
            f"window tuple must have 2 or 3 elements (start, end, [every]), "
            f"got {len(value)}"
        )
    return value


# Connection may arrive as a 3-token string, a 3-tuple, or a dict.
def _coerce_connection(value: Any) -> Any:
    if isinstance(value, str):
        tokens = value.split()
        if len(tokens) != 3:
            raise ValueError(
                f"connection string {value!r} must have exactly three "
                f"whitespace-separated tokens: '<metric_a> <relationship> "
                f"<metric_b>'"
            )
        a, rel, b = tokens
        return {"metric_a": a, "relationship": rel, "metric_b": b}
    if isinstance(value, (list, tuple)):
        if len(value) != 3:
            raise ValueError(
                f"connection tuple {value!r} must have three elements: "
                f"(metric_a, relationship, metric_b)"
            )
        a, rel, b = value
        return {"metric_a": a, "relationship": rel, "metric_b": b}
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
            normalised["connections"] = [
                _coerce_connection(c) for c in normalised["connections"]
            ]
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
        graph: dict[str, str] = {
            m.name: m.follows for m in self.metrics if m.follows is not None
        }
        for start in graph:
            seen = {start}
            node = graph[start]
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
    def _archetype_specs_parse(self) -> "UserInput":
        n_periods = self._compute_n_periods()
        for s in self.segments:
            try:
                parse_archetype(s.archetype, n_periods=n_periods)
            except ArchetypeParseError as err:
                raise ValueError(
                    f"segment {s.name!r} archetype {s.archetype!r}: {err}"
                ) from err
        return self

    # ── Semantic warnings (do not block construction) ──────────────────────

    @model_validator(mode="after")
    def _semantic_warnings(self) -> "UserInput":
        n_periods = self._compute_n_periods()

        # Short window + seasonal: < 24 monthly periods (or equivalent).
        # We compare the period count, not the calendar window — daily/weekly
        # configs need proportional density to capture two cycles.
        seasonal_used = any(
            "seasonal" in s.archetype for s in self.segments
        )
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
