"""plotsim.config — Pydantic v2 config models, YAML loader, validation.

What it does:
    Defines the schema that is the contract between the LLM scaffolder
    (Phase A) and the generation engine (Phase B). Loads a YAML file,
    validates it end-to-end (types, enum membership, cross-reference
    integrity), and returns a frozen PlotsimConfig. Any schema violation
    raises pydantic.ValidationError with a locatable error path.

    Mission 001a additions:
      - parse_source() returns a typed object for every column source
        string (pk, fk, metric, generated, static, derived, threshold,
        proportional, lag).
      - MetricConfig gains causal_lag (driver + lag_periods).
      - Table gains row_count_source (event tables only).
      - PlotsimConfig gains optional top-level stages (StageSequence).

Input:
    Path to a YAML file conforming to the plotsim schema.

Output:
    A frozen PlotsimConfig instance. Round-trippable via dump_config.
"""

from __future__ import annotations

import calendar
import re
import sys
import warnings
from datetime import date, timedelta
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# SEC-02: SQL-safe identifier pattern applied to Table.name and Column.name.
# Leading underscore or letter, then letters / digits / underscores, capped at
# 128 characters. Rejects anything that could escape the output sandbox via
# filesystem path construction (``../foo``, ``/etc/passwd``) and anything that
# would trip a SQL import on the generated CSVs.
_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}")


def _validate_identifier(kind: str, value: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{kind} must be a string, got {type(value).__name__}")
    if not _IDENTIFIER_RE.fullmatch(value):
        raise ValueError(
            f"{kind} {value!r} is not a valid identifier: must match "
            f"[A-Za-z_][A-Za-z0-9_]{{0,127}} (SQL-safe, 1-128 chars, no "
            f"path separators, no leading digit)"
        )
    return value


CURVE_TYPES: frozenset[str] = frozenset({
    "sigmoid", "exp_decay", "step", "logistic",
    "plateau", "oscillating", "compound", "sawtooth",
})
DISTRIBUTIONS: frozenset[str] = frozenset({
    "lognorm", "gamma", "poisson", "beta", "normal", "weibull",
})
POLARITIES: frozenset[str] = frozenset({"positive", "negative"})
TABLE_TYPES: frozenset[str] = frozenset({"dim", "fact", "event"})
# Grain values — each tells the table builder exactly which loop to run.
#   per_entity              dim_company: one row per entity
#   per_period              dim_date: one row per time step, no entity axis
#   per_reference           dim_plan, dim_department: static lookup (no time, no entity)
#   per_entity_per_period   fct_engagement: entity × time step
#   variable                evt_login, evt_churn: trajectory-driven row count
GRAINS: frozenset[str] = frozenset({
    "per_entity",
    "per_period",
    "per_reference",
    "per_entity_per_period",
    "variable",
})
COMPOSITE_GRAINS: frozenset[str] = frozenset({"per_entity_per_period"})
DTYPES: frozenset[str] = frozenset({
    "int", "float", "string", "date", "boolean", "id",
})
GRANULARITIES: frozenset[str] = frozenset({"monthly", "weekly", "daily"})

CurveType = Literal[
    "sigmoid", "exp_decay", "step", "logistic",
    "plateau", "oscillating", "compound", "sawtooth",
]
Distribution = Literal["lognorm", "gamma", "poisson", "beta", "normal", "weibull"]
Polarity = Literal["positive", "negative"]
TableType = Literal["dim", "fact", "event"]
Grain = Literal[
    "per_entity",
    "per_period",
    "per_reference",
    "per_entity_per_period",
    "variable",
]
Dtype = Literal["int", "float", "string", "date", "boolean", "id"]
Granularity = Literal["monthly", "weekly", "daily"]


class SurrogateKeyWarning(UserWarning):
    """Warn when a composite-grain table uses a single-column surrogate PK."""


class RedundantCorrelationWarning(UserWarning):
    """Warn when a correlation entry has coefficient 0.0 (the default).

    F-01 / 0.4.0. Unlisted metric pairs already get zero off-diagonal, so an
    explicit ``coefficient: 0.0`` entry is either a mistake (the user meant
    a different value and typed zero) or unnecessary (it has no effect). We
    warn but don't reject — the entry is still structurally valid, and the
    built matrix is unchanged.
    """


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# --- Typed source parsing (Mission 001a) -------------------------------------
#
# Every column `source` string and every table `row_count_source` string can
# be parsed into one of the typed objects below. parse_source() is the single
# entry point; downstream missions (006 facts/events) should call it once per
# column and dispatch on the returned type rather than re-doing string
# matching inline.

class PKSource(_Frozen):
    """Column is the table's primary key column."""


class FKSource(_Frozen):
    table: str
    column: str


class MetricSource(_Frozen):
    metric: str


class GeneratedSource(_Frozen):
    provider: str


class FakerSource(_Frozen):
    """A Faker provider call, optionally parameterized.

    FIX-05 / MF-2. Grammar:

      * ``generated:faker.name`` → ``FakerSource(method="name", kwargs={})``
      * ``generated:faker.date_between:start_date:2020-01-01:end_date:2024-12-31``
        → ``FakerSource(method="date_between",
                        kwargs={"start_date": "2020-01-01",
                                "end_date": "2024-12-31"})``

    Non-faker providers (``timestamp``, ``date_key``, ``period_label``,
    ``entity_name``) continue to parse as :class:`GeneratedSource`. The
    split exists so parameter parsing and type coercion live on faker
    calls only, where they're meaningful.
    """
    method: str
    kwargs: dict[str, str] = Field(default_factory=dict)


class StaticSource(_Frozen):
    value: str


class DerivedSource(_Frozen):
    field: str


class ThresholdSource(_Frozen):
    """Event row exists when a metric stays above/below a threshold for N periods."""
    metric: str
    direction: Literal["above", "below"]
    value: float
    consecutive: int = Field(ge=1)


class ProportionalSource(_Frozen):
    """Number of event rows per entity per period = metric_value * scale.

    ``scale`` is capped at 100 per Category B / SEC-07: the scalability report
    (§6) showed that event-row construction dominates wall-clock and RSS at
    high scales. The cap preserves legitimate event-per-period multipliers
    while keeping the per-run row count bounded.
    """
    metric: str
    scale: float = Field(gt=0.0, le=100.0)


class LagSource(_Frozen):
    """Column value is driven by another metric's value N periods in the past."""
    metric: str
    periods: int = Field(ge=1)


ParsedSource = (
    PKSource | FKSource | MetricSource | GeneratedSource | FakerSource
    | StaticSource | DerivedSource
    | ThresholdSource | ProportionalSource | LagSource
)

_SOURCE_FORMAT_HELP = (
    "source must be one of: 'pk', 'fk:<table>.<column>', 'metric:<name>', "
    "'generated:<provider>', 'generated:faker.<method>[:<key>:<value>]*', "
    "'static:<value>', 'derived:<field>', "
    "'threshold:<metric>:<above|below>:<value>:for:<consecutive>', "
    "'proportional:<metric>:scale:<multiplier>', "
    "'lag:<metric>:periods:<N>'"
)


def parse_source(source: str) -> ParsedSource:
    """Parse a source string into a typed object. Raises ValueError on bad input.

    Callers: Column.source validator, Table.row_count_source validator,
    PlotsimConfig cross-reference integrity, and Mission 006 dispatch.
    """
    if not isinstance(source, str):
        raise ValueError(
            f"source must be a string, got {type(source).__name__}"
        )
    if source == "pk":
        return PKSource()

    if source.startswith("fk:"):
        ref = source[3:]
        if not ref:
            raise ValueError(f"source {source!r}: prefix 'fk:' requires a value")
        if "." not in ref:
            raise ValueError(
                f"fk source {source!r} must be 'fk:<table>.<column>' format"
            )
        table, column = ref.split(".", 1)
        if not table or not column:
            raise ValueError(
                f"fk source {source!r} must have non-empty table and column"
            )
        return FKSource(table=table, column=column)

    if source.startswith("generated:"):
        body = source[len("generated:"):]
        if not body:
            raise ValueError(
                f"source {source!r}: prefix 'generated:' requires a value"
            )
        if body.startswith("faker."):
            rest = body[len("faker."):]
            if not rest:
                raise ValueError(
                    f"source {source!r}: 'generated:faker.' requires a method"
                )
            parts = rest.split(":")
            method = parts[0]
            if not method:
                raise ValueError(
                    f"source {source!r}: empty faker method"
                )
            param_parts = parts[1:]
            if len(param_parts) % 2 != 0:
                raise ValueError(
                    f"source {source!r}: parameterized faker requires "
                    f"matched 'key:value' pairs after the method name"
                )
            kwargs: dict[str, str] = {}
            for i in range(0, len(param_parts), 2):
                key = param_parts[i]
                if not key:
                    raise ValueError(
                        f"source {source!r}: empty parameter name"
                    )
                if key in kwargs:
                    raise ValueError(
                        f"source {source!r}: duplicate parameter {key!r}"
                    )
                kwargs[key] = param_parts[i + 1]
            return FakerSource(method=method, kwargs=kwargs)
        return GeneratedSource(provider=body)

    for prefix, ctor_kw in (
        ("metric:", ("metric", MetricSource)),
        ("static:", ("value", StaticSource)),
        ("derived:", ("field", DerivedSource)),
    ):
        if source.startswith(prefix):
            value = source[len(prefix):]
            if not value:
                raise ValueError(
                    f"source {source!r}: prefix {prefix!r} requires a value"
                )
            kw, ctor = ctor_kw
            return ctor(**{kw: value})

    if source.startswith("threshold:"):
        parts = source.split(":")
        if len(parts) != 6 or parts[0] != "threshold" or parts[4] != "for":
            raise ValueError(
                f"threshold source {source!r} must be "
                f"'threshold:<metric>:<above|below>:<value>:for:<consecutive>'"
            )
        _, metric, direction, value_str, _, consecutive_str = parts
        if not metric:
            raise ValueError(
                f"threshold source {source!r} has empty metric name"
            )
        if direction not in ("above", "below"):
            raise ValueError(
                f"threshold source {source!r}: direction must be "
                f"'above' or 'below', got {direction!r}"
            )
        try:
            value = float(value_str)
        except ValueError as e:
            raise ValueError(
                f"threshold source {source!r}: non-numeric value {value_str!r}"
            ) from e
        try:
            consecutive = int(consecutive_str)
        except ValueError as e:
            raise ValueError(
                f"threshold source {source!r}: non-integer consecutive "
                f"{consecutive_str!r}"
            ) from e
        return ThresholdSource(
            metric=metric, direction=direction,
            value=value, consecutive=consecutive,
        )

    if source.startswith("proportional:"):
        parts = source.split(":")
        if len(parts) != 4 or parts[0] != "proportional" or parts[2] != "scale":
            raise ValueError(
                f"proportional source {source!r} must be "
                f"'proportional:<metric>:scale:<multiplier>'"
            )
        _, metric, _, scale_str = parts
        if not metric:
            raise ValueError(
                f"proportional source {source!r} has empty metric name"
            )
        try:
            scale = float(scale_str)
        except ValueError as e:
            raise ValueError(
                f"proportional source {source!r}: non-numeric scale {scale_str!r}"
            ) from e
        return ProportionalSource(metric=metric, scale=scale)

    if source.startswith("lag:"):
        parts = source.split(":")
        if len(parts) != 4 or parts[0] != "lag" or parts[2] != "periods":
            raise ValueError(
                f"lag source {source!r} must be 'lag:<metric>:periods:<N>'"
            )
        _, metric, _, periods_str = parts
        if not metric:
            raise ValueError(f"lag source {source!r} has empty metric name")
        try:
            periods = int(periods_str)
        except ValueError as e:
            raise ValueError(
                f"lag source {source!r}: non-integer periods {periods_str!r}"
            ) from e
        return LagSource(metric=metric, periods=periods)

    raise ValueError(f"invalid source {source!r}: {_SOURCE_FORMAT_HELP}")


# --- Schema models -----------------------------------------------------------

class Domain(_Frozen):
    name: str
    description: str
    entity_type: str
    entity_label: str


# Category B / SEC-06: per-granularity span ceilings. Matches the envelope
# measured in plotsim-scalability-report.md §6: monthly/weekly get 30 years,
# daily gets 10 years (daily is the granularity that dominates period counts
# and was observed to time out at long spans in the benchmark).
_SPAN_LIMITS: dict[str, int] = {
    "monthly": 360,
    "weekly": 1_560,
    "daily": 3_650,
}


class TimeWindow(_Frozen):
    start: str
    end: str
    granularity: Granularity

    @field_validator("start", "end")
    @classmethod
    def _valid_yyyy_mm(cls, v: str) -> str:
        parts = v.split("-")
        if len(parts) != 2 or len(parts[0]) != 4 or len(parts[1]) != 2:
            raise ValueError(f"expected YYYY-MM, got {v!r}")
        try:
            year = int(parts[0])
            month = int(parts[1])
        except ValueError as e:
            raise ValueError(f"expected YYYY-MM, got {v!r}") from e
        if year < 1900 or year > 2999:
            raise ValueError(f"year out of range in {v!r}")
        if not 1 <= month <= 12:
            raise ValueError(f"month out of range in {v!r}")
        return v

    @model_validator(mode="after")
    def _start_before_end(self) -> "TimeWindow":
        if self.start >= self.end:
            raise ValueError(
                f"time_window.start ({self.start}) must be before end ({self.end})"
            )
        return self

    def period_count(self) -> int:
        """Number of periods the engine will emit for this window+granularity.

        Matches :func:`plotsim.trajectory.compute_time_steps` exactly:
          - monthly: inclusive month span
          - daily: first-of-start-month through last-of-end-month, inclusive
          - weekly: count of distinct ISO-week labels across the daily span
        """
        sy, sm = int(self.start[:4]), int(self.start[5:7])
        ey, em = int(self.end[:4]), int(self.end[5:7])
        if self.granularity == "monthly":
            return (ey - sy) * 12 + (em - sm) + 1
        start_d = date(sy, sm, 1)
        end_d = date(ey, em, calendar.monthrange(ey, em)[1])
        if self.granularity == "daily":
            return (end_d - start_d).days + 1
        seen: set[tuple[int, int]] = set()
        d = start_d
        one_day = timedelta(days=1)
        while d <= end_d:
            iso_year, iso_week, _ = d.isocalendar()
            seen.add((iso_year, iso_week))
            d += one_day
        return len(seen)

    @model_validator(mode="after")
    def _span_within_limit(self) -> "TimeWindow":
        n = self.period_count()
        limit = _SPAN_LIMITS[self.granularity]
        if n > limit:
            raise ValueError(
                f"TimeWindow spans {n:,} {self.granularity} periods "
                f"({self.start}..{self.end}). Maximum for "
                f"{self.granularity} granularity is {limit:,}. Use monthly "
                f"or weekly granularity for longer ranges."
            )
        return self


class ValueRange(_Frozen):
    min: Optional[float] = None
    max: Optional[float] = None

    @model_validator(mode="after")
    def _min_le_max(self) -> "ValueRange":
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError(f"value_range.min ({self.min}) > max ({self.max})")
        return self


class CausalLag(_Frozen):
    """A metric whose value trails another metric by N periods.

    support_tickets.causal_lag = {driver: engagement, lag_periods: 2} means
    support_tickets at period T reads engagement at period T-2, not T.
    Cross-reference integrity enforces: driver exists in metrics; driver is
    not the metric itself; the induced lag graph has no cycles.

    ``blend_weight`` controls how strongly the driver's past position
    overrides the metric's own current trajectory position. Blend formula:
    ``current_position * (1 - w) + driver_past * w``. The 0.4.0 default of
    ``1.0`` means "full override" — metric at T equals driver at T-N, and
    cross-correlation peaks at exactly N. Values below 1.0 soften the lag
    (xcorr peak shifts toward ``round(w × N)``). The pre-0.4.0 hardcoded
    behavior is recovered with ``blend_weight: 0.6``.
    """
    driver: str
    lag_periods: int = Field(ge=1, le=120)
    blend_weight: float = Field(default=1.0, ge=0.0, le=1.0)


class Metric(_Frozen):
    name: str
    label: str
    distribution: Distribution
    params: dict[str, float]
    polarity: Polarity
    value_range: Optional[ValueRange] = None
    causal_lag: Optional[CausalLag] = None

    @model_validator(mode="after")
    def _causal_lag_not_self(self) -> "Metric":
        if self.causal_lag is not None and self.causal_lag.driver == self.name:
            raise ValueError(
                f"metric {self.name!r} has causal_lag.driver={self.name!r}; "
                f"a metric cannot lag itself"
            )
        return self


class CurveSegment(_Frozen):
    curve: CurveType
    # float | int | bool — sigmoid takes rising: bool, sawtooth/oscillating
    # take period as int-ish. Curve functions in plotsim.curves validate
    # their own param types; this schema just passes the dict through.
    params: dict[str, float | int | bool] = Field(default_factory=dict)
    start_pct: float = Field(ge=0.0, le=1.0)
    end_pct: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _start_before_end(self) -> "CurveSegment":
        if self.start_pct >= self.end_pct:
            raise ValueError(
                f"curve segment start_pct ({self.start_pct}) must be "
                f"< end_pct ({self.end_pct})"
            )
        return self


class MetricOverride(_Frozen):
    distribution: Optional[Distribution] = None
    params: Optional[dict[str, float]] = None


class Archetype(_Frozen):
    name: str
    label: str
    description: str
    curve_segments: list[CurveSegment] = Field(max_length=10)
    metric_overrides: dict[str, MetricOverride] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _segments_cover_full_range(self) -> "Archetype":
        if not self.curve_segments:
            raise ValueError(
                f"archetype {self.name!r} must have at least one curve segment"
            )
        sorted_segs = sorted(self.curve_segments, key=lambda s: s.start_pct)
        if sorted_segs[0].start_pct != 0.0:
            raise ValueError(
                f"archetype {self.name!r} curve_segments must start at 0.0, "
                f"got {sorted_segs[0].start_pct}"
            )
        if sorted_segs[-1].end_pct != 1.0:
            raise ValueError(
                f"archetype {self.name!r} curve_segments must end at 1.0, "
                f"got {sorted_segs[-1].end_pct}"
            )
        for prev, curr in zip(sorted_segs, sorted_segs[1:]):
            if curr.start_pct != prev.end_pct:
                raise ValueError(
                    f"archetype {self.name!r} has a gap or overlap between "
                    f"segments: prev ends at {prev.end_pct}, next starts at "
                    f"{curr.start_pct}"
                )
        return self


class EntityOverrides(_Frozen):
    """Per-entity trajectory adjustments.

    F9 / 0.5: replaces the previous ``Entity.overrides: dict[str, Any]``
    permissive-dict surface. Closes the schema discipline gap that
    silently accepted arbitrary unknown keys (the same pattern that
    motivated the 0.2.0 dead-field cleanup).

    Currently the only recognized override is ``inflection_month``,
    which shifts the archetype's curve segments so the archetype's
    canonical inflection lands on the specified period index.
    Adding new override keys is intentionally schema-gated — extend
    this model rather than threading new dict keys through
    ``compute_trajectory``.
    """
    inflection_month: Optional[int] = None


class Entity(_Frozen):
    name: str
    archetype: str
    size: int = Field(ge=1, le=5_000)
    overrides: Optional[EntityOverrides] = None
    # FIX-04: per-cohort cross-dim FK anchoring. Maps a child column name
    # to a parent PK value. Every entity in this cohort gets that exact
    # value for the named FK column, overriding any Column.distribution.
    # Use case: bind expansion-champion accounts to the enterprise plan,
    # connecting archetype narrative to reference data.
    cross_dim_fks: dict[str, str] = Field(default_factory=dict)


class FKDistribution(_Frozen):
    """Optional sampling spec for FK columns whose parent dim has > 1 row.

    Only meaningful on FK columns. ``weights=None`` means uniform sampling.
    ``weights={pk_value: weight, ...}`` means weighted sampling — keys must
    match parent PK values exactly, and weights must be non-negative with a
    positive sum.

    Schema accepts either a bare ``"uniform"`` string or a full mapping;
    ``Column._normalize_distribution`` coerces the string form to
    ``FKDistribution(weights=None)`` before validation.
    """
    weights: Optional[dict[str, float]] = None

    @model_validator(mode="after")
    def _weights_valid(self) -> "FKDistribution":
        if self.weights is not None:
            if not self.weights:
                raise ValueError("FKDistribution.weights cannot be empty dict")
            for k, v in self.weights.items():
                if v < 0.0:
                    raise ValueError(
                        f"FKDistribution weight for {k!r} is negative ({v})"
                    )
            if sum(self.weights.values()) <= 0.0:
                raise ValueError("FKDistribution weights sum must be > 0")
        return self


class Column(_Frozen):
    name: str
    dtype: Dtype
    source: str

    @field_validator("name")
    @classmethod
    def _name_is_identifier(cls, v: str) -> str:
        return _validate_identifier("column name", v)
    # Optional human-readable note that this column may contain PII (names,
    # emails, addresses generated via Faker). Pure metadata — no generation
    # behavior change. Surfaces in dump_config so downstream consumers can
    # filter or replace these columns before publishing. See README "Generated
    # data and PII".
    pii_note: Optional[str] = None
    # FIX-04: per-FK-column sampling spec. Only consumed when source is
    # ``fk:<table>.<col>``. Accepts a bare string ``"uniform"`` (uniform random
    # over parent PKs) or ``{weights: {pk: weight, ...}}`` (weighted). When
    # absent, the default is uniform if the parent has > 1 row, else the
    # single PK value (preserves the pre-FIX-04 single-row behavior).
    distribution: Optional[FKDistribution] = None
    # FIX-05: opt out of ``validate_temporal_coherence``. True means the
    # column's date values may legitimately fall outside ``time_window``
    # (hire dates, birth dates, trial-ended-before-start). Default False.
    allow_outside_window: bool = False

    @field_validator("source")
    @classmethod
    def _source_format(cls, v: str) -> str:
        # Delegate format validation to parse_source. Cross-reference
        # checks (metric/table names exist) happen in PlotsimConfig.
        parse_source(v)
        return v

    @field_validator("distribution", mode="before")
    @classmethod
    def _normalize_distribution(cls, v):
        # Accept the YAML shorthand ``distribution: "uniform"`` and coerce
        # it to ``FKDistribution(weights=None)``. Anything else falls
        # through to Pydantic's normal model parsing.
        if v is None:
            return None
        if isinstance(v, str):
            if v == "uniform":
                return {"weights": None}
            raise ValueError(
                f"Column.distribution string must be 'uniform', got {v!r}; "
                f"use a mapping {{weights: {{pk: weight, ...}}}} for weighted"
            )
        return v


class Table(_Frozen):
    name: str
    type: TableType
    grain: Grain
    columns: list[Column] = Field(max_length=100)
    primary_key: str | list[str]
    foreign_keys: list[str] = Field(default_factory=list)
    row_count_source: Optional[str] = None

    @property
    def primary_key_cols(self) -> list[str]:
        """Return the PK as a list, whether declared as str or list[str]."""
        return [self.primary_key] if isinstance(self.primary_key, str) else list(self.primary_key)

    @field_validator("name")
    @classmethod
    def _name_is_identifier(cls, v: str) -> str:
        return _validate_identifier("table name", v)

    @field_validator("row_count_source")
    @classmethod
    def _row_count_source_format(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        # Must be a parseable source. Semantics of "which source types are
        # meaningful as a row count" are enforced at generation time
        # (Mission 006); at load time we only enforce syntactic validity.
        parse_source(v)
        return v

    @model_validator(mode="after")
    def _pk_validation(self) -> "Table":
        col_names = {c.name for c in self.columns}
        pk_cols = self.primary_key_cols
        if not pk_cols:
            raise ValueError(f"table {self.name!r} primary_key is empty")
        missing = [k for k in pk_cols if k not in col_names]
        if missing:
            raise ValueError(
                f"table {self.name!r} primary_key columns {missing} "
                f"not in columns {sorted(col_names)}"
            )
        if len(set(pk_cols)) != len(pk_cols):
            raise ValueError(
                f"table {self.name!r} primary_key has duplicate columns: {pk_cols}"
            )
        # Composite-grain tables with a single-column PK: warn, don't block.
        # A (entity_id, date_key) composite natural key is cleaner, but some
        # users prefer surrogate row_id keys — honour that choice.
        if self.grain in COMPOSITE_GRAINS and isinstance(self.primary_key, str):
            warnings.warn(
                f"table {self.name!r} has grain {self.grain!r} but declares a "
                f"single-column primary key {self.primary_key!r}. A composite "
                f"natural key (e.g. [entity_id, date_key]) is cleaner; pass a "
                f"list to primary_key if that's what you want. This is a "
                f"surrogate-key warning, not a validation error.",
                SurrogateKeyWarning,
                stacklevel=2,
            )
        return self

    @model_validator(mode="after")
    def _row_count_source_only_on_event(self) -> "Table":
        if self.row_count_source is not None and self.type != "event":
            raise ValueError(
                f"table {self.name!r} has row_count_source but type is "
                f"{self.type!r}; row_count_source is only allowed on event tables"
            )
        return self


class CorrelationPair(_Frozen):
    metric_a: str
    metric_b: str
    coefficient: float = Field(ge=-1.0, le=1.0)


class NoiseConfig(_Frozen):
    gaussian_sigma: float = Field(default=0.0, ge=0.0, le=5.0)
    outlier_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    mcar_rate: float = Field(default=0.0, ge=0.0, le=1.0)


class OutputConfig(_Frozen):
    format: Literal["csv"] = "csv"
    directory: str


PERFECTLY_CLEAN = NoiseConfig(gaussian_sigma=0.0, outlier_rate=0.0, mcar_rate=0.0)
SLIGHTLY_MESSY = NoiseConfig(gaussian_sigma=0.03, outlier_rate=0.01, mcar_rate=0.005)
REALISTIC = NoiseConfig(gaussian_sigma=0.05, outlier_rate=0.02, mcar_rate=0.01)
DIRTY = NoiseConfig(gaussian_sigma=0.10, outlier_rate=0.05, mcar_rate=0.03)

NOISE_PRESETS: dict[str, NoiseConfig] = {
    "Perfectly clean": PERFECTLY_CLEAN,
    "Slightly messy": SLIGHTLY_MESSY,
    "Realistic": REALISTIC,
    "Dirty": DIRTY,
}


class StageDefinition(_Frozen):
    """One stage in a lifecycle funnel (onboarding → active → at_risk → churned)."""
    name: str
    threshold_enter: float = Field(ge=0.0, le=1.0)
    # Terminal stage has threshold_exit=None (entity never leaves once entered).
    threshold_exit: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class StageSequence(_Frozen):
    """Ordered lifecycle stages gated by a driving metric/field.

    Two semantics for ``threshold_exit`` are accepted; pick one
    consistently across all non-terminal stages:

    * **Legacy** (``threshold_exit > threshold_enter``) — non-overlap
      upper-bound semantic. Each stage spans
      ``[threshold_enter, threshold_exit)``; stages must not overlap
      (``prev.threshold_exit <= curr.threshold_enter``); contiguous
      stages (``prev.exit == curr.enter``) are allowed and typical.
      ``threshold_exit`` is decorative at runtime — both
      ``_monotonic_stage_walk`` and ``_free_mode_stages`` consume only
      ``threshold_enter``. The five bundled templates use this mode.

    * **Hysteresis** (``threshold_exit ≤ threshold_enter``) — F8 / 0.5
      addition. ``threshold_enter`` is the upward entry threshold;
      ``threshold_exit`` is the downward demotion threshold (lower than
      enter, providing a hysteresis band where the entity stays in the
      higher stage on transient dips). Constraint:
      ``prev.threshold_enter ≤ this.threshold_exit ≤ this.threshold_enter``.
      Activates the runtime demote path in ``_monotonic_stage_walk``
      (under ``enforce_order=True``); ``enforce_order=False`` is
      stateless and ignores the hysteresis distinction.

    Mixed sequences (some stages legacy, others hysteresis) are
    rejected at load.

    Last stage is always terminal (``threshold_exit=None``); non-
    terminal stages must have ``threshold_exit`` set under either
    semantic. ``field`` reference is checked against metrics in
    PlotsimConfig. ``enforce_order`` is stored for Mission 006 to
    consume at generation time.

    FIX-06 / SF-8: ``downgrade_delay`` relaxes strict monotonicity
    under ``enforce_order=True`` by letting the cursor step backwards
    once an entity has sat below the demote threshold for
    ``downgrade_delay`` consecutive periods. ``None`` (default)
    preserves strict-monotonic behavior under legacy mode and
    immediate-demote behavior under hysteresis mode. Ignored when
    ``enforce_order=False``.
    """
    field: str
    sequence: list[StageDefinition] = Field(min_length=2, max_length=10)
    enforce_order: bool = True
    downgrade_delay: Optional[int] = Field(default=None, ge=1, le=120)

    @property
    def mode(self) -> str:
        """Derived per-config: ``'legacy'`` or ``'hysteresis'``.

        F8 / 0.5: derived from the relationship between
        ``threshold_exit`` and ``threshold_enter`` on the first
        non-terminal stage. ``_sequence_is_valid`` enforces consistency,
        so the mode is unambiguous after load. Sequences with only a
        terminal stage cannot exist (``min_length=2``); a sequence whose
        non-terminal stages all have ``exit > enter`` is ``'legacy'``;
        otherwise ``'hysteresis'``.
        """
        seq_non_terminal = self.sequence[:-1]
        first = seq_non_terminal[0]
        # Validator guarantees first.threshold_exit is not None for non-
        # terminal stages.
        assert first.threshold_exit is not None
        if first.threshold_exit > first.threshold_enter:
            return "legacy"
        return "hysteresis"

    @model_validator(mode="after")
    def _sequence_is_valid(self) -> "StageSequence":
        seq = self.sequence
        if seq[-1].threshold_exit is not None:
            raise ValueError(
                f"stage sequence: last stage {seq[-1].name!r} must have "
                f"threshold_exit: null (terminal)"
            )
        for stage in seq[:-1]:
            if stage.threshold_exit is None:
                raise ValueError(
                    f"stage {stage.name!r} is not terminal but has "
                    f"threshold_exit: null"
                )

        # F8: detect per-stage mode and enforce consistency. Mixing
        # legacy (exit > enter) with hysteresis (exit ≤ enter) within
        # one sequence makes the runtime semantic ambiguous; reject.
        per_stage_modes: dict[str, str] = {}
        for stage in seq[:-1]:
            assert stage.threshold_exit is not None
            if stage.threshold_exit > stage.threshold_enter:
                per_stage_modes[stage.name] = "legacy"
            else:
                per_stage_modes[stage.name] = "hysteresis"
        if len(set(per_stage_modes.values())) > 1:
            raise ValueError(
                f"stage sequence mixes 'legacy' (threshold_exit > "
                f"threshold_enter; non-overlap upper-bound semantic) and "
                f"'hysteresis' (threshold_exit <= threshold_enter; "
                f"downward-band semantic). Pick one consistent mode. "
                f"Per-stage modes: {per_stage_modes}"
            )
        detected_mode = next(iter(set(per_stage_modes.values())))

        if detected_mode == "legacy":
            # Existing rule: prev.exit ≤ curr.enter (no overlap).
            for prev, curr in zip(seq, seq[1:]):
                assert prev.threshold_exit is not None
                if prev.threshold_exit > curr.threshold_enter:
                    raise ValueError(
                        f"stage {prev.name!r} threshold_exit "
                        f"({prev.threshold_exit}) > {curr.name!r} "
                        f"threshold_enter ({curr.threshold_enter}); "
                        f"stages must not overlap (legacy mode)"
                    )
        else:
            # Hysteresis mode rules:
            #   1. threshold_enter strictly ascending (no two stages share
            #      an entry point).
            #   2. For each non-terminal stage at index i ≥ 1:
            #      threshold_exit ≥ seq[i-1].threshold_enter — the
            #      hysteresis band of stage i lies above the previous
            #      stage's entry, so demoting from i lands the entity in
            #      stage (i-1) (or lower).
            for prev, curr in zip(seq, seq[1:]):
                if curr.threshold_enter <= prev.threshold_enter:
                    raise ValueError(
                        f"stage {curr.name!r} threshold_enter "
                        f"({curr.threshold_enter}) <= {prev.name!r} "
                        f"threshold_enter ({prev.threshold_enter}); stage "
                        f"threshold_enter must be strictly ascending "
                        f"(hysteresis mode)"
                    )
            for prev, curr in zip(seq[:-2], seq[1:-1]):
                # curr is non-terminal (skipped seq[-1]); has exit set.
                assert curr.threshold_exit is not None
                if curr.threshold_exit < prev.threshold_enter:
                    raise ValueError(
                        f"stage {curr.name!r} threshold_exit "
                        f"({curr.threshold_exit}) < {prev.name!r} "
                        f"threshold_enter ({prev.threshold_enter}); "
                        f"hysteresis exit must be >= previous stage's "
                        f"threshold_enter so demotion lands in a defined "
                        f"lower stage"
                    )
        return self


class PlotsimConfig(_Frozen):
    domain: Domain
    time_window: TimeWindow
    seed: int
    metrics: list[Metric] = Field(max_length=50)
    archetypes: list[Archetype] = Field(max_length=20)
    entities: list[Entity] = Field(min_length=1, max_length=100)
    tables: list[Table] = Field(max_length=50)
    # 1_225 = 50 choose 2 — the upper bound on unique pairwise correlations
    # given the 50-metric cap. Anything larger is either duplicates (rejected
    # at matrix assembly) or references to non-existent metrics.
    correlations: list[CorrelationPair] = Field(default_factory=list, max_length=1_225)
    noise: NoiseConfig = Field(default_factory=NoiseConfig)
    output: OutputConfig
    stages: Optional[StageSequence] = None
    # FIX-05 / SF-3: locale threaded to every Faker instance built by the
    # dim/fact/event layers. String (``"en_US"``, ``"ja_JP"``) or list
    # (multi-locale mix). Default ``"en_US"`` preserves prior behavior.
    locale: str | list[str] = "en_US"

    @model_validator(mode="after")
    def _total_entity_size_within_limit(self) -> "PlotsimConfig":
        total = sum(e.size for e in self.entities)
        if total > 100_000:
            raise ValueError(
                f"Total entity count across all groups is {total:,}. "
                f"Maximum is 100,000."
            )
        return self

    @model_validator(mode="after")
    def _combined_scale_estimator(self) -> "PlotsimConfig":
        """Category B Layer 2: detect multiplicative compounding at load.

        Per-field bounds (Layer 1) don't catch configs where every field is
        within its cap but the product ``sum(entities.size) × period_count``
        is ruinously large. The scalability report (§6) identified that product
        as the dominant predictor of wall-clock and memory.

        Behavior:
          * Always prints a summary line to stderr.
          * Appends a warning line when ``cell_count > 500_000``.
          * Raises ``ValueError`` when ``cell_count > 2_000_000``.

        Event-row estimate is informational only — metric values can exceed
        ``ValueRange.max`` once noise is applied, so it is not a reliable gate.
        The reject threshold uses the exact ``cell_count`` from the config.
        """
        n_entities = sum(e.size for e in self.entities)
        n_periods = self.time_window.period_count()
        cell_count = n_entities * n_periods

        n_fact_fields = sum(
            len(t.columns) for t in self.tables if t.type == "fact"
        )
        metrics_bytes = n_entities * n_periods * len(self.metrics) * 8
        fact_bytes = n_entities * n_periods * n_fact_fields * 8
        peak_mb = (metrics_bytes + fact_bytes) / 1_000_000 + 100

        event_rows_upper = 0
        for tbl in self.tables:
            if tbl.type != "event" or tbl.row_count_source is None:
                continue
            if not tbl.row_count_source.startswith("proportional"):
                continue
            parsed_rc = parse_source(tbl.row_count_source)
            if not isinstance(parsed_rc, ProportionalSource):
                continue
            driver = next(
                (m for m in self.metrics if m.name == parsed_rc.metric), None,
            )
            if driver is None or driver.value_range is None:
                continue
            v_max = driver.value_range.max
            if v_max is None:
                continue
            event_rows_upper += int(
                n_entities * n_periods * v_max * parsed_rc.scale
            )

        summary = (
            f"Config summary: {n_entities:,} entities × {n_periods:,} periods "
            f"= {cell_count:,} cells, {len(self.metrics)} metrics, "
            f"{len(self.tables)} tables. Estimated peak memory: ~{peak_mb:.0f} MB."
        )
        if event_rows_upper > 0:
            summary += f" Expected event rows (upper bound): ~{event_rows_upper:,}."
        sys.stderr.write(summary + "\n")

        if cell_count > 2_000_000:
            raise ValueError(
                f"Config produces {cell_count:,} cells (entities × periods), "
                f"which exceeds the maximum of 2,000,000. Reduce entity count, "
                f"time window span, or switch to a coarser granularity."
            )
        if cell_count > 500_000:
            sys.stderr.write(
                f"Warning: {cell_count:,} cells exceeds 500,000. Generation "
                f"may take several minutes and use significant memory.\n"
            )
        return self

    @model_validator(mode="after")
    def _cross_reference_integrity(self) -> "PlotsimConfig":
        metric_names = {m.name for m in self.metrics}
        archetype_names = {a.name for a in self.archetypes}
        table_names = {t.name for t in self.tables}

        if len(metric_names) != len(self.metrics):
            raise ValueError("duplicate metric names in metrics list")
        if len(archetype_names) != len(self.archetypes):
            raise ValueError("duplicate archetype names in archetypes list")
        if len(table_names) != len(self.tables):
            raise ValueError("duplicate table names in tables list")

        for arch in self.archetypes:
            for override_metric in arch.metric_overrides:
                if override_metric not in metric_names:
                    raise ValueError(
                        f"archetype {arch.name!r} overrides unknown metric "
                        f"{override_metric!r}; known metrics: {sorted(metric_names)}"
                    )

        for ent in self.entities:
            if ent.archetype not in archetype_names:
                raise ValueError(
                    f"entity {ent.name!r} references unknown archetype "
                    f"{ent.archetype!r}; known: {sorted(archetype_names)}"
                )

        for tbl in self.tables:
            for col in tbl.columns:
                parsed = parse_source(col.source)
                if isinstance(parsed, MetricSource):
                    if parsed.metric not in metric_names:
                        raise ValueError(
                            f"table {tbl.name!r} column {col.name!r} source "
                            f"{col.source!r} references unknown metric "
                            f"{parsed.metric!r}; known: {sorted(metric_names)}"
                        )
                elif isinstance(parsed, (ThresholdSource, ProportionalSource, LagSource)):
                    if parsed.metric not in metric_names:
                        raise ValueError(
                            f"table {tbl.name!r} column {col.name!r} source "
                            f"{col.source!r} references unknown metric "
                            f"{parsed.metric!r}; known: {sorted(metric_names)}"
                        )
                elif isinstance(parsed, FKSource):
                    if parsed.table not in table_names:
                        raise ValueError(
                            f"table {tbl.name!r} column {col.name!r} has FK to "
                            f"unknown table {parsed.table!r}; known: "
                            f"{sorted(table_names)}"
                        )

            if tbl.row_count_source is not None:
                rcs_parsed = parse_source(tbl.row_count_source)
                ref_metric = getattr(rcs_parsed, "metric", None)
                if ref_metric is not None and ref_metric not in metric_names:
                    raise ValueError(
                        f"table {tbl.name!r} row_count_source "
                        f"{tbl.row_count_source!r} references unknown metric "
                        f"{ref_metric!r}; known: {sorted(metric_names)}"
                    )

            for fk in tbl.foreign_keys:
                if "." not in fk:
                    raise ValueError(
                        f"table {tbl.name!r} foreign_keys entry {fk!r} must be "
                        f"'<table>.<column>' format"
                    )
                fk_table = fk.split(".", 1)[0]
                if fk_table not in table_names:
                    raise ValueError(
                        f"table {tbl.name!r} foreign_keys references unknown "
                        f"table {fk_table!r}; known: {sorted(table_names)}"
                    )

        # F7 (M102): reject duplicate (metric_a, metric_b) entries before
        # the PSD check picks one with last-write-wins. Treat the pair as
        # unordered: (a, b) == (b, a). Pre-fix, _build_correlation_matrix
        # silently overwrote earlier entries with later ones, so a config
        # with conflicting coefficients on the same pair picked one with
        # no signal to the user.
        seen_pairs: dict[frozenset, float] = {}
        for corr in self.correlations:
            pair = frozenset((corr.metric_a, corr.metric_b))
            if pair in seen_pairs:
                prior = seen_pairs[pair]
                raise ValueError(
                    f"duplicate correlation entries for unordered pair "
                    f"({corr.metric_a!r}, {corr.metric_b!r}): "
                    f"coefficients {prior} and {corr.coefficient}; "
                    f"declare each metric pair at most once"
                )
            seen_pairs[pair] = corr.coefficient

        for corr in self.correlations:
            for m in (corr.metric_a, corr.metric_b):
                if m not in metric_names:
                    raise ValueError(
                        f"correlation references unknown metric {m!r}; "
                        f"known: {sorted(metric_names)}"
                    )
            # F-01 / 0.4.0: flag explicit zero-coefficient entries.
            if corr.coefficient == 0.0:
                warnings.warn(
                    f"Correlation between {corr.metric_a!r} and "
                    f"{corr.metric_b!r} is configured as 0.0, which is "
                    f"already the default for unlisted pairs. This entry "
                    f"has no effect.",
                    RedundantCorrelationWarning,
                    stacklevel=2,
                )

        for m in self.metrics:
            if m.causal_lag is not None:
                if m.causal_lag.driver not in metric_names:
                    raise ValueError(
                        f"metric {m.name!r} causal_lag.driver "
                        f"{m.causal_lag.driver!r} is not a known metric; "
                        f"known: {sorted(metric_names)}"
                    )

        # Detect cycles in the induced lag graph (A lags B lags A, or longer).
        lag_graph = {
            m.name: m.causal_lag.driver
            for m in self.metrics if m.causal_lag is not None
        }
        for start in lag_graph:
            seen = {start}
            curr = lag_graph[start]
            while curr in lag_graph:
                if curr in seen:
                    raise ValueError(
                        f"circular causal_lag chain detected involving "
                        f"metric {start!r}"
                    )
                seen.add(curr)
                curr = lag_graph[curr]

        if self.stages is not None:
            if self.stages.field not in metric_names:
                raise ValueError(
                    f"stages.field {self.stages.field!r} is not a known metric; "
                    f"known: {sorted(metric_names)}"
                )

        return self

    @model_validator(mode="after")
    def _correlation_matrix_is_psd(self) -> "PlotsimConfig":
        """F-04: reject non-PSD correlation matrices at load time.

        Every other config defect surfaces as a pydantic ValidationError at
        load. Pre-FIX-F04, a non-PSD matrix passed load and only raised
        (as a plain ValueError) at the top of ``generate_tables``. The
        defense-in-depth call at ``generate_tables`` is retained to catch
        programmatic PlotsimConfig construction that bypasses YAML loading.
        """
        if not self.correlations:
            return self
        # Local import: plotsim.validation imports from plotsim.config,
        # so a module-level import would create a cycle.
        from plotsim.validation import validate_correlation_psd

        issues = validate_correlation_psd(self)
        if issues:
            names = [m.name for m in self.metrics]
            raise ValueError(
                f"correlations: correlation matrix is not positive "
                f"semi-definite for metrics {names}. {issues[0].message}"
            )
        return self


def load_config(path: str | Path) -> PlotsimConfig:
    """Load and validate a plotsim YAML config file."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(
            f"config file {p} did not parse to a mapping (got {type(data).__name__})"
        )
    return PlotsimConfig(**data)


def dump_config(config: PlotsimConfig) -> str:
    """Dump a PlotsimConfig back to a YAML string (round-trippable)."""
    return yaml.safe_dump(
        config.model_dump(mode="python"),
        sort_keys=False,
        default_flow_style=False,
    )
