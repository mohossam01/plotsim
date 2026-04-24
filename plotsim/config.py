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

import warnings
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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
#   per_subentity_per_period user-level facts: sub-entity × time step
#   variable                evt_login, evt_churn: trajectory-driven row count
GRAINS: frozenset[str] = frozenset({
    "per_entity",
    "per_period",
    "per_reference",
    "per_entity_per_period",
    "per_subentity_per_period",
    "variable",
})
COMPOSITE_GRAINS: frozenset[str] = frozenset({
    "per_entity_per_period", "per_subentity_per_period",
})
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
    "per_subentity_per_period",
    "variable",
]
Dtype = Literal["int", "float", "string", "date", "boolean", "id"]
Granularity = Literal["monthly", "weekly", "daily"]


class SurrogateKeyWarning(UserWarning):
    """Warn when a composite-grain table uses a single-column surrogate PK."""


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
    """Number of event rows per entity per period = metric_value * scale."""
    metric: str
    scale: float = Field(gt=0.0)


class LagSource(_Frozen):
    """Column value is driven by another metric's value N periods in the past."""
    metric: str
    periods: int = Field(ge=1)


ParsedSource = (
    PKSource | FKSource | MetricSource | GeneratedSource
    | StaticSource | DerivedSource
    | ThresholdSource | ProportionalSource | LagSource
)

_SOURCE_FORMAT_HELP = (
    "source must be one of: 'pk', 'fk:<table>.<column>', 'metric:<name>', "
    "'generated:<provider>', 'static:<value>', 'derived:<field>', "
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

    for prefix, ctor_kw in (
        ("metric:", ("metric", MetricSource)),
        ("generated:", ("provider", GeneratedSource)),
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
    """
    driver: str
    lag_periods: int = Field(ge=1)


class Metric(_Frozen):
    name: str
    label: str
    distribution: Distribution
    params: dict[str, float]
    polarity: Polarity
    default_curve: CurveType
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
    curve: Optional[CurveType] = None
    distribution: Optional[Distribution] = None
    params: Optional[dict[str, float]] = None


class Archetype(_Frozen):
    name: str
    label: str
    description: str
    curve_segments: list[CurveSegment]
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


class Entity(_Frozen):
    name: str
    archetype: str
    size: int = Field(ge=1)
    overrides: dict[str, Any] = Field(default_factory=dict)


class Column(_Frozen):
    name: str
    dtype: Dtype
    source: str

    @field_validator("source")
    @classmethod
    def _source_format(cls, v: str) -> str:
        # Delegate format validation to parse_source. Cross-reference
        # checks (metric/table names exist) happen in PlotsimConfig.
        parse_source(v)
        return v


class Table(_Frozen):
    name: str
    type: TableType
    grain: Grain
    columns: list[Column]
    primary_key: str | list[str]
    foreign_keys: list[str] = Field(default_factory=list)
    row_count_source: Optional[str] = None

    @property
    def primary_key_cols(self) -> list[str]:
        """Return the PK as a list, whether declared as str or list[str]."""
        return [self.primary_key] if isinstance(self.primary_key, str) else list(self.primary_key)

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
    gaussian_sigma: float = Field(default=0.0, ge=0.0)
    outlier_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    mcar_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    temporal_jitter_days: int = Field(default=0, ge=0)
    duplicate_rate: float = Field(default=0.0, ge=0.0, le=1.0)


class OutputConfig(_Frozen):
    format: Literal["csv"] = "csv"
    directory: str


PERFECTLY_CLEAN = NoiseConfig(
    gaussian_sigma=0.0, outlier_rate=0.0, mcar_rate=0.0,
    temporal_jitter_days=0, duplicate_rate=0.0,
)
SLIGHTLY_MESSY = NoiseConfig(
    gaussian_sigma=0.03, outlier_rate=0.01, mcar_rate=0.005,
    temporal_jitter_days=0, duplicate_rate=0.0,
)
REALISTIC = NoiseConfig(
    gaussian_sigma=0.05, outlier_rate=0.02, mcar_rate=0.01,
    temporal_jitter_days=2, duplicate_rate=0.0,
)
DIRTY = NoiseConfig(
    gaussian_sigma=0.10, outlier_rate=0.05, mcar_rate=0.03,
    temporal_jitter_days=5, duplicate_rate=0.01,
)

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

    Validated here: at least 2 stages, last stage is terminal (exit=None),
    non-terminal stages have exit set and enter < exit, and
    stage N's exit <= stage (N+1)'s enter (no overlap; contiguous allowed).
    `field` reference is checked against metrics in PlotsimConfig.
    `enforce_order` is stored for Mission 006 to consume at generation time.
    """
    field: str
    sequence: list[StageDefinition] = Field(min_length=2)
    enforce_order: bool = True

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
            if stage.threshold_enter >= stage.threshold_exit:
                raise ValueError(
                    f"stage {stage.name!r} threshold_enter ({stage.threshold_enter}) "
                    f">= threshold_exit ({stage.threshold_exit})"
                )
        for prev, curr in zip(seq, seq[1:]):
            # prev.threshold_exit is not None (prev is not last).
            assert prev.threshold_exit is not None
            if prev.threshold_exit > curr.threshold_enter:
                raise ValueError(
                    f"stage {prev.name!r} threshold_exit ({prev.threshold_exit}) "
                    f"> {curr.name!r} threshold_enter ({curr.threshold_enter}); "
                    f"stages must not overlap"
                )
        return self


class PlotsimConfig(_Frozen):
    domain: Domain
    time_window: TimeWindow
    seed: int
    metrics: list[Metric]
    archetypes: list[Archetype]
    entities: list[Entity]
    tables: list[Table]
    correlations: list[CorrelationPair] = Field(default_factory=list)
    noise: NoiseConfig = Field(default_factory=NoiseConfig)
    output: OutputConfig
    stages: Optional[StageSequence] = None

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

        for corr in self.correlations:
            for m in (corr.metric_a, corr.metric_b):
                if m not in metric_names:
                    raise ValueError(
                        f"correlation references unknown metric {m!r}; "
                        f"known: {sorted(metric_names)}"
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
