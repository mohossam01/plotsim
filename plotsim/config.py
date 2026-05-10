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
import os
import re
import sys
import warnings
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator

# Engine Literal aliases live in ``plotsim._types`` so ``recipes.py`` (which
# is documented as engine-import-free) can import them without pulling in
# pydantic. Re-exported here so existing ``from plotsim.config import Dtype``
# imports continue to work.
from plotsim._types import (
    CurveType,
    Distribution,
    Dtype,
    Grain,
    Granularity,
    Polarity,
    TableType,
    is_threshold_direction,
)

__all__ = [
    "CurveType",
    "Distribution",
    "Dtype",
    "Grain",
    "Granularity",
    "Polarity",
    "TableType",
]


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


def _identifier_field_validator(kind: str):
    """Return a ``field_validator(\"name\")`` callable that delegates to
    ``_validate_identifier(kind, value)``.

    Five Pydantic models (``PoolSource``, ``Column``, ``Table``, ``BridgeMetric``,
    ``BridgeTableConfig``) used to declare a near-verbatim 3-line classmethod
    ``_name_is_identifier`` — they only differed in the ``kind`` label
    surfaced in the error message. This factory collapses the duplication
    while preserving identical error text.
    """

    def _check(cls, v: str) -> str:
        return _validate_identifier(kind, v)

    return classmethod(_check)


CURVE_TYPES: frozenset[str] = frozenset(
    {
        "sigmoid",
        "exp_decay",
        "step",
        "logistic",
        "plateau",
        "oscillating",
        "compound",
        "sawtooth",
    }
)
DISTRIBUTIONS: frozenset[str] = frozenset(
    {
        "lognorm",
        "gamma",
        "poisson",
        "beta",
        "normal",
        "weibull",
    }
)
POLARITIES: frozenset[str] = frozenset({"positive", "negative"})
TABLE_TYPES: frozenset[str] = frozenset({"dim", "fact", "event"})
# Grain values — each tells the table builder exactly which loop to run.
#   per_entity              dim_company: one row per entity
#   per_period              dim_date: one row per time step, no entity axis
#   per_reference           dim_plan, dim_department: static lookup (no time, no entity)
#   per_entity_per_period   fct_engagement: entity × time step
#   variable                evt_login, evt_churn: trajectory-driven row count
GRAINS: frozenset[str] = frozenset(
    {
        "per_entity",
        "per_period",
        "per_reference",
        "per_entity_per_period",
        "variable",
    }
)
COMPOSITE_GRAINS: frozenset[str] = frozenset({"per_entity_per_period"})
DTYPES: frozenset[str] = frozenset(
    {
        "int",
        "float",
        "string",
        "date",
        "boolean",
        "id",
    }
)
GRANULARITIES: frozenset[str] = frozenset({"monthly", "weekly", "daily"})


class SurrogateKeyWarning(UserWarning):
    """Warn when a composite-grain table uses a single-column surrogate PK."""


class RedundantCorrelationWarning(UserWarning):
    """Warn when a correlation entry has coefficient 0.0 (the default).

    Unlisted metric pairs already get zero off-diagonal, so an explicit
    ``coefficient: 0.0`` entry is either a mistake (the user meant a
    different value and typed zero) or unnecessary (it has no effect).
    We warn but don't reject — the entry is still structurally valid,
    and the built matrix is unchanged.
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

    Grammar:

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

    ``scale`` is capped at 100: event-row construction dominates wall-clock
    and RSS at high scales, and the cap preserves legitimate
    event-per-period multipliers while keeping the per-run row count
    bounded.
    """

    metric: str
    scale: float = Field(gt=0.0, le=100.0)


class LagSource(_Frozen):
    """Column value is driven by another metric's value N periods in the past."""

    metric: str
    periods: int = Field(ge=1)


class SCDType2Source(_Frozen):
    """Marker source for a dim column carrying SCD Type 2 labels.

    The literal source string ``scd_type2`` parses to this marker; the
    actual versioning configuration (trigger metric, thresholds, labels)
    lives on ``Column.scd_type2`` so the SCD machinery in
    ``plotsim.tables`` has structured access without re-parsing strings.

    A dim column with this source MUST also declare
    ``scd_type2: {trigger_metric, thresholds, labels}`` (validated on
    ``Column``); a column with a non-``scd_type2`` source MUST NOT
    declare ``scd_type2``. The two are paired or both absent — there is
    no "set the source but not the config" path.
    """


class TextBucketSource(_Frozen):
    """Text emission keyed by trajectory-position bands.

    Grammar: ``text:bucket:[<label1>, <label2>, ..., <labelN>]``. Labels are
    comma-separated, whitespace around each label is stripped, and the
    bracket pair is required. With N labels, the [0, 1] trajectory range
    is split into N evenly-spaced bands: position ``p`` maps to bucket
    ``min(int(p * N), N - 1)``. The lowest position lands in label[0],
    the highest in label[N-1] — monotonic by construction.

    Trajectory-first invariant: the bucket is selected from the same
    archetype-driven trajectory position every metric on the same row
    is derived from. A negative-polarity sentiment ("delighted →
    churned") is expressed by ordering the labels with the *most
    favorable* outcome at the *highest* position, mirroring how
    positive-polarity metrics shape values from position. Configs that
    want the inverse just reverse the bucket list.
    """

    buckets: tuple[str, ...] = Field(min_length=2, max_length=20)


class PoolSource(_Frozen):
    """Per-entity value pool on a dimension column.

    Grammar: ``pool:<name>``. ``<name>`` is a free-form identifier that
    distinguishes multiple pool columns on the same table (e.g. ``industry``
    vs ``segment``); the actual values are stored on the column under
    ``Column.value_pool: dict[entity_name, list[str]]``.

    A column with this source MUST also declare ``value_pool`` (validated
    on ``Column``); a column with a non-``pool:`` source MUST NOT declare
    ``value_pool``. The two are paired or both absent — same discipline
    as the SCD Type 2 pairing.

    Architectural firewall: pools resolve entity-membership only, never
    trajectory-derived. The dim layer never sees a trajectory; pool
    selection consumes one RNG draw per row, deterministic under the
    engine's single-seed contract.
    """

    name: str

    _name_is_identifier = field_validator("name")(_identifier_field_validator("pool name"))


ParsedSource = (
    PKSource
    | FKSource
    | MetricSource
    | GeneratedSource
    | FakerSource
    | StaticSource
    | DerivedSource
    | ThresholdSource
    | ProportionalSource
    | LagSource
    | TextBucketSource
    | SCDType2Source
    | PoolSource
)

_SOURCE_FORMAT_HELP = (
    "source must be one of: 'pk', 'fk:<table>.<column>', 'metric:<name>', "
    "'generated:<provider>', 'generated:faker.<method>[:<key>:<value>]*', "
    "'static:<value>', 'derived:<field>', "
    "'threshold:<metric>:<above|below>:<value>:for:<consecutive>', "
    "'proportional:<metric>:scale:<multiplier>', "
    "'lag:<metric>:periods:<N>', "
    "'text:bucket:[<label1>, <label2>, ...]', "
    "'scd_type2', "
    "'pool:<name>'"
)


def parse_source(source: str) -> ParsedSource:
    """Parse a source string into a typed object. Raises ValueError on bad input.

    Callers: Column.source validator, Table.row_count_source validator,
    PlotsimConfig cross-reference integrity, and Mission 006 dispatch.
    """
    if not isinstance(source, str):
        raise ValueError(f"source must be a string, got {type(source).__name__}")
    if source == "pk":
        return PKSource()
    if source == "scd_type2":
        return SCDType2Source()

    if source.startswith("fk:"):
        ref = source[3:]
        if not ref:
            raise ValueError(f"source {source!r}: prefix 'fk:' requires a value")
        if "." not in ref:
            raise ValueError(f"fk source {source!r} must be 'fk:<table>.<column>' format")
        table, column = ref.split(".", 1)
        if not table or not column:
            raise ValueError(f"fk source {source!r} must have non-empty table and column")
        return FKSource(table=table, column=column)

    if source.startswith("generated:"):
        body = source[len("generated:") :]
        if not body:
            raise ValueError(f"source {source!r}: prefix 'generated:' requires a value")
        if body.startswith("faker."):
            rest = body[len("faker.") :]
            if not rest:
                raise ValueError(f"source {source!r}: 'generated:faker.' requires a method")
            parts = rest.split(":")
            method = parts[0]
            if not method:
                raise ValueError(f"source {source!r}: empty faker method")
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
                    raise ValueError(f"source {source!r}: empty parameter name")
                if key in kwargs:
                    raise ValueError(f"source {source!r}: duplicate parameter {key!r}")
                kwargs[key] = param_parts[i + 1]
            return FakerSource(method=method, kwargs=kwargs)
        return GeneratedSource(provider=body)

    for prefix, ctor_kw in (
        ("metric:", ("metric", MetricSource)),
        ("static:", ("value", StaticSource)),
        ("derived:", ("field", DerivedSource)),
    ):
        if source.startswith(prefix):
            payload = source[len(prefix) :]
            if not payload:
                raise ValueError(f"source {source!r}: prefix {prefix!r} requires a value")
            kw, ctor = ctor_kw
            return ctor(**{kw: payload})

    if source.startswith("threshold:"):
        parts = source.split(":")
        if len(parts) != 6 or parts[0] != "threshold" or parts[4] != "for":
            raise ValueError(
                f"threshold source {source!r} must be "
                f"'threshold:<metric>:<above|below>:<value>:for:<consecutive>'"
            )
        _, metric, direction, value_str, _, consecutive_str = parts
        if not metric:
            raise ValueError(f"threshold source {source!r} has empty metric name")
        if not is_threshold_direction(direction):
            raise ValueError(
                f"threshold source {source!r}: direction must be "
                f"'above' or 'below', got {direction!r}"
            )
        try:
            value = float(value_str)
        except ValueError as e:
            raise ValueError(f"threshold source {source!r}: non-numeric value {value_str!r}") from e
        try:
            consecutive = int(consecutive_str)
        except ValueError as e:
            raise ValueError(
                f"threshold source {source!r}: non-integer consecutive " f"{consecutive_str!r}"
            ) from e
        return ThresholdSource(
            metric=metric,
            direction=direction,
            value=value,
            consecutive=consecutive,
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
            raise ValueError(f"proportional source {source!r} has empty metric name")
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
            raise ValueError(f"lag source {source!r} must be 'lag:<metric>:periods:<N>'")
        _, metric, _, periods_str = parts
        if not metric:
            raise ValueError(f"lag source {source!r} has empty metric name")
        try:
            periods = int(periods_str)
        except ValueError as e:
            raise ValueError(f"lag source {source!r}: non-integer periods {periods_str!r}") from e
        return LagSource(metric=metric, periods=periods)

    if source.startswith("pool:"):
        body = source[len("pool:") :]
        if not body:
            raise ValueError(
                f"pool source {source!r}: prefix 'pool:' requires a name " f"(e.g. 'pool:industry')"
            )
        # Reject embedded colons: ``pool:industry:extra`` would be ambiguous
        # under any future grammar extension. Surface it now.
        if ":" in body:
            raise ValueError(
                f"pool source {source!r} must be 'pool:<name>' with no extra "
                f"colons; the actual values live on Column.value_pool"
            )
        return PoolSource(name=body)

    if source.startswith("text:bucket:"):
        body = source[len("text:bucket:") :]
        if not body.startswith("[") or not body.endswith("]"):
            raise ValueError(
                f"text-bucket source {source!r} must wrap labels in '[ ... ]': "
                f"e.g. 'text:bucket:[low, mid, high]'"
            )
        inner = body[1:-1].strip()
        if not inner:
            raise ValueError(f"text-bucket source {source!r} has empty bucket list")
        labels = [p.strip() for p in inner.split(",")]
        if any(not label for label in labels):
            raise ValueError(
                f"text-bucket source {source!r} has an empty label "
                f"(check for stray commas or whitespace-only entries)"
            )
        if len(labels) < 2:
            raise ValueError(
                f"text-bucket source {source!r} requires at least 2 labels "
                f"(banding requires distinguishable bands)"
            )
        if len(set(labels)) != len(labels):
            raise ValueError(
                f"text-bucket source {source!r} has duplicate labels; each "
                f"bucket must be uniquely named so a downstream consumer can "
                f"reverse-map a value to its position band"
            )
        return TextBucketSource(buckets=tuple(labels))

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


# Cell-budget gate: tiered thresholds for the multiplicative-compounding
# guard in ``PlotsimConfig._combined_scale_estimator``.
#
#   * Below ``_CELL_ADVISORY_THRESHOLD``: silent (just the summary line).
#   * Between advisory and the soft budget: stderr advisory recommending
#     parquet + auto generation.
#   * Above the soft budget without opt-in: ``ValueError`` pointing at
#     ``PLOTSIM_CELL_BUDGET`` and ``--allow-large-dataset``.
#   * Above the soft budget with opt-in: stderr large-dataset notice,
#     proceed.
#   * Above ``_CELL_HARD_CEILING`` regardless of opt-in: ``ValueError``.
#     The hard ceiling is non-configurable; configs this large should be
#     split or chunked rather than coerced through the engine.
_CELL_ADVISORY_THRESHOLD = 500_000
_CELL_SOFT_BUDGET_DEFAULT = 2_000_000
_CELL_HARD_CEILING = 50_000_000


def _resolve_cell_budget(config_override: Optional[int] = None) -> int:
    """Return the effective soft budget. ``0`` disables the soft gate.

    Precedence (M7):

      1. Explicit ``OutputConfig.cell_budget`` field if not ``None``.
      2. ``PLOTSIM_CELL_BUDGET`` environment variable.
      3. ``_CELL_SOFT_BUDGET_DEFAULT`` (2,000,000).

    Non-integer env values fall back to the default — invalid env
    config shouldn't silently raise the cap. The config-field path
    is pre-validated by Pydantic (``ge=0``) so no parse step is
    needed for it.
    """
    if config_override is not None:
        return max(config_override, 0)
    raw = os.environ.get("PLOTSIM_CELL_BUDGET")
    if raw is None:
        return _CELL_SOFT_BUDGET_DEFAULT
    try:
        n = int(raw)
    except ValueError:
        return _CELL_SOFT_BUDGET_DEFAULT
    return max(n, 0)


def _allow_large_dataset() -> bool:
    """True when the operator has opted into above-soft-budget runs.

    Set by the CLI ``--allow-large-dataset`` flag, or directly by
    library callers via ``PLOTSIM_ALLOW_LARGE_DATASET=1``.
    """
    raw = os.environ.get("PLOTSIM_ALLOW_LARGE_DATASET", "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# F10 (M102): granularity-aware ``causal_lag.lag_periods`` ceilings.
# Period count corresponds to ~10 years of lag at each granularity —
# tight enough to keep the lag-buffer per-period work bounded and to
# reject obviously-misconfigured values (e.g. 5,000 monthly = 416
# years), wide enough to let daily configs configure quarterly /
# semi-annual / multi-year lags that the previous flat ``le=120`` cap
# blocked. Pre-F10 the field-level cap was a uniform 120, which read
# as "≈ 4 months at daily, 10 years at monthly" — granularity-blind in
# either direction. The field-level cap is now relaxed to the daily
# maximum; ``PlotsimConfig._lag_periods_within_granularity_limit``
# enforces the per-granularity bound at the model level so the error
# names which granularity rejected the value.
_LAG_PERIOD_LIMITS: dict[str, int] = {
    "monthly": 120,
    "weekly": 520,
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
            raise ValueError(f"time_window.start ({self.start}) must be before end ({self.end})")
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

    def period_calendar_months(self) -> list[int]:
        """Return the calendar month (1-12) for each period in this window.

        Anchors ``SeasonalEffect.months`` to period indices.

        - monthly: month from each ``YYYY-MM`` step.
        - weekly:  month of the FIRST day in the window for that ISO week —
                   matches the spec rule "a week belongs to the month of its
                   start date" while staying inside the configured window so
                   weeks at the boundary aren't miscredited to a date outside
                   ``[start, end]``.
        - daily:   month from each ``YYYY-MM-DD`` step.
        """
        sy, sm = int(self.start[:4]), int(self.start[5:7])
        ey, em = int(self.end[:4]), int(self.end[5:7])
        if self.granularity == "monthly":
            months: list[int] = []
            total = (ey - sy) * 12 + (em - sm) + 1
            for k in range(total):
                offset = sm - 1 + k
                months.append(offset % 12 + 1)
            return months
        start_d = date(sy, sm, 1)
        end_d = date(ey, em, calendar.monthrange(ey, em)[1])
        if self.granularity == "daily":
            out: list[int] = []
            d = start_d
            one_day = timedelta(days=1)
            while d <= end_d:
                out.append(d.month)
                d += one_day
            return out
        # weekly — track each ISO-week's first in-window date
        seen: dict[tuple[int, int], int] = {}
        order: list[tuple[int, int]] = []
        d = start_d
        one_day = timedelta(days=1)
        while d <= end_d:
            iso_year, iso_week, _ = d.isocalendar()
            key = (iso_year, iso_week)
            if key not in seen:
                seen[key] = d.month
                order.append(key)
            d += one_day
        return [seen[k] for k in order]


class ValueRange(_Frozen):
    min: Optional[float] = None
    max: Optional[float] = None

    @model_validator(mode="after")
    def _min_le_max(self) -> "ValueRange":
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError(f"value_range.min ({self.min}) > max ({self.max})")
        return self


class SeasonalEffect(_Frozen):
    """A calendar-driven multiplicative center modifier.

    At each period whose calendar month is in ``months``, every metric's
    distribution center is scaled by ``(1 + effective)`` BEFORE distribution
    sampling, where::

        effective = strength × metric.seasonal_sensitivity × entity.seasonal_sensitivity

    Multiple ``SeasonalEffect`` entries are summed at the global level (per
    period) before per-metric and per-entity sensitivities multiply.
    Sensitivities default to ``1.0`` (full follow), can go negative (invert),
    or zero (immunity).

    Modulation is a center modifier, not a trajectory modifier — the
    trajectory-first invariant is preserved. The trajectory says where the
    entity is; seasonality says what the world does to that entity at that
    calendar moment.
    """

    months: tuple[int, ...] = Field(min_length=1, max_length=12)
    strength: float

    @field_validator("months")
    @classmethod
    def _months_in_range_unique(cls, v: tuple[int, ...]) -> tuple[int, ...]:
        for m in v:
            if not 1 <= int(m) <= 12:
                raise ValueError(f"seasonal effect month {m} out of range; valid: 1..12")
        if len(set(v)) != len(v):
            raise ValueError(f"seasonal effect months must be unique, got {list(v)}")
        return v


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
    # F10 (M102): field-level cap relaxed to a sanity bound above the
    # daily-granularity maximum. The authoritative per-granularity
    # bound is enforced at the model level by
    # PlotsimConfig._cross_reference_integrity, which produces a
    # clearer error message naming the granularity that rejected the
    # value. The field-level cap of 10_000 catches obvious garbage
    # (e.g. typos that produce 1_000_000) when CausalLag is
    # constructed outside a PlotsimConfig.
    lag_periods: int = Field(ge=1, le=10_000)
    blend_weight: float = Field(default=1.0, ge=0.0, le=1.0)


class Metric(_Frozen):
    name: str
    label: str
    distribution: Distribution
    params: dict[str, float]
    polarity: Polarity
    value_range: Optional[ValueRange] = None
    causal_lag: Optional[CausalLag] = None
    # M119: per-metric seasonal sensitivity. Multiplies the global
    # ``SeasonalEffect.strength`` (and the entity's sensitivity) at each
    # period in the effect's months. ``1.0`` follows global exactly,
    # ``0.0`` is immune, negatives invert.
    seasonal_sensitivity: float = 1.0

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
                f"curve segment start_pct ({self.start_pct}) must be " f"< end_pct ({self.end_pct})"
            )
        return self


class MetricOverride(_Frozen):
    distribution: Optional[Distribution] = None
    params: Optional[dict[str, float]] = None
    # M114: per-archetype value range override. When present, this range
    # replaces the global ``Metric.value_range`` for entities assigned to
    # the owning archetype — the threading happens in
    # ``plotsim.metrics._apply_archetype_overrides``. The override range
    # MUST be a subset of the global range; overrides restrict, never
    # expand. Subset enforcement is cross-model (needs the metric's range)
    # and lives in ``PlotsimConfig._cross_reference_integrity``.
    value_range: Optional[ValueRange] = None


class Archetype(_Frozen):
    name: str
    label: str
    description: str
    curve_segments: list[CurveSegment] = Field(max_length=10)
    metric_overrides: dict[str, MetricOverride] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _segments_cover_full_range(self) -> "Archetype":
        if not self.curve_segments:
            raise ValueError(f"archetype {self.name!r} must have at least one curve segment")
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
    # ``size`` is the sub-entity dim row multiplier on engine-direct configs:
    # one ``Entity(size=50)`` produces 50 rows in any per_entity dim's child
    # sub-entity (variable-grain) dim. The builder path (M117) instead emits
    # ``size=1`` for every expanded entity and pushes the row multiplier onto
    # ``Table.count``; the two compose multiplicatively in
    # ``dimensions.build_dim_subentity``, so engine-direct configs keep their
    # pre-M117 semantics unchanged.
    size: int = Field(ge=1, le=5_000)
    overrides: Optional[EntityOverrides] = None
    # FIX-04: per-cohort cross-dim FK anchoring. Maps a child column name
    # to a parent PK value. Every entity in this cohort gets that exact
    # value for the named FK column, overriding any Column.distribution.
    # Use case: bind expansion-champion accounts to the enterprise plan,
    # connecting archetype narrative to reference data.
    cross_dim_fks: dict[str, str] = Field(default_factory=dict)
    # M119: per-entity seasonal sensitivity. Multiplies the global
    # ``SeasonalEffect.strength`` (and the metric's sensitivity) at each
    # period in the effect's months. ``1.0`` follows global exactly,
    # ``0.0`` is immune, negatives invert. Builder users typically set
    # this on a ``SegmentInput``; the interpreter copies the value onto
    # every expanded entity in that segment.
    seasonal_sensitivity: float = 1.0
    # 0.6-M8a: per-entity arrival period (cold-start). ``0`` (default) =
    # entity is present for the full window — preserves pre-M8a behaviour
    # byte-for-byte. ``k > 0`` = entity is dormant for periods ``[0, k)``
    # (trajectory NaN-filled, fact rows dropped) and active from period
    # ``k`` onward, with the archetype's full curve playing out across
    # the entity's own active window. ``dim_<entity>`` always includes
    # the entity regardless of arrival period.
    start_period: int = Field(default=0, ge=0)


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
                    raise ValueError(f"FKDistribution weight for {k!r} is negative ({v})")
            if sum(self.weights.values()) <= 0.0:
                raise ValueError("FKDistribution weights sum must be > 0")
        return self


class SCDType2Config(_Frozen):
    """Slowly Changing Dimension Type 2 spec on a dim column.

    Attached to a ``Column`` whose source is the literal string
    ``"scd_type2"``; the source marker and this config object are
    paired (both present, or both absent — never one without the
    other; ``Column._scd_pairing`` enforces).

    Fields:

      * ``trigger_metric`` — ``"<table_name>.<metric_name>"``. The
        ``table_name`` must be a fact table declared in
        ``config.tables``; the ``metric_name`` must be a metric in
        ``config.metrics`` AND must appear as a ``metric:<name>``
        source on a column of that fact table. The fact-table reference
        is documentary: thresholds are evaluated against the
        per-entity trajectory positions (the same trajectory that
        feeds every other metric the engine generates), not against
        the metric's noisy/distributed cell values. Naming the fact
        column anchors the SCD label to a concrete downstream join
        target so a reader of the config knows "plan tier changes
        when MRR moves" rather than "plan tier changes when an
        opaque trajectory shifts."

      * ``thresholds`` — strictly ascending floats in the open
        interval ``(0, 1)``. Bands are ``[0, t0), [t0, t1), ..., [tN, 1]``
        — N thresholds yield N+1 bands.

      * ``labels`` — one per band; ``len(labels) == len(thresholds) + 1``.
        Order follows the bands: ``labels[0]`` is the lowest-position
        band, ``labels[-1]`` is the highest. The SCD column's value
        for an entity in band k is ``labels[k]``.

    Hysteresis: an entity that crosses upward into a new band and
    later returns to a lower band does NOT spawn a new dim row on the
    return — band assignment uses ``np.maximum.accumulate`` over the
    raw per-period band index, so a dim row is emitted only when the
    cursor advances. Mirrors ``StageDefinition.threshold_exit`` monotonic
    semantics so SCD versioning has a single behavioural contract with
    the rest of the engine.
    """

    trigger_metric: str
    thresholds: tuple[float, ...] = Field(min_length=1, max_length=20)
    labels: tuple[str, ...] = Field(min_length=2, max_length=21)

    @field_validator("trigger_metric")
    @classmethod
    def _trigger_metric_format(cls, v: str) -> str:
        if not isinstance(v, str) or "." not in v:
            raise ValueError(f"scd_type2.trigger_metric {v!r} must be " f"'table_name.metric_name'")
        table, metric = v.split(".", 1)
        if not table or not metric:
            raise ValueError(
                f"scd_type2.trigger_metric {v!r} must have non-empty "
                f"table_name and metric_name on either side of '.'"
            )
        return v

    @field_validator("thresholds")
    @classmethod
    def _thresholds_in_open_interval(cls, v: tuple[float, ...]) -> tuple[float, ...]:
        for t in v:
            if not (0.0 < float(t) < 1.0):
                raise ValueError(
                    f"scd_type2.thresholds values must lie in the open " f"interval (0, 1); got {t}"
                )
        for prev, curr in zip(v, v[1:]):
            if curr <= prev:
                raise ValueError(
                    f"scd_type2.thresholds must be strictly increasing; " f"got {list(v)}"
                )
        return v

    @model_validator(mode="after")
    def _label_count_matches(self) -> "SCDType2Config":
        expected = len(self.thresholds) + 1
        if len(self.labels) != expected:
            raise ValueError(
                f"scd_type2.labels has {len(self.labels)} entries; expected "
                f"{expected} (one per band, i.e. len(thresholds) + 1). "
                f"thresholds={list(self.thresholds)}, labels={list(self.labels)}"
            )
        if len(set(self.labels)) != len(self.labels):
            raise ValueError(
                f"scd_type2.labels has duplicate entries {list(self.labels)}; "
                f"each band label must be unique so the dim column round-trips "
                f"to a band index without ambiguity"
            )
        return self


class Column(_Frozen):
    name: str
    dtype: Dtype
    source: str

    _name_is_identifier = field_validator("name")(_identifier_field_validator("column name"))

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
    # M106: when present, this column carries SCD Type 2 labels. Must be
    # paired with ``source: "scd_type2"`` (validated by ``_scd_pairing``).
    # The trigger metric, thresholds, and labels live on the nested model;
    # the engine in ``plotsim.tables`` consumes ``scd_type2.thresholds``
    # and ``scd_type2.labels`` to expand the dim into versioned rows.
    scd_type2: Optional[SCDType2Config] = None
    # M114: per-entity value pool. Keys are ``Entity.name`` values that
    # produce rows in this dim table; each entity's row(s) get a value
    # sampled from its list. Must be paired with ``source: "pool:<name>"``
    # (validated by ``_pool_pairing``); entity coverage is enforced
    # cross-model in ``plotsim.validation.validate_value_pool_coverage``.
    value_pool: Optional[dict[str, list[str]]] = None

    @field_validator("source")
    @classmethod
    def _source_format(cls, v: str) -> str:
        # Delegate format validation to parse_source. Cross-reference
        # checks (metric/table names exist) happen in PlotsimConfig.
        parse_source(v)
        return v

    @model_validator(mode="after")
    def _scd_pairing(self) -> "Column":
        """Source ``scd_type2`` and the ``scd_type2`` config must be paired.

        Either both are present (the column emits SCD Type 2 labels) or
        both are absent (the column resolves through one of the other
        source types). Mixing — source ``"scd_type2"`` without the
        config, or any other source with the config — is rejected at
        load. Mirrors the ``Column.distribution`` discipline (only
        meaningful on FK sources): the schema is structurally exact
        rather than permissive.
        """
        is_scd_source = self.source == "scd_type2"
        has_scd_cfg = self.scd_type2 is not None
        if is_scd_source and not has_scd_cfg:
            raise ValueError(
                f"column {self.name!r} declares source 'scd_type2' but no "
                f"'scd_type2' config block; add 'scd_type2: {{trigger_metric, "
                f"thresholds, labels}}' or change the source"
            )
        if has_scd_cfg and not is_scd_source:
            raise ValueError(
                f"column {self.name!r} has an scd_type2 config block but "
                f"source {self.source!r}; SCD labels replace the column "
                f"value, so set source to 'scd_type2' or remove the "
                f"scd_type2 config"
            )
        return self

    @model_validator(mode="after")
    def _pool_pairing(self) -> "Column":
        """Source ``pool:<name>`` and ``value_pool`` must be paired.

        Either both are present (the column samples per-entity from the
        declared pool) or both are absent. Mirrors ``_scd_pairing``: a
        column with ``pool:`` source but no ``value_pool`` would emit
        nothing meaningful, and a column with ``value_pool`` but a
        non-``pool:`` source is silently ignored data — both reject at
        load.

        Per-entity value lists are also locally validated here:

          * each list has at least one value (an empty list would force
            an undefined RNG draw at generation time);
          * each value is a non-empty string.

        Cross-model checks — that the dict's keys cover every entity
        producing rows in this column's dim table — happen in
        ``plotsim.validation.validate_value_pool_coverage`` because they
        need the ``PlotsimConfig`` entity list, not just the column.
        """
        is_pool_source = self.source.startswith("pool:")
        has_pool_cfg = self.value_pool is not None
        if is_pool_source and not has_pool_cfg:
            raise ValueError(
                f"column {self.name!r} declares source {self.source!r} but "
                f"no 'value_pool' block; add 'value_pool: {{<entity_name>: "
                f"[<value>, ...], ...}}' or change the source"
            )
        if has_pool_cfg and not is_pool_source:
            raise ValueError(
                f"column {self.name!r} has a value_pool block but source "
                f"{self.source!r}; pool sampling replaces the column value, "
                f"so set source to 'pool:<name>' or remove value_pool"
            )
        if self.value_pool is not None:
            for entity_name, values in self.value_pool.items():
                if not entity_name:
                    raise ValueError(
                        f"column {self.name!r} value_pool has an empty " f"entity-name key"
                    )
                if not values:
                    raise ValueError(
                        f"column {self.name!r} value_pool for entity "
                        f"{entity_name!r} is empty; provide at least one "
                        f"value to sample from"
                    )
                for v in values:
                    if not isinstance(v, str) or not v:
                        raise ValueError(
                            f"column {self.name!r} value_pool for entity "
                            f"{entity_name!r} has an empty or non-string "
                            f"value {v!r}; pool values must be non-empty "
                            f"strings"
                        )
        return self

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
    # M117: sub-entity dim row multiplier. Composes multiplicatively with
    # ``Entity.size`` in ``dimensions.build_dim_subentity`` so a builder
    # config (every ``Entity(size=1)``) with ``Table.count=3`` and an
    # engine-direct config (``Entity.size=50``, ``Table.count=1`` default)
    # both resolve their parent's child-row count without branching. Only
    # meaningful on variable-grain dim tables — rejected at load on any
    # other type/grain.
    count: int = Field(default=1, ge=1)

    @property
    def primary_key_cols(self) -> list[str]:
        """Return the PK as a list, whether declared as str or list[str]."""
        return [self.primary_key] if isinstance(self.primary_key, str) else list(self.primary_key)

    _name_is_identifier = field_validator("name")(_identifier_field_validator("table name"))

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
            raise ValueError(f"table {self.name!r} primary_key has duplicate columns: {pk_cols}")
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

    @model_validator(mode="after")
    def _count_only_on_variable_dim(self) -> "Table":
        # M117: ``count`` is the sub-entity dim row multiplier and is only
        # meaningful on variable-grain dim tables. Default 1 is always valid;
        # any larger value on a non-(dim+variable) table is rejected so the
        # field never silently no-ops.
        if self.count > 1 and not (self.type == "dim" and self.grain == "variable"):
            raise ValueError(
                f"table {self.name!r} has count={self.count} but type="
                f"{self.type!r} grain={self.grain!r}; Table.count > 1 is only "
                f"valid on dim tables with grain='variable' (sub-entity dims). "
                f"Default count=1 produces 'one row per parent' on per_entity "
                f"dim children."
            )
        return self


class BridgeMetric(_Frozen):
    """A single metric column on a bridge (M:M) table.

    Bridges are static (one row per association, no period axis), so a
    bridge metric resolves to a single value per row rather than a
    per-period series. Supported sources mirror the fact-column dispatch
    for non-temporal columns:

      * ``metric:<name>`` — value derived from the first-dim entity's
        already-generated metric series (the engine collapses the
        per-period series to its mean, so the bridge cell reflects the
        entity's overall trajectory-driven level for that metric).
      * ``static:<value>`` — literal cell value.
      * ``generated:faker.<method>[...]`` — Faker-driven cell.

    Sources that depend on a period (``fk:dim_date.*``,
    ``derived:period_index``, ``threshold:``, ``proportional:``,
    ``lag:``, ``generated:timestamp/date_key/period_label``) are
    rejected at load — bridges have no time axis to anchor those
    sources against.
    """

    name: str
    dtype: Dtype
    source: str

    _name_is_identifier = field_validator("name")(
        _identifier_field_validator("bridge metric column name")
    )

    @field_validator("source")
    @classmethod
    def _source_format(cls, v: str) -> str:
        parsed = parse_source(v)
        if not isinstance(parsed, (MetricSource, StaticSource, FakerSource)):
            raise ValueError(
                f"bridge metric source {v!r} resolves to "
                f"{type(parsed).__name__}, which is not supported on bridge "
                f"rows. Bridge metrics support metric:, static:, and "
                f"generated:faker.* sources only — period-anchored sources "
                f"like fk:dim_date.*, threshold:, proportional:, lag:, and "
                f"generated:timestamp/date_key/period_label have no time "
                f"axis on a static bridge row."
            )
        return v


class BridgeCardinality(_Frozen):
    """How many associations each entity in the first dim makes with the second.

    ``min`` and ``max`` are inclusive integers. ``trajectory_driven`` on
    the parent ``BridgeTableConfig`` decides whether ``n`` is sampled
    from the entity's trajectory position (closer to ``max`` for higher
    positions) or uniformly at random in ``[min, max]``.
    """

    min: int = Field(ge=0)
    max: int = Field(ge=1)

    @model_validator(mode="after")
    def _min_le_max(self) -> "BridgeCardinality":
        if self.min > self.max:
            raise ValueError(f"bridge cardinality.min ({self.min}) must be <= max ({self.max})")
        return self


class BridgeTableConfig(_Frozen):
    """Many-to-many bridge between two dim tables.

    Bridge tables sit alongside fact and event tables but in their own
    list — ``PlotsimConfig.bridges`` — so the existing ``Table`` model
    keeps its dim/fact/event constraints intact. A bridge produces one
    row per (first-dim entity, second-dim entity) association. The
    number of associations per first-dim entity ranges over
    ``cardinality.min..max``; when ``trajectory_driven=True`` the count
    is biased toward ``max`` for entities at high trajectory positions
    and toward ``min`` for low ones, so bridge density tracks the same
    archetype-driven signal that shapes fact metrics.

    SCD-aware FKs: when a connected dim is SCD-enabled (carries an
    ``scd_type2`` column), the bridge FK references the ``dim_row_id``
    of the active (``is_current=True``) row for that entity rather than
    the natural business key. Non-SCD dims use their PK column directly.

    Bridge rows are static — they're generated once per run, not per
    period. Temporal M:M (a relationship that opens and closes over
    time) is intentionally out of scope for V1.
    """

    name: str
    type: Literal["bridge"] = "bridge"
    connects: list[str] = Field(min_length=2, max_length=2)
    cardinality: BridgeCardinality
    trajectory_driven: bool = True
    metrics: list[BridgeMetric] = Field(default_factory=list, max_length=20)

    _name_is_identifier = field_validator("name")(_identifier_field_validator("bridge table name"))

    @field_validator("connects")
    @classmethod
    def _connects_are_two_distinct_dims(cls, v: list[str]) -> list[str]:
        if len(v) != 2:
            raise ValueError(f"bridge.connects must list exactly 2 dim tables, got {len(v)}")
        if v[0] == v[1]:
            raise ValueError(
                f"bridge.connects entries must be distinct dim tables; got "
                f"both as {v[0]!r} (self-join bridges are not supported)"
            )
        for entry in v:
            _validate_identifier("bridge.connects entry", entry)
        return v

    @model_validator(mode="after")
    def _no_duplicate_metric_names(self) -> "BridgeTableConfig":
        names = [m.name for m in self.metrics]
        if len(set(names)) != len(names):
            raise ValueError(
                f"bridge {self.name!r} has duplicate metric names: {names}; "
                f"each bridge metric column name must be unique"
            )
        return self


class QualityIssue(_Frozen):
    """One configured data-quality corruption to apply post-generation.

    Five issue types are supported, each producing a distinct corruption
    pattern but sharing the same config shape:

      * ``null_injection`` — set ``rate`` of cells in each target column to
        null (NaN for numeric, ``None`` for string/object).
      * ``duplicate_rows`` — insert exact copies of ``rate`` of rows at
        random positions.
      * ``type_mismatch`` — convert ``rate`` of values in each target
        column to the wrong type (numerics rendered as strings, etc.).
      * ``late_arrival`` — for ``rate`` of rows, append an
        ``_arrival_period`` column equal to the original period plus a
        random 1-5 offset; unaffected rows get null in the new column.
      * ``schema_drift`` — for ``rate`` of rows in each target column,
        copy the cell value to a new ``{column}_v2`` column and set the
        original to null; unaffected rows retain the original and have
        null in ``_v2``.

    Determinism: each issue draws from a dedicated
    ``np.random.default_rng(global_seed + seed_offset)`` so reordering
    issues in the config does not perturb earlier issues' draws.
    Default ``seed_offset=0`` is fine for single-issue configs but
    multi-issue configs should set distinct offsets to keep the
    affected row sets independent.

    ``target_columns`` accepts ``"*"`` as a sentinel meaning "all
    eligible metric and attribute columns" (FK and period/date_key
    columns are excluded automatically). Explicit lists are validated
    at load against the resolved target_table's columns.
    """

    type: Literal[
        "null_injection",
        "duplicate_rows",
        "type_mismatch",
        "late_arrival",
        "schema_drift",
    ]
    target_table: str
    target_columns: list[str] = Field(min_length=1, max_length=100)
    rate: float = Field(ge=0.0, le=1.0)
    seed_offset: int = Field(default=0, ge=0, le=2**31 - 1)

    @field_validator("target_columns")
    @classmethod
    def _columns_nonempty_strings(cls, v: list[str]) -> list[str]:
        for entry in v:
            if not isinstance(entry, str) or not entry:
                raise ValueError(
                    f"quality_issues.target_columns entries must be non-empty "
                    f"strings (got {entry!r}); use '*' for the auto-expansion "
                    f"sentinel"
                )
        return v


class QualityConfig(_Frozen):
    """Top-level wrapper for the post-generation quality injection layer.

    Default empty list — configs that don't opt in produce clean output
    identical to baseline. ``output.write_tables`` skips the quality
    pipeline entirely when ``quality_issues`` is empty so the cost is
    zero for non-injected runs.
    """

    quality_issues: list[QualityIssue] = Field(default_factory=list, max_length=50)


class CorrelationPair(_Frozen):
    metric_a: str
    metric_b: str
    coefficient: float = Field(ge=-1.0, le=1.0)


class NoiseConfig(_Frozen):
    gaussian_sigma: float = Field(default=0.0, ge=0.0, le=5.0)
    outlier_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    mcar_rate: float = Field(default=0.0, ge=0.0, le=1.0)


class ManifestConfig(_Frozen):
    """Ground-truth manifest emission config.

    When ``include`` is True (default), ``write_tables`` writes a
    ``manifest.json`` alongside the table files. The manifest records:

      * archetype assigned to each entity (from ``config.entities``)
      * trajectory position at every period for a deterministic sample
        of entities (``trajectory_sample_rate`` of them, picked by
        sorted-name order)
      * event firing periods per entity per event table (the period
        indices where the event fired at least one row)
      * the seed and a SHA-256 of the config dump

    The manifest is the ground-truth signal layer for downstream ML
    feature engineering — anyone training a churn / behavior model on
    plotsim output can read ``manifest.json`` to get the
    archetype-level labels the engine was driving without re-deriving
    them from cell values. ``include: false`` suppresses the file
    entirely (tests, micro-benchmarks).

    ``trajectory_sample_rate`` is bounded ``(0, 1]`` and applied as
    ``max(1, round(n_entities * sample_rate))`` so even at very small
    rates at least one entity's trajectory lands. Determinism: the
    sampled subset is the first N entities under sorted-name order, so
    the same config always selects the same rows regardless of seed.
    """

    include: bool = True
    trajectory_sample_rate: float = Field(default=1.0, gt=0.0, le=1.0)


class EntityFeaturesConfig(_Frozen):
    """Per-entity flat feature table emission config.

    When ``enabled`` is True, ``write_tables`` derives a single
    one-row-per-entity DataFrame by aggregating every fact metric over
    its time series and writes it as ``_entity_features.csv`` (or
    ``.parquet``) alongside the regular table files. The aggregation
    schema per metric is fixed: ``{metric}_mean``, ``{metric}_std``,
    ``{metric}_slope`` (linear regression of value over period index),
    ``{metric}_first``, ``{metric}_last``, ``{metric}_peak_period``.

    ``metrics`` filters which metric names participate; the empty
    default expands to "every numeric metric the engine generated into
    a fact table". Names must reference existing metrics on
    ``config.metrics``; the load-time validator rejects unknowns.
    Bridge metrics are NEVER aggregated regardless of this list — the
    bridge table is associative, not temporal.

    ``include_labels`` controls emission of two ground-truth columns:
    ``archetype`` (sourced from ``config.entities[i].archetype``) and
    ``final_trajectory_position`` (sourced from the last
    ``trajectory_samples`` entry per entity in the manifest). The
    second is NaN for entities outside the manifest's
    ``trajectory_sample_rate`` subset; ``trajectory_sample_rate=1.0``
    (the default) covers every entity.

    ``enabled=true`` requires ``manifest.include=true`` (labels read
    from the manifest payload) and forbids non-empty
    ``quality.quality_issues`` (entity features aggregate the
    pre-corruption fact tables; combining the two would silently mix
    clean and corrupted aggregates). Both rules raise at load time —
    see ``plotsim.validation.validate_entity_features_config``.
    """

    enabled: bool = False
    metrics: list[str] = Field(default_factory=list, max_length=50)
    include_labels: bool = True


class HoldoutConfig(_Frozen):
    """Temporal holdout split for ML target / training workflows.

    When ``enabled`` is True, ``write_tables`` writes two extra companion
    files alongside every per-entity-per-period fact table:

      * ``{table}_train.<csv|parquet>`` — rows whose period index lies in
        ``[0, n_periods - holdout_periods)``.
      * ``{table}_holdout.<csv|parquet>`` — rows whose period index lies
        in ``[n_periods - holdout_periods, n_periods)``.

    The unsplit fact table is still written; the splits are pure
    additions. Dim, bridge, and event tables are NOT split: dims and
    bridges are not period-indexed at all, and event tables carry a
    variable per-period grain that doesn't slice cleanly by a fact-table
    cutoff.

    ``target_metric`` records the prediction target on the run's
    manifest (``target_metric``, ``holdout_periods``, and the resolved
    ``cutoff_period_index``). Downstream feature engineering — when
    ``entity_features.enabled`` is True — restricts every aggregation
    to the training window AND drops the
    ``{target_metric}_{mean,std,slope,first,last,peak_period}`` columns
    so the per-entity feature row never leaks the label.

    ``min_training_periods`` (default 3) is the floor on
    ``n_periods - holdout_periods`` enforced at config load — splits
    that leave fewer than this many training periods raise rather than
    silently producing a one-or-two-period training set with
    pathological ``slope`` values.

    ``enabled=true`` requires:
      * ``target_metric`` set,
      * ``target_metric`` resolves to a numeric metric on a fact table,
      * ``holdout_periods >= 1``,
      * ``n_periods - holdout_periods >= min_training_periods``,
      * ``quality.quality_issues == []`` (the splits operate on the
        clean tables; combining the two would leave train/holdout files
        whose semantics depend silently on whether quality was applied
        before or after the slice. Deferred to a future mission).

    Disabled (default) skips the split entirely.
    """

    enabled: bool = False
    target_metric: Optional[str] = None
    holdout_periods: int = Field(default=0, ge=0, le=10_000)
    min_training_periods: int = Field(default=3, ge=1, le=10_000)


class OutputConfig(_Frozen):
    """Output format selector and target directory.

    ``format`` accepts ``"parquet"`` in addition to the default ``"csv"``.
    CSV remains the default; configs that omit ``format`` (or set
    ``format: csv``) write ``.csv`` files. Parquet output is column-typed
    and typically 5-10x smaller on the bundled templates; the engine path
    is identical, only the on-disk encoding differs.

    Parquet writes go through ``pyarrow``, declared as the optional
    extra ``plotsim[parquet]``. When pyarrow is not installed and
    ``format: parquet`` is configured, ``write_tables`` raises an
    ``ImportError`` naming the install command — fail-fast at the
    write call rather than mid-iteration.

    ``cell_budget`` (M7) is the per-config override of the soft
    cell-count gate enforced by ``_combined_scale_estimator``. ``None``
    (default) falls through to ``PLOTSIM_CELL_BUDGET`` env var, then
    to ``_CELL_SOFT_BUDGET_DEFAULT`` (2,000,000). ``0`` disables the
    soft gate entirely (the 50,000,000-cell hard ceiling still
    applies). A positive integer raises (or lowers) the soft cap to
    that value. Promoting this knob into the config makes 10M+ cell
    runs reproducible from the YAML alone — no env vars required —
    which is the contract the bundled ``lakehouse.yaml`` template
    documents for large-scale generation.
    """

    format: Literal["csv", "parquet"] = "csv"
    directory: str
    cell_budget: Optional[int] = Field(default=None, ge=0)


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

    * **Hysteresis** (``threshold_exit ≤ threshold_enter``) —
      ``threshold_enter`` is the upward entry threshold;
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

    ``enforce_order`` defaults to ``False`` (free-mode per-period
    assignment). Each period independently picks the highest-enter
    stage the realized value satisfies — no forward-only cursor, no
    lock-in. Set ``enforce_order: true`` explicitly to opt into the
    monotonic stage walk (cursor advances only, optionally with the
    ``downgrade_delay`` relaxation below). Irreversible lifecycle
    transitions are already covered by SCD Type 2; stages are intended
    to reflect *current* lifecycle state, which is why free-mode is
    the default.

    ``downgrade_delay`` relaxes strict monotonicity under
    ``enforce_order=True`` by letting the cursor step backwards once an
    entity has sat below the demote threshold for ``downgrade_delay``
    consecutive periods. ``None`` (default) preserves strict-monotonic
    behavior under legacy mode and immediate-demote behavior under
    hysteresis mode. Ignored when ``enforce_order=False``.
    """

    field: str
    sequence: list[StageDefinition] = Field(min_length=2, max_length=10)
    enforce_order: bool = False
    downgrade_delay: Optional[int] = Field(default=None, ge=1, le=120)

    @property
    def mode(self) -> str:
        """Derived per-config: ``'legacy'`` or ``'hysteresis'``.

        Derived from the relationship between ``threshold_exit`` and
        ``threshold_enter`` on the first non-terminal stage.
        ``_sequence_is_valid`` enforces consistency, so the mode is
        unambiguous after load. Sequences with only a terminal stage
        cannot exist (``min_length=2``); a sequence whose non-terminal
        stages all have ``exit > enter`` is ``'legacy'``; otherwise
        ``'hysteresis'``.
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
                    f"stage {stage.name!r} is not terminal but has " f"threshold_exit: null"
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
    # M117: cap raised from 100 → 100,000 to accommodate the builder's
    # per-segment expansion (one ``Entity(size=1)`` per row in the resulting
    # dim table). The combined-scale gate
    # (``_combined_scale_estimator``) still bounds runtime cell-count via
    # ``sum(entities.size) × period_count``, so this raise relaxes the
    # per-row-count cap without weakening the runtime envelope.
    entities: list[Entity] = Field(min_length=1, max_length=100_000)
    tables: list[Table] = Field(max_length=50)
    # 1_225 = 50 choose 2 — the upper bound on unique pairwise correlations
    # given the 50-metric cap. Anything larger is either duplicates (rejected
    # at matrix assembly) or references to non-existent metrics.
    correlations: list[CorrelationPair] = Field(default_factory=list, max_length=1_225)
    noise: NoiseConfig = Field(default_factory=NoiseConfig)
    output: OutputConfig
    # M105: manifest emission. Default ``include=true`` so every ``plotsim
    # run`` lands a ``manifest.json`` next to the table files. Configs that
    # want to skip the file (microbenchmarks, sandboxed CI) set
    # ``manifest: {include: false}``.
    manifest: ManifestConfig = Field(default_factory=ManifestConfig)
    stages: Optional[StageSequence] = None
    # M107: many-to-many bridge tables. Sit alongside ``tables`` (dim/fact/
    # event) but in their own list so the existing Table model keeps its
    # grain and PK constraints untouched. Empty default — configs that
    # don't opt in pay zero cost.
    bridges: list[BridgeTableConfig] = Field(default_factory=list, max_length=20)
    # M107: post-generation data-quality injection. Empty default; configs
    # without quality_issues produce clean output identical to pre-M107
    # baselines.
    quality: QualityConfig = Field(default_factory=QualityConfig)
    # M108: per-entity flat feature table. Default ``enabled=false`` —
    # configs that don't opt in produce no extra file. When opted in,
    # the writer emits ``_entity_features.csv`` (or ``.parquet``)
    # alongside the standard table set.
    entity_features: EntityFeaturesConfig = Field(default_factory=EntityFeaturesConfig)
    # M109: temporal holdout split. Default ``enabled=false`` — configs
    # without an opt-in produce no train/holdout companion files.
    # When enabled, every per_entity_per_period fact table gets two
    # extra files written, the manifest records the split, and entity
    # features (if enabled) aggregate over training periods only with
    # the target-metric columns excluded.
    holdout: HoldoutConfig = Field(default_factory=HoldoutConfig)
    # FIX-05 / SF-3: locale threaded to every Faker instance built by the
    # dim/fact/event layers. String (``"en_US"``, ``"ja_JP"``) or list
    # (multi-locale mix). Default ``"en_US"`` preserves prior behavior.
    locale: str | list[str] = "en_US"
    # M119: global seasonal modulation. Default empty list — configs without
    # opt-in produce output byte-identical to pre-M119 baselines because the
    # per-period summed strength is 0.0 and the metrics pipeline short-circuits
    # the modulation step. ``max_length=12`` matches the calendar-month domain.
    seasonal_effects: list[SeasonalEffect] = Field(
        default_factory=list,
        max_length=12,
    )

    # M120: trajectory-aware correlation pre-compensation. Default ``false``
    # for engine-direct configs so the bundled templates (and any pre-M120
    # YAML on disk) keep producing byte-identical CSV output. The builder
    # interpreter sets ``true`` explicitly so user-declared connections land
    # as table-wide correlations matching the configured signs and
    # magnitudes — the trajectory's own structural covariance otherwise
    # dominates the copula at mixed-archetype scale.
    compensate_correlations: bool = False

    # M121: dual-path generation. ``serial`` walks entities one at a time
    # through ``generate_entity_metrics`` (the pre-M121 hot path);
    # ``vectorized`` groups entities by archetype and runs batched numpy
    # samplers + a batched copula across the entity axis at each period;
    # ``auto`` selects ``vectorized`` when the total entity count meets
    # ``_VECTORIZED_AUTO_THRESHOLD`` (50) and ``serial`` below it. Default
    # ``serial`` for engine-direct configs preserves bundled-template
    # byte-identity on disk; the builder interpreter sets ``auto``
    # explicitly. The two paths consume RNG in different orders, so output
    # is statistically equivalent but not byte-identical between modes —
    # within a mode, ``(config, seed, generation_mode)`` reproduces bytes
    # exactly. Documented in `docs/engine-internals.md` §2.4 and §2.4a.
    generation_mode: Literal["serial", "vectorized", "auto"] = "auto"

    # Populated by ``_correlation_matrix_is_psd`` when the user's
    # correlation matrix had to be Higham-projected to nearest PD.
    # ``None`` for runs where the matrix was already PD (the common case)
    # or where no correlations were configured. Read by
    # ``plotsim.manifest.build_manifest`` to surface the adjustment list
    # in ``manifest.correlation_adjustments``. PrivateAttr because the
    # value is engine-derived, not a user input — round-tripping it
    # through ``model_dump`` would pollute the YAML round-trip and the
    # config_sha256 fingerprint.
    _correlation_adjustments: Optional[list[dict]] = PrivateAttr(default=None)

    # Populated by ``plotsim.tables.generate_tables_with_state`` when
    # ``compensate_correlations=True`` and at least one user-declared pair
    # was compensated. ``None`` for runs without compensation (engine-direct
    # default, builder runs without ``connections``). Read by
    # ``plotsim.manifest.build_manifest`` to surface
    # ``manifest.correlation_compensations``. Distinct from the
    # ``_correlation_adjustments`` attr: that records "your matrix wasn't PD,
    # we projected it"; this records "your target's been compensated for the
    # trajectory's structural contribution before reaching the copula." Both
    # may populate on a single run.
    _correlation_compensations: Optional[list[dict]] = PrivateAttr(default=None)

    # Populated by ``plotsim.tables.generate_tables_with_state`` only
    # in vectorized mode. ``dict[archetype_name → cell_count]`` of cells
    # that triggered ``_apply_correlations_batch``'s per-row scalar
    # fallback. ``None`` in serial mode so the manifest's
    # ``bypass_fallback_counts`` field can distinguish "vectorized with
    # zero bypass" (empty dict) from "serial — never measured" (None).
    # PrivateAttr because the value is a per-run side-effect of generation,
    # not a user input — round-tripping it through ``model_dump`` would
    # pollute the YAML and the config_sha256 fingerprint, same reasoning
    # as the sibling attrs above.
    _bypass_fallback_counts: Optional[dict[str, int]] = PrivateAttr(default=None)

    # 0.6-M5: stashed by ``plotsim.tables.generate_tables_with_state`` at
    # the Cholesky-build site so ``plotsim.manifest.build_manifest`` can
    # surface the projected (post-Higham, post-M120 compensation) coefficient
    # for every user-declared correlation pair. ``None`` for runs without
    # ``correlations`` configured. The companion ``_metric_correlation_order``
    # records the toposorted metric order used to assemble the matrix —
    # needed to translate ``(metric_a, metric_b)`` to row/col indices when
    # the manifest reads back. PrivateAttr for the same reason as siblings:
    # engine-derived, not a user input; would pollute config_sha256.
    _projected_correlation_matrix: Optional[Any] = PrivateAttr(default=None)
    _metric_correlation_order: Optional[list[str]] = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _total_entity_size_within_limit(self) -> "PlotsimConfig":
        total = sum(e.size for e in self.entities)
        if total > 100_000:
            raise ValueError(
                f"Total entity count across all groups is {total:,}. " f"Maximum is 100,000."
            )
        return self

    @model_validator(mode="after")
    def _combined_scale_estimator(self) -> "PlotsimConfig":
        """Category B Layer 2: detect multiplicative compounding at load.

        Per-field bounds (Layer 1) don't catch configs where every field is
        within its cap but the product ``sum(entities.size) × period_count``
        is ruinously large. The scalability report (§6) identified that product
        as the dominant predictor of wall-clock and memory.

        Tiered behavior driven by the ``cell_count`` and two env vars:
          * Always prints the config summary line to stderr.
          * ``cell_count`` ≤ ``_CELL_ADVISORY_THRESHOLD``: silent.
          * ``_CELL_ADVISORY_THRESHOLD`` < ``cell_count`` ≤ soft budget:
            stderr advisory recommending parquet + vectorized.
          * soft budget < ``cell_count`` ≤ ``_CELL_HARD_CEILING`` and
            opt-in not given: ``ValueError`` pointing at how to opt in.
          * Same range, opt-in given: stderr advisory, proceed.
          * ``cell_count`` > ``_CELL_HARD_CEILING``: ``ValueError``
            regardless of opt-in.

        Soft budget precedence (M7): explicit ``output.cell_budget``
        config field > ``PLOTSIM_CELL_BUDGET`` env var >
        ``_CELL_SOFT_BUDGET_DEFAULT``. ``0`` (via either surface)
        disables the soft gate. Opt-in is read from
        ``PLOTSIM_ALLOW_LARGE_DATASET`` (set by the CLI
        ``--allow-large-dataset`` flag, or directly by library
        callers); raising ``output.cell_budget`` past the projected
        cell count is the YAML-only equivalent.
        """
        n_entities = sum(e.size for e in self.entities)
        n_periods = self.time_window.period_count()
        cell_count = n_entities * n_periods

        n_fact_fields = sum(len(t.columns) for t in self.tables if t.type == "fact")
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
                (m for m in self.metrics if m.name == parsed_rc.metric),
                None,
            )
            if driver is None or driver.value_range is None:
                continue
            v_max = driver.value_range.max
            if v_max is None:
                continue
            event_rows_upper += int(n_entities * n_periods * v_max * parsed_rc.scale)

        summary = (
            f"Config summary: {n_entities:,} entities × {n_periods:,} periods "
            f"= {cell_count:,} cells, {len(self.metrics)} metrics, "
            f"{len(self.tables)} tables. Estimated peak memory: ~{peak_mb:.0f} MB."
        )
        if event_rows_upper > 0:
            summary += f" Expected event rows (upper bound): ~{event_rows_upper:,}."
        sys.stderr.write(summary + "\n")

        soft_budget = _resolve_cell_budget(self.output.cell_budget)
        allow_large = _allow_large_dataset()

        if cell_count > _CELL_HARD_CEILING:
            raise ValueError(
                f"Config produces {cell_count:,} cells (entities × periods), "
                f"which exceeds the hard ceiling of {_CELL_HARD_CEILING:,}. "
                f"Reduce entity count, time window span, or coarsen "
                f"granularity. The hard ceiling is not configurable; configs "
                f"this large should be split or chunked."
            )

        if soft_budget > 0 and cell_count > soft_budget and not allow_large:
            raise ValueError(
                f"Config produces {cell_count:,} cells (entities × periods), "
                f"which exceeds the soft budget of {soft_budget:,}. To "
                f"proceed, raise the budget in your config "
                f"(output.cell_budget: N, or 0 to disable), set "
                f"PLOTSIM_CELL_BUDGET=N in the environment, or pass "
                f"--allow-large-dataset on the CLI / set "
                f"PLOTSIM_ALLOW_LARGE_DATASET=1. Recommend "
                f"output.format=parquet and generation_mode=auto for runs "
                f"this size — see the 'Limits and performance gates' "
                f"section of the config reference."
            )

        if soft_budget > 0 and cell_count > soft_budget and allow_large:
            sys.stderr.write(
                f"Large dataset opt-in: {cell_count:,} cells exceeds the "
                f"{soft_budget:,} soft budget but PLOTSIM_ALLOW_LARGE_DATASET "
                f"is set — proceeding. Recommend output.format=parquet and "
                f"generation_mode=auto for memory and runtime bounds.\n"
            )
        elif cell_count > _CELL_ADVISORY_THRESHOLD:
            sys.stderr.write(
                f"Advisory: {cell_count:,} cells exceeds "
                f"{_CELL_ADVISORY_THRESHOLD:,}. Recommend "
                f"output.format=parquet for memory bounds; "
                f"generation_mode=auto picks the vectorized lane "
                f"automatically.\n"
            )
        return self

    @model_validator(mode="after")
    def _cross_reference_integrity(self) -> "PlotsimConfig":
        """Cross-table integrity orchestrator.

        Each rule group lives in ``plotsim.validation`` so it can be
        called and tested independently. The first error from any group
        is raised here; advisory warnings (e.g. zero-coefficient
        correlations) are emitted as side effects of the validator
        functions and survive through to the user.
        """
        # Local import: plotsim.validation imports from plotsim.config,
        # so a module-level import would create a cycle.
        from plotsim.validation import (
            validate_advanced,
            validate_archetype_refs,
            validate_correlations,
            validate_names,
            validate_stages,
            validate_table_schema,
        )

        for check in (
            validate_names,
            validate_archetype_refs,
            validate_table_schema,
            validate_correlations,
            validate_stages,
            validate_advanced,
        ):
            errors = check(self)
            if errors:
                raise ValueError(errors[0])
        return self

    # The pre-split implementation lived here as the body of
    # ``_cross_reference_integrity``. Its rule groups now live in
    # ``plotsim.validation`` as ``validate_names``,
    # ``validate_archetype_refs``, ``validate_table_schema``,
    # ``validate_correlations``, ``validate_stages``, and
    # ``validate_advanced``. The orchestrator above is the only entry
    # point.

    @model_validator(mode="after")
    def _entity_features_gates(self) -> "PlotsimConfig":
        """Load-time gates for the entity-features feature.

        Mutual-exclusion rules are enforced here rather than at
        generation time so a misconfigured YAML fails before the engine
        burns work. The check function lives in ``plotsim.validation``
        (mirroring the ``validate_correlation_psd`` pattern) so the
        post-generation suite and the load-time validator share the
        same source of truth.
        """
        if not self.entity_features.enabled:
            return self
        # Local import: plotsim.validation imports from plotsim.config,
        # so a module-level import would create a cycle.
        from plotsim.validation import validate_entity_features_config

        errors = validate_entity_features_config(self)
        if errors:
            raise ValueError(errors[0])
        return self

    @model_validator(mode="after")
    def _cold_start_active_periods_gate(self) -> "PlotsimConfig":
        """0.6-M8b: load-time gate for cold-start ``Entity.start_period``.

        Every entity must have at least ``MIN_ACTIVE_PERIODS`` periods
        active (``start_period + MIN_ACTIVE_PERIODS <= n_periods``).
        Catches both engine-direct configs that set ``start_period``
        past the deadline AND builder configs whose segment arrival
        distributions drew start_periods past it. The check function
        lives in ``plotsim.validation`` so the same gate applies to
        any caller (CLI, builder, programmatic) that round-trips
        through PlotsimConfig.
        """
        # Local import: plotsim.validation imports from plotsim.config,
        # so a module-level import would create a cycle.
        from plotsim.validation import validate_cold_start_active_periods

        errors = validate_cold_start_active_periods(self)
        if errors:
            raise ValueError(errors[0])
        return self

    @model_validator(mode="after")
    def _value_pool_gates(self) -> "PlotsimConfig":
        """Load-time gates for ``PoolSource`` columns.

        Cross-model coverage check (per_entity dim restriction + key set
        equals entity set) lives in
        ``plotsim.validation.validate_value_pool_coverage``; the local
        Column-level pairing check is on ``Column._pool_pairing``. Same
        split as the SCD Type 2 / entity-features gates: structural
        per-Column rules belong on the Column model, cross-model rules
        belong here.
        """
        # Local import: plotsim.validation imports from plotsim.config,
        # so a module-level import would create a cycle.
        from plotsim.validation import validate_value_pool_coverage

        errors = validate_value_pool_coverage(self)
        if errors:
            raise ValueError(errors[0])
        return self

    @model_validator(mode="after")
    def _holdout_gates(self) -> "PlotsimConfig":
        """Load-time gates for the holdout-split feature.

        Mirrors the ``_entity_features_gates`` pattern — pure config
        check, no DataFrame inputs. Delegates to
        ``plotsim.validation.validate_holdout_config`` so the gate's
        decision logic lives next to the rest of the load-time
        validators and stays reusable from contexts that want a full
        diagnostic instead of fail-fast.
        """
        if not self.holdout.enabled:
            return self
        from plotsim.validation import validate_holdout_config

        errors = validate_holdout_config(self)
        if errors:
            raise ValueError(errors[0])
        return self

    @model_validator(mode="after")
    def _correlation_matrix_is_psd(self) -> "PlotsimConfig":
        """Project non-PD correlation matrices and warn — don't reject.

        8+ correlated metrics with moderate coefficients are often non-PD
        by combinatorial accident, not by user error. We run Higham
        nearest-PD projection (in ``plotsim.metrics``), emit a per-pair
        adjustment warning so the user sees what changed, and stash the
        adjustment records on a private attribute that the manifest reads.

        The projected matrix is NOT stashed: it lives in declaration
        order, while ``plotsim.tables.generate_tables`` builds the
        hoisted Cholesky in toposort order. Re-projection downstream
        (deterministic, ~ms) is cheaper than threading order-aware
        permutations through the engine.

        Only raises if Higham AND the eigenvalue-clipping fallback both
        fail to produce a PD matrix — should be impossible for
        symmetric input.
        """
        if not self.correlations:
            return self
        # Local import: plotsim.validation imports from plotsim.config,
        # so a module-level import would create a cycle.
        from plotsim.metrics import _format_correlation_adjustment_warning
        from plotsim.validation import project_correlation_or_issue

        issues, adjustments, _projected = project_correlation_or_issue(self)
        if issues:
            names = [m.name for m in self.metrics]
            raise ValueError(
                f"correlations: correlation matrix could not be projected "
                f"to positive-definite for metrics {names}. "
                f"{issues[0].message}"
            )
        if adjustments:
            warnings.warn(
                _format_correlation_adjustment_warning(adjustments),
                UserWarning,
                stacklevel=2,
            )
            self._correlation_adjustments = adjustments
        return self


def load_config(path: str | Path) -> PlotsimConfig:
    """Load and validate a plotsim YAML config file."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"config file {p} did not parse to a mapping (got {type(data).__name__})")
    return PlotsimConfig(**data)


def dump_config(config: PlotsimConfig) -> str:
    """Dump a PlotsimConfig back to a YAML string (round-trippable)."""
    return yaml.safe_dump(
        config.model_dump(mode="python"),
        sort_keys=False,
        default_flow_style=False,
    )
