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
#                           (also allowed on fact tables as a parent of
#                           per_parent_row children — fct_orders header
#                           with row count driven by row_count_source)
#   per_parent_row          fct_order_items: one row per parent fact row
#                           times a uniform draw in children_per_row.
#                           Inherits entity + period from parent, carries
#                           an fk:fct_<parent>.<pk> column and independent
#                           draws on remaining columns (no trajectory).
GRAINS: frozenset[str] = frozenset(
    {
        "per_entity",
        "per_period",
        "per_reference",
        "per_entity_per_period",
        "variable",
        "per_parent_row",
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

    @field_validator("kwargs")
    @classmethod
    def _kwargs_size_capped(cls, v: dict[str, str]) -> dict[str, str]:
        if len(v) > 20:
            raise ValueError(
                f"FakerSource.kwargs has {len(v)} entries; the per-call "
                f"kwargs dict is capped at 20 to keep config-load and "
                f"per-row faker dispatch bounded"
            )
        return v


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
    """Per-entity value pool on a column.

    Grammar: ``pool:<name>``. ``<name>`` is a free-form identifier that
    distinguishes multiple pool columns on the same table (e.g. ``industry``
    vs ``segment``); the actual values are stored on the column under
    ``Column.value_pool: dict[entity_name, list[str]]``.

    A column with this source MUST also declare ``value_pool`` (validated
    on ``Column``); a column with a non-``pool:`` source MUST NOT declare
    ``value_pool``. The two are paired or both absent — same discipline
    as the SCD Type 2 pairing.

    Architectural firewall: pools resolve entity-membership only, never
    trajectory-derived. The dim/fact/event layers never read a
    trajectory at pool dispatch time; pool selection consumes one RNG
    draw per row, deterministic under the engine's single-seed contract.

    0.6-M19 Fix 1: pool sources work on per_entity dims (the original
    M114 surface), variable-grain facts, per_parent_row child facts,
    and event tables. The per-row pool lookup uses the row's entity
    PK → entity-name → ``value_pool[entity_name]`` chain; rows whose
    entity is not in ``value_pool.keys()`` cause a load-time error
    (``validate_value_pool_coverage`` requires the key set to cover
    every entity that produces rows in this table).
    """

    name: str

    _name_is_identifier = field_validator("name")(_identifier_field_validator("pool name"))


class RangeSource(_Frozen):
    """Per-row uniform draw between bounds.

    Grammar: ``range:<min>:<max>``. Both bounds parse as ``float``;
    the engine derives the draw mode from the column's ``dtype``:

      * ``dtype: int`` → ``rng.integers(int(min), int(max) + 1)`` —
        inclusive upper bound, integer output.
      * ``dtype: float`` → ``rng.uniform(min, max)`` — exclusive
        upper bound by numpy convention, float output.

    Valid on every table type — per_entity dims, sub-entity dims,
    reference dims, per_entity_per_period facts, per_period facts,
    variable-grain facts, per_parent_row child facts, and event
    tables. Each per-row draw consumes one RNG call so output stays
    deterministic under the engine's single-seed contract.

    Bounds validation: ``max >= min``. Equal bounds produce a
    constant column (single draw collapses to ``min``); the validator
    permits this rather than error, matching numpy's behavior.

    0.6-M19 Fix 2: introduced as a structured alternative to
    ``faker.pyfloat`` / ``faker.random_int`` for cases where the
    author wants an explicit numeric range rather than Faker's
    default bounds. Use ``faker.random_int`` for the legacy unbounded
    default; use ``range`` for ``quantity ∈ [1, 5]``, ``unit_price
    ∈ [10.0, 500.0]`` and similar shape constraints.
    """

    min: float
    max: float

    @model_validator(mode="after")
    def _max_not_less_than_min(self) -> "RangeSource":
        if self.max < self.min:
            raise ValueError(f"range source: max ({self.max}) must be >= min ({self.min})")
        return self


# Placeholder pattern for narrative templates. ``{slot}`` literals where
# ``slot`` is a valid identifier. Used both at config-load to validate
# that the lexicon's slot keys match the template's placeholders, and at
# cell-build time to enumerate slots in declaration order.
_NARRATIVE_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


class NarrativeSource(_Frozen):
    """Marker source for a fact column carrying trajectory- and archetype-driven text.

    Grammar: ``narrative:<key>``. ``<key>`` is a free-form identifier that
    distinguishes multiple narrative columns on the same row (e.g.
    ``narrative:review_text`` vs ``narrative:support_ticket``); the
    actual template + per-archetype lexicons live on
    ``Column.narrative: NarrativeConfig`` so the dispatcher in
    ``plotsim.tables`` has structured access without re-parsing strings.

    A column with this source MUST also declare a ``narrative`` config
    block (validated on ``Column``); a column with a non-narrative source
    MUST NOT declare ``narrative``. The two are paired or both absent —
    same discipline as :class:`SCDType2Source` / :class:`PoolSource`.

    Architectural placement: narrative columns are fact-only and force
    the scalar fact-builder path because phrase sampling consumes one
    RNG draw per slot per row — vectorizing would re-order draws and
    break byte-parity for the Layer-4 reference fixtures. The dim layer
    rejects narrative sources at load time; per-period and event grains
    are out of scope (no per-row trajectory plumbing wired).

    Trajectory-first invariant: each emitted cell is a deterministic
    function of ``(trajectory position, archetype, RNG state)``. Same
    seed → byte-identical text column. The signal a downstream
    classifier learns is the per-archetype lexicon × per-band vocabulary
    mapping; the sampling is intentionally non-degenerate (intra-band
    uniform across the phrase pool, with operator-controlled overlap
    between archetypes' bands) so the signal is learnable but not a
    one-to-one phrase→archetype lookup.
    """

    key: str

    _key_is_identifier = field_validator("key")(_identifier_field_validator("narrative key"))


class NarrativeConfig(_Frozen):
    """Lexicon + template config for a :class:`NarrativeSource` column.

    Each fact-cell is built by:

      1. Reading the entity's trajectory position ``p ∈ [0, 1]`` at the
         current period.
      2. Mapping ``p`` to one of ``N = len(bands)`` evenly-spaced bands
         via ``min(int(p * N), N - 1)`` (same arithmetic
         :class:`TextBucketSource` uses; ``p == 1.0`` lands in the last
         band rather than overflowing).
      3. Looking up the entity's archetype to pick which lexicon applies.
      4. For each ``{slot}`` placeholder in ``template``, sampling one
         phrase from ``lexicons[archetype][slot][bands[band_idx]]`` via
         the seeded engine RNG.
      5. ``template.format(**slot_values)`` → final cell value.

    Lexicon shape:
        ``{archetype_name: {slot_name: {band_name: [phrase, ...]}}}``

    Load-time validation gates (this model):

      * ``template`` contains at least one ``{slot}`` placeholder, no
        duplicate placeholders, slot names are valid identifiers.
      * ``bands`` has 2–20 entries (mirrors :class:`TextBucketSource`;
        the bound is for lexicon-authoring ergonomics, not engine math —
        the band-index arithmetic supports any ``N >= 1``).
      * ``bands`` entries are non-empty unique strings.
      * For every archetype, ``lexicons[archetype]`` keys equal the
        template's ``{slot}`` placeholder set.
      * For every slot, ``lexicons[archetype][slot]`` keys equal the
        ``bands`` tuple as a set.
      * Each band's phrase list is non-empty and contains only non-empty
        strings.

    Cross-config gates (in ``PlotsimConfig._narrative_gates``):

      * ``lexicons`` keys are a subset of ``config.archetypes[].name``.
      * The owning column's table has ``type == "fact"`` and
        ``grain == "per_entity_per_period"``.
    """

    template: str = Field(min_length=1, max_length=1000)
    lexicons: dict[str, dict[str, dict[str, list[str]]]]
    bands: tuple[str, ...] = ("low", "mid", "high")

    @field_validator("bands")
    @classmethod
    def _bands_shape(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        if len(v) < 2 or len(v) > 20:
            raise ValueError(
                f"narrative bands must have between 2 and 20 entries, got {len(v)} ({list(v)!r})"
            )
        if any(not isinstance(b, str) or not b for b in v):
            raise ValueError(f"narrative bands must all be non-empty strings, got {list(v)!r}")
        if len(set(v)) != len(v):
            raise ValueError(f"narrative bands must be unique, got {list(v)!r}")
        return v

    @model_validator(mode="after")
    def _template_and_lexicons_consistent(self) -> "NarrativeConfig":
        slots = _NARRATIVE_PLACEHOLDER_RE.findall(self.template)
        if not slots:
            raise ValueError(
                f"narrative template {self.template!r} has no {{slot}} "
                f"placeholders; a fully static sentence does not need a "
                f"narrative source — use 'static:<value>' instead"
            )
        slot_set = set(slots)
        if len(slots) != len(slot_set):
            seen: set[str] = set()
            dups: list[str] = []
            for s in slots:
                if s in seen and s not in dups:
                    dups.append(s)
                seen.add(s)
            raise ValueError(
                f"narrative template {self.template!r} has duplicate slot "
                f"placeholders {sorted(dups)}; each {{slot}} must be unique "
                f"so the per-slot phrase sampling is unambiguous"
            )

        if not self.lexicons:
            raise ValueError(
                "narrative lexicons must declare at least one archetype "
                "(map archetype name → slot → band → phrase list)"
            )

        bands_set = set(self.bands)
        for archetype, by_slot in self.lexicons.items():
            if not isinstance(archetype, str) or not archetype:
                raise ValueError(
                    f"narrative lexicons archetype key must be a non-empty "
                    f"string, got {archetype!r}"
                )
            slots_present = set(by_slot)
            if slots_present != slot_set:
                missing = slot_set - slots_present
                extra = slots_present - slot_set
                raise ValueError(
                    f"narrative lexicons[{archetype!r}] slots "
                    f"{sorted(slots_present)} do not match template "
                    f"placeholders {sorted(slot_set)}: "
                    f"missing {sorted(missing)}, unexpected {sorted(extra)}"
                )
            for slot, by_band in by_slot.items():
                if not _IDENTIFIER_RE.fullmatch(slot):
                    raise ValueError(
                        f"narrative lexicons[{archetype!r}][{slot!r}]: "
                        f"slot name is not a valid identifier"
                    )
                bands_present = set(by_band)
                if bands_present != bands_set:
                    missing = bands_set - bands_present
                    extra = bands_present - bands_set
                    raise ValueError(
                        f"narrative lexicons[{archetype!r}][{slot!r}] bands "
                        f"{sorted(bands_present)} do not match declared "
                        f"bands {sorted(bands_set)}: "
                        f"missing {sorted(missing)}, unexpected {sorted(extra)}"
                    )
                for band, phrases in by_band.items():
                    if not isinstance(phrases, list) or not phrases:
                        raise ValueError(
                            f"narrative lexicons[{archetype!r}][{slot!r}]"
                            f"[{band!r}]: phrase list must be a non-empty list"
                        )
                    if len(phrases) > 100:
                        raise ValueError(
                            f"narrative lexicons[{archetype!r}][{slot!r}]"
                            f"[{band!r}] has {len(phrases)} phrases; the "
                            f"per-band phrase list is capped at 100 to keep "
                            f"config-load memory bounded"
                        )
                    for phrase in phrases:
                        if not isinstance(phrase, str) or not phrase:
                            raise ValueError(
                                f"narrative lexicons[{archetype!r}][{slot!r}]"
                                f"[{band!r}]: phrases must be non-empty "
                                f"strings, got {phrase!r}"
                            )
        return self

    def template_slots(self) -> list[str]:
        """Return the ordered list of ``{slot}`` placeholders in ``template``.

        Order is template scan order (``re.findall`` left-to-right). The
        cell builder iterates this list to draw one phrase per slot in a
        deterministic order under the seeded RNG.
        """
        return _NARRATIVE_PLACEHOLDER_RE.findall(self.template)


class NestedSource(_Frozen):
    """Marker source for a column carrying a nested (struct or array) value per cell.

    Grammar: the literal source string ``nested`` parses to this marker.
    The cell shape (``dict`` for struct, ``list`` for array) is determined
    by ``Column.dtype`` plus ``Column.nested_schema`` (struct) or
    ``Column.array_element_type`` + ``Column.array_length`` (array).

    A column with this source MUST also declare ``dtype: struct`` (with
    ``nested_schema``) or ``dtype: array`` (with ``array_element_type``);
    the pairing is enforced by ``Column._nested_pairing``. A column with
    one of those dtypes that uses any other source is rejected — nested
    cells need the dedicated builder path.

    Scope (V1): one level of nesting. Struct fields are primitive
    (``int`` / ``float`` / ``string`` / ``boolean``); array elements are
    primitive. No struct-of-struct, array-of-struct, or nested-of-nested.
    The cell builder draws a value per field/element from a seeded RNG so
    same-seed runs produce byte-identical nested cells.
    """


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
    | RangeSource
    | NarrativeSource
    | NestedSource
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
    "'pool:<name>', "
    "'range:<min>:<max>', "
    "'narrative:<key>', "
    "'nested'"
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
    if source == "nested":
        return NestedSource()

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
                f"threshold source {source!r}: non-integer consecutive {consecutive_str!r}"
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
                f"proportional source {source!r} must be 'proportional:<metric>:scale:<multiplier>'"
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
                f"pool source {source!r}: prefix 'pool:' requires a name (e.g. 'pool:industry')"
            )
        # Reject embedded colons: ``pool:industry:extra`` would be ambiguous
        # under any future grammar extension. Surface it now.
        if ":" in body:
            raise ValueError(
                f"pool source {source!r} must be 'pool:<name>' with no extra "
                f"colons; the actual values live on Column.value_pool"
            )
        return PoolSource(name=body)

    if source.startswith("range:"):
        parts = source.split(":")
        if len(parts) != 3 or parts[0] != "range":
            raise ValueError(f"range source {source!r} must be 'range:<min>:<max>'")
        _, min_str, max_str = parts
        try:
            min_val = float(min_str)
        except ValueError as e:
            raise ValueError(f"range source {source!r}: non-numeric min {min_str!r}") from e
        try:
            max_val = float(max_str)
        except ValueError as e:
            raise ValueError(f"range source {source!r}: non-numeric max {max_str!r}") from e
        return RangeSource(min=min_val, max=max_val)

    if source.startswith("narrative:"):
        body = source[len("narrative:") :]
        if not body:
            raise ValueError(
                f"narrative source {source!r}: prefix 'narrative:' requires "
                f"a key (e.g. 'narrative:review_text')"
            )
        if ":" in body:
            raise ValueError(
                f"narrative source {source!r} must be 'narrative:<key>' "
                f"with no extra colons; the actual template + lexicons "
                f"live on Column.narrative"
            )
        # NarrativeSource's own field validator enforces identifier rules
        # on ``key`` — re-using the same path keeps the error consistent
        # whether the source is parsed from a string or constructed directly.
        return NarrativeSource(key=body)

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

    Adstock-style decay (opt-in): set ``decay=True`` plus
    ``decay_window=W`` to read the driver's past as a normalized weighted
    sum over W periods ``[T-N-W+1, T-N]`` instead of a single ``T-N``
    cell. Weights are determined by ``decay_kernel`` and normalize to
    sum to 1, so the blend-weight semantic is unchanged. ``decay=False``
    (default) preserves the single-cell read byte-for-byte.

      * ``geometric`` (default) — weights ∝ ``0.5^s`` for offset
        ``s = 0, 1, ..., W-1`` (most-recent first). Gives a half-life of
        one period — the most recent driver value dominates and
        contribution halves with each additional period back. Models the
        "marketing carryover" intuition where last week mattered most.
      * ``linear`` — weights ∝ ``W - s``, dropping linearly from
        ``W`` at the most-recent cell to ``1`` at the oldest. A flatter
        spread; appropriate when the contribution should fade more
        gradually.

    Cold-start NaN handling under decay: NaN cells in the buffer slice
    are dropped and the surviving weights renormalised. If every cell
    in the slice is NaN, the lag falls through to the unmodified
    current position — matching the discrete-lag fallback contract.
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
    # 0.6-M9b: opt-in adstock-style decay over a window.
    decay: bool = False
    decay_window: Optional[int] = Field(default=None, ge=1, le=10_000)
    decay_kernel: Literal["geometric", "linear"] = "geometric"

    @model_validator(mode="after")
    def _decay_window_paired_with_decay(self) -> "CausalLag":
        if self.decay:
            if self.decay_window is None:
                raise ValueError(
                    "causal_lag.decay=True requires `decay_window` to be "
                    "set (the number of periods over which the driver's "
                    "effect spreads)"
                )
        else:
            if self.decay_window is not None:
                raise ValueError(
                    f"causal_lag.decay_window={self.decay_window} is set "
                    f"but decay=False; either set decay=True or remove "
                    f"decay_window"
                )
        return self


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
                f"curve segment start_pct ({self.start_pct}) must be < end_pct ({self.end_pct})"
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
    # 0.6-M8c: treatment / control assignment for A/B test datasets.
    #
    # Three independent fields (all defaults reproduce pre-M8c output
    # byte-for-byte):
    #
    #   * ``treatment_group`` — optional string label. Used by the manifest
    #     for per-cohort grouping (e.g. ``"treatment"`` / ``"control"``).
    #     Plotsim treats it as opaque metadata; the engine never branches
    #     on the label string.
    #   * ``treatment_lift_log_odds`` — optional float. The known effect
    #     size in log-odds units (logit space). When set, every metric's
    #     pre-polarity effective position is shifted by this amount via
    #     ``sigmoid(logit(p) + lift)`` for periods ``>= treatment_start_period``.
    #     ``None`` = no shift (the control-arm contract).
    #   * ``treatment_start_period`` — absolute period index at which the
    #     shift kicks in. ``0`` (default) = treatment from the entity's
    #     first active period. Setting it ``> entity.start_period`` carves
    #     out a baseline window where treatment and control entities share
    #     identical metric distributions (the AC for "pre-treatment baseline
    #     is identical across groups").
    #
    # The label is decoupled from the lift so a "control" entity can carry
    # a label without applying a shift, AND so the user can opt out of
    # labelling for debug runs while still applying lift. Cross-field
    # validation (``_treatment_start_period_within_window``) enforces
    # ``treatment_start_period < n_periods`` when lift is set, otherwise
    # the shift would never apply.
    treatment_group: Optional[str] = None
    treatment_lift_log_odds: Optional[float] = None
    treatment_start_period: int = Field(default=0, ge=0)


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
            raise ValueError(f"scd_type2.trigger_metric {v!r} must be 'table_name.metric_name'")
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
                    f"scd_type2.thresholds values must lie in the open interval (0, 1); got {t}"
                )
        for prev, curr in zip(v, v[1:]):
            if curr <= prev:
                raise ValueError(f"scd_type2.thresholds must be strictly increasing; got {list(v)}")
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
    # When present, this column emits trajectory- and archetype-driven
    # text. Must be paired with ``source: "narrative:<key>"`` (validated
    # by ``_narrative_pairing``); archetype coverage and fact-only
    # placement are enforced cross-model in
    # ``plotsim.validation.validate_narrative_columns``.
    narrative: Optional["NarrativeConfig"] = None
    # 0.6-M14c: nested column type — struct or array. Both fields are
    # paired with ``source: "nested"`` and ``dtype: "struct"`` /
    # ``dtype: "array"`` respectively (validated by ``_nested_pairing``).
    # ``nested_schema`` maps struct field names → primitive type words
    # (``int`` / ``float`` / ``string`` / ``boolean``); the cell builder
    # generates a Python dict of those typed values per row.
    # ``array_element_type`` + ``array_length`` describe an array of
    # primitive values; the cell builder generates a Python list of
    # ``array_length`` values of ``array_element_type`` per row.
    # Output writer: Parquet uses native nested schema (pyarrow); CSV
    # serializes via ``json.dumps``. V1 supports one level of nesting only.
    nested_schema: Optional[dict[str, str]] = None
    array_element_type: Optional[str] = None
    array_length: Optional[int] = Field(default=None, ge=1, le=100)

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
    def _narrative_pairing(self) -> "Column":
        """Source ``narrative:<key>`` and the ``narrative`` config must be paired.

        Mirrors ``_scd_pairing`` / ``_pool_pairing``: either both are
        present (the column emits trajectory- and archetype-driven text)
        or both are absent. A column with ``narrative:`` source but no
        config block has no template / lexicons to draw from; a column
        with a config block but a non-``narrative:`` source carries
        silently-ignored data. Both reject at load.

        Cross-model checks — that ``lexicons`` keys cover the config's
        archetypes and the column's table is a fact table — happen in
        ``plotsim.validation.validate_narrative_columns`` because they
        need the full ``PlotsimConfig`` context, not just the column.
        """
        is_narrative_source = self.source.startswith("narrative:")
        has_narrative_cfg = self.narrative is not None
        if is_narrative_source and not has_narrative_cfg:
            raise ValueError(
                f"column {self.name!r} declares source {self.source!r} but "
                f"no 'narrative' config block; add 'narrative: {{template, "
                f"lexicons, bands}}' or change the source"
            )
        if has_narrative_cfg and not is_narrative_source:
            raise ValueError(
                f"column {self.name!r} has a narrative config block but "
                f"source {self.source!r}; the narrative template replaces "
                f"the column value, so set source to 'narrative:<key>' or "
                f"remove the narrative block"
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
                        f"column {self.name!r} value_pool has an empty entity-name key"
                    )
                if not values:
                    raise ValueError(
                        f"column {self.name!r} value_pool for entity "
                        f"{entity_name!r} is empty; provide at least one "
                        f"value to sample from"
                    )
                if len(values) > 1000:
                    raise ValueError(
                        f"column {self.name!r} value_pool for entity "
                        f"{entity_name!r} has {len(values)} entries; the "
                        f"per-entity pool is capped at 1000 to keep "
                        f"config-load memory and per-row draw bounded"
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

    @model_validator(mode="after")
    def _range_dtype(self) -> "Column":
        """0.6-M19 Fix 2: ``range:<min>:<max>`` source requires
        ``dtype: int`` or ``dtype: float`` — string / id / date /
        boolean / struct / array columns have no meaningful uniform
        draw between numeric bounds. Fail at load rather than emit
        coerced nonsense at generation time.
        """
        if not self.source.startswith("range:"):
            return self
        if self.dtype not in ("int", "float"):
            raise ValueError(
                f"column {self.name!r} declares source {self.source!r} but "
                f"dtype={self.dtype!r}; range sources require dtype 'int' or "
                f"'float' so the per-row draw produces the right cell type"
            )
        return self

    @model_validator(mode="after")
    def _nested_pairing(self) -> "Column":
        """Source ``nested`` and ``dtype: struct|array`` must be paired.

        Three pairing rules:
          * ``source: "nested"`` requires ``dtype: "struct"`` or
            ``dtype: "array"`` — anything else has no nested-cell semantics.
          * ``dtype: "struct"`` requires ``source: "nested"`` plus a
            ``nested_schema`` mapping field names → primitive types.
            ``array_element_type`` and ``array_length`` are not meaningful
            for struct columns and are rejected.
          * ``dtype: "array"`` requires ``source: "nested"`` plus an
            ``array_element_type`` (and an ``array_length``, defaulted to
            3 if omitted). ``nested_schema`` is rejected for arrays.

        Primitive types inside ``nested_schema`` and ``array_element_type``
        are restricted to ``int`` / ``float`` / ``string`` / ``boolean``.
        Nested-of-nested (struct-of-struct, array-of-struct, ...) is
        rejected — V1 supports one level of nesting only.
        """
        from plotsim._types import is_nested_primitive

        is_nested_source = self.source == "nested"
        is_struct_dtype = self.dtype == "struct"
        is_array_dtype = self.dtype == "array"

        # Source vs dtype consistency.
        if is_nested_source and not (is_struct_dtype or is_array_dtype):
            raise ValueError(
                f"column {self.name!r} declares source 'nested' but "
                f"dtype={self.dtype!r}; nested cells require dtype "
                f"'struct' or 'array'"
            )
        if (is_struct_dtype or is_array_dtype) and not is_nested_source:
            raise ValueError(
                f"column {self.name!r} has dtype={self.dtype!r} but "
                f"source={self.source!r}; nested dtypes require "
                f"source 'nested' (the cell builder reads nested_schema "
                f"or array_element_type to materialise the value)"
            )

        # Struct-specific shape.
        if is_struct_dtype:
            if not self.nested_schema:
                raise ValueError(
                    f"column {self.name!r} has dtype 'struct' but no "
                    f"nested_schema; declare 'nested_schema: {{<field>: "
                    f"<int|float|string|boolean>, ...}}'"
                )
            if len(self.nested_schema) > 20:
                raise ValueError(
                    f"column {self.name!r} nested_schema has "
                    f"{len(self.nested_schema)} fields; struct columns are "
                    f"capped at 20 fields to keep per-row materialization "
                    f"bounded"
                )
            if self.array_element_type is not None or self.array_length is not None:
                raise ValueError(
                    f"column {self.name!r} has dtype 'struct' but also "
                    f"declares array_element_type / array_length; those "
                    f"fields are array-only"
                )
            for field_name, field_type in self.nested_schema.items():
                if not field_name:
                    raise ValueError(f"column {self.name!r} nested_schema has an empty field name")
                if not is_nested_primitive(field_type):
                    raise ValueError(
                        f"column {self.name!r} nested_schema field "
                        f"{field_name!r} has type {field_type!r}; valid "
                        f"primitive types are int / float / string / boolean "
                        f"(nested-of-nested not supported in V1)"
                    )

        # Array-specific shape.
        if is_array_dtype:
            if self.array_element_type is None:
                raise ValueError(
                    f"column {self.name!r} has dtype 'array' but no "
                    f"array_element_type; declare 'array_element_type: "
                    f"<int|float|string|boolean>'"
                )
            if not is_nested_primitive(self.array_element_type):
                raise ValueError(
                    f"column {self.name!r} array_element_type "
                    f"{self.array_element_type!r} is not a valid primitive; "
                    f"valid: int / float / string / boolean"
                )
            if self.nested_schema is not None:
                raise ValueError(
                    f"column {self.name!r} has dtype 'array' but also "
                    f"declares nested_schema; that field is struct-only"
                )

        # Reject the nested config fields outside their dtype scope.
        if not is_struct_dtype and self.nested_schema is not None:
            raise ValueError(
                f"column {self.name!r} has nested_schema but dtype="
                f"{self.dtype!r}; nested_schema is only valid on dtype "
                f"'struct'"
            )
        if not is_array_dtype and (
            self.array_element_type is not None or self.array_length is not None
        ):
            raise ValueError(
                f"column {self.name!r} has array_element_type/array_length "
                f"but dtype={self.dtype!r}; those fields are only valid on "
                f"dtype 'array'"
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


_TABLE_TYPE_PREFIXES: tuple[str, ...] = ("dim_", "fct_", "evt_")


def _strip_table_type_prefix(name: str) -> str:
    """0.6-M19 Fix 8: drop the ``dim_`` / ``fct_`` / ``evt_`` prefix
    so PK-prefix derivation works off the semantic name (``orders``
    rather than ``fct_orders``)."""
    for prefix in _TABLE_TYPE_PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def _table_uses_sequential_pk(tbl: "Table") -> bool:
    """0.6-M19 Fix 8: whether a table's PK column gets a sequential
    ``<prefix>-NNNN`` value at generation time.

    Excludes:

      * ``dim_date`` — calendar-derived ``date_key`` values, not
        ``_make_ids``.
      * per_entity_per_period and per_period facts — composite /
        column-name surrogate PKs (``date_key``+``entity_id`` or
        ``<col>-<period>-<entity>``).
      * Bridge tables — composite from the two connected entity FKs.

    Includes every other dim, every event, and variable-grain /
    per_parent_row facts (the M18 child-grain).
    """
    if tbl.name == "dim_date":
        return False
    if tbl.type == "dim":
        return True
    if tbl.type == "event":
        return True
    if tbl.type == "fact":
        return tbl.grain in ("variable", "per_parent_row")
    return False


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
    # 0.6-M9c: opt-in fact-side CDC. When True, the fact picks up three
    # audit columns at generation time: ``_inserted_at``, ``_updated_at``
    # (both ISO period strings derived from each row's date_key via
    # ``dim_date``), and ``_op`` (``"I"`` for the initial insert,
    # ``"U"`` for rows mutated by a column-level quality issue
    # post-generation). Only valid on fact tables — rejected at load
    # on dim/event/bridge by ``_cdc_only_on_fact``. Default False
    # preserves pre-M9c output byte-for-byte.
    cdc: bool = False
    # 0.6-M14b: opt-in log-file companion writer for event tables. When
    # ``log_format`` is set on an event table, ``write_tables`` formats
    # each event row through the template (Python ``str.format`` against
    # the row's column dict) and writes one ``.log`` file per event
    # table alongside the CSV/Parquet. Placeholder names must match
    # column names on the table (e.g.
    # ``"{event_ts} [INFO] {company_id} login {event_id}"``).
    # ``log_filename`` is the on-disk name (default
    # ``<table_name>.log``). Both fields are valid only on event
    # tables — rejected at load on fact/dim/bridge by
    # ``_log_format_only_on_event``. Default ``None`` preserves
    # pre-M14b output byte-for-byte.
    log_format: Optional[str] = None
    log_filename: Optional[str] = None
    # 0.6-M18: parent/child fact grain. ``parent_table`` names the parent
    # fact for a ``per_parent_row``-grain child; ``children_per_row`` is
    # the inclusive ``(min, max)`` fan-out range drawn per parent row.
    # Both required when ``grain == "per_parent_row"`` and rejected
    # otherwise (validators below). Default ``None`` keeps every existing
    # config byte-identical — no validator fires unless a child is
    # declared.
    parent_table: Optional[str] = None
    children_per_row: Optional[tuple[int, int]] = None
    # 0.6-M19 Fix 8: explicit override for the per-row sequential PK
    # prefix. When ``None`` (default), the engine derives the prefix
    # from the table name's first character after stripping the
    # ``dim_`` / ``fct_`` / ``evt_`` type prefix. If two tables would
    # otherwise share the same first character (e.g. ``fct_orders``
    # and ``fct_order_items`` both → ``o``), ``PlotsimConfig``'s
    # ``_resolve_pk_prefixes`` validator auto-promotes both to their
    # full stripped names (``orders`` / ``order_items``) so the
    # emitted PKs are distinguishable. Set ``pk_prefix`` explicitly
    # to pin a custom value — useful when the auto-derived name is
    # long or you want SQL-friendly short codes. Validated as a short
    # alphanumeric identifier (start with a letter, 1–12 chars).
    pk_prefix: Optional[str] = None

    @property
    def primary_key_cols(self) -> list[str]:
        """Return the PK as a list, whether declared as str or list[str]."""
        return [self.primary_key] if isinstance(self.primary_key, str) else list(self.primary_key)

    _name_is_identifier = field_validator("name")(_identifier_field_validator("table name"))

    @field_validator("pk_prefix")
    @classmethod
    def _pk_prefix_format(cls, v: Optional[str]) -> Optional[str]:
        """0.6-M19 Fix 8: validate explicit ``pk_prefix`` is a short
        alphanumeric token. Leading letter avoids ID-as-number
        ambiguity in CSV/SQL; the 12-char cap keeps emitted PKs
        readable (``orders-0001`` is fine; ``some_really_long_thing-
        0001`` is not).
        """
        if v is None:
            return v
        if not re.match(r"^[a-zA-Z][a-zA-Z0-9_]{0,11}$", v):
            raise ValueError(
                f"pk_prefix {v!r} must start with a letter and contain "
                f"only letters / digits / underscores (1-12 characters)"
            )
        return v

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
    def _row_count_source_only_on_event_or_variable_fact(self) -> "Table":
        # 0.6-M18: row_count_source is now also valid on variable-grain
        # fact tables (the parent of a per_parent_row child). Reject on
        # any other shape so a stray field on a per_entity_per_period
        # fact fails loud instead of silently no-op'ing.
        if self.row_count_source is None:
            return self
        if self.type == "event":
            return self
        if self.type == "fact" and self.grain == "variable":
            return self
        raise ValueError(
            f"table {self.name!r} has row_count_source but type="
            f"{self.type!r} grain={self.grain!r}; row_count_source is "
            f"only allowed on event tables and on variable-grain fact "
            f"tables (parent fact of a per_parent_row child)"
        )

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

    @model_validator(mode="after")
    def _cdc_only_on_fact(self) -> "Table":
        # 0.6-M9c: ``cdc=True`` is fact-side audit columns. Dims have
        # SCD Type 2 / Type 1 for analogous semantics; events and
        # bridges are not in scope. Reject at load to flag the
        # mis-configuration with a clear message.
        if self.cdc and self.type != "fact":
            raise ValueError(
                f"table {self.name!r} has cdc=True but type={self.type!r}; "
                f"`cdc` is only valid on fact tables (dim tables use "
                f"SCD Type 2 for an analogous audit trail; event and "
                f"bridge tables are out of scope)"
            )
        return self

    @model_validator(mode="after")
    def _log_format_only_on_event(self) -> "Table":
        # 0.6-M14b: ``log_format`` / ``log_filename`` are event-side
        # only — fact/dim/bridge tables don't have the "one row per
        # discrete event" shape that a log file expresses. Reject at
        # load. Either field set on a non-event table is the
        # mis-configuration; mention both in the message because
        # users frequently set one without the other.
        if self.type != "event" and (self.log_format is not None or self.log_filename is not None):
            raise ValueError(
                f"table {self.name!r} has log_format/log_filename set "
                f"but type={self.type!r}; both fields are only valid on "
                f"event tables (each fact/dim row is a state snapshot, "
                f"not a discrete event)"
            )
        # ``log_filename`` without ``log_format`` is meaningless —
        # there's nothing to write. Reject so the user isn't surprised
        # when no log file appears.
        if self.log_filename is not None and self.log_format is None:
            raise ValueError(
                f"table {self.name!r} sets log_filename={self.log_filename!r} "
                f"but log_format is None; the filename alone produces no "
                f"output. Set log_format (the template string) or drop "
                f"log_filename."
            )
        return self

    @model_validator(mode="after")
    def _per_parent_row_field_pairing(self) -> "Table":
        # 0.6-M18: ``parent_table`` and ``children_per_row`` are required
        # on per_parent_row tables and rejected everywhere else. The
        # paired-fields discipline matches SCD / pool / narrative — the
        # grain selects the dispatch and the fields parameterize it; one
        # without the other is a mis-configuration we catch at load.
        is_child = self.grain == "per_parent_row"
        has_parent = self.parent_table is not None
        has_range = self.children_per_row is not None
        if is_child:
            if self.type != "fact":
                raise ValueError(
                    f"table {self.name!r} has grain='per_parent_row' but "
                    f"type={self.type!r}; per_parent_row is only valid on "
                    f"fact tables"
                )
            if not has_parent:
                raise ValueError(
                    f"table {self.name!r} has grain='per_parent_row' but "
                    f"parent_table is unset; declare the parent fact name "
                    f"(e.g. parent_table: 'fct_orders')"
                )
            if not has_range:
                raise ValueError(
                    f"table {self.name!r} has grain='per_parent_row' but "
                    f"children_per_row is unset; declare a (min, max) "
                    f"fan-out range (e.g. children_per_row: [1, 5])"
                )
            mn, mx = self.children_per_row  # type: ignore[misc]
            if mn < 1:
                raise ValueError(
                    f"table {self.name!r} children_per_row min={mn} must "
                    f"be >= 1 (a per_parent_row child fans out at least "
                    f"one row per parent row)"
                )
            if mx < mn:
                raise ValueError(
                    f"table {self.name!r} children_per_row max={mx} must be >= min={mn}"
                )
            if self.row_count_source is not None:
                raise ValueError(
                    f"table {self.name!r} has grain='per_parent_row' but "
                    f"row_count_source is set; child row count is driven "
                    f"by (parent rows × children_per_row), not by a "
                    f"row_count_source. Drop the field."
                )
        else:
            if has_parent:
                raise ValueError(
                    f"table {self.name!r} sets parent_table="
                    f"{self.parent_table!r} but grain={self.grain!r}; "
                    f"parent_table is only valid on grain='per_parent_row'"
                )
            if has_range:
                raise ValueError(
                    f"table {self.name!r} sets children_per_row but "
                    f"grain={self.grain!r}; children_per_row is only "
                    f"valid on grain='per_parent_row'"
                )
        return self

    @model_validator(mode="after")
    def _variable_grain_fact_no_metric_columns(self) -> "Table":
        # 0.6-M18: variable-grain fact tables are designed for parents
        # of per_parent_row children (orders/header records). Metric
        # columns at this grain would mean "one metric value per
        # discrete instance" which doesn't compose cleanly with the
        # trajectory-first invariant (multiple rows per (entity, period)
        # share one trajectory position). M18 forbids ``metric:`` sources
        # on variable-grain facts; per-instance metrics belong on the
        # child fact table (one row per discrete instance is exactly
        # what per_parent_row delivers). Deferred to a future mission
        # if needed.
        if self.type != "fact" or self.grain != "variable":
            return self
        for col in self.columns:
            try:
                parsed = parse_source(col.source)
            except ValueError:
                # Bad source format — let the column-level validator
                # surface the parse error. This validator only checks
                # the variable-fact / metric pairing.
                continue
            if isinstance(parsed, MetricSource):
                raise ValueError(
                    f"variable-grain fact column {col.name!r} on table "
                    f"{self.name!r} has source {col.source!r}; "
                    f"metric: sources are not supported on variable-grain "
                    f"fact tables in 0.6 (per-instance metric semantics "
                    f"are ambiguous when multiple rows share one "
                    f"trajectory position). Move the metric to a "
                    f"per_parent_row child table, or use generated:/"
                    f"static:/derived: on the parent."
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

    Six issue types are supported, each producing a distinct corruption
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
      * ``volume_anomaly`` — at ``target_period`` (or every period in
        ``target_periods``), either ``mode="spike"`` duplicates
        ``rate`` of the rows whose period matches, or ``mode="drop"``
        removes ``rate`` of them. Row-level like ``duplicate_rows``;
        ``target_columns`` must be ``["*"]`` (the corruption is not
        per-column). Manifest record uses the ``column="_rows"``
        sentinel.

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
        "volume_anomaly",
    ]
    target_table: str
    target_columns: list[str] = Field(min_length=1, max_length=100)
    rate: float = Field(ge=0.0, le=1.0)
    seed_offset: int = Field(default=0, ge=0, le=2**31 - 1)
    # 0.6-M9a: volume_anomaly extras. ``mode`` picks spike (duplicate
    # rows whose period matches) vs drop (remove them). Either
    # ``target_period`` (single int) or ``target_periods`` (list) names
    # the period(s) to corrupt — exactly one of the two must be set
    # when ``type == "volume_anomaly"``. All three fields default to
    # None and are required-only on volume_anomaly; a load-time
    # validator enforces the conditional contract. Period values are
    # 0-based indices into ``dim_date`` — the handler maps them to
    # date_key at apply time so the period spec stays granularity-
    # agnostic.
    mode: Optional[Literal["spike", "drop"]] = None
    target_period: Optional[int] = Field(default=None, ge=0)
    target_periods: Optional[list[int]] = Field(default=None, max_length=10_000)

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

    @field_validator("target_periods")
    @classmethod
    def _target_periods_non_negative(cls, v: Optional[list[int]]) -> Optional[list[int]]:
        if v is None:
            return v
        if len(v) == 0:
            raise ValueError(
                "quality_issues.target_periods must be a non-empty list "
                "when set; omit the field or use target_period for the "
                "single-period case"
            )
        for entry in v:
            if not isinstance(entry, int) or entry < 0:
                raise ValueError(
                    f"quality_issues.target_periods entries must be "
                    f"non-negative integers (got {entry!r})"
                )
        return v

    @model_validator(mode="after")
    def _volume_anomaly_required_fields(self) -> "QualityIssue":
        is_va = self.type == "volume_anomaly"
        has_period = self.target_period is not None
        has_periods = self.target_periods is not None
        has_mode = self.mode is not None
        if is_va:
            if not has_mode:
                raise ValueError(
                    "quality_issues.type='volume_anomaly' requires "
                    "`mode` to be set to 'spike' or 'drop'"
                )
            if has_period == has_periods:
                # both set or both unset
                if has_period and has_periods:
                    raise ValueError(
                        "quality_issues.type='volume_anomaly' accepts "
                        "exactly one of `target_period` or "
                        "`target_periods`, not both"
                    )
                raise ValueError(
                    "quality_issues.type='volume_anomaly' requires "
                    "`target_period` (single int) or `target_periods` "
                    "(list of ints) — neither was set"
                )
        else:
            extras = []
            if has_mode:
                extras.append("mode")
            if has_period:
                extras.append("target_period")
            if has_periods:
                extras.append("target_periods")
            if extras:
                raise ValueError(
                    f"quality_issues fields {extras!r} are only valid "
                    f"when type='volume_anomaly'; got type={self.type!r}"
                )
        return self


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


class CorrelationPhase(_Frozen):
    """0.6-M11: a window over the time axis with its own correlation matrix.

    Phases declare per-window correlation pairs that override the baseline
    ``PlotsimConfig.correlations`` for periods inside ``[start_period,
    end_period]`` (both inclusive). Phases are non-overlapping (validated
    at config load) and global across all entities — every entity sees
    the same phase boundaries.

    Periods not covered by any phase fall back to the baseline
    ``correlations`` list. A config with ``correlation_phases`` set must
    also declare a non-empty ``correlations`` baseline (the baseline is
    required even if phases tile the window exhaustively, so a later
    edit to ``time_window`` cannot leave uncovered periods with no
    correlation set).

    Engine treatment per phase: each phase's ``correlations`` list runs
    through the same M120 trajectory-aware compensation (when
    ``compensate_correlations=True``) and M111 Higham nearest-PD
    projection that the baseline list goes through, independently. Each
    phase produces its own Cholesky factor; the engine resolves the
    active factor per period at sample time.
    """

    start_period: int = Field(ge=0)
    end_period: int = Field(ge=0)
    correlations: list[CorrelationPair] = Field(default_factory=list, max_length=1_225)

    @model_validator(mode="after")
    def _end_after_start(self) -> "CorrelationPhase":
        if self.end_period < self.start_period:
            raise ValueError(
                f"correlation_phases entry has end_period={self.end_period} "
                f"< start_period={self.start_period}; phases must satisfy "
                f"start_period <= end_period (both inclusive)"
            )
        return self


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

    ``format`` accepts ``"parquet"``, ``"jsonl"``, and ``"sql"`` in
    addition to the default ``"csv"``. CSV remains the default; configs
    that omit ``format`` (or set ``format: csv``) write ``.csv`` files.
    Parquet output is column-typed and typically 5-10x smaller on the
    bundled templates; the engine path is identical, only the on-disk
    encoding differs. JSONL (0.6-M16b) writes newline-delimited JSON for
    streaming-ingestion / schema-on-read consumers. SQL (0.6-M16c)
    writes a single ``data.sql`` file with dialect-aware DDL + batched
    INSERTs for database-exercise workflows (``psql < data.sql``,
    ``sqlite3 db.sqlite < data.sql``).

    Parquet writes go through ``pyarrow``, declared as the optional
    extra ``plotsim[parquet]``. When pyarrow is not installed and
    ``format: parquet`` is configured, ``write_tables`` raises an
    ``ImportError`` naming the install command — fail-fast at the
    write call rather than mid-iteration.

    JSONL writes use ``DataFrame.to_json(orient='records', lines=True,
    date_format='iso')`` so nested struct / array cells serialize as
    native JSON objects / arrays, NaN values become ``null``, and date
    columns land as ISO-8601 strings (rather than pandas' default
    epoch-ms milliseconds for ``orient='records'``).

    SQL writes emit one ``data.sql`` file containing dialect-specific
    DDL (``CREATE TABLE`` with PK + FK constraints) and batched
    ``INSERT`` statements (~100 rows per statement). Dimension tables
    appear before fact / event / bridge tables so FK targets exist when
    the dump is replayed top-to-bottom. ``sql_dialect`` selects between
    ``postgresql`` (default; ``"col"`` quoting, ``NUMERIC`` for floats),
    ``mysql`` (`` `col` `` quoting, ``DOUBLE``, ``VARCHAR(255)`` for id
    cols since MySQL forbids ``TEXT`` primary keys), and ``sqlite``
    (``"col"`` quoting, ``REAL`` for floats, ``INTEGER`` for booleans).
    Denormalized wide tables and holdout splits — when enabled —
    appear as additional ``CREATE TABLE`` + INSERT blocks AFTER the
    star schema, without FK constraints (their multi-dim shape doesn't
    fit the FK model). ``entity_features.enabled`` is rejected at load
    when ``format == "sql"`` (the per-entity feature DataFrame mixes
    aggregates across all metrics and doesn't compose cleanly into the
    single-file SQL dump).

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

    ``denormalized`` (0.6-M14a) opts into a wide-table companion
    write: for each fact table, ``write_tables`` left-joins every
    FK'd dim onto the fact and emits ``<fct_name>_wide.{csv|parquet}``
    alongside the normalized output. Off by default so existing
    output is byte-identical. SCD2 dims are filtered to current-
    state rows (``is_current == True``) before the join; SCD2 audit
    columns are excluded from the wide output. Dim columns are
    prefixed with the dim's table name plus ``__`` to avoid
    collisions; the dim-side join key is dropped post-join because
    it duplicates the fact's FK column.

    ``partition_by`` names a column to partition Parquet output on.
    When set (and ``format == "parquet"``), every table that has a
    column with this name is written as a Hive-style directory of
    Parquet files (``<output_dir>/<table_name>/<col>=<value>/...``)
    via ``pyarrow.parquet.write_to_dataset``. Tables without the
    column are written as a single Parquet file unchanged. Default
    ``None`` preserves the single-file-per-table layout. The
    streaming-Parquet row-group optimization (M121b) bypasses cleanly
    when ``partition_by`` is set — partitioning is the user-visible
    knob, streaming is an internal memory tactic. The cross-table
    column-type check on the named partition column runs in
    ``PlotsimConfig._validate_partition_column``: ``float`` /
    ``struct`` / ``array`` partition keys are rejected at load.
    """

    format: Literal["csv", "parquet", "jsonl", "sql"] = "csv"
    directory: str
    cell_budget: Optional[int] = Field(default=None, ge=0)
    denormalized: bool = False
    partition_by: Optional[str] = None
    sql_dialect: Literal["postgresql", "mysql", "sqlite"] = "postgresql"

    @model_validator(mode="after")
    def _partition_by_requires_parquet(self) -> "OutputConfig":
        if self.partition_by is not None and self.format != "parquet":
            raise ValueError(
                f"output.partition_by={self.partition_by!r} requires "
                f"output.format='parquet' (got {self.format!r}); "
                f"partitioning only applies to the columnar format"
            )
        return self

    @model_validator(mode="after")
    def _sql_dialect_requires_sql_format(self) -> "OutputConfig":
        # The default ``postgresql`` is allowed under any format (it
        # round-trips through ``dump_config`` regardless of whether the
        # SQL writer ever consumes it). An explicit ``mysql`` / ``sqlite``
        # paired with a non-sql format is a misconfiguration the user
        # should hear about at load rather than have silently ignored.
        if self.sql_dialect != "postgresql" and self.format != "sql":
            raise ValueError(
                f"output.sql_dialect={self.sql_dialect!r} requires "
                f"output.format='sql' (got {self.format!r}); the dialect "
                f"is only consumed when emitting the data.sql dump"
            )
        return self


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
                    f"stage {stage.name!r} is not terminal but has threshold_exit: null"
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


# 0.6-M13: multi-source / overlap mode. Each `SourceDeclaration` describes
# one upstream system the engine emits a divergent copy of the canonical
# per_entity dim for. The canonical `dim_<entity>` is unchanged; per-source
# emissions land as `dim_<entity>_<source>` with name, key-scheme, and
# attribute drift applied at the configured rates. Ground truth (entity →
# source-specific-id, list of drifted fields) is recorded in the manifest's
# `source_entity_mappings` section so an entity-resolution exercise has an
# answer key.
SourceKeyScheme = Literal["prefix_padded", "numeric", "uuid_short"]


class SourceDeclaration(_Frozen):
    """One upstream system in the multi-source / overlap layout.

    Fields:

      * ``name`` — SQL-safe identifier used as the per-source dim
        suffix (``dim_<entity>_<name>``) and the source label on
        manifest mapping records. Must be unique within
        ``MultiSourceConfig.sources``.

      * ``key_scheme`` — how this source represents entity IDs.
        ``prefix_padded`` mimics a CRM (``CUST-001``);
        ``numeric`` mimics a billing system (``1001``);
        ``uuid_short`` mimics a record-keeping system (``c3f9a``,
        5-char hex). The canonical ``dim_<entity>`` PK is preserved;
        the per-source dim's PK column is renamed to
        ``<entity_type>_id_<source>`` so a join on the canonical PK
        is impossible by construction — entity resolution must
        bridge through the manifest mapping or fuzzy-match on
        drifted attributes.

      * ``name_drift_rate`` — fraction of entities (per source) whose
        first ``generated:faker.{name,first_name,last_name,company}``
        column gets a name typo applied (adjacent-char swap, casing
        flip, or abbreviation). ``0.0`` = no drift (the same names
        as the canonical dim); ``1.0`` = every entity is drifted.

      * ``attribute_drift_rate`` — fraction of entities (per source)
        whose first non-PK, non-FK, non-name string column gets a
        deterministic conflicting value. Builds the per-entity
        attribute disagreement that record-linkage exercises learn
        to score over.

    Drift mechanics live in :mod:`plotsim.multi_source`. Per-source
    RNGs are derived by sequential ``integers`` draws on the
    dim-build RNG in declaration order, so toggling source order
    shifts each source's drift but stays deterministic under a
    fixed ``(seed, sources)`` pair.
    """

    name: str
    key_scheme: SourceKeyScheme = "prefix_padded"
    name_drift_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    attribute_drift_rate: float = Field(default=0.0, ge=0.0, le=1.0)

    _name_is_identifier = field_validator("name")(_identifier_field_validator("source name"))


class MultiSourceConfig(_Frozen):
    """Multi-source / overlap mode config block.

    Activates per-source dim emission when present on a
    ``PlotsimConfig``. ``sources`` must declare 2–5 named sources;
    1 source is degenerate (no overlap to resolve against) and >5
    moves out of the teaching range called out in the mission spec.

    Default-empty ``PlotsimConfig.multi_source = None`` means the
    feature is dormant — output is byte-identical to pre-0.6-M13.
    """

    sources: list[SourceDeclaration] = Field(min_length=2, max_length=5)

    @model_validator(mode="after")
    def _source_names_unique(self) -> "MultiSourceConfig":
        seen: set[str] = set()
        for src in self.sources:
            if src.name in seen:
                raise ValueError(
                    f"multi_source.sources: duplicate source name "
                    f"{src.name!r}; each declared source must be unique "
                    f"because the source name doubles as the per-source "
                    f"dim suffix and the manifest mapping label"
                )
            seen.add(src.name)
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
    # 0.6-M11: time-varying correlations. Each phase declares its own
    # CorrelationPair list active for the inclusive period window
    # ``[start_period, end_period]``. Empty default — configs without
    # phases keep the single-Cholesky behavior (byte-identical to
    # pre-M11). When non-empty, ``correlations`` is required as the
    # baseline for any period not covered by a phase (validated by
    # ``_correlation_phases_require_baseline``). ``max_length=64``
    # accommodates monthly-granularity exercises (e.g. one phase per
    # month over a multi-year window) without unbounded growth.
    correlation_phases: list[CorrelationPhase] = Field(default_factory=list, max_length=64)
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

    # 0.6-M13: multi-source / overlap mode. ``None`` (default) keeps output
    # byte-identical to pre-M13: the dim builder skips the multi-source pass
    # entirely. When set, the canonical ``dim_<entity>`` is still emitted;
    # each source declared under ``multi_source.sources`` produces an
    # additional ``dim_<entity>_<source>`` table with name typos, an
    # alternate ID scheme, and attribute conflicts applied at the configured
    # rates. The manifest's ``source_entity_mappings`` records the
    # (entity, source, source_entity_id, drifted_fields) tuples as the
    # ground-truth answer key for entity-resolution exercises.
    multi_source: Optional[MultiSourceConfig] = None

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

    # 0.6-M11: per-phase analogues of the three sibling stashes above.
    # Each is keyed by phase index (0..len(correlation_phases)-1). The
    # baseline matrix continues to use the non-phase attrs. ``None`` when
    # the run has no ``correlation_phases``; an empty dict is reserved
    # for "phases configured but every per-phase matrix already PD" /
    # "phases configured but compensation didn't run for any phase".
    # Same PrivateAttr rationale as the siblings: engine-derived,
    # excluded from ``model_dump`` / ``config_sha256``.
    _phase_correlation_adjustments: Optional[dict[int, list[dict]]] = PrivateAttr(default=None)
    _phase_correlation_compensations: Optional[dict[int, list[dict]]] = PrivateAttr(default=None)
    _phase_projected_correlation_matrices: Optional[dict[int, Any]] = PrivateAttr(default=None)

    # 0.6-M13: per-(entity, source, dim_table) ground-truth mapping records
    # produced by the dim builder when ``multi_source`` is set. ``None`` on
    # configs without multi-source (the default lane — keeps manifest's
    # ``source_entity_mappings`` empty). PrivateAttr because this is a per-run
    # side-effect of generation, not user input — round-tripping it through
    # ``model_dump`` would pollute the YAML and ``config_sha256``, same
    # reasoning as the sibling adjustment / compensation stashes above.
    _source_entity_mappings: Optional[list[dict]] = PrivateAttr(default=None)

    # 0.6-M19 Fix 8: cached map of ``table_name → resolved PK prefix``,
    # populated by ``_resolve_pk_prefixes`` at load. Engine call sites
    # read via ``pk_prefix_for(table_name)`` rather than re-deriving
    # the prefix inline (avoids the pre-fix collision pattern where
    # ``fct_orders`` and ``fct_order_items`` both rendered ``o-0001``).
    _pk_prefixes: dict[str, str] = PrivateAttr(default_factory=dict)

    @model_validator(mode="after")
    def _total_entity_size_within_limit(self) -> "PlotsimConfig":
        total = sum(e.size for e in self.entities)
        if total > 100_000:
            raise ValueError(
                f"Total entity count across all groups is {total:,}. Maximum is 100,000."
            )
        return self

    @model_validator(mode="after")
    def _resolve_pk_prefixes(self) -> "PlotsimConfig":
        """0.6-M19 Fix 8: compute the per-table PK prefix used by the
        engine's sequential-PK builders. Two paths:

        * **Explicit** — ``Table.pk_prefix`` set on the table model.
          Used verbatim.
        * **Auto-derive** — first character of the table name after
          stripping the ``dim_`` / ``fct_`` / ``evt_`` type prefix. If
          two tables would land on the same first character, both
          (the whole colliding group) promote to their full stripped
          names so the emitted PKs stay distinguishable.

        Cross-collision check runs after both paths resolve — if an
        explicit value lands on the same string as an auto-derived
        one (or two explicit values overlap), raise so the user picks.

        Only sequential-PK tables participate: per_entity_per_period
        and per_period facts use column-name surrogate PKs that don't
        share the prefix scheme; ``dim_date`` is calendar-derived;
        bridge tables compose their PK from two FK columns.
        """
        prefix_eligible_tables = [t for t in self.tables if _table_uses_sequential_pk(t)]
        explicit: dict[str, str] = {}
        auto_candidates: list["Table"] = []
        for tbl in prefix_eligible_tables:
            if tbl.pk_prefix is not None:
                explicit[tbl.name] = tbl.pk_prefix
            else:
                auto_candidates.append(tbl)

        # Group auto-derived candidates by first-character default.
        by_first_char: dict[str, list["Table"]] = {}
        for tbl in auto_candidates:
            stripped = _strip_table_type_prefix(tbl.name)
            if not stripped:
                raise ValueError(
                    f"table {tbl.name!r}: cannot derive a PK prefix from "
                    f"an empty stripped name; set ``pk_prefix`` explicitly"
                )
            by_first_char.setdefault(stripped[0], []).append(tbl)

        resolved: dict[str, str] = dict(explicit)
        for _first, group in by_first_char.items():
            if len(group) == 1:
                resolved[group[0].name] = _strip_table_type_prefix(group[0].name)[0]
            else:
                # Collision in the auto-derived bucket — promote every
                # colliding table to its full stripped name so the
                # emitted PKs disambiguate without user intervention.
                for tbl in group:
                    resolved[tbl.name] = _strip_table_type_prefix(tbl.name)

        # Final cross-collision detection. The auto promotion above is
        # safe in isolation (full stripped names of distinct tables
        # can't collide because table names are unique), but an
        # explicit override might collide with an auto-derived prefix
        # on another table — surface that here so the user picks.
        prefix_to_tables: dict[str, list[str]] = {}
        for tname, prefix in resolved.items():
            prefix_to_tables.setdefault(prefix, []).append(tname)
        for prefix, names in prefix_to_tables.items():
            if len(names) > 1:
                raise ValueError(
                    f"PK prefix {prefix!r} resolves to multiple tables "
                    f"{sorted(names)!r}; set ``Table.pk_prefix`` "
                    f"explicitly on at least one to disambiguate"
                )

        object.__setattr__(self, "_pk_prefixes", resolved)
        return self

    def pk_prefix_for(self, table_name: str) -> str:
        """Return the resolved sequential-PK prefix for ``table_name``.

        Populated by :meth:`_resolve_pk_prefixes` at config load. Falls
        back to the first character of the stripped table name when
        the table isn't in the resolved map (defensive — every dim /
        fact / event the validator considered should be present).
        """
        if table_name in self._pk_prefixes:
            return self._pk_prefixes[table_name]
        stripped = _strip_table_type_prefix(table_name)
        return stripped[0] if stripped else "x"

    @model_validator(mode="after")
    def _validate_partition_column(self) -> "PlotsimConfig":
        """0.6-M16a: cross-table partition-column existence and dtype check.

        When ``output.partition_by`` is set, at least one table must
        carry a matching column (otherwise the name is a typo and
        partitioning would silently no-op across the entire write).
        A column matches either by literal name OR — 0.6-M19 Fix 5 —
        by FK target: a column whose source is
        ``fk:<dim>.<partition_by>`` resolves to ``partition_by`` even
        though its local name differs (e.g. ``order_date`` on
        ``fct_orders`` resolves the ``partition_by: date_key``
        declaration via its FK to ``dim_date.date_key``).

        Every matching column must use a partition-eligible dtype —
        ``float`` / ``struct`` / ``array`` are rejected because
        Hive-style equality partitioning is ill-defined for them. The
        remaining dtypes (``int`` / ``string`` / ``date`` /
        ``boolean`` / ``id``) all work.

        Precedence: literal name match runs first; FK-target
        resolution is the fallback when no table declares the literal
        column. Mixing literal and FK matches across tables is fine —
        each table independently resolves its own column at write time
        in :func:`plotsim.output.write_single_table`.
        """
        partition_by = self.output.partition_by
        if partition_by is None:
            return self
        invalid_dtypes = {"float", "struct", "array"}
        matched: list[tuple[Table, Column]] = []
        for tbl in self.tables:
            for col in tbl.columns:
                if col.name == partition_by:
                    matched.append((tbl, col))
        if not matched:
            for tbl in self.tables:
                for col in tbl.columns:
                    parsed = parse_source(col.source)
                    if isinstance(parsed, FKSource) and parsed.column == partition_by:
                        matched.append((tbl, col))
        if not matched:
            table_names = ", ".join(t.name for t in self.tables)
            raise ValueError(
                f"output.partition_by={partition_by!r} does not match any "
                f"column on any declared table ({table_names}) — neither "
                f"as a literal column name nor as an FK target column; "
                f"check for typos or remove the field to disable "
                f"partitioning"
            )
        for tbl, col in matched:
            if col.dtype in invalid_dtypes:
                raise ValueError(
                    f"output.partition_by={partition_by!r} resolves to "
                    f"column {tbl.name}.{col.name} with dtype "
                    f"{col.dtype!r}, which is not a valid partition "
                    f"key type; use one of int / string / date / "
                    f"boolean / id"
                )
        return self

    @model_validator(mode="after")
    def _multi_source_requires_per_entity_dim(self) -> "PlotsimConfig":
        """0.6-M13: gate ``multi_source`` on the presence of a per_entity dim.

        The per-source emission pass copies each per_entity dim and applies
        drift. Without at least one per_entity dim there's nothing to
        overlay — surface a precise error at load rather than letting the
        dim orchestrator silently emit zero per-source tables.

        Also reject source names that would generate a ``dim_<entity>_<source>``
        table name colliding with an existing table — the dim writer would
        overwrite one file with the other.
        """
        if self.multi_source is None:
            return self
        per_entity_dims = [t for t in self.tables if t.type == "dim" and t.grain == "per_entity"]
        if not per_entity_dims:
            raise ValueError(
                "multi_source is set but no per_entity dim tables are "
                "declared; multi-source emission overlays drift on top of "
                "an existing per_entity dim, so at least one is required"
            )
        table_names = {t.name for t in self.tables}
        for src in self.multi_source.sources:
            for dim in per_entity_dims:
                emitted = f"{dim.name}_{src.name}"
                if emitted in table_names:
                    raise ValueError(
                        f"multi_source source {src.name!r} would emit dim "
                        f"table {emitted!r}, which collides with an existing "
                        f"table in config.tables; rename the source"
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
        # 0.6-M18: estimate per-table row counts for variable-grain
        # tables (event tables + variable-grain fact tables, both
        # routed through the proportional builder). Stash per-table
        # so the per_parent_row child fan-out can multiply against
        # the parent's row estimate.
        variable_rows_by_table: dict[str, int] = {}
        for tbl in self.tables:
            if tbl.row_count_source is None:
                continue
            is_proportional_owner = tbl.type == "event" or (
                tbl.type == "fact" and tbl.grain == "variable"
            )
            if not is_proportional_owner:
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
            tbl_rows = int(n_entities * n_periods * v_max * parsed_rc.scale)
            variable_rows_by_table[tbl.name] = tbl_rows
            event_rows_upper += tbl_rows

        # 0.6-M18: per_parent_row children fan out from parent rows by
        # the configured (min, max) range. Upper-bound estimate uses
        # the parent's row estimate × children_per_row_max. Parent
        # row estimate:
        #   - parent grain=per_entity_per_period → n_entities × n_periods
        #   - parent grain=variable → variable_rows_by_table[parent]
        # Unknown parent (typo or unbuilt estimate) contributes 0 — the
        # cross-reference validator catches typos at load before this
        # field is used downstream.
        child_rows_upper = 0
        tables_by_name = {t.name: t for t in self.tables}
        for tbl in self.tables:
            if tbl.grain != "per_parent_row":
                continue
            if tbl.parent_table is None or tbl.children_per_row is None:
                continue
            parent_tbl = tables_by_name.get(tbl.parent_table)
            if parent_tbl is None:
                continue
            if parent_tbl.grain == "per_entity_per_period":
                parent_rows = n_entities * n_periods
            elif parent_tbl.grain == "variable":
                parent_rows = variable_rows_by_table.get(parent_tbl.name, 0)
            else:
                parent_rows = 0
            _, mx = tbl.children_per_row
            child_rows_upper += int(parent_rows * mx)
        event_rows_upper += child_rows_upper

        # Estimate row-count growth from post-generation quality
        # injection. Two issue types inflate row counts:
        #   * ``duplicate_rows`` — inserts ``rate`` of the target
        #     table's existing rows as exact copies.
        #   * ``volume_anomaly`` with ``mode='spike'`` — duplicates
        #     ``rate`` of the rows whose period matches each entry in
        #     ``target_period`` / ``target_periods``.
        # Other issue types either replace cells (null_injection,
        # type_mismatch, schema_drift), append a column (late_arrival),
        # or remove rows (volume_anomaly drop) — none of those grow
        # the cell count, so they're not in the estimate.
        tables_by_name_for_quality = {t.name: t for t in self.tables}

        def _table_row_estimate(name: str) -> int:
            tbl = tables_by_name_for_quality.get(name)
            if tbl is None:
                return cell_count  # conservative — caught by other validators
            if tbl.type == "fact" and tbl.grain == "per_entity_per_period":
                return cell_count
            if tbl.name in variable_rows_by_table:
                return variable_rows_by_table[tbl.name]
            return cell_count

        quality_extra_rows = 0
        for issue in self.quality.quality_issues:
            target_rows = _table_row_estimate(issue.target_table)
            if issue.type == "duplicate_rows":
                quality_extra_rows += int(target_rows * issue.rate)
            elif issue.type == "volume_anomaly" and issue.mode == "spike":
                if issue.target_period is not None:
                    n_target_periods = 1
                else:
                    n_target_periods = len(issue.target_periods or ())
                rows_per_period = target_rows // max(n_periods, 1)
                quality_extra_rows += int(rows_per_period * n_target_periods * issue.rate)

        post_quality_cells = cell_count + quality_extra_rows

        summary = (
            f"Config summary: {n_entities:,} entities × {n_periods:,} periods "
            f"= {cell_count:,} cells, {len(self.metrics)} metrics, "
            f"{len(self.tables)} tables. Estimated peak memory: ~{peak_mb:.0f} MB."
        )
        if event_rows_upper > 0:
            summary += f" Expected event rows (upper bound): ~{event_rows_upper:,}."
        if quality_extra_rows > 0:
            summary += (
                f" Quality injection adds ~{quality_extra_rows:,} rows "
                f"(post-injection cells: ~{post_quality_cells:,})."
            )
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

        if (
            soft_budget > 0
            and post_quality_cells > soft_budget
            and cell_count <= soft_budget
            and not allow_large
        ):
            raise ValueError(
                f"Config produces {cell_count:,} cells before quality "
                f"injection but the configured quality_issues grow the "
                f"estimate to ~{post_quality_cells:,} cells, which exceeds "
                f"the soft budget of {soft_budget:,}. Lower the "
                f"`rate` on any `duplicate_rows` / `volume_anomaly` issue, "
                f"raise the budget in your config (output.cell_budget: N, "
                f"or 0 to disable), set PLOTSIM_CELL_BUDGET=N in the "
                f"environment, or pass --allow-large-dataset on the CLI / "
                f"set PLOTSIM_ALLOW_LARGE_DATASET=1."
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
            validate_parent_child_facts,
            validate_stages,
            validate_table_schema,
        )

        for check in (
            validate_names,
            validate_archetype_refs,
            validate_table_schema,
            validate_parent_child_facts,
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
        # 0.6-M16c: entity_features doesn't compose into the single-file
        # SQL dump — its per-entity wide-table shape mixes aggregates
        # across all metrics and breaks the dim/fact star schema that
        # the SQL writer ships. Operator-stated scope decision at M16c
        # kickoff: reject the combination at load rather than silently
        # drop the file.
        if self.output.format == "sql":
            raise ValueError(
                "entity_features.enabled=True is not supported with "
                "output.format='sql'. The per-entity feature DataFrame "
                "doesn't compose cleanly into the single data.sql file. "
                "Use output.format=csv/parquet/jsonl, or disable "
                "entity_features for the SQL run."
            )
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
    def _treatment_assignments_gate(self) -> "PlotsimConfig":
        """0.6-M8c: load-time gate for treatment / control assignments.

        Catches three malformed shapes: ``treatment_start_period``
        outside ``[entity.start_period, n_periods)`` and non-finite
        ``treatment_lift_log_odds``. See
        ``plotsim.validation.validate_treatment_assignments`` for the
        full rule list. Same model_validator pattern as
        ``_cold_start_active_periods_gate``.
        """
        # Local import: plotsim.validation imports from plotsim.config,
        # so a module-level import would create a cycle.
        from plotsim.validation import validate_treatment_assignments

        errors = validate_treatment_assignments(self)
        if errors:
            raise ValueError(errors[0])
        return self

    @model_validator(mode="after")
    def _narrative_gates(self) -> "PlotsimConfig":
        """Load-time gates for ``NarrativeSource`` columns.

        Cross-model: the per-Column structural pairing lives on
        ``Column._narrative_pairing``; this gate enforces the rules that
        only make sense with the full config in hand — that the column's
        table is fact-typed at the per-entity-per-period grain, and that
        every archetype name in ``narrative.lexicons`` resolves to a
        declared archetype. Same split as ``_value_pool_gates``.
        """
        from plotsim.validation import validate_narrative_columns

        errors = validate_narrative_columns(self)
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
    def _correlation_phases_require_baseline(self) -> "PlotsimConfig":
        """0.6-M11: when ``correlation_phases`` is set, ``correlations`` is required.

        The baseline ``correlations`` list serves as the active correlation
        set in any period not covered by a phase. Even if the configured
        phases tile the time window exhaustively today, a later edit to
        ``time_window`` could leave uncovered periods with no correlation
        set — making the engine's per-period resolver behave inconsistently
        across config edits. Requiring an explicit baseline keeps the
        fallback contract structural rather than emergent.
        """
        if self.correlation_phases and not self.correlations:
            raise ValueError(
                "correlation_phases is set but `correlations` (the baseline "
                "list) is empty. The baseline serves as the active correlation "
                "set in any period not covered by a phase; declare at least "
                "one baseline pair even if your phases tile the window "
                "exhaustively. See docs: user-guide/metrics-and-connections.md"
            )
        return self

    @model_validator(mode="after")
    def _correlation_phases_within_window(self) -> "PlotsimConfig":
        """0.6-M11: every phase's period range fits inside the time window.

        ``end_period`` is inclusive; the upper bound is
        ``time_window.period_count() - 1``. A phase whose window extends
        past the last period is rejected (rather than silently clipped)
        so the user catches off-by-one errors at config load.
        """
        if not self.correlation_phases:
            return self
        last = self.time_window.period_count() - 1
        for idx, phase in enumerate(self.correlation_phases):
            if phase.end_period > last:
                raise ValueError(
                    f"correlation_phases[{idx}] has end_period="
                    f"{phase.end_period} but time_window has only "
                    f"{last + 1} periods (max valid period_index={last}). "
                    f"Adjust end_period or extend time_window."
                )
        return self

    @model_validator(mode="after")
    def _correlation_phases_no_overlap(self) -> "PlotsimConfig":
        """0.6-M11: phase windows must not overlap.

        Overlapping phases would make the per-period Cholesky lookup
        ambiguous (which phase wins at the overlapping period?). Rather
        than picking an implicit precedence rule, reject the config so
        the user explicitly resolves the ambiguity by editing the
        windows. Phases may abut (one ends at ``t``, the next starts at
        ``t+1``) without overlap; phases may be in any declaration order.
        """
        if len(self.correlation_phases) < 2:
            return self
        # Sort by start_period to compare consecutive windows after the sort.
        sorted_phases = sorted(
            enumerate(self.correlation_phases),
            key=lambda pair: pair[1].start_period,
        )
        for i in range(1, len(sorted_phases)):
            prev_idx, prev = sorted_phases[i - 1]
            curr_idx, curr = sorted_phases[i]
            if curr.start_period <= prev.end_period:
                raise ValueError(
                    f"correlation_phases[{prev_idx}] (periods "
                    f"{prev.start_period}-{prev.end_period}) overlaps with "
                    f"correlation_phases[{curr_idx}] (periods "
                    f"{curr.start_period}-{curr.end_period}). Phases must "
                    f"not overlap; adjacent windows may abut (one ending at "
                    f"period N and the next starting at N+1)."
                )
        return self

    def resolve_period_to_phase(self) -> list[Optional[int]]:
        """0.6-M11: build the period_index → phase index resolution table.

        Returns a length-``period_count()`` list where entry ``t`` is the
        index into ``correlation_phases`` of the phase covering period
        ``t``, or ``None`` if no phase covers that period (in which case
        the baseline ``correlations`` applies).

        Pure function of ``time_window`` and ``correlation_phases``; called
        once at the engine's Cholesky-build site to avoid per-period
        scanning. Empty ``correlation_phases`` yields a list of ``None``
        (every period falls back to baseline) — equivalent to the
        single-Cholesky pre-M11 path.
        """
        n_periods = self.time_window.period_count()
        result: list[Optional[int]] = [None] * n_periods
        for phase_idx, phase in enumerate(self.correlation_phases):
            for t in range(phase.start_period, phase.end_period + 1):
                if t < n_periods:
                    result[t] = phase_idx
        return result

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

        0.6-M11: when ``correlation_phases`` is set, every phase's
        correlation matrix is independently projected too. Per-phase
        adjustments are stashed on ``_phase_correlation_adjustments``
        keyed by phase index.

        Only raises if Higham AND the eigenvalue-clipping fallback both
        fail to produce a PD matrix — should be impossible for
        symmetric input.
        """
        if not self.correlations:
            return self
        # Local import: plotsim.validation imports from plotsim.config,
        # so a module-level import would create a cycle.
        from plotsim.metrics import _format_correlation_adjustment_warning
        from plotsim.validation import (
            project_correlation_or_issue,
            project_phase_correlation_or_issue,
        )

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

        # 0.6-M11: per-phase PSD projection + warning emission. Each
        # phase carries its own correlation matrix (built against the
        # SAME metric set as the baseline); each is independently
        # checked, projected, and warned about. Per-phase adjustments
        # land on ``_phase_correlation_adjustments`` keyed by phase
        # index for the manifest to surface. The dict is set whenever
        # any phase reports adjustments — phases without adjustments
        # are absent from the dict (None vs empty dict mirrors the
        # baseline _correlation_adjustments contract).
        if self.correlation_phases:
            phase_adj_records: dict[int, list[dict]] = {}
            for phase_idx, phase in enumerate(self.correlation_phases):
                if not phase.correlations:
                    continue
                ph_issues, ph_adjustments, _ph_projected = project_phase_correlation_or_issue(
                    self, phase_idx
                )
                if ph_issues:
                    names = [m.name for m in self.metrics]
                    raise ValueError(
                        f"correlation_phases[{phase_idx}] (periods "
                        f"{phase.start_period}-{phase.end_period}): "
                        f"correlation matrix could not be projected to "
                        f"positive-definite for metrics {names}. "
                        f"{ph_issues[0].message}"
                    )
                if ph_adjustments:
                    warnings.warn(
                        f"correlation_phases[{phase_idx}] (periods "
                        f"{phase.start_period}-{phase.end_period}): "
                        + _format_correlation_adjustment_warning(ph_adjustments),
                        UserWarning,
                        stacklevel=2,
                    )
                    phase_adj_records[phase_idx] = ph_adjustments
            if phase_adj_records:
                self._phase_correlation_adjustments = phase_adj_records
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
