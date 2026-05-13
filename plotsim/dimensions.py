"""plotsim.dimensions — dimension table builders (date spine, entity, sub-entity, reference).

What it does:
    Builds every non-behavioural table the engine needs before fact/event
    generation can resolve foreign keys:

      * dim_date          — true date spine at the configured granularity.
                            Fixed schema, independent of the config's column
                            declarations for that table. Consumers DATE-JOIN
                            to this; we never inline DATE_TRUNC at fact time.
      * dim_<entity>      — one row per Entity (per_entity grain), static
                            attributes drawn from Faker / derived fields.
      * dim_<subentity>   — variable-grain dim table: N = sum(entity.size)
                            rows, each FK-linked to its parent dim_<entity>.
                            (The schema enum has no ``per_subentity`` grain
                            yet; we route on grain == "variable" + type ==
                            "dim" + presence of an FK to a per_entity dim.
                            Mission 005 completion report flags this.)
      * dim_<reference>   — small lookup tables (plans, departments…). Row
                            count is driven by static CSV values; the longest
                            static-valued column wins.

    Zero trajectory involvement — this module does not import from
    ``plotsim.curves`` or ``plotsim.metrics``. It imports
    ``compute_time_steps`` from trajectory.py only for date-label math.

Input:
    PlotsimConfig + a seeded ``numpy.random.Generator``. Faker is seeded
    deterministically from that rng inside each builder, so a given
    (config, seed) pair always yields the same company names.

Output:
    dict mapping table name → pandas.DataFrame. Build order is internal;
    Mission 006 calls ``build_all_dimensions`` once.
"""

from __future__ import annotations

import calendar
import datetime as _dt
from typing import Any, Optional

import numpy as np
import pandas as pd
from faker import Faker

from plotsim._column_dispatch import (
    COLUMN_DISPATCH,
    BuilderKind,
)
from plotsim._faker import _make_faker
from plotsim.config import (
    Column,
    DerivedSource,
    Entity,
    FKSource,
    FakerSource,
    GeneratedSource,
    NestedSource,
    PKSource,
    PlotsimConfig,
    PoolSource,
    SCDType2Source,
    StaticSource,
    Table,
    TimeWindow,
    parse_source,
)
from plotsim.data import GEO_BUNDLE_FIELDS, GEO_LOCATIONS


# --- Helpers ----------------------------------------------------------------


def _id_prefix(table_name: str, explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit
    base = table_name
    if base.startswith("dim_"):
        base = base[4:]
    if not base:
        raise ValueError(f"cannot derive ID prefix from table name {table_name!r}")
    return base[0]


def _id_pad_width(n_rows: int) -> int:
    return max(3, len(str(max(n_rows, 1))))


def _make_ids(table_name: str, n_rows: int) -> list[str]:
    prefix = _id_prefix(table_name)
    width = _id_pad_width(n_rows)
    return [f"{prefix}-{i + 1:0{width}d}" for i in range(n_rows)]


def _coerce_static(value: str, dtype: str) -> Any:
    """Cast a raw static-source string to the column's declared dtype.

    On ``dtype: date`` columns, malformed ISO dates raise instead of
    silently returning the raw string. The primary load-time check at
    ``PlotsimConfig._cross_reference_integrity`` rejects malformed static
    dates before generation runs; this defensive raise catches the same
    bug class on programmatic ``PlotsimConfig`` construction that
    bypasses YAML-loading validators.
    """
    v = value.strip()
    if dtype == "int":
        return int(float(v))
    if dtype == "float":
        return float(v)
    if dtype == "boolean":
        return v.lower() in {"true", "1", "yes", "y"}
    if dtype == "date":
        try:
            return _dt.date.fromisoformat(v)
        except ValueError as exc:
            raise ValueError(
                f"static value {v!r} on a dtype: date column is not a "
                f"valid ISO date (expected YYYY-MM-DD)"
            ) from exc
    return v


def _split_static(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",")]


# Geo bundle provider (`generated:geo.<field>`) — row-coherent reference data.
#
# When ANY column on a dim table declares a ``generated:geo.<field>`` source,
# the builder pre-allocates one bundle entry per row from
# ``plotsim.data.GEO_LOCATIONS`` and threads it through ctx. Each
# ``geo.<field>`` dispatch reads that row's bundle and returns the matching
# field — so two geo columns on the same row always agree (city ∈ stated
# country, latitude/longitude on the named city, etc.). The dispatch lives
# alongside ``_per_entity_generated`` / ``_sub_generated`` /
# ``_ref_generated`` because geo is a dim-only concern: facts and events
# already reject unknown ``generated:`` providers in their own dispatchers.


def _geo_provider_field(provider: str) -> Optional[str]:
    """Return the bundle field name for a ``geo.<field>`` provider, or None.

    ``provider`` is the substring after ``generated:`` (e.g.
    ``geo.country``). Returns ``"country"`` for that, ``None`` if the
    provider is not a geo bundle reference. Unknown ``geo.foo`` raises so
    typos surface at first dispatch instead of later as a missing-key
    KeyError.
    """
    if not provider.startswith("geo."):
        return None
    field = provider[len("geo.") :]
    if field not in GEO_BUNDLE_FIELDS:
        raise ValueError(
            f"unknown geo bundle field {field!r} (provider 'generated:{provider}'); "
            f"supported fields: {sorted(GEO_BUNDLE_FIELDS)}"
        )
    return field


def _table_uses_geo_bundle(columns: list[Column]) -> bool:
    """True if any column on the table declares a ``generated:geo.<field>`` source."""
    for col in columns:
        parsed = parse_source(col.source)
        if isinstance(parsed, GeneratedSource) and parsed.provider.startswith("geo."):
            # Validate the field name eagerly — keeps the bundle-allocation
            # path from silently no-op'ing on a typo'd provider.
            _geo_provider_field(parsed.provider)
            return True
    return False


def _assign_geo_bundles(
    columns: list[Column],
    n_rows: int,
    rng: np.random.Generator,
) -> Optional[list[dict[str, object]]]:
    """Pre-allocate one geo bundle per row, or None if the table has no geo columns.

    A single ``rng.integers`` call draws ``n_rows`` indices from
    ``GEO_LOCATIONS``; the dispatcher then reads the row's bundle field-by-
    field as each ``geo.<field>`` column is resolved. The single rng draw
    keeps determinism predictable: row K's bundle index depends on row K-1
    only via the rng's internal state, the same as every other random
    column on this table.
    """
    if not _table_uses_geo_bundle(columns):
        return None
    if n_rows <= 0:
        return []
    indices = rng.integers(0, len(GEO_LOCATIONS), size=n_rows)
    return [GEO_LOCATIONS[int(idx)] for idx in indices]


# Extended providers that fill gaps in stock Faker. The sample SaaS YAML
# reaches for ``faker.industry`` and ``faker.year`` — neither is a core Faker
# method. Rather than mutate the sample configs (out of M005 scope), we keep a
# tiny shim here. Lookup order in ``_call_faker`` is: extended provider first,
# then stock Faker's getattr.
_EXTENDED_PROVIDERS = {
    "industry": lambda fake: fake.random_element(
        (
            "Technology",
            "Healthcare",
            "Finance",
            "Retail",
            "Manufacturing",
            "Education",
            "Energy",
            "Media",
            "Transportation",
            "Hospitality",
            "Consulting",
            "Real Estate",
        )
    ),
    "year": lambda fake: fake.random_int(min=1950, max=2020),
}


# SEC-04: explicit allowlist of Faker methods the engine will dispatch to.
# Generated from the set needed by the five bundled templates plus the most
# common portfolio/persona/geo/date/text/net fakers. Anything outside this
# list is rejected with a clear error pointing the user at the contribution
# path — new additions go through review rather than ``getattr(fake, ...)``.
ALLOWED_FAKER_METHODS: frozenset[str] = frozenset(
    {
        # Person / identity
        "name",
        "first_name",
        "last_name",
        "email",
        "phone_number",
        "user_name",
        "job",
        # Company / business
        "company",
        "bs",
        "catch_phrase",
        # Address / geo
        "address",
        "street_address",
        "city",
        "state",
        "country",
        "country_code",
        "zipcode",
        "postcode",
        "latitude",
        "longitude",
        # Date / time
        "date",
        "date_between",
        "date_this_decade",
        "date_of_birth",
        "date_time",
        "iso8601",
        "unix_time",
        # Text
        "sentence",
        "paragraph",
        "text",
        "word",
        "words",
        # Network / identifiers
        "url",
        "domain_name",
        "ipv4",
        "ipv6",
        "mac_address",
        "uuid4",
        "md5",
        "sha1",
        "sha256",
        # Numeric / primitives
        "random_int",
        "random_element",
        "random_elements",
        "boolean",
        "pybool",
        "pyint",
        "pyfloat",
        "pydecimal",
        # Misc common
        "currency",
        "currency_code",
        "color_name",
        "hex_color",
        "file_name",
        "file_extension",
        "mime_type",
        "license_plate",
        "vin",
        "ean",
        "ean13",
    }
)

# SEC-04: defense-in-depth denylist. Methods here are rejected even if a
# future change pulls them into the allowlist, because they either mutate
# Faker's RNG state (breaking determinism), dispatch dynamically to bypass
# the allowlist, or amplify memory by orders of magnitude per row.
DENIED_FAKER_METHODS: frozenset[str] = frozenset(
    {
        "seed",
        "seed_instance",
        "seed_locale",
        "add_provider",
        "del_provider",
        "format",
        "parse",
        "pystr_format",
        "provider",
        "providers",
        "binary",
    }
)

# SEC-04: cap on length-like kwarg values. 4096 is generous for sentences,
# paragraphs, and random element counts; anything larger is almost certainly
# an amplification attempt (``binary:length:1_000_000_000`` style).
_FAKER_KWARG_LENGTH_CAP = 4096
_FAKER_LENGTH_KWARGS: frozenset[str] = frozenset(
    {
        "length",
        "max_nb_chars",
        "nb",
        "min_chars",
        "max_chars",
        "nb_elements",
        "max_value",
        "nb_words",
        "nb_sentences",
    }
)


def _check_faker_method_allowed(method: str) -> None:
    """Raise if ``method`` is denylisted or missing from the allowlist.

    Extended providers (``industry``, ``year``) are always permitted — they
    are literal constants, not live Faker dispatch.
    """
    if method in _EXTENDED_PROVIDERS:
        return
    if method in DENIED_FAKER_METHODS:
        raise ValueError(
            f"Faker method {method!r} is explicitly denied for security "
            f"reasons (determinism / dynamic-dispatch / memory amplification). "
            f"If you have a legitimate use case, open an issue."
        )
    if method not in ALLOWED_FAKER_METHODS:
        allowed = sorted(ALLOWED_FAKER_METHODS)
        raise ValueError(
            f"Faker method {method!r} is not in the allowed list. "
            f"Allowed methods: {allowed}. To request an addition, open an issue."
        )


def _check_faker_kwarg_caps(method: str, kwargs: dict[str, Any]) -> None:
    """Raise if any length-related kwarg exceeds the hard cap.

    Runs after :func:`_coerce_faker_kwarg` so the value is already typed —
    only integer values count as ``length`` overruns. Strings and dates
    pass through even if they share a capped name.
    """
    for key, value in kwargs.items():
        if key not in _FAKER_LENGTH_KWARGS:
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, int) and value > _FAKER_KWARG_LENGTH_CAP:
            raise ValueError(
                f"Faker kwarg {key}={value} on method {method!r} exceeds "
                f"the {_FAKER_KWARG_LENGTH_CAP} cap; lower the value or "
                f"request a higher cap via an issue."
            )


def _coerce_faker_kwarg(value: str) -> Any:
    """Best-effort string → int/date coercion for parameterized faker kwargs.

    Applied to every kwarg value parsed out of a
    ``generated:faker.<method>:<key>:<value>:...`` source. Heuristic:

      * ``YYYY-MM-DD`` → ``datetime.date`` (covers ``date_between`` bounds).
      * All-digit (with optional leading ``-``) → ``int`` (covers
        ``random_int(min=N, max=M)``).
      * Everything else → passthrough string.

    Faker accepts strings for several providers natively, so when coercion
    doesn't match, the string goes through untouched — users can still
    pass provider-specific formats we don't recognize.
    """
    if len(value) == 10 and value[4] == "-" and value[7] == "-":
        try:
            return _dt.date.fromisoformat(value)
        except ValueError:
            pass
    trimmed = value[1:] if value.startswith("-") else value
    if trimmed.isdigit():
        try:
            return int(value)
        except ValueError:
            pass
    return value


def _call_faker(
    fake: Faker,
    method: str,
    kwargs: Optional[dict[str, str]] = None,
) -> Any:
    """Invoke ``fake.<method>(**coerced_kwargs)``. Raises at build time on typos.

    Extended providers (``industry``, ``year``) short-circuit the getattr
    lookup and do not accept kwargs. Stock Faker methods accept any kwargs
    the user supplied in the source string, coerced via
    :func:`_coerce_faker_kwarg`.
    """
    if not method:
        raise ValueError("faker method name cannot be empty")
    kwargs = kwargs or {}
    # SEC-04: allowlist+denylist guard runs before any dispatch so attacker-
    # controlled method names (``seed_instance``, ``format``, ``binary``)
    # never reach ``getattr`` / the extended-providers table.
    _check_faker_method_allowed(method)
    extended = _EXTENDED_PROVIDERS.get(method)
    if extended is not None:
        if kwargs:
            raise ValueError(
                f"extended faker provider {method!r} does not accept kwargs; got {sorted(kwargs)}"
            )
        return extended(fake)
    fn = getattr(fake, method, None)
    if fn is None or not callable(fn):
        raise ValueError(
            f"Faker has no provider method {method!r} (source 'generated:faker.{method}')"
        )
    coerced = {k: _coerce_faker_kwarg(v) for k, v in kwargs.items()}
    _check_faker_kwarg_caps(method, coerced)
    return fn(**coerced)


# --- dim_date ----------------------------------------------------------------

_DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _iso_week_monday(label: str) -> _dt.date:
    """Return the Monday of the ISO week named ``YYYY-Www``."""
    year_s, week_s = label.split("-W")
    return _dt.date.fromisocalendar(int(year_s), int(week_s), 1)


def _label_to_anchor_date(label: str, granularity: str) -> _dt.date:
    """Each period's canonical date: 1st-of-month, Monday-of-week, or the day."""
    if granularity == "monthly":
        y, m = label.split("-")
        return _dt.date(int(y), int(m), 1)
    if granularity == "weekly":
        return _iso_week_monday(label)
    if granularity == "daily":
        return _dt.date.fromisoformat(label)
    raise ValueError(f"unknown granularity {granularity!r}")


def build_dim_date(time_window: TimeWindow) -> pd.DataFrame:
    """Date spine covering ``time_window`` with no gaps and no duplicates.

    Granularity-invariant columns (always present):
      date_key, date, year, quarter, month, month_name, week,
      period_label, period_index.

    Daily-only columns: day_of_week, day_of_month, is_weekend.
    """
    # Imported lazily — keeping the dim module's import graph small is handy
    # when someone is iterating on trajectory.py itself.
    from plotsim.trajectory import compute_time_steps

    labels = compute_time_steps(time_window)
    granularity = time_window.granularity

    rows: list[dict[str, Any]] = []
    for idx, lbl in enumerate(labels):
        d = _label_to_anchor_date(str(lbl), granularity)
        row: dict[str, Any] = {
            "date_key": d.year * 10000 + d.month * 100 + d.day,
            "date": d,
            "year": d.year,
            "quarter": (d.month - 1) // 3 + 1,
            "month": d.month,
            "month_name": calendar.month_name[d.month],
            "week": d.isocalendar().week,
            "period_label": str(lbl),
            "period_index": idx,
        }
        if granularity == "daily":
            row["day_of_week"] = _DAY_NAMES[d.weekday()]
            row["day_of_month"] = d.day
            row["is_weekend"] = d.weekday() >= 5
        rows.append(row)

    df = pd.DataFrame(rows)
    # Enforce the column order the acceptance tests expect regardless of
    # insertion ordering quirks.
    base_cols = [
        "date_key",
        "date",
        "year",
        "quarter",
        "month",
        "month_name",
        "week",
        "period_label",
        "period_index",
    ]
    daily_cols = ["day_of_week", "day_of_month", "is_weekend"]
    ordered = base_cols + (daily_cols if granularity == "daily" else [])
    return df[ordered]


# --- dim_entity --------------------------------------------------------------


def _resolve_derived(field: str, entity: Entity) -> Any:
    if field == "size":
        return entity.size
    if field == "archetype":
        return entity.archetype
    if field == "name" or field == "entity_name":
        return entity.name
    raise ValueError(
        f"unsupported derived field {field!r}: expected one of 'size', 'archetype', 'name'"
    )


def _column_value_for_entity(
    col: Column,
    entity: Entity,
    entity_pk: str,
    fake: Faker,
    rng: Optional[np.random.Generator] = None,
    geo_bundle: Optional[dict[str, object]] = None,
) -> Any:
    """Resolve one cell on a per_entity dim row.

    M127b: dispatch handlers live in the shared ``COLUMN_DISPATCH``
    registry so adding a new source type for per_entity dims is a
    single ``register(...)`` call below.

    ``geo_bundle`` is the row-level dict drawn from ``GEO_LOCATIONS`` when
    the table has any ``generated:geo.<field>`` column; ``None`` otherwise.
    """
    parsed = parse_source(col.source)
    ctx = {
        "col": col,
        "entity": entity,
        "entity_pk": entity_pk,
        "fake": fake,
        "rng": rng,
        "geo_bundle": geo_bundle,
    }
    return COLUMN_DISPATCH.dispatch(BuilderKind.PER_ENTITY_DIM, parsed, ctx)


def _per_entity_pk(parsed: PKSource, ctx: dict):
    return ctx["entity_pk"]


def _per_entity_static(parsed: StaticSource, ctx: dict):
    col = ctx["col"]
    values = _split_static(parsed.value)
    # Entity rows are 1:1; broadcast the first value regardless of count.
    return _coerce_static(values[0], col.dtype)


def _per_entity_derived(parsed: DerivedSource, ctx: dict):
    return _resolve_derived(parsed.field, ctx["entity"])


def _per_entity_generated(parsed: GeneratedSource, ctx: dict):
    col = ctx["col"]
    if parsed.provider == "entity_name":
        return ctx["entity"].name
    geo_field = _geo_provider_field(parsed.provider)
    if geo_field is not None:
        bundle = ctx.get("geo_bundle")
        if bundle is None:
            raise ValueError(
                f"column {col.name!r} source {col.source!r} requires a row-level "
                f"geo bundle, but none was assigned; this is an internal wiring "
                f"bug — _table_uses_geo_bundle should have triggered allocation"
            )
        return bundle[geo_field]
    raise ValueError(
        f"column {col.name!r} source {col.source!r} is not supported on "
        f"per_entity dimension tables: non-faker 'generated:' providers "
        f"other than 'entity_name' and 'geo.<field>' (e.g. 'timestamp', "
        f"'date_key') only make sense on fact/event tables"
    )


def _per_entity_faker(parsed: FakerSource, ctx: dict):
    return _call_faker(ctx["fake"], parsed.method, parsed.kwargs)


def _per_entity_fk(parsed: FKSource, ctx: dict):
    # dim_entity rows are 1:1 with entities, so FKs can't be populated
    # meaningfully here — the parent row this entity would point to is
    # itself. Reference tables with many-to-one FKs are built via
    # dim_reference; cross-dim FKs on per_entity tables (e.g. dim_employee
    # → dim_department) are resolved by broadcasting the first reference
    # row's PK, since reference tables are tiny and row-0 is deterministic.
    return None  # filled in by _backfill_fks


def _per_entity_scd2(parsed: SCDType2Source, ctx: dict):
    # M106: SCD label cells are populated by ``tables.expand_scd_dims``,
    # which has the per-entity trajectory and the SCDType2Config's
    # thresholds/labels. Emit a None placeholder here so the per-entity
    # initial dim row carries the column slot; the expansion step
    # rewrites both the label and the row count.
    return None


def _per_entity_pool(parsed: PoolSource, ctx: dict):
    # M114: per-entity value pool. ``Column._pool_pairing`` and
    # ``validate_value_pool_coverage`` already guarantee value_pool is
    # set and contains an entry for this entity. Sample one value via
    # the caller-supplied RNG so determinism is preserved under the
    # engine's single-seed contract; rng is required when any column
    # on the table has a PoolSource.
    col = ctx["col"]
    entity = ctx["entity"]
    rng = ctx["rng"]
    if rng is None:
        raise ValueError(
            f"column {col.name!r} has source {col.source!r} but no "
            f"RNG was supplied to _column_value_for_entity; pool "
            f"sampling requires the per-table RNG"
        )
    assert col.value_pool is not None  # _pool_pairing
    choices = col.value_pool.get(entity.name)
    if not choices:
        raise ValueError(
            f"column {col.name!r} value_pool has no entry for entity "
            f"{entity.name!r}; coverage is enforced at config load — "
            f"reaching this branch means the validator was bypassed"
        )
    idx = int(rng.integers(0, len(choices)))
    return _coerce_static(choices[idx], col.dtype)


def _per_entity_unsupported(parsed: Any, ctx: dict):
    col = ctx["col"]
    raise ValueError(
        f"column {col.name!r} source {col.source!r} is not supported on per_entity dimension tables"
    )


# 0.6-M14c: nested column type cell builder. Generates a Python dict
# (struct) or list (array) per cell from a seeded RNG. Each
# field/element value is drawn independently from a primitive-type
# generator below; one cell consumes ``len(nested_schema)`` (struct)
# or ``array_length`` (array) RNG draws so RNG-consumption order is
# deterministic for a given config.
def _generate_nested_primitive(type_word: str, rng) -> Any:
    """Draw one primitive value of ``type_word`` from ``rng``.

    Same draw count per call regardless of type (one ``rng.integers``
    or one ``rng.random``) so reordering struct fields in config
    never changes the RNG state more than the field count itself.
    """
    if type_word == "int":
        return int(rng.integers(0, 1000))
    if type_word == "float":
        return float(rng.random())
    if type_word == "string":
        # Deterministic short string token. Users who need realistic
        # strings should use a separate ``faker.<method>`` column
        # instead — nested cells are intended for structured
        # JSON-like payloads, not free-form prose.
        return f"v{int(rng.integers(0, 100000)):05d}"
    if type_word == "boolean":
        return bool(int(rng.integers(0, 2)))
    raise ValueError(
        f"unknown nested primitive type {type_word!r}; valid: int / float / string / boolean"
    )


def _generate_nested_value(col: Column, rng) -> Any:
    """Generate one nested cell value (dict for struct, list for array).

    Pre-conditions enforced by ``Column._nested_pairing``:
      * struct columns have ``nested_schema`` set
      * array columns have ``array_element_type`` set
      * primitive types in both are validated
    """
    if col.dtype == "struct":
        assert col.nested_schema is not None  # _nested_pairing
        return {field: _generate_nested_primitive(t, rng) for field, t in col.nested_schema.items()}
    if col.dtype == "array":
        assert col.array_element_type is not None  # _nested_pairing
        n = col.array_length if col.array_length is not None else 3
        return [_generate_nested_primitive(col.array_element_type, rng) for _ in range(n)]
    raise ValueError(
        f"_generate_nested_value called on column {col.name!r} with "
        f"dtype={col.dtype!r}; nested generation requires dtype "
        f"'struct' or 'array'"
    )


def _per_entity_nested(parsed: NestedSource, ctx: dict):
    col = ctx["col"]
    rng = ctx["rng"]
    if rng is None:
        raise ValueError(
            f"column {col.name!r} has source 'nested' but no RNG was "
            f"supplied to _column_value_for_entity; nested cell "
            f"generation requires the per-table RNG"
        )
    return _generate_nested_value(col, rng)


COLUMN_DISPATCH.register(BuilderKind.PER_ENTITY_DIM, PKSource, _per_entity_pk)
COLUMN_DISPATCH.register(BuilderKind.PER_ENTITY_DIM, StaticSource, _per_entity_static)
COLUMN_DISPATCH.register(BuilderKind.PER_ENTITY_DIM, DerivedSource, _per_entity_derived)
COLUMN_DISPATCH.register(
    BuilderKind.PER_ENTITY_DIM,
    GeneratedSource,
    _per_entity_generated,
)
COLUMN_DISPATCH.register(BuilderKind.PER_ENTITY_DIM, FakerSource, _per_entity_faker)
COLUMN_DISPATCH.register(BuilderKind.PER_ENTITY_DIM, FKSource, _per_entity_fk)
COLUMN_DISPATCH.register(BuilderKind.PER_ENTITY_DIM, SCDType2Source, _per_entity_scd2)
COLUMN_DISPATCH.register(BuilderKind.PER_ENTITY_DIM, PoolSource, _per_entity_pool)
COLUMN_DISPATCH.register(BuilderKind.PER_ENTITY_DIM, NestedSource, _per_entity_nested)
COLUMN_DISPATCH.register_unsupported(
    BuilderKind.PER_ENTITY_DIM,
    _per_entity_unsupported,
)


def sample_fk_values(
    column: Column,
    parent_pks: list[Any],
    n: int,
    rng: np.random.Generator,
    anchored_value: Optional[Any] = None,
) -> list[Any]:
    """Draw ``n`` FK values for ``column`` against the parent's PK list.

    Resolution order:

      1. ``anchored_value`` not None → broadcast it for every row. Caller
         supplies this when an Entity's ``cross_dim_fks`` pinned the value.
         No randomness consumed.
      2. ``len(parent_pks) == 1`` → broadcast that single PK.
         No randomness consumed.
      3. ``column.distribution.weights`` is set → weighted sample over the
         keys named in ``weights``; keys must exist in ``parent_pks``.
      4. Otherwise (uniform default) → uniform random over ``parent_pks``.

    All randomness flows through ``rng`` so determinism is preserved.
    """
    if anchored_value is not None:
        return [anchored_value] * n
    if len(parent_pks) == 1:
        return [parent_pks[0]] * n
    spec = column.distribution
    if spec is not None and spec.weights is not None:
        keys = list(spec.weights.keys())
        unknown = [k for k in keys if k not in parent_pks]
        if unknown:
            raise ValueError(
                f"FK column {column.name!r} distribution.weights references "
                f"PK value(s) {unknown} not present in parent (known: "
                f"{parent_pks})"
            )
        weights = np.array([spec.weights[k] for k in keys], dtype=float)
        probs = weights / weights.sum()
        idx = rng.choice(len(keys), size=n, p=probs)
        return [keys[i] for i in idx]
    # Uniform random over parent PKs (default for multi-row parent).
    idx = rng.integers(0, len(parent_pks), size=n)
    return [parent_pks[i] for i in idx]


def _backfill_fks(
    df: pd.DataFrame,
    table: Table,
    dims: dict[str, pd.DataFrame],
    rng: np.random.Generator,
    entities: Optional[list[Entity]] = None,
) -> pd.DataFrame:
    """Populate FK columns left None at build time.

    Single-row parent dim → broadcast the lone PK.
    Multi-row parent dim → distribute via ``sample_fk_values``: uniform by
    default, or weighted when the column declares ``distribution.weights``.

    When ``entities`` is supplied (per_entity dims), each row's
    ``Entity.cross_dim_fks[col.name]`` overrides distribution sampling for
    that row — used to anchor a cohort to a specific reference value
    (e.g. enterprise_accounts → plan_id=enterprise).
    """
    n_rows = len(df)
    for col in table.columns:
        parsed = parse_source(col.source)
        if not isinstance(parsed, FKSource):
            continue
        parent = dims.get(parsed.table)
        if parent is None or parent.empty:
            raise ValueError(
                f"table {table.name!r} FK column {col.name!r} references "
                f"{parsed.table!r}, which is not yet built or is empty"
            )
        if parsed.column not in parent.columns:
            raise ValueError(
                f"table {table.name!r} FK column {col.name!r} references "
                f"{parsed.table}.{parsed.column}, but that column does not exist"
            )
        parent_pks = parent[parsed.column].tolist()
        if (
            entities is not None
            and len(entities) == n_rows
            and any(col.name in e.cross_dim_fks for e in entities)
        ):
            # Per-row dispatch so each entity can pin its own value or fall
            # back to distribution sampling. Single rng draw per uncolumned
            # row keeps determinism predictable.
            values: list[Any] = []
            for entity in entities:
                anchored = entity.cross_dim_fks.get(col.name)
                if anchored is not None:
                    if anchored not in parent_pks:
                        raise ValueError(
                            f"entity {entity.name!r} cross_dim_fks pins "
                            f"{col.name!r}={anchored!r}, not in parent "
                            f"{parsed.table!r} PKs {parent_pks}"
                        )
                    values.append(anchored)
                else:
                    values.extend(sample_fk_values(col, parent_pks, 1, rng))
            df[col.name] = values
        else:
            df[col.name] = sample_fk_values(col, parent_pks, n_rows, rng)
    return df


def build_dim_entity(
    table_config: Table,
    entities: list[Entity],
    rng: np.random.Generator,
    locale: str | list[str] = "en_US",
) -> pd.DataFrame:
    """Build a per_entity dim: one row per Entity, static attributes from Faker/derived."""
    fake = _make_faker(rng, locale)
    ids = _make_ids(table_config.name, len(entities))
    geo_bundles = _assign_geo_bundles(table_config.columns, len(entities), rng)

    rows: list[dict[str, Any]] = []
    for row_idx, (entity, pk) in enumerate(zip(entities, ids)):
        row: dict[str, Any] = {}
        bundle = geo_bundles[row_idx] if geo_bundles is not None else None
        for col in table_config.columns:
            row[col.name] = _column_value_for_entity(col, entity, pk, fake, rng, geo_bundle=bundle)
        rows.append(row)

    df = pd.DataFrame(rows, columns=[c.name for c in table_config.columns])
    return df


# --- dim_subentity -----------------------------------------------------------


def _identify_parent_fk(
    table_config: Table,
    parent_dim_names: set[str],
) -> tuple[str, str, str]:
    """Find the column in ``table_config`` that FKs into a per_entity dim.

    Returns (local_column_name, parent_table, parent_pk_column).
    Raises if there's no such column (we shouldn't have routed here) or if
    there are multiple (the parent is ambiguous — surface it).
    """
    matches: list[tuple[str, str, str]] = []
    for col in table_config.columns:
        parsed = parse_source(col.source)
        if isinstance(parsed, FKSource) and parsed.table in parent_dim_names:
            matches.append((col.name, parsed.table, parsed.column))
    if not matches:
        raise ValueError(
            f"table {table_config.name!r} routed as sub-entity but has no FK "
            f"to a per_entity dim; known per_entity dims: "
            f"{sorted(parent_dim_names)}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"table {table_config.name!r} has multiple FKs to per_entity dims "
            f"{[m[1] for m in matches]}; cannot infer the parent"
        )
    return matches[0]


def build_dim_subentity(
    table_config: Table,
    entities: list[Entity],
    dim_entity: pd.DataFrame,
    rng: np.random.Generator,
    parent_pk_column: Optional[str] = None,
    local_fk_column: Optional[str] = None,
    locale: str | list[str] = "en_US",
) -> pd.DataFrame:
    """Build a sub-entity dim: sum(entity.size) rows keyed back to dim_entity.

    The parent FK column on this table is auto-detected (first FK whose target
    is a per_entity dim). Callers who already know the parent can pass it
    explicitly via ``local_fk_column`` / ``parent_pk_column``.
    """
    if len(entities) != len(dim_entity):
        raise ValueError(
            f"build_dim_subentity: {len(entities)} entities vs "
            f"{len(dim_entity)} dim_entity rows; parent must be 1:1 with entities"
        )
    if local_fk_column is None or parent_pk_column is None:
        # Single-parent auto-detection: whichever column FKs into a table
        # whose name matches the sole parent we know about.
        found = next(
            (
                (col.name, parse_source(col.source))
                for col in table_config.columns
                if isinstance(parse_source(col.source), FKSource)
            ),
            None,
        )
        if found is None:
            raise ValueError(
                f"table {table_config.name!r} has no FK column to anchor "
                f"sub-entity rows against a parent"
            )
        local_fk_column, parsed = found
        assert isinstance(parsed, FKSource)
        parent_pk_column = parsed.column

    fake = _make_faker(rng, locale)
    # M117: per-parent row count is ``entity.size * table_config.count``.
    # Engine-direct configs (Entity.size=N, Table.count=1) resolve to N×1=N;
    # builder configs (Entity.size=1, Table.count=K) resolve to 1×K=K. The
    # multiplication handles both paths without branching.
    total_rows = sum(e.size * table_config.count for e in entities)
    ids = _make_ids(table_config.name, total_rows)
    geo_bundles = _assign_geo_bundles(table_config.columns, total_rows, rng)

    rows: list[dict[str, Any]] = []
    cursor = 0
    for entity, (_, parent_row) in zip(entities, dim_entity.iterrows()):
        parent_pk_value = parent_row[parent_pk_column]
        for _ in range(entity.size * table_config.count):
            row: dict[str, Any] = {}
            local_pk = ids[cursor]
            base_ctx = {
                "entity": entity,
                "parent_pk_value": parent_pk_value,
                "local_fk_column": local_fk_column,
                "local_pk": local_pk,
                "fake": fake,
                "rng": rng,
                "geo_bundle": (geo_bundles[cursor] if geo_bundles is not None else None),
            }
            for col in table_config.columns:
                parsed = parse_source(col.source)
                ctx = dict(base_ctx)
                ctx["col"] = col
                row[col.name] = COLUMN_DISPATCH.dispatch(
                    BuilderKind.SUB_ENTITY_DIM,
                    parsed,
                    ctx,
                )
            rows.append(row)
            cursor += 1

    df = pd.DataFrame(rows, columns=[c.name for c in table_config.columns])
    return df


# --- Sub-entity dim dispatchers ----------------------------------------------


def _sub_pk(parsed: PKSource, ctx: dict):
    return ctx["local_pk"]


def _sub_fk(parsed: FKSource, ctx: dict):
    col = ctx["col"]
    if col.name == ctx["local_fk_column"]:
        return ctx["parent_pk_value"]
    # Unrelated FK (e.g. to a dim_reference); fill with None — _backfill_fks
    # rewrites it after dims are materialised.
    return None


def _sub_generated(parsed: GeneratedSource, ctx: dict):
    col = ctx["col"]
    if parsed.provider == "entity_name":
        return ctx["entity"].name
    geo_field = _geo_provider_field(parsed.provider)
    if geo_field is not None:
        bundle = ctx.get("geo_bundle")
        if bundle is None:
            raise ValueError(
                f"column {col.name!r} source {col.source!r} requires a row-level "
                f"geo bundle, but none was assigned; this is an internal wiring "
                f"bug — _table_uses_geo_bundle should have triggered allocation"
            )
        return bundle[geo_field]
    raise ValueError(
        f"column {col.name!r} source {col.source!r} is "
        f"not supported on sub-entity dimension tables: "
        f"only 'entity_name' and 'geo.<field>' are resolved here"
    )


def _sub_faker(parsed: FakerSource, ctx: dict):
    return _call_faker(ctx["fake"], parsed.method, parsed.kwargs)


def _sub_static(parsed: StaticSource, ctx: dict):
    col = ctx["col"]
    values = _split_static(parsed.value)
    return _coerce_static(values[0], col.dtype)


def _sub_derived(parsed: DerivedSource, ctx: dict):
    return _resolve_derived(parsed.field, ctx["entity"])


def _sub_unsupported(parsed: Any, ctx: dict):
    col = ctx["col"]
    raise ValueError(
        f"column {col.name!r} source {col.source!r} is not supported on sub-entity dimension tables"
    )


def _sub_nested(parsed: NestedSource, ctx: dict):
    col = ctx["col"]
    rng = ctx["rng"]
    if rng is None:
        raise ValueError(
            f"column {col.name!r} on sub-entity dim has source 'nested' "
            f"but no RNG in context; nested cell generation requires rng"
        )
    return _generate_nested_value(col, rng)


COLUMN_DISPATCH.register(BuilderKind.SUB_ENTITY_DIM, PKSource, _sub_pk)
COLUMN_DISPATCH.register(BuilderKind.SUB_ENTITY_DIM, FKSource, _sub_fk)
COLUMN_DISPATCH.register(BuilderKind.SUB_ENTITY_DIM, GeneratedSource, _sub_generated)
COLUMN_DISPATCH.register(BuilderKind.SUB_ENTITY_DIM, FakerSource, _sub_faker)
COLUMN_DISPATCH.register(BuilderKind.SUB_ENTITY_DIM, StaticSource, _sub_static)
COLUMN_DISPATCH.register(BuilderKind.SUB_ENTITY_DIM, DerivedSource, _sub_derived)
COLUMN_DISPATCH.register(BuilderKind.SUB_ENTITY_DIM, NestedSource, _sub_nested)
COLUMN_DISPATCH.register_unsupported(
    BuilderKind.SUB_ENTITY_DIM,
    _sub_unsupported,
)


# --- dim_reference -----------------------------------------------------------


def build_dim_reference(
    table_config: Table,
    rng: np.random.Generator,
    locale: str | list[str] = "en_US",
) -> pd.DataFrame:
    """Build a static reference/lookup dim. Row count = longest static-CSV column."""
    fake = _make_faker(rng, locale)

    # Determine row count: max count across static columns; default 1 if none.
    n_rows = 1
    for col in table_config.columns:
        parsed = parse_source(col.source)
        if isinstance(parsed, StaticSource):
            n_rows = max(n_rows, len(_split_static(parsed.value)))

    ids = _make_ids(table_config.name, n_rows)
    geo_bundles = _assign_geo_bundles(table_config.columns, n_rows, rng)

    rows: list[dict[str, Any]] = []
    for i in range(n_rows):
        row: dict[str, Any] = {}
        base_ctx = {
            "i": i,
            "ids": ids,
            "fake": fake,
            "rng": rng,
            "geo_bundle": geo_bundles[i] if geo_bundles is not None else None,
        }
        for col in table_config.columns:
            parsed = parse_source(col.source)
            ctx = dict(base_ctx)
            ctx["col"] = col
            row[col.name] = COLUMN_DISPATCH.dispatch(
                BuilderKind.REFERENCE_DIM,
                parsed,
                ctx,
            )
        rows.append(row)

    return pd.DataFrame(rows, columns=[c.name for c in table_config.columns])


# --- Reference dim dispatchers -----------------------------------------------


def _ref_pk(parsed: PKSource, ctx: dict):
    return ctx["ids"][ctx["i"]]


def _ref_static(parsed: StaticSource, ctx: dict):
    col = ctx["col"]
    values = _split_static(parsed.value)
    # Broadcast single-value columns; index into multi-value columns.
    pick = values[ctx["i"]] if len(values) > 1 else values[0]
    return _coerce_static(pick, col.dtype)


def _ref_faker(parsed: FakerSource, ctx: dict):
    return _call_faker(ctx["fake"], parsed.method, parsed.kwargs)


def _ref_generated(parsed: GeneratedSource, ctx: dict):
    col = ctx["col"]
    geo_field = _geo_provider_field(parsed.provider)
    if geo_field is not None:
        bundle = ctx.get("geo_bundle")
        if bundle is None:
            raise ValueError(
                f"column {col.name!r} source {col.source!r} requires a row-level "
                f"geo bundle, but none was assigned; this is an internal wiring "
                f"bug — _table_uses_geo_bundle should have triggered allocation"
            )
        return bundle[geo_field]
    raise ValueError(
        f"column {col.name!r} source {col.source!r} is not "
        f"supported on reference dimension tables: use "
        f"'generated:faker.<method>', 'generated:geo.<field>', or "
        f"'static:...' instead"
    )


def _ref_fk(parsed: FKSource, ctx: dict):
    return None  # backfilled after dims are materialised


def _ref_unsupported(parsed: Any, ctx: dict):
    col = ctx["col"]
    raise ValueError(
        f"column {col.name!r} source {col.source!r} is not supported on reference dimension tables"
    )


def _ref_nested(parsed: NestedSource, ctx: dict):
    col = ctx["col"]
    rng = ctx["rng"]
    if rng is None:
        raise ValueError(
            f"column {col.name!r} on reference dim has source 'nested' "
            f"but no RNG in context; nested cell generation requires rng"
        )
    return _generate_nested_value(col, rng)


COLUMN_DISPATCH.register(BuilderKind.REFERENCE_DIM, PKSource, _ref_pk)
COLUMN_DISPATCH.register(BuilderKind.REFERENCE_DIM, StaticSource, _ref_static)
COLUMN_DISPATCH.register(BuilderKind.REFERENCE_DIM, FakerSource, _ref_faker)
COLUMN_DISPATCH.register(BuilderKind.REFERENCE_DIM, GeneratedSource, _ref_generated)
COLUMN_DISPATCH.register(BuilderKind.REFERENCE_DIM, FKSource, _ref_fk)
COLUMN_DISPATCH.register(BuilderKind.REFERENCE_DIM, NestedSource, _ref_nested)
COLUMN_DISPATCH.register_unsupported(
    BuilderKind.REFERENCE_DIM,
    _ref_unsupported,
)


# --- Orchestrator ------------------------------------------------------------


def _is_subentity_table(table: Table, per_entity_names: set[str]) -> bool:
    """Heuristic routing: dim + variable + FK into a per_entity dim."""
    if table.type != "dim" or table.grain != "variable":
        return False
    for col in table.columns:
        parsed = parse_source(col.source)
        if isinstance(parsed, FKSource) and parsed.table in per_entity_names:
            return True
    return False


def _per_entity_pk_column(table: Table) -> str:
    """0.6-M13: locate the canonical PK column on a per_entity dim.

    ``Table.primary_key`` is the canonical source — it's a string for
    single-column PKs (every per_entity dim that exists in the wild) and
    a list for composite PKs (facts only). For a per_entity dim with a
    composite PK the multi-source emission can't generate a coherent
    drifted ID, so we defensively raise.
    """
    pk = table.primary_key
    if isinstance(pk, list):
        raise ValueError(
            f"multi-source emission cannot drift composite-PK dim "
            f"{table.name!r} (primary_key={pk!r}); only single-column PKs "
            f"are supported on per_entity dims"
        )
    return pk


def _entity_type_for_dim(table_name: str) -> str:
    """Strip the leading ``dim_`` prefix; raise if absent.

    The convention enforced elsewhere in the engine is that every dim
    table name starts with ``dim_``. Used to derive the prefix for
    ``prefix_padded`` keys and the suffix on the renamed PK column
    (``<entity_type>_id_<source>``).
    """
    if not table_name.startswith("dim_"):
        raise ValueError(
            f"multi-source dim {table_name!r} does not start with 'dim_'; "
            f"cannot derive the entity-type prefix"
        )
    return table_name[len("dim_") :]


def _emit_per_source_dims(
    config: PlotsimConfig,
    dims: dict[str, pd.DataFrame],
    rng: np.random.Generator,
) -> None:
    """0.6-M13: write per-source drifted copies of every per_entity dim into ``dims``.

    For each ``SourceDeclaration`` (in config declaration order):
      1. Draw one seed integer off ``rng`` and instantiate a per-source RNG.
         Same pattern as quality issues' ``base_seed + seed_offset`` —
         shifts source order to a single deterministic offset on the dim
         RNG without coupling RNGs across sources.
      2. For each per_entity dim already built into ``dims``, copy the
         canonical frame, apply name/attribute drift via
         :func:`plotsim.multi_source.apply_source_drift`, and emit the
         result as ``dim_<entity>_<source>``.
      3. Accumulate per-row mapping records (entity name, source name,
         source PK, drifted fields) onto a single flat list, then stash
         the list on ``config._source_entity_mappings`` so
         ``manifest.build_manifest`` can surface it.

    Side effects:
      * Adds new keys to ``dims`` (one per ``(source, per_entity_dim)``).
      * Consumes RNG draws on ``rng`` (one per source).
      * Writes to ``config._source_entity_mappings`` (PrivateAttr).
    """
    # Local import: ``plotsim.multi_source`` would otherwise create a
    # circular dependency at module load (it imports config classes
    # we re-export from ``plotsim.config``, and ``dimensions`` is
    # imported by ``tables.py`` which the load order would chain into).
    from plotsim.multi_source import apply_source_drift

    if config.multi_source is None:
        return

    per_entity_tables = [t for t in config.tables if t.type == "dim" and t.grain == "per_entity"]
    if not per_entity_tables:
        # Cross-reference validator (``_multi_source_requires_per_entity_dim``)
        # already rejected this path; defensive no-op for any future caller
        # that bypasses validation.
        return

    entity_names_by_position = [e.name for e in config.entities]
    all_mappings: list[dict[str, Any]] = []

    for source in config.multi_source.sources:
        source_seed = int(rng.integers(0, 2**31 - 1))
        source_rng = np.random.default_rng(source_seed)

        for tbl in per_entity_tables:
            canonical_df = dims.get(tbl.name)
            if canonical_df is None:
                # Step 3 above builds every per_entity dim before we run;
                # a missing entry would mean an earlier-step bug, not a
                # multi-source bug. Surface it instead of silently skipping.
                raise ValueError(
                    f"multi-source emission: canonical dim {tbl.name!r} "
                    f"was not built before per-source pass; this is an "
                    f"internal orchestrator wiring bug"
                )
            entity_type = _entity_type_for_dim(tbl.name)
            pk_column = _per_entity_pk_column(tbl)
            drifted_df, _source_pk_col, per_row_mappings = apply_source_drift(
                canonical_df=canonical_df,
                canonical_columns=list(tbl.columns),
                canonical_pk_column=pk_column,
                source=source,
                entity_type=entity_type,
                rng=source_rng,
            )
            dims[f"{tbl.name}_{source.name}"] = drifted_df

            # Stitch entity name + source name onto each mapping record.
            # ``canonical_df`` rows are 1:1 with ``config.entities`` order
            # (per ``build_dim_entity``'s contract), so position N's entity
            # name is ``config.entities[N].name``. SCD expansion runs AFTER
            # ``build_all_dimensions`` returns, so canonical_df has not yet
            # been row-multiplied here.
            for row_idx, mapping in enumerate(per_row_mappings):
                entity_name = (
                    entity_names_by_position[row_idx]
                    if row_idx < len(entity_names_by_position)
                    else mapping["canonical_entity_id"]
                )
                all_mappings.append(
                    {
                        "entity": entity_name,
                        "source": source.name,
                        "dim_table": tbl.name,
                        "canonical_entity_id": mapping["canonical_entity_id"],
                        "source_entity_id": mapping["source_entity_id"],
                        "drifted_fields": mapping["drifted_fields"],
                    }
                )

    # PrivateAttr write — engine-derived runtime state, mirrors the
    # ``_correlation_compensations`` stash pattern. ``None`` (default)
    # signals "multi-source not configured"; an empty list would be
    # ambiguous with that. We only land here when ``multi_source is not
    # None`` so the empty-list case is impossible.
    config._source_entity_mappings = all_mappings


def build_all_dimensions(
    config: PlotsimConfig,
    rng: np.random.Generator,
) -> dict[str, pd.DataFrame]:
    """Build every dim table in ``config.tables``, returning {name: DataFrame}.

    Build order (enforced internally):
        1. dim_date             (no deps)
        2. reference dims       (no deps)
        3. per_entity dims      (FK-backfilled from reference dims)
        3b. per-source dims     (0.6-M13; drift overlay on per_entity dims)
        4. sub-entity dims      (FK to per_entity dim)

    Step 3b runs only when ``config.multi_source`` is set. It does NOT
    touch the canonical per_entity dims — those are passed through to
    sub-entity FK resolution (step 4) untouched, so fact / event tables
    that key off the canonical PK are unaffected by drift.
    """
    per_entity_names = {
        t.name for t in config.tables if t.type == "dim" and t.grain == "per_entity"
    }

    dims: dict[str, pd.DataFrame] = {}

    # 1. dim_date — fixed schema, routed by name.
    for tbl in config.tables:
        if tbl.type == "dim" and tbl.name == "dim_date":
            dims[tbl.name] = build_dim_date(config.time_window)

    # 2. reference dims.
    for tbl in config.tables:
        if tbl.type != "dim" or tbl.name == "dim_date":
            continue
        if tbl.grain == "per_reference":
            dims[tbl.name] = build_dim_reference(tbl, rng, locale=config.locale)

    # 3. per_entity dims — FK backfill from any reference dims built above.
    # Pass entities so per-cohort cross_dim_fks anchoring takes effect.
    for tbl in config.tables:
        if tbl.type == "dim" and tbl.grain == "per_entity":
            df = build_dim_entity(
                tbl,
                list(config.entities),
                rng,
                locale=config.locale,
            )
            dims[tbl.name] = _backfill_fks(
                df,
                tbl,
                dims,
                rng,
                entities=list(config.entities),
            )

    # 3b. 0.6-M13: per-source drifted copies of per_entity dims.
    _emit_per_source_dims(config, dims, rng)

    # 4. sub-entity dims.
    for tbl in config.tables:
        if tbl.type != "dim":
            continue
        if not _is_subentity_table(tbl, per_entity_names):
            continue
        local_fk, parent_table, parent_pk = _identify_parent_fk(tbl, per_entity_names)
        parent_df = dims.get(parent_table)
        if parent_df is None:
            raise ValueError(
                f"sub-entity table {tbl.name!r} needs parent "
                f"{parent_table!r}, which has not been built"
            )
        dims[tbl.name] = build_dim_subentity(
            tbl,
            list(config.entities),
            parent_df,
            rng,
            parent_pk_column=parent_pk,
            local_fk_column=local_fk,
            locale=config.locale,
        )

    return dims
