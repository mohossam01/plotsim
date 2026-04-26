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

from plotsim.config import (
    Column,
    DerivedSource,
    Entity,
    FKSource,
    FakerSource,
    GeneratedSource,
    PlotsimConfig,
    PKSource,
    StaticSource,
    Table,
    TimeWindow,
    parse_source,
)


# --- Helpers ----------------------------------------------------------------

# Providers the dim layer resolves without going through Faker.
_NON_FAKER_GENERATED = {"entity_name"}


def _faker_seed_from_rng(rng: np.random.Generator) -> int:
    """Derive a stable 32-bit seed from ``rng`` and consume one draw to do it."""
    return int(rng.integers(0, 2**31 - 1))


def _make_faker(
    rng: np.random.Generator,
    locale: str | list[str] = "en_US",
) -> Faker:
    fake = Faker(locale)
    fake.seed_instance(_faker_seed_from_rng(rng))
    return fake


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
    return [f"{prefix}-{i+1:0{width}d}" for i in range(n_rows)]


def _coerce_static(value: str, dtype: str) -> Any:
    """Cast a raw static-source string to the column's declared dtype.

    F11 (M102): on ``dtype: date`` columns, malformed ISO dates now
    raise instead of silently returning the raw string. The primary
    load-time check at ``PlotsimConfig._cross_reference_integrity``
    rejects malformed static dates before generation runs; this
    defensive raise catches the same bug class on programmatic
    ``PlotsimConfig`` construction that bypasses YAML-loading
    validators.
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


# Extended providers that fill gaps in stock Faker. The sample SaaS YAML
# reaches for ``faker.industry`` and ``faker.year`` — neither is a core Faker
# method. Rather than mutate the sample configs (out of M005 scope), we keep a
# tiny shim here. Lookup order in ``_call_faker`` is: extended provider first,
# then stock Faker's getattr.
_EXTENDED_PROVIDERS = {
    "industry": lambda fake: fake.random_element((
        "Technology", "Healthcare", "Finance", "Retail", "Manufacturing",
        "Education", "Energy", "Media", "Transportation", "Hospitality",
        "Consulting", "Real Estate",
    )),
    "year": lambda fake: fake.random_int(min=1950, max=2020),
}


# SEC-04: explicit allowlist of Faker methods the engine will dispatch to.
# Generated from the set needed by the five bundled templates plus the most
# common portfolio/persona/geo/date/text/net fakers. Anything outside this
# list is rejected with a clear error pointing the user at the contribution
# path — new additions go through review rather than ``getattr(fake, ...)``.
ALLOWED_FAKER_METHODS: frozenset[str] = frozenset({
    # Person / identity
    "name", "first_name", "last_name", "email", "phone_number",
    "user_name", "job",
    # Company / business
    "company", "bs", "catch_phrase",
    # Address / geo
    "address", "street_address", "city", "state", "country",
    "country_code", "zipcode", "postcode", "latitude", "longitude",
    # Date / time
    "date", "date_between", "date_this_decade", "date_of_birth",
    "date_time", "iso8601", "unix_time",
    # Text
    "sentence", "paragraph", "text", "word", "words",
    # Network / identifiers
    "url", "domain_name", "ipv4", "ipv6", "mac_address",
    "uuid4", "md5", "sha1", "sha256",
    # Numeric / primitives
    "random_int", "random_element", "random_elements",
    "boolean", "pybool", "pyint", "pyfloat", "pydecimal",
    # Misc common
    "currency", "currency_code", "color_name", "hex_color",
    "file_name", "file_extension", "mime_type",
    "license_plate", "vin", "ean", "ean13",
})

# SEC-04: defense-in-depth denylist. Methods here are rejected even if a
# future change pulls them into the allowlist, because they either mutate
# Faker's RNG state (breaking determinism), dispatch dynamically to bypass
# the allowlist, or amplify memory by orders of magnitude per row.
DENIED_FAKER_METHODS: frozenset[str] = frozenset({
    "seed", "seed_instance", "seed_locale",
    "add_provider", "del_provider",
    "format", "parse", "pystr_format",
    "provider", "providers",
    "binary",
})

# SEC-04: cap on length-like kwarg values. 4096 is generous for sentences,
# paragraphs, and random element counts; anything larger is almost certainly
# an amplification attempt (``binary:length:1_000_000_000`` style).
_FAKER_KWARG_LENGTH_CAP = 4096
_FAKER_LENGTH_KWARGS: frozenset[str] = frozenset({
    "length", "max_nb_chars", "nb", "min_chars", "max_chars",
    "nb_elements", "max_value", "nb_words", "nb_sentences",
})


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

    FIX-05 / MF-2. Applied to every kwarg value parsed out of a
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
                f"extended faker provider {method!r} does not accept kwargs; "
                f"got {sorted(kwargs)}"
            )
        return extended(fake)
    fn = getattr(fake, method, None)
    if fn is None or not callable(fn):
        raise ValueError(
            f"Faker has no provider method {method!r} "
            f"(source 'generated:faker.{method}')"
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
        "date_key", "date", "year", "quarter", "month", "month_name",
        "week", "period_label", "period_index",
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
        f"unsupported derived field {field!r}: expected one of "
        f"'size', 'archetype', 'name'"
    )


def _column_value_for_entity(
    col: Column,
    entity: Entity,
    entity_pk: str,
    fake: Faker,
) -> Any:
    parsed = parse_source(col.source)
    if isinstance(parsed, PKSource):
        return entity_pk
    if isinstance(parsed, StaticSource):
        values = _split_static(parsed.value)
        # Entity rows are 1:1; broadcast the first value regardless of count.
        return _coerce_static(values[0], col.dtype)
    if isinstance(parsed, DerivedSource):
        return _resolve_derived(parsed.field, entity)
    if isinstance(parsed, GeneratedSource):
        if parsed.provider == "entity_name":
            return entity.name
        raise ValueError(
            f"column {col.name!r} source {col.source!r} is not supported on "
            f"per_entity dimension tables: non-faker 'generated:' providers "
            f"other than 'entity_name' (e.g. 'timestamp', 'date_key') only "
            f"make sense on fact/event tables"
        )
    if isinstance(parsed, FakerSource):
        return _call_faker(fake, parsed.method, parsed.kwargs)
    if isinstance(parsed, FKSource):
        # dim_entity rows are 1:1 with entities, so FKs can't be populated
        # meaningfully here — the parent row this entity would point to is
        # itself. Reference tables with many-to-one FKs are built via
        # dim_reference; cross-dim FKs on per_entity tables (e.g. dim_employee
        # → dim_department) are resolved by broadcasting the first reference
        # row's PK, since reference tables are tiny and row-0 is deterministic.
        return None  # filled in by _backfill_fks
    raise ValueError(
        f"column {col.name!r} source {col.source!r} is not supported on "
        f"per_entity dimension tables"
    )


def sample_fk_values(
    column: Column,
    parent_pks: list[Any],
    n: int,
    rng: np.random.Generator,
    anchored_value: Optional[Any] = None,
) -> list[Any]:
    """Draw ``n`` FK values for ``column`` against the parent's PK list.

    FIX-04 sampler. Resolution order:

      1. ``anchored_value`` not None → broadcast it for every row. Caller
         supplies this when an Entity's ``cross_dim_fks`` pinned the value.
         No randomness consumed.
      2. ``len(parent_pks) == 1`` → broadcast that single PK. Preserves the
         pre-FIX-04 single-row behavior. No randomness consumed.
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

    Single-row parent dim → broadcast the lone PK (pre-FIX-04 behavior).
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
        if entities is not None and len(entities) == n_rows and any(
            col.name in e.cross_dim_fks for e in entities
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

    rows: list[dict[str, Any]] = []
    for entity, pk in zip(entities, ids):
        row: dict[str, Any] = {}
        for col in table_config.columns:
            row[col.name] = _column_value_for_entity(col, entity, pk, fake)
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
            ((col.name, parse_source(col.source))
             for col in table_config.columns
             if isinstance(parse_source(col.source), FKSource)),
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
    total_rows = sum(e.size for e in entities)
    ids = _make_ids(table_config.name, total_rows)

    rows: list[dict[str, Any]] = []
    cursor = 0
    for entity, (_, parent_row) in zip(entities, dim_entity.iterrows()):
        parent_pk_value = parent_row[parent_pk_column]
        for _ in range(entity.size):
            row: dict[str, Any] = {}
            local_pk = ids[cursor]
            for col in table_config.columns:
                parsed = parse_source(col.source)
                if isinstance(parsed, PKSource):
                    row[col.name] = local_pk
                elif isinstance(parsed, FKSource) and col.name == local_fk_column:
                    row[col.name] = parent_pk_value
                elif isinstance(parsed, FKSource):
                    # Unrelated FK (e.g. to a dim_reference); fill with row-0
                    # of that parent to keep the row valid. M006 may rewrite.
                    row[col.name] = None
                elif isinstance(parsed, GeneratedSource):
                    if parsed.provider == "entity_name":
                        row[col.name] = entity.name
                    else:
                        raise ValueError(
                            f"column {col.name!r} source {col.source!r} is "
                            f"not supported on sub-entity dimension tables: "
                            f"only 'entity_name' is resolved here"
                        )
                elif isinstance(parsed, FakerSource):
                    row[col.name] = _call_faker(fake, parsed.method, parsed.kwargs)
                elif isinstance(parsed, StaticSource):
                    values = _split_static(parsed.value)
                    row[col.name] = _coerce_static(values[0], col.dtype)
                elif isinstance(parsed, DerivedSource):
                    row[col.name] = _resolve_derived(parsed.field, entity)
                else:
                    raise ValueError(
                        f"column {col.name!r} source {col.source!r} is not "
                        f"supported on sub-entity dimension tables"
                    )
            rows.append(row)
            cursor += 1

    df = pd.DataFrame(rows, columns=[c.name for c in table_config.columns])
    return df


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

    rows: list[dict[str, Any]] = []
    for i in range(n_rows):
        row: dict[str, Any] = {}
        for col in table_config.columns:
            parsed = parse_source(col.source)
            if isinstance(parsed, PKSource):
                row[col.name] = ids[i]
            elif isinstance(parsed, StaticSource):
                values = _split_static(parsed.value)
                # Broadcast single-value columns; index into multi-value columns.
                pick = values[i] if len(values) > 1 else values[0]
                row[col.name] = _coerce_static(pick, col.dtype)
            elif isinstance(parsed, FakerSource):
                row[col.name] = _call_faker(fake, parsed.method, parsed.kwargs)
            elif isinstance(parsed, GeneratedSource):
                raise ValueError(
                    f"column {col.name!r} source {col.source!r} is not "
                    f"supported on reference dimension tables: use "
                    f"'generated:faker.<method>' or 'static:...' instead"
                )
            elif isinstance(parsed, FKSource):
                row[col.name] = None  # backfilled after dims are materialised
            else:
                raise ValueError(
                    f"column {col.name!r} source {col.source!r} is not "
                    f"supported on reference dimension tables"
                )
        rows.append(row)

    return pd.DataFrame(rows, columns=[c.name for c in table_config.columns])


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


def build_all_dimensions(
    config: PlotsimConfig,
    rng: np.random.Generator,
) -> dict[str, pd.DataFrame]:
    """Build every dim table in ``config.tables``, returning {name: DataFrame}.

    Build order (enforced internally):
        1. dim_date             (no deps)
        2. reference dims       (no deps)
        3. per_entity dims      (FK-backfilled from reference dims)
        4. sub-entity dims      (FK to per_entity dim)
    """
    per_entity_names = {
        t.name for t in config.tables
        if t.type == "dim" and t.grain == "per_entity"
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
                tbl, list(config.entities), rng, locale=config.locale,
            )
            dims[tbl.name] = _backfill_fks(
                df, tbl, dims, rng, entities=list(config.entities),
            )

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
