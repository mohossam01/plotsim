"""0.6-M19 Fix 8 — PK prefix distinguishability.

Pre-fix, the engine derived a single-character prefix from each
sequential-PK table's name (first char after stripping the
``dim_`` / ``fct_`` / ``evt_`` type prefix). Two tables sharing a
first character — ``fct_orders`` and ``fct_order_items``,
``dim_company`` and ``evt_churn``, etc. — produced colliding PKs
(both ``o-00001`` / both ``c-00001``), which broke joins on those
PK values and made the output ambiguous.

This module covers the post-fix surface:

  * ``Table.pk_prefix`` is an optional explicit override.
  * ``PlotsimConfig._resolve_pk_prefixes`` auto-derives a unique
    prefix per table: first char when there's no collision (the
    pre-fix behavior, preserved byte-for-byte on non-colliding
    configs), the full stripped name when colliding tables would
    otherwise share a first char.
  * The resolved map is read via ``config.pk_prefix_for(table_name)``
    by every sequential-PK builder (dims, variable-grain facts,
    per_parent_row child facts, events).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from plotsim import generate_tables, load_template
from plotsim.config import (
    Archetype,
    Column,
    CurveSegment,
    Domain,
    Entity,
    Metric,
    OutputConfig,
    PlotsimConfig,
    SurrogateKeyWarning,
    Table,
    TimeWindow,
)


# --- Fixture helpers --------------------------------------------------------


def _flat_archetype() -> Archetype:
    return Archetype(
        name="flat",
        label="flat",
        description="constant 0.5 plateau",
        curve_segments=[
            CurveSegment(
                curve="plateau",
                params={"level": 0.5},
                start_pct=0.0,
                end_pct=1.0,
            ),
        ],
    )


def _minimal_metric() -> Metric:
    return Metric(
        name="m",
        label="m",
        distribution="normal",
        params={"mu": 1.0, "sigma": 0.1},
        polarity="positive",
    )


def _make_config(extra_tables: list[Table], entities: list[Entity] | None = None) -> PlotsimConfig:
    """Build a small valid config with one per_entity dim + dim_date +
    one fct_m + any caller-supplied extra tables."""
    if entities is None:
        entities = [Entity(name="e1", archetype="flat", size=1)]
    dim_date = Table(
        name="dim_date",
        type="dim",
        grain="per_period",
        primary_key="date_key",
        columns=[
            Column(name="date_key", dtype="id", source="pk"),
            Column(name="date", dtype="date", source="generated:date_key"),
        ],
    )
    dim_entity = Table(
        name="dim_entity",
        type="dim",
        grain="per_entity",
        primary_key="entity_id",
        columns=[Column(name="entity_id", dtype="id", source="pk")],
    )
    fct = Table(
        name="fct_m",
        type="fact",
        grain="per_entity_per_period",
        primary_key=["date_key", "entity_id"],
        foreign_keys=["dim_date.date_key", "dim_entity.entity_id"],
        columns=[
            Column(name="date_key", dtype="id", source="fk:dim_date.date_key"),
            Column(name="entity_id", dtype="id", source="fk:dim_entity.entity_id"),
            Column(name="m", dtype="float", source="metric:m"),
        ],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SurrogateKeyWarning)
        return PlotsimConfig(
            domain=Domain(
                name="t",
                description="t",
                entity_type="entity",
                entity_label="Entities",
            ),
            time_window=TimeWindow(
                start="2024-01",
                end="2024-04",
                granularity="monthly",
            ),
            seed=42,
            metrics=[_minimal_metric()],
            archetypes=[_flat_archetype()],
            entities=entities,
            tables=[dim_date, dim_entity, fct] + extra_tables,
            output=OutputConfig(format="csv", directory="out/pk_prefix"),
        )


# --- Auto-derivation: single-char prefix preserved when no collision -------


def test_single_letter_prefix_kept_when_no_collision():
    """Backward compat: a config where every table's first stripped
    character is unique keeps the pre-fix single-character prefixes.
    """
    dim_plan = Table(
        name="dim_plan",
        type="dim",
        grain="per_reference",
        primary_key="plan_id",
        columns=[
            Column(name="plan_id", dtype="id", source="pk"),
            Column(name="plan_name", dtype="string", source="static:starter,growth"),
        ],
    )
    cfg = _make_config([dim_plan])
    assert cfg.pk_prefix_for("dim_entity") == "e"
    assert cfg.pk_prefix_for("dim_plan") == "p"


# --- Auto-derivation: collision promotes both to full stripped name --------


def test_first_letter_collision_promotes_to_stripped_names():
    """Two tables sharing a first character — ``dim_company`` and
    ``dim_customer`` both stripped to a ``c``-prefix candidate — get
    auto-promoted to their full stripped names so PKs disambiguate.
    """
    dim_company = Table(
        name="dim_company",
        type="dim",
        grain="per_reference",
        primary_key="company_id",
        columns=[
            Column(name="company_id", dtype="id", source="pk"),
            Column(name="company_name", dtype="string", source="static:acme,globex"),
        ],
    )
    dim_customer = Table(
        name="dim_customer",
        type="dim",
        grain="per_reference",
        primary_key="customer_id",
        columns=[
            Column(name="customer_id", dtype="id", source="pk"),
            Column(name="customer_name", dtype="string", source="static:alice,bob"),
        ],
    )
    cfg = _make_config([dim_company, dim_customer])
    assert cfg.pk_prefix_for("dim_company") == "company"
    assert cfg.pk_prefix_for("dim_customer") == "customer"


def test_event_and_dim_collision_promotes_both():
    """``dim_company`` + ``evt_churn`` (both stripped → ``c``) both
    promote — the M19 fix isn't dim-only, it covers facts and events
    too."""
    dim_company = Table(
        name="dim_company",
        type="dim",
        grain="per_reference",
        primary_key="company_id",
        columns=[
            Column(name="company_id", dtype="id", source="pk"),
            Column(name="company_name", dtype="string", source="static:acme"),
        ],
    )
    evt_churn = Table(
        name="evt_churn",
        type="event",
        grain="per_entity_per_period",
        primary_key="event_id",
        foreign_keys=["dim_date.date_key", "dim_entity.entity_id"],
        columns=[
            Column(name="event_id", dtype="id", source="pk"),
            Column(name="date_key", dtype="id", source="fk:dim_date.date_key"),
            Column(name="entity_id", dtype="id", source="fk:dim_entity.entity_id"),
        ],
    )
    cfg = _make_config([dim_company, evt_churn])
    assert cfg.pk_prefix_for("dim_company") == "company"
    assert cfg.pk_prefix_for("evt_churn") == "churn"


# --- Explicit override on Table.pk_prefix ----------------------------------


def test_explicit_pk_prefix_overrides_auto_derive():
    dim_company = Table(
        name="dim_company",
        type="dim",
        grain="per_reference",
        primary_key="company_id",
        pk_prefix="co",
        columns=[
            Column(name="company_id", dtype="id", source="pk"),
            Column(name="company_name", dtype="string", source="static:acme"),
        ],
    )
    cfg = _make_config([dim_company])
    assert cfg.pk_prefix_for("dim_company") == "co"


def test_explicit_pk_prefix_format_validation():
    """``Table.pk_prefix`` must start with a letter, 1-12 chars,
    alphanumeric+underscore."""
    with pytest.raises(ValueError, match="pk_prefix"):
        Table(
            name="dim_x",
            type="dim",
            grain="per_entity",
            primary_key="x_id",
            pk_prefix="9bad",  # starts with digit
            columns=[Column(name="x_id", dtype="id", source="pk")],
        )
    with pytest.raises(ValueError, match="pk_prefix"):
        Table(
            name="dim_x",
            type="dim",
            grain="per_entity",
            primary_key="x_id",
            pk_prefix="has space",
            columns=[Column(name="x_id", dtype="id", source="pk")],
        )


def test_explicit_pk_prefix_collision_rejected():
    """Two tables with explicit ``pk_prefix`` set to the same value
    must fail at config load — the validator can't auto-disambiguate
    when the user has insisted on both."""
    dim_a = Table(
        name="dim_alpha",
        type="dim",
        grain="per_reference",
        primary_key="alpha_id",
        pk_prefix="ax",
        columns=[Column(name="alpha_id", dtype="id", source="pk")],
    )
    dim_b = Table(
        name="dim_aleph",
        type="dim",
        grain="per_reference",
        primary_key="aleph_id",
        pk_prefix="ax",
        columns=[Column(name="aleph_id", dtype="id", source="pk")],
    )
    with pytest.raises(ValueError, match="resolves to multiple tables"):
        _make_config([dim_a, dim_b])


# --- End-to-end: orders template (the canonical collision case) ------------


def test_orders_template_produces_distinguishable_pks():
    """``orders`` template has fct_orders + fct_order_items (both
    stripped → ``o``) plus fct_returns (``r``). The first two must
    get distinguishable PKs; fct_returns keeps its single ``r``."""
    cfg = load_template("orders")
    tables = generate_tables(cfg, np.random.default_rng(cfg.seed))
    orders_pks = tables["fct_orders"]["order_id"].tolist()
    items_pks = tables["fct_order_items"]["item_id"].tolist()
    returns_pks = tables["fct_returns"]["return_id"].tolist()
    # No overlap between orders and order_items PKs (the M19 bug).
    assert set(orders_pks).isdisjoint(
        items_pks
    ), "fct_orders and fct_order_items PKs overlap — auto-disambiguation broke"
    # Distinct prefix shapes confirm the auto-derive promotion.
    assert orders_pks[0].startswith("orders-")
    assert items_pks[0].startswith("order_items-")
    # fct_returns stayed single-letter (no collision).
    assert returns_pks[0].startswith("r-")


# --- End-to-end: collision-free config preserves single-letter output ------


def test_collision_free_config_preserves_single_letter_pks_end_to_end():
    """Generate from a config with no first-char collisions and
    confirm every dim and event PK keeps the pre-M19 single-letter
    shape — the backward-compat invariant the mission required.
    """
    dim_plan = Table(
        name="dim_plan",
        type="dim",
        grain="per_reference",
        primary_key="plan_id",
        columns=[
            Column(name="plan_id", dtype="id", source="pk"),
            Column(name="plan_name", dtype="string", source="static:starter,growth"),
        ],
    )
    cfg = _make_config([dim_plan])
    tables = generate_tables(cfg, np.random.default_rng(cfg.seed))
    assert tables["dim_entity"]["entity_id"].iloc[0].startswith("e-")
    assert tables["dim_plan"]["plan_id"].iloc[0].startswith("p-")
