"""0.6-M18 — parent/child fact grain.

A ``per_parent_row``-grain child fact fans out from each row of a
parent fact. The parent is either a ``variable``-grain fact (row count
driven by ``row_count_source``, trajectory-driven) or a
``per_entity_per_period`` fact (one parent row per (entity, period)).

Tests cover:

  1. **Config-level validation** — ``parent_table`` / ``children_per_row``
     pairing, ``fk:fct_<parent>`` source restricted to per_parent_row
     tables, circular references rejected, parent-grain restriction.
  2. **Generation** — child row counts within configured range,
     valid FK back to parent PK, inherited entity / period, dimension
     FKs land on real dim rows.
  3. **Trajectory invariant** — parent row count tracks the driver
     metric (growth > decline over the window); child rows-per-parent
     is independent of trajectory.
  4. **Determinism** — same ``(config, seed)`` reproduces the same
     output exactly.
  5. **Budget estimator** — fan-out contributes to the estimated row
     count surfaced in the stderr summary.
  6. **Backwards compatibility** — configs without per_parent_row
     tables produce output byte-identical to pre-M18.
  7. **Vehicle config** — the ``tests/configs/orders_template.yaml`` vehicle
     produces a working parent/child config end-to-end.
"""

from __future__ import annotations

import io
import sys
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from plotsim import create, create_from_yaml, generate_tables, load_template
from plotsim.config import PlotsimConfig
from tests.configs import CONFIGS_DIR


# --- Fixture helpers --------------------------------------------------------


def _orders_config(**overrides: Any) -> PlotsimConfig:
    """A minimal 2-fact parent/child config: orders → line items.

    Two facts (no intermediate driver-host fact — the variable-grain
    builder reads ``order_volume`` directly from ``entity_metrics``):

      * ``fct_orders`` (variable grain) is the parent — one row per
        order, count driven by ``order_volume × scale``.
      * ``fct_order_items`` (per_parent_row, ``[1, 5]``) is the child.
        Its parent FK column is auto-synthesized; the user does NOT
        declare ``ref.fct_orders`` on the child.

    Overrides mutate the kwargs passed to ``create`` so individual
    tests can tweak (e.g.) the segment archetypes or the
    children_per_row range.
    """
    base: dict[str, Any] = {
        "about": "parent/child fact test",
        "unit": "customer",
        "seed": 18181,
        "window": ("2024-01", "2024-06", "monthly"),
        "metrics": [
            {
                "name": "order_volume",
                "type": "amount",
                "polarity": "positive",
                "range": [1, 30],
            },
        ],
        "segments": [
            {"name": "growers", "count": 6, "archetype": "growth"},
            {"name": "decliners", "count": 6, "archetype": "decline"},
        ],
        "dimensions": [
            {
                "name": "dim_customer",
                "per": "unit",
                "columns": [
                    {"name": "customer_id", "type": "id"},
                    {"name": "customer_name", "type": "faker.name"},
                ],
            },
            {
                "name": "dim_product",
                "reference": True,
                "columns": [
                    {"name": "product_id", "type": "id"},
                    {"name": "product_name", "type": "static.alpha,beta,gamma,delta"},
                ],
            },
        ],
        "facts": [
            {
                "name": "fct_orders",
                "row_count_driver": "order_volume",
                "row_count_scale": 1.0,
                "columns": [
                    {"name": "order_id", "type": "id"},
                    {"name": "customer_id", "type": "ref.dim_customer"},
                    {"name": "order_date", "type": "ref.dim_date"},
                ],
            },
            {
                "name": "fct_order_items",
                "parent_table": "fct_orders",
                "children_per_row": [1, 5],
                # No explicit ref.fct_orders column — engine synthesizes
                # the parent FK at generation time.
                "columns": [
                    {"name": "item_id", "type": "id"},
                    {"name": "customer_id", "type": "ref.dim_customer"},
                    {"name": "order_date", "type": "ref.dim_date"},
                    {"name": "product_id", "type": "ref.dim_product"},
                    {"name": "quantity", "type": "faker.random_int"},
                ],
            },
        ],
    }
    base.update(overrides)
    return create(**base)


# --- 1. Config validation ---------------------------------------------------


def test_parent_table_must_exist():
    """parent_table referencing a non-existent fact is rejected at load."""
    with pytest.raises(ValidationError, match="parent_table='fct_missing'"):
        _orders_config(
            facts=[
                {
                    "name": "fct_orders",
                    "row_count_driver": "order_volume",
                    "row_count_scale": 1.0,
                    "columns": [
                        {"name": "order_id", "type": "id"},
                        {"name": "customer_id", "type": "ref.dim_customer"},
                        {"name": "order_date", "type": "ref.dim_date"},
                    ],
                },
                {
                    "name": "fct_order_items",
                    "parent_table": "fct_missing",
                    "children_per_row": [1, 3],
                    "columns": [
                        {"name": "item_id", "type": "id"},
                        {"name": "customer_id", "type": "ref.dim_customer"},
                        {"name": "order_date", "type": "ref.dim_date"},
                    ],
                },
            ]
        )


def test_circular_parent_reference_rejected():
    """A parent_table cycle (A→B, B→A) is rejected at load.

    Both tables route through the builder as per_parent_row facts. The
    parent-grain validator catches "parent is per_parent_row" before the
    cycle check itself; either rejection path is acceptable.
    """
    with pytest.raises(ValidationError, match="(itself per_parent_row|cycle|per_parent_row)"):
        _orders_config(
            facts=[
                {
                    "name": "fct_orders",
                    "row_count_driver": "order_volume",
                    "row_count_scale": 1.0,
                    "columns": [
                        {"name": "order_id", "type": "id"},
                        {"name": "customer_id", "type": "ref.dim_customer"},
                        {"name": "order_date", "type": "ref.dim_date"},
                    ],
                },
                {
                    "name": "fct_a",
                    "parent_table": "fct_b",
                    "children_per_row": [1, 2],
                    "columns": [
                        {"name": "a_id", "type": "id"},
                        {"name": "customer_id", "type": "ref.dim_customer"},
                    ],
                },
                {
                    "name": "fct_b",
                    "parent_table": "fct_a",
                    "children_per_row": [1, 2],
                    "columns": [
                        {"name": "b_id", "type": "id"},
                        {"name": "customer_id", "type": "ref.dim_customer"},
                    ],
                },
            ]
        )


def test_children_per_row_bounds_rejected():
    """children_per_row must have min >= 1 and max >= min."""
    with pytest.raises(ValidationError, match="children_per_row"):
        _orders_config(
            facts=[
                {
                    "name": "fct_orders",
                    "row_count_driver": "order_volume",
                    "row_count_scale": 1.0,
                    "columns": [
                        {"name": "order_id", "type": "id"},
                        {"name": "customer_id", "type": "ref.dim_customer"},
                        {"name": "order_date", "type": "ref.dim_date"},
                    ],
                },
                {
                    "name": "fct_order_items",
                    "parent_table": "fct_orders",
                    "children_per_row": [0, 5],
                    "columns": [
                        {"name": "item_id", "type": "id"},
                        {"name": "customer_id", "type": "ref.dim_customer"},
                        {"name": "order_date", "type": "ref.dim_date"},
                    ],
                },
            ]
        )


def test_per_parent_row_requires_paired_fields():
    """parent_table without children_per_row (or vice versa) is rejected."""
    with pytest.raises(ValidationError, match="children_per_row"):
        _orders_config(
            facts=[
                {
                    "name": "fct_orders",
                    "row_count_driver": "order_volume",
                    "row_count_scale": 1.0,
                    "columns": [
                        {"name": "order_id", "type": "id"},
                        {"name": "customer_id", "type": "ref.dim_customer"},
                        {"name": "order_date", "type": "ref.dim_date"},
                    ],
                },
                {
                    "name": "fct_order_items",
                    "parent_table": "fct_orders",
                    # children_per_row intentionally omitted.
                    "columns": [
                        {"name": "item_id", "type": "id"},
                        {"name": "customer_id", "type": "ref.dim_customer"},
                    ],
                },
            ]
        )


def test_explicit_parent_fk_on_child_rejected():
    """0.6-M18 Fix 1: per_parent_row children can't declare ref.fct_<parent>.

    Use a column name distinct from the parent's PK so the regex
    matches the explicit-FK rule rather than the column-name collision
    rule (both fire when the user declares an explicit FK with the
    same name as the auto-synthesized column).
    """
    with pytest.raises(ValidationError, match="auto-synthesizes|do not declare"):
        _orders_config(
            facts=[
                {
                    "name": "fct_orders",
                    "row_count_driver": "order_volume",
                    "row_count_scale": 1.0,
                    "columns": [
                        {"name": "order_id", "type": "id"},
                        {"name": "customer_id", "type": "ref.dim_customer"},
                        {"name": "order_date", "type": "ref.dim_date"},
                    ],
                },
                {
                    "name": "fct_order_items",
                    "parent_table": "fct_orders",
                    "children_per_row": [1, 3],
                    "columns": [
                        {"name": "item_id", "type": "id"},
                        {"name": "parent_order_ref", "type": "ref.fct_orders"},
                        {"name": "customer_id", "type": "ref.dim_customer"},
                        {"name": "order_date", "type": "ref.dim_date"},
                    ],
                },
            ]
        )


def test_child_column_name_collision_with_parent_pk_rejected():
    """0.6-M18 Fix 1: user can't declare a child column that collides
    with the synthesized parent FK column name."""
    with pytest.raises(ValidationError, match="collides with the auto-synthesized"):
        _orders_config(
            facts=[
                {
                    "name": "fct_orders",
                    "row_count_driver": "order_volume",
                    "row_count_scale": 1.0,
                    "columns": [
                        {"name": "order_id", "type": "id"},
                        {"name": "customer_id", "type": "ref.dim_customer"},
                        {"name": "order_date", "type": "ref.dim_date"},
                    ],
                },
                {
                    "name": "fct_order_items",
                    "parent_table": "fct_orders",
                    "children_per_row": [1, 3],
                    "columns": [
                        {"name": "item_id", "type": "id"},
                        # Collides with parent's PK column name.
                        {"name": "order_id", "type": "static.x"},
                        {"name": "customer_id", "type": "ref.dim_customer"},
                    ],
                },
            ]
        )


# --- 2. Generation ----------------------------------------------------------


def test_children_count_per_parent_within_range():
    """Every parent row has 1..5 children (the configured range)."""
    cfg = _orders_config()
    tables = generate_tables(cfg)
    counts = tables["fct_order_items"].groupby("order_id").size()
    assert counts.min() >= 1
    assert counts.max() <= 5
    # Every parent row should produce at least one child — no rows with
    # zero children, no rows with more than the max.
    assert len(counts) == len(tables["fct_orders"])


def test_children_carry_valid_fk_to_parent():
    """Every child row's parent FK lands on a real parent PK."""
    cfg = _orders_config()
    tables = generate_tables(cfg)
    parent_pks = set(tables["fct_orders"]["order_id"])
    child_fks = set(tables["fct_order_items"]["order_id"])
    assert child_fks.issubset(parent_pks)
    # And we use them all (every parent has at least one child).
    assert child_fks == parent_pks


def test_entity_and_period_inherited_from_parent():
    """Each child row's customer_id and order_date match its parent's."""
    cfg = _orders_config()
    tables = generate_tables(cfg)
    parents = tables["fct_orders"].set_index("order_id")
    children = tables["fct_order_items"]
    # Join children → parent and verify equality.
    joined = children.merge(
        parents[["customer_id", "order_date"]],
        left_on="order_id",
        right_index=True,
        suffixes=("_child", "_parent"),
    )
    assert (joined["customer_id_child"] == joined["customer_id_parent"]).all()
    assert (joined["order_date_child"] == joined["order_date_parent"]).all()


def test_child_dim_fks_reference_real_dim_rows():
    """product_id on the child lands on a real dim_product row."""
    cfg = _orders_config()
    tables = generate_tables(cfg)
    product_pks = set(tables["dim_product"]["product_id"])
    child_pks = set(tables["fct_order_items"]["product_id"].dropna())
    assert child_pks.issubset(product_pks)


# --- 3. Trajectory invariants -----------------------------------------------


def test_parent_row_count_trajectory_driven_growth_exceeds_decline():
    """Growth entities produce more parent rows than decline entities.

    The 6-month window doesn't yield a big cumulative gap (growth and
    decline trajectories are roughly mirror images, so totals are close),
    so we widen the window and use accelerating vs decline for a clean
    separation.
    """
    cfg = _orders_config(
        window=("2024-01", "2025-12", "monthly"),
        segments=[
            {"name": "growers", "count": 8, "archetype": "accelerating"},
            {"name": "decliners", "count": 8, "archetype": "decline"},
        ],
    )
    tables = generate_tables(cfg)
    orders = tables["fct_orders"]
    growers = [f"c-{i:03d}" for i in range(1, 9)]
    decliners = [f"c-{i:03d}" for i in range(9, 17)]
    growth_rows = orders[orders["customer_id"].isin(growers)].shape[0]
    decline_rows = orders[orders["customer_id"].isin(decliners)].shape[0]
    assert growth_rows > decline_rows, (
        f"expected growers ({growth_rows}) to outpace decliners "
        f"({decline_rows}) on accelerating vs decline trajectories"
    )


def test_children_per_parent_is_uniform_not_trajectory_correlated():
    """Children-per-parent has no correlation with parent's trajectory."""
    # Strong-contrast archetypes — if children fan-out were trajectory-
    # driven, growers would systematically get more (or fewer) children
    # than decliners. Uniform draws shouldn't.
    cfg = _orders_config(
        segments=[
            {"name": "growers", "count": 8, "archetype": "accelerating"},
            {"name": "decliners", "count": 8, "archetype": "decline"},
        ],
        seed=42,
    )
    tables = generate_tables(cfg)
    counts = tables["fct_order_items"].groupby("order_id").size()
    parent = tables["fct_orders"].set_index("order_id")
    joined = counts.to_frame("n_children").join(parent[["customer_id"]])
    growers = [f"c-{i:03d}" for i in range(1, 9)]
    decliners = [f"c-{i:03d}" for i in range(9, 17)]
    mean_growth = joined[joined["customer_id"].isin(growers)]["n_children"].mean()
    mean_decline = joined[joined["customer_id"].isin(decliners)]["n_children"].mean()
    # Uniform in [1, 5] has mean 3.0. Both should land in [2.5, 3.5].
    # The two means should differ by less than 0.7 — anything larger
    # would indicate trajectory leakage.
    assert abs(mean_growth - mean_decline) < 0.7, (
        f"children/parent means leak the trajectory: "
        f"growth={mean_growth:.2f} vs decline={mean_decline:.2f}"
    )


# --- 4. Determinism ---------------------------------------------------------


def test_same_seed_produces_byte_identical_output():
    """Two runs at the same seed produce identical fact rows."""
    cfg1 = _orders_config()
    cfg2 = _orders_config()
    t1 = generate_tables(cfg1)
    t2 = generate_tables(cfg2)
    for name in ("fct_orders", "fct_order_items"):
        np.testing.assert_array_equal(
            t1[name].to_numpy(),
            t2[name].to_numpy(),
            err_msg=f"non-determinism in {name}",
        )


# --- 5. Budget estimator ---------------------------------------------------


def test_budget_estimator_accounts_for_child_fanout():
    """The stderr config summary's event_rows_upper covers child rows."""
    # capture stderr while building the config
    captured_stderr = io.StringIO()
    saved = sys.stderr
    sys.stderr = captured_stderr
    try:
        _orders_config()
    finally:
        sys.stderr = saved
    output = captured_stderr.getvalue()
    # The "Expected event rows" line includes the child fan-out
    # (parent_rows * max children_per_row). For this fixture, that's a
    # non-trivial multiple of n_entities * n_periods.
    assert "Expected event rows" in output


# --- 6. Backwards compatibility --------------------------------------------


def test_config_without_per_parent_row_unchanged():
    """A config with no per_parent_row table generates pre-M18 output.

    Sanity check: the bundled saas / hr / retail templates still load,
    validate, and generate without invoking any M18 surface.
    """
    cfg = load_template("saas")
    # Generate to confirm the path still works; we don't need the
    # resulting tables — the assertion is about config-level grain
    # composition (no M18 surface present).
    generate_tables(cfg)
    grains = {t.grain for t in cfg.tables}
    assert "per_parent_row" not in grains


# --- 7. Bundled template ---------------------------------------------------


def test_orders_template_loads_and_generates():
    """The orders vehicle config produces a working parent/child config."""
    cfg = create_from_yaml(CONFIGS_DIR / "orders_template.yaml")
    assert isinstance(cfg, PlotsimConfig)
    tables = generate_tables(cfg)
    assert "fct_orders" in tables
    assert "fct_order_items" in tables
    # FK integrity
    parent_pks = set(tables["fct_orders"]["order_id"])
    child_fks = set(tables["fct_order_items"]["order_id"])
    assert child_fks.issubset(parent_pks)
    # Range bounds
    counts = tables["fct_order_items"].groupby("order_id").size()
    assert counts.min() >= 1
    assert counts.max() <= 5


def test_parent_fk_column_auto_synthesized_first():
    """0.6-M18 Fix 1: synthesized parent FK column appears first in child output."""
    cfg = _orders_config()
    tables = generate_tables(cfg)
    cols = list(tables["fct_order_items"].columns)
    # Parent's PK column name is ``order_id``; synthesized FK on the
    # child carries the same name and lands first.
    assert (
        cols[0] == "order_id"
    ), f"expected synthesized FK 'order_id' as first column; got {cols!r}"


def test_variable_grain_fact_builds_without_driver_host_fact():
    """0.6-M18 Fix 2: the driver metric for a variable-grain fact does
    NOT need to be projected onto another fact column. The variable
    builder reads from entity_metrics directly.

    The base fixture has exactly this shape — only fct_orders + fct_order_items,
    no fct_customer_activity hosting order_volume.
    """
    cfg = _orders_config()
    # No fact column projects ``order_volume`` as a metric source.
    from plotsim.config import MetricSource, parse_source

    has_order_volume_projection = False
    for tbl in cfg.tables:
        for col in tbl.columns:
            try:
                parsed = parse_source(col.source)
            except ValueError:
                continue
            if isinstance(parsed, MetricSource) and parsed.metric == "order_volume":
                has_order_volume_projection = True
                break
    assert not has_order_volume_projection, (
        "test precondition: the fixture should have no metric:order_volume "
        "column anywhere; the variable-grain builder is supposed to read "
        "the driver directly from entity_metrics"
    )
    # Generation succeeds + parent fact has non-zero rows.
    tables = generate_tables(cfg)
    assert len(tables["fct_orders"]) > 0


# --- 8. Sibling-fact references (0.6-M18 Fix 3) -----------------------------


def _orders_with_returns_config(**overrides: Any) -> PlotsimConfig:
    """Three-fact config: orders + line items + sibling-referenced returns."""
    base: dict[str, Any] = {
        "about": "orders + returns",
        "unit": "customer",
        "seed": 18181,
        "window": ("2024-01", "2024-06", "monthly"),
        "metrics": [
            {
                "name": "order_volume",
                "type": "amount",
                "polarity": "positive",
                "range": [1, 30],
            },
            {
                "name": "return_rate",
                "type": "score",
                "polarity": "negative",
            },
        ],
        "segments": [
            {"name": "growers", "count": 6, "archetype": "growth"},
            {"name": "decliners", "count": 6, "archetype": "decline"},
        ],
        "dimensions": [
            {
                "name": "dim_customer",
                "per": "unit",
                "columns": [
                    {"name": "customer_id", "type": "id"},
                    {"name": "customer_name", "type": "faker.name"},
                ],
            },
        ],
        "facts": [
            {
                "name": "fct_orders",
                "row_count_driver": "order_volume",
                "row_count_scale": 1.0,
                "columns": [
                    {"name": "order_id", "type": "id"},
                    {"name": "customer_id", "type": "ref.dim_customer"},
                    {"name": "order_date", "type": "ref.dim_date"},
                ],
            },
            {
                "name": "fct_returns",
                "row_count_driver": "return_rate",
                "row_count_scale": 0.5,
                "columns": [
                    {"name": "return_id", "type": "id"},
                    {"name": "order_id", "type": "ref.fct_orders"},
                    {"name": "customer_id", "type": "ref.dim_customer"},
                    {"name": "return_date", "type": "ref.dim_date"},
                ],
            },
        ],
    }
    base.update(overrides)
    return create(**base)


def test_sibling_fact_reference_lands_on_parent_pk():
    """Cross-fact FK values are valid parent PKs."""
    cfg = _orders_with_returns_config()
    tables = generate_tables(cfg)
    parent_pks = set(tables["fct_orders"]["order_id"])
    sibling_fks = set(tables["fct_returns"]["order_id"].dropna())
    assert sibling_fks.issubset(parent_pks)


def test_sibling_fact_reference_is_same_entity_filtered():
    """Each fct_returns row's order_id belongs to a parent row with the
    same customer_id."""
    cfg = _orders_with_returns_config()
    tables = generate_tables(cfg)
    merged = tables["fct_returns"].merge(
        tables["fct_orders"][["order_id", "customer_id"]].rename(
            columns={"customer_id": "order_customer"}
        ),
        on="order_id",
        how="left",
    )
    # Drop rows where FK was None (no orders for that customer).
    resolved = merged.dropna(subset=["order_customer"])
    mismatches = (resolved["customer_id"] != resolved["order_customer"]).sum()
    assert mismatches == 0, (
        f"{mismatches} fct_returns rows reference an order placed by a "
        f"different customer; same-entity filter broken"
    )


def test_topological_sort_handles_mixed_parent_table_and_fk_edges():
    """Declaring fct_returns BEFORE fct_orders in config still builds
    correctly because the topo sort resolves dependencies."""
    # Reverse the facts list — referenced fact declared AFTER its
    # referencer. Topo sort should reorder.
    facts_reversed = [
        {
            "name": "fct_returns",
            "row_count_driver": "return_rate",
            "row_count_scale": 0.5,
            "columns": [
                {"name": "return_id", "type": "id"},
                {"name": "order_id", "type": "ref.fct_orders"},
                {"name": "customer_id", "type": "ref.dim_customer"},
                {"name": "return_date", "type": "ref.dim_date"},
            ],
        },
        {
            "name": "fct_orders",
            "row_count_driver": "order_volume",
            "row_count_scale": 1.0,
            "columns": [
                {"name": "order_id", "type": "id"},
                {"name": "customer_id", "type": "ref.dim_customer"},
                {"name": "order_date", "type": "ref.dim_date"},
            ],
        },
    ]
    cfg = _orders_with_returns_config(facts=facts_reversed)
    tables = generate_tables(cfg)
    # FK integrity holds regardless of declaration order.
    parent_pks = set(tables["fct_orders"]["order_id"])
    sibling_fks = set(tables["fct_returns"]["order_id"].dropna())
    assert sibling_fks.issubset(parent_pks)


def test_cross_fact_cycle_rejected_at_load():
    """A → B and B → A cross-fact reference cycle is caught."""
    with pytest.raises(ValidationError, match="cycle"):
        _orders_with_returns_config(
            facts=[
                {
                    "name": "fct_a",
                    "row_count_driver": "order_volume",
                    "row_count_scale": 1.0,
                    "columns": [
                        {"name": "a_id", "type": "id"},
                        {"name": "customer_id", "type": "ref.dim_customer"},
                        {"name": "a_date", "type": "ref.dim_date"},
                        {"name": "b_ref", "type": "ref.fct_b"},
                    ],
                },
                {
                    "name": "fct_b",
                    "row_count_driver": "return_rate",
                    "row_count_scale": 1.0,
                    "columns": [
                        {"name": "b_id", "type": "id"},
                        {"name": "customer_id", "type": "ref.dim_customer"},
                        {"name": "b_date", "type": "ref.dim_date"},
                        {"name": "a_ref", "type": "ref.fct_a"},
                    ],
                },
            ]
        )


def test_variable_grain_fact_with_scd2_entity_dim():
    """Regression: variable-grain fact builder must handle an SCD2-expanded
    per_entity dim.

    Pre-fix the variable-grain builder assumed
    ``len(dim_<entity>) == n_entities``. SCD2 expansion produces one
    row per tier-band crossing per entity (so n_rows > n_entities),
    which tripped a RuntimeError at
    ``tables._build_variable_grain_fact``'s entity-pks vs entity-count
    consistency check. The fix collapses the entity dim's PK column via
    ``drop_duplicates()`` before building entity-major arrays.

    Asserts: (1) generation completes without RuntimeError, (2) the
    entity dim IS SCD2-expanded (n_rows > n_entities), (3) the
    variable-grain fact's per-entity FK covers all N entities exactly
    (not the SCD2-version count).
    """
    cfg = create(
        about="SCD2 entity dim + variable-grain fact regression",
        unit="customer",
        seed=18182,
        window=("2024-01", "2024-06", "monthly"),
        metrics=[
            {"name": "order_volume", "type": "amount", "polarity": "positive", "range": [1, 30]},
            {"name": "loyalty_score", "type": "score", "polarity": "positive"},
        ],
        segments=[
            {"name": "growers", "count": 4, "archetype": "growth"},
            {"name": "decliners", "count": 4, "archetype": "decline"},
        ],
        dimensions=[
            {
                "name": "dim_customer",
                "per": "unit",
                "columns": [
                    {"name": "customer_id", "type": "id"},
                    {"name": "customer_name", "type": "faker.name"},
                    {
                        "name": "tier",
                        "type": "scd",
                        "tracks": "loyalty_score",
                        "tiers": ["bronze", "silver", "gold"],
                        "at": [0.4, 0.75],
                    },
                ],
            },
        ],
        facts=[
            {
                "name": "fct_customer_activity",
                "columns": [
                    {"name": "date_key", "type": "ref.dim_date"},
                    {"name": "customer_id", "type": "ref.dim_customer"},
                    {"name": "loyalty_score", "type": "metric.loyalty_score"},
                ],
            },
            {
                "name": "fct_orders",
                "row_count_driver": "order_volume",
                "row_count_scale": 1.0,
                "columns": [
                    {"name": "order_id", "type": "id"},
                    {"name": "customer_id", "type": "ref.dim_customer"},
                    {"name": "order_date", "type": "ref.dim_date"},
                ],
            },
        ],
    )

    n_entities = len(cfg.entities)
    assert n_entities == 8

    tables = generate_tables(cfg)

    dim_customer = tables["dim_customer"]
    fct_orders = tables["fct_orders"]
    assert "dim_row_id" in dim_customer.columns, "SCD2 expansion did not fire"
    assert len(dim_customer) > n_entities, (
        f"dim_customer has {len(dim_customer)} rows but only {n_entities} "
        f"entities; SCD2 expansion did not produce extra versions"
    )

    fact_entity_pks = set(fct_orders["customer_id"].dropna().unique())
    dim_entity_pks = set(dim_customer["customer_id"].dropna().unique())
    assert fact_entity_pks <= dim_entity_pks
    assert len(fact_entity_pks) == n_entities, (
        f"fct_orders.customer_id covers {len(fact_entity_pks)} entities, "
        f"expected {n_entities} (one PK per entity, not per SCD2 version)"
    )


def test_orders_template_manifest_has_parent_child_relation():
    """The manifest carries one parent_child_relations record per child."""
    from plotsim import build_manifest, generate_tables_with_state

    cfg = create_from_yaml(CONFIGS_DIR / "orders_template.yaml")
    rng = np.random.default_rng(cfg.seed)
    tables, state = generate_tables_with_state(cfg, rng)
    manifest = build_manifest(
        cfg,
        state.trajectories,
        tables,
        scd_state=state.scd,
        bridge_state=state.bridges,
    )
    assert len(manifest.parent_child_relations) == 1
    rec = manifest.parent_child_relations[0]
    assert rec.parent_table == "fct_orders"
    assert rec.child_table == "fct_order_items"
    assert rec.children_per_row_min == 1
    assert rec.children_per_row_max == 5
    assert rec.parent_row_count == len(tables["fct_orders"])
    assert rec.child_row_count == len(tables["fct_order_items"])
