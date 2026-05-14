# Parent/child fact grain

> Generate a parent fact (one row per discrete instance, count
> driven by trajectory) with detail child rows fanning out from
> each parent row, AND sibling facts that reference the parent
> by per-entity FK draws. Models the universal header/detail
> pattern (orders + line items, claims + claim lines,
> shipments + parcels) plus the "some-orders-have-returns"
> reference pattern.

---

## What it does

Two new fact grains compose into a header/detail star, and a
third pattern (sibling-fact reference) lets independent facts
point at the parent:

| Pattern | Cardinality | Mechanism |
|---|---|---|
| Header / detail | Every parent has 1..N children (deterministic fan-out) | `parent_table` + `children_per_row` on a `per_parent_row` child |
| Sibling reference | Some rows of fact B reference fact A by same-entity FK draw | `row_count_driver` on B + a `ref.<fact_a>` column |

The trajectory-first signal flows through each fact's row count
(growth-archetype entities generate more parent rows than decline
entities, because their driver metric trends up). For header /
detail children, each child row inherits its parent's entity and
period; for sibling references, each row draws a parent PK from
its own entity's pool — the on-disk star is referentially intact.

Both patterns follow the **bridge precedent**: the relationship
is declared once (via `parent_table` or `ref.<fact>`) and the
engine handles FK column emission and resolution at generation
time. `per_parent_row` children get an auto-synthesized FK
column named after the parent's PK; sibling-fact references
draw same-entity-filtered values into their declared column.

---

## When to use which

- **Orders / line items.** Every order has 1..N items. Use
  header / detail (per_parent_row child).
- **Orders / multiple shipments per order.** Every order is
  shipped in 1..N parcels. Use header / detail.
- **Orders / returns.** Some orders are returned. Use sibling
  reference (independent variable-grain fact with `ref.fct_orders`).
- **Orders / shipments where some orders haven't shipped yet.**
  Same as returns — sibling reference.

If your domain is a flat per-(entity, period) summary, the
default `per_entity_per_period` grain is the right tool — no
parent/child wiring needed.

---

## How to enable: header / detail

The pattern needs **two facts**. The driver metric is declared
in `metrics:` but does NOT need a fact-column projection — the
engine reads it directly from the per-entity-per-period metric
layer:

=== "YAML"

    ```yaml
    metrics:
      - { name: order_volume, type: amount, polarity: positive, range: [1, 30] }

    facts:
      - name: fct_orders
        row_count_driver: order_volume
        row_count_scale: 1.2
        columns:
          - { name: order_id,    type: id }
          - { name: customer_id, type: ref.dim_customer }
          - { name: order_date,  type: ref.dim_date }

      # The parent FK column (named `order_id`, matching fct_orders' PK)
      # is auto-synthesized by the engine. Do NOT declare it here.
      - name: fct_order_items
        parent_table: fct_orders
        children_per_row: [1, 5]
        columns:
          - { name: item_id,     type: id }
          - { name: customer_id, type: ref.dim_customer }
          - { name: order_date,  type: ref.dim_date }
          - { name: product_id,  type: ref.dim_product }
          - { name: quantity,    type: faker.random_int }
    ```

=== "Python"

    ```python
    from plotsim import create

    cfg = create(
        # ... about / unit / window / segments / dimensions ...
        metrics=[
            {"name": "order_volume", "type": "amount",
             "polarity": "positive", "range": [1, 30]},
        ],
        facts=[
            {
                "name": "fct_orders",
                "row_count_driver": "order_volume",
                "row_count_scale": 1.2,
                "columns": [
                    {"name": "order_id",    "type": "id"},
                    {"name": "customer_id", "type": "ref.dim_customer"},
                    {"name": "order_date",  "type": "ref.dim_date"},
                ],
            },
            {
                "name": "fct_order_items",
                "parent_table": "fct_orders",
                "children_per_row": [1, 5],
                # Engine auto-synthesizes the parent FK column.
                "columns": [
                    {"name": "item_id",     "type": "id"},
                    {"name": "customer_id", "type": "ref.dim_customer"},
                    {"name": "order_date",  "type": "ref.dim_date"},
                    {"name": "product_id",  "type": "ref.dim_product"},
                    {"name": "quantity",    "type": "faker.random_int"},
                ],
            },
        ],
    )
    ```

---

## How to enable: sibling-fact reference

A sibling fact is its own variable-grain fact with its own
trajectory-driven row count, and one of its columns is a
`ref.<other_fact>` FK. The engine draws same-entity-filtered
PK values from the referenced fact:

=== "YAML"

    ```yaml
    metrics:
      - { name: order_volume, type: amount, polarity: positive, range: [1, 30] }
      - { name: return_rate,  type: score, polarity: negative }

    facts:
      - name: fct_orders
        row_count_driver: order_volume
        row_count_scale: 1.2
        columns:
          - { name: order_id,    type: id }
          - { name: customer_id, type: ref.dim_customer }
          - { name: order_date,  type: ref.dim_date }

      - name: fct_returns
        row_count_driver: return_rate
        row_count_scale: 0.6
        columns:
          - { name: return_id,     type: id }
          - { name: order_id,      type: ref.fct_orders }   # sibling FK
          - { name: customer_id,   type: ref.dim_customer }
          - { name: return_date,   type: ref.dim_date }
          - { name: return_reason, type: faker.word }
    ```

The engine guarantees:

- Every `fct_returns.order_id` value lands on a real `fct_orders.order_id`.
- Each return references an order placed **by the same customer** —
  same-entity filtered, so a return for customer X never references
  customer Y's order.
- The engine builds `fct_orders` before `fct_returns` regardless of
  the declaration order in the config (topological sort).

If a customer has zero parent rows in the period being processed,
the FK value lands as `null` rather than raising. Use that
behavior to model "this customer hasn't placed any orders, but
the trajectory drew a return event anyway" edge cases — or
configure `return_rate` to ensure your customers all have orders
before returns can fire.

---

## What you get on disk

A run with 18 customers across 6 months produces something like:

```
fct_orders.csv          ~200 rows (trajectory-driven row count)
fct_order_items.csv     ~570 rows (parent rows × uniform(1, 5))
fct_returns.csv          ~30 rows (independent variable-grain fact)
```

Every `fct_order_items.order_id` value matches an
`fct_orders.order_id` (FK integrity enforced by the auto-
synthesis path). Every `fct_returns.order_id` value matches an
`fct_orders.order_id` and the customer_id columns agree
(same-entity filter). Growth-archetype customers generate more
`fct_orders` rows and consequently more `fct_returns` rows
(growth metric correlates with returns volume only if you
configure return_rate that way; in the bundled `orders`
template, return_rate opposes loyalty_score so decliners
return more often).

---

## What's available on the child (per_parent_row)

| Column source | Result on the child |
|---|---|
| `pk` | Auto-generated sequential ID (e.g. `i-00001`) |
| `ref.<dim>` | Independent draw from the dim's PK column |
| `ref.dim_date` / `ref.dim_customer` (per-entity) | Inherited from parent — child shares parent's coordinates |
| `static:<value>` | Literal cell value |
| `derived:entity_id` / `derived:date_key` | Inherited from parent |
| `generated:date_key` / `timestamp` / `period_label` | Resolved from parent's period |
| `generated:faker.<method>` | Independent per-row Faker draw |

**The parent FK column is auto-synthesized** — its name matches
the parent fact's PK column verbatim (e.g. `order_id`) and it
appears first in column order. Do NOT declare it explicitly;
the engine rejects `ref.<parent_fact>` on a per_parent_row child
at config load.

**Not supported on children:**

- `metric:<name>` — child rows don't go through the trajectory
  engine. Per-line metrics (quantity, unit_price) come from
  `faker.*` or `static:` sources.
- `threshold:` / `proportional:` / `lag:` — trajectory-coupled.
- `narrative:` / `text:bucket:` — same reason.
- `pool:` / `scd_type2` — dim-only sources.

---

## What's available on the parent (variable-grain fact)

Variable-grain fact tables (`row_count_driver` + `row_count_scale`
set) support the same column sources as event tables:

- `pk`, `ref.dim_<x>`, `static:`, `derived:`, `generated:*`,
  `faker.*`.
- **Not** `metric:` — per-instance metric values are ambiguous
  when multiple rows share one trajectory position. Per-instance
  numbers belong on the child detail or on a sibling fact.

The driver metric (named in `row_count_driver`) is declared in
the top-level `metrics:` block. It does NOT need a fact-column
projection — the variable-grain builder reads its per-entity
series directly from the metric layer.

---

## What's available on a sibling-referencing fact

A sibling fact (e.g. `fct_returns`) is just another variable-grain
fact. The only special column is `ref.<other_fact>`:

- Source: `ref.<other_fact>` resolves to `fk:<other_fact>.<pk>`.
- Resolution: per-row stochastic same-entity draw from the
  referenced fact's PK column (filtered to the current row's
  entity).
- Empty pool (entity has no parent rows): cell lands as `null`.

The referencing fact MUST declare a per_entity FK column of its
own — otherwise same-entity filtering has nothing to anchor on,
and the engine rejects the config at load.

---

## Engine-enforced rules at config load

- Every `per_parent_row` child has a valid `parent_table` pointing
  at an existing fact with grain `variable` or
  `per_entity_per_period`.
- `children_per_row` is an inclusive `(min, max)` with `min ≥ 1`
  and `max ≥ min`.
- A `per_parent_row` child must NOT declare an explicit
  `ref.<parent_fact>` column — the engine synthesizes it.
- A `per_parent_row` child column name must not collide with the
  parent's PK column name (synthesized column would overwrite).
- The combined dependency graph (parent_table edges +
  cross-fact `ref.fct_*` edges) is acyclic. The engine rejects
  A→B / B→A cycles regardless of which edge type forms them.
- Sibling-fact references (`ref.<other_fact>` on a non-
  per_parent_row table) require the referencing fact to have a
  per_entity FK column for same-entity filtering.

---

## Limits

- **Single level of parent/child only.** Grandchild chains
  (`fct_orders → fct_order_items → fct_returns_per_item`) are
  not supported in the current release.
- **Single-column parent PKs only.** Composite parent PKs would
  need a multi-column FK on the child; out of scope.
- **No metric columns on variable-grain facts.** Per-instance
  metrics belong on the child detail or on a sibling fact.
- **No CDC / SCD Type 2 / quality injection on child tables.**
  The parent fact carries the temporal-audit story; child rows
  are immutable detail.
- **Same-entity filter only.** Cross-fact references always
  filter to the same entity (a return for customer X references
  one of X's orders). Cross-entity references are not in scope.
- **No temporal-coherence enforcement.** A sibling row's
  reference is drawn without regard to the row's own period —
  a return at month 2 may reference an order at month 5. Model
  the timing of returns via the driver metric if precision
  matters.

---

## Related

- [Output formats](./output-formats.md) — CSV, Parquet, JSONL, SQL
  dump all handle parent/child + sibling facts the same way.
- [Schema guide](./schema-guide.md) — the full column-source
  vocabulary.
- [CDC fact-side](./cdc-facts.md) — opt-in audit-column pattern
  on parent facts; combines cleanly with `cdc: true`.
