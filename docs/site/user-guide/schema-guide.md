# Schema guide

> When to let plotsim auto-build your tables and when to take over.
> How to design dimensions, facts, events, bridges, and SCD columns.

---

## Auto-schema vs explicit schema

Two ways to declare a config:

### Auto-schema — the default

If you skip the `dimensions` / `facts` / `events` blocks entirely,
plotsim emits a sensible default star schema:

- `dim_date` — date spine for the configured window
- `dim_<unit>` — one row per entity
- `fct_<metric>` — one fact table per metric, with FKs to `dim_<unit>`
  and `dim_date`

```yaml
about: Subscription customers
unit: customer
window: ["2024-01", "2024-12", "monthly"]

metrics:
  - { name: engagement, type: score,  polarity: positive }
  - { name: mrr,        type: amount, polarity: positive, range: [10, 5000] }

segments:
  - { name: growers,   count: 30, archetype: growth }
  - { name: decliners, count: 20, archetype: decline }
```

Run that and you get:

```
dim_date.csv
dim_customer.csv
fct_engagement.csv
fct_mrr.csv
```

No schema declaration needed. This is the fastest path from prompt to
data.

### Explicit schema

When you want named columns, multiple metrics in one fact table, custom
event tables, or bridge tables, declare the schema yourself:

```yaml
dimensions:
  - name: dim_customer
    columns:
      - { name: customer_id,   type: id }
      - { name: company_name,  type: faker.company }
      - { name: industry,      type: pool.industry }

facts:
  - name: fct_engagement
    metrics: [engagement, mrr]
    columns:
      - { name: customer_id, type: ref.dim_customer }
      - { name: date_key,    type: ref.dim_date }

events:
  - name: evt_login
    trigger: proportional
    driver: engagement
    scale: 5
    columns:
      - { name: customer_id, type: ref.dim_customer }
      - { name: timestamp,   type: timestamp }
```

`dim_date` is auto-generated even when you supply explicit dims; you
don't have to declare it (though you can to add custom columns).

When you declare *any* schema block, plotsim uses *only* what you
declared — there's no merging of auto-defaults into your declarations.

---

## Designing dimensions

Dim tables hold static (or slowly-changing) attributes. There are three
kinds:

| Kind | When | Cardinality |
|---|---|---|
| Per-entity dim | One row per entity | `count(entities)` |
| Reference dim | Lookup table (industries, plans, departments) | Whatever you put in it |
| Sub-entity dim | Multiple rows per entity (e.g. users per company) | `count(entities) × count` |

### Per-entity dim

The default. One row per entity, keyed off the entity's PK.

```yaml
dimensions:
  - name: dim_customer
    columns:
      - { name: customer_id,   type: id }
      - { name: company_name,  type: faker.company }
      - { name: signup_date,   type: faker.date_between }
      - { name: cohort_size,   type: segment.count }
      - { name: industry,      type: pool.industry }
```

`pool.{attr}` columns pull from per-segment attributes — see
[Column types](../column-types.md#poolattr).

### Reference dim

Set `reference: true` for static lookup tables.

```yaml
dimensions:
  - name: dim_plan
    reference: true
    columns:
      - { name: plan_id,    type: id }
      - { name: plan_name,  type: static.starter }
      - { name: monthly_fee, type: static.49 }
```

Reference dims aren't tied to entities or periods. They're typically
small (3–10 rows) and exist to enable FK joins from facts.

### Sub-entity dim

Set `count: <int>` for sub-entity expansion. Each entity produces `count`
rows.

```yaml
dimensions:
  - name: dim_user
    count: 5     # 5 users per customer
    columns:
      - { name: user_id,     type: id }
      - { name: customer_id, type: ref.dim_customer }
      - { name: full_name,   type: faker.name }
```

Useful for one-to-many entity hierarchies — companies with multiple
users, departments with multiple employees, schools with multiple
students.

---

## Designing facts

Facts are the temporal heart of the dataset. One row per entity per
period (or whatever grain you configure).

```yaml
facts:
  - name: fct_engagement
    metrics: [engagement, mrr, support_tickets]
    columns:
      - { name: customer_id, type: ref.dim_customer }
      - { name: date_key,    type: ref.dim_date }
```

Two ways to put metrics on a fact table:

### `metrics:` shorthand (recommended)

The `metrics:` array is the simplest path. Each name auto-adds a column
with `type: metric.<name>` and the right dtype. The example above
produces five columns: `customer_id`, `date_key`, `engagement`, `mrr`,
`support_tickets`.

### Explicit `metric.X` columns

When you want to control the column name independently of the metric
name (or skip the metric in the schema even though it's declared), use
`metric.{name}` in the columns array directly:

```yaml
facts:
  - name: fct_revenue
    columns:
      - { name: customer_id,   type: ref.dim_customer }
      - { name: date_key,      type: ref.dim_date }
      - { name: monthly_revenue, type: metric.mrr }
```

Mix and match — you can use both shorthand (`metrics: [engagement]`)
and explicit columns (`{ name: foo, type: metric.mrr }`) on the same
fact table.

---

## Designing events

Event tables have *variable* row counts per `(entity, period)`. Two
trigger types:

### Proportional events

Number of rows per entity per period scales with a metric value:
`rows = metric_value × scale`.

```yaml
events:
  - name: evt_login
    trigger: proportional
    driver: engagement
    scale: 5     # peak ~5 logins per period at engagement = 1.0
    columns:
      - { name: customer_id, type: ref.dim_customer }
      - { name: timestamp,   type: timestamp }
```

Use for activity that recurs whenever the entity is "active" —
logins, transactions, page views.

### Threshold events

A row fires when a metric crosses a threshold and stays past it for N
consecutive periods.

```yaml
events:
  - name: evt_churn
    trigger: threshold
    metric: churn_risk
    above: 0.8
    for_periods: 2
    columns:
      - { name: customer_id, type: ref.dim_customer }
      - { name: date_key,    type: ref.dim_date }
      - { name: churn_flag,  type: flag }
```

Use for state-change moments — churn, conversion, escalation,
qualification. The `flag` column reads True when the threshold fires.

`above` and `below` are mutually exclusive; pick one.

---

## Designing bridges

Bridge tables capture many-to-many relationships between two dim tables.

```yaml
bridges:
  - name: customer_subscription
    left: dim_customer
    right: dim_subscription
    cardinality: [1, 3]
    driver: mrr
    columns:
      - { name: weight, type: metric.mrr }
```

| Field | Notes |
|---|---|
| `left` / `right` | Both must be dim tables (auto or declared); must differ |
| `cardinality` | `[min, max]` second-dim entries per left entity |
| `driver` | Optional metric — biases sampling toward trajectory position |
| `columns` | Up to 20 bridge-row columns (`metric.{name}`, `static.{value}`, `faker.{kind}` only) |

Bridges produce one row per `(left_entity, right_entity)` association.
With `cardinality: [1, 3]` and 100 left entities, expect roughly 200
bridge rows.

When `driver` is set, entities with high trajectory position get more
associations than entities with low position. When omitted, sampling is
uniform.

---

## SCD Type 2 columns

For dim columns whose value *changes* as an entity's behavior crosses
configured thresholds, use the `scd` column type:

```yaml
dimensions:
  - name: dim_customer
    columns:
      - { name: customer_id, type: id }
      - name: plan_tier
        type: scd
        tracks: mrr
        tiers: [free, starter, pro, enterprise]
        at: [0.0, 0.25, 0.50, 0.75]
```

The dim is expanded into multiple rows per entity — one for each band
crossing — with `valid_from` / `valid_to` semantics. Fact rows
automatically reference the active version through a `dim_row_id`
column the engine appends.

Use SCD when:

- The attribute genuinely changes over time (plan tier, role, status)
- Downstream queries need point-in-time joins
- You want history, not just the current state

For static attributes that don't change (industry, signup date), use a
plain column. SCD adds row count and join complexity — only opt in when
you need the history.

---

## What plotsim *won't* let you do

A handful of structural rules the builder enforces at construction:

- **Self-bridges** — `left == right` is rejected. To model intra-dim
  relationships (manager / report, parent / child), use a sub-entity
  dim instead.
- **`pool.{attr}` on facts/events/reference dims** — pool columns are
  only valid on per-entity dim columns. Per-segment attributes are
  entity-level facts, not row-level facts.
- **Bare dtype words on non-`dim_date` tables** — `type: int` is only
  valid on `dim_date`. Other tables need a source-bearing type
  (`metric.X`, `faker.X`, `static.X`, `ref.X`, etc.).
- **`flag` on proportional events** — flags are only valid on threshold
  events; proportional events don't have a threshold to flag.

---

## What to read next

- [Column types](../column-types.md) — every type with valid-on-which-table-type
- [Config field reference §dimensions / §facts / §events / §bridges](../config-reference.md) —
  field-level constraints
- [Output formats](./output-formats.md) — what the schema produces on disk
- [Tutorials → schema and dimensions](../tutorial-notebooks/schema_and_dimensions.ipynb) — runnable example
- [Tutorials → bridges and advanced](../tutorial-notebooks/bridges_and_advanced.ipynb) — bridges + SCD walkthrough
