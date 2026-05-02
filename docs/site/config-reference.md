# Config Reference

> Every input field accepted by `create()` / `create_from_yaml()`.
> Source of truth is the code; this page is the field map.
>
> For column types see [`column-types.md`](./column-types.md).

---

## Builder input shape

```yaml
about: <one-line description>
unit: <singular noun>
window: { start, end, every }
metrics: [ ... ]
segments: [ ... ]
connections: [ ... ]
lifecycle: { track, stages }
dimensions: [ ... ]
facts: [ ... ]
events: [ ... ]
seasonality: [ ... ]
bridges: [ ... ]
quality: [ ... ]
holdout: { target, periods, min_training_periods }
entity_features: true | false | { metrics, include_labels }
seed: <int>
```

Required keys: `about`, `unit`, `window`, `metrics` (at least one),
`segments` (at least one). Everything else is optional. The same shape
is accepted from both `create(**kwargs)` and `create_from_yaml(path)`.

---

## Top-level fields

### `about`

| | |
|---|---|
| Type | `str` |
| Required | yes |
| Constraints | non-empty |

One-line description of what the dataset represents. Surfaces in
`config.domain.description`.

```yaml
about: "Subscription customers churning across 24 months"
```

### `unit`

| | |
|---|---|
| Type | `str` |
| Required | yes |
| Constraints | non-empty; lowercase singular noun recommended |

The thing each entity represents. Used to name the auto-generated entity
dim table (`dim_<unit>`) and to label the entity FK column.

```yaml
unit: customer    # auto-generates dim_customer, customer_id
unit: employee    # auto-generates dim_employee, employee_id
```

### `window`

| | |
|---|---|
| Type | object or 2/3-tuple |
| Required | yes |

Time span and granularity. Three accepted shapes:

```yaml
# Object form
window:
  start: 2024-01
  end: 2024-12
  every: monthly

# Two-tuple (default granularity = monthly)
window: ["2024-01", "2024-12"]

# Three-tuple
window: ["2024-01", "2024-12", "monthly"]
```

| Field | Type | Default | Constraints |
|---|---|---|---|
| `start` | `str` | required | `YYYY-MM` or `YYYY-MM-DD` |
| `end` | `str` | required | same format as `start` |
| `every` | `"daily"` / `"weekly"` / `"monthly"` | `"monthly"` | — |

YAML's relaxed scalar parser turns `2024-01` into a date object; the
builder coerces it back to a string before validation, so both quoted
and unquoted forms work.

### `seed`

| | |
|---|---|
| Type | `int` |
| Required | no |
| Default | drawn from `secrets.randbelow(2**32)` |
| Constraints | `0 ≤ seed ≤ 2**32 - 1` |

Pin this for reproducible output. Same `(config, seed)` always produces
byte-identical files. When omitted, the builder draws a fresh seed from
the system CSPRNG.

---

## `metrics`

Array of metric declarations. At least one required, max 50.

```yaml
metrics:
  - name: engagement
    type: score
    polarity: positive

  - name: mrr
    type: amount
    polarity: positive
    range: [10, 5000]

  - name: churn_risk
    type: score
    polarity: negative
    follows: engagement
    delay: 2
    seasonal_sensitivity: 0.5
```

### Metric fields

| Field | Type | Required | Default | Notes |
|---|---|---|---|---|
| `name` | `str` | yes | — | Alphanumeric / underscore only |
| `type` | enum | yes | — | `score`, `amount`, `count`, `index` |
| `polarity` | enum | yes | — | `positive` (high position → high value) or `negative` (high position → low value) |
| `label` | `str` | no | `None` | Display label; defaults to `name` |
| `range` | `[float, float]` | conditional | `None` | Required for `amount` and `index`; forbidden for `count` |
| `follows` | `str` | no | `None` | Name of another metric this one lags behind. Must pair with `delay` |
| `delay` | `int` | no | `None` | Lag in periods. Must be `≥ 1` and pair with `follows` |
| `seasonal_sensitivity` | `float` | no | `1.0` | Per-metric multiplier on global seasonality. `0.0` immune; `-0.5` halves and inverts |

### Metric types

| Type | Distribution | Range | Use for |
|---|---|---|---|
| `score` | beta(2, 5) | implicit `[0, 1]` | Health scores, engagement indices, satisfaction |
| `count` | poisson(λ=5) | non-negative integers | Logins, transactions, ticket counts |
| `amount` | lognorm or beta (auto-picked) | required | Money, weights, durations |
| `index` | normal | required | Bounded indicators where mean matters |

For `amount`, the builder picks `lognorm` when `min == 0` or
`max / min ≥ 10`, else `beta`. The `index` distribution is centered on
the range midpoint with sigma chosen to keep ~99.7% of draws inside the
range.

### Causal lag (`follows` / `delay`)

`follows: <other_metric>` and `delay: <int>` declare that this metric
trails the named driver by `delay` periods. The two must appear together
or not at all. A metric cannot follow itself, and the chain must be
acyclic — both are checked at construction time.

---

## `segments`

Array of cohort declarations — each segment is a count of entities all
sharing one archetype. At least one required.

```yaml
segments:
  - name: growers
    count: 30
    archetype: growth

  - name: decliners
    count: 20
    archetype: decline
    baseline:
      mrr: high
      engagement: mid
    attributes:
      industry: ["tech", "retail"]

  - name: hybrids
    count: 25
    archetype: "growth > decline @ 0.6"
    seasonal_sensitivity: 0.0
```

### Segment fields

| Field | Type | Required | Default | Notes |
|---|---|---|---|---|
| `name` | `str` | yes | — | Alphanumeric / underscore |
| `count` | `int` | yes | — | `3 ≤ count ≤ 5000` per segment |
| `archetype` | `str` | yes | — | Shape word or composition DSL — see below |
| `label` | `str` | no | `None` | Display label |
| `attributes` | `dict[str, str \| list[str]]` | no | `{}` | Per-segment static attributes; doubles as the source for `pool.{attr}` columns |
| `baseline` | `dict[str, str]` | no | `{}` | Per-metric value-range narrowing — `high` / `mid` / `low` |
| `seasonal_sensitivity` | `float` | no | `1.0` | Per-segment multiplier on global seasonality |

### Archetype DSL

Six base shapes, listed in `BASELINE_RECIPES` / `SHAPE_RECIPES`:

| Shape | Behavior |
|---|---|
| `growth` | Sigmoid rise from low to high |
| `decline` | Exponential decay from high to low |
| `seasonal` | Oscillating around 0.5 |
| `flat` | Constant around 0.15 |
| `spike_then_crash` | Sigmoid rise → step drop → low plateau |
| `accelerating` | Compounding growth with acceleration |

Shapes compose with two operators:

- **Sequence** `>` — chain shapes in order. Default split is even.
  `growth > decline` is half growth, half decline.
- **Anchor** `@` — explicit transition period. `growth > decline @ 8`
  spends periods 0–7 on growth then transitions to decline at period 8.
  With N shapes, supply N-1 `@` clauses (one between every pair).

Examples: `growth`, `growth > decline`, `flat > growth > seasonal @ 4 @ 12`,
`growth > spike_then_crash @ 6`. See
[`user-guide/archetypes.md`](./user-guide/archetypes.md) for the full
DSL.

### Baseline vocabulary

Three words that narrow the metric's value range to a third of its
declared band:

| Word | Range fraction |
|---|---|
| `high` | upper third — `(2/3, 1)` of `[min, max]` |
| `mid` | middle third — `(1/3, 2/3)` |
| `low` | lower third — `(0, 1/3)` |

Useful for "this segment runs hot" / "this segment runs cold" without
authoring a full archetype variant.

---

## `connections`

Array of correlation pairs. Optional.

```yaml
connections:
  - "mrr driven_by engagement"             # 3-token string
  - ["churn_risk", "inverts", "mrr"]        # tuple
  - {a: "support_tickets", relationship: "related", b: "churn_risk"}  # dict
```

Three accepted shapes for each entry — pick whichever reads best.

### Relationship vocabulary

Nine words spanning `-0.75` to `+0.75`:

| Word | Coefficient |
|---|---|
| `mirrors` | +0.75 |
| `driven_by` | +0.55 |
| `related` | +0.40 |
| `hints_at` | +0.20 |
| `independent` | 0.00 |
| `hints_against` | -0.20 |
| `resists` | -0.40 |
| `opposes` | -0.55 |
| `inverts` | -0.75 |

Both endpoints must reference declared metrics. Self-pairs and connections
on metrics named in `lifecycle.track` are rejected at construction time.

If your declared correlation matrix is not positive semi-definite, the
engine projects it to the nearest valid matrix using Higham's algorithm
and records the adjustment in the manifest under
`correlation_adjustments`.

---

## `lifecycle`

Optional ladder of named thresholds against a chosen metric. When set,
the engine emits a stage column on the relevant fact table.

```yaml
lifecycle:
  track: engagement
  stages:
    - { onboarding: 0.0 }
    - { active: 0.3 }
    - { at_risk: 0.6 }
    - { churned: 0.9 }
```

Stage entries accept four shapes:
`{onboarding: 0.0}`, `(onboarding, 0.0)`,
`{name: onboarding, threshold: 0.0}`, or canonical form. Each thresh
must be in `[0, 1]`; thresholds must be strictly ascending; stage names
must be unique.

| Field | Type | Required | Notes |
|---|---|---|---|
| `track` | `str` | yes | Must be a declared metric |
| `stages` | array | yes | At least 2 entries |

The keyword `lifecycle` is canonical; `stages` is also accepted as the
outer block name as an alias.

---

## `seasonality`

Optional global seasonal effects, each spanning a set of calendar months.

```yaml
seasonality:
  - { months: [11, 12], strength: 0.30 }   # +30% in Nov-Dec
  - { months: [6, 7, 8], strength: -0.10 } # -10% in summer
```

| Field | Type | Required | Notes |
|---|---|---|---|
| `months` | tuple of int | yes | Values in `1..12`, unique within one effect, max 12 entries |
| `strength` | `float` | yes | Multiplier added to `1.0` at each named month |

Multiple effects may overlap — strengths sum at each period. The summed
effect is then multiplied by per-metric `seasonal_sensitivity` and
per-segment `seasonal_sensitivity` before being applied to the metric's
distribution center.

The empty default `[]` produces output byte-identical to runs without a
seasonality block.

---

## Schema overrides — `dimensions`, `facts`, `events`

When you want named columns that aren't auto-generated. Each entry uses
the same shape:

```yaml
dimensions:
  - name: dim_customer
    columns:
      - { name: customer_id, type: id }
      - { name: signup_date, type: date }
      - { name: industry,    type: pool.industry }

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

See [`column-types.md`](./column-types.md) for every supported `type`.

### Dim fields

| Field | Type | Required | Default | Notes |
|---|---|---|---|---|
| `name` | `str` | yes | — | Table name; conventionally `dim_<thing>` |
| `columns` | array | yes | — | At least one column |
| `per` | `"period"` / `"unit"` | no | `None` | Cardinality hint — one row per period or per unit |
| `reference` | `bool` | no | `false` | Pure lookup table (no per-entity / per-period rows) |
| `count` | `int` | no | `1` | Sub-entity multiplier (e.g. `dim_user` with `count=3` produces 3 users per customer) |

`reference: true` and `per` are mutually exclusive.

### Fact fields

| Field | Type | Required | Default | Notes |
|---|---|---|---|---|
| `name` | `str` | yes | — | Conventionally `fct_<thing>` |
| `columns` | array | yes | — | At least one column |
| `metrics` | array of `str` | no | `[]` | Metric names whose `metric.{name}` columns are added automatically |

### Event fields

| Field | Type | Required | Default | Notes |
|---|---|---|---|---|
| `name` | `str` | yes | — | Conventionally `evt_<thing>` |
| `columns` | array | yes | — | At least one column |
| `trigger` | `"proportional"` / `"threshold"` | yes | — | How row count is determined |
| `driver` | `str` | proportional only | — | Metric whose value drives row count |
| `scale` | `float` | proportional only | — | `≥ 0`. Rows per entity per period = `metric_value × scale` |
| `metric` | `str` | threshold only | — | Metric to watch |
| `above` | `float` | threshold only | — | Fire when value crosses above this |
| `below` | `float` | threshold only | — | Fire when value crosses below this |
| `for_periods` (alias `for`) | `int` | no | `None` | Hold the threshold for N periods before firing |

`above` and `below` are mutually exclusive on a single event.

---

## `bridges`

Many-to-many associations between two dimension tables.

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

| Field | Type | Required | Default | Notes |
|---|---|---|---|---|
| `name` | `str` | yes | — | Alphanumeric / underscore |
| `left` | `str` | yes | — | Dim table name; auto-dims (`dim_date`, `dim_<unit>`) are valid |
| `right` | `str` | yes | — | Same; must differ from `left` |
| `cardinality` | `[int, int]` | yes | — | Inclusive `[min, max]` second-dim entries per left entity |
| `driver` | `str` | no | `None` | Optional metric — non-null biases sampling toward trajectory position |
| `columns` | array | no | `[]` | Up to 20 bridge-row columns (`metric.{name}`, `static.{value}`, `faker.{kind}` only) |

Limit: 20 bridges per config.

---

## `quality`

Post-generation data corruption — null injection, duplicates, type
mismatches, late arrivals, schema drift.

```yaml
quality:
  - { table: fct_engagement, issue: null_injection,  rate: 0.02, column: engagement }
  - { table: fct_engagement, issue: duplicate_rows, rate: 0.01 }
```

| Field | Type | Required | Default | Notes |
|---|---|---|---|---|
| `table` | `str` | yes | — | Target table |
| `issue` | enum | yes | — | `null_injection`, `duplicate_rows`, `type_mismatch`, `late_arrival`, `schema_drift` |
| `rate` | `float` | yes | — | `0.0 ≤ rate ≤ 1.0` |
| `column` | `str` | conditional | `None` | Required for `null_injection`, `type_mismatch`, `schema_drift`; optional otherwise |
| `seed_offset` | `int` | no | `0` | Sub-seed offset to vary which rows are corrupted under the same config seed |

Limit: 50 quality issues per config. The clean copy of the data is
preserved in memory; the manifest's `quality_injections` list records
exactly which rows / columns / clean values were corrupted so a downstream
consumer can recover ground truth.

`quality` is mutually exclusive with `entity_features` and with
`holdout` — both rules raise at config load.

---

## `holdout`

Temporal train/holdout split for ML target workflows.

```yaml
holdout:
  target: mrr
  periods: 3
  min_training_periods: 6
```

| Field | Type | Required | Default | Notes |
|---|---|---|---|---|
| `target` | `str` | yes | — | Metric you intend to predict |
| `periods` | `int` | yes | — | Trailing periods reserved for evaluation. `1 ≤ periods ≤ 10000` |
| `min_training_periods` | `int` | no | `3` | Floor on `n_periods - periods`; rejected at load if violated |

When set, every per-entity-per-period fact table writes two extra files:
`<fact>_train.<ext>` (`[0, n - periods)`) and `<fact>_holdout.<ext>`
(`[n - periods, n)`). The unsplit fact is also written. Dim, bridge, and
event tables are not split.

When entity features are also enabled, aggregation restricts to the
training window and the target metric's six aggregate columns are
dropped to prevent label leakage.

---

## `entity_features`

Per-entity flat feature table emission.

Two accepted shapes:

```yaml
# Shorthand — emit every numeric metric, with labels
entity_features: true

# Detailed
entity_features:
  metrics: [engagement, mrr]
  include_labels: true
```

| Field | Type | Required | Default | Notes |
|---|---|---|---|---|
| `metrics` | array of `str` | no | `[]` (every numeric metric on a fact table) | Each name must reference a numeric fact metric. Max 50 |
| `include_labels` | `bool` | no | `true` | Emits `archetype` and `final_trajectory_position` columns |

For every selected metric, six aggregate columns are added per entity:
`<m>_mean`, `<m>_std`, `<m>_slope`, `<m>_first`, `<m>_last`,
`<m>_peak_period`. See [`build_entity_features`](./api-reference.md#build_entity_features).

Pre-conditions enforced at load: `manifest.include` must be True;
`quality.quality_issues` must be empty.
