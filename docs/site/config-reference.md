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
lifecycle: { track, stages, enforce_order, downgrade_delay }
dimensions: [ ... ]
facts: [ ... ]
events: [ ... ]
seasonality: [ ... ]
bridges: [ ... ]
quality: [ ... ]
holdout: { target, periods, min_training_periods }
entity_features: true | false | { metrics, include_labels }
noise: <preset_name> | { gaussian_sigma, outlier_rate, mcar_rate }
output: csv | parquet | { format, directory }
locale: <faker locale or list of locales>
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
| `decay_window` | `int` | no | `None` | Spread the lagged read over `N` periods ending at `t-delay`. Requires `follows` / `delay` |
| `decay_kernel` | enum | no | `"geometric"` | `geometric` (half-life one period) or `linear` weight shape; ignored without `decay_window` |
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

Array of correlation pairs. Optional. Each entry has three slots —
left metric, relationship-or-coefficient, right metric — and accepts
six shorthand forms:

```yaml
connections:
  # Word form
  - "mrr driven_by engagement"                        # 3-token string
  - ["churn_risk", "inverts", "mrr"]                  # tuple
  - {metric_a: "support_tickets", relationship: "related", metric_b: "churn_risk"}

  # Numeric form (any coefficient in [-1.0, 1.0])
  - "engagement 0.42 retention"                       # numeric middle token
  - ["mrr", -0.31, "support_tickets"]                  # numeric in tuple
  - {metric_a: "nps", coefficient: 0.18, metric_b: "feature_adoption"}
```

The middle slot is parsed as a number when it tokenizes to a float;
otherwise it's looked up against the relationship vocabulary. Each
canonical entry sets *exactly one* of `relationship` / `coefficient`
— passing both raises at construction time, since the word already
implies a coefficient.

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

The numeric form accepts any value in `[-1.0, 1.0]` — useful when you've
calibrated the coefficient from a real dataset and the nine-word
vocabulary doesn't land on the right magnitude. Coefficients of exactly
`0.0` are dropped (treated as "independent") with a warning, matching
the engine's redundant-pair contract.

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
  enforce_order: false        # default — stateless free-mode
  downgrade_delay: null       # ignored when enforce_order is false
```

Stage entries accept four shapes:
`{onboarding: 0.0}`, `(onboarding, 0.0)`,
`{name: onboarding, threshold: 0.0}`, or canonical form. Each thresh
must be in `[0, 1]`; thresholds must be strictly ascending; stage names
must be unique.

| Field | Type | Required | Default | Notes |
|---|---|---|---|---|
| `track` | `str` | yes | — | Must be a declared metric |
| `stages` | array | yes | — | At least 2 entries |
| `enforce_order` | `bool` | no | `false` | When `false`, every period independently picks the highest threshold the realised value satisfies — stateless free-mode. When `true`, the cursor advances only and an entity can't jump back on a transient dip — a monotonic stage walk |
| `downgrade_delay` | `int` or `null` | no | `null` | Hysteresis under `enforce_order: true`. The cursor steps back once the entity has sat below the demote threshold for `downgrade_delay` consecutive periods. `null` keeps strict monotonicity. Range `1`–`120` |

The keyword `lifecycle` is canonical; `stages` is also accepted as the
outer block name as an alias (a back-compat path for the early-spec
keyword — both forms parse identically).

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

### How to enable

Append to any config that has (or auto-generates) at least two dim
tables. Replace `dim_a` / `dim_b` with the names of two distinct dims
already in your config — `dim_date` and `dim_{unit}` are auto-generated
and always valid targets.

```yaml
bridges:
  - name: a_b
    left: dim_a
    right: dim_b
    cardinality: [1, 3]
```

### Detailed example

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
mismatches, late arrivals, schema drift, volume anomalies.

### How to enable

Append to any config. Replace `<fact>` with one of your fact-table
names and `<metric_col>` with a column on that fact. Mutually exclusive
with `holdout` and `entity_features` — pick one corruption strategy
per config.

```yaml
quality:
  - { table: <fact>, issue: null_injection, rate: 0.02, column: <metric_col> }
  - { table: <fact>, issue: duplicate_rows, rate: 0.01 }
```

### Detailed example

```yaml
quality:
  - { table: fct_engagement, issue: null_injection,  rate: 0.02, column: engagement }
  - { table: fct_engagement, issue: duplicate_rows, rate: 0.01 }
  - { table: fct_engagement, issue: volume_anomaly, rate: 1.0, mode: spike, period: 5 }
  - { table: fct_engagement, issue: volume_anomaly, rate: 0.5, mode: drop,  periods: [11, 17] }
```

| Field | Type | Required | Default | Notes |
|---|---|---|---|---|
| `table` | `str` | yes | — | Target table |
| `issue` | enum | yes | — | `null_injection`, `duplicate_rows`, `type_mismatch`, `late_arrival`, `schema_drift`, `volume_anomaly` |
| `rate` | `float` | yes | — | `0.0 ≤ rate ≤ 1.0`. For `volume_anomaly` it scales per-period (rows at the target period), not whole-table |
| `column` | `str` | conditional | `None` | Required for `null_injection`, `type_mismatch`, `schema_drift`. Forbidden on `volume_anomaly` (row-level) |
| `mode` | enum | conditional | `None` | `volume_anomaly` only. `spike` appends duplicate rows; `drop` removes rows |
| `period` | `int` | conditional | `None` | `volume_anomaly` only. 0-based period index. Exactly one of `period` / `periods` |
| `periods` | `list[int]` | conditional | `None` | `volume_anomaly` only. List form for multiple target periods |
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

### How to enable

Append to any config. Replace `<metric>` with any numeric metric
emitted on a per-entity-per-period fact table. The minimum is two
lines (`target` + `periods`); `min_training_periods` defaults to 3.
Requires `quality: []` — the splits work on the clean tables.

```yaml
holdout:
  target: <metric>
  periods: 3
```

### Detailed example

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

### How to enable

Append one line to any config. Defaults to "every numeric metric
emitted on a fact table, with archetype + final-trajectory labels."
Requires `quality: []` and `manifest.include: true` (the default).

```yaml
entity_features: true
```

### Detailed example

```yaml
# Narrow the metric set or strip labels for unsupervised pipelines
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

---

## `noise`

Distributional noise applied on top of the trajectory-driven distribution
center. Three independent dials, all defaulting to zero (no noise — the
default produces clean output identical to pre-noise baselines).

```yaml
# Preset shorthand
noise: realistic

# Detailed
noise:
  gaussian_sigma: 0.05
  outlier_rate: 0.02
  mcar_rate: 0.01
```

| Field | Type | Default | Range | Effect |
|---|---|---|---|---|
| `gaussian_sigma` | `float` | `0.0` | `0.0`–`5.0` | Multiplicative log-normal jitter on each draw — `value *= exp(N(0, σ²))`. Bigger σ = wider spread |
| `outlier_rate` | `float` | `0.0` | `0.0`–`1.0` | Probability per cell of replacing the value with a 3-σ tail draw |
| `mcar_rate` | `float` | `0.0` | `0.0`–`1.0` | Probability per cell of dropping the value to NaN (missing-completely-at-random) |

Four named presets accept the lower-case canonical name OR a friendly
alias — pick whichever reads naturally:

| Preset | `gaussian_sigma` | `outlier_rate` | `mcar_rate` | Aliases |
|---|---|---|---|---|
| `perfectly_clean` *(default — same as omitting `noise`)* | 0.00 | 0.00 | 0.000 | `clean` |
| `slightly_messy` | 0.03 | 0.01 | 0.005 | — |
| `realistic` | 0.05 | 0.02 | 0.010 | `messy` |
| `dirty` | 0.10 | 0.05 | 0.030 | `very_messy` |

The same constants are exported from `plotsim` for engine-direct
mutation: `PERFECTLY_CLEAN`, `SLIGHTLY_MESSY`, `REALISTIC`, `DIRTY`.

`noise` is independent of the `quality` block — `noise` perturbs metric
values *during* generation (correlations and trajectory still hold);
`quality` corrupts the output table *after* generation.

---

## `output`

Output-format selector and target directory.

```yaml
# Word shorthand (uses default directory ./output)
output: parquet

# Detailed
output:
  format: parquet
  directory: ./fixtures
```

| Field | Type | Default | Notes |
|---|---|---|---|
| `format` | `"csv"` / `"parquet"` | `"csv"` | CSV is the default; `parquet` requires `pip install plotsim[parquet]` (pyarrow). Same engine path, ~5–10× smaller on disk |
| `directory` | `str` | `"output"` | Where `write_tables` writes. Override at call time with `write_tables(..., output_dir=...)` |
| `cell_budget` | `int ≥ 0` / `null` | `null` | M7 — soft cell-count cap consumed by the load-time scale estimator. `null` falls through to `PLOTSIM_CELL_BUDGET` env var, then to the 2,000,000 default. `0` disables the soft cap entirely. See [Cell-count budget](#cell-count-budget) for precedence and the bundled `lakehouse` template for a worked example |

When `format: parquet` and `pyarrow` is missing, `write_tables` raises
`ImportError` naming the install command — fail-fast at the write call
rather than mid-iteration. See [Output formats](./user-guide/output-formats.md)
for the full pickup of column dtypes, dim_date typing, and downstream
loader notes.

---

## `locale`

Faker locale (or list of locales) threaded to every `faker.*` column.

```yaml
locale: en_GB                   # single locale
locale: [en_US, ja_JP, de_DE]   # multi-locale mix
```

| Type | Default | Notes |
|---|---|---|
| `str` or `list[str]` | `"en_US"` | Any locale supported by your installed `faker` package. Lists round-robin across providers — useful when seeded fixtures should look multinational |

Locale only affects `faker.*` columns; `static.*`, `metric.*`, and
`pool.*` columns are unaffected.

---

## Engine-direct fields

A handful of `PlotsimConfig` fields are not surfaced in the builder YAML
above. They live on the engine config — set them with
`load_config()`/`dump_config()` round-trips, or by passing them to a
hand-authored engine-direct YAML.

### `compensate_correlations`

| | |
|---|---|
| Type | `bool` |
| Default (engine-direct) | `False` |
| Default (builder) | `True` |

When `True`, the engine pre-compensates the trajectory-driven mean shift
so the realized Pearson correlations land closer to the declared
`connections` coefficients on configs with strong archetype mixes.
Records each adjustment in `manifest.correlation_compensations`. The
builder layer sets `True` explicitly because `connections` is a
table-wide intent contract; engine-direct configs default to `False`
to preserve byte-identical output for pre-M120 YAML on disk.

### `generation_mode`

| | |
|---|---|
| Type | `"serial"` / `"vectorized"` / `"auto"` |
| Default (engine-direct) | `"serial"` |
| Default (builder) | `"auto"` |

`"vectorized"` batches all entities in an archetype group through one
copula draw — large speedups on configs above ~5,000 entities, identical
results modulo the deliberate copula bypass-fallback contract.
`"auto"` picks per archetype group by entity count; `create()` /
`create_from_yaml()` set `"auto"` explicitly. Manifest records the
mode and any bypass-fallback counts under `bypass_fallback_counts`.

### Per-entity overrides — `cross_dim_fks` and `inflection_month`

Both fields live on individual `Entity` objects (the resolved
counterpart to a builder `segment`). They steer per-entity behavior
that doesn't belong at the segment level:

```python
from plotsim.config import EntityOverrides
cfg.entities[0].cross_dim_fks = {"plan_id": "plan_enterprise"}
cfg.entities[0].overrides = EntityOverrides(inflection_month=4)
```

| Field | Type | Default | Purpose |
|---|---|---|---|
| `cross_dim_fks` | `dict[str, str]` | `{}` | Pin specific FK column values to specific PKs in another dim — e.g. bind expansion-champion accounts to a specific plan row. Bypasses the column's `distribution` for that entity |
| `overrides.inflection_month` | `int` or `None` | `None` | Shift the archetype's curve segments so its canonical inflection lands on this period index. Per-entity narrative timing (e.g. "this account turned around in March") |

### `manifest`

The manifest emission config. Defaults to `include: true`,
`trajectory_sample_rate: 1.0` — every run lands a `manifest.json` next
to the table files. Set `manifest: {include: false}` for microbenchmarks
or sandboxed CI runs that don't need the ground-truth payload. See
[Manifest reference](./manifest-reference.md).

### Per-archetype overrides — `curve_segments` and `metric_overrides`

Two mechanisms let an archetype diverge from the global metric defaults.

**`Archetype.curve_segments`** — per-archetype list of `CurveSegment`
entries defining the full `[0.0, 1.0]` trajectory shape. Segments must
cover the range without gaps or overlaps (validated at config load).
Every metric reads its position from this curve; there is no
per-metric curve override.

**`Archetype.metric_overrides`** — `dict[str, MetricOverride]` keyed
by metric name. Each entry can override `distribution`, `params`, or
`value_range` for that metric *only when sampled for entities of this
archetype*. `polarity` and `causal_lag` are never overridable.

`value_range` overrides must be a *subset* of the global range —
overrides narrow, never expand. Subset enforcement runs at config
load.

**Resolution order:** for each `(entity, metric)` draw, the engine
looks up `archetype.metric_overrides[metric.name]`. If present, listed
fields replace the global `Metric` fields; unset fields fall through
to the global metric. Partial overrides compose cleanly via
`model_copy(update=…)`.

The builder API surfaces `metric_overrides.value_range` only;
`distribution` and `params` overrides require an engine-direct config.

---

## Limits and performance gates

Every config is checked against per-field caps and a global
cell-count budget at load time. The bounds are intentionally
conservative — well above any realistic dashboard dataset, well
below the point where a single laptop run becomes painful.

| Limit | Cap | Behavior on breach |
|---|---|---|
| `metrics` count | 50 | Pydantic rejects at load |
| Per-segment `count` | 5000 | Pydantic rejects at load |
| Total entities (`Σ segments.count`) | 100,000 | Custom validator rejects at load |
| `quality` issues | 50 | Pydantic rejects at load |
| `bridges` count | 20 | Pydantic rejects at load |
| Per-bridge `columns` | 20 | Pydantic rejects at load |
| `seasonality` effects | 12 | Pydantic rejects at load |
| Causal lag `delay` | `1`–`10000` periods | Pydantic rejects at load |

### Cell-count budget

The cell count (`Σ segments.count × n_periods`) drives a tiered
budget. The thresholds protect against runaway configs while keeping
big datasets a real feature for users who deliberately want them.

| Cell count | Behavior |
|---|---|
| ≤ 500,000 | Silent (just the always-printed summary line) |
| > 500,000 | Stderr **advisory** recommending `output.format: parquet` and `generation_mode: auto` |
| > soft budget (default 2,000,000) | `ValueError` at load with instructions to opt in |
| > soft budget, opt-in given | Stderr **large-dataset notice**, generation proceeds |
| > 50,000,000 | Hard ceiling — `ValueError` regardless of opt-in |

Two ways to opt into above-soft-budget runs:

1. **CLI flag** — `--allow-large-dataset` on `plotsim run`,
   `plotsim validate`, or `plotsim info`.
2. **Environment variable** — `PLOTSIM_ALLOW_LARGE_DATASET=1` for
   library callers and CI scripts.

Three ways to change the soft-budget threshold itself, in
precedence order (the first one that resolves wins):

1. **Config field (recommended)** — set `output.cell_budget: N` in
   the YAML (or pass `output={"cell_budget": N}` to `create()`).
   Reproducible from the config alone — no env vars or flags
   required, which is the contract the bundled `lakehouse`
   template relies on.
2. **Environment variable** — `PLOTSIM_CELL_BUDGET=N` sets the
   soft cap to `N` cells when no config field is set.
3. **Default** — `2,000,000` cells.

`output.cell_budget: 0` (or `PLOTSIM_CELL_BUDGET=0`) disables the
soft cap entirely; only the 50,000,000-cell hard ceiling still
applies. Setting `output.cell_budget` past the projected cell
count is the YAML-only equivalent of `--allow-large-dataset`:
because the cap is raised, the run no longer "exceeds" it and no
opt-in is needed.

The hard ceiling is non-configurable. Configs above 50,000,000 cells
should be split or chunked rather than coerced through a single run.

A summary line is *always* printed to stderr at load time so the
projected cell count and peak memory estimate are visible even on
runs well below the threshold:

```
Config summary: 80 entities × 24 periods = 1,920 cells, 4 metrics, 6 tables. Estimated peak memory: ~100 MB.
```
