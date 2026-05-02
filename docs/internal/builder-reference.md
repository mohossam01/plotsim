# Builder Reference

> **Purpose.** Complete vocabulary and field reference for the plain-language
> input format consumed by `plotsim.create()` / `plotsim.create_from_yaml()`.
>
> **Source of truth.** Recipe values (correlation coefficients, distribution
> picks, baseline fractions) live in `plotsim/builder/recipes.py`. Field
> requirements live in `plotsim/builder/input.py`. The acceptance test
> `tests/test_builder_schema.py` keeps the vocabulary enums in
> `plotsim/builder/schema.py` aligned with the recipes — drift surfaces as
> a test failure.
>
> **JSON Schema.** Programmatic consumers (IDE tooling, UI, lint rules)
> should consume the schema from
> `plotsim.builder.schema.generate_user_input_schema()`. This document is
> the human-readable companion.

---

## 1. The two surfaces

`plotsim.create()` accepts keyword arguments; `plotsim.create_from_yaml(path)`
loads a YAML file. Both normalise into the same `UserInput` model and the
same `interpret(...)` step. Anything you can write in YAML you can express
as Python kwargs and vice versa — see
[`plotsim/configs/new/saas_template.py`](https://github.com/mohossam-ae/plotsim/blob/main/plotsim/configs/new/saas_template.py)
for the full Python-side equivalent of
[`saas_template.yaml`](https://github.com/mohossam-ae/plotsim/blob/main/plotsim/configs/new/saas_template.yaml).

```python
from plotsim import create_from_yaml, generate_tables
import numpy as np

cfg = create_from_yaml("plotsim/configs/new/saas_template.yaml")
tables = generate_tables(cfg, np.random.default_rng(cfg.seed))
```

---

## 2. Required and optional fields

`UserInput` is the root model. Required fields raise immediately on
omission; optional fields default to the values shown.

| Field         | Type                                  | Required | Default       | Notes                                                                  |
|---------------|---------------------------------------|----------|---------------|------------------------------------------------------------------------|
| `about`       | string                                | yes      | —             | Free-text domain description; surfaces in `Domain.name`.               |
| `unit`        | string                                | yes      | —             | Singular noun (e.g. `company`, `customer`); drives `dim_{unit}` naming. |
| `window`      | object \| 2/3-tuple                   | yes      | —             | `{start, end, every}` or `(start, end[, every])`.                       |
| `metrics`     | list of metric objects                | yes      | —             | Min length 1.                                                          |
| `segments`    | list of segment objects               | yes      | —             | Min length 1; each segment becomes one archetype + one entity.         |
| `connections` | list of connection objects/strings    | no       | `[]`          | Correlation pair declarations.                                          |
| `lifecycle`   | object                                | no       | `null`        | Lifecycle ladder (track + ordered named thresholds).                    |
| `dimensions`  | list of dim objects                   | no       | `[]`          | Omit to auto-generate `dim_date` + `dim_{unit}`.                        |
| `facts`       | list of fact objects                  | no       | `[]`          | Omit to auto-generate `fct_{unit}` covering all metrics.                |
| `events`      | list of event objects                 | no       | `[]`          | None auto-generated.                                                    |
| `seasonality` | list of seasonal effects              | no       | `[]`          | Global calendar-month modulation. See §2.12.                            |
| `bridges`     | list of bridge objects                | no       | `[]`          | Many-to-many associations. See §2.8.                                    |
| `quality`     | list of quality issues                | no       | `[]`          | Post-generation data corruption. See §2.9.                              |
| `holdout`     | object                                | no       | `null`        | Train/holdout split. See §2.10.                                         |
| `entity_features` | bool or object                    | no       | `null`        | Flat per-entity feature table. See §2.11.                               |
| `seed`        | int in `[0, 2**32 - 1]`               | no       | random        | Pin for reproducibility. Omitted → drawn from `secrets.randbelow(2**32)`. |

Omitting **any** of `dimensions`, `facts`, and `events` together (all three
empty) tells the interpreter to generate a minimal viable schema:
`dim_date` (per-period) + `dim_{unit}` (per-entity) + `fct_{unit}`
(per-entity-per-period) carrying every declared metric. See
[`plotsim/configs/new/bare_minimum.yaml`](https://github.com/mohossam-ae/plotsim/blob/main/plotsim/configs/new/bare_minimum.yaml).

### 2.1 `window`

| Field    | Type                              | Required | Default    | Notes                                  |
|----------|-----------------------------------|----------|------------|----------------------------------------|
| `start`  | string `YYYY-MM` or `YYYY-MM-DD`  | yes      | —          |                                        |
| `end`    | string `YYYY-MM` or `YYYY-MM-DD`  | yes      | —          |                                        |
| `every`  | `daily` \| `weekly` \| `monthly`  | no       | `monthly`  |                                        |

YAML's relaxed scalar parsing turns `2023-01` into a `datetime.date` —
`create_from_yaml` coerces window dates back to strings before validation.

### 2.2 `metrics[*]`

| Field      | Type                                      | Required | Default      | Notes                                                                |
|------------|-------------------------------------------|----------|--------------|----------------------------------------------------------------------|
| `name`     | identifier (`[a-zA-Z0-9_]+`)              | yes      | —            | Must be unique across `metrics`.                                     |
| `type`     | `score` \| `amount` \| `count` \| `index` | yes      | —            | Drives distribution selection.                                       |
| `polarity` | `positive` \| `negative`                  | yes      | —            | `negative` flips the trajectory position before the distribution.    |
| `label`    | string                                    | no       | titled name  | Display-only.                                                        |
| `range`    | `[min, max]`                              | conditional | —         | **Required** for `amount`, `index`. **Forbidden** for `count`.       |
| `follows`  | metric name                               | conditional | `null`    | Causal-lag driver. Must reference another declared metric; cycles forbidden. |
| `delay`    | int >= 1                                  | conditional | `null`    | Periods to lag behind `follows`. Must be paired with `follows`.       |
| `seasonal_sensitivity` | float                          | no       | `1.0`        | Per-metric multiplier on global seasonality. `0.0` → immune. Negative inverts. |

`score` defaults to range `[0, 1]`; `count` has no `value_range`.

### 2.3 `segments[*]`

| Field        | Type                                   | Required | Default            | Notes                                                                            |
|--------------|----------------------------------------|----------|--------------------|----------------------------------------------------------------------------------|
| `name`       | identifier                             | yes      | —                  | Must be unique across `segments`. Becomes the archetype name and the entity name. |
| `count`      | int in `[3, 5000]`                     | yes      | —                  | Number of records in the cohort.                                                  |
| `archetype`  | DSL string (see §4)                    | yes      | —                  | Validated against the window length at construction time.                         |
| `label`      | string                                 | no       | titled name        | Description shown in the manifest and notebooks.                                  |
| `attributes` | dict of arbitrary string/list values   | no       | `{}`               | Free-form metadata; doubles as the `pool.{attr}` source — see §2.6.               |
| `baseline`   | dict of `{metric_name: baseline_word}` | no       | `{}`               | Per-segment value-range restriction. Words: `high`, `mid`, `low` (§5).            |
| `seasonal_sensitivity` | float                        | no       | `1.0`              | Per-segment multiplier on global seasonality. Composes with the per-metric value.  |

### 2.4 `connections[*]`

Three accepted shapes — all normalise to `(metric_a, relationship, metric_b)`:

| Shape                                                | Example                                            |
|------------------------------------------------------|----------------------------------------------------|
| 3-token string (whitespace-separated)                | `"engagement driven_by mrr"`                       |
| 3-tuple                                              | `("engagement", "driven_by", "mrr")`               |
| dict                                                 | `{"metric_a": "engagement", "relationship": "driven_by", "metric_b": "mrr"}` |

`relationship` must be a vocabulary word (§3); endpoints must reference
declared metrics; endpoints must be distinct.

### 2.5 `lifecycle`

| Field     | Type                                   | Required | Default | Notes                                                                          |
|-----------|----------------------------------------|----------|---------|--------------------------------------------------------------------------------|
| `track`   | metric name                            | yes      | —       | The metric whose trajectory drives stage transitions.                           |
| `stages`  | list of stage objects                  | yes      | —       | Min length 2. Strictly ascending thresholds in `[0.0, 1.0]`. Names unique.      |

Stage shapes accepted for each entry in `stages`:

| Shape                                              | Example                                       |
|----------------------------------------------------|-----------------------------------------------|
| Single-key dict                                    | `{onboarding: 0.0}`                           |
| 2-tuple                                            | `("onboarding", 0.0)`                         |
| Canonical dict                                     | `{"name": "onboarding", "threshold": 0.0}`    |

The interpreter emits a `StageSequence` with `enforce_order=False` (free-mode
stages — entities can revisit any stage at any time).

`stages` is also accepted as the outer block name for back-compatibility
(`stages: {track: ..., stages: [...]}`); `lifecycle` is canonical.

### 2.6 `segments[*].attributes` and `pool.{attr}` columns

`SegmentInput.attributes` is a free-form dict of extra per-segment
properties — `industry`, `region`, `tier`. Two effects:

1. A column with `type: pool.{attr}` on a per-entity dim resolves to
   `(string, pool:{attr})` with a `value_pool` keyed by every expanded
   entity. Each entity's pool entry is the attribute list declared on
   its segment; scalar values wrap into a single-element list. The
   engine's `validate_value_pool_coverage` check requires every entity
   to have a pool entry, which means **every segment must declare the
   attribute** referenced by the column. Missing it on any segment
   raises a clear builder-side error.
2. When `dimensions`/`facts`/`events` are all omitted (auto-schema
   path), the auto-generated `dim_{unit}` gains one `pool:{attr}`
   column per attribute declared on **every** segment, alphabetically
   ordered. Attributes declared on only some segments are silently
   skipped from the auto-schema. Explicit-schema users can still
   reference the attribute via `pool.{attr}` if they're willing to
   declare it on every segment.

```yaml
segments:
  - name: enterprise_co
    count: 25
    archetype: growth
    attributes:
      industry: [Technology, Finance]   # list → multi-value pool
      region: [US, EMEA]
      tier: enterprise                  # scalar → 1-element pool

dimensions:
  - name: dim_company
    per: unit
    columns:
      - {name: company_id, type: id}
      - {name: industry,   type: pool.industry}   # → pool:industry
      - {name: tier,       type: pool.tier}       # → pool:tier
```

### 2.7 `dimensions[*]`, `facts[*]`, `events[*]`

See §6 for column types and §7 for event triggers.

`dim`:

| Field        | Type                          | Required | Default | Notes                                                          |
|--------------|-------------------------------|----------|---------|----------------------------------------------------------------|
| `name`       | identifier                    | yes      | —       |                                                                |
| `columns`    | list of column objects        | yes      | —       | Min length 1.                                                  |
| `per`        | `period` \| `unit`            | conditional | `null` | Mutually exclusive with `reference`.                          |
| `reference`  | bool                          | no       | `false` | Static lookup; FK on a fact table is documentary, not in PK.   |
| `count`      | int >= 1                      | no       | `1`     | Sub-entity row multiplier (deferred — bundled templates use 1). |

`fact`:

| Field      | Type                          | Required | Default | Notes                                              |
|------------|-------------------------------|----------|---------|----------------------------------------------------|
| `name`     | identifier                    | yes      | —       |                                                    |
| `columns`  | list of column objects        | yes      | —       | Min length 1.                                      |
| `metrics`  | list of metric names          | no       | `[]`    | Documentary hint; column types drive the wiring.   |

`event`:

| Field         | Type                              | Required    | Default | Notes                                                         |
|---------------|-----------------------------------|-------------|---------|---------------------------------------------------------------|
| `name`        | identifier                        | yes         | —       |                                                               |
| `columns`     | list of column objects            | yes         | —       | Min length 1.                                                 |
| `trigger`     | `proportional` \| `threshold`     | yes         | —       |                                                               |
| `driver`      | metric name                       | proportional only | —  | Row count = `driver × scale` per period.                      |
| `scale`       | float >= 0                        | proportional only | —  |                                                               |
| `metric`      | metric name                       | threshold only    | —  | The metric whose crossings trigger the event.                  |
| `above`       | float                             | threshold only    | —  | Pick **one** of `above` or `below`.                            |
| `below`       | float                             | threshold only    | —  |                                                               |
| `for_periods` | int >= 1                          | threshold only    | `1` | Threshold must hold for N periods. YAML alias: `for`.          |

Threshold events fire **once per entity** at the first period the
threshold holds for `for_periods` consecutive periods (see
`engine-internals.md` §4.4).

---

### 2.8 `bridges[*]` (optional)

Many-to-many bridge between two dim tables. Bridges sit alongside the
`tables` list but in their own engine collection (`PlotsimConfig.bridges`)
so they don't share `Table`'s grain constraints.

| Field         | Type                | Required | Default | Notes                                                                         |
|---------------|---------------------|----------|---------|-------------------------------------------------------------------------------|
| `name`        | identifier          | yes      | —       |                                                                               |
| `left`        | dim table name      | yes      | —       | Must reference a declared dim or auto-schema `dim_{unit}` / `dim_date`.       |
| `right`       | dim table name      | yes      | —       | Must differ from `left`. Cannot be a `per_period` dim (engine restriction).   |
| `cardinality` | `[min, max]`        | yes      | —       | Inclusive ints. Each `left` entity associates with `min..max` `right` rows.   |
| `driver`      | metric name         | no       | `null`  | Validated against declared metrics. **Documentary only** — see note below.     |
| `columns`     | list of bridge cols | no       | `[]`    | Per-row bridge metric columns. Types: `metric.X`, `static.X`, `faker.X` only. |

```yaml
bridges:
  - name: bridge_company_user
    left: dim_company
    right: dim_user
    cardinality: [1, 3]
    driver: engagement
    columns:
      - {name: engagement_share, type: metric.engagement}
```

Bridge rows are static — generated once per run, not per period — so
period-anchored column types (`ref.dim_date`, `timestamp`, etc.) are
rejected by the bridge column translator with a context-rich error.

#### A note on `bridges[*].driver`

`driver` is **documentary**. The interpreter sets
`BridgeTableConfig.trajectory_driven=True` whenever bridges are declared
on a builder config — independent of whether `driver` is set or which
metric it names. Bridge cardinality sampling reads the entity's
trajectory position directly, not a specific metric column.

The field exists to:

  1. Validate (the `driver` name must be a declared metric — typos surface
     at build time, not at runtime).
  2. Self-document (a future maintainer reading the YAML can see *which*
     metric the author had in mind for the bias).

Two consequences:

  * Omitting `driver` does **not** disable trajectory-biased sampling on
    the builder path. To get uniform random cardinality, drop into
    engine-direct YAML and set `trajectory_driven: false`.
  * Changing `driver: engagement` to `driver: mrr` on the same archetype
    set produces byte-identical bridge output. The bias source is the
    archetype curve, not the named metric.

### 2.9 `quality[*]` (optional)

Post-generation data-quality corruption. Each entry maps 1:1 to one
engine `QualityIssue`.

| Field         | Type                                                                                       | Required           | Default | Notes                                                            |
|---------------|--------------------------------------------------------------------------------------------|--------------------|---------|------------------------------------------------------------------|
| `table`       | table name                                                                                 | yes                | —       | Engine validates the table exists and the column belongs to it.  |
| `issue`       | `null_injection` \| `duplicate_rows` \| `type_mismatch` \| `late_arrival` \| `schema_drift` | yes                | —       |                                                                  |
| `column`      | column name                                                                                | conditional        | —       | Required for `null_injection`, `type_mismatch`, `schema_drift`. Ignored for `duplicate_rows` and `late_arrival` (omit → engine `"*"` auto-expand sentinel). |
| `rate`        | float `[0, 1]`                                                                             | yes                | —       |                                                                  |
| `seed_offset` | int >= 0                                                                                   | no                 | `0`     | Distinct offsets keep multi-issue runs' affected rows independent. |

```yaml
quality:
  - table: fct_revenue
    column: mrr
    issue: null_injection
    rate: 0.05
  - table: fct_engagement
    issue: duplicate_rows
    rate: 0.02
```

Note: the engine forbids combining `quality` with `entity_features` or
`holdout` — corruption ordering would silently change downstream
aggregates / splits. Pick one.

### 2.10 `holdout` (optional)

Temporal train/holdout split for ML target workflows. Maps to
`HoldoutConfig(enabled=True)`.

| Field                  | Type           | Required | Default | Notes                                                                                                        |
|------------------------|----------------|----------|---------|--------------------------------------------------------------------------------------------------------------|
| `target`               | metric name    | yes      | —       | Engine validates the metric resolves to a numeric (`int`/`float`) column on a fact table.                    |
| `periods`              | int >= 1       | yes      | —       | Last N periods land in the `_holdout` slice; earlier periods in `_train`.                                    |
| `min_training_periods` | int >= 1       | no       | `3`     | Floor on `n_periods - periods`. Lower → engine raises at config load.                                        |

```yaml
holdout:
  target: mrr
  periods: 6
```

### 2.11 `entity_features` (optional)

Per-entity flat feature table — one row per entity, columns are
`{metric}_{mean,std,slope,first,last,peak_period}` aggregations.
Maps to `EntityFeaturesConfig(enabled=True)`.

Boolean shorthand:

```yaml
entity_features: true   # enabled, all numeric metrics, labels on
```

Dict form for explicit settings:

| Field            | Type             | Required | Default | Notes                                                                                  |
|------------------|------------------|----------|---------|----------------------------------------------------------------------------------------|
| `metrics`        | list of names    | no       | `[]`    | Empty → every numeric fact metric. Names must reference declared metrics.              |
| `include_labels` | bool             | no       | `true`  | Emits `archetype` + `final_trajectory_position` label columns (read from manifest).    |

```yaml
entity_features:
  metrics: [mrr, engagement]
  include_labels: true
```

Engine constraints (raised at PlotsimConfig load):

- `manifest.include` must be true (labels read from the manifest payload).
- `quality.quality_issues` must be empty (entity features aggregate
  pre-corruption tables).

When combined with `holdout`, aggregations restrict to the training
window and drop the target metric's columns to prevent label leakage.

---

### 2.12 `seasonality` (optional)

Global calendar-month modulation. Each entry names a set of months and
a strength multiplier; effects sum at each period before per-metric and
per-segment `seasonal_sensitivity` apply.

| Field      | Type            | Required | Default | Notes                                                                                  |
|------------|-----------------|----------|---------|----------------------------------------------------------------------------------------|
| `months`   | list of int     | yes      | —       | Month numbers `1..12`, unique within one effect. Min length 1, max 12.                  |
| `strength` | float           | yes      | —       | Added to `1.0` at each matching month before metric multiplication.                     |

```yaml
seasonality:
  - { months: [11, 12], strength: 0.30 }    # +30% in Nov-Dec
  - { months: [6, 7, 8], strength: -0.10 }  # -10% in summer
```

Composition formula at each `(entity, period, metric)` cell:

```
seasonal_factor = global_strength × metric_sensitivity × segment_sensitivity
modulated_center = distribution_center × (1 + seasonal_factor)
```

Empty default (`seasonality: []`) produces output identical to
pre-seasonality runs. The `seasonal` archetype shape is independent —
it ships oscillation as a curve, not a global multiplier.

---

## 3. Vocabulary

All vocabulary lookup dicts are exported by
`plotsim.builder.schema` for programmatic consumers — keys match the
recipe dicts in `plotsim/builder/recipes.py`.

### 3.1 Metric types — `METRIC_TYPES`

| Word     | Engine distribution                          | When to use                                              |
|----------|----------------------------------------------|----------------------------------------------------------|
| `score`  | `beta(2, 5)`                                 | Bounded `[0, 1]` rates: engagement, adoption, risk.      |
| `amount` | `lognorm(s=0.85)` if zero-touching or hi/lo ratio ≥ 10, else `beta(2, 5)` | Currency / quantity in a declared range. |
| `count`  | `poisson(λ=5)`                               | Non-negative integer event counts (tickets, logins).     |
| `index`  | `normal(mu=midpoint, sigma=range/6)`         | Signed centered metric (NPS, sentiment).                 |

The `amount` ratio threshold (`AMOUNT_LOGNORM_RATIO_THRESHOLD = 10.0`) and
the `index` sigma fraction (`1/6` — three-sigma fits the declared range)
live in `recipes.py`.

### 3.2 Shape words — `SHAPE_WORDS` ↔ `SHAPE_RECIPES`

| Word               | Curve segments                                             | Shape                                |
|--------------------|------------------------------------------------------------|--------------------------------------|
| `growth`           | `sigmoid(midpoint=0.5, steepness=6, rising=True)`          | Smooth S-curve rise.                 |
| `decline`          | `exp_decay(rate=2)`                                        | Exponential fade.                    |
| `seasonal`         | `oscillating(period=2, amplitude=0.4, center=0.5)`         | Two oscillation cycles.              |
| `flat`             | `plateau(level=0.15)`                                      | Low constant.                        |
| `spike_then_crash` | `sigmoid → step(threshold=0.5, before=1, after=0.2) → plateau(level=0.2)` | Rapid rise, sharp drop, low plateau. |
| `accelerating`     | `compound(base_rate=0.05, acceleration=0.02)`              | Compound growth.                     |

### 3.3 Relationship words — `RELATIONSHIP_WORDS` ↔ `RELATIONSHIP_RECIPES`

| Word            | Coefficient | Meaning                              |
|-----------------|-------------|--------------------------------------|
| `mirrors`       | +0.75       | Nearly the same signal.              |
| `driven_by`     | +0.55       | Strong positive link.                |
| `related`       | +0.40       | Moderate positive.                   |
| `hints_at`      | +0.20       | Weak positive.                       |
| `independent`   |  0.00       | No relationship (interpreter drops it). |
| `hints_against` | −0.20       | Weak inverse.                        |
| `resists`       | −0.40       | Moderate inverse.                    |
| `opposes`       | −0.55       | Strong inverse.                      |
| `inverts`       | −0.75       | Nearly mirror-opposite.              |

`independent` (coefficient 0) is silently dropped by the interpreter — the
engine warns on explicit-zero pairs (`RedundantCorrelationWarning`) and
unlisted pairs already get zero off-diagonal. List them only when the
absence is meaningful documentation.

### 3.4 Baseline words — `BASELINE_WORDS` ↔ `BASELINE_RECIPES`

| Word   | Range fraction         | Effect                                                            |
|--------|------------------------|-------------------------------------------------------------------|
| `high` | upper third (`2/3`–`1`)| Restricts the segment's value range to the upper third.           |
| `mid`  | middle third (`1/3`–`2/3`) | Middle third (the implicit default if `baseline` is omitted).  |
| `low`  | lower third (`0`–`1/3`)| Lower third.                                                      |

Baselines apply per-metric per-segment via a `MetricOverride.value_range`
on the archetype. Skipped when the metric has no value range (`count`).

---

## 4. Composite archetype DSL

### 4.1 Grammar

```
spec      ::= shape ( ">" shape )* ( "@" period )*
shape     ::= one of SHAPE_WORDS keys
period    ::= integer in [1, n_periods - 1]
```

### 4.2 Rules

- The number of `@` tokens must equal the number of `>` tokens. A spec
  with N shapes carries N − 1 transition periods. A single-shape spec
  carries zero `@` tokens and covers the whole `[0.0, 1.0]` window.
- Periods are strictly ascending; each lies in `[1, n_periods − 1]`.
- `+` is reserved for layered patterns — rejected at parse time. Use `>`
  for sequential composition.
- Multi-segment shapes (`spike_then_crash`) are rescaled into the phase
  window assigned to the shape, preserving internal sub-segment ratios.

### 4.3 Examples

| Spec                                          | Result                                                                    |
|-----------------------------------------------|---------------------------------------------------------------------------|
| `growth`                                      | 1 segment, sigmoid covering `[0.0, 1.0]`.                                  |
| `flat > decline @ 12`                         | 2 phases at `[0.0, 0.5]` and `[0.5, 1.0]` for a 24-period window.          |
| `growth > seasonal @ 6`                       | Sigmoid for the first 6 periods, then 2 cycles of oscillation.            |
| `decline > flat > growth @ 6 @ 14`            | 3 phases on a 24-period window: drop, plateau, rise.                       |
| `growth > spike_then_crash > flat @ 8 @ 16`   | 5 segments — `spike_then_crash` rescaled into `[1/3, 2/3]` of the window.  |

The parser validates the spec against the actual window length at
`UserInput` construction time, so a `@ 25` transition on a 24-period
window raises immediately.

---

## 5. Column types — `COLUMN_TYPES`

The column-type vocabulary the interpreter handles in
`_translate_column`. Each row maps the builder type to the engine's
`(dtype, source)` pair.

| Type             | Sub-fields            | Engine `(dtype, source)`                                                | Where allowed                  |
|------------------|-----------------------|-------------------------------------------------------------------------|--------------------------------|
| `id`             | —                     | `(id, pk)`                                                              | dim, fact, event               |
| `ref.{dim}`      | —                     | `(id, fk:{dim}.{dim_pk})`                                               | dim, fact, event               |
| `metric.{name}`  | —                     | `(int|float, metric:{name})` — `int` for poisson metrics, else `float`. | fact (engine wires from here)  |
| `faker.{kind}`   | —                     | `(string, generated:faker.{kind})` — `int` for `faker.year`.            | dim, event                     |
| `static.{value}` | —                     | `(float, static:{value})` if numeric literal, else `(string, …)`.       | dim, fact                      |
| `segment.count`  | —                     | `(int, pool:cohort_size)` + `value_pool` keyed by entity                | dim                            |
| `pool.{attr}`    | —                     | `(string, pool:{attr})` + `value_pool` keyed by entity                  | dim (per_entity only)          |
| `timestamp`      | —                     | `(date, generated:timestamp)`                                           | event                          |
| `flag`           | —                     | `(boolean, threshold:{metric}:{above|below}:{value}:for:{periods})`     | event (threshold trigger only) |
| `bucket`         | `labels` (list[str])  | `(string, text:bucket:[labels])`                                        | fact                           |
| `scd`            | `tracks`, `tiers`, `at` | `(string, scd_type2)` + `SCDType2Config(trigger_metric, thresholds, labels)` | dim |
| `date`           | —                     | `(date, generated:date_key)` — **dim_date columns only**.               | dim_date                       |
| `int`            | —                     | `(int, generated:date_key)` — **dim_date columns only**.                | dim_date                       |
| `string`         | —                     | `(string, generated:date_key)` — **dim_date columns only**.             | dim_date                       |
| `float`          | —                     | `(float, generated:date_key)` — **dim_date columns only**.              | dim_date                       |

### 5.1 SCD sub-fields

| Field    | Type        | Meaning                                                                 |
|----------|-------------|-------------------------------------------------------------------------|
| `tracks` | metric name | The metric whose value drives band changes.                             |
| `tiers`  | list[str]   | Ordered band labels, low → high.                                        |
| `at`     | list[float] | Ascending threshold cuts in trajectory space; `len(at) == len(tiers)-1`.|

### 5.2 Bucket sub-field

| Field    | Type      | Meaning                                                                                  |
|----------|-----------|------------------------------------------------------------------------------------------|
| `labels` | list[str] | Quantile-binned bucket labels, low → high. The engine bins by realized metric quantile.  |

---

## 6. Event triggers

### 6.1 `proportional`

Row count per entity per period = `driver_metric_value × scale`, rounded.
The interpreter encodes this as `row_count_source = "proportional:{driver}:scale:{scale}"`.

### 6.2 `threshold`

Fires once per entity at the first period where the named `metric` has
been on the named side of the threshold for `for_periods` consecutive
periods. The flag column emits `True` at the firing period; subsequent
crossings do not re-fire (see `engine-internals.md` §4.4).

---

## 7. Auto-generated schema

When `dimensions`, `facts`, and `events` are all omitted, the interpreter
generates a minimal viable schema:

| Table          | Grain                  | Columns                                                                    |
|----------------|------------------------|----------------------------------------------------------------------------|
| `dim_date`     | `per_period`           | `date_key` (id), `date`, `year`, `month`, `quarter`.                       |
| `dim_{unit}`   | `per_entity`           | `{unit}_id` (id), `{unit}_name` (faker — see §7.1).                         |
| `fct_{unit}`   | `per_entity_per_period`| `date_key`, `{unit}_id`, plus one column per declared metric.              |

### 7.1 Unit-to-faker map

| `unit` value             | Faker kind        |
|--------------------------|-------------------|
| `company`                | `faker.company`   |
| `employee`               | `faker.name`      |
| `customer`               | `faker.name`      |
| (any other word)         | `faker.company`   |

To override the auto-generated schema, declare any of `dimensions`,
`facts`, or `events` explicitly — the auto-path is all-or-nothing.

---

## 8. Errors and warnings

The full catalog lives in [`builder-errors.md`](./builder-errors.md). Two
quick-reference summaries:

- **Errors** are `pydantic.ValidationError` (or `ArchetypeParseError`
  wrapped in one) raised at `UserInput` construction time. They block
  `create()` / `create_from_yaml()`.
- **Warnings** are `UserWarning` emitted alongside successful
  construction. They flag semantic concerns (short window + seasonal,
  single-segment input, `mirrors`/`inverts` with many metrics) the user
  may have intended.

---

## 9. See also

- [`builder-quickstart.md`](./builder-quickstart.md) — annotated walkthroughs.
- [`builder-errors.md`](./builder-errors.md) — full error and warning catalog.
- [`plotsim/configs/new/saas_template.yaml`](https://github.com/mohossam-ae/plotsim/blob/main/plotsim/configs/new/saas_template.yaml) — full reference template.
- [`plotsim/configs/new/saas_template.py`](https://github.com/mohossam-ae/plotsim/blob/main/plotsim/configs/new/saas_template.py) — Python-API equivalent.
- [`plotsim/configs/new/bare_minimum.yaml`](https://github.com/mohossam-ae/plotsim/blob/main/plotsim/configs/new/bare_minimum.yaml) — minimal working config.
