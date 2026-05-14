# Column Types

> Every column type the builder accepts on `dimensions`, `facts`,
> `events`, and `bridges` — what it produces, where it's valid, and a
> schema snippet. The companion field reference is
> [`config-reference.md`](./config-reference.md).

---

## How columns work

Every column declaration has two required fields:

```yaml
- name: <column_name>
  type: <type_string>
```

Some types take additional fields (`labels` for `bucket`, `tracks` /
`tiers` / `at` for `scd`).

---

## Quick map — where each type is valid

|  Type | Dimension | Fact | Event | Bridge |
|---|---|---|---|---|
| `id` | yes (PK) | — | — | — |
| `ref.{table}` | yes (FK) | yes (FK) | yes (FK) | yes (FK)* |
| `metric.{name}` | — | yes | — | yes |
| `faker.{kind}` | yes | yes | yes | yes |
| `geo.{field}` | yes (dim only) | — | — | — |
| `static.{value}` | yes | yes | yes | yes |
| `pool.{attr}` | yes (per-entity dim only) | yes (variable-grain + per_parent_row) | yes | — |
| `range` | — | yes | yes | — |
| `segment.count` | yes (per-entity dim only) | — | — | — |
| `timestamp` | — | — | yes | — |
| `flag` | — | — | yes (threshold trigger only) | — |
| `bucket` | yes | yes | yes | — |
| `narrative` | — | yes (per_entity_per_period only) | — | — |
| `scd` | yes (per-entity dim only) | — | — | — |
| `struct` | yes | yes | — | — |
| `array` | yes | yes | — | — |
| `date` / `int` / `string` / `float` | yes (`dim_date` only) | — | — | — |

\* Bridge `ref.` columns are auto-generated for the two endpoints; you
typically don't declare them by hand.

---

## `id`

Marks the table's primary-key column. Exactly one per dim table.

```yaml
- { name: customer_id, type: id }
```

The value is a zero-padded integer string, padded wide enough to keep
lexicographic order matching numeric order across the row count. Useful
for SQL imports where IDs need a stable string representation.

---

## `ref.{table}`

Foreign key to another table's primary key.

```yaml
- { name: customer_id, type: ref.dim_customer }
- { name: date_key,    type: ref.dim_date }
- { name: dept_id,     type: ref.dim_department }
```

The target table's PK column is discovered automatically — you don't
need to spell it out. Both auto-generated dims (`dim_date`, `dim_<unit>`)
and your own declared dims are valid targets.

By default the engine samples uniformly across the parent's PK values.

---

## `metric.{name}`

A column whose value is the realized value of the named metric for the
row's `(entity, period)` pair.

```yaml
- { name: engagement, type: metric.engagement }
- { name: mrr,        type: metric.mrr }
```

Output dtype is `int` for poisson (`count`-type) metrics and `float`
for everything else.

**Valid on**: facts and bridges only. Bridges treat the column as a
static value derived once per association (no period axis).

The metric must be declared in the top-level `metrics:` array — the
builder raises at construction if `metric.engagement` references a
metric that doesn't exist.

---

## `faker.{kind}`

A faker-generated string (or year integer).

```yaml
- { name: company_name, type: faker.company }
- { name: industry,     type: faker.industry }
- { name: full_name,    type: faker.name }
- { name: launch_year,  type: faker.year }
- { name: reason,       type: faker.sentence }
- { name: keyword,      type: faker.word }
```

Output dtype is `string`, except `faker.year` which produces an `int`.

**Common methods**: `company`, `name`, `industry`, `sentence`, `word`,
`year`, `address`, `email`, `phone_number`, `city`, `country`. Any
provider on the installed `faker` package's locale is accepted.

---

## `geo.{field}`

A row-coherent geo bundle drawn from a curated reference dataset.
Multiple `geo.<field>` columns on the same dim row read from the
same bundle entry, so country / region / city / postcode /
latitude / longitude all agree.

```yaml
- { name: country,      type: geo.country }
- { name: country_code, type: geo.country_code }
- { name: region,       type: geo.region }
- { name: city,         type: geo.city }
- { name: postcode,     type: geo.postcode }
- { name: latitude,     type: geo.latitude }
- { name: longitude,    type: geo.longitude }
```

Output dtype is `float` for `latitude` / `longitude` and `string`
for everything else. `geo.<field>` is dim-only; on facts and
events the engine raises `unsupported generated provider`. See
[Geo hierarchy](./user-guide/geo-hierarchy.md) for the underlying
dataset, determinism, and the bundled `geo_retail` template.

---

## `narrative`

Trajectory- and archetype-driven sentence text on a fact column.
Each row's text is built by sampling per-slot phrases from a
per-archetype lexicon, banded by the row's trajectory position.

```yaml
- name: review_text
  type: narrative
  template: "{opener} {object}. {comment}"
  lexicons:
    promoters:
      opener:
        low:  ["I tried"]
        mid:  ["I am using"]
        high: ["I love"]
      object:
        low:  ["the app"]
        mid:  ["this product"]
        high: ["this product"]
      comment:
        low:  ["Decent start."]
        mid:  ["Glad we picked it."]
        high: ["Highly recommend."]
    detractors:
      # ... one entry per assigned segment ...
```

Output dtype is `string`. Lexicon archetype keys must match the
**segment names** (which equal the engine archetype names in the
builder API). `narrative` is fact-only and per_entity_per_period;
the cell builder forces the scalar fact path because it consumes one
RNG draw per slot per row. See
[Narrative text source](./user-guide/narrative-source.md) for the
lexicon-design playbook, validation gates, and the bundled
`narrative_reviews` template.

---

## `static.{value}`

Constant value across every row.

```yaml
- { name: dataset_version, type: static.v1.0 }
- { name: pi,              type: static.3.14159 }
```

Output dtype is `float` when the value parses as a number, `string`
otherwise.

For variable timestamps use `timestamp`; for a fixed date you typically
don't need a static — declare the field on `dim_date` or a reference
dim instead.

---

## `pool.{attr}`

Per-entity value pool, one of the strings declared on the segment's
`attributes` map.

```yaml
segments:
  - name: enterprise
    count: 30
    archetype: growth
    attributes:
      industry: ["tech", "finance"]
      region: "us-east"
  - name: smb
    count: 50
    archetype: decline
    attributes:
      industry: ["retail", "services"]
      region: "global"

dimensions:
  - name: dim_customer
    columns:
      - { name: customer_id, type: id }
      - { name: industry,    type: pool.industry }
      - { name: region,      type: pool.region }
```

Output dtype is `string`.

**Valid on**: per-entity dimension columns, variable-grain fact
columns, per_parent_row child-fact columns, and event columns. The
engine reads the row's entity FK and draws from
`attributes[attr_name]` for that entity's segment.
Per_entity_per_period and per_period facts, reference dims, and
sub-entity dims are out of scope — the M19 lift required either a
per-row entity binding (facts / events) or a 1:1 row-to-entity
mapping (per_entity dim).

**Coverage** — every segment must declare the attribute. A `pool.region`
column rejects at construction if even one segment omits `region`. The
error message lists the attributes declared on every segment.

**Scalars vs lists** — segments can declare a scalar (`region: "us-east"`)
or a list (`industry: ["tech", "finance"]`). Scalar values wrap into a
single-element list; list values are sampled uniformly per row.

---

## `range`

Per-row uniform draw between two numeric bounds. Use it when a
column needs a bounded random number with explicit limits rather
than Faker's defaults.

```yaml
facts:
  - name: fct_orders
    row_count_driver: order_volume
    row_count_scale: 1.0
    columns:
      - { name: order_id,    type: id }
      - { name: customer_id, type: ref.dim_customer }
      - { name: order_date,  type: ref.dim_date }
      # Inclusive [1, 5] integer draw — quantity ∈ {1, 2, 3, 4, 5}.
      - { name: quantity,    type: range, range: [1, 5] }
      # Uniform float in [10.0, 500.0).
      - { name: unit_price,  type: range, range: [10.0, 500.0] }
```

**Bounds dtype** — both bounds in `range: [...]` must be numeric.
Integer-typed bounds produce `dtype: int` and draw via
`rng.integers(min, max + 1)` (inclusive upper bound). Any float
bound produces `dtype: float` and draws via `rng.uniform(min, max)`
(exclusive upper bound — numpy's continuous-range convention).

**Valid on**: variable-grain fact columns, per_parent_row child
fact columns, event columns, and per_entity_per_period fact columns.
Dimension columns are out of scope in this version.

**Deterministic** — every draw goes through the engine RNG so the
same seed yields the same column.

---

## `segment.count`

The original cohort population — useful when you want to surface "this
customer came from a 30-customer growth segment" as a column.

```yaml
- { name: cohort_size, type: segment.count }
```

Output dtype is `int`. Each row carries the count from its segment's
declaration.

**Valid on**: per-entity dimension columns only.

---

## `timestamp`

A wall-clock timestamp for the row's period.

```yaml
- { name: occurred_at, type: timestamp }
```

Output dtype is `date`.

**Valid on**: event columns. Resolves to the start of the period the
row falls in (period for monthly, week-start for weekly, day for daily).

---

## `flag`

Boolean column for threshold-triggered events. Set to True when the
event's threshold condition fires; False otherwise.

```yaml
events:
  - name: evt_churn
    trigger: threshold
    metric: churn_risk
    above: 0.8
    for_periods: 2
    columns:
      - { name: customer_id, type: ref.dim_customer }
      - { name: churn_flag,  type: flag }
```

Output dtype is `boolean`. The threshold the column reports is taken
from the event's own `trigger`/`metric`/`above`/`below`/`for_periods`
fields — you don't repeat them on the column.

**Valid on**: event columns where `trigger: threshold`. Proportional
events don't have a threshold; using `flag` there raises at construction.

---

## `bucket`

Banded text label keyed off the row's trajectory position. With N labels,
position in `[0, 1]` is split into N evenly-spaced bands.

```yaml
- name: sentiment
  type: bucket
  labels: ["churned", "at_risk", "engaged", "delighted"]
```

Output dtype is `string`.

**Valid on**: dim, fact, and event columns. The realized label is fully
determined by trajectory position — the same row always lands in the
same bucket given the same seed.

**Polarity convention** — order labels with the *most favorable*
outcome at the *highest* position, mirroring positive-polarity metrics.
Reverse the list for negative-polarity sentiments. The label list takes
2 to 20 entries.

---

## `scd`

Slowly Changing Dimension Type 2 — versioned dim rows whose label
advances when a tracked metric crosses configured thresholds.

```yaml
dimensions:
  - name: dim_customer
    columns:
      - { name: customer_id, type: id }
      - name: plan_tier
        type: scd
        tracks: mrr
        tiers: ["free", "starter", "pro", "enterprise"]
        at: [0.0, 0.25, 0.50, 0.75]
```

Output dtype is `string`.

**Required sub-fields**:

| Field | Type | Description |
|---|---|---|
| `tracks` | `str` | Metric name whose trajectory drives band changes |
| `tiers` | array of `str` | Band labels, lowest band first |
| `at` | array of `float` | Threshold crossings, ascending, in `[0, 1]`. Same length as `tiers` |

**Valid on**: per-entity dimension columns only. The dim is expanded
into multiple rows per entity (one per band crossing); each fact row
automatically references the active version through a `dim_row_id`
column appended for you.

The tracked metric must already be emitted by some fact table — the
builder rejects `tracks: <metric>` at construction time when no fact
references that metric.

---

## `dim_date` dtype words — `date`, `int`, `string`, `float`

When you want named columns on `dim_date` that aren't auto-generated.

```yaml
dimensions:
  - name: dim_date
    columns:
      - { name: date_key, type: id }
      - { name: date,     type: date }
      - { name: year,     type: int }
      - { name: month,    type: int }
      - { name: quarter,  type: int }
      - { name: fiscal,   type: string }
```

The output dtype matches the word you wrote (`int` → `int`, `date` → `date`).
Each column is derived from the date-key spine — `year`, `month`,
`quarter`, `weekday`, `is_weekend` are all supported via name-matching.

**Valid on**: `dim_date` columns only. Other tables that try to use a
bare dtype word (e.g. `type: int`) raise — non-`dim_date` columns must
declare a source-bearing type (`metric.X`, `faker.X`, `static.X`,
`ref.X`, etc.).

---

## `struct`

A nested column where each cell is a Python `dict` with a fixed set of
typed fields. The cell shape is declared by `nested_schema` mapping
field names to primitive types (`int`, `float`, `string`, `boolean`).

```yaml
- name: metadata
  type: struct
  nested_schema:
    tier_score: int
    is_pilot: boolean
    region_code: string
```

Output:

* **Parquet** — written as a native pyarrow `struct<...>` field,
  preserving the typed schema. Round-trips through `pd.read_parquet`
  as a column of dicts.
* **CSV** — each cell serialised via `json.dumps`, so a row's value
  looks like `{"tier_score": 654, "is_pilot": false, "region_code": "v091"}`.
  Round-trip via `json.loads`.

Engine-direct shape:

```yaml
- name: metadata
  dtype: struct
  source: nested
  nested_schema: { tier_score: int, is_pilot: boolean, region_code: string }
```

The current release supports one level of nesting only — struct field
types are primitive (no struct-of-struct). Field values are drawn
independently per row from a seeded RNG, so the same seed produces
byte-identical nested cells across runs. Need realistic strings? Use a
separate `faker.<method>` column instead — `string` field values
inside a struct are short deterministic tokens (`"v00042"`).

**Valid on**: dim and fact tables. Not supported on event or bridge
tables in the current release.

---

## `array`

A nested column where each cell is a Python `list` of fixed length,
holding values of one primitive type. Declared by `array_element_type`
(required) and optional `array_length` (defaults to 3, capped at 100).

```yaml
- name: tags
  type: array
  array_element_type: string
  array_length: 5
```

Output:

* **Parquet** — written as a native `list<element: ...>` field.
* **CSV** — each cell serialised via `json.dumps`, e.g.
  `["v43301", "v85859", "v08594"]`.

Engine-direct shape:

```yaml
- name: tags
  dtype: array
  source: nested
  array_element_type: string
  array_length: 5
```

Element type is one of `int` / `float` / `string` / `boolean` (no
nested-of-nested in the current release). All cells in a column have
the same length — it's a homogeneous array shape, not a list of
variable-length lists.

**Valid on**: dim and fact tables. Not supported on event or bridge
tables in the current release.

---

## Engine-direct sources

The builder DSL covers the column types most configs need. The engine
also accepts four lower-level `source:` strings that the builder doesn't
surface — useful when authoring or editing an engine-direct YAML
(`PlotsimConfig` shape) directly. All four parse through
`plotsim.config.parse_source` and live on `Column.source`.

### `derived:<field>`

Computed-column source. Copies an already-realized column on the same
row into the new column, applying the declared `dtype` for coercion.
Useful for surfacing a metric value under a second column name without
a redundant fact entry, or for narrowing a wide dtype to a small one.

```yaml
- name: engagement_int
  dtype: int
  source: derived:engagement_score
```

The referenced field must already be present on the row at compute time.

### `lag:<metric>:periods:<N>`

Embed a metric's value from `N` periods ago as a column in its own right
(distinct from the metric's own `causal_lag` declaration, which only
shifts the trajectory used to draw the *current* metric value). Useful
for "previous month MRR" / "last quarter NPS" columns expected by
downstream dashboards.

```yaml
- name: mrr_3mo_ago
  dtype: float
  source: lag:mrr:periods:3
```

`<N>` must be `≥ 1`. Periods before window start return null.

### `threshold:<metric>:<above|below>:<value>:for:<consecutive>`

Event-row driver. Expressed on `Table.row_count_source` rather than a
column — the table emits one event row per `(entity, period)` where the
named metric stays above/below `value` for at least `consecutive`
consecutive periods. Equivalent to the builder's
`events: { trigger: threshold, ... }` block but exposed as a parsed
source string in the engine YAML:

```yaml
- name: evt_churn
  type: event
  row_count_source: threshold:churn_risk:above:0.7:for:3
  columns:
    - { name: company_id, source: fk:dim_company.company_id }
```

Mirrors `flag` columns: pair this `row_count_source` with a `flag` column
inside the table to surface the boolean fired/not-fired marker per row.

### `proportional:<metric>:scale:<multiplier>`

The other event-row driver. Row count per `(entity, period)` =
`metric_value × multiplier`. Scale is capped at `100.0` (engine guards
event-table memory growth at very large multipliers).

```yaml
- name: evt_login
  type: event
  row_count_source: proportional:engagement:scale:5
  columns:
    - { name: company_id, source: fk:dim_company.company_id }
    - { name: event_ts,   source: generated:timestamp }
```

The cell-count gate documented under
[Limits](./config-reference.md#limits-and-performance-gates) does *not*
account for event-row volume — high-scale proportional events on
high-cell configs can still produce surprisingly large event tables.
