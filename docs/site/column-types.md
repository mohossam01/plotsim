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
| `static.{value}` | yes | yes | yes | yes |
| `pool.{attr}` | yes (per-entity dim only) | — | — | — |
| `segment.count` | yes (per-entity dim only) | — | — | — |
| `timestamp` | — | — | yes | — |
| `flag` | — | — | yes (threshold trigger only) | — |
| `bucket` | yes | yes | yes | — |
| `scd` | yes (per-entity dim only) | — | — | — |
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

**Valid on**: per-entity dimension columns only. Fact, event, and
reference-dim columns can't pull from per-segment attributes —
attribute values are an entity-level fact, not a per-row one.

**Coverage** — every segment must declare the attribute. A `pool.region`
column rejects at construction if even one segment omits `region`. The
error message lists the attributes declared on every segment.

**Scalars vs lists** — segments can declare a scalar (`region: "us-east"`)
or a list (`industry: ["tech", "finance"]`). Scalar values wrap into a
single-element list; list values are sampled uniformly per row.

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
