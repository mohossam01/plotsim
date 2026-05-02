# Builder Quickstart

> Two annotated walkthroughs — a bare-minimum config and the full saas
> reference template — to get from "I want a dataset" to "I have CSVs"
> in under five minutes.
>
> The companion vocabulary reference is [`builder-reference.md`](./builder-reference.md).
> The full validation surface is [`builder-errors.md`](./builder-errors.md).

---

## 1. Install and run

```bash
pip install plotsim
```

```python
import numpy as np
from plotsim import create_from_yaml, generate_tables, validate, write_tables

cfg = create_from_yaml("plotsim/configs/new/bare_minimum.yaml")
tables = generate_tables(cfg, np.random.default_rng(cfg.seed))
report = validate(cfg, tables)
write_tables(tables, cfg, report)  # default output dir: ./output
```

Same flow with kwargs instead of YAML:

```python
from plotsim import create
cfg = create(
    about="Subscription customers",
    unit="customer",
    window=("2024-01", "2024-12", "monthly"),
    metrics=[
        {"name": "engagement", "type": "score", "polarity": "positive"},
        {"name": "payments",   "type": "count", "polarity": "positive"},
    ],
    segments=[
        {"name": "active",   "count": 50, "archetype": "growth"},
        {"name": "inactive", "count": 30, "archetype": "decline"},
    ],
)
```

---

## 2. Bare-minimum walkthrough

The smallest config that loads. Required fields only — everything else
is filled in by the interpreter.

```yaml
# plotsim/configs/new/bare_minimum.yaml

about: "Subscription customers"
unit: customer

window:
  start: 2024-01
  end: 2024-12
  every: monthly

metrics:
  - name: engagement
    type: score
    polarity: positive

  - name: payments
    type: count
    polarity: positive

segments:
  - name: active
    count: 50
    archetype: growth

  - name: inactive
    count: 30
    archetype: decline
```

### What the interpreter generates

| Block         | What you wrote                           | What the interpreter fills in                                                              |
|---------------|------------------------------------------|--------------------------------------------------------------------------------------------|
| `Domain`      | `about`, `unit`                          | `name = "Subscription customers"`, `entity_type = "customer"`, `entity_label = "Customers"`. |
| `TimeWindow`  | `window`                                 | 12 monthly periods, `granularity = "monthly"`.                                              |
| `Metric` × 2  | `name`, `type`, `polarity`               | `engagement → beta(2, 5)` with range `[0, 1]`; `payments → poisson(λ=5)` with no range.     |
| `Archetype` × 2 | `archetype: growth`, `archetype: decline` | `growth → sigmoid(midpoint=0.5, steepness=6)`; `decline → exp_decay(rate=2)`.            |
| `Entity` × 2  | `segments[*].count`                      | One entity per segment, `size = count`. Entity name = segment name.                        |
| `connections` | omitted                                  | Empty correlation list — engine uses zero off-diagonal.                                    |
| `lifecycle`   | omitted                                  | `stages = None` — no stage column on the fact tables.                                      |
| `tables`      | omitted                                  | Auto-generated: `dim_date` + `dim_customer` + `fct_customer` (one column per metric).      |
| `seed`        | omitted                                  | Drawn from `secrets.randbelow(2**32)` — written into the config and the manifest.          |
| `output`      | omitted                                  | `format="csv", directory="output"`.                                                        |

### Auto-generated schema in detail

Because `dimensions`, `facts`, and `events` were all empty, the interpreter
emits a minimal viable three-table schema:

```
dim_date
  date_key (id, pk)             # primary key
  date    (date,   generated:date_key)
  year    (int,    generated:date_key)
  month   (int,    generated:date_key)
  quarter (int,    generated:date_key)

dim_customer
  customer_id   (id,     pk)
  customer_name (string, generated:faker.name)   # `customer` → faker.name; unknown units fall back to faker.company

fct_customer
  date_key   (id,    fk:dim_date.date_key)
  customer_id (id,   fk:dim_customer.customer_id)
  engagement  (float, metric:engagement)
  payments    (int,   metric:payments)         # poisson → int dtype
```

Grain notes:

- `dim_date` — one row per period (12 rows for a year of monthly data).
- `dim_customer` — one row per *cohort* (Entity), not per individual record.
  With two segments you get two rows. The `count` field on each segment
  controls the number of *participants* a cohort represents at the engine
  level (e.g. for proportional event row counts).
- `fct_customer` — `cohorts × periods` rows (2 × 12 = 24).

To get per-individual records, declare a `dim_user` (or similar) with a
`ref.dim_customer` column — see the saas template for an example.

---

## 3. Full reference walkthrough — `saas_template.yaml`

The full template at [`plotsim/configs/new/saas_template.yaml`](https://github.com/mohossam-ae/plotsim/blob/main/plotsim/configs/new/saas_template.yaml)
exercises every accepted shape: 6 metrics, 6 segments with composite
archetypes and baselines, 3 connections, a 4-stage lifecycle, an
explicit schema with SCD and bucket columns, and two events.

### 3.1 Metrics — all four types

```yaml
metrics:
  - name: engagement                                    # score: bounded [0, 1] — beta(2, 5)
    type: score
    polarity: positive

  - name: mrr                                           # amount: declared range — lognorm
    type: amount
    polarity: positive
    range: [100, 50000]

  - name: support_tickets                               # count: poisson — no range allowed
    type: count
    polarity: negative
    follows: engagement                                 # causal lag — driver metric
    delay: 2                                            # … by 2 periods

  - name: feature_adoption                              # another score
    type: score
    polarity: positive

  - name: churn_risk                                    # score with negative polarity
    type: score                                         # high trajectory → low churn-risk
    polarity: negative

  - name: nps                                           # index: signed centered metric
    type: index
    polarity: positive
    range: [-100, 100]
```

**What the interpreter does with `range: [100, 50000]`:**
`50000 / 100 = 500 ≥ 10`, so `mrr` is mapped to a lognormal distribution
with `s=0.85`, `scale=midpoint`, not a beta.

**What `follows`/`delay` does:**
Builds a `CausalLag(driver="engagement", lag_periods=2)` on the engine-
side `Metric`. The trajectory of `support_tickets` is shifted to lag
`engagement` by two periods — a drop in engagement at month 8 surfaces
as elevated support tickets at month 10.

### 3.2 Connections — three correlation pairs

```yaml
connections:
  - engagement driven_by mrr        # +0.55
  - engagement opposes churn_risk   # −0.55
  - support_tickets related churn_risk   # +0.40
```

The 3-token string form. The interpreter parses each into a
`CorrelationPair` and feeds them to the engine's Gaussian copula. Pairs
not listed default to zero off-diagonal — no need to enumerate
"independent" pairs.

### 3.3 Segments — composite archetypes + baselines

```yaml
segments:

  - name: promising_client
    count: 20
    archetype: growth > spike_then_crash > flat @ 8 @ 16     # 3-phase composite
    label: "Strong start, lost champion at month 8, went dormant by 16"
    attributes:
      industry: [Technology, Finance, Healthcare]
      region: [US, EMEA]
      tier: enterprise
    baseline:                                                # high baseline on these metrics
      mrr: high
      engagement: high
      support_tickets: low

  - name: steady_enterprise
    count: 25
    archetype: growth                                        # single-phase
    baseline:
      mrr: high
      engagement: high
      support_tickets: low
```

The archetype `growth > spike_then_crash > flat @ 8 @ 16` is parsed into
a 5-segment curve over a 24-period window:

| Phase | Range (periods) | Segments                                                |
|-------|-----------------|---------------------------------------------------------|
| 1     | `[0, 8)`        | `growth` — sigmoid rise.                                |
| 2     | `[8, 16)`       | `spike_then_crash` rescaled into `[1/3, 2/3]`: a sigmoid → step → plateau triplet. |
| 3     | `[16, 24]`      | `flat` — low plateau at 0.15.                           |

`baseline: {mrr: high}` becomes
`MetricOverride(value_range=ValueRange(min=33433, max=50000))` for this
archetype — restricted to the upper third of mrr's `[100, 50000]` range.

### 3.4 Lifecycle — free-mode stages

```yaml
lifecycle:
  track: churn_risk
  stages:
    - onboarding: 0.0
    - active: 0.2
    - at_risk: 0.5
    - churned: 0.8
```

Translates to a `StageSequence` with `enforce_order=False` (M115 settled
default) — entities can re-enter any stage at any time when their
`churn_risk` trajectory crosses a threshold. To force one-way ladders,
use the engine-level config directly.

### 3.5 Schema — explicit dimensions, facts, events

The template declares all three explicitly. Highlights:

```yaml
dimensions:

  - name: dim_date
    per: period
    columns:
      - {name: date_key,  type: id}
      - {name: date,      type: date}        # date dtype — dim_date columns only
      - {name: year,      type: int}         # int dtype — dim_date columns only
      - {name: month,     type: int}
      - {name: quarter,   type: int}

  - name: dim_company
    per: unit
    columns:
      - {name: company_id,    type: id}
      - {name: company_name,  type: faker.company}
      - {name: industry,      type: faker.industry}
      - {name: founded_year,  type: faker.year}      # faker.year → int dtype
      - {name: cohort_size,   type: segment.count}   # engine fills with the cohort row count
      - name: plan_tier                              # SCD Type 2 column
        type: scd
        tracks: mrr                                  # band changes when mrr crosses thresholds
        tiers: [starter, growth, enterprise]
        at: [0.4, 0.7]                               # in trajectory space (0–1)

  - name: dim_user                                   # sub-entity dim
    per: unit
    columns:
      - {name: user_id,     type: id}
      - {name: company_id,  type: ref.dim_company}   # FK to parent dim
      - {name: user_name,   type: faker.name}
      - {name: role,        type: static.member}     # fixed string for every row

  - name: dim_plan                                   # reference dim — static lookup
    reference: true
    columns:
      - {name: plan_id,       type: id}
      - {name: plan_name,     type: static.starter}
      - {name: monthly_price, type: static.99.00}    # numeric literal → float dtype
```

```yaml
facts:

  - name: fct_engagement
    metrics: [engagement, feature_adoption]
    columns:
      - {name: date_key,           type: ref.dim_date}
      - {name: company_id,         type: ref.dim_company}
      - {name: engagement_score,   type: metric.engagement}
      - {name: feature_adoption,   type: metric.feature_adoption}
      - name: customer_sentiment                                     # bucket column
        type: bucket
        labels: [at_risk, lukewarm, satisfied, delighted]
```

The fact's primary key is the composite of all `ref.X` columns whose
target dim is **not** a reference dim. So `fct_engagement.PK = (date_key, company_id)`,
even when `plan_id` would also be present.

```yaml
events:

  - name: evt_login
    trigger: proportional
    driver: engagement                                # row count per period = engagement × scale
    scale: 5
    columns:
      - {name: event_id,    type: id}
      - {name: date_key,    type: ref.dim_date}
      - {name: user_id,     type: ref.dim_user}
      - {name: company_id,  type: ref.dim_company}
      - {name: event_ts,    type: timestamp}

  - name: evt_churn
    trigger: threshold
    metric: churn_risk
    above: 0.7                                        # fires when churn_risk > 0.7 …
    for: 3                                            # … sustained for 3 periods
    columns:
      - {name: event_id,      type: id}
      - {name: date_key,      type: ref.dim_date}
      - {name: company_id,    type: ref.dim_company}
      - {name: churn_reason,  type: faker.sentence}
      - {name: churn_flag,    type: flag}             # boolean — only valid in threshold events
```

`for` is an alias for `for_periods` — both names are accepted in YAML.
The Python kwargs surface uses `for_periods` (the YAML alias avoids the
Python reserved word).

### 3.6 What's required vs optional in this template

| Field type        | Required (omission raises)                                  | Optional refinement (omission accepts)                  |
|-------------------|-------------------------------------------------------------|---------------------------------------------------------|
| Top-level         | `about`, `unit`, `window`, `metrics`, `segments`            | `connections`, `lifecycle`, `dimensions`, `facts`, `events` |
| Per metric        | `name`, `type`, `polarity` (and `range` for amount/index)   | `label`, `range` (for score), `follows`, `delay`        |
| Per segment       | `name`, `count`, `archetype`                                | `label`, `attributes`, `baseline`                       |
| Per connection    | `metric_a`, `relationship`, `metric_b` (or 3-token string)  | —                                                       |
| Per dim           | `name`, `columns`                                           | `per`, `reference`, `count`                             |
| Per fact          | `name`, `columns`                                           | `metrics` (documentary)                                 |
| Per event         | `name`, `columns`, `trigger`, plus trigger-specific fields  | `for_periods` (defaults to 1)                           |

---

## 4. From config to disk

```python
import numpy as np
from plotsim import create_from_yaml, generate_tables, validate, write_tables

cfg = create_from_yaml("plotsim/configs/new/saas_template.yaml")
rng = np.random.default_rng(cfg.seed)
tables = generate_tables(cfg, rng)
report = validate(cfg, tables)
out_dir = write_tables(tables, cfg, report)
print(out_dir)   # → ./output/<run_timestamp>/
```

Determinism: the same `(cfg, seed)` produces byte-identical CSV / Parquet
output, manifest, and validation report. The seed is captured on `cfg`
when the builder draws it; copy that value to reproduce the same dataset
from a fresh interpreter call.

---

## 5. Next steps

- Read [`builder-reference.md`](./builder-reference.md) for the complete vocabulary, the field table, and the auto-generation rules.
- Read [`builder-errors.md`](./builder-errors.md) before debugging a failing config — every structural error and warning is enumerated there with a triggering example.
- Read the [Config field reference](./config-reference.md) and [Column types](./column-types.md) when you need the full input field map.
