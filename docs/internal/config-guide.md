# Config guide

A plotsim config is a YAML file that maps to a frozen Pydantic v2 model. Every field is validated at load time; cross-references (table-to-metric, archetype-to-curve, FK-to-parent) are resolved before generation starts. If your config loads, it will generate.

This page walks through every section, then maps the ten errors you're most likely to see to causes and fixes.

## Mental model

A config has eight top-level sections. Five are required, three are optional.

| Section         | Required | Purpose                                                 |
| --------------- | :------: | ------------------------------------------------------- |
| `domain`        |    ✓     | Display strings — name, entity label                    |
| `time_window`   |    ✓     | Date spine: start, end (YYYY-MM), granularity           |
| `seed`          |    ✓     | Integer — pins all randomness                           |
| `metrics`       |    ✓     | Distributions, polarity, optional causal lag            |
| `archetypes`    |    ✓     | Trajectory curves built from segments                   |
| `entities`      |    ✓     | Cohorts → archetype assignment                          |
| `tables`        |    ✓     | Dim, fact, event schemas with typed columns and FKs     |
| `output`        |    ✓     | Format (`csv` or `parquet`) and target directory        |
| `correlations`  |          | Pairwise metric coefficients (Gaussian copula)          |
| `noise`         |          | Gaussian sigma, outlier rate, MCAR rate                 |
| `stages`        |          | Lifecycle funnel keyed on a metric                      |

Tables are generated in dependency order: dimensions first, then facts, then events. Within each table, every cell value traces back to a single trajectory position computed for that (entity, period). That's the trajectory-first invariant — no metric is ever sampled independently of the entity's archetype.

## Domain and time window

```yaml
domain:
  name: "Tutorial"
  description: "Anything — display only"
  entity_type: "team"
  entity_label: "Teams"

time_window:
  start: "2024-01"      # YYYY-MM
  end: "2024-12"        # YYYY-MM, must be > start
  granularity: "monthly"  # monthly | weekly | daily

seed: 42
```

Span limits: 360 monthly periods, 1,560 weekly, 3,650 daily. Above the cap, load fails — pick a coarser granularity.

## Metrics

Every metric needs a `name`, `label`, `distribution`, `params`, and `polarity`. Six distributions are supported: `lognorm`, `gamma`, `poisson`, `beta`, `normal`, `weibull`.

```yaml
metrics:
  - name: "engagement"
    label: "Product engagement"
    distribution: "beta"
    params: {alpha: 2.0, beta: 5.0}
    polarity: "positive"
    value_range: {min: 0.0, max: 1.0}
```

`polarity` is the most consequential field. `positive` means the metric rises when the trajectory rises; `negative` inverts that — high trajectory position becomes a low value. Churn risk, attrition, and support tickets are negative; revenue, engagement, and feature adoption are positive.

`value_range` is optional but recommended for bounded metrics (rates in [0, 1], scores in [0, 100]).

### Causal lag

A metric can trail another by N periods. Both metrics still read from the trajectory; the lagged one reads from a *past* trajectory position.

```yaml
metrics:
  - name: "support_tickets"
    distribution: "poisson"
    params: {lambda: 5.0}
    polarity: "negative"
    causal_lag:
      driver: "engagement"
      lag_periods: 2
      blend_weight: 1.0    # default — pure period shift
```

`blend_weight: 1.0` (the default) is a pure shift — period T reads driver at T−2. Lower values blend the metric's own current trajectory with the driver's past value. Lags chain: if A→B(lag=2) and B→C(lag=3), C reflects A at lag 5.

Per-granularity caps on `lag_periods`: 120 monthly, 520 weekly, 3,650 daily.

## Archetypes

Archetypes are master curves built from segments. Each segment is one of eight curve types (`sigmoid`, `exp_decay`, `step`, `logistic`, `plateau`, `oscillating`, `compound`, `sawtooth`) over a portion of the [0, 1] trajectory range.

```yaml
archetypes:
  - name: "rocket_then_cliff"
    label: "Grows fast then crashes"
    description: "Sigmoid rise, abrupt drop, low plateau"
    curve_segments:
      - curve: "sigmoid"
        params: {midpoint: 0.3, steepness: 10.0}
        start_pct: 0.0
        end_pct: 0.55
      - curve: "step"
        params: {threshold: 0.5, before: 1.0, after: 0.2}
        start_pct: 0.55
        end_pct: 0.65
      - curve: "plateau"
        params: {level: 0.2}
        start_pct: 0.65
        end_pct: 1.0
```

Segment rules:

- `start_pct` < `end_pct` for every segment
- The first segment must start at `0.0`; the last must end at `1.0`
- Adjacent segments must touch — no gaps, no overlaps (`prev.end_pct == next.start_pct`)
- Up to 10 segments per archetype

A single-segment archetype is fine and common (`steady_grower` is one sigmoid from 0 to 1).

### Per-archetype metric overrides

An archetype can override the distribution, params, and (M114) `value_range` of any metric for entities in cohorts assigned to it:

```yaml
metrics:
  - name: "mrr"
    distribution: "lognorm"
    params: {s: 0.85, scale: 1500.0}
    polarity: "positive"
    value_range: {min: 100, max: 50000}

archetypes:
  - name: "expansion_champion"
    curve_segments: [...]
    metric_overrides:
      mrr:
        distribution: "lognorm"
        params: {s: 0.6, scale: 2500.0}
        # M114: per-archetype value range override.
        # Must be a subset of the global metric value_range —
        # overrides restrict, never expand. Validated at load.
        value_range: {min: 33400, max: 50000}
```

Use this when one cohort's metric values legitimately come from a different distribution, or when one cohort's archetype reaches a different baseline than the rest of the population. The `value_range` override propagates through the full metric pipeline (center, sampling, copula CDF, clamp), so the override is visible in realised cell values even when the metric is correlated with others.

## Entities

```yaml
entities:
  - name: "acme_corp_cohort"
    archetype: "rocket_then_cliff"
    size: 50
```

Each entity is a *cohort*. `size` is the count of sub-entities under that cohort — used for sub-entity dim row counts (`grain: variable`) and for proportional event row counts. The fact tables collapse to one row per cohort per period.

Per-cohort cap: 5,000. Total across all cohorts: 100,000.

## Tables

Three table types: `dim`, `fact`, `event`. Five grains:

| Grain                    | Used by             | Row count                                           |
| ------------------------ | ------------------- | --------------------------------------------------- |
| `per_period`             | `dim_date`          | One row per time step                               |
| `per_entity`             | `dim_<entity>`      | One row per cohort                                  |
| `per_reference`          | `dim_plan`, etc.    | Static lookup (one row per declared static value)   |
| `per_entity_per_period`  | Most fact tables    | Cohorts × periods                                   |
| `variable`               | Sub-entity dims, events | Driven by row_count_source or sub-entity size       |

```yaml
tables:
  - name: "fct_engagement"
    type: "fact"
    grain: "per_entity_per_period"
    columns:
      - {name: "date_key",          dtype: "id",    source: "fk:dim_date.date_key"}
      - {name: "company_id",        dtype: "id",    source: "fk:dim_company.company_id"}
      - {name: "engagement_score",  dtype: "float", source: "metric:engagement"}
    primary_key: ["date_key", "company_id"]
    foreign_keys: ["dim_date.date_key", "dim_company.company_id"]
```

### Column source types

The `source` field tells the engine where a column's values come from. Ten source kinds:

| Source                                                   | Purpose                                             |
| -------------------------------------------------------- | --------------------------------------------------- |
| `pk`                                                     | Primary key                                         |
| `fk:<table>.<column>`                                    | Foreign key into another table                      |
| `metric:<name>`                                           | Sample from a metric at the trajectory position     |
| `lag:<metric>:periods:<N>`                               | Same, but at the trajectory position N periods ago  |
| `static:<value>`                                          | Constant — same value in every row                  |
| `derived:<field>`                                         | Computed (`size`, `entity_id`, `period_index`, etc.) |
| `generated:date_key`, `generated:timestamp`, etc.        | Engine-internal generators                          |
| `generated:faker.<method>[:<key>:<value>]*`              | Faker-backed (names, companies, dates)              |
| `threshold:<metric>:<above\|below>:<value>:for:<N>`     | Boolean — true when condition holds for N periods   |
| `proportional:<metric>:scale:<X>`                        | Row count = metric value × scale (event tables)     |
| `pool:<name>`                                            | M114: per-entity sample from `Column.value_pool[entity_name]` (per_entity dim only) |
| `text:bucket:[<label1>, <label2>, ...]`                  | M105: bucket the trajectory position to a fixed text label set (negative polarity reverses) |
| `scd_type2`                                              | M106: SCD Type 2 band label; requires `Column.scd_type2` config |

### Event tables

Event tables have `grain: variable` and either a `row_count_source` (proportional) or a threshold-source column (threshold).

```yaml
- name: "evt_login"
  type: "event"
  grain: "variable"
  row_count_source: "proportional:engagement:scale:5"
  columns: [...]

- name: "evt_churn"
  type: "event"
  grain: "variable"
  columns:
    - {name: "churn_flag", dtype: "boolean", source: "threshold:churn_risk:above:0.7:for:3"}
```

`scale` cap on proportional sources is 100. Threshold sources require the metric to stay above (or below) the threshold for the configured `consecutive` count.

## Correlations

```yaml
correlations:
  - {metric_a: "engagement", metric_b: "mrr",         coefficient: 0.72}
  - {metric_a: "engagement", metric_b: "churn_risk", coefficient: -0.55}
```

plotsim uses a Gaussian copula to inject configured correlations while preserving each metric's marginal distribution. The matrix is checked for positive semi-definiteness at config load — if your three pairwise coefficients are mutually inconsistent (e.g. all three near ±0.9 in a triangle), load fails.

Empirical tolerance, measured in [`docs/statistical-fidelity.md`](https://github.com/mohossam-ae/plotsim/blob/main/docs/statistical-fidelity.md): 9 of 10 distribution pairings land within ±0.10 of configured Pearson; `lognorm × lognorm` widens to ±0.15 at high magnitudes.

Each pair declared at most once. Order doesn't matter — `(a, b)` and `(b, a)` are the same pair, and a duplicate raises.

## Stages

A lifecycle funnel keyed on one metric. Two assignment modes (free / monotonic) and two threshold semantics (legacy / hysteresis):

```yaml
stages:
  field: "churn_risk"
  sequence:
    - {name: "onboarding", threshold_enter: 0.0, threshold_exit: 0.2}
    - {name: "active",     threshold_enter: 0.2, threshold_exit: 0.5}
    - {name: "at_risk",    threshold_enter: 0.5, threshold_exit: 0.8}
    - {name: "churned",    threshold_enter: 0.8, threshold_exit: null}
  # enforce_order: false  # default — per-period free-mode assignment;
                          # set true for forward-only monotonic walk.
```

**Free mode (default, `enforce_order: false`):** each period independently picks the highest-enter stage the realized value satisfies. Stages can move backward when the value falls. `threshold_exit` and `downgrade_delay` are ignored. Use this when stages should reflect *current* lifecycle state. Irreversible transitions (e.g. tier upgrades) belong in SCD Type 2.

**Monotonic mode (`enforce_order: true`):** forward-only cursor — once an entity enters a later stage it stays there. Two threshold semantics apply only here:

- **Legacy** (`threshold_exit > threshold_enter`, the bundled-template default for monotonic): non-overlap upper-bound semantic — each stage spans `[threshold_enter, threshold_exit)`. Adjacent stages must not overlap (`prev.exit ≤ curr.enter`). Optional `downgrade_delay: N` lets the cursor step back after N consecutive sub-threshold periods.
- **Hysteresis** (`threshold_exit ≤ threshold_enter`): `threshold_enter` is the upward entry threshold; `threshold_exit` is the downward demote threshold. The band `[exit, enter]` keeps an entity in the higher stage on transient dips.

The last stage must have `threshold_exit: null` (terminal). Mixing the two threshold semantics in one sequence is rejected at load.

## Noise

```yaml
noise:
  gaussian_sigma: 0.05    # multiplicative noise around the trajectory-derived center
  outlier_rate: 0.02      # fraction of cells that get a 3×–10× value blowup
  mcar_rate: 0.01         # fraction of cells set to null (missing completely at random)
```

Noise applies *after* correlation injection, so it doesn't disturb the configured Pearson coefficients. Outliers are intentional and are excluded from the trajectory-first invariant by design — see the fidelity contract for the exact accounting.

---

## Common errors and fixes

The ten errors you're most likely to see when authoring a config. Each shows the message text plotsim emits inside the Pydantic `ValidationError`, the cause, and the fix.

### 1. `correlations: correlation matrix is not positive semi-definite`

Three pairwise coefficients can't simultaneously be at the magnitudes you declared. The matrix has a non-positive eigenvalue.

**Fix.** Soften one coefficient. If you want `corr(a, b) = 0.8` and `corr(a, c) = 0.8`, then `corr(b, c)` cannot be near −1 — it has to be at least about +0.28 to keep the eigenvalues non-negative. Tighten the triangle.

### 2. `archetype 'X' curve_segments must start at 0.0` / `must end at 1.0` / `gap or overlap between segments`

Your segments don't tile the [0, 1] range cleanly.

**Fix.** First segment's `start_pct: 0.0`. Last segment's `end_pct: 1.0`. Adjacent segments share their boundary exactly (`prev.end_pct == next.start_pct`).

### 3. `entity 'X' references unknown archetype 'Y'; known: [...]`

The `archetype:` value on an entity doesn't match any name in the `archetypes:` list.

**Fix.** Typo. The list of known archetype names is in the error message.

### 4. `circular causal_lag chain detected involving metric 'X'`

A → B → … → A. The lag dependency graph has a cycle.

**Fix.** Break the cycle. A metric cannot lag itself, even transitively.

### 5. `metric 'X' causal_lag.lag_periods (N) exceeds the 'monthly' granularity cap of 120`

Lag exceeds the granularity-aware ceiling. Caps: 120 monthly, 520 weekly, 3,650 daily.

**Fix.** Either reduce `lag_periods` or switch granularity. A daily config can carry a 365-period lag; a monthly config can't.

### 6. `duplicate correlation entries for unordered pair ('a', 'b'): coefficients X and Y; declare each metric pair at most once`

You declared `(a, b)` and `(b, a)` (or `(a, b)` twice).

**Fix.** Remove one entry. Plotsim treats the pair as unordered.

### 7. `table 'T' column 'C' source 'fk:X.Y' references unknown metric / table 'X'`

The FK target table or the source-string metric reference doesn't exist.

**Fix.** Check the target name. Cross-reference integrity is enforced at load — every `fk:`, `metric:`, `lag:`, `threshold:`, `proportional:` source is checked against the declared metrics and tables.

### 8. `table 'T' column 'C' declares dtype: boolean with metric-source 'metric:X', which produces a continuous value the boolean cast collapses to a near-constant True`

You typed a metric-driven column as `boolean`. `bool(continuous_metric_value)` is `True` for any positive value, so the column carries no information.

**Fix.** Use `dtype: float` (or `int` for poisson) to preserve the metric value, or switch the source to `threshold:<metric>:above:<value>:for:<N>` if a boolean indicator is what you want.

### 9. `Total entity count across all groups is N. Maximum is 100,000.` / `Config produces N cells (entities × periods), which exceeds the maximum of 2,000,000`

You crossed one of the resource ceilings.

**Fix.** Reduce `entities[].size`, shorten `time_window`, or coarsen `granularity`. The config-load summary line on stderr also warns at 500k cells (below the cap, but worth noting for wall-clock).

### 10. `time_window.start (X) must be before end (Y)` / `expected YYYY-MM, got 'V'`

Date format or order issue.

**Fix.** Both `start` and `end` are `YYYY-MM` strings (not dates, not `YYYY-MM-DD`). `end` is exclusive of the end month for daily, inclusive for monthly/weekly — see [reference](reference.md) for the period-count formula.

### Bonus: `column name 'X' is not a valid identifier: must match [A-Za-z_][A-Za-z0-9_]{0,127}`

Column / table names must be SQL-safe identifiers. No spaces, no hyphens, no leading digits.

**Fix.** Rename. `customer_id` not `customer-id`; `mrr` not `MRR (USD)`.

### Bonus: `extra inputs are not permitted` (Pydantic-native)

You added a key the schema doesn't recognize (probably a typo of an existing field).

**Fix.** Check spelling. Plotsim configures `extra="forbid"` on every model so unknown keys fail loudly rather than silently.

---

## Validating before generating

```bash
plotsim validate config.yaml              # bare command — runs all load-time validators
plotsim validate config.yaml --config-only  # explicit fast-path name; same behavior
```

Both invocations are behaviorally identical today: load the YAML, run every validator, exit. `--config-only` makes the fast-path contract explicit for CI scripts and reserves the bare command for a future deeper-validation mode without breaking the fast path.

```bash
plotsim info config.yaml      # preview entity count, period count, table list, estimated rows
plotsim run config.yaml -v    # generate + run post-generation integrity checks
```

## Where to next

- **[Templates](templates.md)** — pick a starting config instead of building from scratch
- **[Reference](reference.md)** — full CLI, full schema, changelog
