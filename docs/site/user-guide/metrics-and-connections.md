# Metrics and connections

> What `type`, `polarity`, and `range` actually control. How to declare
> cross-metric correlations. How to make one metric trail another in
> time.

---

## Metric anatomy

A metric declaration has up to seven fields. Two are required:

```yaml
metrics:
  - name: engagement
    type: score
    polarity: positive
```

The other five are optional:

| Field | Purpose |
|---|---|
| `range` | `[min, max]` bounds (required for `amount` / `index`) |
| `label` | Display name; defaults to `name` |
| `follows` | Name of another metric this one trails (causal lag) |
| `delay` | Lag in periods (must pair with `follows`) |
| `seasonal_sensitivity` | How strongly this metric responds to seasonality |

---

## The four metric types

The `type` field tells plotsim what *kind* of value to produce.

| Type | Distribution | Range | What it's for |
|---|---|---|---|
| `score` | Beta(α=2, β=5) | implicit `[0, 1]` | Health scores, engagement indices, satisfaction |
| `count` | Poisson(λ=5) | non-negative integers | Logins, transactions, ticket counts |
| `amount` | Lognormal or Beta (auto-picked) | required | Money, weights, durations |
| `index` | Normal | required | Bounded indicators where mean matters more than tail |

### `score`

Bounded `[0, 1]`. The Beta(2, 5) shape skews low — most rows land in
the lower half, with a long tail toward 1.0. Good default for anything
"intensity-like": engagement, churn risk, customer satisfaction.

```yaml
- { name: engagement, type: score, polarity: positive }
```

### `count`

Non-negative integers from a Poisson distribution. The `λ` parameter is
shaped by trajectory position, so the *expected* count varies with the
entity's behavioral state. No `range` field — counts are unbounded above.

```yaml
- { name: logins, type: count, polarity: positive }
```

### `amount`

Continuous values with a configured `range`. plotsim auto-picks the
shape:

- **Lognormal** when `min == 0` or `max / min ≥ 10`. Heavy right tail —
  good for revenue, transaction sizes, where most values are modest
  but a few are large.
- **Beta** otherwise. Smooth, bounded — good for percentages, ratios.

```yaml
- name: mrr
  type: amount
  polarity: positive
  range: [10, 5000]   # lognorm — large dynamic range
- name: utilization
  type: amount
  polarity: positive
  range: [0.4, 0.9]   # beta — narrow band
```

### `index`

Normal distribution centered on the range midpoint, with sigma chosen
so ~99.7% of draws land inside the declared range (the 3-sigma rule).
Use when the *average* matters more than the tail.

```yaml
- { name: nps, type: index, polarity: positive, range: [-100, 100] }
```

### `amount` with weibull (engine-direct)

For lifetime / time-to-event style metrics — session duration,
days-to-renewal, contract length — set the engine-direct
`distribution: weibull` on a `Metric`. The shape parameter controls
the tail: `shape < 1` produces an aging-out distribution
(many short, few long), `shape = 1` is exponential, `shape > 1` skews
toward the mean. The trajectory position scales the realized value, so
trajectory and shape compose in the usual way. Not exposed in the
builder DSL — set it on `cfg.metrics[i]` after `create_from_yaml(...)`:

```python
cfg = create_from_yaml("config.yaml")
cfg.metrics[0] = cfg.metrics[0].model_copy(update={
    "distribution": "weibull",
    "params": {"shape": 1.5},
    "value_range": {"min": 1.0, "max": 365.0},
})
```

---

## Polarity

Polarity is the relationship between trajectory position and metric value:

| Polarity | High trajectory position → | Low trajectory position → |
|---|---|---|
| `positive` | high metric value | low metric value |
| `negative` | low metric value | high metric value |

A `growth` archetype with two metrics:

```yaml
metrics:
  - { name: engagement, type: score, polarity: positive }
  - { name: churn_risk, type: score, polarity: negative }

segments:
  - { name: growers, count: 30, archetype: growth }
```

produces engagement *rising* and churn risk *falling* over the window —
both from the same trajectory. You don't need a separate archetype for
each direction.

---

## Connections — declaring correlations

Connections add a *cross-metric* correlation on top of the trajectory.
Trajectory shapes already produce strong shared signal across metrics
on the same entity; connections fine-tune the relationship.

```yaml
connections:
  - "mrr driven_by engagement"
  - "churn_risk inverts engagement"
  - "support_tickets related churn_risk"
```

Three accepted shapes per entry — pick whichever reads best:

```yaml
connections:
  - "mrr driven_by engagement"                                   # 3-token string
  - ["mrr", "driven_by", "engagement"]                           # tuple
  - { a: "mrr", relationship: "driven_by", b: "engagement" }     # dict
```

### The vocabulary

Nine words spanning `-0.75` to `+0.75`:

| Word | Coefficient | Reads as |
|---|---|---|
| `mirrors` | +0.75 | "moves with" |
| `driven_by` | +0.55 | "follows" |
| `related` | +0.40 | "partly tracks" |
| `hints_at` | +0.20 | "weakly suggests" |
| `independent` | 0.00 | "no relationship" |
| `hints_against` | -0.20 | "weakly opposes" |
| `resists` | -0.40 | "partly opposes" |
| `opposes` | -0.55 | "tends opposite" |
| `inverts` | -0.75 | "moves opposite" |

Both endpoints must reference declared metrics. Self-pairs are rejected.

### What the engine does with this

plotsim builds a correlation matrix from your connections, then samples
metrics through a Gaussian copula at each `(entity, period)` cell. The
realized table-wide Pearson correlation lands close to the declared
coefficient — usually within `0.05–0.10` for production-shape configs.

If the matrix you declare is mathematically inconsistent (not positive
semi-definite), plotsim projects it to the nearest valid matrix and
records the adjustments in `manifest.correlation_adjustments`. Strong
mirrors (`mirrors`, `inverts`) on lots of metrics tends to over-constrain
the matrix — a warning fires.

---

## Causal lag — `follows` + `delay`

Sometimes one metric should *trail* another in time. Engagement spikes,
then a few periods later support tickets rise. plotsim models this with
the `follows` and `delay` fields:

```yaml
metrics:
  - name: engagement
    type: score
    polarity: positive
  - name: support_tickets
    type: count
    polarity: negative
    follows: engagement
    delay: 2
```

This says: at every period, the trajectory position used for
`support_tickets` is blended with the position from 2 periods ago for
`engagement`. The result — support tickets visibly trail engagement
moves by 2 periods.

**Rules**:

- `follows` and `delay` must pair (both present or both absent).
- `delay` is in periods, `>= 1`.
- A metric can't follow itself.
- The chain must be acyclic — a `follows` graph can't loop back on
  itself.

---

## Per-metric seasonal sensitivity

Each metric has a `seasonal_sensitivity` that controls how much it
responds to globally configured seasonality:

| Value | Effect |
|---|---|
| `1.0` (default) | Follow global seasonal strength at face value |
| `0.0` | Immune to seasonality |
| `-0.5` | Halve the effect *and* invert direction |
| `2.0` | Amplify by 2× |

This pairs with `seasonality` and per-segment sensitivity. See
[`seasonality.md`](./seasonality.md) for the full multiplication
formula.

---

## A complete example

```yaml
about: Subscription customers
unit: customer
window: ["2024-01", "2024-12", "monthly"]

metrics:
  - { name: engagement,  type: score, polarity: positive }
  - { name: mrr,         type: amount, polarity: positive, range: [10, 5000] }
  - { name: tickets,     type: count, polarity: negative, follows: engagement, delay: 2 }
  - { name: churn_risk,  type: score, polarity: negative }

connections:
  - "mrr driven_by engagement"
  - "tickets resists mrr"
  - "churn_risk inverts engagement"

segments:
  - { name: growers,    count: 30, archetype: growth }
  - { name: decliners,  count: 20, archetype: decline }

seed: 42
```

What this produces:

- A `growth` segment whose engagement rises, MRR rises with it (driven by),
  tickets stay low (negative polarity, lagging engagement by 2 periods),
  churn risk stays low.
- A `decline` segment with the mirror story — engagement falls, MRR
  falls, tickets rise (with a 2-period lag), churn risk rises.
- All four metrics on each row share the same underlying trajectory
  position. The connections shape the *deviation* from that shared
  signal.

---

## What to read next

- [How it works](./how-it-works.md) — the trajectory-first invariant
- [Archetypes](./archetypes.md) — the curve side of the same story
- [Seasonality](./seasonality.md) — adding seasonal modulation on top
- [Config field reference](../config-reference.md) — every metric and
  connection field with constraints
- [Tutorials → seasonality and correlations](../tutorial-notebooks/seasonality_and_correlations.ipynb) — runnable example
