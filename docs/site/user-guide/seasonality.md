# Seasonality

> Global seasonal effects, per-metric sensitivity, per-segment
> sensitivity, and the multiplication formula that ties them together.

---

## Why seasonality

Some patterns aren't well captured by an archetype curve. Calendar-month
spikes (Black Friday, summer slumps, fiscal-year-end pushes) recur
across every entity at the same calendar time, regardless of where each
entity sits on its trajectory.

plotsim's `seasonality` block lets you declare these as a *global*
modulation that multiplies onto every metric value, with per-metric and
per-segment dials to dampen, amplify, or invert the effect.

---

## Declaring global effects

Each entry in the `seasonality` array names a set of calendar months
and a strength multiplier:

```yaml
seasonality:
  - { months: [11, 12], strength: 0.30 }
  - { months: [6, 7, 8], strength: -0.10 }
```

| Field | Type | Notes |
|---|---|---|
| `months` | array of `int` | Month numbers `1..12`, unique within one effect |
| `strength` | `float` | Added to `1.0` at each matching month before metric multiplication |

The example above produces:

- **November / December**: every metric value scaled by `1.30` (a 30%
  lift)
- **June / July / August**: every metric value scaled by `0.90` (a 10%
  dip)
- **All other months**: no scaling

Multiple effects may overlap — strengths *sum* at each period before the
multiplier is applied. So `strength: 0.20` on `[11, 12]` and another
entry of `strength: 0.10` on `[12]` produces `1.20` in November and
`1.30` in December.

The empty default (`seasonality: []`) produces output identical to runs
without the block.

---

## Per-metric sensitivity

Each metric has a `seasonal_sensitivity` field that scales the global
effect for that metric only:

```yaml
metrics:
  - name: revenue
    type: amount
    polarity: positive
    range: [100, 10000]
    seasonal_sensitivity: 1.0   # default — full seasonal effect

  - name: support_tickets
    type: count
    polarity: negative
    seasonal_sensitivity: 0.0   # immune

  - name: engagement
    type: score
    polarity: positive
    seasonal_sensitivity: 0.5   # half effect
```

| Value | Effect |
|---|---|
| `1.0` (default) | Follow global seasonal strength at face value |
| `0.0` | Metric is immune to seasonality |
| `0.5` | Half the configured effect |
| `2.0` | Double the configured effect |
| `-1.0` | Invert direction (a global lift becomes a dip on this metric) |

Negative sensitivities are useful when a metric reasonably moves
*against* the calendar trend — e.g., support tickets fall during a
holiday lift in revenue because customers don't file tickets while
they're shopping.

---

## Per-segment sensitivity

Each segment has its own `seasonal_sensitivity`, applied to every
entity expanded from that segment:

```yaml
segments:
  - name: retail
    count: 50
    archetype: growth
    seasonal_sensitivity: 1.5   # extra-seasonal cohort

  - name: enterprise
    count: 20
    archetype: growth
    seasonal_sensitivity: 0.0   # B2B contracts don't follow calendar
```

Same `1.0` default and same range of accepted values as the per-metric
field.

This dial lets you mix seasonal-sensitive and seasonal-insensitive
cohorts in a single config — e.g., consumer customers track the
calendar while enterprise customers don't.

---

## The multiplication formula

For each `(entity, period, metric)` cell:

```
seasonal_factor = global_strength × metric_sensitivity × segment_sensitivity
modulated_center = distribution_center × (1 + seasonal_factor)
```

`distribution_center` is the value the engine derived from the
trajectory position before seasonality was applied; `modulated_center`
is what the distribution actually samples around.

A worked example for December:

```yaml
seasonality:
  - { months: [11, 12], strength: 0.30 }    # global = 0.30 in Dec

metrics:
  - name: revenue
    type: amount
    polarity: positive
    range: [100, 10000]
    seasonal_sensitivity: 1.0                # metric = 1.0

segments:
  - name: retail
    count: 50
    archetype: growth
    seasonal_sensitivity: 1.5                # segment = 1.5
```

For a retail entity in December:

```
seasonal_factor = 0.30 × 1.0 × 1.5 = 0.45
modulated_center = distribution_center × 1.45
```

The metric samples around 1.45× its non-seasonal center for that cell.

For an enterprise entity (segment sensitivity `0.0`) in the same cell:

```
seasonal_factor = 0.30 × 1.0 × 0.0 = 0.0
modulated_center = distribution_center × 1.0
```

No seasonal effect at all.

---

## Window-length caveats

Seasonal patterns need at least two cycles of the configured period to
recover cleanly downstream. Rule of thumb:

- For monthly windows, declare seasonality only when the window covers
  ≥ 24 months (two years).
- Shorter windows still produce seasonal-looking values, but the
  pattern is hard to pull back out of the noisy fact-table cells.

The builder warns at construction when `seasonal` archetypes are
combined with windows shorter than 24 periods.

---

## What the manifest records

Seasonal modulation is a deterministic function of the config — same
`(config, seed)` produces the same `seasonal_factor` at every cell. The
manifest doesn't record per-cell seasonal factors directly; you can
reconstruct them from the config alone.

If you need per-cell verification, [`trace_metric_cell`](../api-reference.md#trace_metric_cell)
returns the `seasonal_factor` and `modulated_center` for any single
`(entity, period, metric)` triple.

---

## What to read next

- [Metrics and connections](./metrics-and-connections.md) —
  `seasonal_sensitivity` in the broader metric anatomy
- [How it works](./how-it-works.md) — where seasonality fits in the
  pipeline (after distribution center, before noise)
- [Config field reference §seasonality](../config-reference.md) —
  field-level constraints
- [Tutorials → seasonality and correlations](../tutorial-notebooks/seasonality_and_correlations.ipynb) — runnable example
