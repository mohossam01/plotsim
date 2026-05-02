# Archetypes

> The DSL for "shape over time." Six base shapes; two operators to
> compose them.

---

## Why archetypes

An archetype tells plotsim *how an entity's behavior evolves over the
time window*. It's a curve over `[0, 1]`. At every period the engine
evaluates the curve to get one number — the trajectory position — and
every metric for that `(entity, period)` cell reads from that single
number.

You don't author distributions or per-period values. You declare the
shape, and the engine handles distribution sampling, polarity flipping,
correlation, and noise on top of that shape.

---

## The six base shapes

| Shape | Curve | Use for |
|---|---|---|
| `growth` | Sigmoid rising from low to high | Adoption, ramping engagement |
| `decline` | Exponential decay from high to low | Churn, fading interest |
| `seasonal` | Oscillating around 0.5 (period ≈ 2 cycles in window) | Cyclical demand, recurring patterns |
| `flat` | Constant near 0.15 | Steady-state low activity |
| `spike_then_crash` | Sigmoid rise → step drop → low plateau | Honeymoon-period customers |
| `accelerating` | Compounding growth with positive acceleration | Viral / network-effect adoption |

Each is a single word in the YAML:

```yaml
segments:
  - { name: ramping,  count: 30, archetype: growth }
  - { name: leaving,  count: 20, archetype: decline }
  - { name: cyclical, count: 25, archetype: seasonal }
  - { name: dormant,  count: 50, archetype: flat }
  - { name: doomed,   count: 15, archetype: spike_then_crash }
  - { name: viral,    count:  8, archetype: accelerating }
```

---

## Composition with `>` and `@`

You combine shapes into composite archetypes when one phase doesn't
capture the behavior. Two operators:

### `>` — sequence

Chain shapes in order. Default split is even.

```yaml
archetype: "growth > decline"
```

Half the window on `growth`, half on `decline`. With three shapes:

```yaml
archetype: "flat > growth > decline"
```

Each shape gets one third of the window.

### `@ <period>` — explicit transition

Set the transition period explicitly. Periods are integers, counted from
the start of the time window.

```yaml
window: ["2024-01", "2024-12", "monthly"]   # 12 periods total
archetype: "growth > decline @ 8"
```

Period 0 through period 7 follow `growth`; period 8 onward follows
`decline`.

With N shapes, you supply N−1 `@` clauses — one transition between every
pair, in ascending order.

```yaml
archetype: "flat > growth > spike_then_crash @ 4 @ 16"
```

Periods 0–3 are `flat`; 4–15 are `growth`; 16 onward is `spike_then_crash`.

**Constraints**:

- Each `@` value must be in `[1, n_periods - 1]` (you can't anchor at
  the very first or very last period).
- Anchors must be strictly ascending.
- The `@` count must be exactly one less than the shape count.

---

## What each shape looks like

```
growth                          decline
  ┌──────                        ──┐
  │                                │
──┘                                └──────


seasonal                        flat
   ╱╲      ╱╲      ╱╲           ──────────
  ╱  ╲    ╱  ╲    ╱  ╲
 ╱    ╲  ╱    ╲  ╱
╱      ╲╱      ╲╱


spike_then_crash                accelerating
  ╱──╲                                  ╱
  │   ╲                                ╱
  │    └────────                     ╱
──┘                              ──╱
```

Curves are smooth functions over `[0, 1]`; the ASCII is for shape only.

---

## Worked examples

**One pattern, full window**

```yaml
segments:
  - name: stable_cohort
    count: 50
    archetype: growth
```

Single sigmoid from low to high across the entire window.

**Hand-off mid-window**

```yaml
segments:
  - name: rampers_then_dippers
    count: 40
    archetype: "growth > decline"
```

Even split — sigmoid for the first half, exponential decay for the
second.

**Three-phase lifecycle on a 24-period window**

```yaml
window: ["2023-01", "2024-12", "monthly"]   # 24 periods
segments:
  - name: full_lifecycle
    count: 30
    archetype: "flat > growth > decline @ 4 @ 18"
```

Periods 0–3 dormant, 4–17 ramping, 18–23 declining.

**Honeymoon-then-flatline**

```yaml
segments:
  - name: doomed_after_quarter
    count: 25
    archetype: "spike_then_crash > flat @ 6"
```

`spike_then_crash` plays out across periods 0–5, then `flat` from
period 6 to the end.

---

## Polarity flipping

Trajectories are positive by default — high position means "more of the
behavior." Whether a *metric* trends up or down with the trajectory
depends on the metric's `polarity`.

| Metric polarity | Trajectory rises | Trajectory falls |
|---|---|---|
| `positive` | metric rises | metric falls |
| `negative` | metric falls | metric rises |

So a single archetype like `growth` produces engagement *rising* and
churn risk *falling* simultaneously when you tag them with the right
polarity. You don't need a separate "churn growth" archetype.

[`metrics-and-connections.md`](./metrics-and-connections.md) covers
polarity in full.

---

## What about my own shapes?

The six base shapes are intentionally limited — they cover the common
cases without forcing you into curve-fitting territory. If you need
finer control, compose. `flat > growth > spike_then_crash > flat`
covers more shapes than it looks like.

If a single curve genuinely doesn't fit, that's usually a sign your
"one segment" is actually two — split it into two segments with
different archetypes and let the engine give you the mix.

---

## What to read next

- [Metrics and connections](./metrics-and-connections.md) — the metric
  side of the same story
- [Seasonality](./seasonality.md) — adding a global seasonal modulation
  on top of any archetype
