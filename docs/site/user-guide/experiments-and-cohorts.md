# Experiments & cohorts

> Late-arriving entities, cohort-mix evolution, and treatment / control
> assignment — the three engine features that turn plotsim from a
> static-roster simulator into an A/B-test-ready dataset generator.

---

## Why these features

The default plotsim model has every entity present from period 0 with
the same archetype playing across the full window. That's the right
shape for a steady-state simulation — but real datasets aren't
steady-state.

- **Cohort analysis** needs entities that arrive at different times.
- **Retention curves** need a left-truncated population.
- **A/B test analysis** needs treatment and control arms with a known
  ground-truth lift.

The three features below — cold-start, segment drift, and treatment /
control — compose: a single segment can declare *when its entities
arrive* AND *which arm of an experiment they land in*, and the engine
maintains determinism (same seed → same dataset) across all three.

---

## Cold-start entities (`Entity.start_period`)

Per-entity arrival period. An entity with `start_period = k` is
**dormant** for periods `[0, k)` — its trajectory is NaN-filled, and
the fact tables drop those rows. From period `k` onward the entity
becomes active and the archetype's full curve plays out across the
remaining `n_periods - k` periods (the entity lives its own lifecycle
from t=0; cohort analysis depends on this).

```yaml
entities:
  - name: customer_42
    archetype: growth
    start_period: 5    # arrives in month 6 of a 12-month window
```

`dim_<entity>` always includes the entity regardless of its arrival
period — the registry is complete; only the fact tables are
left-truncated.

### Validator

`MIN_ACTIVE_PERIODS = 2`. Every entity must satisfy
`start_period <= n_periods - MIN_ACTIVE_PERIODS`. An entity with one
or zero active periods is degenerate (a metric needs at least two
data points to have shape) and the validator rejects at config load.

### Causal-lag interaction

Cold-start periods append `NaN` to the per-entity lag buffer so the
buffer stays period-index-aligned. When the lag pipeline reads a
`NaN` driver value, it falls back to the entity's current trajectory
position — same fallback path the engine already uses for
`period_index < lag_periods` and "history too short" cases. No new
behaviour to learn; the cold-start gap is the third lane of an
existing pattern.

---

## Segment proportion drift (`SegmentInput.arrival`)

The builder layer materialises cold-start cohorts via four arrival
distributions. Each segment can declare an `arrival` shape; the
interpreter draws per-entity `start_period` values deterministically
from a seed-derived RNG.

### `uniform`

Entities arrive evenly across `[start, end)`.

```yaml
segments:
  - name: trial_signups
    count: 60
    archetype: growth
    arrival:
      kind: uniform
      start: 0
      end: 10        # exclusive upper bound (numpy convention)
```

`end=None` (the default) auto-fills to `n_periods - MIN_ACTIVE_PERIODS`
so every drawn entity has at least two active periods.

### `linear`

Entities arrive at a linearly varying rate. `direction='increasing'`
back-loads (more entities arrive later — typical of an organic growth
ramp); `direction='decreasing'` front-loads (most entities arrive
early — a promotional spike).

```yaml
arrival:
  kind: linear
  start: 0
  end: 12
  direction: increasing
```

Implementation: triangular CDF inversion via `sqrt(u)` for the
back-loaded direction, `1 - sqrt(1 - u)` for the front-loaded
direction.

### `step`

Entities arrive in discrete blocks. Each block names a period and a
fraction of the segment to land at that period. Fractions must sum to
`1.0` (±0.001 for floating-point inputs).

```yaml
arrival:
  kind: step
  blocks:
    - { period: 0, fraction: 0.5 }    # half arrive at launch
    - { period: 6, fraction: 0.5 }    # half arrive mid-year
```

Deterministic by construction — no RNG draw. The last block absorbs
rounding remainder so the total entity count exactly matches
`segment.count`.

### `explicit`

Per-entity start_periods, length must equal `segment.count`. Useful
for golden fixtures or research configs where cohort timing is the
experimental variable.

```yaml
arrival:
  kind: explicit
  start_periods: [0, 0, 3, 3, 6, 6, 9, 9]    # length == count
```

No RNG draw.

### Determinism contract

Same seed + same `UserInput` → same per-entity arrival schedule.
`step` and `explicit` consume zero RNG draws — so adding either
between two random shapes does NOT shift the random shapes' draws.

---

## Treatment / control (`SegmentInput.treatment`)

The third feature: A/B test assignment with a known effect size. Each
segment can declare a `treatment` block; the interpreter randomly
assigns a fraction of the segment's entities to the treatment arm and
the rest to the control arm.

```yaml
segments:
  - name: trial_users
    count: 100
    archetype: growth
    arrival:
      kind: linear
      start: 0
      end: 10
      direction: increasing
    treatment:
      fraction: 0.5             # 50/50 split
      lift_log_odds: 0.6         # known effect size
      start_period: 6            # rollout date
      treatment_label: new_onboarding
      control_label: original_onboarding
      target_metric: mrr         # optional — see below
```

### What the lift does

Every metric's pre-polarity effective trajectory position (in `[0, 1]`)
is shifted in **logit space** for treatment-arm entities at periods
`>= treatment_start_period`:

```
shifted_position = sigmoid(logit(position) + lift_log_odds)
```

Working in log-odds space gives the right diminishing-returns
behaviour: a `+0.5` lift moves `p=0.5` to ~0.62, but only moves
`p=0.9` to ~0.94. Same intervention, less impact when the metric is
already near saturation.

### Targeting a single metric (`target_metric`)

By default the lift applies to **every** metric for the treatment arm —
useful when modelling an intervention that shifts overall trajectory
position (a global engagement boost, a churn-reduction programme).
Add `target_metric: <metric_name>` to restrict the lift to a single
named metric. Every other metric in the same period is drawn
identically to the control arm, even for entities in the treatment
cohort.

```yaml
treatment:
  fraction: 0.5
  lift_log_odds: 0.6
  start_period: 6
  target_metric: mrr   # only mrr shifts; engagement, churn_risk, etc. stay flat
```

Use the targeted form when the experimental hypothesis names one
outcome metric ("the pricing experiment lifts revenue, not
engagement"), or when you want a placebo metric in the dataset whose
mean must be statistically identical across arms. Omit it for a
trajectory-wide intervention.

Correlated metrics: if `target_metric` names a metric that participates
in a `connections` correlation, the copula still operates on residuals
around each metric's own (un-shifted) centre — so the lift does **not**
propagate to the correlated metric's mean. The targeted metric shifts,
the correlated metric stays at its control distribution.

### Pre-treatment baseline

At `period_index < treatment_start_period`, the shift is `0.0` for
every entity — treatment and control arms see the same trajectory.
Pre-treatment population distributions are statistically identical
(modulo per-entity RNG noise at the distributional draw level), which
is the AC for "pre-treatment baseline is identical across groups" —
exactly what the analyst needs to confirm randomisation worked before
running difference-in-means.

### RNG isolation

The interpreter uses **two independent RNG streams** for arrival and
treatment draws, both seeded from the config's `seed` but salted apart.
Concretely: `arrival_rng = default_rng(seed)` and
`treatment_rng = default_rng(seed ^ TREATMENT_SALT)`. The salt
decouples the two:

- Same seed + same treatment config + ANY arrival shape (uniform,
  step, explicit, or none) → identical treatment assignments.
- Same seed + same arrival config + ANY treatment shape (different
  fraction, label, start_period, or none) → identical arrival
  schedule.

This is the same principle behind the `step` / `explicit` arrival
shapes' zero-RNG-draw contract, applied across feature boundaries:
changing one feature's shape doesn't shift another feature's outputs.

### Manifest

Two manifest fields surface treatment ground-truth:

- `EntityArchetypeAssignment.treatment` — per-entity assignment
  record. Carries the entity's group label, lift (or `None` for
  control), `start_period`, and `target_metric` (`null` for the
  trajectory-wide default). `null` for entities with no treatment
  fields set.
- `ManifestSchema.treatment_cohorts` — aggregate per-cohort records.
  One entry per distinct `treatment_group` label. Reports the cohort
  size, mean lift, modal `start_period`, and modal `target_metric`
  (`null` when every entity in the cohort uses the trajectory-wide
  default).

The `target_metric` field on both records is additive; manifests
emitted for configs that do not set `target_metric` keep the field
`null`, so older readers continue to parse cleanly.

### Validator

Rejected at config load:

- `treatment_start_period >= n_periods` (the lift would never apply).
- `treatment_lift_log_odds = ±inf` or `nan` (would propagate NaN cells).
- `target_metric` set to a name that doesn't match any declared metric.
  Without this check a typo would silently fall through the per-metric
  gate (no metric matches → the lift is never applied) and the
  treatment would be invisible in the generated data.

**NOT** rejected (intentionally):

- `treatment_start_period < entity.start_period`. An entity arriving
  at period 6 with `treatment_start_period=4` is fine — periods 4–5
  are dormant (cold-start NaN trajectory, no rows generated), and the
  shift kicks in naturally at the entity's first active period. The
  builder uses this slack to assign one segment-level
  `TreatmentConfig.start_period` to a cohort whose entities have
  arrival-distribution-drawn `start_period` values varying per entity.

---

## Worked example

`tests/configs/ab_trial.yaml` is a SaaS trial-conversion A/B test
dataset that exercises all three features together: a legacy cohort
present from period 0, organic trial signups arriving via a
back-loaded linear ramp, paid-ad trial signups arriving in two step
blocks, and a 50/50 onboarding-flow rollout to the trial cohorts at
period 6. The resulting `manifest.json` carries per-entity treatment
labels and per-cohort lift in `treatment_cohorts` so a downstream
analysis can recover the configured effect from the generated data
via difference-in-means.

```bash
plotsim run tests/configs/ab_trial.yaml --output ./datasets/ab_trial
```

The paired `ab_trial.py` in the same directory is the recommended
starting point for adapting plotsim to your own A/B test scenarios.
Per-metric treatment also ships on the bundled `marketing` and
`health` domain templates if you want a turnkey example.

---

## How the three features compose

The features have a hard dependency chain:

1. **Cold-start** is the foundation — the engine surface
   (`Entity.start_period`) that the other two build on.
2. **Drift** is the builder mechanism for materialising cold-start
   across a cohort with one declarative shape per segment.
3. **Treatment / control** is independent of arrival but uses the
   cold-start infrastructure as a baseline window: setting
   `treatment_start_period > 0` carves out a pre-treatment window
   where the shift hasn't yet kicked in.

For determinism, all three features draw from RNG streams that are
seed-coupled but draw-independent. Practical consequence: you can
iterate on one feature's shape (try a different arrival distribution,
or a different lift magnitude) without disturbing the others'
realised values, which makes parameter exploration fast.
