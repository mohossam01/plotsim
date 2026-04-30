# Engine Internals

> **Purpose.** This is the single self-contained map of the plotsim engine.
> It exists so a fresh reader — human or LLM — can reconstruct what the engine
> does, what each module is responsible for, and where the load-bearing
> invariants live, **without opening any `.py` file**.
>
> **Audience.** Maintainers picking up after a context reset, reviewers
> grading a PR, and AI assistants asked to reason about plotsim without
> being handed the source tree.
>
> **Aligns with:** `__version__ = 0.5.0` · post-M119 (global seasonality) · pending commit · 2026-04-30
>
> **Maintenance contract.** Whenever a change to `plotsim/*.py` (other than
> `cli.py`) lands, this file is updated in the same session. The
> `/handoff` command enforces the gate — see `Maintenance` at the bottom.

---

## 1. Mental model

Two ideas explain almost everything about plotsim.

### 1.1 Trajectory-first

For every entity at every time step, the engine computes one **trajectory
position** in `[0, 1]` from the entity's archetype curve. **All metrics for
that entity at that period are derived from that single position.** Metrics
are never sampled independently. If you can't trace a metric value back to a
trajectory position, the generation is broken.

The corollary is structural: tables are not generated independently. The
fact-table builder receives the trajectory array and passes it to each metric
generator. There is no code path where a metric generator runs without a
trajectory input.

### 1.2 Three-phase system

| Phase | What | Network? | Deterministic? |
|-------|------|----------|----------------|
| **A — Scaffold** *(V2 only)*   | Plain-language prompt → Groq/Llama → YAML config | yes | no (LLM is non-deterministic) |
| **B — Build** *(M115/M116)*    | Plain-language input (YAML or kwargs) → `PlotsimConfig` (`plotsim.builder`) | **no** | yes — same input produces the same `PlotsimConfig` (modulo `seed`, drawn fresh per call) |
| **C — Generate** *(V1)*         | `PlotsimConfig` → tables, manifest, validation report | **no** | yes — same `(config, seed)` produces byte-identical output |

Plotsim 0.5.0 ships Phase B + Phase C. Phase A returns in V2 layered on
top of Phase B's same input format. The Phase B layer (`plotsim.builder`)
is documented separately: [`builder-reference.md`](./builder-reference.md)
covers the vocabulary + field surface, [`builder-quickstart.md`](./builder-quickstart.md)
walks two annotated examples, and [`builder-errors.md`](./builder-errors.md)
catalogs every error and warning.

---

## 2. Pipeline — stage by stage

Top-to-bottom data flow. Each stage names its module(s), its input/output
contract, and the invariants it owns.

```
┌──────────────────────────────────────────────────────────────────────┐
│  YAML config  →  PlotsimConfig (frozen, validated)                   │
└──────────────────────────────────────────────────────────────────────┘
            │
            ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Archetype curve segments  →  trajectory[period] in [0, 1]           │
│  (per entity, no randomness)                                         │
└──────────────────────────────────────────────────────────────────────┘
            │
            ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Per entity, per period: position → center → distribution sample →   │
│  Gaussian-copula correlation → noise → clamp/round → MCAR null       │
└──────────────────────────────────────────────────────────────────────┘
            │
            ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Dimensions (date spine, entity, sub-entity, reference)              │
│  Facts (per_entity_per_period, per_period)                           │
│  Events (proportional, threshold)                                    │
│  Bridges (M:M)  ·  SCD Type 2 expansion  ·  Stage assignment         │
└──────────────────────────────────────────────────────────────────────┘
            │
            ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Holdout split  ·  Entity features  ·  Quality injection  ·          │
│  Validation report  ·  Manifest                                      │
└──────────────────────────────────────────────────────────────────────┘
            │
            ▼
┌──────────────────────────────────────────────────────────────────────┐
│  output.write_tables() — the only filesystem-touching module         │
└──────────────────────────────────────────────────────────────────────┘
```

### 2.1 Config load — `plotsim/config.py`

**Role.** Parses the YAML, runs Pydantic v2 validation (types, enum
membership, identifier safety, cross-reference closure), returns a frozen
`PlotsimConfig`.

**Inputs:** path to a YAML file conforming to the plotsim schema.
**Outputs:** an immutable `PlotsimConfig`. Round-trippable via `dump_config`.

**Top-level fields on `PlotsimConfig`:**

- `domain` — name + label.
- `time_window` — `start` (`YYYY-MM`), `end` (`YYYY-MM`), `granularity` ∈
  {`monthly`, `weekly`, `daily`}.
- `seed` — integer; the only entropy source for the entire pipeline.
- `metrics` — distribution + polarity + value_range + optional `causal_lag`.
- `archetypes` — ordered `curve_segments` covering `[0, 1]`.
- `entities` — name + archetype name + size + optional `overrides`
  (`inflection_month`, per-metric `noise`, etc.). Capped at 100,000
  entries (raised from 100 in M117 to accommodate per-segment expansion
  in the builder path); the runtime envelope is bounded by the
  cell-count gate (`sum(entities.size) × period_count`), not the
  list cap.
- `tables` — schema, columns, dtypes, and `source` for every table.
  Variable-grain dim tables accept `count: int = 1` (M117), the
  sub-entity row multiplier; rejected on any other type/grain
  combination at load.
- `correlations` — pairwise coefficients; defaults to identity off-diagonal.
- `noise` — gaussian σ, outlier rate, MCAR rate (with `NOISE_PRESETS`:
  `PERFECTLY_CLEAN`, `SLIGHTLY_MESSY`, `REALISTIC`, `DIRTY`).
- `output` — `format` (`csv` or `parquet`), `output_dir`, `float_format`.
- `manifest` — `include` toggle + `trajectory_sample_rate`.
- `stages` *(optional)* — `StageSequence` for per-period stage assignment (free-mode by default; opt into the monotonic walk via `enforce_order: true`).
- `bridges` *(optional)* — list of `BridgeTableConfig` (M:M tables).
- `quality` *(optional)* — `QualityConfig.quality_issues` for corruption.
- `entity_features` *(optional)* — `enabled`, label inclusion.
- `holdout` *(optional)* — `enabled`, `target_metric`, `holdout_periods`,
  `min_training_periods`.
- `locale` — Faker locale; string or list.

**Source taxonomy** — every column declares a typed `source` parsed by
`parse_source(s) -> ParsedSource`:

| Source class       | Maps to                                                      |
|--------------------|--------------------------------------------------------------|
| `PKSource`         | primary key (auto or composite)                              |
| `FKSource`         | foreign key into another table                               |
| `MetricSource`     | values from `plotsim.metrics`                                |
| `GeneratedSource`  | engine-resolved (e.g. `entity_name`, `timestamp`)            |
| `FakerSource`      | Faker provider (`faker.company`, `faker.name`, …)            |
| `StaticSource`     | inline list/value                                            |
| `DerivedSource`    | computed from other columns                                  |
| `ThresholdSource`  | event firing when a metric crosses a band                    |
| `ProportionalSource` | event row count proportional to a metric                   |
| `LagSource`        | a metric's value at a lagged period                          |
| `SCDType2Source`   | the band label for an SCD Type 2 column                      |
| `TextBucketSource` | bucket a numeric metric to one of a fixed text label set     |
| `PoolSource`       | per-entity sample from `Column.value_pool[entity_name]` (per_entity dim only; M114) |

**Identifier safety.** Table and column names must match
`[A-Za-z_][A-Za-z0-9_]{0,127}` — a SQL-safe pattern that also rejects path
traversal in output filenames.

### 2.2 Curves — `plotsim/curves.py`

**Role.** Composable mathematical building blocks. Each curve consumes a
normalised `t ∈ [0, 1]` array and returns values clipped to `[0, 1]`.

| Curve         | Parameters                          | Shape                          |
|---------------|-------------------------------------|--------------------------------|
| `sigmoid`     | `midpoint`, `steepness`, `rising`   | S-curve rise or fall           |
| `exp_decay`   | `rate`                              | Exponential decline            |
| `step`        | `threshold`, `before`, `after`      | Abrupt level change            |
| `logistic`    | `k`, `midpoint`, `ceiling`          | Bounded growth                 |
| `plateau`     | `level`                             | Flat constant                  |
| `oscillating` | `period`, `amplitude`, `center`     | Cyclical wave                  |
| `compound`    | `base_rate`, `acceleration`         | Accelerating cumulative growth |
| `sawtooth`    | `period`, `amplitude`, `base`       | Periodic ramp-and-drop         |

**Single entry point.** `evaluate_segment(t_segment, curve_type, params)`
dispatches via `CURVE_REGISTRY` and clamps the output. The trajectory engine
is the only production caller.

### 2.3 Trajectory — `plotsim/trajectory.py`

**Role.** Stitches an archetype's ordered `curve_segments` into one
length-`n_periods` array in `[0, 1]`. Zero randomness — pure function of
its inputs.

**Public surface:**

- `compute_time_steps(time_window) -> np.ndarray[str]` — period labels
  (`YYYY-MM`, `YYYY-Www`, `YYYY-MM-DD` per granularity).
- `compute_trajectory(archetype, n_periods, overrides=None) -> np.ndarray`
  — single-entity trajectory; `overrides.inflection_month` shifts segment
  boundaries.
- `compute_all_trajectories(config, n_periods) -> dict[entity_name → np.ndarray]`
  — batch builder used by `tables.generate_tables`.

**Segment boundary discipline.** Boundaries are computed once via
`floor((end_pct + shift) * n_periods)`, clamped to `[0, n_periods]`, and
enforced monotonically non-decreasing. A shift that would push one boundary
past the next collapses the intermediate segment to length 0 rather than
producing a negative-length slice. The first boundary is always `0`; the
last is always `n_periods` so the final segment absorbs rounding remainder.

**Discontinuities at segment boundaries are preserved.** The engine does
not smooth them.

### 2.4 Metrics — `plotsim/metrics.py`

**Role.** Where randomness enters the system. Turns a trajectory position
into an actual metric value through a fixed pipeline.

**Per-period pipeline (every metric, every entity, every period):**

1. **Polarity flip** — `negative` polarity replaces `position` with
   `1 - position`.

   *Polarity is applied here and only here.* Once `_apply_polarity` runs
   inside `position_to_center` ([`metrics.py:90`](../plotsim/metrics.py#L90)),
   every downstream feature reads either the resulting metric value
   (stages, threshold events, proportional events) or the raw trajectory
   position (SCD Type 2). None of `_monotonic_stage_walk`,
   `_free_mode_stages`, `_build_threshold_event`,
   `_build_proportional_event`, or `_compute_scd_versions` consults
   `metric.polarity` (verified by grep — only `inspect.py` reads it,
   for trace reporting). Stage and event thresholds therefore operate
   in **realized metric-value space** with polarity already baked in:
   for a `negative`-polarity metric, "above 0.7" fires when the
   *trajectory* is low. SCD Type 2 is the exception — it reads raw
   trajectory positions, so SCD thresholds live in **trajectory space
   `[0, 1]`** and are unaware of polarity entirely. See §2.6 for the
   full input-source table.

2. **Position → center** — `position_to_center(position, metric)` maps
   the polarity-adjusted position `p` to the distribution's location
   parameter:

   | Distribution | Formula                                                | Linear in `p`? | E[X] tracks center?                                                                  |
   |--------------|--------------------------------------------------------|----------------|---------------------------------------------------------------------------------------|
   | `beta`       | `p` (no `value_range`) · `vr.min + p · (vr.max − vr.min)` (with `vr`) | yes (affine) | yes — sampler shifts E[X] to center exactly via `center + (raw − base_mean) · span`   |
   | `normal`     | `mu · p`                                               | yes            | yes — `rng.normal(loc=center, scale=sigma)` so E[X] = center                          |
   | `poisson`    | `lam · p`                                              | yes            | yes — `rng.poisson(lam=center)` so E[X] = center                                       |
   | `gamma`      | `shape · scale · p`                                    | yes            | yes — `rng.gamma(shape, scale=center/shape)` so E[X] = `shape · (center/shape)` = center |
   | `lognorm`    | `loc + scale · p`                                      | yes (affine)   | scaled — `lognormal(mean=log(center), sigma=s)` so **median = center** but E[X] = center · exp(s²/2) |
   | `weibull`    | `scale · p`                                            | yes            | scaled — `rng.weibull(shape) · center` so E[X] = center · Γ(1 + 1/shape)              |

   All `position_to_center` mappings are linear in `p`, so the configured
   archetype shape always reaches `center`. Realized-cell-mean recovery
   of that shape is exact for `beta`, `normal`, `poisson`, and `gamma`
   (E[X] = center → archetype recoverable from the realized cell mean);
   `lognorm` and `weibull` pick up a distribution-dependent multiplicative
   constant (`exp(s²/2)` and `Γ(1 + 1/shape)` respectively), so the cell
   mean is a *scaled* version of the trajectory. For `lognorm`, the
   cell **median** still recovers the trajectory exactly — useful when
   downstream consumers need to back out the archetype from realized
   values. Pick `beta` / `normal` / `poisson` / `gamma` when the user
   needs to see the configured archetype directly in the realized cell
   mean.

3. **Seasonal modulation (M119)** — when `config.seasonal_effects` is
   non-empty, the per-period summed global strength is computed once in
   `tables._build_seasonal_factors`, anchored to calendar months by
   `TimeWindow.period_calendar_months()`. Inside the metric loop, each
   metric's center is scaled by

   ```
   modulated = base_center × (1 + global × metric.seasonal_sensitivity × entity.seasonal_sensitivity)
   ```

   then clamped to `metric.value_range` BEFORE the distributional draw.
   When the effective strength is zero (or `seasonal_effects` is empty),
   the branch short-circuits and the center is unchanged — the
   pre-M119 cell value is byte-identical. Trajectory positions are
   never touched here; seasonality is a center modifier, not a
   trajectory modifier, so the trajectory-first invariant survives.
   `inspect.trace_metric_cell` exposes both `seasonal_factor` (the
   effective multiplier) and `modulated_center` for verification.
4. **Independent draw** — `sample_single_metric` draws from
   `lognorm | gamma | poisson | beta | normal | weibull` via `scipy.stats`.
5. **Correlate** — `apply_correlations` uses a Gaussian copula: each
   independent draw is mapped to a uniform via the distribution's CDF, then
   to a Gaussian via the inverse normal, jointly correlated against a
   Cholesky factor of the (PD-projected) correlation matrix, then mapped
   back through the original CDF. This preserves marginals while honouring
   pairwise correlations.

   **M120: trajectory-aware pre-compensation.** When
   `config.compensate_correlations=True` (default for builder-produced
   configs, opt-in for engine-direct configs), an extra step runs once
   before the Cholesky factor is built: `estimate_trajectory_covariance`
   computes the within-archetype Pearson the trajectory's centers
   themselves induce between every metric pair (with M119 seasonal
   modulation applied), weighted by entity proportion across archetypes.
   `compensate_correlation_matrix` then subtracts that contribution from
   each declared `CorrelationPair` before the matrix reaches Higham. The
   per-cell copula is unchanged — it just receives a different Cholesky
   factor. Without this compensation, mixed-archetype configs land
   table-wide Pearson values dominated by the trajectory's structural
   covariance and the configured `connections` are invisible. Undeclared
   off-diagonals are NOT compensated: forcing `r_copula = -r_traj` for
   every silent zero pushes the matrix into a near-degenerate region
   that Higham then heavily distorts, leaking back into the declared
   pairs and undoing the compensation we wanted there. The metric cap
   `_MAX_METRICS_FOR_COMPENSATION = 20` falls back to the legacy direct
   path with a warning above that count — the additive trajectory +
   copula decomposition becomes too noisy at higher M to satisfy the
   sign-match floor.
6. **Noise** — `apply_noise` adds gaussian σ-jitter, replaces a small
   fraction with outliers, and inserts MCAR nulls.
7. **Clamp / round** — `_clamp_and_round` enforces `value_range` and
   rounds integer dtypes.

**Causal lag.** `CausalLag(driver, lag_periods, blend_weight)` lets a target
metric blend its own position with the driver's effective position from a
past period. Drivers are resolved in topological order via
`_toposort_metrics`; cycles raise `CycleError`. `_compute_effective_position`
maintains a `lag_buffer` of *effective* positions (not raw trajectory) so
multi-hop chains `A → B → C` compose truthfully.

**Per-archetype `MetricOverride`.** An archetype can substitute a metric's
`distribution`, `params`, or (M114) `value_range` for entities assigned to
it. Overrides are applied once per metric in `_apply_archetype_overrides`
([`metrics.py`](../plotsim/metrics.py)) and the resulting *effective*
Metric is what every downstream helper reads — so `position_to_center`,
`sample_single_metric`, `_get_scipy_dist`, and `_clamp_and_round` all
honour the override consistently. Polarity and `causal_lag` are never
overridable. The `value_range` override is enforced as a *subset* of the
global metric range at config load
(`PlotsimConfig._cross_reference_integrity`); overrides restrict, never
expand. A metric without a global `value_range` cannot carry a
range override.

**Higham nearest-PD projection.** Configured correlation matrices that
aren't positive-definite are projected to the nearest PD matrix via Higham
2002. The projection is recorded as `CorrelationAdjustment` records and
surfaced on the manifest. Eigenvalue clipping with a small tolerance margin
is the fallback when Higham doesn't converge fast enough.

**Public functions:**

- `position_to_center(position, metric) -> float`
- `sample_single_metric(rng, metric, center, n) -> np.ndarray`
- `apply_correlations(samples_dict, metrics, correlations, rng) -> dict`
- `apply_noise(values, rng, noise) -> np.ndarray`
- `project_correlation_matrix(M) -> (M_pd, adjustment_records)`
- `generate_metrics_for_period(...)` — single period across all metrics.
- `generate_entity_metrics(...)` — full per-period series for one entity.
- `generate_archetype_batch(...)` — full per-period series for a batch
  of entities sharing one archetype (M121 vectorized path).

### 2.4a Vectorized generation — `plotsim/metrics.py` (M121)

**Role.** A second, optional metric-generation path that groups entities
by archetype and runs batched numpy samplers + a batched copula across
the entity axis at each period. Selected per-config via
`PlotsimConfig.generation_mode`:

| Mode          | Behavior                                                                    | Default for          |
|---------------|-----------------------------------------------------------------------------|----------------------|
| `serial`      | Pre-M121 per-(entity, period) loop in `generate_entity_metrics`.            | Engine-direct configs |
| `vectorized`  | Archetype-batched numpy via `generate_archetype_batch`.                     | Builder configs (via `auto`) |
| `auto`        | Resolves to `vectorized` when `len(config.entities) >= 50`, else `serial`. | Builder configs      |

The dispatch lives in `tables._compute_entity_metrics`. In `vectorized`
mode the function:

1. Groups entities by archetype, preserving the order each archetype
   first appears in `config.entities` so RNG draw order is stable.
2. Excludes entities with `EntityOverrides` (e.g., `inflection_month`)
   from the batch — overridden entities run the per-entity serial path
   *after* the batch closes, so all batch RNG draws happen first in a
   fixed order.
3. Calls `generate_archetype_batch(...)` per archetype, then the serial
   fallback for the overridden subset, and unions both into the
   `entity_metrics` dict the rest of the pipeline consumes.

**Per-period vectorized pipeline** (inside `generate_archetype_batch`):

1. **Effective positions** — for each metric in topological driver→target
   order, blend the trajectory position at `t` with the driver's past
   effective position from a `(n_batch, n_periods)` lag buffer. No
   cross-entity leakage: each batch row carries its own lag history.
2. **Centers** — `_position_to_center_batch` over the batch axis, then
   M119 seasonal modulation `× (1 + S_t × em.sens × ent.sens[i])`,
   then `value_range` clamp.
3. **Independent draws** — one batched `rng.<dist>(size=n_batch)` call
   per metric (dispatched in `sample_single_metric_batch`).
4. **Batched copula** — `_apply_correlations_batch` pushes each
   marginal through scipy's `cdf`, clips to `[1e-10, 1−1e-10]`, applies
   the inverse normal, multiplies by `cholesky_L.T`, then closes the
   round-trip via `cdf(N(0,1))` → marginal `ppf`. Per-cell degenerate
   cells (`_bypass_mask_batch`) trigger a per-row scalar fallback to
   preserve the bypass-aware submatrix Cholesky behavior of the scalar
   path; the all-active fast path covers production configs with
   positive centers throughout the trajectory.
5. **Noise** — `_apply_noise_batch` runs gaussian → outlier → MCAR with
   one batched RNG call per branch; MCAR cells become `np.nan`.
6. **Clamp / round** — `_clamp_and_round_batch` enforces `value_range`
   and rounds poisson columns, preserving NaN cells.

**Cholesky factor sharing.** The orchestrator's `cholesky_L` (built
once after M120 compensation + M111 Higham projection in toposort
order) is passed straight through to `_apply_correlations_batch` —
the vectorized path does *not* recompute it per archetype.

**RNG-order divergence.** Vectorized and serial paths consume RNG in
different orders (`size=n_batch` calls per metric per period vs. `n`
scalar calls per metric per period). Same seed across modes →
**statistically equivalent but distinct cell values**. Same seed
within a mode → byte-identical. The two contracts coexist:

- `(config[mode=serial], seed)` → byte-identical bytes (matches
  pre-M121 baseline → bundled templates round-trip unchanged).
- `(config[mode=vectorized], seed)` → byte-identical bytes across
  runs of the same plotsim version on the same platform.
- Cross-mode comparison: same per-archetype mean trajectories,
  same correlation sign matches, same shape recovery — but distinct
  individual cell values.

**Bypass fallback.** When *any* cell in the batch at the current
period has a degenerate distribution (gamma `shape ≤ 0`, lognorm
center `≤ 1e-9`, etc.), `_apply_correlations_batch` falls back to a
per-row scalar `apply_correlations` call. This is correct but slow.

**M121b bypass observability.** `_apply_correlations_batch` now
returns `(values, bypass_cell_count)`; `generate_archetype_batch`
accumulates the count across periods and writes it into an optional
mutable `bypass_counter` dict the orchestrator threads through. The
orchestrator stashes the resulting `dict[archetype_name → int]` on
`PlotsimConfig._bypass_fallback_counts` (PrivateAttr, mirrors the
M111 / M120 sibling pattern); `manifest.build_manifest` surfaces it
as `bypass_fallback_counts: Optional[dict[str, int]]`:

* `None` → serial-mode run; bypass is unmeasured because the path
  has no batched copula to fall back from.
* `{}` → vectorized run with zero bypass cells; the all-active fast
  path covered every period.
* `{archetype: count, ...}` → vectorized run where one or more
  archetypes hit degenerate centers. A user investigating
  "vectorized isn't faster on my config" reads this directly to see
  whether bypass is the cause.

**Auto-threshold provenance.** `_VECTORIZED_AUTO_THRESHOLD = 50`
is justified by `analysis/perf/m121_vectorized.py`: at 95 entities
across 6 archetypes (smallest archetype-batch ~10) the saas builder
template still beats serial by 3.75×; at 1,020 entities across 3
archetypes the stress config lands at ~70×. `manifest.vectorized_threshold_used`
records the constant value at generation time so a re-run after a
threshold change can be detected by comparing against the current
constant. Open question deferred: should the threshold key on
`max(archetype_group_size)` instead of `len(config.entities)`? At
~10 entities per archetype the numpy-overhead amortization is
narrow; the bundled templates all clear that bar but a multi-
archetype edge config might not.

**What stays per-entity even in vectorized mode.** Entities with any
non-`None` `Entity.overrides` (currently `inflection_month`) are routed
to the serial path inside the dispatcher. Their batch is the empty
set; the rest of the archetype's entities still batch as usual. The
conservative check is `if entity.overrides:` — any `EntityOverrides`
instance triggers serial fallback, future-proofing the dispatch
against new override fields without silent batching of fields the
batch doesn't yet honour.

**Public functions added:**

- `generate_archetype_batch(...)` — batch generator for one archetype.
- `sample_single_metric_batch(centers, metric, rng) -> np.ndarray` —
  batched draw for one metric across `n` entities.

### 2.5 Dimensions — `plotsim/dimensions.py`

**Role.** Builds every non-behavioural table the engine needs before fact
generation can resolve foreign keys.

| Output                | Grain               | Role                                         |
|-----------------------|---------------------|----------------------------------------------|
| `dim_date`            | `per_period`        | Date spine — fixed schema. Consumers **always** join by `date_key`; never inline `DATE_TRUNC`. |
| `dim_<entity>`        | `per_entity`        | One row per entity, static attributes from Faker / derived fields. |
| `dim_<subentity>`     | `variable` + dim    | `sum(entity.size × table.count)` rows, FK-linked to parent dim_<entity>. The two factors compose multiplicatively — engine-direct configs use `Entity.size > 1` with default `Table.count=1`; the builder path uses `Entity.size=1` with `Table.count` taking the multiplier role. |
| `dim_<reference>`     | `per_reference`     | Lookup tables (plans, departments). Row count = longest static-valued column. |

**Determinism.** Faker is seeded from the same RNG that drives generation;
`(config, seed)` always produces the same names.

**`PoolSource` — per-entity value pools (M114).** A column on a
`per_entity` dim can declare `source: "pool:<name>"` and pair it with
`value_pool: {<entity_name>: [<value>, ...], ...}`. Each entity's row
gets one value sampled (uniform integer draw) from that entity's list
via the per-table RNG. The pairing is enforced on `Column._pool_pairing`
and the per-entity coverage check (every entity producing rows in the
table must appear in `value_pool`) is enforced cross-model in
`plotsim.validation.validate_value_pool_coverage`. Pools are restricted
to per_entity dims in this version — sub-entity and reference dims are
rejected at load. **The architectural firewall holds**: pool selection
resolves entity-membership only and never reads a trajectory.

**Architectural firewall.** No import of `plotsim.curves` or
`plotsim.metrics`. The dimension layer never sees a trajectory.

### 2.6 Facts, events, bridges, SCD, stages — `plotsim/tables.py`

The largest module (~2,500 lines). Composes trajectories + metrics +
dimensions into a complete table set.

**Public entry points:**

- `generate_tables(config, rng) -> dict[name → DataFrame]` — the
  user-facing one.
- `generate_tables_with_state(config, rng) -> (tables, GenerationState)` —
  same output plus the `GenerationState` (trajectories, SCD state, bridge
  associations) needed by `plotsim.inspect`.

**`GenerationState`.** Dataclass holding trajectories, SCD state, bridge
associations, and the toposorted metric order. Hands `inspect` everything
it needs to replay generation deterministically.

**Build order inside `generate_tables`:**

1. `build_all_dimensions` — date spine, per-entity dims, sub-entity dims,
   reference dims.
2. `compute_all_trajectories` — once per entity, cached.
3. `expand_scd_dims` — for any dim column with `SCDType2Source`, expand
   one row per (entity × band) version.
4. `build_fact_tables` — per_entity_per_period and per_period facts.
   Internally walks each entity once, calling
   `generate_entity_metrics` to produce all metrics for that entity in
   one pass (the topological order is computed once and reused).
5. `attach_dim_row_id_to_facts` — for facts referencing SCD dims, resolve
   to the correct version row per period.
6. `build_event_tables` — proportional events (row count proportional to
   the driving metric) and threshold events (one firing per entity when
   the trajectory first crosses a threshold).
7. `build_bridge_tables` — M:M associations between two dim tables, with
   `BridgeMetric` aggregates (career-aggregated NaN-aware mean).
8. `assign_stages` — per-period stage assignment over a driving metric,
   written into the `stage` column of the fact table that owns it.
   Default is **free mode** (`enforce_order=False`): each period
   independently picks the highest-enter stage the realized value
   satisfies, so stages can move backward when the value falls. Set
   `enforce_order: true` in the config to opt into the monotonic walk
   (forward-only cursor, optionally relaxed by `downgrade_delay`).
   Irreversible lifecycle transitions belong in SCD Type 2; stages
   reflect *current* state.

**Threshold input sources.** Five engine features compare values against
thresholds. They divide cleanly between *trajectory space* (operates on
`[0, 1]` trajectory positions before polarity, sampling, and noise) and
*metric-value space* (operates on realized fact-table values, with
polarity and noise already baked in):

| Feature             | Function                                                                          | Input read from                                                              | Compared against                                                                  | Space                       |
|---------------------|-----------------------------------------------------------------------------------|------------------------------------------------------------------------------|------------------------------------------------------------------------------------|-----------------------------|
| SCD Type 2          | `_compute_scd_versions` ([`tables.py:1703`](../plotsim/tables.py#L1703))         | `trajectory` array (`np.ndarray` in `[0, 1]`)                                | `scd_cfg.thresholds` via `np.searchsorted(side="right")`                          | trajectory `[0, 1]`         |
| Stages (free mode, **default**) | `_free_mode_stages` ([`tables.py:1454`](../plotsim/tables.py#L1454))     | realized metric values from the fact table                                   | `[s.threshold_enter for s in seq]` only — `threshold_exit` ignored in free mode    | metric value                |
| Stages (monotonic, opt-in via `enforce_order: true`) | `_monotonic_stage_walk` ([`tables.py:1369`](../plotsim/tables.py#L1369)) | realized metric values from the fact table (`values: np.ndarray`)            | `[s.threshold_enter for s in seq]`; optional `s.threshold_exit` for hysteresis demote | metric value                |
| Threshold events    | `_build_threshold_event` ([`tables.py:1208`](../plotsim/tables.py#L1208))        | `fact_row[metric_col]` per period                                            | `ts.value` with `ts.direction ∈ {above, below}` and `ts.consecutive`               | metric value                |
| Proportional events | `_build_proportional_event` ([`tables.py:973`](../plotsim/tables.py#L973))       | `fact_df[metric_col]` (vectorized)                                            | *no comparison* — row count = `np.rint(value · rc.scale).astype(int64)`            | metric value (scaled to count) |

A reader reasoning about how thresholds interact with polarity or
`value_range` needs to know which space a feature lives in. SCD Type 2
thresholds are unaware of polarity (it's applied later, inside
`position_to_center` — see §2.4). Stage and event thresholds see
polarity-baked-in values; "above 0.7" on a `negative`-polarity metric
fires when the *trajectory* is low. Proportional events do not threshold
at all — they scale the realized metric value to a row count.

**Critical invariant.** Event tables consume completed *fact values* —
never trajectories. The function signature of `build_event_tables` accepts
`fact_tables` and not trajectories; this is the mechanical enforcement.

**Threshold events fire ONCE per entity.** The first period where the
trajectory crosses the configured threshold emits one row; subsequent
crossings of the same threshold do not re-fire. This is by design — see
`_build_threshold_event` (≈ line 1208 in `tables.py`). If you need
per-period firings, use a proportional event.

### 2.7 Holdout — `plotsim/holdout.py`

**Role.** Temporal train/holdout split for fact tables. Strictly temporal
— random shuffling on time-series data is a leakage pattern.

**Cutoff.** `cutoff_period_index(config) = n_periods - holdout_periods`.
Rows with `period_index < cutoff` are training; `>= cutoff` are holdout.

**Eligibility.** Only fact tables with grain `per_entity_per_period` and
non-empty data are split. Dim, reference, event, and bridge tables are
not split.

**Architectural rules.**

- No import of `plotsim.tables` — receives DataFrames as arguments.
- Pure: same `(config, tables)` produces byte-identical splits every call.
- The writer in `plotsim.output` is the sole production caller.

**Public surface:**

- `cutoff_period_index(config) -> int`
- `split_fact_tables(config, tables) -> dict[name → (train_df, holdout_df)]`

### 2.8 Entity features — `plotsim/entity_features.py`

**Role.** Aggregates the temporal fact tables into a single
one-row-per-entity DataFrame for downstream tabular ML or notebook joining.

**Per-metric aggregates** (six per numeric metric):

- `{metric}_mean` — `np.nanmean` over the entity's series.
- `{metric}_std` — `np.nanstd` (population, ddof=0).
- `{metric}_slope` — degree-1 polyfit of value vs period index.
- `{metric}_first` — value at the entity's earliest period.
- `{metric}_last` — value at the entity's latest period.
- `{metric}_peak_period` — period index where value is max.

**Optional ground-truth columns** (`include_labels=true`):

- `archetype` — from `config.entities[i].archetype`.
- `final_trajectory_position` — from manifest's `trajectory_samples`.

**Architectural rules.**

- No import of `plotsim.tables`.
- Bridge tables are never aggregated — bridges are associative, not
  temporal.
- Mutual exclusion with `quality.quality_issues` enforced at config load
  via `validate_entity_features_config`.

### 2.9 Quality injection — `plotsim/quality.py`

**Role.** Post-generation data-quality corruption layer. Additive over
generation: never re-derives metric values, never reads trajectories,
never touches FK or period columns.

**Five issue types** — each gets its own seeded RNG (`base_seed +
seed_offset`) so reordering issues in the config never perturbs other
issues' affected row sets:

| Issue                | Behavior                                                                             |
|----------------------|---------------------------------------------------------------------------------------|
| `null_injection`     | Set `rate` of cells in target columns to NaN / `None`.                                |
| `duplicate_rows`     | Insert exact copies of `rate` of randomly chosen rows back at random positions.       |
| `type_mismatch`      | Convert `rate` of values to the wrong type (numerics → strings, strings → ints, …). Column dtype promoted to `object`. |
| `late_arrival`       | Append `_arrival_period` column = original period + `random(1, 5)` for `rate` of rows. |
| `schema_drift`       | Copy value into `{column}_v2` and null original at `rate` of rows.                    |

**Protected columns:** `date_key`, `period`, `period_index`,
`period_label`, plus all FK columns. The validator at config load rejects
target columns that resolve to a protected name; the `"*"` sentinel
expansion excludes them defensively.

**Pure:** clean tables passed in are NOT mutated. A deep copy is taken
before any corruption. Manifest construction reads the clean copy; the
corrupted dict is what callers write to disk.

**Public surface:**

- `apply_issues(tables, config, base_seed) -> (corrupted_tables, ground_truth_records)`

### 2.10 Validation — `plotsim/validation.py`

**Role.** Post-generation integrity and coherence checks; one
pre-generation check on the config alone.

**Checks** (constants on the module, all in `ALL_CHECKS`):

| Check                          | What it verifies                                                                  |
|--------------------------------|-----------------------------------------------------------------------------------|
| `correlation_psd`              | Configured correlation matrix is PD (or projection succeeded).                    |
| `pk_uniqueness`                | Single and composite PKs are unique per table.                                    |
| `fk_integrity`                 | Every FK value resolves to a parent PK.                                           |
| `date_spine`                   | `dim_date` is gap-free; facts' `date_keys` ⊆ `dim_date`.                          |
| `causal_coherence`             | Causal-lag alignment + threshold-event coherence.                                 |
| `null_policy`                  | Metric nulls within `mcar_rate`'s 3σ; non-metric columns null-free.               |
| `empty_event_table`            | Configured event tables are non-empty unless explicitly allowed.                  |
| `cross_dim_fk_cardinality`     | Cross-dim FK reach matches table grain.                                           |
| `temporal_coherence`           | All date columns within `time_window`.                                            |
| `scd_integrity`                | SCD Type 2 versions cover the period axis without gaps or overlaps.               |
| `bridge_integrity`             | Bridge cardinalities and FK resolution.                                           |

**Output.** `ValidationReport` — immutable list of `ValidationIssue`,
with `.ok`, `.errors`, `.warnings`, `.by_check(name)` accessors.

**Public surface:**

- `validate_tables(config, tables) -> ValidationReport` (also exported
  as `validate`).
- `validate_correlation_psd(config) -> list[ValidationIssue]` — single
  pre-generation check.

### 2.11 Manifest — `plotsim/manifest.py`

**Role.** Builds a structured JSON sidecar (`manifest.json`) recording the
*signal layer* of a run — the inputs an ML pipeline would predict against
rather than re-derive from noisy cell values.

**Wire shape** — single source of truth is `ManifestSchema`:

- `schema_version` — string tag (currently `"1.0"`).
- `seed` — int.
- `config_sha256` — full SHA-256 of the JSON-dumped config (see Hidden
  Contracts §4).
- `archetype_assignments: list[EntityArchetypeAssignment]`
- `trajectory_samples: list[TrajectorySample]` — `(entity, period_index,
  position)` for a deterministic subset of entities (size determined by
  `manifest.trajectory_sample_rate`).
- `event_firings: list[EventFiring]` — `(entity, table, period_indices)`.
- `scd_events: list[SCDEvent]` — band crossings (initial assignments are
  not events).
- `bridge_associations: list[BridgeAssociationRecord]`
- `quality_injections: list[QualityInjection]` — *populated at write time
  (§4)*.
- `holdout: HoldoutInfo | None` — *populated at write time (§4)*.
- `correlation_adjustments: list[CorrelationAdjustment] | None` — Higham
  projection records.
- `correlation_compensations: list[CorrelationCompensation] | None` —
  M120 trajectory-aware pre-compensation records.
- `bypass_fallback_counts: dict[archetype, int] | None` — M121b
  per-archetype count of cells that triggered the per-row scalar
  fallback in `_apply_correlations_batch`. `None` in serial mode
  (the path doesn't measure bypass); `{}` in vectorized mode with
  no bypass; populated dict surfaces "vectorized isn't faster on
  my config" investigations.
- `vectorized_threshold_used: int | None` — M121b value of
  `_VECTORIZED_AUTO_THRESHOLD` at generation time, recorded so old
  manifests stay reproducible if the constant changes in a later
  release.

**Determinism.** JSON serialization is byte-identical: `sort_keys=True`,
`indent=2`, `ensure_ascii=False`, trailing newline. Every numeric field is
funnelled through `float(...)` so numpy types never leak into the wire.

**Trajectory sampling determinism.** The entity subset is the first
`ceil(n * sample_rate)` entities under sorted-name order. No RNG consumed.

**Public surface:**

- `build_manifest(config, tables, state) -> ManifestSchema`
- `write_manifest(manifest, output_dir) -> Path`
- `config_sha256(config) -> str`

### 2.12 Output — `plotsim/output.py`

**Role.** **The only filesystem-touching module in plotsim.** Every other
module returns DataFrames / reports in memory.

**Per-table file format.** `OutputConfig.format` selects CSV (default)
or Parquet. Both share the same column-ordering and `Int64` coercion;
only the encoder differs.

**CSV conventions.**

- UTF-8, no DataFrame index.
- Float precision: `%.4f` (configurable via `float_format`).
- NaN / `None` → empty string; non-numeric quoted (`csv.QUOTE_NONNUMERIC`).
- Integer-typed columns render without `.0` even when pandas promoted them
  to float for NaN handling.
- Column order: PKs first, then FKs in config order, then remaining
  columns in config order. Engine-added columns (e.g. `stage`) appended
  last.

**Parquet conventions.**

- Engine: `pyarrow` (`plotsim[parquet]` extra). No other engines in V1.
- Same column ordering and Int64 coercion as CSV.
- Compression: snappy (explicit).
- Same DataFrame + same plotsim/pyarrow versions → byte-identical output
  for the non-streaming path. Streaming and non-streaming Parquet
  agree on read-back DataFrames but **not** raw bytes — row group
  metadata (count, sizes) differs by construction (see §2.12a).

### 2.12a Streaming Parquet — `plotsim/output.py` (M121b)

**When it fires.** `_streaming_parquet_eligible(config)` returns
True iff `output.format == "parquet"` AND
`_resolve_generation_mode(config) == "vectorized"`. Engine-direct
configs default to `serial` so bundled-template Parquet output (when
opted in via `output.format: parquet`) keeps the single-shot
`to_parquet` path. Builder configs above the auto threshold pick up
the streaming branch automatically.

**What it does.** Fact tables (`per_entity_per_period` and
`per_period`) are written via `pyarrow.parquet.ParquetWriter` with
one row group per archetype batch (yielded by
`tables.iter_fact_chunks`). `per_period` facts and any fact whose
row count doesn't equal `len(config.entities) × n_periods` (e.g.,
sub-entity-FK facts the helper conservatively rejects) write as a
single row group under the sentinel chunk key `"__per_period__"`.
Dim, event, and bridge tables continue through the existing
`write_single_table` path — they don't have the per-archetype row
layout the streaming model needs.

**Memory contract.** The unified DataFrames are still resident in
memory because downstream consumers (`attach_dim_row_id_to_facts`,
`assign_stages`, `build_event_tables`, `build_bridge_tables`,
`entity_features`) all consumed them before the writer ran. The
streaming win is the *additional* peak from `to_parquet`'s pyarrow
conversion of the full DataFrame, which the architecture-scalability
doc identified as the dominant transient overhead. Realistic
projection (replaces architecture-scalability §4's optimistic
numbers):

| Scale | Non-streaming | Streaming Parquet | Notes |
|---|---|---|---|
| DE daily (10K × 365 × 20) | ~1.7 GB | ~1.0–1.2 GB | unified DF still resident |
| Extreme (50K × 365 × 20) | ~8.7 GB | ~4–5 GB | same |

The original M121 600 MB / 1.2 GB targets are only reachable if
events, stages, and entity_features also stream — out of scope here
and tracked as a follow-up.

**Determinism contract.** Streaming and non-streaming Parquet
produce read-back DataFrames that are equal cell-for-cell for the
same `(config, seed, generation_mode)`. Raw file bytes differ:
streaming writes one row group per archetype, non-streaming writes
one row group containing all rows. Use `pd.read_parquet` to compare;
do not compare file bytes directly.

**Schema inference.** ParquetWriter requires a schema upfront, but
empty-DataFrame inference returns `null` type for object columns
(no values to type-resolve from). The streaming path defers writer
creation until the first non-empty chunk for each fact table,
infers schema from that chunk's pyarrow Table, and reuses the
schema for every subsequent row group via `pa.Table.cast(schema)`
when needed. All chunks of the same fact are slices of the same
prepared DataFrame, so dtypes are consistent.

**Side-effect ordering inside `write_tables`** (this is what makes the
hidden contracts in §4 land where they do):

1. Quality injection — `apply_issues(tables, config, seed)` runs first.
   Returns `(corrupted_tables, ground_truth_records)`. **Manifest is
   patched here via `model_copy(update={"quality_injections": ...})`.**
2. Holdout split — `split_fact_tables(config, corrupted_tables)` if
   `holdout.enabled`. **Manifest patched again with `HoldoutInfo`.**
3. Entity features — `build_entity_features(...)` if `enabled`.
4. Per-table writes — CSV or Parquet, plus `_train` / `_holdout`
   companions when holdout fired.
5. `config.yaml` (round-trippable) and `validation_report.txt` written
   alongside.
6. `manifest.json` written last (so it reflects every write-time patch).

**Public surface:**

- `write_tables(tables, config, report) -> Path`
- `write_single_table(name, df, config) -> Path`
- `write_config_copy(config, output_dir) -> Path`
- `write_validation_report(report, output_dir) -> Path`

### 2.13 Inspect — `plotsim/inspect.py`

**Role.** Single-cell trace of the metric pipeline. Reconstructs the full
path from `(entity, period, metric)` through to the realized fact-table
cell value: polarity-flipped position → center → independent draw →
correlated draw → noise → clamp/round → realized cell.

The only sanctioned external consumer of `plotsim.metrics` private
internals. Notebooks and tests use `trace_metric_cell` as the ground-truth
verifier of the trajectory-first invariant.

**How.** Re-runs generation deterministically with `seed`, captures
`GenerationState`, then replays for entities `[0..target_idx)` to
synchronise RNG consumption, and walks the period loop manually for the
target entity capturing each intermediate value.

**Replay must NOT reorder entities or skip periods.** Doing either
desynchronises the RNG draws and breaks the bit-exact traceback assertion
the acceptance notebook uses.

**Public surface:**

- `TraceResult` (dataclass with all intermediates).
- `trace_metric_cell(config, entity, period, metric, seed=None) -> TraceResult`

### 2.14 Schema — `plotsim/schema.py`

**Role.** Thin wrapper over Pydantic v2 that emits a Draft 2020-12 JSON
Schema for `PlotsimConfig`. Editor integrations (VSCode, JetBrains)
point at the produced `plotsim-schema.json` for autocomplete and inline
validation on `sample_*.yaml` configs.

**Public surface:**

- `generate_schema() -> dict`
- `write_schema(path) -> Path`
- `SCHEMA_FILENAME = "plotsim-schema.json"`

### 2.15 CLI — `plotsim/cli.py`

Thin argparse shell over the library. Every command calls a public
function importable from `plotsim`.

| Command                                | Effect                                                  |
|----------------------------------------|---------------------------------------------------------|
| `plotsim run <config.yaml>`            | Generate CSV/Parquet from a config.                     |
| `plotsim validate <config.yaml>`       | Validate the config without generating tables.          |
| `plotsim info <config.yaml>`           | Summarise what a config would generate.                 |
| `plotsim list-templates`               | List bundled `sample_*` configs.                        |
| `plotsim template <name> [--output]`   | Copy a sample config out for editing.                   |
| `plotsim schema [--output]`            | Emit JSON Schema for `PlotsimConfig`.                   |

### 2.16 Builder — `plotsim/builder/{__init__,recipes,parser,input,interpreter,schema}.py`

**Role.** Phase B: translate a plain-language declaration of "what data
do you want" into a `PlotsimConfig` that Phase C can generate from. Two
public surfaces — `plotsim.create(**kwargs)` (Python) and
`plotsim.create_from_yaml(path)` (YAML) — both routed through the same
`UserInput` model and the same `interpret(...)` step.

**Layered as:**

- `recipes` — pure-data lookup tables: vocabulary words → engine
  parameters. Three families (`SHAPE_RECIPES`, `RELATIONSHIP_RECIPES`,
  `BASELINE_RECIPES`) plus the conditional metric distribution rules
  (`AMOUNT_LOGNORM_RATIO_THRESHOLD`, `INDEX_SIGMA_FRACTION`). Imports
  nothing engine-side.
- `parser` — composite archetype DSL parser
  (`parse_archetype(spec, n_periods) -> list[CurveSegment]`). Grammar:
  `shape ( ">" shape )* ( "@" period )*` — see `builder-reference.md` §4.
- `input` — `UserInput` Pydantic model + structural validators
  (cross-reference closure, causal-lag cycles, archetype DSL,
  vocabulary membership, baseline targets, lifecycle ordering).
  Raises `pydantic.ValidationError` at construction.
- `interpreter` — `interpret(UserInput) -> PlotsimConfig`. 10 ordered
  steps: domain → window → metrics → archetypes/entities → correlations
  → stages → tables (or auto-generated `dim_date` + `dim_{unit}` +
  `fct_{unit}`) → sub-entity dims → seed → final `PlotsimConfig`
  validation. Anything raised here is an interpreter bug; user-facing
  validation already ran inside `UserInput`. **M117 segment expansion**:
  each segment with `count: N` produces N individual `Entity(size=1)`
  objects named `{segment_name}_{i:04d}`; the per-row multiplier (when
  declared via `DimInput.count`) travels onto the engine `Table.count`
  field, and the two compose multiplicatively in `dimensions.build_dim_subentity`.
  The `segment.count` column type translates to a `pool:cohort_size`
  PoolSource whose value_pool maps each expanded entity to the original
  cohort population — the pre-M117 `derived:size` would have emitted 1
  for every row after expansion.
- `schema` (M116) — `generate_user_input_schema()` returning a Draft
  2020-12 JSON Schema for `UserInput`, plus five vocabulary lookup
  dicts (`METRIC_TYPES`, `SHAPE_WORDS`, `RELATIONSHIP_WORDS`,
  `BASELINE_WORDS`, `COLUMN_TYPES`) for IDE / UI tooling.
- `__init__` — public surface: `create`, `create_from_yaml`, plus the
  raw `UserInput`, `interpret`, `parse_archetype`, the four recipe
  dicts, and `ArchetypeParseError` for downstream tooling.

**Inputs:** plain-language YAML or kwargs per
`plotsim/configs/new/saas_template.{yaml,py}` (full reference) or
`plotsim/configs/new/bare_minimum.yaml` (smallest working).
**Outputs:** a fully-validated `PlotsimConfig` ready for `generate_tables`.

**Public surface:**

- `plotsim.create(**kwargs) -> PlotsimConfig`
- `plotsim.create_from_yaml(path) -> PlotsimConfig`
- `plotsim.builder.UserInput` (model_validate / model_dump)
- `plotsim.builder.interpret(user_input) -> PlotsimConfig`
- `plotsim.builder.parse_archetype(spec, n_periods) -> list[CurveSegment]`
- `plotsim.builder.ArchetypeParseError` (subclass of `ValueError`)
- `plotsim.builder.{BASELINE,METRIC,RELATIONSHIP,SHAPE}_RECIPES` (re-exports)
- From `plotsim.builder.schema`: `generate_user_input_schema`,
  `write_user_input_schema`, `SCHEMA_FILENAME`, `METRIC_TYPES`,
  `SHAPE_WORDS`, `RELATIONSHIP_WORDS`, `BASELINE_WORDS`, `COLUMN_TYPES`.

**Determinism caveat.** `interpret` draws `seed` from
`secrets.randbelow(2**32)` — calling `create()` twice on the same input
yields two `PlotsimConfig`s with different seeds. To reproduce a
dataset, copy the seed from the first config (or its manifest) before
the second call.

---

## 3. Public API surface

Everything importable from `plotsim`:

```python
from plotsim import (
    __version__,
    # Introspection
    inspect,
    # Builder (M115) — plain-language input → PlotsimConfig
    create, create_from_yaml,
    # Config
    PlotsimConfig, SurrogateKeyWarning, ManifestConfig, TextBucketSource,
    load_config, dump_config,
    NOISE_PRESETS, PERFECTLY_CLEAN, SLIGHTLY_MESSY, REALISTIC, DIRTY,
    # Generation
    generate_tables, generate_tables_with_state, GenerationState,
    # Manifest
    ManifestSchema, EntityArchetypeAssignment, TrajectorySample, EventFiring,
    build_manifest, write_manifest,
    # Validation
    validate, validate_tables, ValidationReport, ValidationIssue,
    # Output
    write_tables, write_single_table, write_config_copy, write_validation_report,
)
```

`create(**kwargs)` and `create_from_yaml(path)` accept the plain-language
input format (see `plotsim/configs/new/saas_template.{yaml,py}` for the
canonical shape) and return a fully-validated `PlotsimConfig` ready for
`generate_tables`. The builder package (`plotsim.builder`) is layered:
`recipes` (vocabulary → engine parameters) → `parser` (composite archetype
DSL) → `input` (UserInput pydantic model + structural validation) →
`interpreter` (UserInput → PlotsimConfig). Construction errors raise
`pydantic.ValidationError` with the offending field named.

**Quick start:**

```python
import numpy as np
from plotsim import load_config, generate_tables, validate, write_tables

config = load_config("config.yaml")
tables = generate_tables(config, np.random.default_rng(config.seed))
report = validate(config, tables)
write_tables(tables, config, report)
```

---

## 4. Hidden API contracts

These are **non-obvious, load-bearing behaviors** that are not visible
from a quick skim of the public API. Promoted here so they survive past
the session that discovered them.

### 4.1 `manifest.config_sha256` is a model-dump-JSON SHA, not a YAML-bytes SHA

`config_sha256(config)` hashes
`json.dumps(config.model_dump(mode="json"), sort_keys=True, default=str)`.
It is **not** the SHA-256 of the YAML file's raw bytes. Two configs with
identical semantics but different YAML formatting (whitespace, key order,
comments) hash to the same value. Any tool that wants to compare against
"the config on disk" must dump-and-rehash, not file-hash.

### 4.2 `quality_injections` and `manifest.holdout` are populated at write time

`build_manifest` returns `quality_injections=[]` and `holdout=None` —
always. The fields are populated by `output.write_tables`:

- `quality_injections` is filled from `quality.apply_issues(...)`.
- `holdout` is filled from `holdout.split_fact_tables(...)`.

Both are attached to the manifest via `model_copy(update={...})` right
before `write_manifest` is called.

**Implication for in-memory consumers:** notebooks or tests that want to
inspect `quality_injections` / `manifest.holdout` *without* writing to
disk must call `apply_issues` and `split_fact_tables` themselves and
patch the manifest the same way `output.py` does. Calling `build_manifest`
alone will not populate either field.

### 4.3 Bridge `MetricSource` resolves to a NaN-aware career mean

When a `BridgeMetric` references a metric via `MetricSource`, the
resolver in `_bridge_metric_value` collapses the per-entity series via
`np.nanmean(...)` — the **career-aggregated** mean across periods, not
a single-period reference. This is correct for associative tables (a
bridge row represents a relationship over time, not at a moment) but
will surprise anyone expecting a snapshot at the bridge's effective
period.

### 4.4 Threshold events fire ONCE per entity

In `tables.py:_build_threshold_event`, the firing logic emits exactly
one row per entity for the first period the trajectory crosses the
configured threshold. Subsequent crossings of the same threshold do
not re-fire. Use a `ProportionalSource` event if you need per-period
firings.

### 4.5 Bounded-distribution non-rise on `gaussian_sigma`

Multiplicative noise jitter is applied via `apply_noise`, but
`_clamp_and_round` enforces `value_range` bounds **after** the noise is
applied (see `metrics.py:_clamp_and_round`, ≈ line 715). For a metric
near the upper bound of `value_range`, increasing `gaussian_sigma` does
not raise the realized values — every excess gets clipped. Variance
appears one-sided in that regime. Use a wider `value_range`, a different
distribution, or accept the asymmetry.

### 4.6 `per_entity` dim row count is `len(entities)`, not `sum(e.size)`

`Entity.size` is a cohort-population value carried as a metadata column
on the dim (via `derived:size`). It is *not* a row multiplier on the dim
itself: `build_dim_entity` emits exactly one row per `Entity` regardless
of `size`. The sub-entity (variable-grain) child dim is where `size`
*does* multiply rows — composed with `Table.count` per §2.5.

This was a hidden contract because two consumers in `config.py` initially
disagreed:

- `_cross_reference_integrity` (bridge cardinality gate) sized per_entity
  dims as `sum(e.size for e in entities)` — wrong; corrected in M118.
- `_total_entity_size_within_limit` (`config.py:1722`) and
  `_combined_scale_estimator` (`config.py:1748`) intentionally use
  `sum(e.size)` — those gate the cohort *population* envelope, not the
  dim row count. They are correct as-is.

Practical consequence: a bridge whose second dim is `per_entity` has a
row-count ceiling of `len(config.entities)`, which the cardinality.max
validator now enforces. None of the bundled engine templates exercise
this path (no bundled bridge points at a per_entity second dim).

### 4.7 Trajectory-aware correlation compensation only touches declared pairs

`compensate_correlation_matrix` (M120) subtracts
`estimate_trajectory_covariance` from each `CorrelationPair` the user
declared in `config.correlations` (or the builder's `connections`
list). Undeclared off-diagonals — pairs the user never wrote — keep
their auto-zero value in the matrix instead of being compensated to
`-r_traj`.

This was a hidden contract because the mission spec's "for each metric
pair" wording reads like "every off-diagonal," but compensating
undeclared pairs has a subtle Higham interaction: with mixed-archetype
configs the within-archetype trajectory contribution is close to ±1
between same-polarity metrics, so the compensated matrix lands at
unit-magnitude off-diagonals everywhere — a near-rank-1 / near-
degenerate input. Higham's Frobenius-optimal projection then collapses
the structure back toward identity, and the *declared* pair's
compensation gets lost in the projection drift. Restricting
compensation to declared pairs keeps Higham close to identity in the
non-target rows/columns and lets the declared pair land where the
math says it should.

Practical consequence: pairs the user didn't mention follow whatever
the trajectory does (the pre-M120 contract), and the manifest's
`correlation_compensations` list contains exactly one record per
declared pair (no auto-emitted records for silent zeros). If a future
mission wants to extend the compensation surface, it should also
generalize the Higham step (e.g., shrinkage projection that preserves
target rows) — the present implementation does not.

---

## 5. Engine invariants

The non-negotiables. A change that breaks any of these is a bug.

1. **Trajectory-first.** Every realized metric value is reproducible from
   `compute_trajectory(...)` at the corresponding period via the
   documented pipeline (verified by `inspect.trace_metric_cell` in the
   acceptance notebook's bit-exact `§7` assertion).
2. **Determinism.** Same `(config, seed)` produces byte-identical
   tables, manifest, validation report, and CSV / Parquet output —
   across runs and across machines (modulo the same Python / pyarrow
   versions for Parquet).
3. **Single entropy source.** All randomness flows through one
   `numpy.random.Generator` constructed with `config.seed`. Faker is
   seeded from a draw on that generator inside each builder.
4. **FK closure.** Every FK value in every fact / event / bridge table
   resolves to a parent PK. Validation enforces this; output does not
   assume it.
5. **Dimension-before-fact ordering.** Dims are built first; facts
   resolve FKs against the already-built dim PK columns.
6. **Date spine, never `DATE_TRUNC`.** Consumers join `dim_date` by
   `date_key`; the engine never inlines `DATE_TRUNC` at fact time.
7. **Config immutability.** `PlotsimConfig` is frozen
   (`pydantic.ConfigDict(frozen=True)`). The `_correlation_adjustments`
   PrivateAttr is the single engine-side write, used only to surface
   Higham projection records on the manifest.
8. **One filesystem touchpoint.** Only `plotsim.output` writes files.
   Every other module is in-memory and pure.

---

## 6. Module reference (one-liners)

| Module                       | LOC   | Role                                                       |
|------------------------------|-------|------------------------------------------------------------|
| `plotsim/__init__.py`        | 94    | Public API re-exports.                                     |
| `plotsim/config.py`          | 2,390 | Pydantic v2 schema + YAML loader + cross-ref validation.   |
| `plotsim/curves.py`          | 158   | 8 mathematical curves + `evaluate_segment` dispatcher.     |
| `plotsim/trajectory.py`      | 228   | Stitch curve segments → length-`n_periods` array.          |
| `plotsim/metrics.py`         | 950   | Position → distribution → copula → noise → clamp.          |
| `plotsim/dimensions.py`      | 846   | `dim_date`, `dim_<entity>`, sub-entity, reference dims.    |
| `plotsim/tables.py`          | 2,570 | Facts + events + bridges + SCD + stages.                   |
| `plotsim/holdout.py`         | 182   | Temporal train/holdout split.                              |
| `plotsim/entity_features.py` | 461   | Per-entity flat feature aggregation.                       |
| `plotsim/quality.py`         | 473   | 5 post-generation corruption types.                        |
| `plotsim/validation.py`      | 1,693 | 11 named integrity / coherence checks.                     |
| `plotsim/manifest.py`        | 599   | Ground-truth JSON sidecar.                                 |
| `plotsim/output.py`          | 606   | CSV / Parquet writer — sole filesystem touchpoint.         |
| `plotsim/inspect.py`         | 530   | Single-cell pipeline trace (`trace_metric_cell`).          |
| `plotsim/schema.py`          | 51    | JSON Schema export of `PlotsimConfig`.                     |
| `plotsim/cli.py`             | 466   | `argparse` CLI shell.                                      |
| `plotsim/builder/__init__.py` | 85   | Builder public surface — `create`, `create_from_yaml` (M115). |
| `plotsim/builder/recipes.py` | 133   | Vocabulary → engine parameters (pure data; M115).          |
| `plotsim/builder/parser.py`  | 159   | Composite archetype DSL parser (M115).                     |
| `plotsim/builder/input.py`   | 640   | `UserInput` pydantic model + structural validation (M115). |
| `plotsim/builder/interpreter.py` | 790 | `interpret(UserInput) → PlotsimConfig` (M115).            |
| `plotsim/builder/schema.py`  | 121   | `UserInput` JSON Schema export + vocab enum dicts (M116).  |

---

## 7. Glossary

- **Archetype** — named ordered sequence of curve segments covering
  `[0, 1]`. Maps an entity's time axis to a master curve.
- **Curve segment** — one `(curve_type, params, end_pct)` tuple inside an
  archetype. Segments compose end-to-end with no gaps.
- **Master curve** — the stitched-together output of an archetype's
  segments before per-entity overrides.
- **Trajectory position** — scalar in `[0, 1]` representing where on the
  master curve an entity is at a given period. The single source of
  truth for "where is this entity at time t."
- **Polarity** — `positive` (high position → high value) or `negative`
  (high position → low value, e.g. churn risk).
- **Source** — the typed declaration on a column that tells the engine
  how to fill it (`pk`, `fk:dim.col`, `metric:engagement`,
  `generated:faker.company`, …). See the table in §2.1.
- **Fact** — table at grain `per_entity_per_period` or `per_period`,
  trajectory-driven.
- **Event** — table at grain `variable`, with a row count driven by
  trajectory position (proportional) or trajectory crossing (threshold).
- **Bridge** — M:M associative table between two dim tables. Not
  trajectory-driven directly; carries career-aggregated metric values.
- **SCD Type 2** — slowly-changing dimension where each band change
  produces a new row, versioned by `dim_row_id`.
- **Holdout cutoff** — `n_periods - holdout_periods`. Rows with
  `period_index < cutoff` are training; `>=` are holdout.
- **MCAR** — *missing completely at random*. The null-injection knob.
- **PSD / PD** — positive semidefinite / positive definite. A correlation
  matrix that isn't PD is Higham-projected to its nearest PD form before
  Cholesky.

---

## 8. Maintenance

This file is a denormalized cache of the codebase. Like every cache, it
needs an invalidation strategy. The strategy is **enforced by the
`/handoff` command**, not by hooks or social discipline.

**The contract:**

1. Whenever a session touches any `plotsim/*.py` other than `cli.py`,
   this file is updated *in the same session*.
2. The header's `Aligns with: <commit>` line is bumped to the new commit.
3. If a change adds, renames, or removes a public function in
   `plotsim/__init__.py`, §3 is updated in lockstep.
4. If a change reveals a new hidden contract, §4 grows.

**The gate:** the `/handoff` command reads the session's git diff. If
any `plotsim/*.py` file other than `cli.py` was modified and
`docs/engine-internals.md` was *not*, handoff halts with a prompt and
will not finalize the state rewrite until the operator answers.

The full procedure is in [`/handoff`](.claude/commands/handoff.md), step 1.5.
