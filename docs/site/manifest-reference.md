# Manifest Reference

> The manifest is a JSON sidecar (`manifest.json`) written next to the
> generated tables. It captures the *signal layer* — archetype labels,
> trajectory positions, event firings, SCD transitions, bridge
> associations, and reproducibility metadata — that a downstream ML
> pipeline can use as ground truth without re-deriving it from noisy
> cell values.
>
> See [`build_manifest`](./api-reference.md#build_manifest) for the
> programmatic builder. The companion docs are
> [`config-reference.md`](./config-reference.md) (the `manifest` config
> block) and [`api-reference.md`](./api-reference.md).

---

## When the manifest is written

`write_tables` writes `manifest.json` when both:

1. `config.manifest.include` is True (default), **and**
2. A `manifest` argument was passed to `write_tables`.

The function-level CLI does this automatically. Programmatic callers
must build the manifest first via `build_manifest(...)` and pass it
through:

```python
from plotsim import generate_tables_with_state, build_manifest, write_tables

tables, state = generate_tables_with_state(cfg)
manifest = build_manifest(
    cfg, state.trajectories, tables,
    scd_state=state.scd, bridge_state=state.bridges,
    entity_metrics=state.entity_metrics,
)
write_tables(tables, cfg, manifest=manifest)
```

The JSON serialization is byte-deterministic — same `(config, seed)`
produces a byte-identical `manifest.json`. Encoding: UTF-8,
`sort_keys=True`, `indent=2`, trailing newline.

---

## Top-level fields

```json
{
  "schema_version": "1.10",
  "seed": 42,
  "config_sha256": "<64-char hex>",
  "archetype_assignments": [...],
  "trajectory_samples": [...],
  "event_firings": [...],
  "scd_events": [...],
  "bridge_associations": [...],
  "quality_injections": [...],
  "holdout": {...} | null,
  "correlation_adjustments": [...] | null,
  "correlation_compensations": [...] | null,
  "bypass_fallback_counts": {...} | null,
  "vectorized_threshold_used": 50 | null,
  "causal_graph": [...],
  "correlations": [...],
  "outlier_injections": [...] | null,
  "parent_child_relations": [...],
  "noise_config": {...} | null,
  "seasonal_decomposition": {...},
  "regression_pairs_global": [...],
  "regression_pairs_by_archetype": {...},
  "variance_partitions": [...],
  "variance_partitions_by_segment": [...],
  "gp_kernel_fits": [...]
}
```

| Field | Type | Description |
|---|---|---|
| `schema_version` | `str` | Wire-shape version. Currently `"1.10"` (bumped over time as new additive sections — `causal_graph`, `correlations`, `outlier_injections`, multi-source mappings, `parent_child_relations`, `noise_config` — landed; 1.7 → 1.8 extended `noise_config` with `noise_family` / `degrees_of_freedom`; 1.8 → 1.9 added the optional `target_metric` field on the per-entity `treatment` and per-cohort `treatment_cohorts` records; 1.9 → 1.10 added the `seasonal_decomposition` snapshot plus per-pair OLS summaries in `regression_pairs_global` / `regression_pairs_by_archetype`; the `variance_partitions` / `variance_partitions_by_segment` / `gp_kernel_fits` sections landed additively on 1.10 with no version bump) |
| `seed` | `int` | The seed used for generation — `config.seed` |
| `config_sha256` | `str` | Full SHA-256 hex of the JSON-serialized config. Detects config drift between generation and consumption |
| `archetype_assignments` | array | One entry per entity; see below |
| `trajectory_samples` | array | Per-period position cells for a sampled subset of entities |
| `event_firings` | array | Which periods each entity fired in for each event table |
| `scd_events` | array | SCD Type 2 band crossings (empty when no SCD columns are configured) |
| `bridge_associations` | array | Per-bridge M:N association ground truth (empty when no bridges are configured) |
| `quality_injections` | array | Per-issue ground truth — corrupted rows and clean values (empty when `quality.quality_issues` is empty) |
| `holdout` | object or `null` | Train/holdout split metadata. `null` when `holdout.enabled` is False |
| `correlation_adjustments` | array or `null` | Higham nearest-PD projections. `null` when the user matrix was already PD |
| `correlation_compensations` | array or `null` | Trajectory-aware compensation records. `null` when `compensate_correlations` is False or the metric cap was exceeded |
| `bypass_fallback_counts` | object or `null` | Per-archetype count of cells that fell back to the scalar copula path. `null` in serial mode |
| `vectorized_threshold_used` | `int` or `null` | The auto-mode entity-count threshold at generation time. `null` for manifests produced before this field was added |
| `causal_graph` | array | One `CausalEdge` per metric with a non-None `causal_lag`. Empty list when no metric uses `causal_lag` |
| `correlations` | array | One entry per user-declared `config.correlations` pair, with the realized (post-Higham, post-compensation) coefficient. Empty list when no correlations are configured |
| `outlier_injections` | array or `null` | Per-cell outlier-fire log. `null` when skipped (no `outlier_rate`, vectorized mode, or cell budget exceeded). `[]` when the detector ran and observed no firings |
| `noise_config` | object or `null` | Noise-model record. `null` when the run uses the default magnitude-scaled gaussian lane; populated when EITHER `noise.scale_with_trajectory` is `true` OR `noise.noise_family` is non-default (`"student_t"` / `"laplace"`) |
| `seasonal_decomposition` | object | Snapshot of the seasonal-strength inputs the engine consumed. Always emitted; configs without `seasonal_effects` get the empty-sentinel shape (empty list / empty dicts) |
| `regression_pairs_global` | array | Pair-wise OLS summary (slope, intercept, r², residual variance) for every declared correlation pair, pooled across every entity. Empty list when no correlations are configured |
| `regression_pairs_by_archetype` | object | Same OLS summary as `regression_pairs_global` but grouped by `Entity.archetype`. Keys are archetype names; values mirror the global list shape. Empty dict when no correlations are configured |
| `variance_partitions` | array | Nested-ANOVA variance decomposition per metric, with `Entity.archetype` as the between-group axis. One record per metric. Empty list when the config declares no metrics |
| `variance_partitions_by_segment` | array | Same nested-ANOVA decomposition with curve segment as the between-group axis, computed per archetype. One record per `(metric, archetype)` pair. Segments are never pooled across archetypes. Empty list when the config declares no metrics |
| `gp_kernel_fits` | array | RBF Gaussian-process kernel fits over each archetype's trajectory shape, plus per-entity records for entities that carry trajectory `overrides`. Empty list when the config declares no metrics |

---

## `archetype_assignments`

One ground-truth label per entity — the archetype the engine drove their
trajectory from.

```json
{
  "archetype_assignments": [
    { "entity": "growers_001",   "archetype": "growth" },
    { "entity": "decliners_002", "archetype": "decline" }
  ]
}
```

| Field | Type | Description |
|---|---|---|
| `entity` | `str` | Entity name (matches `config.entities[i].name`) |
| `archetype` | `str` | Archetype name (matches `config.archetypes[i].name`) |

Sorted by `entity` for stable diff under the same config.

**Use case** — train a classifier on the fact-table aggregates and
score it against this column. The archetype is the latent class label
your model is trying to recover; this list is the answer key.

---

## `trajectory_samples`

Per-period trajectory positions for a sampled subset of entities.

```json
{
  "trajectory_samples": [
    { "entity": "growers_001", "period_index": 0,  "position": 0.05 },
    { "entity": "growers_001", "period_index": 1,  "position": 0.08 },
    { "entity": "growers_001", "period_index": 2,  "position": 0.13 }
  ]
}
```

| Field | Type | Description |
|---|---|---|
| `entity` | `str` | Entity name |
| `period_index` | `int` | Zero-based period index. `0` is the first period of `time_window` |
| `position` | `float` | Position in `[0, 1]` |

Position is the noise-free, distribution-free behavioral state the
engine derived every metric from. It's not present in the fact table —
the fact table holds *realized* values shaped by polarity, distribution,
correlation, and noise.

**Sampled subset** — controlled by `config.manifest.trajectory_sample_rate`
(default `1.0`, meaning every entity). The selection is the first
`ceil(n_entities × sample_rate)` entities under sorted-name order, so it
stays stable regardless of seed. Set this below `1.0` for very large
configs where the per-period tape would dominate manifest size.

**Use case** — verify the trajectory-first invariant from the manifest:
combine with [`trace_metric_cell`](./api-reference.md#trace_metric_cell)
to confirm `position → realized cell` for any entity in the sample.

---

## `event_firings`

For each event table, which periods each entity fired in.

```json
{
  "event_firings": [
    {
      "entity": "growers_001",
      "table": "evt_login",
      "period_indices": [0, 1, 2, 3, 5, 7]
    }
  ]
}
```

| Field | Type | Description |
|---|---|---|
| `entity` | `str` | Entity name |
| `table` | `str` | Event-table name |
| `period_indices` | array of `int` | Sorted ascending; the periods this entity contributed at least one row in |

Empty `period_indices` are kept rather than omitted, so a downstream
consumer can iterate the full entity × event-table matrix without
fallback logic.

Both threshold and proportional events surface here. The manifest
records *observed* firings, not the configured triggers.

---

## `scd_events`

SCD Type 2 band crossings — emitted only for transitions, not the
initial band.

```json
{
  "scd_events": [
    {
      "dim_table": "dim_customer",
      "entity": "growers_001",
      "period_index": 5,
      "old_label": "starter",
      "new_label": "pro",
      "old_dim_row_id": 12,
      "new_dim_row_id": 13,
      "trigger_metric": "fct_engagement.mrr",
      "trigger_position": 0.52
    }
  ]
}
```

| Field | Type | Description |
|---|---|---|
| `dim_table` | `str` | The dim table the SCD column lives on |
| `entity` | `str` | Entity name |
| `period_index` | `int` | Period the crossing happened |
| `old_label` | `str` | Band the entity was in before |
| `new_label` | `str` | Band the entity advanced to |
| `old_dim_row_id` | `int` | Surrogate row ID of the closing version |
| `new_dim_row_id` | `int` | Surrogate row ID of the opening version |
| `trigger_metric` | `str` | The metric whose threshold was crossed (`<fact_table>.<metric>`) |
| `trigger_position` | `float` | Trajectory position at the crossing period |

Empty when no `scd` columns are configured. Sorted by dim table, then
entity, then period for stable ordering.

**Use case** — join against `trajectory_samples` to recover the exact
position that triggered each band change.

---

## `bridge_associations`

Many-to-many associations recorded as ground truth.

```json
{
  "bridge_associations": [
    {
      "bridge": "customer_subscription",
      "entity": "growers_001",
      "targets": ["sub_007", "sub_023", "sub_041"],
      "cardinality": 3
    }
  ]
}
```

| Field | Type | Description |
|---|---|---|
| `bridge` | `str` | Bridge-table name |
| `entity` | `str` | First-dim entity name |
| `targets` | array | Second-dim FK values (PKs for non-SCD dims; `dim_row_id` for SCD dims) |
| `cardinality` | `int` | `len(targets)`. Surfaced separately so consumers can aggregate without iterating each tuple |

Empty when no `bridges` are configured. Sorted by bridge name, then
entity name.

---

## `quality_injections`

Ground truth for post-generation data corruption.

```json
{
  "quality_injections": [
    {
      "issue_index": 0,
      "issue_type": "null_injection",
      "table": "fct_engagement",
      "column": "engagement",
      "row_indices": [3, 17, 42],
      "clean_values": [0.42, 0.71, 0.18]
    },
    {
      "issue_index": 1,
      "issue_type": "duplicate_rows",
      "table": "fct_engagement",
      "column": "_rows",
      "row_indices": [8, 19],
      "clean_values": []
    }
  ]
}
```

| Field | Type | Description |
|---|---|---|
| `issue_index` | `int` | Position in `config.quality.quality_issues` — distinguishes multiple issues |
| `issue_type` | `str` | `null_injection`, `duplicate_rows`, `type_mismatch`, `late_arrival`, `schema_drift`, or `volume_anomaly` |
| `table` | `str` | Target table |
| `column` | `str` | Target column. For row-level issues this is a sentinel — `_rows` for duplicates and volume anomalies, `_arrival_period` for late arrivals |
| `row_indices` | array of `int` | Row positions in the corrupted DataFrame — the rows that were affected |
| `clean_values` | array | Original values at those rows. Empty for `duplicate_rows`, `late_arrival`, and `volume_anomaly` (the corruption is row-level, not per-cell) |

Empty when `config.quality.quality_issues` is empty.

**Use case** — recover the clean dataset from the corrupted output
without re-running generation, or train a model that explicitly handles
the corruption pattern.

---

## `holdout`

Train/holdout split metadata. Present only when `config.holdout.enabled`
is True; `null` otherwise.

```json
{
  "holdout": {
    "target_metric": "mrr",
    "holdout_periods": 3,
    "cutoff_period_index": 21
  }
}
```

| Field | Type | Description |
|---|---|---|
| `target_metric` | `str` | Mirror of `config.holdout.target` |
| `holdout_periods` | `int` | Mirror of `config.holdout.periods` |
| `cutoff_period_index` | `int` | The resolved boundary — `n_periods - holdout_periods`. Periods `[0, cutoff)` are training; `[cutoff, n_periods)` are holdout |

**Use case** — slice an unsplit fact table or its derivative on the
same axis without recomputing `period_count` from `time_window`.

---

## `correlation_adjustments`

Pairs whose configured correlation was projected to a nearby PD value
because the user-declared matrix wasn't positive semi-definite.

```json
{
  "correlation_adjustments": [
    {
      "metric_a": "engagement",
      "metric_b": "support_tickets",
      "requested": -0.75,
      "achieved":  -0.68,
      "adjustment": 0.07
    }
  ]
}
```

| Field | Type | Description |
|---|---|---|
| `metric_a` / `metric_b` | `str` | The pair |
| `requested` | `float` | Coefficient declared in the config |
| `achieved` | `float` | Value at the same `(i, j)` cell after Higham projection |
| `adjustment` | `float` | `abs(requested - achieved)` |

`null` when the user-declared matrix was already PD (the common case)
or when no correlations were configured. Pairs whose adjustment falls
below the numerical noise floor (~1e-12) are dropped, so an empty array
distinguishes "all pairs were tolerance-clean" from `null` ("no
projection needed").

**Use case** — flag configs whose declared correlations couldn't be
delivered exactly, and decide whether to relax the matrix or accept the
projected value.

---

## `correlation_compensations`

Pairs the engine pre-compensated for trajectory-induced covariance —
recorded only when `compensate_correlations` is True.

```json
{
  "correlation_compensations": [
    {
      "metric_a": "engagement",
      "metric_b": "mrr",
      "user_target": 0.55,
      "trajectory_contribution": 0.32,
      "compensated_target": 0.23,
      "achievable": 0.23,
      "infeasible": false,
      "adjustment": 0.32
    }
  ]
}
```

| Field | Type | Description |
|---|---|---|
| `metric_a` / `metric_b` | `str` | The pair |
| `user_target` | `float` | Coefficient declared in the config's `connections` block |
| `trajectory_contribution` | `float` | Within-archetype-weighted Pearson the trajectory's centers induce, in `[-1, 1]` |
| `compensated_target` | `float` | Pre-clamp `user_target - trajectory_contribution`. May fall outside `[-1, 1]` |
| `achievable` | `float` | `compensated_target` clamped to `[-1, 1]`. The value the copula actually targets |
| `infeasible` | `bool` | True when `compensated_target` fell outside `[-1, 1]`. The realized table-wide Pearson lands at `user_target ± something < |user_target|` for these |
| `adjustment` | `float` | `abs(user_target - achievable)` |

`null` when:

- `compensate_correlations` is False, or
- the config has no `correlations` / `connections` block, or
- the metric count exceeded the cap (20) and the engine fell back to
  the direct-copula path.

Distinct from `correlation_adjustments`: that records "your matrix
wasn't PD, we projected"; this records "your target was compensated for
the trajectory's structural contribution before reaching the copula."
Both can populate on a single run.

**Use case** — sort by `adjustment` to find pairs whose realized
correlation drifts most from the configured target. Pairs flagged
`infeasible: true` can never reach the user target on the current
config — relax the trajectory mix or lower the magnitude.

---

## `bypass_fallback_counts`

Per-archetype count of cells that triggered the per-row scalar fallback
in vectorized generation mode.

```json
{
  "bypass_fallback_counts": {
    "growth": 0,
    "decline": 12,
    "spike_then_crash": 47
  }
}
```

| Form | Meaning |
|---|---|
| `null` | Serial mode — bypass was never measured |
| `{}` | Vectorized ran with zero bypass cells (the production-shape case) |
| `{name: count, ...}` | Vectorized hit the scalar fallback for `count` cells under archetype `name` |

A non-zero count means vectorized mode wasn't fully effective for that
archetype on this config. Surfaces "vectorized isn't faster than serial
here" investigations directly.

---

## `vectorized_threshold_used`

The value of the auto-mode entity-count threshold at generation time.

| Form | Meaning |
|---|---|
| `int` | Recorded threshold (currently `50`) |
| `null` | Older manifest produced before this field existed |

Recorded so old manifests stay reproducible if the constant changes in
a later release — comparing this against the current threshold lets a
consumer detect that a re-run would land in a different
`generation_mode`.

---

## `causal_graph`

The run's causal-lag DAG, derived from `config.metrics`.

```json
{
  "causal_graph": [
    {
      "driver": "engagement",
      "target": "support_tickets",
      "lag_periods": 2,
      "blend_weight": 1.0
    }
  ]
}
```

| Field | Type | Description |
|---|---|---|
| `driver` | `str` | Source metric name. Mirrors `metric.causal_lag.driver` |
| `target` | `str` | Target metric name (the metric whose `causal_lag` field declared the edge) |
| `lag_periods` | `int` | Period offset the target reads the driver at. Mirrors `metric.causal_lag.lag_periods` |
| `blend_weight` | `float` | Blend coefficient — `1.0` is full lag override, `0.0` ignores the lag, intermediate values blend between the lagged driver and the target's own current trajectory position |

One edge per metric whose `causal_lag` field is set. Empty list when no
metric uses `causal_lag`. Sorted by `(driver, target)` for stable JSON
output.

**Use case** — reconstruct the run's directed causal graph without
re-parsing the YAML config. A downstream lineage tool can build "what
upstream metrics could have caused this metric to move" queries
directly from this list.

---

## `correlations`

One entry per user-declared correlation pair, with the realized
coefficient the engine actually drove the copula against.

```json
{
  "correlations": [
    {
      "metric_a": "engagement",
      "metric_b": "mrr",
      "requested": 0.82,
      "projected": 0.7332
    }
  ]
}
```

| Field | Type | Description |
|---|---|---|
| `metric_a` | `str` | First metric of the user-declared pair |
| `metric_b` | `str` | Second metric of the pair |
| `requested` | `float` | The coefficient written in `config.correlations` — what the user asked for |
| `projected` | `float` | The coefficient at `(metric_a, metric_b)` of the matrix the engine drove the copula against — i.e. after trajectory-aware compensation (when enabled) and Higham nearest-PD projection (when needed). May differ from `requested` when those steps adjusted the matrix |

One entry per pair in `config.correlations`. Auto-zero off-diagonals
(pairs the user did not declare) are not recorded. Sorted by
`(metric_a, metric_b)` for stable JSON output.

**Distinct from** `correlation_adjustments` (which only fires when
Higham had to project) and `correlation_compensations` (which only
fires when trajectory-aware compensation ran). `correlations` fires on every run
that has correlations, so consumers always see the realized value
regardless of whether the matrix needed adjustment.

**Use case** — verify that the realized coefficient matches the user's
intent. A pair where `abs(requested - projected) > tolerance` is a
signal that the matrix was incompatible with the trajectory's
structural covariance and the engine had to bend it; a learner can
rank these by deviation magnitude to flag the configuration choices
that introduced the most drift.

---

## `outlier_injections`

Per-cell record of which cells had `noise.outlier_rate` fire during
generation.

```json
{
  "outlier_injections": [
    { "entity": "acme_corp_cohort", "period_index": 8, "metric": "engagement" },
    { "entity": "acme_corp_cohort", "period_index": 9, "metric": "churn_risk" }
  ]
}
```

| Field | Type | Description |
|---|---|---|
| `entity` | `str` | Entity name — matches `config.entities[i].name` |
| `period_index` | `int` | Zero-based period index. `0` is the first period of `time_window` |
| `metric` | `str` | Metric name — matches `config.metrics[i].name` |

Sorted by `(entity, period_index, metric)` for stable JSON output.

### When the section is `null`

The detector replays the engine pipeline with an inline noise hook to
observe outlier firings. It skips three cases, all of which surface as
`outlier_injections: null`:

| Skip reason | Why |
|---|---|
| `noise.outlier_rate == 0.0` | The noise pipeline never consults the outlier branch — re-running the engine to observe zero firings would be wasted work |
| Vectorized generation mode | `_apply_noise_batch` consumes RNG in a different order than per-cell `apply_noise`. A serial-mode replay would record firings at cells that don't match the vectorized fact tables. Recording vectorized outliers needs a parallel batch detector — out of scope for this release |
| Cell count exceeds budget | The detector replays the full metric pipeline once. Total cells (`n_entities × n_periods × n_metrics`) above `OUTLIER_DETECTION_CELL_BUDGET` (1,000,000) trigger a skip — the replay cost is not justified for what is effectively a debug aid |

`[]` (empty list) means the detector ran and observed no firings — a
valid outcome at low `outlier_rate` and small cell counts. Distinct
from `null` (skipped).

**Use case** — score an anomaly-detection model. Each outlier
injection is ground truth: the cell got an outlier multiplier from
`apply_noise`, so a detector that fails to flag it has missed a known
positive. An empty list means clean data with no anomalies to find.

---

## `parent_child_relations`

Parent-fact / child-fact pairing records — one entry per
`per_parent_row` child table declared in the config.

```json
{
  "parent_child_relations": [
    {
      "parent_table": "fct_orders",
      "child_table": "fct_order_items",
      "children_per_row_min": 1,
      "children_per_row_max": 5,
      "parent_row_count": 221,
      "child_row_count": 662
    }
  ]
}
```

| Field | Type | Description |
|---|---|---|
| `parent_table` | `str` | Name of the parent fact table |
| `child_table` | `str` | Name of the `per_parent_row` child fact table |
| `children_per_row_min` | `int` | Inclusive lower bound declared on the child (`Table.children_per_row[0]`) |
| `children_per_row_max` | `int` | Inclusive upper bound declared on the child (`Table.children_per_row[1]`) |
| `parent_row_count` | `int` | Actual row count of the parent fact in the generated output |
| `child_row_count` | `int` | Actual row count of the child fact in the generated output |

Empty list when the config declares no `per_parent_row` tables.
Sorted by `child_table` for stable diff under the same config.

**Use case** — verify the trajectory-driven parent fan-out without
re-reading the CSVs. `parent_row_count` reflects the
trajectory-driven row counts; `child_row_count` reflects the
configured `children_per_row` range × parent rows. A divergent
`child_row_count / parent_row_count` ratio across runs at the same
seed signals a generation regression.

---

## `noise_config`

Noise-model record — emitted whenever the run diverges from the
historical magnitude-scaled gaussian lane. Two triggers, either
sufficient: `noise.scale_with_trajectory: true` (heteroscedastic
amplitude) OR `noise.noise_family` is non-default (heavy-tailed
family — `"student_t"` or `"laplace"`). `null` for the default lane
(and absent from manifests produced before `schema_version: "1.7"`).

```json
{
  "noise_config": {
    "gaussian_sigma": 0.20,
    "outlier_rate": 0.0,
    "mcar_rate": 0.0,
    "scale_with_trajectory": true,
    "noise_family": "student_t",
    "degrees_of_freedom": 4.0
  }
}
```

| Field | Type | Description |
|---|---|---|
| `gaussian_sigma` | `float` | The σ multiplier from `config.noise.gaussian_sigma`. Under the heteroscedastic lane the realized scale at a cell is `gaussian_sigma × trajectory_position`; otherwise `gaussian_sigma × \|value\|`. Used by every family as the scale parameter |
| `outlier_rate` | `float` | Mirrors `config.noise.outlier_rate`. Unaffected by the family or heteroscedastic flag — recorded here for completeness so the manifest fully describes the noise model |
| `mcar_rate` | `float` | Mirrors `config.noise.mcar_rate`. Unaffected by the family or heteroscedastic flag |
| `scale_with_trajectory` | `bool` | `true` when the heteroscedastic lane was engaged. `false` when the record was emitted purely because `noise_family` diverged from the default |
| `noise_family` | `str` | The additive-jitter distribution — one of `"gaussian"`, `"student_t"`, `"laplace"`. Mirrors `config.noise.noise_family` |
| `degrees_of_freedom` | `float` or `null` | Populated only when `noise_family == "student_t"`; `null` otherwise |

**Use case** — distinguish a run that opted into position-scaled or
heavy-tailed gaussian noise from one that didn't, without re-reading
the YAML config. Anomaly-detection scoring that assumes uniform
gaussian noise variance can read this record to switch to a
position-aware or family-aware likelihood model — e.g., switching to
a t-distribution likelihood when `noise_family == "student_t"` keeps
the scorer well-calibrated under the heavier-tailed residuals.

---

## `seasonal_decomposition`

Snapshot of the seasonal-strength inputs the engine consumed during
metric generation.

```json
{
  "seasonal_decomposition": {
    "seasonal_factors": [0.0, 0.8, 0.8, 0.0, 0.0, -0.3, -0.3, 0.0, 0.0, 0.0, 0.0, 0.8],
    "metric_seasonal_sensitivities": {
      "engagement": 1.0,
      "mrr": 0.6
    },
    "entity_seasonal_sensitivities": {
      "growers_001": 1.0,
      "decliners_002": 0.0
    }
  }
}
```

| Field | Type | Description |
|---|---|---|
| `seasonal_factors` | array of `float` | Length-`n_periods` global strength array. Entry `t` is the sum of every `SeasonalEffect.strength` whose `months` set contains period `t`'s calendar month |
| `metric_seasonal_sensitivities` | object | One entry per metric, keyed by `Metric.name` and valued by `Metric.seasonal_sensitivity`. The per-metric multiplier the engine applies on top of the global strength |
| `entity_seasonal_sensitivities` | object | One entry per entity, keyed by `Entity.name` and valued by `Entity.seasonal_sensitivity`. The per-entity multiplier the engine applies on top of the global strength |

### When the section is the empty sentinel

Configs without any `seasonal_effects` declared get the empty-sentinel
shape — `seasonal_factors: []`, `metric_seasonal_sensitivities: {}`,
`entity_seasonal_sensitivities: {}` — rather than `null`. The
sensitivity multipliers are inert in that lane (the engine short-
circuits before applying them), so recording them would just be noise.
Always present so a downstream consumer can iterate the section without
a None-check.

**Use case** — reconstruct the engine's effective seasonal lift at any
cell without re-reading the YAML config. For an `(entity, period, metric)`
triple:

```python
lift = (
    manifest["seasonal_decomposition"]["seasonal_factors"][period]
    * manifest["seasonal_decomposition"]["metric_seasonal_sensitivities"][metric]
    * manifest["seasonal_decomposition"]["entity_seasonal_sensitivities"][entity]
)
```

A seasonality-aware anomaly detector can subtract this lift before
scoring; a feature pipeline can expose `seasonal_factor` as a regressor
that exactly mirrors the engine's modulation.

---

## `regression_pairs_global`

Pair-wise ordinary-least-squares fit for every declared correlation,
pooled across every entity and period.

```json
{
  "regression_pairs_global": [
    {
      "metric_a": "engagement",
      "metric_b": "mrr",
      "beta_a_to_b": 0.84,
      "intercept_a_to_b": 12.3,
      "beta_b_to_a": 0.71,
      "intercept_b_to_a": -4.1,
      "r_squared": 0.6,
      "residual_variance_a_to_b": 18.7,
      "residual_variance_b_to_a": 0.04,
      "n_observations": 720
    }
  ]
}
```

| Field | Type | Description |
|---|---|---|
| `metric_a` / `metric_b` | `str` | The pair, in the order the user declared them in `config.correlations` |
| `beta_a_to_b` | `float` | OLS slope for `b = beta * a + intercept` over the pooled `(a, b)` observations |
| `intercept_a_to_b` | `float` | OLS intercept for the same regression |
| `beta_b_to_a` | `float` | OLS slope for the reverse regression `a = beta * b + intercept` |
| `intercept_b_to_a` | `float` | OLS intercept for the reverse regression |
| `r_squared` | `float` | Direction-invariant coefficient of determination. Equal to `corr(a, b) ** 2` on the same observations |
| `residual_variance_a_to_b` | `float` | Variance of `b - (beta_a_to_b * a + intercept_a_to_b)` — the unexplained-noise scale for the `a → b` direction |
| `residual_variance_b_to_a` | `float` | Same for the reverse direction |
| `n_observations` | `int` | Count of finite `(a, b)` pairs used. Cells with NaN in either metric (cold-start lead-ins, MCAR-rewritten values) are excluded |

One entry per pair in `config.correlations`. Auto-zero off-diagonals
(pairs the user did not declare) are not recorded. Sorted by
`(metric_a, metric_b)` for stable JSON output.

**Distinct from** `correlations` (which records the realized Pearson
coefficient the copula targeted). `regression_pairs_global` describes
the *fitted linear relationship* between the realized series — slope
and intercept, plus the unexplained variance. A high `r_squared`
combined with a small `residual_variance` says the pair moves
tightly together along a straight line; a high `r_squared` with
asymmetric residual variances says one direction predicts the other
better than vice-versa (which is normal under unequal metric scales).

`n_observations < 2` is a degenerate case (sparse cold-start, no
overlap between metric domains); the record's β / intercept / variance
fields are all `0.0` and downstream consumers should gate on the count
before reading the coefficients.

**Use case** — score a regression baseline. A predictor of `mrr` from
`engagement` should land near `beta_a_to_b` with residual variance
close to `residual_variance_a_to_b`. Larger deviations flag either
model misspecification or that the consumer is over-fitting noise the
manifest already attributes to residuals.

---

## `regression_pairs_by_archetype`

The same OLS surface as `regression_pairs_global`, but restricted to
each archetype's entity subset so a consumer can see which archetypes
carry the declared correlations.

```json
{
  "regression_pairs_by_archetype": {
    "growth": [
      {
        "metric_a": "engagement",
        "metric_b": "mrr",
        "beta_a_to_b": 0.91,
        "intercept_a_to_b": 9.2,
        "beta_b_to_a": 0.86,
        "intercept_b_to_a": -7.0,
        "r_squared": 0.78,
        "residual_variance_a_to_b": 10.4,
        "residual_variance_b_to_a": 0.02,
        "n_observations": 360
      }
    ],
    "decline": [
      {
        "metric_a": "engagement",
        "metric_b": "mrr",
        "beta_a_to_b": 0.62,
        "intercept_a_to_b": 15.8,
        "beta_b_to_a": 0.41,
        "intercept_b_to_a": 1.2,
        "r_squared": 0.31,
        "residual_variance_a_to_b": 25.6,
        "residual_variance_b_to_a": 0.08,
        "n_observations": 360
      }
    ]
  }
}
```

The top-level object's keys are archetype names (matching
`Entity.archetype`); each value list mirrors the
`regression_pairs_global` shape, one entry per declared pair.
Archetypes that contribute no finite observations are omitted entirely
(rather than mapped to an empty list) — the dict reflects archetypes
that actually contributed to the fit.

Empty `{}` when no correlations are declared.

**Use case** — diagnose where in the population a declared correlation
is strongest. A pair with a high pooled `r_squared` but per-archetype
values that swing widely is a signal that the correlation is a mixture
artefact, not a within-archetype relationship — a model trained on the
pooled fit will mispredict for the archetype whose β diverges most.

---

## `variance_partitions`

Nested-ANOVA variance decomposition per metric, with `Entity.archetype`
as the between-group axis.

```json
{
  "variance_partitions": [
    {
      "metric": "engagement",
      "scope": "archetype",
      "scope_name": "all",
      "ss_between": 12.4,
      "ss_within_entity": 6.8,
      "ss_residual": 41.0,
      "fraction_between": 0.206,
      "fraction_within_entity": 0.113,
      "fraction_residual": 0.681,
      "degrees_of_freedom_between": 1,
      "degrees_of_freedom_within": 18,
      "degrees_of_freedom_residual": 220,
      "n_observations": 240,
      "cold_start_entities_excluded": 0
    }
  ]
}
```

| Field | Type | Description |
|---|---|---|
| `metric` | `str` | The metric this record decomposes (matches `config.metrics[i].name`) |
| `scope` | `str` | Always `"archetype"` for this section |
| `scope_name` | `str` | Always the literal sentinel `"all"` for this section — the partition spans every archetype the config declares |
| `ss_between` | `float` | Sum-of-squares attributable to the grouping axis (variance between archetype means around the grand mean) |
| `ss_within_entity` | `float` | Sum-of-squares between entity means within the same archetype |
| `ss_residual` | `float` | Within-entity sum-of-squares over time (residual to each entity's own mean) |
| `fraction_between` / `fraction_within_entity` / `fraction_residual` | `float` | Each `ss_*` divided by `ss_total = ss_between + ss_within_entity + ss_residual`. The three fractions sum to `1.0` (or to `0.0` for a fully constant metric) |
| `degrees_of_freedom_between` | `int` | `n_groups - 1` where `n_groups` is the number of archetypes that contributed observations |
| `degrees_of_freedom_within` | `int` | `n_cells - n_groups` where `n_cells` is the number of distinct `(archetype, entity)` pairs |
| `degrees_of_freedom_residual` | `int` | `n_observations - n_cells` |
| `n_observations` | `int` | Count of finite `(entity, period)` cells used. Cells with NaN (cold-start lead-ins, MCAR-rewritten values) are excluded |
| `cold_start_entities_excluded` | `int` | Count of entities that contributed at least one NaN cell to this partition. Surfaces "this section dropped data" without forcing a re-derivation of the NaN tally |

The three sums-of-squares satisfy `ss_between + ss_within_entity +
ss_residual == ss_total` exactly (modulo floating-point rounding at
`rtol≈1e-10`). One record per metric, sorted by `metric` for stable JSON
output.

**Use case** — diagnose how much of a metric's spread is explained by
the latent archetype label vs. by entity-level idiosyncrasy vs. by
within-entity time-series noise. A high `fraction_between` says the
archetype is the dominant driver of metric values; a high
`fraction_residual` with a low `fraction_between` says metric values
are essentially noise on top of an entity-specific mean.

---

## `variance_partitions_by_segment`

The same nested-ANOVA decomposition with curve segment as the between-
group axis, computed per archetype.

```json
{
  "variance_partitions_by_segment": [
    {
      "metric": "engagement",
      "scope": "segment",
      "scope_name": "growth",
      "ss_between": 8.1,
      "ss_within_entity": 4.6,
      "ss_residual": 22.3,
      "fraction_between": 0.231,
      "fraction_within_entity": 0.131,
      "fraction_residual": 0.638,
      "degrees_of_freedom_between": 2,
      "degrees_of_freedom_within": 27,
      "degrees_of_freedom_residual": 90,
      "n_observations": 120,
      "cold_start_entities_excluded": 0
    }
  ]
}
```

The schema is identical to `variance_partitions`. The differences:

| Field | Difference |
|---|---|
| `scope` | Always `"segment"` |
| `scope_name` | The archetype name. Each archetype's segments are decomposed in isolation; the section never pools observations across archetypes |
| `degrees_of_freedom_between` | `n_curve_segments - 1` for the named archetype |
| `n_observations` | Restricted to entities of `scope_name`'s archetype only |

One record per `(metric, archetype)` pair whose entities contributed at
least one finite observation. Sorted by `(metric, scope_name)` for
stable JSON output.

Each entity's segment membership is derived from its own boundary
computation — entities with `start_period > 0` (cold-start) or with
trajectory `overrides` contribute observations to the segment they
actually occupied at each period, not to the archetype baseline
segment. Period membership is reported as `segment_0`, `segment_1`, …
internally and rolls up into the per-segment SS terms; downstream
consumers see only the partition totals.

**Use case** — locate where in an archetype's curve a metric's
variance spreads most. A high `ss_between` for an archetype with three
distinct curve segments says the metric tracks the curve; a low
`ss_between` says the metric is decoupled from the archetype's
narrative phase structure.

---

## `gp_kernel_fits`

RBF Gaussian-process kernel fits over each archetype's trajectory
shape. Surfaces a smoothness characterization the trajectory tape
itself doesn't directly expose.

```json
{
  "gp_kernel_fits": [
    {
      "scope_type": "archetype",
      "scope_name": "growth",
      "kernel_type": "rbf",
      "hyperparameters": {
        "length_scale": 4.7,
        "signal_variance": 0.31,
        "noise_variance": 0.0008
      },
      "log_marginal_likelihood": -3.2,
      "n_train": 12,
      "converged": true
    },
    {
      "scope_type": "archetype",
      "scope_name": "flat",
      "kernel_type": "rbf",
      "hyperparameters": {},
      "log_marginal_likelihood": null,
      "n_train": 12,
      "converged": false
    },
    {
      "scope_type": "entity",
      "scope_name": "growers_001",
      "kernel_type": "rbf",
      "hyperparameters": {
        "length_scale": 6.1,
        "signal_variance": 0.28,
        "noise_variance": 0.0012
      },
      "log_marginal_likelihood": -2.9,
      "n_train": 10,
      "converged": true
    }
  ]
}
```

| Field | Type | Description |
|---|---|---|
| `scope_type` | `str` | Either `"archetype"` (one fit per declared archetype, against the archetype's clean trajectory) or `"entity"` (one fit per override-bearing entity, against that entity's specific trajectory) |
| `scope_name` | `str` | The archetype name or the entity name, depending on `scope_type` |
| `kernel_type` | `str` | Always `"rbf"` for now. Reserved as a discriminator for future kernel families |
| `hyperparameters` | object | Three keys when `converged=true`: `length_scale`, `signal_variance`, `noise_variance`. All in the natural (unstandardized) scale — `length_scale` is in units of period indices. Empty `{}` when `converged=false` |
| `log_marginal_likelihood` | `float` or `null` | The maximized value (positive sign — the fitter minimizes the negative log likelihood internally and negates before reporting). `null` when `converged=false` |
| `n_train` | `int` | Count of finite `(period, position)` training pairs used. NaN cells (cold-start prefix periods) are excluded |
| `converged` | `bool` | `true` when the optimizer reported success AND produced finite hyperparameters. `false` otherwise — see below for the failure modes |

### Records emitted

* One `scope_type="archetype"` record per archetype the config
  declares. The fit consumes the archetype's *clean* trajectory (no
  overrides, no cold-start shift) so the kernel characterizes the
  archetype's intrinsic shape, not any individual entity's realized
  data.
* One `scope_type="entity"` record per entity carrying a non-`None`
  `overrides` field. Default-trajectory entities do *not* produce
  per-entity records — only override-bearing entities do.

### When `converged=false`

The optimizer's failure paths are surfaced as a non-fatal record (the
manifest build never raises on a failed fit):

* **Flat trajectory** — variance below the floor (≈ `1e-12`). The RBF
  likelihood surface is degenerate when the signal has no variance to
  fit.
* **Sparse data** — fewer than three finite training points (the
  kernel has three hyperparameters; under-determined fits are
  short-circuited).
* **Optimizer failure** — `scipy.optimize.minimize` reports
  non-success or returns a non-finite NLL.
* **Numerical blow-up** — Cholesky factorization fails on the
  covariance matrix despite the noise-variance floor.

`converged=false` records carry an empty `hyperparameters` dict and a
`null` log marginal likelihood. Consumers should gate downstream
usage on the flag rather than inspecting `hyperparameters` directly.

**Use case** — characterize trajectory smoothness without re-fitting a
GP downstream. A short `length_scale` (≪ `n_periods`) says the
archetype oscillates or has fast transitions; a long `length_scale`
(≈ `n_periods`) says it's gradual or monotone. Compare per-entity
records against their parent archetype to detect override-driven
shape divergence — a per-entity `length_scale` that disagrees with
the archetype baseline is direct evidence that the override pushed
the entity's curve onto a different smoothness regime.

---

## Reading the manifest in Python

```python
import json
from pathlib import Path

manifest = json.loads(Path("output/manifest.json").read_text())

# Build the entity → archetype lookup
labels = {a["entity"]: a["archetype"] for a in manifest["archetype_assignments"]}

# Reconstruct an entity's trajectory tape
positions = sorted(
    (s["period_index"], s["position"])
    for s in manifest["trajectory_samples"]
    if s["entity"] == "growers_001"
)

# Detect quality corruption on a column
nullified_rows = [
    inj["row_indices"]
    for inj in manifest["quality_injections"]
    if inj["issue_type"] == "null_injection"
    and inj["table"] == "fct_engagement"
    and inj["column"] == "engagement"
]
```

`pydantic` users can validate the on-disk JSON against the typed
manifest model directly:

```python
from plotsim import ManifestSchema

manifest = ManifestSchema.model_validate_json(Path("output/manifest.json").read_text())
```

The model has `extra="forbid"`, so a malformed or out-of-version manifest
fails loudly during validation rather than silently dropping unknown
fields.
