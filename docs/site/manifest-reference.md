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
  "schema_version": "1.1",
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
  "outlier_injections": [...] | null
}
```

| Field | Type | Description |
|---|---|---|
| `schema_version` | `str` | Wire-shape version. Currently `"1.1"` (bumped from `"1.0"` in 0.6-M5 for the additive `causal_graph`, `correlations`, `outlier_injections` sections) |
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
| `vectorized_threshold_used` | `int` or `null` | The auto-mode entity-count threshold at generation time. `null` for pre-M121b manifests on disk |
| `causal_graph` | array | One `CausalEdge` per metric with a non-None `causal_lag`. Empty list when no metric uses `causal_lag` |
| `correlations` | array | One entry per user-declared `config.correlations` pair, with the realized (post-Higham, post-compensation) coefficient. Empty list when no correlations are configured |
| `outlier_injections` | array or `null` | Per-cell outlier-fire log. `null` when skipped (no `outlier_rate`, vectorized mode, or cell budget exceeded). `[]` when the detector ran and observed no firings |

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
| `issue_type` | `str` | `null_injection`, `duplicate_rows`, `type_mismatch`, `late_arrival`, or `schema_drift` |
| `table` | `str` | Target table |
| `column` | `str` | Target column. For row-level issues this is a sentinel — `_rows` for duplicates, `_arrival_period` for late arrivals |
| `row_indices` | array of `int` | Row positions in the corrupted DataFrame — the rows that were affected |
| `clean_values` | array | Original values at those rows. Empty for `duplicate_rows` and `late_arrival` (the corruption is row-level, not per-cell) |

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
| `null` | Pre-M121b manifest on disk |

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
| `projected` | `float` | The coefficient at `(metric_a, metric_b)` of the matrix the engine drove the copula against — i.e. after M120 trajectory-aware compensation (when enabled) and M111 Higham nearest-PD projection (when needed). May differ from `requested` when those steps adjusted the matrix |

One entry per pair in `config.correlations`. Auto-zero off-diagonals
(pairs the user did not declare) are not recorded. Sorted by
`(metric_a, metric_b)` for stable JSON output.

**Distinct from** `correlation_adjustments` (which only fires when
Higham had to project) and `correlation_compensations` (which only
fires when M120 compensation ran). `correlations` fires on every run
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
| Vectorized generation mode | `_apply_noise_batch` consumes RNG in a different order than per-cell `apply_noise`. A serial-mode replay would record firings at cells that don't match the vectorized fact tables. Recording vectorized outliers needs a parallel batch detector — out of scope for 0.6-M5 |
| Cell count exceeds budget | The detector replays the full metric pipeline once. Total cells (`n_entities × n_periods × n_metrics`) above `OUTLIER_DETECTION_CELL_BUDGET` (1,000,000) trigger a skip — the replay cost is not justified for what is effectively a debug aid |

`[]` (empty list) means the detector ran and observed no firings — a
valid outcome at low `outlier_rate` and small cell counts. Distinct
from `null` (skipped).

**Use case** — score an anomaly-detection model. Each outlier
injection is ground truth: the cell got an outlier multiplier from
`apply_noise`, so a detector that fails to flag it has missed a known
positive. An empty list means clean data with no anomalies to find.

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
