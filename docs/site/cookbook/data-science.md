# ML training data & evaluation

> plotsim for ML training data, feature-engineering practice, and
> evaluation under known-truth conditions. Trajectory positions and
> archetype labels are exposed as ground truth via the manifest.

---

## Why plotsim for ML work

Most synthetic data tools generate columns that don't tell a story —
revenue, engagement, and churn move independently, and any model
trained on them learns noise. Real data has *coherence*: the same
underlying behavioral state shapes every metric on the row.

plotsim builds that coherence in. Every metric value at every
`(entity, period)` cell is derived from the same trajectory position.
The trajectory itself is exposed as ground truth — so you can train a
classifier on the noisy fact tables and *score it against the answer
key*.

The companion notebooks are
[**ds_use_cases.ipynb**](https://github.com/mohossam01/plotsim/blob/main/docs/site/tutorial-notebooks/ds_use_cases.ipynb)
(end-to-end ML workflows) and
[**ml_readiness.ipynb**](https://github.com/mohossam01/plotsim/blob/main/docs/site/tutorial-notebooks/ml_readiness.ipynb)
(features, holdout, sklearn).

---

## Quick start — two paths to the same dataset

The bundled `saas` template is the fastest way to get a coherent,
multi-metric dataset with archetype ground truth.

=== "CLI + YAML"

    ```bash
    plotsim template saas -o saas_fixture.yaml
    plotsim run saas_fixture.yaml -o ./data
    ```

=== "Python API"

    ```python
    from plotsim import create_from_yaml, generate_tables, write_tables

    cfg = create_from_yaml("saas_fixture.yaml")
    tables = generate_tables(cfg)
    write_tables(tables, cfg, output_dir="./data")
    ```

The [`saas_template.py`](https://github.com/mohossam01/plotsim/blob/main/plotsim/configs/templates/saas_template.py)
companion (paired with `saas.yaml` in the same directory) shows the
same SaaS template authored as a `create(**kwargs)` call — every YAML
field maps 1-1 to a Python keyword.

---

## Training-ready feature table

Set `entity_features: true` and a flat one-row-per-entity DataFrame
is written alongside the fact tables. Six aggregate columns per
metric, plus archetype + final-trajectory ground truth.

=== "YAML"

    ```yaml
    entity_features: true
    ```

=== "Python"

    ```python
    cfg = create(
        # ... about / unit / window / metrics / segments ...
        entity_features=True,                # or {"metrics": [...], "include_labels": False}
    )
    ```

```python
import pandas as pd
features = pd.read_csv("data/_entity_features.csv")
# Columns: <metric>_mean, <metric>_std, <metric>_slope,
#          <metric>_first, <metric>_last, <metric>_peak_period,
#          archetype, final_trajectory_position

X = features.drop(columns=["archetype", "final_trajectory_position", "company_id"])
y = features["archetype"]
```

The detailed shape (`metrics:` filter, `include_labels: false` for
unsupervised pipelines) is documented in
[`config-reference.md` §entity_features](../config-reference.md#entity_features).

---

## Temporal train/holdout split

Reserve trailing periods for evaluation. Every per-entity-per-period
fact table writes `<fact>_train.<ext>` and `<fact>_holdout.<ext>`
companions. Aggregations on `entity_features` restrict to the
training window automatically — no leakage.

=== "YAML"

    ```yaml
    holdout:
      target: mrr
      periods: 3
      min_training_periods: 9
    ```

=== "Python"

    ```python
    cfg = create(
        # ...
        holdout={"target": "mrr", "periods": 3, "min_training_periods": 9},
    )
    ```

The target metric's six aggregate columns are dropped from
`_entity_features.csv` to prevent label leakage. Cutoff period
index lands on `manifest.holdout.cutoff_period_index` for
downstream code that needs the boundary.

---

## Lagged features via `follows` / `delay`

Cross-metric causal lag is part of the metric declaration — no
separate post-processing step. `support_tickets` trails
`engagement` by 2 periods:

=== "YAML"

    ```yaml
    metrics:
      - { name: engagement, type: score, polarity: positive }
      - name: support_tickets
        type: count
        polarity: negative
        follows: engagement
        delay: 2
    ```

=== "Python"

    ```python
    metrics = [
        {"name": "engagement", "type": "score", "polarity": "positive"},
        {"name": "support_tickets", "type": "count", "polarity": "negative",
         "follows": "engagement", "delay": 2},
    ]
    ```

A `follows` graph cannot loop back on itself — the engine validates
the chain at config load. See
[Causal lag](../user-guide/metrics-and-connections.md#causal-lag-follows-delay).

---

## Custom correlation magnitudes

Calibrate a coefficient to whatever your domain measurement
suggests rather than picking from the nine-word vocabulary. Three
shorthand forms:

=== "YAML"

    ```yaml
    connections:
      - "engagement 0.42 retention"          # 3-token string with numeric middle
      - ["mrr", -0.31, "support_tickets"]    # tuple
      - {metric_a: nps, coefficient: 0.18, metric_b: feature_adoption}
    ```

=== "Python"

    ```python
    connections=[
        ("engagement", 0.42, "retention"),
        ("mrr", -0.31, "support_tickets"),
        {"metric_a": "nps", "coefficient": 0.18, "metric_b": "feature_adoption"},
    ]
    ```

Useful when a real dataset gave you a measured correlation (`r=0.42`)
that doesn't match any of `mirrors`/`driven_by`/etc. Coefficients are
clamped to `[-1, 1]`; the engine projects non-PSD matrices to the
nearest valid one and records the adjustment in the manifest.

---

## Time-to-event metrics — Weibull

For survival-style metrics (session duration, days-to-renewal,
contract length, p99 latency), pin the Weibull distribution
explicitly on the metric. The shape parameter controls the tail;
trajectory still scales the realised value, so a `growth`
archetype with `weibull` produces *lengthening* sessions over
time, not just larger ones.

=== "Python (builder)"

    ```python
    create(metrics=[{
        "name": "session_duration_days",
        "type": "amount",
        "polarity": "positive",
        "range": [1, 365],
        "distribution": "weibull",
        "distribution_params": {"shape": 1.5},  # >1 → lengthening tail; <1 → aging out
    }])
    ```

=== "YAML (engine-direct)"

    ```yaml
    # load with plotsim.load_config when you're past the builder surface
    metrics:
      - name: session_duration_days
        label: Session duration (days)
        distribution: weibull
        params: { shape: 1.5 }
        polarity: positive
        value_range: { min: 1.0, max: 365.0 }
    ```

All six builder distribution families (`lognorm`, `gamma`,
`weibull`, `beta`, `normal`, `poisson`) are pinnable the same way
via `MetricInput.distribution` + `distribution_params`. The
`tests/configs/latency_skew.yaml` worked example exercises all six
on a single config. Full mechanics:
[`metrics-and-connections.md` §pinning the distribution explicitly](../user-guide/metrics-and-connections.md#pinning-the-distribution-explicitly).

---

## SCD Type 2 dimensions for time-aware joins

Slowly-changing dimensions are first-class — declare a `scd`
column on a per-entity dim and the engine emits one row per band
crossing. Each fact row joins to the *active version* via an
auto-appended `dim_row_id` column.

=== "YAML"

    ```yaml
    dimensions:
      - name: dim_company
        per: unit
        columns:
          - { name: company_id, type: id }
          - name: plan_tier
            type: scd
            tracks: mrr
            tiers: [starter, growth, enterprise]
            at: [0.4, 0.7]
    ```

=== "Python"

    ```python
    dimensions = [{
        "name": "dim_company", "per": "unit", "columns": [
            {"name": "company_id", "type": "id"},
            {"name": "plan_tier", "type": "scd",
             "tracks": "mrr", "tiers": ["starter", "growth", "enterprise"],
             "at": [0.4, 0.7]},
        ],
    }]
    ```

`manifest.scd_events` records every transition with old/new label,
old/new `dim_row_id`, trigger period, and trigger position —
perfect ground truth for tenure-aware feature engineering. See
[Schema guide §SCD2](../user-guide/schema-guide.md#scd-type-2-columns).

---

## Bridge tables for relationship features

Many-to-many associations between two dimensions land as a bridge
table. `manifest.bridge_associations` records every
`(left_entity, right_entities)` tuple — the answer key for
relationship-graph features.

=== "YAML"

    ```yaml
    bridges:
      - name: customer_subscription
        left: dim_company
        right: dim_plan
        cardinality: [1, 3]
        driver: mrr
        columns:
          - { name: weight, type: metric.mrr }
    ```

=== "Python"

    ```python
    bridges = [{
        "name": "customer_subscription",
        "left": "dim_company", "right": "dim_plan",
        "cardinality": [1, 3], "driver": "mrr",
        "columns": [{"name": "weight", "type": "metric.mrr"}],
    }]
    ```

`driver: mrr` biases sampling toward the entity's trajectory
position — high-MRR companies hold more subscriptions. See
[`bridges_and_advanced.ipynb`](https://github.com/mohossam01/plotsim/blob/main/docs/site/tutorial-notebooks/bridges_and_advanced.ipynb).

---

## Seasonality + per-metric sensitivity

Calendar-month effects layer on top of the trajectory. Each metric
declares its own `seasonal_sensitivity` — `0.0` immune, `1.0`
default, `-0.5` halve and invert.

=== "YAML"

    ```yaml
    seasonality:
      - { months: [11, 12], strength: 0.30 }    # Q4 lift
      - { months: [6, 7, 8], strength: -0.10 }  # summer dip

    metrics:
      - { name: mrr,            type: amount, polarity: positive,
          range: [10, 5000], seasonal_sensitivity: 1.0 }
      - { name: support_tickets, type: count,  polarity: negative,
          seasonal_sensitivity: 0.0 }            # immune
    ```

=== "Python"

    ```python
    cfg = create(
        # ...
        seasonality=[
            {"months": [11, 12], "strength": 0.30},
            {"months": [6, 7, 8], "strength": -0.10},
        ],
        metrics=[
            {"name": "mrr", "type": "amount", "polarity": "positive",
             "range": [10, 5000], "seasonal_sensitivity": 1.0},
            {"name": "support_tickets", "type": "count", "polarity": "negative",
             "seasonal_sensitivity": 0.0},
        ],
    )
    ```

Per-segment sensitivity is also available — `seasonal_sensitivity:
1.5` on a B2C cohort, `0.0` on a B2B cohort. See
[Seasonality](../user-guide/seasonality.md) for the full
multiplication formula.

---

## Reading ground truth from the manifest

The manifest is the ML answer key — every entity's archetype,
every event firing, every band transition, every adjustment the
engine made (Higham PSD projection, correlation compensation,
copula bypass fallback).

```python
import json
from pathlib import Path

mf = json.loads(Path("./data/manifest.json").read_text(encoding="utf-8"))

# Entity → archetype label
labels = {a["entity"]: a["archetype"] for a in mf["archetype_assignments"]}

# Trajectory position at every period for sampled entities
positions = {
    (s["entity"], s["period_index"]): s["position"]
    for s in mf["trajectory_samples"]
}
```

Train a classifier on the fact-table aggregates, predict on the
holdout entities, compare against `labels` for accuracy / F1 /
confusion matrix. plotsim is the only synthetic data tool that
gives you that answer key for free.

See [Manifest reference](../manifest-reference.md) for every
section.

---

## Trace one weird cell — `trace_metric_cell`

When a particular `(entity, period, metric)` value looks
surprising, `trace_metric_cell` reconstructs every step of the
pipeline that produced it: trajectory position → distribution
center → correlated draw → noise → clamp.

```python
from plotsim.inspect import trace_metric_cell

result = trace_metric_cell(
    cfg, seed=cfg.seed,
    entity_name="promising_client_0001",
    period_index=8,
    metric_name="mrr",
)
print(result.trajectory_position, result.distribution_center,
      result.correlated_draw, result.noised_value, result.realized_cell)
```

Useful for debugging unexpected outliers or validating that a
custom archetype curve actually produced the shape you intended.
See [`trace_metric_cell`](../api-reference.md#trace_metric_cell).

---

## A complete ML-ready config

The bundled `saas` template — extended with `entity_features`,
`holdout`, and `quality: []` (entity_features and holdout both
require quality empty) — is ML-ready out of the box:

```bash
plotsim template saas -o saas_ml.yaml
# Edit saas_ml.yaml: append entity_features: true and a holdout block
plotsim run saas_ml.yaml -o ./data --validate
```

Then load `_entity_features.csv` for tabular ML, the per-fact
`_train` / `_holdout` files for time-series ML, and
`manifest.json` for ground truth. ml_readiness.ipynb walks this
recipe end-to-end against a sklearn classifier and a per-account
linear-regression forecast.

---

## See also

- [ml_readiness.ipynb](https://github.com/mohossam01/plotsim/blob/main/docs/site/tutorial-notebooks/ml_readiness.ipynb) —
  end-to-end ML pipeline with plotsim fixtures
- [designing_archetypes.ipynb](https://github.com/mohossam01/plotsim/blob/main/docs/site/tutorial-notebooks/designing_archetypes.ipynb) —
  the archetype DSL in depth
- [seasonality_and_correlations.ipynb](https://github.com/mohossam01/plotsim/blob/main/docs/site/tutorial-notebooks/seasonality_and_correlations.ipynb) —
  full seasonality + connection-grammar walk-through
- [Manifest reference](../manifest-reference.md) — every ground-truth field
- [Config field reference §holdout / §entity_features](../config-reference.md#holdout) —
  the ML-aware config blocks
- [Engine-direct fields](../config-reference.md#engine-direct-fields) —
  weibull distribution, `cross_dim_fks`, `inflection_month`
- [API reference §build_entity_features / §trace_metric_cell](../api-reference.md) —
  programmatic access to features and per-cell pipeline traces
