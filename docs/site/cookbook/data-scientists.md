# Cookbook — for data scientists

> plotsim for ML training data, feature-engineering practice, and
> evaluation under known-truth conditions. Trajectory positions and
> archetype labels are exposed as ground truth via the manifest.

---

## Why plotsim for DS work

Most synthetic data tools generate columns that don't tell a story —
revenue, engagement, and churn move independently, and any model
trained on them learns noise. Real data has *coherence*: the same
underlying behavioral state shapes every metric on the row.

plotsim builds that coherence in. Every metric value at every
`(entity, period)` cell is derived from the same trajectory position.
The trajectory itself is exposed as ground truth — so you can train a
classifier on the noisy fact tables and *score it against the answer
key*.

---

## What to know

The data-science use-case notebook has the runnable, end-to-end
walkthrough:

[**`ds_use_cases.ipynb`**](https://github.com/mohossam-ae/plotsim/blob/main/docs/tutorial-notebooks/ds_use_cases.ipynb)

It covers:

- Building a config with explicit archetype labels (the latent class
  variable a downstream model learns)
- Generating per-entity flat features for tabular ML
- Setting up a temporal holdout split for time-series evaluation
- Reading the manifest to get trajectory positions for ground-truth
  comparison

---

## Common patterns

### Training-ready feature table

```yaml
entity_features: true
```

Generates a flat one-row-per-entity DataFrame:
`<metric>_mean`, `<metric>_std`, `<metric>_slope`, `<metric>_first`,
`<metric>_last`, `<metric>_peak_period`, plus `archetype` and
`final_trajectory_position` ground-truth labels.

```python
import pandas as pd
features = pd.read_csv("output/_entity_features.csv")
X = features.drop(columns=["archetype", "final_trajectory_position", "customer_id"])
y = features["archetype"]
```

### Temporal train/holdout split

```yaml
holdout:
  target: mrr
  periods: 3
  min_training_periods: 9
```

Every per-entity-per-period fact table writes two extra files:
`<fact>_train.<ext>` and `<fact>_holdout.<ext>`. The unsplit fact is
also written. When `entity_features: true` is also set, aggregation
restricts to the training window and the target metric's columns are
dropped to prevent label leakage.

### Reading ground truth from the manifest

```python
import json
from pathlib import Path

manifest = json.loads(Path("output/manifest.json").read_text())

# entity → archetype label (the latent class)
labels = {a["entity"]: a["archetype"] for a in manifest["archetype_assignments"]}

# trajectory position at every period for sampled entities
positions = {
    (s["entity"], s["period_index"]): s["position"]
    for s in manifest["trajectory_samples"]
}
```

Train a classifier on the fact-table aggregates, predict on
`features_test`, compare against `labels` for accuracy / F1 / confusion
matrix. plotsim is the only synthetic data tool that gives you that
answer key for free.

---

## See also

- [ML readiness](https://github.com/mohossam-ae/plotsim/blob/main/docs/tutorial-notebooks/ml_readiness.ipynb) —
  end-to-end ML pipeline with plotsim fixtures
- [Designing archetypes](https://github.com/mohossam-ae/plotsim/blob/main/docs/tutorial-notebooks/designing_archetypes.ipynb) —
  the archetype DSL in depth
- [Manifest reference](../manifest-reference.md) — every ground-truth
  field
- [Config field reference §holdout / §entity_features](../config-reference.md) —
  the ML-aware config blocks
- [API reference §build_entity_features / §trace_metric_cell](../api-reference.md) —
  programmatic access to features and per-cell pipeline traces
