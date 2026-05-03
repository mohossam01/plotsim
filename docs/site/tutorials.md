# Tutorials

> Runnable notebooks covering each feature surface. Each notebook is
> self-contained — clone the repo, open in Jupyter, run top to bottom.

The notebooks live in
[`docs/tutorial-notebooks/`](https://github.com/mohossam01/plotsim/tree/main/docs/tutorial-notebooks)
and render best on GitHub or in a local Jupyter session.

---

## Start here

| Notebook | What it covers |
|---|---|
| [`getting_started.ipynb`](https://github.com/mohossam01/plotsim/blob/main/docs/tutorial-notebooks/getting_started.ipynb) | The happy-path on-ramp: install, write a builder config, generate, inspect, write |

---

## Feature surfaces

Each notebook picks one feature surface, builds a focused config that
exercises it, and shows what to look for in the output.

| Notebook | Feature |
|---|---|
| [`designing_archetypes.ipynb`](https://github.com/mohossam01/plotsim/blob/main/docs/tutorial-notebooks/designing_archetypes.ipynb) | Six base shapes; the `>` and `@` composition DSL |
| [`seasonality_and_correlations.ipynb`](https://github.com/mohossam01/plotsim/blob/main/docs/tutorial-notebooks/seasonality_and_correlations.ipynb) | Global seasonality, per-metric/segment sensitivity, declared correlations |
| [`schema_and_dimensions.ipynb`](https://github.com/mohossam01/plotsim/blob/main/docs/tutorial-notebooks/schema_and_dimensions.ipynb) | Auto-schema vs explicit schema; designing dims, facts, events |
| [`bridges_and_advanced.ipynb`](https://github.com/mohossam01/plotsim/blob/main/docs/tutorial-notebooks/bridges_and_advanced.ipynb) | Many-to-many bridges; SCD Type 2 versioned dims |
| [`data_quality.ipynb`](https://github.com/mohossam01/plotsim/blob/main/docs/tutorial-notebooks/data_quality.ipynb) | Quality injection — null, duplicate, type-mismatch, late-arrival |

---

## Workflows

End-to-end recipes for downstream consumers. See also
[Cookbook → Data engineers](./cookbook/data-engineering.md) and
[Cookbook → Data scientists](./cookbook/data-science.md) for the
opinionated cohort versions.

| Notebook | Recipe |
|---|---|
| [`pipeline_testing.ipynb`](https://github.com/mohossam01/plotsim/blob/main/docs/tutorial-notebooks/pipeline_testing.ipynb) | Use plotsim output as fixtures for ETL / dbt pipeline tests |
| [`ml_readiness.ipynb`](https://github.com/mohossam01/plotsim/blob/main/docs/tutorial-notebooks/ml_readiness.ipynb) | Holdout splits + entity-features for downstream ML training |

---

## What to read alongside

- [How it works](./user-guide/how-it-works.md) — the mental model the
  notebooks assume
- [Config field reference](./config-reference.md) — every field each
  notebook exercises
- [API reference](./api-reference.md) — function signatures the
  notebooks call
