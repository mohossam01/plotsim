<p align="center">
  <img src="https://raw.githubusercontent.com/mohossam01/plotsim/main/docs/site/assets/brand/readme-banner.png" alt="plotsim — datasets that tell a story" width="100%">
</p>

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![Tests: 1673 passed](https://img.shields.io/badge/tests-1673%20passed-brightgreen)]()
[![PyPI](https://img.shields.io/pypi/v/plotsim)](https://pypi.org/project/plotsim/)
[![Docs](https://img.shields.io/badge/docs-mohossam01.github.io-blue)](https://mohossam01.github.io/plotsim/)

**Generate multi-table relational datasets where every metric tells the same story. Config-driven. No real data required.**

```
pip install plotsim
```

plotsim generates multi-table relational datasets from a behavioral
description. You define metrics, segments, and how entities behave over
time — the engine produces a star schema where every value traces back to
one trajectory. **shape**: every entity follows a behavioral trajectory, and every metric across every table reads from the same trajectory position. When engagement rises, revenue follows. When it declines, churn fires.

## Quick start

```python
from plotsim import create, generate_tables

cfg = create(
    about="Subscription customers",
    unit="customer",
    window=("2024-01", "2024-12", "monthly"),
    metrics=[
        {"name": "engagement", "type": "score", "polarity": "positive"},
        {"name": "payments",   "type": "count", "polarity": "positive"},
    ],
    segments=[
        {"name": "active",   "count": 50, "archetype": "growth"},
        {"name": "inactive", "count": 30, "archetype": "decline"},
    ],
)
tables = generate_tables(cfg)
for name, df in tables.items():
    print(f"{name}: {len(df)} rows")
# dim_date: 12 rows
# dim_customer: 80 rows
# fct_customer: 960 rows
```

## What you get

A complete dataset of CSV (or Parquet) files, ready to load into a warehouse, notebook, or BI tool. Same config plus same seed produces byte-identical output every time — see the [output guide](https://mohossam01.github.io/plotsim/user-guide/output-formats/) for details.



## Who is this for

**Educators and students** who need realistic datasets for SQL 
courses, data modeling workshops, analytics training, or portfolio 
projects — five domain templates ready to go, same seed produces 
the same data every time.

**Data engineers** who need test fixtures that behave like production 
data — with FK integrity, realistic distributions, and configurable 
corruption — without copying production or hand-rolling three-row 
CSVs.

**Data scientists** who need labeled training data with known ground 
truth — archetype labels, trajectory positions, and temporal holdout 
splits — to validate models before touching real data.

**Analytics engineers** who need a star schema to build dbt models, 
test transformations, or demonstrate a pipeline end-to-end without 
waiting for upstream data.

**BI and analytics teams** who need a populated star schema to 
build dashboards, test reports, or demo a new tool to stakeholders 
— dims, facts, events, and SCD versioning out of the box.

**Demo builders** who need a convincing dataset for a conference 
talk, a product walkthrough, or a proof of concept — correlated 
metrics that tell a realistic story, not random noise.



## How it works

Every entity in the dataset follows a **behavioral trajectory** — a 
curve shape like growth, decline, seasonal, or spike-then-crash. At 
each time period, the entity's position on that curve determines 
every metric value across every table. Revenue, engagement, churn 
risk, and support tickets all read from the same position, so they 
move together the way real business metrics do.

Metric relationships are enforced through a **Gaussian copula** — 
declare `engagement opposes churn_risk` and the engine delivers the 
configured correlation coefficient regardless of whether one metric 
is beta-distributed and the other is Poisson. Causal lags compose: 
if `A → B (lag 2) → C (lag 3)`, then C reflects A from 5 periods ago.

Output is **deterministic**. Every random draw flows through a single 
seeded `numpy.Generator`. Same config + same seed = byte-identical 
tables across machines, OS, and Python versions. The manifest records 
every generation decision — archetype assignments, trajectory 
positions, correlation adjustments, quality injections — so any 
cell value can be traced back to its origin.

Config-time **validation** catches problems before generation starts: 
circular causal chains, non-positive-definite correlation matrices, 
broken foreign key references, duplicate metric names, and 
SQL-unsafe identifiers all surface as parse errors with fix 
suggestions.


See the [docs site](https://mohossam01.github.io/plotsim/) for the full pipeline.

## Docs

[mohossam01.github.io/plotsim](https://mohossam01.github.io/plotsim/) — quickstart, user guide, tutorials, API reference, cookbooks.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for dev setup, test commands, and how to add templates.

## License

Apache-2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
