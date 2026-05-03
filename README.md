<p align="center">
  <img src="https://raw.githubusercontent.com/mohossam01/plotsim/main/docs/site/assets/brand/readme-banner.png" alt="plotsim — behavioral patterns, simulated" width="100%">
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

## Under the hood

- **Trajectory-driven** — every entity follows one behavioral curve, and every metric reads from the same position on that curve.
- **Correlated** — declared relationships between metrics hold to a measured tolerance, regardless of distribution shape.
- **Deterministic** — every random draw flows through one seeded generator; reproduction is part of the contract.

See the [docs site](https://mohossam01.github.io/plotsim/) for the full pipeline.

## Docs

[mohossam01.github.io/plotsim](https://mohossam01.github.io/plotsim/) — quickstart, user guide, tutorials, API reference, cookbooks.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for dev setup, test commands, and how to add templates.

## License

Apache-2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
