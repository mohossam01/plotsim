# plotsim

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![Tests: 1657 passed](https://img.shields.io/badge/tests-1657%20passed-brightgreen)]()
[![PyPI](https://img.shields.io/pypi/v/plotsim)](https://pypi.org/project/plotsim/)

**Generate multi-table relational datasets where every metric tells the same story. Config-driven. No real data required.**

```
pip install plotsim
```

---

Most synthetic data tools generate columns independently — a customer's revenue is random, their engagement is random, their churn is random. The numbers fill a schema, but they don't behave like real data. plotsim generates relational test data with **shape**: every entity follows a behavioral trajectory, and every metric across every table reads from the same trajectory position. When engagement rises, revenue follows. When it declines, churn fires.

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

`create()` accepts the same shape as the bundled YAML templates (`plotsim template <name>`). For a CLI-only flow, `plotsim run config.yaml -o ./output` does the same end-to-end.

## What you get

A complete star schema written as CSV (or Parquet, with the `[parquet]` extra): a date dimension, one entity dimension per `unit`, fact tables joining entity × period, and event tables for proportional or threshold-driven occurrences. Every foreign key resolves; the date spine has no gaps; same config + same seed = byte-identical output.

## Under the hood

- **Trajectory-first generation** — each entity is assigned a behavioral curve (`growth`, `decline`, `seasonal`, `spike_then_crash`, …). Every metric value at every period reads from that curve, so positive-polarity metrics rise as the trajectory rises and negative-polarity metrics fall.
- **Gaussian copula correlations** — declare `engagement opposes churn_risk` and the engine delivers the configured Pearson coefficient regardless of the underlying distribution pair, within a measured tolerance ([statistical fidelity](docs/statistical-fidelity.md)).
- **Causal lag with composable chains** — one metric can trail another by N periods; lags compose, so `A → B(lag=2) → C(lag=3)` produces a `C` that reads `A` from 5 periods ago.
- **Deterministic** — every random draw flows through a single seeded `numpy.Generator`. Cross-process reproduction is part of the contract.
- **Config-time validation** — Pydantic V2 cross-validates the entire input before generation: circular causal chains, non-PSD correlation matrices, broken FK references, and SQL-unsafe identifiers all surface as parse errors.

## Where plotsim sits

Unlike [Faker](https://github.com/joke2k/faker) (independent random columns, no relationships across tables) and unlike [SDV](https://github.com/sdv-dev/SDV) (machine learning trained on real data), plotsim takes a YAML or kwargs spec and emits a multi-table dataset with the configured statistical properties. No training, no privacy concerns, no seed data.

## Generated data and PII

plotsim uses [Faker](https://github.com/joke2k/faker) for string-valued columns (names, companies, emails). Faker output is realistic-looking but not globally unique — a generated name can coincidentally match a real person. Treat Faker output as synthetic, not anonymized. Mark a column with `pii_note: "<description>"` in your config to flag it for downstream catalogs and governance tools.

## Docs

- **[Getting started](docs/getting-started.md)** — install, run a template, build your first config
- **[Builder quickstart](docs/builder-quickstart.md)** — the `create()` / `create_from_yaml()` walkthrough
- **[Builder reference](docs/builder-reference.md)** — every keyword, every recipe
- **[Templates](docs/templates.md)** — six bundled domain configs you can copy and edit
- **[Statistical fidelity](docs/statistical-fidelity.md)** — measured correlation tolerances and the determinism contract

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for dev setup, test commands, and how to add templates.

## License

Apache-2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
