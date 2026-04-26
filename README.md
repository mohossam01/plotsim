# plotsim

**Generate multi-table synthetic datasets with behavioral trajectories, correlations, and causal lags. Config-driven. No real data required.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![Tests: 667 passed](https://img.shields.io/badge/tests-667%20passed-brightgreen)]()
[![PyPI](https://img.shields.io/pypi/v/plotsim)](https://pypi.org/project/plotsim/)

---

Most synthetic data tools generate columns independently. A customer's revenue is random. Their engagement is random. Their churn is random. The numbers fill a schema, but they don't behave like real data — because in real data, these things move together.

plotsim generates **multi-table relational datasets** where every metric tells the same story. Each simulated entity follows a behavioral trajectory — a mathematical curve that evolves over time. Revenue, engagement, churn risk, and support tickets all derive from the same trajectory position. When engagement rises, revenue follows. When it declines, churn events fire. Across every table, every foreign key, every time period.

The result is synthetic test data with **shape** — not just structure.

```
pip install plotsim
```

---

## Quick start

Generate a synthetic dataset from a bundled template:

```bash
plotsim template saas -o config.yaml
plotsim run config.yaml -o ./output --validate
```

Or from Python:

```python
from plotsim import load_config, generate_tables, write_tables

config = load_config("config.yaml")
tables = generate_tables(config)
write_tables(tables, config)
```

A single config produces a complete star schema:

```
output/
├── dim_date.csv                # complete date spine
├── dim_company.csv             # entity attributes
├── dim_user.csv                # sub-entity attributes
├── dim_plan.csv                # reference lookup
├── fct_engagement.csv          # entity × period metrics
├── fct_revenue.csv             # entity × period metrics
├── fct_support_tickets.csv     # entity × period metrics
├── evt_login.csv               # behavioral events
├── evt_churn.csv               # threshold-triggered events
└── validation_report.txt       # integrity checks
```

If a company's engagement trajectory declines, its login events decrease in `evt_login.csv` and churn events appear in `evt_churn.csv` — because both tables read from the same underlying trajectory, not from separate random generators.

---

## What makes plotsim different

**Trajectory-driven generation.** Each entity is assigned an archetype — a curve built from segments like sigmoid, exponential decay, plateau, or oscillation. At every time step, the engine reads the entity's position on that curve (a value between 0 and 1) and derives all metrics from it. Positive-polarity metrics rise when the trajectory rises. Negative-polarity metrics fall.

**Cross-metric correlation.** Configure the correlation strength between any pair of metrics. plotsim uses a Gaussian copula to inject the configured correlation, regardless of the underlying distribution pairing. Set engagement and revenue to covary at r=0.8, while support tickets moves inversely at r=-0.5 — and observe those values in the output within a measured tolerance (±0.10 for most distribution pairings; see [statistical fidelity](docs/statistical-fidelity.md) for the per-pair numbers).

**Causal lag with composable chains.** One metric can trail another by N periods, blended at a configurable weight against the metric's own trajectory. Configure engagement to drive revenue with a 3-period lag and the engine implements that shift faithfully at the metric-generator level; small lags (1–2) are also recoverable in output-level cross-correlation, while larger lags on smooth-archetype drivers require non-cross-correlation detection methods (see [statistical fidelity](docs/statistical-fidelity.md#causal-lag-fidelity)). Lags compose through chains: if A drives B with lag 2 and B drives C with lag 3, C reflects A's signal at lag 5.

**Star schema output.** plotsim generates dimensional models — date dimensions, entity dimensions, fact tables, event tables — with referential integrity enforced. Every foreign key resolves. Zero orphans.

**Deterministic output.** Same config + same seed = byte-identical CSVs. Always. The seeded numpy random state flows through every layer of generation.

**Config-time validation.** Pydantic V2 cross-validates your entire config before generation starts. Circular causal dependencies, non-positive-semi-definite correlation matrices, broken FK references, empty entity lists — all caught at parse time with clear error messages.

**Six distribution families.** Normal, lognormal, beta, poisson, gamma, and weibull — each configurable per metric. The engine samples from the distribution you specify and preserves marginal fidelity through the correlation injection.

---

## When to use plotsim

**Analytics portfolios** — showcase dbt models, dashboards, or SQL analysis with data that has real temporal patterns and cross-metric relationships, not random noise.

**Data engineering pipelines** — test with relational input where referential integrity holds, metrics correlate, and temporal ordering is causal. No production data access needed.

**Dashboard prototyping** — build with synthetic data that trends, correlates, and responds to filters the way real data would, before production access exists.

**Data science practice** — explore datasets with known ground truth. The correlations, trajectories, and causal lags are all configured — so you can verify whether your analysis recovers them.

**ML training data** — generate labeled datasets with controlled statistical properties for classification, regression, or causal discovery benchmarks.

**Teaching and courses** — give students multi-table schemas that behave like production data, where joins reveal actual business patterns.

---

## Templates

Five domain configs ship with the package:

| Template    | Domain       | Entities                  | Tables |
|-------------|--------------|---------------------------|--------|
| `saas`      | B2B SaaS     | accounts with users       | 10     |
| `hr`        | HR analytics | employees in departments  | 7      |
| `ecommerce` | E-commerce   | customer segments         | 8      |
| `education` | University   | student cohorts           | 7      |
| `healthcare`| Clinic       | patient groups            | 8      |

```bash
plotsim list-templates          # see all available
plotsim template hr -o hr.yaml  # export one to edit
```

Each template is a YAML file you can modify. Or describe what you need to any LLM:

> "Change this SaaS config to model a food delivery service with restaurants, orders, delivery times, and customer ratings."

---

## How it works

plotsim's generation pipeline:

1. **Config** — YAML defines entity types, metrics, distributions, archetypes, tables, correlations, causal lags, and noise levels. Pydantic V2 validates everything at load time.
2. **Trajectories** — each entity is assigned an archetype curve. The trajectory engine computes a position between 0 and 1 for every time period.
3. **Metrics** — processed in causal-dependency order (topologically sorted). Each metric's distribution is sampled at the trajectory-derived center. Causal lags propagate through the dependency chain.
4. **Correlations** — a Gaussian copula transforms independent samples through CDF → standard normal space, applies the Cholesky factor, and inverse-transforms back. Configured correlations are preserved regardless of distribution pairing.
5. **Noise** — Gaussian noise, outliers, and missing-completely-at-random nulls are injected after correlation, so they don't contaminate the configured statistical properties.
6. **Tables** — dimension, fact, and event tables are assembled with enforced referential integrity. Output is deterministic CSV.

---

## Config overview

A plotsim config has these sections:

- **domain** — name and entity label
- **time_window** — start, end, granularity (monthly / weekly / daily)
- **seed** — integer controlling all randomness
- **metrics** — name, distribution (normal, lognormal, beta, poisson, gamma, weibull), polarity, optional causal lag with configurable blend weight
- **archetypes** — named trajectory shapes built from curve segments (sigmoid, decay, step, plateau, oscillation, compound, linear)
- **entities** — groups assigned to archetype distributions
- **tables** — dim / fact / event schemas with typed columns and FK references
- **correlations** — metric-pair coefficients delivered via Gaussian copula
- **noise** — gaussian sigma, outlier rate, MCAR rate, temporal jitter
- **stages** — optional lifecycle sequence with enforceable ordering

Full schema with type annotations: [`plotsim/config.py`](plotsim/config.py)

---

## CLI reference

```
plotsim run <config>              Generate dataset from config
  -o, --output-dir <path>         Output directory (default: from config)
  -s, --seed <int>                Override seed
  -v, --validate                  Run validation after generation
  --strict                        Fail on validation warnings
  -q, --quiet                     Suppress output

plotsim validate <config>         Check config without generating
plotsim info <config>             Preview tables, rows, entities
plotsim list-templates            Show bundled templates
plotsim template <name>           Print template YAML to stdout
  -o, --output <path>             Write to file instead
```

---

## Post-generation validation

The engine runs these checks after generation:

- **FK integrity** — every foreign key resolves to a parent row (0 orphans across all templates)
- **PK uniqueness** — no duplicate primary keys
- **Date spine** — no gaps or duplicates in the date dimension
- **Causal coherence** — lagged metrics inflect after their drivers
- **Null policy** — no unexpected nulls outside configured MCAR rates
- **Correlation PSD** — correlation matrix is positive semi-definite (checked at config load)

```bash
plotsim run config.yaml --validate
```

For the empirical bounds these guarantees hold within — measured per-pair correlation tolerance, the recoverable-lag boundary, the trajectory-first cell-level envelope, and the determinism contract — see [`docs/statistical-fidelity.md`](docs/statistical-fidelity.md). The smoke test [`tests/test_fidelity_smoke.py`](tests/test_fidelity_smoke.py) re-checks the headline tolerances on every CI run.

---

## Ecosystem positioning

plotsim sits between tools like [Faker](https://github.com/joke2k/faker) / [plaitpy](https://github.com/plaitpy/plaitpy) (random values from templates) and [SDV](https://github.com/sdv-dev/SDV) (machine learning from real data).

Unlike Faker: plotsim produces multi-table relational datasets with cross-metric correlations, causal lags, and temporal trajectories — not independent random columns.

Unlike SDV: plotsim doesn't need real data. You specify the statistical properties you want in a YAML config, and the engine generates data matching that specification. No training, no privacy concerns, no seed data required.

---

## Generated data and PII

plotsim uses [Faker](https://github.com/joke2k/faker) for string-valued columns (names, companies, emails). Faker output is realistic-looking but not globally unique — a generated name can coincidentally match a real person. **Treat Faker output as synthetic, not anonymized.**

Mark a column with `pii_note: "<description>"` in your config to flag it as producing realistic-sounding data about people or organizations. plotsim threads the note through schema introspection so downstream consumers (data catalogs, governance tools, documentation generators) can identify those fields. It is metadata only and does not change generation behavior.

---

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for dev setup, test commands, and how to add templates or curve types.

## License

Apache-2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
