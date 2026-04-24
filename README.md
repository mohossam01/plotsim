# plotsim

**A Python synthetic data generator for realistic multi-table relational
datasets — with no real data required.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/mohossam-ae/plotsim/blob/main/LICENSE)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)](#)

plotsim is a synthetic data generator that produces multi-table CSV files
with foreign keys, temporal patterns, and cross-table consistency. Unlike
field-level fakers, every metric for an entity (engagement, revenue, churn)
moves together over time because they derive from the same behavioral
trajectory.

Use it to generate test data for dbt projects, analytics portfolios,
dashboard prototypes, training courses, or any scenario where you need
realistic relational data but have no access to production systems.

```
pip install plotsim
```

## Usage

### Generate from a template

```bash
plotsim template saas -o config.yaml
plotsim run config.yaml -o ./output --validate
```

### Generate from Python

```python
from plotsim import load_config, generate_tables, write_tables

config = load_config("config.yaml")
tables = generate_tables(config)
write_tables(tables, config)
```

Same config and seed produces identical output every run.

## When to use plotsim

- Building an **analytics portfolio** and need realistic data to showcase
  dbt models, dashboards, or SQL analysis
- Teaching a **data engineering course** and need students to work with
  multi-table schemas that behave like production data
- Prototyping a **dashboard or report** before production data is available
- Writing **integration tests** for a data pipeline that expects relational
  input with referential integrity
- Practicing **dbt, SQL, or data modeling** with data that has temporal
  patterns, not just random values

## What it generates

A single config produces a complete relational schema:

```
output/
├── dim_date.csv                # date spine
├── dim_company.csv             # entity attributes
├── dim_user.csv                # sub-entity attributes
├── dim_plan.csv                # reference lookup
├── fct_engagement.csv          # entity × period metrics
├── fct_revenue.csv             # entity × period metrics
├── fct_support_tickets.csv     # entity × period metrics
├── evt_login.csv               # behavioral events
├── evt_churn.csv               # threshold-triggered events
├── config.yaml                 # config that produced this output
└── validation_report.txt       # integrity checks
```

All foreign keys resolve. Event tables derive from fact values, not from
independent random generation. If an entity's engagement declines, its
login events decrease and churn events fire — across separate CSV files.

## Templates

Five domain configs ship with the package:

| Template | Domain | Entities | Tables |
| --- | --- | --- | --- |
| `saas` | B2B SaaS | accounts with users | 10 |
| `hr` | HR department | employees in departments | 7 |
| `ecommerce` | E-commerce | customer segments | 8 |
| `education` | University | student cohorts | 7 |
| `healthcare` | Clinic | patient groups | 8 |

```bash
plotsim list-templates          # see all available
plotsim template hr -o hr.yaml  # export one to edit
```

## Custom domains

The config file defines everything: entity types, metrics, behavioral
archetypes, table schemas, correlations, and noise levels. Copy any
template and modify it, or generate one with any LLM:

> "Change this SaaS config to model a food delivery service with
> restaurants, orders, delivery times, and customer ratings."

Or generate from scratch by feeding the schema to the LLM:

```python
import json
from plotsim.config import PlotsimConfig

schema = json.dumps(PlotsimConfig.model_json_schema(), indent=2)
# Paste this schema into any LLM and describe the domain you need
```

Validate before generating:

```bash
plotsim validate my_config.yaml
plotsim run my_config.yaml -o ./output
```

## How it works

Each entity is assigned an archetype — a trajectory curve composed of
segments like sigmoid, exponential decay, step, plateau, or oscillation.

At each time step, the engine reads the entity's trajectory position
(a value between 0 and 1) and derives every metric from it:

- **Positive polarity** metrics (engagement, revenue) rise when the
  trajectory rises.
- **Negative polarity** metrics (churn risk, support tickets) rise when
  the trajectory falls.

Distributions (lognormal, gamma, poisson, beta, normal, weibull) shape
the raw values. Correlated noise is applied via Cholesky decomposition
on the configured correlation matrix. Causal lag lets one metric trail
another by N periods.

Dimension tables are generated first (dates, entities, reference lookups),
then fact tables (trajectory-driven metrics per period), then event tables
(derived from completed fact values, never from raw trajectories).

## Config overview

A plotsim config has these sections:

- **domain** — name and entity label
- **time_window** — start, end, granularity (monthly / weekly / daily)
- **seed** — integer controlling all randomness
- **metrics** — name, distribution, polarity, optional causal lag
- **archetypes** — named trajectory shapes built from curve segments
- **entities** — instances assigned to archetypes
- **tables** — dim / fact / event schemas with typed columns
- **correlations** — optional metric-pair coefficients
- **noise** — gaussian sigma, outlier rate, missing data rate, temporal jitter
- **stages** — optional lifecycle sequence with enforceable ordering

Full schema with type annotations: [plotsim/config.py](https://github.com/mohossam-ae/plotsim/blob/main/plotsim/config.py)

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

## Validation

The engine runs these checks after generation:

- **FK integrity** — every foreign key resolves to a parent row
- **PK uniqueness** — no duplicate primary keys
- **Date spine** — no gaps in the date dimension
- **Causal coherence** — lagged metrics inflect after their drivers
- **Null policy** — no unexpected nulls outside configured missing rates
- **Correlation PSD** — correlation matrix is positive semi-definite

```bash
plotsim run config.yaml --validate
```

The validation report is written alongside the CSVs as `validation_report.txt`.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for dev setup, test commands,
and how to add templates or curve types.

## License

Apache-2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).