# plotsim

**Generate realistic multi-table datasets from behavioral archetypes.
No real data needed. No ML training. No cloud dependency. Seed in, tables out.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)](#)

## The problem

Analytics engineers, dbt learners, and portfolio builders need realistic
multi-table datasets — not just a single CSV, but a fact/dim schema where
the stories across tables line up. Existing tools don't fit:

- **SDV, Gretel, MOSTLY AI, YData** need real data as training input.
- **Faker, Mimesis** produce field-level randomness with no temporal or
  cross-table coherence.
- **Jaffle Shop, dbt-fake** are static snapshots — no behavior over time.

The gap: nothing turns a plain YAML description of a domain into a
causally-consistent multi-table dataset with no source data required.

## How plotsim is different

One trajectory drives every metric for an entity. When engagement drops,
MRR drops, support tickets spike, churn risk rises, and churn events fire
— not because of a post-hoc correlation matrix, but because all four
metrics read from the same archetype-derived position at each time step.

That's the core invariant: **every value is traceable to a trajectory
position.** Correlations, noise, and distributions shape the numbers on
top, but the causal spine is set once per entity.

## Install

```
pip install plotsim
```

Python 3.10 or newer. Core dependencies are `numpy`, `scipy`, `pandas`,
`pyyaml`, `pydantic`, and `faker`.

## Quick start

First, pull a template out:

```
plotsim template saas -o config.yaml
```

Then generate from Python — three lines:

```python
from plotsim import load_config, generate_tables, write_tables

config = load_config("config.yaml")
tables = generate_tables(config)
write_tables(tables, config)
```

`generate_tables` seeds its RNG from `config.seed` when no generator is
passed; `write_tables` runs the validator internally when no report is
passed. Both can be supplied explicitly for tighter control — see
[examples/quickstart.py](examples/quickstart.py) for an end-to-end version
that branches on the validation report and prints a head of one table.

The engine is fully offline and deterministic — same config + same seed
produces byte-identical CSVs every time.

## CLI

```
# Pull a bundled template out and generate from it
plotsim template saas -o my_config.yaml
plotsim run my_config.yaml -o ./output --validate

# Preview what a config would generate (no writes)
plotsim info my_config.yaml

# Validate a config without running the engine
plotsim validate my_config.yaml

# List every bundled template
plotsim list-templates
```

Run `plotsim --help` or `plotsim <command> --help` for flag detail.

## Available templates

| Template     | Domain               | What you get                                           |
|--------------|----------------------|--------------------------------------------------------|
| `saas`       | B2B SaaS             | accounts, engagement, MRR, support tickets, churn      |
| `hr`         | HR department        | employees, performance, training, attrition            |
| `ecommerce`  | E-commerce           | customer segments, orders, returns, cart abandonment   |
| `education`  | University           | student cohorts, grades, engagement                    |
| `healthcare` | Clinic               | patient groups, visits, treatments, outcomes           |

Every template is a plain YAML file you can copy, edit, and re-run. The
engine is domain-agnostic — the templates are there for a fast start, not
as a closed list.

## Custom domains

The config file is the contract. Copy a template, edit the `domain`,
`metrics`, `archetypes`, and `tables` sections to your use case, then:

```
plotsim validate my_config.yaml
plotsim run my_config.yaml
```

If you don't feel like editing YAML by hand, paste a template into
ChatGPT/Claude and describe the transformation:

> "Change this SaaS config to model a food delivery service with
> restaurants, orders, delivery times, and customer ratings."

Then validate and run. The Groq/Llama auto-scaffolder in the original
design spec is deferred to V2; a clipboard and an LLM are enough for now.

## What gets generated

Running `plotsim run configs/sample_saas.yaml -o ./out` produces:

```
out/
├── dim_date.csv              # 24 rows — the date spine
├── dim_company.csv           # one row per cohort
├── dim_user.csv              # per-cohort user rows
├── dim_plan.csv              # static reference
├── fct_engagement.csv        # entity × period engagement + feature_adoption
├── fct_revenue.csv           # entity × period MRR
├── fct_support_tickets.csv   # entity × period ticket_count + churn_risk + nps
├── evt_login.csv             # login events, count proportional to engagement
├── evt_churn.csv             # churn events, fired when churn_risk stays high
├── config.yaml               # exact config that produced this run
└── validation_report.txt     # FK/PK/date_spine/causal/null checks
```

Fact columns carry a `stage` field if the config defines a lifecycle
sequence (`onboarding → active → at_risk → churned` for SaaS). The
validator enforces stage monotonicity when `enforce_order: true`.

## How it works

```
YAML config → Pydantic schema → PlotsimConfig (frozen)
                                      │
                      per-entity archetype curve segments
                                      ▼
                      trajectory engine → position arrays [0,1]
                                      │
                         polarity × distribution × correlation
                                      ▼
                      metric generator → per-metric series
                                      │
                            fact + event table builders
                                      ▼
                              dict[str, DataFrame]
                                      │
                                  write_tables
                                      ▼
                         CSV files + config.yaml + report
```

Every randomness path flows through a single seeded
`numpy.random.Generator`. Events consume completed fact values, never
raw trajectories — that firewall is enforced by the function signature
of `build_event_tables`. Correlations between metrics are applied via
Cholesky decomposition on the configured coefficient matrix; the
validator catches non-PSD matrices at config-load time.

## Simple vs expert configs

The same archetype can be described either way — the engine doesn't care:

| Control            | Simple label                     | Expert parameters                              |
|--------------------|----------------------------------|------------------------------------------------|
| Archetype          | "Grows fast then crashes"        | `sigmoid(0..0.55) → step(0.55..0.65) → plateau` |
| Metric relationship| "Tightly linked"                 | `correlation_coefficient: 0.72`                |
| Data messiness     | "Slightly messy"                 | `gaussian_sigma=0.03, mcar_rate=0.005`         |
| Turning point      | "Midway through"                 | `inflection_month: 14`                         |
| Value spread       | "Tight clustering"               | `lognorm(s=0.85, scale=1.2)`                   |

A UI layer (deferred to V2) swaps the labels on the same underlying
parameters. Same data underneath.

## Compared to alternatives

| Feature                           | plotsim | SDV     | Faker | dbt-fake |
|-----------------------------------|----------|---------|-------|----------|
| Needs real data                   | No       | Yes     | No    | No       |
| Multi-table with FKs              | Yes      | Yes     | No    | Limited  |
| Temporal behavioral patterns      | Yes      | No      | No    | No       |
| Causal chain enforcement          | Yes      | No      | No    | No       |
| Domain-agnostic                   | Yes      | Yes     | N/A   | No       |
| Deterministic (seeded)            | Yes      | Partial | Yes   | No       |

## Config schema reference

The full schema lives in [plotsim/config.py](plotsim/config.py) — it's
a Pydantic v2 model, every field is typed, every cross-reference is
validated at load time. The top-level sections are:

- `domain` — name and entity label
- `time_window` — start, end, granularity (monthly/weekly/daily)
- `seed` — integer, drives every random decision
- `metrics` — name, distribution, polarity, default curve, optional causal lag
- `archetypes` — curve segments composed of sigmoid, step, plateau, etc.
- `entities` — cohorts with an archetype and size
- `tables` — dim/fact/event schemas with column-level `source` strings
- `correlations` — optional metric-pair coefficients
- `noise` — gaussian, outlier, MCAR, temporal jitter
- `stages` — optional lifecycle sequence with enforceable ordering
- `output` — CSV directory

Start from a template (`plotsim template saas`) rather than from a
blank file — the sample configs exercise every supported section.

## Limitations (V1)

- Output format is CSV only.
- All randomness is deterministic given a seed; we don't simulate time-
  varying noise rates.
- The LLM auto-scaffolder (Groq + Llama) and the 4-page Streamlit UI
  from the design spec are V2 features.
- `per_subentity_per_period` facts aren't exercised by any bundled
  template yet; sub-entity dim tables work, but sub-entity facts are
  unvalidated against real configs.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the dev setup, test commands,
and the checklist for adding a new domain template or curve type.

## License

Apache-2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
