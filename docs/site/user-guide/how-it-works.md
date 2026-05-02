# How plotsim works

> The mental model in five minutes. No code internals — just the four
> ideas that explain every other page in this guide.

---

## The problem plotsim solves

Most synthetic data tools generate columns independently. Revenue is
random. Engagement is random. Churn is random. The numbers fill a
schema, but they don't behave like real data — because in real data,
these things *move together*.

plotsim generates relational test data with **shape**. Every entity
follows a behavioral trajectory, and every metric — across every table,
every foreign key, every time period — is derived from the same
trajectory position.

When engagement rises, revenue follows. When it declines, churn fires.

---

## The four ideas

### 1. You describe entities by their *shape over time*

plotsim doesn't ask for distributions or per-period values. It asks
"what does this kind of customer / employee / patient *do* over time?"
You answer with an **archetype** — a named behavioral pattern.

```yaml
segments:
  - { name: growers,  count: 30, archetype: growth }
  - { name: churners, count: 20, archetype: spike_then_crash }
  - { name: steady,   count: 50, archetype: flat }
```

Each archetype is a curve over `[0, 1]`: how the entity's underlying
behavior evolves from the start of the time window to the end.
[`archetypes.md`](./archetypes.md) covers the six base shapes and how
to compose them.

### 2. Every metric reads from the same trajectory position

For every entity at every period, plotsim computes one number — the
**trajectory position** in `[0, 1]` — from the archetype curve. Every
metric value at that `(entity, period)` cell is derived from that one
position.

This is the *trajectory-first invariant*:

> Engagement at 0.85, MRR at $4,200, churn risk at 0.05, support
> tickets at 1 — all four cells are derived from the same trajectory
> position. They aren't independent draws that happen to correlate.
> They literally come from the same underlying number.

When the trajectory dips, *everything* dips together (or *everything*
dips appropriately given each metric's polarity). When it rises,
everything rises. That's why plotsim data tells a story — because the
story is the trajectory, and the metrics are just reflections of it.

### 3. You configure the input layer; plotsim builds the star schema

You declare four things:

| You declare | plotsim builds |
|---|---|
| Domain (`about`, `unit`) | Entity dim table |
| Time window | `dim_date` spine |
| Metrics + segments | Behavior + cohorts |
| Schema (optional) | Dim, fact, event, bridge tables |

If you skip the schema declaration, plotsim emits a sensible default:
`dim_date`, `dim_<unit>`, one fact per metric, with FK columns wired
correctly. If you declare the schema, plotsim respects exactly what you
wrote.

[`schema-guide.md`](./schema-guide.md) covers when to use each path.

### 4. Output is deterministic and reproducible

Same `(config, seed)` → byte-identical files. Always. Set a seed once,
share the YAML, and anyone can regenerate exactly the dataset you saw.

The output bundle:

```
output/
├── dim_date.csv
├── dim_<unit>.csv
├── fct_<metric>.csv          # one per metric, by default
├── evt_<event>.csv           # if you declared events
├── config.yaml               # round-trippable copy of the input
├── validation_report.txt     # FK + PK + date-spine checks
└── manifest.json             # archetype labels, trajectory tape, etc.
```

Configure `output_format: parquet` and you get `.parquet` files instead.
[`output-formats.md`](./output-formats.md) covers the differences.

---

## The minimum viable config

```yaml
about: Subscription customers
unit: customer
window: ["2024-01", "2024-12", "monthly"]

metrics:
  - { name: engagement, type: score, polarity: positive }
  - { name: payments,   type: count, polarity: positive }

segments:
  - { name: active,   count: 50, archetype: growth }
  - { name: inactive, count: 30, archetype: decline }

seed: 42
```

Five required keys, two segments, two metrics. Run it:

```python
from plotsim import create_from_yaml, generate_tables, write_tables

cfg = create_from_yaml("my_config.yaml")
tables = generate_tables(cfg)
write_tables(tables, cfg)
```

You get a complete star schema with realistic temporal correlations
between engagement and payments, deterministic under the seed, ready
to load into pandas / polars / DuckDB.

---

## What to read next

- [Archetypes](./archetypes.md) — the DSL for shape-over-time
- [Metrics and connections](./metrics-and-connections.md) — what `type`,
  `polarity`, and `range` actually control, and how to declare
  cross-metric correlations
- [Schema guide](./schema-guide.md) — when to let plotsim auto-build
  your tables and when to override
- [Config field reference](../config-reference.md) — every input field
- [Tutorials](../tutorials.md) — runnable notebooks for every feature surface
