# Denormalization mode

Star schemas are how analytics warehouses store data. Wide spreadsheets
are how analysts (and a lot of business stakeholders) want to *read*
data. Denormalization mode produces the wide-spreadsheet view alongside
the normalized output — same data, same row count, every dim column
joined onto each fact.

This is also the canonical setup for a normalization exercise: hand a
student the wide table, ask them to decompose it into 3NF. The
ground-truth normalized form is sitting right next to the wide table,
so the answer key is built in.

## How it works

When you set `output.denormalized: true`, plotsim writes the normal
star-schema output (`dim_*.csv`, `fct_*.csv`, `evt_*.csv`) and **also**
writes one wide companion per fact table:

```text
out/
  dim_company.csv              ← unchanged star-schema dims
  dim_plan.csv
  dim_date.csv
  fct_revenue.csv              ← unchanged star-schema facts
  fct_revenue_wide.csv         ← NEW: fact + every FK'd dim joined
  fct_engagement.csv
  fct_engagement_wide.csv      ← NEW
  ...
```

Each `*_wide` file has:

* **Every column from the original fact**, unchanged (PK, FKs, metric
  values, `_op` / `_inserted_at` / `_updated_at` if CDC is on).
* **Every column from every dim the fact FKs to**, prefixed with the
  dim table name and a double underscore. So `dim_company.company_name`
  becomes `dim_company__company_name`, `dim_plan.plan_tier` becomes
  `dim_plan__plan_tier`, `dim_date.period_label` becomes
  `dim_date__period_label`. The prefix prevents collisions when two
  dims share a column name like `name` or `created_at`.
* **One row per original fact row** — the join is a left-join, so
  fact-row count is preserved exactly.

The dim's join-key column is dropped from the wide output (it
duplicates the fact's FK column — no point carrying it twice).

## SCD Type 2 dims

When a dim has SCD Type 2 versioning enabled, it has multiple rows per
entity (one per historical version). The wide output filters that dim
to **current state only** (`is_current == True`) before joining, so
each fact row picks up the dim's latest attributes — not historical
ones.

The four SCD2 audit columns (`dim_row_id`, `valid_from`, `valid_to`,
`is_current`) are excluded from the wide output. They're internals of
the dim's history layer; the wide table is the as-of-now view, so they
don't add value here. If you need historical joins, work directly with
the normalized tables — `dim_row_id` resolution is exposed there.

## Configuration

Add `denormalized: true` to the `output` block:

```yaml
output:
  format: "csv"          # or "parquet"
  directory: "out/saas"
  denormalized: true     # ← opt-in
```

Or via the builder surface:

```python
from plotsim import create

config = create(
    about="B2B SaaS",
    unit="company",
    window=("2023-01", "2024-12", "monthly"),
    metrics=[...],
    segments=[...],
    output={"format": "csv", "directory": "out", "denormalized": True},
)
```

The flag defaults to `false`, so existing configs produce
byte-identical output until you opt in.

## Use cases

### Normalization exercise (1NF → 3NF)

Hand a student `fct_revenue_wide.csv`. Have them identify
the functional dependencies, separate transitive dependencies into
their own tables, and rebuild the star schema. Compare against the
already-normalized `dim_company.csv` / `dim_plan.csv` / `fct_revenue.csv`
that plotsim shipped alongside.

### BI report drafts

Many BI platforms perform better when fed wide tables (Tableau and
Power BI both materialize joins under the hood; doing the join once
upstream is faster than re-joining at every report load). Dropping
`fct_revenue_wide.csv` straight into a workbook gets you to a chart in
seconds.

### Spreadsheet-shaped analysis

Stakeholders who live in Excel often want a single rectangular file
with every attribute already attached. The `_wide` file is exactly
that — no joins required, no SQL knowledge required.

## Scope and limitations

* **Facts only.** Events and bridge tables are not denormalized in V1.
  An event's row count depends on the trajectory and would multiply
  out into the wide view in confusing ways; bridges are M:M
  associations whose denormalized form is rarely useful.
* **One join hop.** Facts join to their direct FK targets, not to the
  dims-of-dims. Snowflake schemas are flattened only one level.
* **Current state for SCD2.** As above — historical joins are out of
  scope; use the normalized tables for those.
* **No cross-fact joins.** Each fact denormalizes independently. Joining
  two facts together would multiply rows in ways that almost never
  match what an analyst actually wants; if you need that, write the
  SQL.
