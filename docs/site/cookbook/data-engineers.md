# Cookbook — for data engineers

> plotsim as a fixture generator for ETL pipelines, dbt projects, and
> warehouse loaders. Configurable corruption, deterministic output,
> realistic FK / PK / dimensional structure.

---

## Why plotsim for DE work

Data engineering tests usually live in one of three uncomfortable spots:

1. **Hand-rolled fixtures** — three rows per table, drift away from
   production over time, miss every interesting edge case.
2. **Production snapshots** — privacy headaches, slow to refresh,
   expensive to store.
3. **Random data tools** — Faker fills cells but the cells don't agree
   with each other; FK / PK relationships have to be wired up by hand.

plotsim sits in between: realistic *shape* and *structure* without real
data. Same config + same seed → byte-identical output, so your pipeline
tests reproduce. Configurable quality injection lets you assert dirty-
data handling explicitly.

---

## What to know

The data-engineering use-case notebook has the runnable, end-to-end
walkthrough:

[**`de_use_cases.ipynb`**](https://github.com/mohossam-ae/plotsim/blob/main/docs/tutorial-notebooks/de_use_cases.ipynb)

It covers:

- Generating a multi-table star schema with auto-wired FKs
- Injecting controlled data-quality issues for pipeline testing
- Round-tripping through Parquet for warehouse-loader rehearsal
- The deterministic seed contract — pinning a config for fixture stability

---

## Common patterns

### Fixture generation for dbt / ETL tests

```python
from plotsim import create_from_yaml, generate_tables, write_tables

cfg = create_from_yaml("tests/fixtures/saas_clean.yaml")
tables = generate_tables(cfg)
write_tables(tables, cfg, output_dir="tests/fixtures/output")
```

Pin `seed:` in the YAML and the fixture is byte-stable across CI runs.

### Dirty-data tests

```yaml
quality:
  - { table: fct_engagement, issue: null_injection,  rate: 0.05, column: engagement }
  - { table: fct_engagement, issue: duplicate_rows, rate: 0.02 }
  - { table: dim_customer,   issue: type_mismatch,  rate: 0.01, column: industry }
```

The corrupted rows are recorded in `manifest.quality_injections` —
your test can recover the clean values for assertion comparisons
without re-running generation.

### Warehouse-loader rehearsal

Switch output format to Parquet:

```yaml
output_format: parquet
```

Files land as `.parquet` instead of `.csv` — typed columns, Snappy
compression, ready for DuckDB / Snowflake / BigQuery loader testing.

---

## See also

- [Data quality](https://github.com/mohossam-ae/plotsim/blob/main/docs/tutorial-notebooks/data_quality.ipynb) —
  every issue type with examples
- [Pipeline testing](https://github.com/mohossam-ae/plotsim/blob/main/docs/tutorial-notebooks/pipeline_testing.ipynb) —
  deeper end-to-end recipe
- [Schema guide](../user-guide/schema-guide.md) — designing dim / fact / event tables
- [Output formats](../user-guide/output-formats.md) — CSV vs Parquet
- [Config field reference §quality](../config-reference.md) — every quality issue field
