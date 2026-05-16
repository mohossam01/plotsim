# Pipeline fixtures & warehouse loading

> plotsim as a fixture generator for ETL pipelines, dbt projects, and
> warehouse loaders. Configurable corruption, deterministic output,
> realistic FK / PK / dimensional structure.

---

## Why plotsim for warehouse work

Warehouse-side tests usually live in one of three uncomfortable spots:

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

The companion notebook is the runnable, end-to-end walkthrough:
[**de_use_cases.ipynb**](https://github.com/mohossam01/plotsim/blob/main/docs/site/tutorial-notebooks/de_use_cases.ipynb).

---

## Quick start — two paths to the same dataset

The bundled `saas` template exists in two equivalent forms — pick
whichever fits your workflow.

=== "CLI + YAML"

    ```bash
    plotsim template saas -o saas_fixture.yaml
    plotsim run saas_fixture.yaml -o ./fixtures --validate --strict
    ```

    `--strict` aborts the write if any validation check fails.
    `--validate` prints the report after generation. The output dir
    contains every dim / fact / event table, plus
    `manifest.json` and `validation_report.txt`.

=== "Python API"

    ```python
    from plotsim import create_from_yaml, generate_tables, write_tables

    cfg = create_from_yaml("saas_fixture.yaml")
    tables = generate_tables(cfg)
    write_tables(tables, cfg, output_dir="./fixtures")
    ```

    Or skip the YAML round-trip entirely — the
    [`saas_template.py`](https://github.com/mohossam01/plotsim/blob/main/plotsim/configs/templates/saas_template.py)
    bundled with plotsim shows the same SaaS template authored as
    `create(**kwargs)` directly, paired with `saas.yaml` in the
    same directory.

Pin `seed:` in the YAML (or pass `seed=42` to `create`) and the fixture
is byte-stable across CI runs.

---

## Pin a deterministic fixture for CI

Check the YAML into `tests/fixtures/`. The fixture is the *config*,
not the generated CSVs. Any test can rebuild byte-identical tables
in O(seconds).

=== "CLI"

    ```bash
    plotsim run tests/fixtures/saas_clean.yaml -o tests/_tmp --quiet
    ```

=== "pytest"

    ```python
    import pytest
    import numpy as np
    from plotsim import create_from_yaml, generate_tables

    @pytest.fixture(scope="session")
    def saas_fixture():
        cfg = create_from_yaml("tests/fixtures/saas_clean.yaml")
        return generate_tables(cfg, np.random.default_rng(cfg.seed))

    def test_fact_grain(saas_fixture):
        fct = saas_fixture["fct_revenue"]
        assert not fct.duplicated(subset=["company_id", "date_key"]).any()
    ```

`generate_tables` returns a `dict[str, pandas.DataFrame]` keyed by
table name — exactly what most test assertions want.

---

## Inject dirty data — all six quality issue types

The `quality` block runs *after* generation, corrupting rows
post-hoc. The clean values are recorded in the manifest under
`quality_injections` so a test can recover ground truth without
re-running generation.

=== "YAML"

    ```yaml
    quality:
      - { table: fct_engagement,      issue: null_injection,  rate: 0.05, column: engagement_score }
      - { table: fct_engagement,      issue: duplicate_rows,  rate: 0.02 }
      - { table: dim_company,         issue: type_mismatch,   rate: 0.01, column: industry }
      - { table: evt_login,           issue: late_arrival,    rate: 0.03 }
      - { table: fct_support_tickets, issue: schema_drift,    rate: 0.05, column: ticket_count }
      - { table: fct_engagement,      issue: volume_anomaly,  rate: 1.0, mode: spike, period: 5 }
      - { table: fct_engagement,      issue: volume_anomaly,  rate: 0.5, mode: drop,  periods: [11, 17] }
    ```

=== "Python"

    ```python
    from plotsim import create, generate_tables, write_tables

    cfg = create(
        # ... about / unit / window / metrics / segments as in your template ...
        quality=[
            {"table": "fct_engagement", "issue": "null_injection",
             "rate": 0.05, "column": "engagement_score"},
            {"table": "fct_engagement", "issue": "duplicate_rows", "rate": 0.02},
            {"table": "dim_company",    "issue": "type_mismatch",
             "rate": 0.01, "column": "industry"},
            {"table": "evt_login",      "issue": "late_arrival", "rate": 0.03},
            {"table": "fct_support_tickets", "issue": "schema_drift",
             "rate": 0.05, "column": "ticket_count"},
            {"table": "fct_engagement", "issue": "volume_anomaly",
             "rate": 1.0, "mode": "spike", "period": 5},
            {"table": "fct_engagement", "issue": "volume_anomaly",
             "rate": 0.5, "mode": "drop", "periods": [11, 17]},
        ],
    )
    tables = generate_tables(cfg)
    write_tables(tables, cfg, output_dir="./fixtures_dirty")
    ```

`schema_drift` adds a `{col}_v2` companion column and nulls the
original on the affected rows. `late_arrival` adds an
`_arrival_period` column. `volume_anomaly` is row-level: `mode: spike`
appends duplicates of `floor(rate × N)` rows at the target period(s);
`mode: drop` removes them. PK / FK / `date_key` columns are skipped
automatically — quality never breaks referential integrity.

See the [data_quality.ipynb](https://github.com/mohossam01/plotsim/blob/main/docs/site/tutorial-notebooks/data_quality.ipynb)
notebook for assertions against the recovered clean values.

---

## Distributional noise (separate from quality)

`quality` corrupts the on-disk table after generation. **`noise`**
perturbs metric values *during* generation — the realised values
deviate from the trajectory-driven center, but FK / PK / grain
contracts are untouched. Four named presets cover the typical
fuzz-test ladder:

=== "YAML"

    ```yaml
    # Preset shorthand (alias 'messy' also accepted; see config-reference)
    noise: realistic

    # Or set the three dials explicitly
    noise:
      gaussian_sigma: 0.05
      outlier_rate: 0.02
      mcar_rate: 0.01
    ```

=== "Python"

    ```python
    cfg = create(
        # ... about / unit / window / metrics / segments ...
        noise="realistic",          # σ=0.05, outliers=2%, NaN=1%
    )
    ```

Use `perfectly_clean` (the default — same as omitting `noise`) for
golden-path tests, `realistic` or `dirty` for fuzz tests where the
pipeline has to handle imperfect data without breaking. See the full
preset table at
[`config-reference.md` §noise](../config-reference.md#noise).

---

## Warehouse-loader rehearsal — Parquet output

CSV is the default; flip one field for typed, compressed Parquet
that any DuckDB / Snowflake / BigQuery / Redshift loader will accept:

=== "YAML"

    ```yaml
    output:
      format: parquet
      directory: ./fixtures_parquet
    ```

=== "Python"

    ```python
    cfg = create(
        # ... about / unit / window / metrics / segments ...
        output={"format": "parquet", "directory": "./fixtures_parquet"},
    )
    write_tables(tables, cfg)
    ```

Same engine path, same seed, ~5–10× smaller on disk. Requires
`pip install plotsim[parquet]`. See
[Output formats](../user-guide/output-formats.md).

---

## Multi-locale fixtures

Single faker locale (default `en_US`) or a list — useful when
fixtures need to look multinational.

=== "YAML"

    ```yaml
    locale: [en_US, ja_JP, de_DE, pt_BR]
    ```

=== "Python"

    ```python
    cfg = create(
        # ...
        locale=["en_US", "ja_JP", "de_DE", "pt_BR"],
    )
    ```

Affects every `faker.*` column (company names, person names, addresses).
Static, metric, and pool columns are untouched.

---

## Schema-evolution / migration testing

Generate two fixtures from the same base config with different
metric or table sets, then diff. Two `create_from_yaml` calls is the
cleanest path; both YAML files share `seed:` so the columns that
overlap are byte-identical:

```python
import numpy as np
from plotsim import create_from_yaml, generate_tables

cfg_v1 = create_from_yaml("saas_v1.yaml")
cfg_v2 = create_from_yaml("saas_v2.yaml")    # adds tickets_v2 metric

v1 = generate_tables(cfg_v1, np.random.default_rng(cfg_v1.seed))
v2 = generate_tables(cfg_v2, np.random.default_rng(cfg_v2.seed))

added   = set(v2["fct_support_tickets"].columns) - set(v1["fct_support_tickets"].columns)
removed = set(v1["fct_support_tickets"].columns) - set(v2["fct_support_tickets"].columns)
```

`v2` includes the new metric in every fact whose `metrics:` list
references it; existing rows still match `v1` for shared columns
(deterministic seed contract). Authoring both as YAML keeps the
diff between the two configs reviewable by hand.

---

## Validate fixtures programmatically

`validate()` runs every cross-table integrity check the engine
supports — PK uniqueness, FK closure, grain, date-spine integrity,
SCD continuity, bridge integrity, causal-lag coherence,
correlation positive-semi-definiteness, null policy, empty event
tables, temporal coherence:

```python
from plotsim import create_from_yaml, generate_tables, validate

cfg = create_from_yaml("saas_fixture.yaml")
tables = generate_tables(cfg)
report = validate(cfg, tables)

if not report.ok:
    for issue in report.errors:
        print(f"[{issue.check}] {issue.table}: {issue.message}")
    raise AssertionError(f"{len(report.errors)} validation errors")
```

In a CI script, `plotsim run config.yaml --strict` does the same
gate at the CLI layer.

---

## The manifest as oracle for pipeline-output comparison

`manifest.json` records every entity's archetype, every event
firing, every quality injection, every SCD band crossing, every
bridge association. Use it as ground truth for assertions about
your pipeline's row counts and aggregates.

```python
import json
from pathlib import Path

mf = json.loads(Path("./fixtures/manifest.json").read_text(encoding="utf-8"))

# Pipeline aggregate vs. expected event count from the manifest
n_pipeline_logins = pipeline_output["dim_company"]["lifetime_logins"].sum()
n_manifest_logins = sum(
    sum(f["row_counts"]) for f in mf["event_firings"]
    if f["table"] == "evt_login"
)
assert n_pipeline_logins == n_manifest_logins
```

See [Manifest reference](../manifest-reference.md) for every
section.

---

## Performance — know the cell-count budget

`entities × periods` drives wall-clock and memory. The engine
prints a one-line summary at config load and gates absurd configs
explicitly:

| Cell count | Behavior |
|---|---|
| `≤ 500,000` | Silent (just the summary line) |
| `> 500,000` | Stderr advisory recommending `output.format: parquet` and `generation_mode: auto` |
| `> soft budget` (default `2,000,000`) | `ValueError` at load with instructions to opt in |
| `> soft budget` with opt-in | Stderr large-dataset notice, generation proceeds |
| `> 50,000,000` | Hard ceiling — `ValueError` regardless of opt-in |

Two ways to opt into above-soft-budget runs: `--allow-large-dataset` on
the CLI, or `PLOTSIM_ALLOW_LARGE_DATASET=1` in the environment. Three
ways to change the soft-budget threshold itself: `output.cell_budget`
in the config (recommended; reproducible from YAML alone),
`PLOTSIM_CELL_BUDGET=N` env var, or the `2,000,000`-cell default.
`output.cell_budget: 0` (or `PLOTSIM_CELL_BUDGET=0`) disables the soft
cap entirely; only the `50,000,000`-cell hard ceiling still applies.
See [Limits](../config-reference.md#limits-and-performance-gates) for
the full ladder; `tests/configs/lakehouse.yaml` in the repo is a
worked example of a config near the 1.5M-cell range.

---

## See also

- [data_quality.ipynb](https://github.com/mohossam01/plotsim/blob/main/docs/site/tutorial-notebooks/data_quality.ipynb) —
  every quality issue type with examples
- [pipeline_testing.ipynb](https://github.com/mohossam01/plotsim/blob/main/docs/site/tutorial-notebooks/pipeline_testing.ipynb) —
  deeper end-to-end recipe
- [CLI reference](../cli-reference.md) — every subcommand and flag
- [Schema guide](../user-guide/schema-guide.md) — designing dim / fact / event tables
- [Output formats](../user-guide/output-formats.md) — CSV vs Parquet
- [Config field reference §quality](../config-reference.md#quality) — every quality issue field
- [Config field reference §noise / §output / §locale](../config-reference.md#noise) — the three top-level dials
