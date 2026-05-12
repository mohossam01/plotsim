# Output formats

> What lands on disk, in which format, and how to load it. Plus the
> manifest sidecar and the optional companion files.

---

## CSV (default)

```python
from plotsim import create_from_yaml, generate_tables, write_tables

cfg = create_from_yaml("my_config.yaml")
tables = generate_tables(cfg)
write_tables(tables, cfg)
```

Default output in `./output/`:

```
output/
├── dim_date.csv
├── dim_customer.csv
├── fct_engagement.csv
├── fct_mrr.csv
├── evt_login.csv
├── config.yaml
├── validation_report.txt
└── manifest.json
```

**File format conventions**:

- UTF-8 encoding
- Float format `%.6g` (6 significant digits, mixed scientific / fixed)
- `pd.NA` and `NaN` written as empty strings
- Strings quoted only when they contain commas, newlines, or quotes
  (CSV `QUOTE_NONNUMERIC` style — numeric cells are unquoted)

To load with pandas:

```python
import pandas as pd
df = pd.read_csv("output/fct_engagement.csv")
```

---

## Parquet

Set the output format on the config:

```yaml
output_format: parquet
```

Then run as normal — every table writes as `.parquet` instead of `.csv`.

Parquet writes require `pyarrow`:

```bash
pip install plotsim[parquet]
# or
pip install pyarrow
```

**When to use Parquet**:

- Files are 5–10× smaller than CSV on the bundled templates (typed
  columns + Snappy compression)
- Column dtypes round-trip exactly — no string-vs-int ambiguity
- Faster to load into DuckDB / pandas / polars at scale
- Streaming write per archetype group keeps memory bounded for very
  large fact tables

**When CSV is fine**:

- Smaller datasets (< 1M rows) where the file-size delta doesn't matter
- Tooling that doesn't speak Parquet (some shell pipelines, some legacy
  loaders)
- Eyeballing data in a text editor

To load Parquet with pandas / polars:

```python
import pandas as pd
df = pd.read_parquet("output/fct_engagement.parquet")

# or
import polars as pl
df = pl.read_parquet("output/fct_engagement.parquet")
```

---

## Partitioned Parquet

Set `partition_by` to a column name and every table that carries that
column is written as a Hive-style partitioned directory instead of a
single file. The shape matches what a lakehouse landing zone (S3 +
Glue / Iceberg, GCS + BigLake, ABFS + Synapse) expects — drop the
output directory into the landing bucket and a `MSCK REPAIR TABLE` /
crawler picks up every partition without further setup.

```yaml
output:
  format: parquet
  partition_by: date_key
```

A run on the bundled saas template produces:

```
output/
├── dim_date/                       # has date_key → partitioned
│   ├── date_key=20240101/part-0.parquet
│   ├── date_key=20240201/part-0.parquet
│   └── ...
├── dim_company.parquet             # no date_key → single file
├── dim_user.parquet                # no date_key → single file
├── dim_plan.parquet                # no date_key → single file
├── fct_engagement/
│   ├── date_key=20240101/part-0.parquet
│   └── ...
├── fct_revenue/
│   └── ...
├── fct_support_tickets/
│   └── ...
├── evt_login/
│   └── ...
├── evt_churn/
│   └── ...
├── config.yaml
└── validation_report.txt
```

**Rules:**

- Only applies when `format: parquet`. A `partition_by` paired with
  `format: csv` is rejected at config load.
- Tables that have a column with the named name are partitioned;
  tables without it stay as single files. The validator confirms the
  name resolves on at least one table (typos fail fast) and rejects
  `float` / `struct` / `array` partition keys (Hive-style equality
  matching is ill-defined for those types).
- Companion files (`config.yaml`, `validation_report.txt`,
  `manifest.json`) are always single top-level files — they are not
  table data.
- Denormalized wide-table sidecars (when `denormalized: true`) and
  holdout splits (`<fct>_train` / `<fct>_holdout`) partition on the
  same column when they carry it.
- The per-entity feature file (`_entity_features.parquet`) is
  per-entity with no time axis, so it stays as a single file.

### Loading partitioned datasets

pandas, polars, pyarrow, and DuckDB all read a partitioned directory
without any glob ceremony — point them at the table directory:

```python
import pandas as pd
df = pd.read_parquet("output/fct_engagement")  # directory, not file

import polars as pl
df = pl.read_parquet("output/fct_engagement/**/*.parquet")

import duckdb
duckdb.sql("SELECT * FROM 'output/fct_engagement/**/*.parquet'")
```

The partition column is added back from the directory names on read.

### Streaming + partitioning

The streaming-Parquet row-group writer (used when `generation_mode:
vectorized` + `format: parquet` for very large fact tables) bypasses
cleanly when partitioning is on — the partitioned writer emits one
file per partition value rather than one row group per archetype.
Partitioning is the user-visible knob; streaming is an internal
memory tactic that loses precedence on collision.

---

## JSONL

Set the output format to `jsonl` and every table is written as
newline-delimited JSON — one self-contained JSON object per line. The
shape matches what a Kafka producer, an SQS / Kinesis replay tool, or a
schema-on-read pipeline (Spark / DuckDB / jq) expects to consume.

```yaml
output:
  format: jsonl
```

A run on the bundled saas template produces:

```
output/
├── dim_date.jsonl
├── dim_company.jsonl
├── dim_user.jsonl
├── dim_plan.jsonl
├── fct_engagement.jsonl
├── fct_revenue.jsonl
├── fct_support_tickets.jsonl
├── evt_login.jsonl
├── evt_churn.jsonl
├── config.yaml
├── validation_report.txt
└── manifest.json
```

A handful of lines from `fct_engagement.jsonl`:

```json
{"date_key":20230101,"company_id":"c-001","engagement_score":0.7088,"feature_adoption":0.0,"customer_sentiment":"at_risk","dim_row_id":1}
{"date_key":20230201,"company_id":"c-001","engagement_score":0.6667,"feature_adoption":0.0,"customer_sentiment":"at_risk","dim_row_id":1}
```

**Format conventions**:

- One JSON object per line, terminated by `\n` (LF pinned, even on
  Windows, so files are byte-identical across platforms)
- UTF-8 encoding; non-ASCII characters land verbatim (not as `\uXXXX`
  escapes) — useful for international templates and entity names
- `NaN` / `pd.NA` / `None` serialise as JSON `null`
- Date and datetime columns emit as ISO-8601 strings (`"2024-01-15"`),
  not pandas' default epoch-ms milliseconds
- Nested `struct` columns serialise as native JSON objects; `array`
  columns as native JSON arrays — no JSON-string wrapping (the CSV
  writer wraps because flat-string cells can't carry nested types;
  JSONL doesn't have that constraint)
- Column key order in each row matches the config's column order
  (PK → FK → others)

**When to use JSONL**:

- Streaming-ingestion workflows: drop the file into Kafka / Kinesis /
  SQS as a replay source, one message per line
- Schema-on-read pipelines (Spark `spark.read.json`, DuckDB
  `read_json_auto`, jq, ripgrep over the raw file)
- Nested-data exercises where you want students to see the JSON shape
  directly rather than parse a CSV column
- Hand-inspection of a few rows — `head -3 fct_engagement.jsonl | jq`
  beats opening a Parquet file in a hex editor

**When CSV or Parquet is fine**:

- Tabular BI tooling (Excel, Google Sheets, Looker, Tableau) — they
  speak CSV / Parquet natively, JSONL needs a transform step
- Maximum file-size compactness — Parquet's columnar binary beats
  JSONL's per-row key-name repetition by 5-15x on the bundled templates

### Loading JSONL

pandas, polars, DuckDB, and Spark all read JSONL without ceremony:

```python
import pandas as pd
df = pd.read_json("output/fct_engagement.jsonl", lines=True)

# or
import polars as pl
df = pl.read_ndjson("output/fct_engagement.jsonl")

# or
import duckdb
duckdb.sql("SELECT * FROM read_json_auto('output/fct_engagement.jsonl')")
```

### Replaying through Kafka

The on-disk format is wire-ready — each line is a complete message.
Pipe straight into a producer:

```bash
while IFS= read -r line; do
  kafka-console-producer --topic engagement --broker-list localhost:9092 <<< "$line"
done < output/fct_engagement.jsonl
```

Or in Python:

```python
from kafka import KafkaProducer
producer = KafkaProducer(bootstrap_servers="localhost:9092")
with open("output/fct_engagement.jsonl") as f:
    for line in f:
        producer.send("engagement", line.rstrip("\n").encode("utf-8"))
producer.flush()
```

### Sidecars under JSONL

The same encoding extends to every per-table sidecar so a run never
produces mixed-format output:

- Denormalized wide tables (when `denormalized: true`) →
  `<fct_name>_wide.jsonl`
- Holdout splits (when `holdout` is configured) →
  `<fct_name>_train.jsonl` / `<fct_name>_holdout.jsonl`
- Per-entity features (when `entity_features` is enabled) →
  `_entity_features.jsonl`

Companions are not table data and stay in their canonical text form:
`config.yaml`, `validation_report.txt`, and `manifest.json` are never
re-encoded.

---

## What `write_tables` produces

| File | Always written? | Description |
|---|---|---|
| `<table>.csv` / `.parquet` / `.jsonl` | yes | One file per generated table |
| `config.yaml` | yes | Round-trippable copy of the config used for generation |
| `validation_report.txt` | yes | Human-readable list of FK / PK / spine / null-policy issues |
| `manifest.json` | conditional | Ground-truth signal layer (see below) |
| `<fact>_train.<ext>` / `<fact>_holdout.<ext>` | conditional | Train/holdout split when `holdout` is configured |
| `_entity_features.<ext>` | conditional | Flat per-entity feature table when `entity_features` is enabled |

### `config.yaml`

A complete, round-trippable copy of the config used for this run. Pass
it to `create_from_yaml(...)` and you regenerate the same dataset under
the same plotsim version.

The copy includes engine-derived defaults the original input may have
omitted — useful when you want to see exactly what plotsim filled in
for you.

### `validation_report.txt`

Human-readable validation summary. Header carries error / warning
counts and overall `VALID` / `INVALID` status. Body lists each issue
with check name, table, message, and detail block.

```
Plotsim Validation Report
==========================
Generated: deterministic (config-sha256[:16]=a1b2c3d4...)
Errors: 0 | Warnings: 1 | Total: 1
Status: VALID

[WARN ] empty_event_tables (evt_churn) — 0 rows generated; threshold may be too aggressive
        threshold: above 0.95
```

`Status: VALID` requires zero errors. Warnings don't block — they
inform.

---

## The manifest

`manifest.json` is the ground-truth sidecar. It captures the *signal
layer* — the inputs an ML pipeline would predict against, rather than
re-derive from noisy fact-table cells.

```python
import json
from pathlib import Path

manifest = json.loads(Path("output/manifest.json").read_text())

# Entity → archetype label
labels = {a["entity"]: a["archetype"] for a in manifest["archetype_assignments"]}

# Trajectory position at every period for sampled entities
positions = manifest["trajectory_samples"]
```

The manifest is byte-deterministic — same `(config, seed)` produces the
same JSON. Full field reference in [`manifest-reference.md`](../manifest-reference.md).

To opt out of manifest emission, set `manifest: { include: false }` in
the config. The file is then never written.

---

## Holdout split (optional)

When you declare a `holdout` block, plotsim writes two extra files for
every per-entity-per-period fact table:

```yaml
holdout:
  target: mrr
  periods: 3
  min_training_periods: 6
```

```
output/
├── fct_engagement.csv
├── fct_engagement_train.csv      # periods [0, n - 3)
├── fct_engagement_holdout.csv    # periods [n - 3, n)
├── fct_mrr.csv
├── fct_mrr_train.csv
└── fct_mrr_holdout.csv
```

The unsplit fact table is still written. Dim, bridge, and event tables
are not split — they're not period-indexed in a way that slices cleanly.

The manifest's `holdout` block records `target_metric`,
`holdout_periods`, and the resolved `cutoff_period_index` so a
downstream consumer can re-derive the split without re-reading the
config.

---

## Per-entity features (optional)

When `entity_features: true`, plotsim writes one extra file:
`_entity_features.csv` (or `.parquet`).

```yaml
entity_features: true
```

One row per entity. For every numeric metric the engine landed in a
fact table, six aggregate columns are added per entity:

```
customer_id,
engagement_mean, engagement_std, engagement_slope,
engagement_first, engagement_last, engagement_peak_period,
mrr_mean, mrr_std, mrr_slope, mrr_first, mrr_last, mrr_peak_period,
archetype, final_trajectory_position
```

The `archetype` and `final_trajectory_position` columns are
ground-truth labels pulled from the manifest. They give a downstream
classifier the answer key to learn against.

When `holdout` is also enabled, aggregation is restricted to the
training window and the target metric's six aggregate columns are
dropped to prevent label leakage.

---

## Output directory and overrides

Default location is `./output/` (relative to the working directory).
Override via:

```python
write_tables(tables, cfg, output_dir="path/to/somewhere")
```

If the directory doesn't exist, plotsim creates it. Existing files at
the same paths are overwritten — there's no append, no timestamped
subdirectories. Run twice and the second run replaces the first.

For hosted deployments where you want to constrain output to a sandbox
root:

```python
write_tables(tables, cfg, output_dir="user_request_dir", base_dir="/sandbox")
```

Absolute-path overrides and `..` traversal are rejected when `base_dir`
is set.

---

## Putting it together

```python
from plotsim import (
    create_from_yaml,
    generate_tables_with_state,
    build_manifest,
    write_tables,
)

cfg = create_from_yaml("my_config.yaml")

# Generate tables and the trajectory state alongside
tables, state = generate_tables_with_state(cfg)

# Build the manifest from the state
manifest = build_manifest(
    cfg, state.trajectories, tables,
    scd_state=state.scd, bridge_state=state.bridges,
)

# Write everything
out_path = write_tables(tables, cfg, manifest=manifest)
print(f"Wrote to {out_path}")
```

Or the one-liner version (no manifest):

```python
from plotsim import create_from_yaml, generate_tables, write_tables

cfg = create_from_yaml("my_config.yaml")
write_tables(generate_tables(cfg), cfg)
```

---

## What to read next

- [Manifest reference](../manifest-reference.md) — every manifest field
- [API reference §write_tables](../api-reference.md#write_tables) —
  full parameter list
- [How it works](./how-it-works.md) — what the pipeline produces and why
- [Tutorials → getting started](../tutorial-notebooks/getting_started.ipynb) — runnable end-to-end example
