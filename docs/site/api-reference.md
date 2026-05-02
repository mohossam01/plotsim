# API Reference

> Every public function on `plotsim` — signature, parameters, return type,
> exceptions, example. The companion docs are
> [`config-reference.md`](./config-reference.md) for input fields,
> [`column-types.md`](./column-types.md) for column declarations, and
> [`manifest-reference.md`](./manifest-reference.md) for the ground-truth
> sidecar.
>
> Tutorial walkthroughs live in [Tutorials](./tutorials.md).

---

## Quick map

| Function | Module | What it does |
|---|---|---|
| [`create`](#create) | `plotsim` | Build a config from Python kwargs |
| [`create_from_yaml`](#create_from_yaml) | `plotsim` | Build a config from a YAML file |
| [`generate_tables`](#generate_tables) | `plotsim` | Generate every dim/fact/event/bridge table |
| [`generate_tables_with_state`](#generate_tables_with_state) | `plotsim` | Same, plus the trajectory tape |
| [`validate`](#validate) | `plotsim` | Run every post-generation check on tables |
| [`write_tables`](#write_tables) | `plotsim` | Write tables, config copy, validation report, manifest |
| [`build_manifest`](#build_manifest) | `plotsim` | Build the ground-truth manifest payload |
| [`build_entity_features`](#build_entity_features) | `plotsim` | Flatten facts into one row per entity |
| [`trace_metric_cell`](#trace_metric_cell) | `plotsim.inspect` | Reconstruct one cell's full pipeline path |

Constants exported from `plotsim` for engine-direct mutation:
`PERFECTLY_CLEAN`, `SLIGHTLY_MESSY`, `REALISTIC`, `DIRTY`,
`NOISE_PRESETS`, `PlotsimConfig`, `ManifestSchema`, `GenerationState`,
`ValidationReport`, `ValidationIssue`, `load_config`, `dump_config`.

---

## `create`

Build a `PlotsimConfig` from keyword arguments.

```python
def create(**kwargs) -> PlotsimConfig
```

The keywords mirror the YAML template — see
[`config-reference.md`](./config-reference.md) for the full input shape.
Validation runs at construction time: structural problems
(duplicate names, orphan references, causal-lag cycles, malformed archetype
DSL) raise `pydantic.ValidationError` with the offending field named.

**Returns** — a frozen `PlotsimConfig` ready for `generate_tables`.

**Raises**

- `pydantic.ValidationError` — structural problem in the input.
- `ValueError` — semantic problem the engine catches at config load
  (e.g. archetype refers to an unknown metric, scale gates exceeded).

**Example**

```python
from plotsim import create, generate_tables, write_tables

cfg = create(
    about="Subscription customers",
    unit="customer",
    window=("2024-01", "2024-12", "monthly"),
    metrics=[
        {"name": "engagement", "type": "score", "polarity": "positive"},
        {"name": "mrr",        "type": "amount", "polarity": "positive",
         "range": [10, 5000]},
    ],
    segments=[
        {"name": "growers",  "count": 30, "archetype": "growth"},
        {"name": "decliners","count": 20, "archetype": "decline"},
    ],
    seed=42,
)
tables = generate_tables(cfg)
write_tables(tables, cfg)
```

---

## `create_from_yaml`

Build a `PlotsimConfig` from a YAML file.

```python
def create_from_yaml(path: str | Path) -> PlotsimConfig
```

The YAML follows the same shape as `create(**kwargs)`. YAML's relaxed
scalar parser turns `2024-01` into a date object; `create_from_yaml`
coerces window fields back to strings before construction so the same
validators run for both surfaces.

**Returns** — a frozen `PlotsimConfig`.

**Raises**

- `ValueError` — the file does not parse to a top-level mapping.
- `pydantic.ValidationError` — structural problem in the input.
- `FileNotFoundError` — the path does not exist.

**Example**

```python
from plotsim import create_from_yaml, generate_tables, write_tables

cfg = create_from_yaml("my_config.yaml")
tables = generate_tables(cfg)
write_tables(tables, cfg)
```

---

## `generate_tables`

Run the full pipeline (dimensions → trajectories → facts → events → bridges).

```python
def generate_tables(
    config: PlotsimConfig,
    rng: numpy.random.Generator | None = None,
) -> dict[str, pandas.DataFrame]
```

Returns a dict keyed by table name (`dim_date`, `dim_customer`,
`fct_engagement`, `evt_login`, ...) with one DataFrame per table.

**Determinism** — same `(config, seed)` produces byte-identical output.
If `rng` is omitted, a fresh `numpy.random.default_rng(config.seed)` is
used; passing your own RNG lets you sequence multiple runs against a
single seed stream.

**Pre-flight gates**

The function checks the configured correlation matrix is positive
semi-definite before consuming any randomness. A non-PSD matrix raises
`ValueError` here rather than silently producing partial output.

**Returns** — `dict[str, DataFrame]`.

**Raises**

- `ValueError` — correlation matrix is not positive semi-definite.

**Example**

```python
import numpy as np
from plotsim import create_from_yaml, generate_tables

cfg = create_from_yaml("my_config.yaml")
tables = generate_tables(cfg, np.random.default_rng(42))
print(tables["fct_engagement"].head())
```

---

## `generate_tables_with_state`

Same pipeline, plus the per-entity trajectory tape used during generation.

```python
def generate_tables_with_state(
    config: PlotsimConfig,
    rng: numpy.random.Generator | None = None,
) -> tuple[dict[str, pandas.DataFrame], GenerationState]
```

Use this when you need the ground-truth trajectory positions — the
[manifest builder](#build_manifest) and downstream feature pipelines are
the primary consumers. Recovering positions from noisy fact-table cells
is impossible in general; this function exposes them directly.

`GenerationState` is a frozen dataclass with three fields:

| Field | Type | Contents |
|---|---|---|
| `trajectories` | `dict[str, ndarray]` | Per-entity position array, length `n_periods`, values in `[0, 1]` |
| `scd` | `SCDState` | Per-dim SCD Type 2 versioning (empty when no SCD columns are configured) |
| `bridges` | `BridgeAssociations` | Per-bridge association ground truth (empty when no bridges are configured) |

**Returns** — `(tables, state)`.

**Raises** — same as [`generate_tables`](#generate_tables).

**Example**

```python
from plotsim import generate_tables_with_state, build_manifest

tables, state = generate_tables_with_state(cfg)
manifest = build_manifest(cfg, state.trajectories, tables,
                          scd_state=state.scd, bridge_state=state.bridges)
```

---

## `validate`

Run every post-generation check on a `(config, tables)` pair.

```python
def validate(
    config: PlotsimConfig,
    tables: dict[str, pandas.DataFrame],
) -> ValidationReport
```

`validate` is an alias for `plotsim.validation.validate_tables`. Both
names are exported so existing imports keep working — they refer to the
same function.

Checks run in fixed order so the issue list is deterministic for the
same input:

1. correlation matrix PSD
2. primary-key uniqueness
3. foreign-key integrity
4. date-spine completeness
5. causal-coherence (lag chains land where they should)
6. null-policy adherence
7. empty-event-table heuristic
8. cross-dim FK cardinality
9. temporal coherence (dates inside the window)
10. SCD Type 2 integrity
11. bridge integrity

**`ValidationReport`** — frozen dataclass with these accessors:

| Attribute | Type | Description |
|---|---|---|
| `issues` | `tuple[ValidationIssue, ...]` | Every issue, errors and warnings interleaved in check order |
| `errors` | `tuple[ValidationIssue, ...]` | Filter to `severity == "error"` |
| `warnings` | `tuple[ValidationIssue, ...]` | Filter to `severity == "warning"` |
| `ok` | `bool` | True when `errors` is empty |
| `by_check(name)` | `tuple[ValidationIssue, ...]` | Filter to `check == name` |

**`ValidationIssue`** — frozen dataclass: `check`, `severity`
(`"error"` / `"warning"`), `table` (or `None`), `message`, `details`
(arbitrary key-value dict).

**Returns** — `ValidationReport`.

**Example**

```python
from plotsim import validate

report = validate(cfg, tables)
if not report.ok:
    for issue in report.errors:
        print(f"[{issue.check}] {issue.message}")
```

---

## `write_tables`

Write every generated table, the config copy, the validation report,
and (optionally) the manifest.

```python
def write_tables(
    tables: dict[str, pandas.DataFrame],
    config: PlotsimConfig,
    report: ValidationReport | None = None,
    output_dir: str | Path | None = None,
    float_format: str = "%.6g",
    base_dir: str | Path | None = None,
    generated_at: datetime.datetime | None = None,
    manifest: ManifestSchema | None = None,
) -> Path
```

**Parameters**

| Parameter | Description |
|---|---|
| `tables` | The dict returned by [`generate_tables`](#generate_tables). |
| `config` | The config used for generation. Drives column ordering, dtypes, output format, and quality / holdout / entity-features companion files. |
| `report` | A pre-built `ValidationReport`. When `None`, the full check suite runs first. |
| `output_dir` | Target directory. When `None`, uses `config.output.directory`. |
| `float_format` | Format string for floats in CSV output. Default `"%.6g"`. |
| `base_dir` | Sandbox root for hosted deployments. When set, the resolved target must live under it; absolute-path overrides and `..` traversal raise `ValueError`. |
| `generated_at` | Wall-clock timestamp for the validation-report header. When `None`, the header carries a deterministic config-fingerprint identifier instead. |
| `manifest` | A pre-built `ManifestSchema`. Required when `config.entity_features.enabled` is True; otherwise optional. Written as `manifest.json` when `config.manifest.include` is True. |

**What gets written**

- `<table>.csv` (or `.parquet`) for every key in `tables`.
- `config.yaml` — round-trippable copy of `config`.
- `validation_report.txt` — human-readable.
- `manifest.json` — when `manifest` was passed and `config.manifest.include` is True.
- `<fact>_train.<ext>` and `<fact>_holdout.<ext>` — when `config.holdout.enabled`.
- `_entity_features.<ext>` — when `config.entity_features.enabled`.

The output format (`csv` vs `parquet`) is read off `config.output.format`.
Parquet writes require `pyarrow`; an `ImportError` with the install hint
is raised if it's missing.

Generation failures are not masked: when `report.ok` is False the files
are still written so you can inspect the broken data. Block on
`report.ok` before calling this if you need clean-only output.

**Returns** — the resolved output directory path.

**Raises**

- `ValueError` — `base_dir` violation, or `entity_features.enabled` is
  True but `manifest` was not supplied.
- `ImportError` — `format: parquet` configured but `pyarrow` is not
  installed.

**Example**

```python
from plotsim import generate_tables_with_state, build_manifest, write_tables

tables, state = generate_tables_with_state(cfg)
manifest = build_manifest(cfg, state.trajectories, tables,
                          scd_state=state.scd, bridge_state=state.bridges)
out = write_tables(tables, cfg, manifest=manifest)
print(f"Wrote to {out}")
```

---

## `build_manifest`

Assemble the ground-truth manifest from a generation run.

```python
def build_manifest(
    config: PlotsimConfig,
    trajectories: dict[str, numpy.ndarray],
    tables: dict[str, pandas.DataFrame],
    sample_rate: float | None = None,
    scd_state: SCDState | None = None,
    bridge_state: BridgeAssociations | None = None,
) -> ManifestSchema
```

The manifest captures the *signal layer* a noisy fact table can't
recover: archetype assignments, trajectory positions, event-firing
periods, SCD band crossings, bridge associations, and reproducibility
metadata.

**Parameters**

| Parameter | Description |
|---|---|
| `config` | The config used for generation. |
| `trajectories` | The `state.trajectories` dict from [`generate_tables_with_state`](#generate_tables_with_state). |
| `tables` | The generated tables, used to extract event-firing periods. |
| `sample_rate` | Override for `config.manifest.trajectory_sample_rate`. `None` reads the config value. |
| `scd_state` | Pass `state.scd` to record SCD Type 2 band crossings. `None` leaves `manifest.scd_events` empty. |
| `bridge_state` | Pass `state.bridges` to record M:N associations. `None` leaves `manifest.bridge_associations` empty. |

The function is pure — same inputs produce a byte-identical manifest.
No RNG, no clock, no filesystem.

**Returns** — `ManifestSchema`. See
[`manifest-reference.md`](./manifest-reference.md) for the full field map.

**Example**

```python
from plotsim import generate_tables_with_state, build_manifest, write_manifest
from pathlib import Path

tables, state = generate_tables_with_state(cfg)
manifest = build_manifest(
    cfg, state.trajectories, tables,
    scd_state=state.scd, bridge_state=state.bridges,
)
write_manifest(manifest, Path("output"))
```

---

## `build_entity_features`

Aggregate temporal facts into a single one-row-per-entity DataFrame.

```python
def build_entity_features(
    config: PlotsimConfig,
    tables: dict[str, pandas.DataFrame],
    manifest: ManifestSchema,
) -> pandas.DataFrame
```

For every numeric metric the engine landed in a fact table, six
aggregate columns are emitted per entity: `mean`, `std`, `slope` (linear
fit over period index), `first`, `last`, `peak_period`. When
`config.entity_features.include_labels` is True, two ground-truth
columns are appended: `archetype` and `final_trajectory_position`.

**Pre-conditions** (enforced at config load):

- `config.entity_features.enabled` is True.
- `config.manifest.include` is True (labels read from the manifest).
- `config.quality.quality_issues` is empty (entity features aggregate
  the pre-corruption tables; mixing the two is not supported).
- Every name in `config.entity_features.metrics` resolves to a numeric
  metric on a fact table.

**Holdout interaction** — when `config.holdout.enabled`, aggregation is
restricted to the training window and the target metric's six aggregate
columns are dropped. This is the leakage-prevention rule for downstream
ML.

**Returns** — `pandas.DataFrame` with one row per entity. Column order
is fully determined by config order; same `(config, tables, manifest)`
produces a byte-identical DataFrame every call.

**Raises**

- `ValueError` — config has no `per_entity` dim table, the dim has no
  PK column, the dim was not generated or is empty, or `dim_date` is
  missing.

**Example**

```python
from plotsim import build_entity_features

features = build_entity_features(cfg, tables, manifest)
features.head()
```

In normal use you don't call this directly — `write_tables` invokes it
when `config.entity_features.enabled` is True and writes
`_entity_features.csv` (or `.parquet`).

---

## `trace_metric_cell`

Reconstruct the full pipeline path for one `(entity, period, metric)` cell.

```python
def trace_metric_cell(
    config: PlotsimConfig,
    entity_name: str,
    period_index: int,
    metric_name: str,
    seed: int | None = None,
) -> TraceResult
```

This is the trajectory-first invariant verifier: every realized cell
value can be traced back through trajectory position → polarity flip →
distribution center → seasonal modulation → independent draw →
correlated draw → noise → clamp/round → realized cell. `TraceResult`
captures every intermediate.

Use cases — debugging surprising cell values, asserting trajectory-first
behavior in tests, and pedagogical exploration of the pipeline.

**Parameters**

| Parameter | Description |
|---|---|
| `config` | A loaded `PlotsimConfig`. |
| `entity_name` | Name of an entity in `config.entities`. |
| `period_index` | Zero-based period index. Must be in `[0, n_periods)`. |
| `metric_name` | Name of a metric in `config.metrics`. |
| `seed` | Optional override for `config.seed`. `None` reuses the config seed (matches `generate_tables_with_state`). |

**`TraceResult`** — frozen dataclass with the cell's pipeline path.
The load-bearing assertion the dataclass exists to support is
`result.realized_cell == fct.<metric>` at the matching `(entity, period)`
row. Key fields:

| Field | Description |
|---|---|
| `trajectory_position` | Position in `[0, 1]` from the archetype curve |
| `effective_position` | After causal-lag blend, if any |
| `distribution_center` | After polarity flip + distribution map |
| `seasonal_factor` | Combined global × per-metric × per-entity multiplier |
| `modulated_center` | `distribution_center × (1 + seasonal_factor)`, clamped |
| `independent_draw` | Raw distributional sample |
| `correlated_draw` | After Gaussian-copula transform |
| `noised_value` | After gaussian / outlier / MCAR noise |
| `clamped_value` | After value-range clamp + Poisson round |
| `realized_cell` | The value as found in the generated fact table |

**Returns** — `TraceResult`.

**Raises**

- `EntityNotFound` (subclass of `KeyError`) — entity name not in config.
- `PeriodOutOfRange` (subclass of `IndexError`) — period index outside
  the generated range.
- `MetricNotFound` (subclass of `KeyError`) — metric name not in config.

**Example**

```python
from plotsim import create
from plotsim.inspect import trace_metric_cell

cfg = create(
    about="Subscription customers",
    unit="customer",
    window=("2024-01", "2024-12", "monthly"),
    metrics=[
        {"name": "engagement", "type": "score", "polarity": "positive"},
        {"name": "mrr", "type": "amount", "polarity": "positive",
         "range": [10, 5000]},
    ],
    segments=[{"name": "growers", "count": 30, "archetype": "growth"}],
    seed=42,
)
# Builder expands segments to entity names like "<segment>_0001",
# "<segment>_0002", ... in zero-padded order.
result = trace_metric_cell(cfg, entity_name="growers_0001",
                           period_index=6, metric_name="mrr")
print(f"trajectory={result.trajectory_position:.3f} "
      f"→ realized={result.realized_cell:.2f}")
```
