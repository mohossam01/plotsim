# Reference

Full surface area of plotsim 0.5.0 — every CLI command, every config field, the determinism contract, and links to deeper reference material.

## CLI

```text
$ plotsim --help
usage: plotsim [-h] [--version] {run,validate,info,list-templates,template,schema} ...

Generate realistic multi-table datasets from behavioral archetypes.
```

### `plotsim run`

Generate a dataset from a config.

```text
plotsim run <config> [-o OUTPUT_DIR] [-s SEED] [-v] [--strict] [-q] [--allow-absolute-output]
```

| Flag                         | Effect                                                                                       |
| ---------------------------- | -------------------------------------------------------------------------------------------- |
| `-o`, `--output-dir`         | Override the directory in `output.directory`                                                 |
| `-s`, `--seed`               | Override `config.seed`                                                                       |
| `-v`, `--validate`           | Print the post-generation validation report on stdout                                        |
| `--strict`                   | Exit non-zero if validation has any errors                                                   |
| `-q`, `--quiet`              | Suppress the "Generating ..." / "Wrote N tables" status lines                                |
| `--allow-absolute-output`    | Bypass the cwd path sandbox (SEC-01) — required for absolute `output_dir` paths              |

```text
$ plotsim run plotsim/configs/sample_saas.yaml -o ./output --validate
Config summary: 90 entities × 24 periods = 2,160 cells, 6 metrics, 9 tables. Estimated peak memory: ~100 MB. Expected event rows (upper bound): ~10,800.
Generating dataset from plotsim/configs/sample_saas.yaml (seed=42)...
Validation: VALID - 0 error(s), 0 warning(s)
Wrote 9 table(s), 474 total row(s) to ./output/
```

The `Config summary` line on stderr always prints — it's plotsim's resource-bound estimator. Above 500k cells it warns; above 2M it raises.

### `plotsim validate`

Validate a config without running the engine.

```text
plotsim validate <config> [--config-only]
```

`--config-only` is behaviorally identical to the bare command today — both load the YAML and run every Pydantic validator. The flag pins the fast-path contract for CI scripts and reserves the bare command for a future deeper-validation mode.

```text
$ plotsim validate plotsim/configs/sample_saas.yaml
Config summary: 90 entities × 24 periods = 2,160 cells, 6 metrics, 9 tables. Estimated peak memory: ~100 MB. Expected event rows (upper bound): ~10,800.
VALID: plotsim/configs/sample_saas.yaml
```

On failure, exit code is 1 and the message includes the Pydantic error path:

```text
$ plotsim validate broken.yaml
INVALID: broken.yaml
  1 validation error for PlotsimConfig
  archetypes.0
    Value error, archetype 'rocket' curve_segments must end at 1.0, got 0.95 [type=value_error, ...]
```

### `plotsim info`

Preview what a config will generate, without generating.

```text
$ plotsim info plotsim/configs/sample_saas.yaml
Domain: B2B SaaS customer success
Entity type: Customer accounts
Entities: 90 across 3 cohort(s) (acme_corp_cohort, globex_cohort, hooli_cohort)
Time window: 2023-01 to 2024-12 (24 months)
Metrics: 6 (engagement, mrr, support_tickets, feature_adoption, churn_risk, nps)
Archetypes: 6 defined, 3 in use
Tables: 9 (4 dim, 3 fact, 2 event)
Estimated rows: ~6,532
Seed: 42
```

The `Estimated rows` count is dim + per-entity-per-period fact rows — events are not estimated (they depend on trajectory-driven row counts).

### `plotsim list-templates`

List the bundled sample configs.

```text
$ plotsim list-templates
Available templates:
  ecommerce   E-commerce - customers, orders, cart abandonment, returns
  education   University - students, courses, grades, enrollment
  healthcare  Clinic - patients, visits, treatments, outcomes
  hr          HR department - employees, performance, training, attrition
  saas        B2B SaaS - customer accounts, engagement, revenue, churn

Usage: plotsim template ecommerce -o my_config.yaml && plotsim run my_config.yaml
```

### `plotsim template`

Print a bundled template to stdout, or copy it to a file.

```text
plotsim template <name> [-o OUTPUT]
```

```text
$ plotsim template saas -o config.yaml
Wrote config.yaml

$ plotsim template saas | head -5
domain:
  name: "B2B SaaS customer success"
  description: "Customer account behavior across the subscription lifecycle"
  entity_type: "customer_account"
  entity_label: "Customer accounts"
```

### `plotsim schema`

Emit the JSON Schema for `PlotsimConfig`. Used for editor autocomplete and inline validation.

```text
plotsim schema [-o OUTPUT]
```

| `-o` value      | Behavior                                                          |
| --------------- | ----------------------------------------------------------------- |
| (omitted)       | Write to `./plotsim-schema.json`                                  |
| `-o <path>`     | Write to that path                                                |
| `-o -`          | Write to stdout (useful for `plotsim schema -o - \| jq`)          |

```text
$ plotsim schema -o -
{
  "$defs": {
    "Archetype": {
      "additionalProperties": false,
      "properties": {
        "name": {
          "title": "Name",
          "type": "string"
        ...
```

The schema is JSON Schema Draft 2020-12 with `additionalProperties: false` enforced everywhere (matching Pydantic's `extra="forbid"`). Editor integrations (VSCode, JetBrains) point at the produced `plotsim-schema.json` for autocomplete on `sample_*.yaml` files.

Same plotsim version produces byte-identical schema output.

---

## Config schema reference

The authoritative source is [`plotsim/config.py`](https://github.com/mohossam-ae/plotsim/blob/main/plotsim/config.py). What follows is a structural walkthrough.

### Top-level

| Field           | Type                       | Required | Default                       |
| --------------- | -------------------------- | :------: | ----------------------------- |
| `domain`        | `Domain`                   |    ✓     |                               |
| `time_window`   | `TimeWindow`               |    ✓     |                               |
| `seed`          | `int`                      |    ✓     |                               |
| `metrics`       | `list[Metric]` (1–50)      |    ✓     |                               |
| `archetypes`    | `list[Archetype]` (1–20)   |    ✓     |                               |
| `entities`      | `list[Entity]` (1–100)     |    ✓     |                               |
| `tables`        | `list[Table]` (1–50)       |    ✓     |                               |
| `output`        | `OutputConfig`             |    ✓     |                               |
| `correlations`  | `list[CorrelationPair]`    |          | `[]`                          |
| `noise`         | `NoiseConfig`              |          | All zero (clean)              |
| `stages`        | `StageSequence \| null`    |          | `null`                        |
| `locale`        | `str \| list[str]`         |          | `"en_US"`                     |

### `Domain`

| Field          | Type    | Notes                          |
| -------------- | ------- | ------------------------------ |
| `name`         | `str`   | Display name                   |
| `description`  | `str`   | Free text                      |
| `entity_type`  | `str`   | Singular label (`"customer"`)  |
| `entity_label` | `str`   | Plural display (`"Customers"`) |

### `TimeWindow`

| Field          | Type                           | Constraints                                                        |
| -------------- | ------------------------------ | ------------------------------------------------------------------ |
| `start`        | `str`                          | `YYYY-MM`, year ∈ [1900, 2999], month ∈ [1, 12]                    |
| `end`          | `str`                          | `YYYY-MM`, must be > `start`                                       |
| `granularity`  | `"monthly" \| "weekly" \| "daily"` | Span caps: 360 / 1,560 / 3,650 periods respectively         |

### `Metric`

| Field          | Type                           | Notes                                                |
| -------------- | ------------------------------ | ---------------------------------------------------- |
| `name`         | `str`                          | Unique identifier                                    |
| `label`        | `str`                          | Display label                                        |
| `distribution` | `"lognorm" \| "gamma" \| "poisson" \| "beta" \| "normal" \| "weibull"` |    |
| `params`       | `dict[str, float]`             | Distribution-specific (see scipy docs)               |
| `polarity`     | `"positive" \| "negative"`     | Direction of trajectory-to-value mapping             |
| `value_range`  | `ValueRange \| null`           | Optional `{min, max}` clipping                       |
| `causal_lag`   | `CausalLag \| null`            | Optional driver + lag_periods + blend_weight         |

### `CausalLag`

| Field          | Type    | Constraints                                                              |
| -------------- | ------- | ------------------------------------------------------------------------ |
| `driver`       | `str`   | Name of another metric (cycles rejected at load)                         |
| `lag_periods`  | `int`   | ≥ 1; per-granularity cap 120 monthly / 520 weekly / 3,650 daily         |
| `blend_weight` | `float` | [0.0, 1.0], default 1.0 (pure shift)                                     |

### `Archetype`

| Field              | Type                              | Constraints                                                |
| ------------------ | --------------------------------- | ---------------------------------------------------------- |
| `name`             | `str`                             | Unique                                                     |
| `label`            | `str`                             | Display label                                              |
| `description`      | `str`                             | Free text                                                  |
| `curve_segments`   | `list[CurveSegment]` (1–10)       | First starts at 0.0, last ends at 1.0, no gaps/overlaps    |
| `metric_overrides` | `dict[str, MetricOverride]`       | Per-metric `distribution` / `params` / `value_range` override (M114) |

### `CurveSegment`

| Field        | Type                                                                 | Constraints                       |
| ------------ | -------------------------------------------------------------------- | --------------------------------- |
| `curve`      | `"sigmoid" \| "exp_decay" \| "step" \| "logistic" \| "plateau" \| "oscillating" \| "compound" \| "sawtooth"` |  |
| `params`     | `dict[str, float \| int \| bool]`                                    | Curve-specific                    |
| `start_pct`  | `float`                                                              | [0.0, 1.0], strictly < `end_pct`  |
| `end_pct`    | `float`                                                              | [0.0, 1.0]                        |

### `Entity`

| Field            | Type                  | Constraints                                                          |
| ---------------- | --------------------- | -------------------------------------------------------------------- |
| `name`           | `str`                 | Unique                                                               |
| `archetype`      | `str`                 | Must reference an archetype name                                     |
| `size`           | `int`                 | [1, 5,000]; sum across all entities ≤ 100,000                        |
| `overrides`      | `EntityOverrides \| null` | Optional `inflection_month`                                       |
| `cross_dim_fks`  | `dict[str, str]`      | FIX-04: per-cohort cross-dim FK anchoring                            |

### `Table`

| Field                | Type                                                | Notes                                                              |
| -------------------- | --------------------------------------------------- | ------------------------------------------------------------------ |
| `name`               | `str`                                               | SQL-safe identifier                                                |
| `type`               | `"dim" \| "fact" \| "event"`                        |                                                                    |
| `grain`              | `"per_entity" \| "per_period" \| "per_reference" \| "per_entity_per_period" \| "variable"` |  |
| `columns`            | `list[Column]` (1–100)                              |                                                                    |
| `primary_key`        | `str \| list[str]`                                  | Single or composite                                                |
| `foreign_keys`       | `list[str]`                                         | `<table>.<column>` format                                          |
| `row_count_source`   | `str \| null`                                       | Event tables only — `proportional:` source                         |

### `Column`

| Field                    | Type                  | Notes                                                                      |
| ------------------------ | --------------------- | -------------------------------------------------------------------------- |
| `name`                   | `str`                 | SQL-safe identifier                                                        |
| `dtype`                  | `"int" \| "float" \| "string" \| "date" \| "boolean" \| "id"` |                                |
| `source`                 | `str`                 | One of the source kinds (see [Config guide](config-guide.md))              |
| `pii_note`               | `str \| null`         | Free-text PII descriptor — surfaced in schema introspection                |
| `distribution`           | `FKDistribution \| "uniform" \| null` | FK sampling — uniform or weighted                          |
| `allow_outside_window`   | `bool`                | Default `false` — set to `true` for hire dates, birth dates, etc.          |
| `scd_type2`              | `SCDType2Config \| null` | SCD Type 2 config; required when `source: "scd_type2"` (M106)           |
| `value_pool`             | `dict[str, list[str]] \| null` | Per-entity sample pool; required when `source: "pool:<name>"` (M114) |

### `MetricOverride`

| Field          | Type                          | Notes                                                                  |
| -------------- | ----------------------------- | ---------------------------------------------------------------------- |
| `distribution` | `Distribution \| null`        | Override the global metric distribution family for this archetype       |
| `params`       | `dict[str, float] \| null`    | Override distribution shape params                                     |
| `value_range`  | `ValueRange \| null`          | M114: per-archetype value range; must be a subset of the global metric `value_range` (overrides restrict, never expand) |

### `CorrelationPair`

| Field         | Type    | Constraints                                                          |
| ------------- | ------- | -------------------------------------------------------------------- |
| `metric_a`    | `str`   | Must reference a metric                                              |
| `metric_b`    | `str`   | Must reference a different metric                                    |
| `coefficient` | `float` | [−1.0, 1.0]; `0.0` warns (no-op)                                     |

Pair is unordered; declaring both `(a, b)` and `(b, a)` is rejected.

### `NoiseConfig`

| Field            | Type    | Constraints                                                          |
| ---------------- | ------- | -------------------------------------------------------------------- |
| `gaussian_sigma` | `float` | [0.0, 5.0], default 0.0                                              |
| `outlier_rate`   | `float` | [0.0, 1.0], default 0.0                                              |
| `mcar_rate`      | `float` | [0.0, 1.0], default 0.0                                              |

Presets: `Perfectly clean`, `Slightly messy`, `Realistic`, `Dirty` — see `plotsim.config.NOISE_PRESETS`.

### `OutputConfig`

| Field        | Type                       | Default  | Notes                                                                       |
| ------------ | -------------------------- | -------- | --------------------------------------------------------------------------- |
| `format`     | `"csv" \| "parquet"`       | `"csv"`  | Parquet requires `pip install plotsim[parquet]` (pyarrow)                   |
| `directory`  | `str`                      |          | Relative path; absolute requires `--allow-absolute-output` on the CLI       |

### `StageSequence`

| Field              | Type                       | Notes                                                                     |
| ------------------ | -------------------------- | ------------------------------------------------------------------------- |
| `field`            | `str`                      | Must reference a metric                                                   |
| `sequence`         | `list[StageDefinition]` (2–10) | Last stage must have `threshold_exit: null`                          |
| `enforce_order`    | `bool`                     | Default `false` — per-period free-mode assignment (highest-enter stage). Set `true` for monotonic forward-only progression |
| `downgrade_delay`  | `int \| null`              | [1, 120]; consecutive periods below threshold before demote                |

Two semantic modes — see [Config guide § Stages](config-guide.md#stages).

---

## Determinism contract

**Same config + same seed = byte-identical output**, with the following scope:

| Axis                                | Tested? | Guarantee     |
| ----------------------------------- | :-----: | ------------- |
| Same Python process                 |    ✓    | Byte-identical |
| Cross-process, same cwd             |    ✓    | Byte-identical |
| Cross-process, different cwd        |    ✓    | Byte-identical |
| Cross-Python-version (3.10 → 3.11)  |         | Not tested    |
| Cross-numpy-version (1.26 → 2.0)    |         | Not tested    |
| Cross-OS (Windows → Linux)          |         | Not tested    |

For the cross-version dimensions, pin the toolchain in CI rather than relying on plotsim's determinism contract.

`validation_report.txt` is byte-identical via a config-fingerprint header (16-character SHA-256 prefix); the CLI also stamps wall-clock time, but library callers (`write_tables` without `generated_at=...`) get the deterministic header.

---

## Statistical fidelity

The empirical bounds plotsim's correlation, lag, and trajectory-first guarantees hold within are documented in [`docs/statistical-fidelity.md`](https://github.com/mohossam-ae/plotsim/blob/main/docs/statistical-fidelity.md). Headlines:

- **Correlation**: 9 of 10 distribution pairings within ±0.10 of configured Pearson; `lognorm × lognorm` widens to ±0.15 at high magnitudes.
- **Trajectory-first cell-level invariant**: median deviation −0.04 σ, 99th-percentile 3.60 σ across 11,865 cells; cells beyond 4 σ (0.84 %) accounted for by `outlier_rate`.
- **Causal lag (output cross-correlation)**: lags 1 and 2 recoverable; lag ≥ 5 with smooth drivers fails *detection* (engine-layer correctness unchanged).

The smoke test [`tests/test_fidelity_smoke.py`](https://github.com/mohossam-ae/plotsim/blob/main/tests/test_fidelity_smoke.py) re-checks the headline tolerances on every CI run.

---

## Changelog

Full release history: [`CHANGELOG.md`](https://github.com/mohossam-ae/plotsim/blob/main/CHANGELOG.md).

| Version  | Date       | Headline                                                                              |
| -------- | ---------- | ------------------------------------------------------------------------------------- |
| 0.5.0    | 2026-04-25 | Statistical fidelity addendum; hypothesis property-based tests; F1–F17 fix burndown   |
| 0.4.0    | 2026-04-23 | Gaussian copula correlations; lag chains; load-time PSD gate; vectorized fact path    |
| 0.3.0    | 2026-04-22 | Parameterized Faker; cross-dim FK distributions; downgrade_delay; 35–46× stage speedup |
| 0.2.0    | 2026-04-22 | Schema cleanup (dead fields removed); `archetypes[].metric_overrides` wired           |
| 0.1.0    | 2026-04    | Initial PyPI release                                                                  |

---

## Source

| Component         | File                              |
| ----------------- | --------------------------------- |
| Config schema     | `plotsim/config.py`               |
| Curve library     | `plotsim/curves.py`               |
| Trajectory engine | `plotsim/trajectory.py`           |
| Metric generators | `plotsim/metrics.py`              |
| Dim builders      | `plotsim/dimensions.py`           |
| Fact/event builders | `plotsim/tables.py`             |
| Validation        | `plotsim/validation.py`           |
| Output writer     | `plotsim/output.py`               |
| JSON Schema       | `plotsim/schema.py`               |
| CLI               | `plotsim/cli.py`                  |
| Bundled templates | `plotsim/configs/sample_*.yaml`   |
