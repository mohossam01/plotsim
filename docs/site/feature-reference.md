# Feature reference

> What plotsim can do today, what config knobs exist, and what lands on
> disk after a run. High-level catalog — for syntax see
> [API](./api-reference.md), [Config fields](./config-reference.md),
> [Column types](./column-types.md), [Manifest](./manifest-reference.md).
>
> Snapshot against `plotsim` `0.6.1`.

---

## At a glance

plotsim takes one YAML (or one `create(...)` call) and writes a
multi-table relational dataset where every metric value, on every row of
every table, is derived from one shared archetype trajectory per entity.
Same seed in → byte-identical files out. Generation is fully offline.

Three surfaces today:

| Surface | Entry point | Audience |
|---|---|---|
| Library | `plotsim.create`, `create_from_yaml`, `generate_tables`, `write_tables` | Python users in an IDE or notebook |
| CLI | `plotsim run`, `validate`, `info`, `template`, `schema` | Terminal, CI, scripts |
| YAML | bundled templates: `bare_minimum`, `saas`, `hr`, `education`, `retail`, `marketing` | Anyone who wants to hand-edit a config |

---

## Capability catalog

### Generation

| Feature | What it produces | Public API |
|---|---|---|
| Trajectory-first metric generation | Every metric for an entity at time *t* is derived from one archetype-curve position | `generate_tables(cfg)` |
| Composable archetype curves | 8 curves (sigmoid, exp_decay, step, logistic, plateau, oscillating, compound, sawtooth) chained as time-windowed segments | YAML `archetypes[].curve_segments`; archetype DSL string |
| Archetype DSL | One-line phrase like `growth: sigmoid then plateau` parsed to segments | `MetricInput.archetype` shorthand; YAML `segments[].archetype` |
| Six metric distributions | `lognorm`, `normal`, `beta`, `poisson`, `gamma`, `weibull` shaping values around the trajectory position | YAML `metrics[].distribution` (advanced); auto-picked by `type` otherwise |
| Causal lag (cause → effect) | Metric A at *t-k* drives metric B at *t* | YAML `metrics[].causal_lag` |
| Connections (correlations) | Cholesky-induced cross-metric correlation, applied per period | YAML `connections[]` |
| Lifecycle stages | Per-entity stage sequence with stage-specific archetype overrides | YAML `lifecycle:` |
| Seasonality | Period-of-year multipliers on metrics (`oscillating` curve + `seasonal_effects:`) | YAML `metrics[].seasonal_effects` |
| Quality issues | Targeted MCAR nulls, outliers, and value-injection at configurable rates | YAML `quality_issues[]` |
| Noise presets | `PERFECTLY_CLEAN`, `SLIGHTLY_MESSY`, `REALISTIC`, `DIRTY` numeric bundles | constants on `plotsim`; YAML `noise:` numeric form |
| Determinism | Single seeded `numpy.random.Generator` flows through every random draw | YAML `seed:` (integer) |

### Tables emitted

| Table type | Grain | Source |
|---|---|---|
| `dim_date` | one row per period | engine — always emitted |
| `dim_<entity>` | one row per entity (or N rows under SCD2) | engine `dimensions.build_dim_entity` |
| `dim_<reference>` | one row per category / plan / region | YAML `dims[]` |
| `fct_<metric>` | one row per entity per period | trajectory + metric pipeline |
| `evt_<event>` | variable rows per entity per period (event firings) | trajectory-driven Poisson |
| Bridge tables | M:N join tables between entities | YAML `bridges[]` |
| `_entity_features.csv` | one row per entity, six aggregates per metric | `plotsim.entity_features.build_entity_features` |

### Slowly-changing dimensions

| Feature | Behavior |
|---|---|
| SCD Type 2 | `dim_<entity>` expanded to N×versions with `valid_from_period` and band-crossing events surfaced in the manifest |
| SCD Type 1 | default (no-op) |

### Holdout splits

`output.holdout: {fraction|periods}` writes
`{table}_train.<csv\|parquet>` + `{table}_holdout.<csv\|parquet>` instead
of one file per fact, split by period index.

### Validation

| Check | Catches |
|---|---|
| FK integrity | orphan rows, missing parents |
| Null rates | configured `noise.missing` exceeded |
| Date spine | gaps, dupes |
| Distribution range | values outside `metrics[].range` |
| Causal-lag fidelity | xcorr peak at the configured lag |
| PK uniqueness | duplicate primary keys |

Run via `plotsim.validate(tables, cfg)` → `ValidationReport` with
`ValidationIssue` records. Same checks gate `plotsim run` (non-zero exit
on critical issues).

### Inspect / provenance

`plotsim.inspect.trace_metric_cell(state, table, row_index, column)`
returns a `TraceResult` containing the trajectory position, curve
segment, archetype, distribution call, noise application, and final
written value for one cell. The differentiator the manifest can't give.

### Manifest sidecar

`manifest.json` accompanies every run. Captures: schema, run metadata,
seed, archetype assignments per entity, sampled trajectory positions
(rate-controlled), event firings, SCD-2 band crossings, configured
quality-issue rates. The "what was true at generation time" record.

### Faker-backed text + identifiers

PII-shape providers wired into the engine: `name`, `email`,
`phone_number`, `company`, `address`, `postcode`, `country`, `city`,
`latitude`, `longitude`, `sentence`. Deterministic under the run seed.
Useful for masking exercises and regex-validation scenarios; **does not
read entity, archetype, or trajectory** (each call is an independent
draw — see "Known limits" below).

---

## Config options at a glance

A `PlotsimConfig` is built from a YAML file (`create_from_yaml`) or
Python kwargs (`create(...)`). The two forms are equivalent — same
fields, same validation, same output. Below is the section-level map;
field-level shape lives in [Config fields](./config-reference.md).

### Top-level sections

| Section | Controls | Required? |
|---|---|---|
| `about`, `unit` | Domain text and the entity noun | yes |
| `window` | Start, end, granularity (`monthly`, `weekly`, `daily`) | yes |
| `seed` | Determinism root | optional (defaults provided) |
| `metrics` | What numeric series exist + their type/polarity/range/distribution/causal_lag | yes (≥1) |
| `archetypes` | Named curve compositions | yes (≥1) |
| `segments` | Entity counts per archetype | yes (≥1) |
| `connections` | Pairwise metric correlations | optional |
| `lifecycle` | Stage sequence with per-stage overrides | optional |
| `dims` | Reference dimension tables (plans, regions, …) | optional |
| `facts` | Custom fact-table layout overrides | optional (auto-derived) |
| `events` | Event firing definitions (`evt_*` tables) | optional |
| `bridges` | M:N join tables | optional |
| `quality_issues` | Targeted nulls / outliers / overrides | optional |
| `noise` | Global gaussian σ, MCAR rate, outlier rate | optional (presets available) |
| `holdout` | Train/holdout split parameters | optional |
| `entity_features` | Whether to emit the per-entity rollup CSV | optional |
| `output` | Format (`csv` or `parquet`), filenames, `cell_budget` env gating | optional |
| `manifest` | Whether to emit, sample rate, schema version | optional |

### Builder shortcuts

`create(...)` accepts the same sections as kwargs, plus a few
convenience shapes:

- `metrics=[{"name": "x", "type": "score|amount|count|rate", "polarity": "positive|negative"}]` — engine picks distribution.
- `segments=[{"name": "...", "count": N, "archetype": "growth"}]` — DSL string parsed into a curve.
- `noise="realistic"` resolves through `NOISE_PRESETS`.
- `window=("2024-01", "2024-12", "monthly")` shorthand.

Templates: `plotsim.list_templates()` →
`["bare_minimum", "education", "hr", "marketing", "retail", "saas"]`.
`plotsim.load_template("saas")` returns a `PlotsimConfig` ready to mutate
or pass to `generate_tables`.

---

## Output produced

Default `./output/` after `plotsim run config.yaml`:

```
output/
├── dim_date.csv             # date spine
├── dim_<entity>.csv         # one row per entity (or N rows under SCD2)
├── dim_<ref>.csv            # zero or more reference dims
├── fct_<metric>.csv         # one row per entity per period
├── evt_<event>.csv          # zero or more event tables
├── <bridge>.csv             # zero or more M:N bridges
├── _entity_features.csv     # optional per-entity rollup
├── config.yaml              # exact config used (round-trip)
├── validation_report.txt    # human-readable validation
├── validation.json          # structured validation
└── manifest.json            # ground-truth sidecar
```

Format conventions:

- UTF-8, `pd.NA` written as empty string, `%.6g` floats.
- Parquet emitted instead of CSV when `output.format: parquet` (requires
  `plotsim[parquet]` extra). Columns and dtypes identical otherwise.
- `output.holdout` swaps each `fct_*` and `evt_*` for paired
  `_train` / `_holdout` files.

Reproducibility contract: run `plotsim run output/config.yaml` against
the same plotsim version → byte-identical output.

---

## Engine internals not yet exposed in the public API

The library has more capability than `plotsim/__init__.py:__all__`
exposes. Anything below works today via direct submodule import, but
isn't part of the supported front door — meaning it can be renamed or
restructured between minor releases without notice. External wrappers
(FastAPI studio, Streamlit demo) should treat this list as gap-to-close,
not as a parallel API.

Symbols are reachable via the path shown; promotion to `plotsim/__all__`
is a straightforward re-export.

| Capability | Where it lives | Why a wrapper might want it |
|---|---|---|
| 8 curve functions + `evaluate_segment` + `CURVE_REGISTRY` | `plotsim.curves` | Render archetype previews ("show me what `sigmoid then plateau` looks like") without spinning up a config |
| Archetype DSL parser | `plotsim.builder.parse_archetype` (re-exported via `plotsim.builder`, not via top-level `plotsim`) | Validate or preview a DSL string from a UI editor |
| Trajectory engine | `plotsim.trajectory.compute_trajectory`, `compute_all_trajectories`, `compute_time_steps` | Generate the position curve for one entity for live preview, without `generate_tables` |
| Distribution registry | `plotsim._distribution_registry.DISTRIBUTION_REGISTRY`, `get_family`, `DistributionFamily` | List available distribution families to a UI dropdown |
| Quality-issue dispatcher | `plotsim.quality.apply_issues` | Apply quality issues to an existing table in a sandbox |
| Holdout helpers | `plotsim.holdout.cutoff_period_index`, `split_fact_tables` | Slice tables on demand without rerunning generation |
| Entity-features builder | `plotsim.entity_features.build_entity_features` | **Already documented in [api-reference.md](./api-reference.md) as if public — but absent from `__all__`. Doc/code drift; promotion would close the gap.** |
| Dimension builders | `plotsim.dimensions.build_dim_date`, `build_dim_entity`, `build_dim_subentity`, `build_dim_reference`, `build_all_dimensions` | Build a single dim for inspection / fixture work |
| JSON-Schema generators | `plotsim.schema.generate_schema`, `plotsim.builder.schema.generate_user_input_schema`, `write_user_input_schema` | Serve the input schema to an editor for autocomplete and lint |
| Builder input models | `plotsim.builder.input.MetricInput`, `SegmentInput`, `ConnectionInput`, `LifecycleInput`, `ColumnInput`, `DimInput`, `FactInput`, `EventInput`, `SeasonalEffectInput`, `QualityIssueInput`, `HoldoutInput`, `EntityFeaturesInput`, `BridgeInput`, `NoiseInput`, `OutputInput` | Pydantic-based field-level validation in a UI; only `UserInput` is plausibly user-facing |
| Manifest constants | `plotsim.manifest.MANIFEST_FILENAME`, `MANIFEST_SCHEMA_VERSION`; `plotsim.entity_features.ENTITY_FEATURES_BASENAME`; `plotsim.schema.SCHEMA_FILENAME` | Identify expected sidecar files without hard-coding strings |

### Known limits (intentional, not "not yet exposed")

These are engine-level limitations, not closed doors waiting to be
opened — surfaced here so a feature catalog reader doesn't go looking:

- **Faker text is non-semantic.** `generated:faker.sentence` and
  similar are independent draws — no access to entity, archetype, or
  trajectory position. Text-classification / sentiment lessons score
  to chance. (Tracked: narrative-text-source mission.)
- **Single source per config.** No multi-system overlap, no CDC change
  log on facts (SCD2 covers dim CDC only), no schema-evolution
  emission. (Tracked: multi-source mode mission.)
- **Flat scalars only.** No struct/array/JSON column types; CSV and
  Parquet writers don't emit nested types.
- **3NF by construction.** No denormalized "wide" output mode.
- **Geo providers independent.** Faker `country` / `city` /
  `postcode` / `lat-lng` are independent draws — no enforced hierarchy
  on a single row.

These are listed in `project/notes/engine-features-maturity.md`
(internal) with mission-shaped fixes. They are out of scope for "what
the library does today."
