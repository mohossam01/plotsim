# Feature reference

> What plotsim can do today, what config knobs exist, and what lands on
> disk after a run. High-level catalog — for syntax see
> [API](./api-reference.md), [Config fields](./config-reference.md),
> [Column types](./column-types.md), [Manifest](./manifest-reference.md).
>
> Snapshot against `plotsim` `0.7.0-dev` (main HEAD `fbbf6ab`, post-M14c).

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
| YAML | bundled templates: `ab_trial`, `bare_minimum`, `cdc_demo`, `crm_billing_overlap`, `education`, `geo_retail`, `hr`, `lakehouse`, `latency_skew`, `marketing`, `narrative_reviews`, `retail`, `saas` | Anyone who wants to hand-edit a config |

---

## Capability catalog

Grouped by typical journey — start at the top with what every run
produces and what most beginners reach for first, work down toward
correlations, lifecycle, audit columns, and the advanced
integrity / provenance tooling.

### 1. Foundations — automatic on every run

| Feature | What it produces | Public API |
|---|---|---|
| Trajectory-first metric generation | Every metric for an entity at time *t* is derived from one archetype-curve position | `generate_tables(cfg)` |
| Determinism | Single seeded `numpy.random.Generator` flows through every random draw | YAML `seed:` (integer) |
| Cell-budget scale gate | Soft pre-flight guard that aborts runs above the configured cell ceiling. Precedence: `output.cell_budget` field > `PLOTSIM_CELL_BUDGET` env > 2M default; `0` disables. Bundled template `lakehouse` exercises a 1.5M-cell config. | YAML `output.cell_budget: <int>`; env override `PLOTSIM_CELL_BUDGET` / `PLOTSIM_ALLOW_LARGE_DATASET` |

#### Tables emitted

| Table type | Grain | Source |
|---|---|---|
| `dim_date` | one row per period | engine — always emitted |
| `dim_<entity>` | one row per entity (or N rows under SCD2) | engine `dimensions.build_dim_entity` |
| `dim_<reference>` | one row per category / plan / region | YAML `dims[]` |
| `fct_<metric>` | one row per entity per period | trajectory + metric pipeline |
| `evt_<event>` | variable rows per entity per period (event firings) | trajectory-driven Poisson |
| Bridge tables | M:N join tables between entities | YAML `bridges[]` |
| `_entity_features.csv` | one row per entity, six aggregates per metric | `plotsim.entity_features.build_entity_features` |

### 2. Shape metrics — pick the curve and distribution

| Feature | What it produces | Public API |
|---|---|---|
| Composable archetype curves | 8 curves (sigmoid, exp_decay, step, logistic, plateau, oscillating, compound, sawtooth) chained as time-windowed segments | YAML `archetypes[].curve_segments`; archetype DSL string |
| Archetype DSL | One-line phrase like `growth: sigmoid then plateau` parsed to segments | `MetricInput.archetype` shorthand; YAML `segments[].archetype` |
| Six metric distributions | `lognorm`, `normal`, `beta`, `poisson`, `gamma`, `weibull` shaping values around the trajectory position | YAML `metrics[].distribution` (advanced); auto-picked by `type` otherwise |

### 3. Add realism — noise and messiness

| Feature | What it produces | Public API |
|---|---|---|
| Noise presets | `PERFECTLY_CLEAN`, `SLIGHTLY_MESSY`, `REALISTIC`, `DIRTY` numeric bundles | constants on `plotsim`; YAML `noise:` numeric form |
| Quality issues | Targeted MCAR nulls, duplicates, type mismatches, late arrivals, schema drift, and per-period volume anomalies (spike / drop) | YAML `quality_issues[]` |

### 4. Relate metrics across time

| Feature | What it produces | Public API |
|---|---|---|
| Connections (correlations) | Cholesky-induced cross-metric correlation between two metrics — a single coefficient held across the whole time window | YAML `connections[]` |
| Time-varying correlations | Phase-keyed Cholesky: declare period windows where a different correlation matrix applies (regime change, market shock, treatment rollout). The engine resolves the active factor per period; the baseline `connections[]` covers periods outside any phase. Each phase runs through the same Higham nearest-PD projection and trajectory-aware compensation as the baseline. | YAML `correlation_phases[]`; builder kwarg `connection_phases` |
| Causal lag (cause → effect) | Metric A at *t-k* drives metric B at *t* | YAML `metrics[].causal_lag` |
| Seasonality | Period-of-year multipliers on metrics (`oscillating` curve + `seasonal_effects:`) | YAML `metrics[].seasonal_effects` |

### 5. Entity lifecycle

| Feature | What it produces | Public API |
|---|---|---|
| Lifecycle stages | Per-entity stage sequence with stage-specific archetype overrides | YAML `lifecycle:` |
| Cohort arrival distribution | Per-segment entity arrival shape — `uniform` / `linear` / `step` / `explicit` — driving `Entity.start_period`, so the entity body grows or contracts across the window. Cold-start cells are NaN-filled and dropped pre-write. Validator enforces every entity has ≥2 active periods. | builder kwarg `arrival:` on segments (4-shape discriminated union); YAML `Entity.start_period` directly |
| Treatment / control cohorts | Per-entity treatment assignment with a logit-shift on trajectory position from `treatment_start_period` onward (`treatment_lift_log_odds`). Known effect → A/B test analysis, uplift modeling, causal inference. Manifest carries `TreatmentAssignment` per entity + `TreatmentCohort` per segment. Bundled template `ab_trial`. | YAML `Entity.treatment_group` / `treatment_lift_log_odds` / `treatment_start_period` |

### 6. Dim columns + fact-grain text — fill non-metric cells with realistic content

| Feature | Behavior |
|---|---|
| Geo bundle provider | `geo.<field>` column types pull country / region / city / postcode / lat-lng from a curated 200-entry, 17-country reference dataset. All fields on the same dim row come from a single bundle, so the city is in the stated country, the postcode looks right for that country, and lat/lng land on the named city. Dim-only; the engine rejects geo on facts/events. See [Geo hierarchy](./user-guide/geo-hierarchy.md). |
| Faker-backed text + identifiers | PII-shape providers wired into the engine: `name`, `email`, `phone_number`, `company`, `address`, `postcode`, `country`, `city`, `latitude`, `longitude`, `sentence`. Deterministic under the run seed. Useful for masking exercises and regex-validation scenarios; **does not read entity, archetype, or trajectory** (each call is an independent draw — see "Known limits" below). |
| Narrative text source (trajectory-aware) | Per-archetype lexicons + a sentence template rendered into a `narrative` column on a fact table. Output vocabulary tracks the entity's trajectory position (a high-position `growth` entity produces systematically different text than a low-position `decline` entity); a simple bag-of-words classifier hits ≥0.55 accuracy on archetype prediction. Deterministic under seed; preserves the trajectory-first invariant. **Fact-only** (rejected on dim / event tables at config load). **Performance:** forces the scalar fact builder path (~3-10× slower than vectorized metric-only facts), so keep narrative on tables that genuinely need text. Bundled template `narrative_reviews`. See [Narrative source](./user-guide/narrative-source.md). |

### 7. Audit + downstream-pipeline outputs

| Feature | Behavior |
|---|---|
| SCD Type 2 | `dim_<entity>` expanded to N×versions with `valid_from_period` and band-crossing events surfaced in the manifest |
| SCD Type 1 | default (no-op) |
| Fact-side CDC | `facts[].cdc: true` emits `_inserted_at` / `_updated_at` / `_op` audit columns; column-level quality issues flip `_op` to `"U"` on affected rows. Demonstrated in `cdc_demo` (dedicated) and `retail` (realistic POS purchase ledger). |
| Holdout splits | `output.holdout: {fraction\|periods}` writes `{table}_train.<csv\|parquet>` + `{table}_holdout.<csv\|parquet>` instead of one file per fact, split by period index |
| Denormalization | `output.denormalized: true` joins each fact with its FK'd dims (SCD2 current-only, audit columns excluded, dim columns prefixed `<dim>__<col>`); emits `<fct>_wide.{csv\|parquet}` alongside normalized output for 1NF–3NF decomposition exercises. Demonstrated in `saas`. |
| Log-file writer | Event tables with `log_format: "{ts} ... "` + `log_filename: "..."` emit a structured `.log` file alongside the CSV/Parquet event table. Format string is `template.format(**row.to_dict())` per row; unknown placeholders raise. Demonstrated in `saas` (`evt_login` as syslog-flavoured lines). |
| Multi-source / overlap | `multi_source:` block emits per-source dim copies with controlled drift (casing / abbreviation / swap) and per-source key schemes; `source_entity_mappings` ground truth in the manifest. Demonstrated in `crm_billing_overlap` (CRM + billing dual-source, 40 mapping records). |
| Nested / JSON columns | `dtype: struct` (with `nested_schema`) or `dtype: array` (with `array_element_type`) paired with `source: nested` on dim columns. Parquet preserves native nested schema (`pa.struct(...)`); CSV serializes as JSON string. Dim-only, one level of nesting, primitive leaves in V1. Demonstrated in `retail` (`dim_product_category.catalog_metadata`). |

### 8. Validation, manifest, and provenance (advanced)

#### Validation

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

#### Manifest sidecar

`manifest.json` accompanies every run. Captures: schema, run metadata,
seed, archetype assignments per entity, sampled trajectory positions
(rate-controlled), event firings, SCD-2 band crossings, treatment
assignments + cohort definitions, bridge associations, and configured
quality-issue rates. Plus three M5 ground-truth sections that turn the
manifest into a scoreable answer key:

| Section | What it carries | Used for |
|---|---|---|
| `causal_graph` | One `CausalEdge` record per non-None `causal_lag` in the config (source metric → target metric, lag, decay window when set) | DE lineage / cataloging exercises; AE semantic-layer modeling |
| `correlations` | One `CorrelationEntry` per metric pair (projected coefficient after Higham nearest-PD); M11 extends with per-phase entries tagged by `phase_index` | DS/ML feature-engineering guidance; correlation-recovery exercises |
| `outlier_injections` | Per-cell log of where the noise outlier branch fired (`None` when skipped — zero outlier rate or above the `OUTLIER_DETECTION_CELL_BUDGET` of 1M cells) | DS/ML anomaly-detection scoring (Isolation Forest, LOF); DE data-quality testing |

The "what was true at generation time" record.

#### Inspect — trace one cell

`plotsim.inspect.trace_metric_cell(state, table, row_index, column)`
returns a `TraceResult` containing the trajectory position, curve
segment, archetype, distribution call, noise application, and final
written value for one cell. The differentiator the manifest can't give.

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
| Dimension builders | `plotsim.dimensions.build_dim_date`, `build_dim_entity`, `build_dim_subentity`, `build_dim_reference`, `build_all_dimensions` | Build a single dim for inspection / fixture work |
| JSON-Schema generators | `plotsim.schema.generate_schema`, `plotsim.builder.schema.generate_user_input_schema`, `write_user_input_schema` | Serve the input schema to an editor for autocomplete and lint |
| Builder input models | `plotsim.builder.input.MetricInput`, `SegmentInput`, `ConnectionInput`, `ConnectionPhase`, `LifecycleInput`, `ColumnInput`, `DimInput`, `FactInput`, `EventInput`, `SeasonalEffectInput`, `QualityIssueInput`, `HoldoutInput`, `EntityFeaturesInput`, `BridgeInput`, `NoiseInput`, `OutputInput` | Pydantic-based field-level validation in a UI; only `UserInput` is plausibly user-facing |
| Manifest constants | `plotsim.manifest.MANIFEST_FILENAME`, `MANIFEST_SCHEMA_VERSION`; `plotsim.entity_features.ENTITY_FEATURES_BASENAME`; `plotsim.schema.SCHEMA_FILENAME` | Identify expected sidecar files without hard-coding strings |
| Manifest sub-models (not in `__all__`) | `plotsim.manifest.CausalEdge`, `CorrelationAdjustment`, `CorrelationCompensation`, `CorrelationEntry`, `CorrelationPhaseInfo`, `OutlierInjection`, `QualityInjection`, `SCDEvent`, `BridgeAssociationRecord`, `HoldoutInfo` | Typed parsing of `manifest.json` from a wrapper (e.g. an evaluator scoring anomaly-detection candidates against the outlier ground truth). The six manifest sub-models already in `__all__` (`EntityArchetypeAssignment`, `ActiveWindow`, `TreatmentAssignment`, `TreatmentCohort`, `TrajectorySample`, `EventFiring`) show the pattern; these would complete the typed surface. |

### Known limits (intentional, not "not yet exposed")

These are engine-level limitations, not closed doors waiting to be
opened — surfaced here so a feature catalog reader doesn't go looking:

- **Faker text is non-semantic.** `generated:faker.sentence` and
  similar are independent draws — no access to entity, archetype, or
  trajectory position. Text-classification / sentiment lessons score
  to chance. For trajectory- and archetype-driven text on fact tables,
  use the `narrative` column type instead — see
  [Narrative text source](./user-guide/narrative-source.md).
- **Single source per config.** No multi-system overlap, no CDC change
  log on facts (SCD2 covers dim CDC only), no schema-evolution
  emission. (Tracked: multi-source mode mission.)
- **Flat scalars only.** No struct/array/JSON column types; CSV and
  Parquet writers don't emit nested types.
- **3NF by construction.** No denormalized "wide" output mode.
- **Faker geo providers are independent draws.** `generated:faker.city`,
  `generated:faker.country`, etc. don't agree on a single row — Faker
  picks each value independently. Use the `geo.<field>` column type
  (or `generated:geo.<field>` source) for row-coherent
  country / region / city / postcode / lat-lng bundles drawn from a
  curated reference dataset; see
  [Geo hierarchy](./user-guide/geo-hierarchy.md).

These are listed in `project/notes/engine-features-maturity.md`
(internal) with mission-shaped fixes. They are out of scope for "what
the library does today."
