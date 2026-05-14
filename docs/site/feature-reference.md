# Feature reference

> What plotsim can do today, what config knobs exist, and what lands on
> disk after a run. High-level catalog — for syntax see
> [API](./api-reference.md), [Config fields](./config-reference.md),
> [Column types](./column-types.md), [Manifest](./manifest-reference.md).
>
> Snapshot against `plotsim` `0.6.1` on `main`.

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
| Parent fact (`variable` grain) | one row per discrete instance; count driven by a metric × scale | YAML `facts[]` with `row_count_driver` + `row_count_scale` |
| Child fact (`per_parent_row` grain) | one row per parent row × uniform fan-out; inherits parent's entity + period. Parent FK column auto-synthesized from `parent_table` (bridge pattern). | YAML `facts[]` with `parent_table` + `children_per_row`. See [Parent/child fact grain](./user-guide/parent-child-facts.md). |
| Sibling-fact reference | A second variable-grain fact references the parent via `ref.<other_fact>` with same-entity FK draw (e.g. `fct_returns` referencing `fct_orders`) | YAML `facts[]` column `{type: ref.<fact_name>}`. See [Parent/child fact grain — sibling references](./user-guide/parent-child-facts.md). |
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

#### Trajectory coupling and realized correlation magnitude

Two metrics that ride the same archetype trajectory share variance
from the trajectory itself, on top of whatever the copula injects.
A declared `connections: [a driven_by b: 0.5]` therefore realizes a
table-wide Pearson somewhat **higher** than 0.5 when the entity body
is concentrated in a single archetype, and closer to 0.5 as the
archetype mix broadens.

Plotsim exposes `compensate_correlations` (see
[Config fields](./config-reference.md#compensate_correlations))
to pre-subtract the trajectory contribution from declared targets
before the copula runs. Defaults are split by entry point:

| Entry point | `compensate_correlations` default | Realized magnitude vs. declared |
|---|---|---|
| Builder (`create` / `create_from_yaml`) | `True` | Within ±0.15 across documented distribution pairs |
| Engine-direct (`load_config` on a YAML) | `False` | Higher than declared on mono-archetype configs; widens to ±0.20–0.25 on mixed sweeps |

Each compensation is recorded in `manifest.correlation_compensations`
so the realized-vs-declared delta is auditable per pair. The
compensation cap is 20 metrics — configs with more metrics skip
compensation (with a `UserWarning`) because the additive
trajectory + copula decomposition gets noisy above that count.
Engine-direct configs that need the tight magnitude envelope can
opt in by setting `compensate_correlations: true` in YAML; output
is no longer byte-identical to a pre-flag run of the same file.

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
| Range source | `type: range` with `range: [min, max]` on fact / event columns produces a per-row uniform draw between the bounds. Integer bounds → `dtype: int` and inclusive upper bound; float bounds → `dtype: float` and exclusive upper bound (numpy conventions). Use it for `quantity ∈ [1, 5]`, `unit_price ∈ [10.0, 500.0]`, and similar shape constraints that `faker.random_int` / `faker.pyfloat` express less precisely. Deterministic under seed. |
| Pool source on facts and events | `type: pool.<attribute>` lifts the per-entity value pool (previously dim-only) onto variable-grain facts, per_parent_row child facts, and event tables. Every row resolves to its entity's segment, then draws uniformly from `attributes[<attr>]` — so a `loyal` cohort customer's `channel` always lands in `[app, web]` while a `casual` customer's lands in `[sms, email]`. |
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
quality-issue rates. Plus three ground-truth sections that turn the
manifest into a scoreable answer key:

| Section | What it carries | Used for |
|---|---|---|
| `causal_graph` | One `CausalEdge` record per non-None `causal_lag` in the config (source metric → target metric, lag, decay window when set) | DE lineage / cataloging exercises; AE semantic-layer modeling |
| `correlations` | One `CorrelationEntry` per metric pair (projected coefficient after Higham nearest-PD); time-varying `correlation_phases` add per-phase entries tagged by `phase_index` | DS/ML feature-engineering guidance; correlation-recovery exercises |
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
`["ab_trial", "bare_minimum", "cdc_demo", "crm_billing_overlap", "education", "geo_retail", "hr", "lakehouse", "latency_skew", "marketing", "narrative_reviews", "retail", "saas"]`.
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

- UTF-8, `pd.NA` written as empty string, `%.4f` floats (4 decimal
  places, fixed-point).
- `output.format` selects the per-table encoding: `csv` (default),
  `parquet` (requires `plotsim[parquet]` extra), `jsonl` (one JSON
  object per line), or `sql` (single `data.sql` with dialect-aware DDL
  + INSERTs). Columns and dtypes identical across encodings.
- `output.holdout` swaps each `fct_*` and `evt_*` for paired
  `_train` / `_holdout` files.

Reproducibility contract: run `plotsim run output/config.yaml` against
the same plotsim version → byte-identical output.
