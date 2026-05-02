# Builder / Docs / Template Coverage Gap Audit

**Date:** 2026-05-02
**Scope:** Engine (`plotsim/config.py`) → Builder (`plotsim/builder/`) → Docs (`docs/site/`) → Templates (`plotsim/configs/new/`)
**Method:** Read-only inspection of all files in each layer, cross-referenced against engine pydantic schema.

This report is the basis for a docs-and-templates parity pass. Notebook tutorials are explicitly out of scope per operator directive; logged in §3.

---

## 1. Coverage Matrix

Legend: ✓ supported · ✗ not supported · ~ partial / behind alias / undocumented branch

### Domain & Time

| Capability | Builder | Docs | Template |
|---|:-:|:-:|:-:|
| `about` (free-text domain summary) | ✓ `UserInput.about` | ✓ config-reference | ✓ all |
| `unit` (entity name) | ✓ `UserInput.unit` | ✓ config-reference | ✓ all |
| `window.start` / `window.end` (`YYYY-MM`) | ✓ `WindowInput` | ✓ config-reference | ✓ all |
| `window.every` (`daily` / `weekly` / `monthly`) | ✓ `WindowInput.every` | ✓ config-reference | ✓ all |
| Window 2/3-tuple shorthand | ✓ `_coerce_window_tuple` | ✓ config-reference | ✓ all |
| Explicit `seed` | ✓ `UserInput.seed` (M124) | ✓ config-reference | ~ commented in saas YAML, never set |

### Metrics

| Capability | Builder | Docs | Template |
|---|:-:|:-:|:-:|
| Type `score` (beta default) | ✓ recipe | ✓ config-reference | ✓ all |
| Type `count` (poisson default) | ✓ recipe | ✓ config-reference | ✓ all |
| Type `amount` (lognorm/beta auto) | ✓ recipe + interpreter | ~ branch threshold (10×) undocumented | ✓ saas/hr |
| Type `index` (normal default) | ✓ recipe | ✓ config-reference | ✓ saas |
| `polarity` (positive/negative) | ✓ `MetricInput.polarity` | ✓ config-reference | ✓ all |
| `range` (`[min, max]`) | ✓ `MetricInput.range` | ✓ config-reference | ✓ all |
| `label` (display name) | ✓ `MetricInput.label` | ✓ config-reference | ~ a few |
| Causal-lag (`follows` + `delay`) | ✓ paired fields | ✓ config-reference | ✓ saas/hr |
| Per-metric `seasonal_sensitivity` | ✓ `MetricInput.seasonal_sensitivity` | ✓ seasonality.md | ✗ no template |

### Archetypes & Curves

| Capability | Builder | Docs | Template |
|---|:-:|:-:|:-:|
| Shape DSL: `growth` | ✓ `parse_archetype` | ✓ archetypes.md | ✓ all |
| Shape DSL: `decline` | ✓ | ✓ | ✓ |
| Shape DSL: `seasonal` | ✓ | ✓ | ✓ saas/retail |
| Shape DSL: `flat` | ✓ | ✓ | ✓ |
| Shape DSL: `spike_then_crash` | ✓ | ✓ | ✓ saas |
| Shape DSL: `accelerating` | ✓ | ✓ | ✓ |
| DSL composition `>` | ✓ parser | ✓ archetypes.md | ✓ all |
| DSL anchor `@N` | ✓ parser | ✓ archetypes.md | ✓ saas |
| Engine curves not in DSL: `sigmoid`, `exp_decay`, `step`, `logistic`, `plateau`, `oscillating`, `compound`, `sawtooth` | ~ behind shape recipes only | ~ engine-internals only | N/A |

### Distributions

| Capability | Builder | Docs | Template |
|---|:-:|:-:|:-:|
| `beta` | ✓ recipe | ✓ config-reference | ✓ all |
| `poisson` | ✓ recipe | ✓ config-reference | ✓ all |
| `lognorm` (range-conditional on amount) | ✓ interpreter | ~ threshold not documented | ~ implicit in saas |
| `normal` (default for index) | ✓ INDEX_DISTRIBUTION | ~ undocumented at builder layer | ~ implicit in saas |

### Correlations / Connections

| Capability | Builder | Docs | Template |
|---|:-:|:-:|:-:|
| 9 connection words (`mirrors` … `inverts`) | ✓ vocabulary | ✓ metrics-and-connections.md | ✓ saas |
| 3-token string / 3-tuple / dict shorthand | ✓ `_coerce_connection` | ✓ config-reference | ✓ saas (string form) |
| **Arbitrary `coefficient` (any r ∈ [-1,1])** | **✗ NOT EXPOSED** | ✗ | ✗ |
| PSD projection (Higham) | ✓ engine-side | ~ engine-internals | N/A |
| Pre-compensation toggle | ~ builder forces `True` | ~ engine-internals | N/A |

### Stages / Lifecycle

| Capability | Builder | Docs | Template |
|---|:-:|:-:|:-:|
| Lifecycle ladder (`track` + `stages`) | ✓ `LifecycleInput` | ✓ config-reference | ✓ saas/hr |
| Stage shorthand forms | ✓ `_normalise_stage_shapes` | ~ brief mention | ✓ saas |
| `stages` alias for `lifecycle` keyword | ✓ alias | ✗ undocumented | N/A |
| **`enforce_order` (monotonic walk)** | **✗ NOT EXPOSED** (always `False`) | ✓ engine-direct only | ✗ |
| **`downgrade_delay` (hysteresis)** | **✗ NOT EXPOSED** | ✓ engine-direct only | ✗ |

### Noise / Quality

| Capability | Builder | Docs | Template |
|---|:-:|:-:|:-:|
| **Noise presets (`PERFECTLY_CLEAN` / `SLIGHTLY_MESSY` / `REALISTIC` / `DIRTY`)** | **✗ NOT EXPOSED** | ✓ engine-direct only | ✗ |
| Noise raw fields (`gaussian_sigma`, `outlier_rate`, `mcar_rate`) | ✗ not exposed | ✓ engine-direct only | ✗ |
| Quality issue types (5: null_injection / duplicate_rows / type_mismatch / late_arrival / schema_drift) | ✓ `QualityIssueInput` | ✓ config-reference | ✗ no template |
| `seed_offset` per quality issue | ✓ | ✓ | ✗ |

### Output & Manifest

| Capability | Builder | Docs | Template |
|---|:-:|:-:|:-:|
| **Output format (`csv` / `parquet`)** | **✗ NOT EXPOSED** (always csv) | ✓ output-formats.md, config-reference | ✗ |
| Output directory | ✗ not exposed | ✓ | ✗ |
| Manifest `include` toggle | ✗ not exposed | ✓ manifest-reference.md | ✗ |
| Manifest `trajectory_sample_rate` | ✗ not exposed | ✓ manifest-reference.md | ✗ |

### Faker / Locale

| Capability | Builder | Docs | Template |
|---|:-:|:-:|:-:|
| `faker.{kind}` column type | ✓ ColumnInput | ✓ column-types.md | ✓ all |
| **Faker `locale` (string or list)** | **✗ NOT EXPOSED** (engine default `en_US` applies) | ✓ engine-direct only | ✗ |
| Faker parameterized providers (e.g. `faker.date_between:start_date:...`) | ✗ not exposed | ✗ | ✗ |

### Schema (Tables / Columns)

| Capability | Builder | Docs | Template |
|---|:-:|:-:|:-:|
| Dim `per` (`period` / `unit`) | ✓ DimInput | ✓ schema-guide.md | ✓ saas |
| Dim `reference: true` | ✓ | ✓ | ✓ saas |
| Dim sub-entity `count` (M115) | ✓ | ~ brief mention | ✗ |
| Fact `metrics` array | ✓ | ✓ | ~ rarely used |
| Event trigger `proportional` (driver+scale) | ✓ | ✓ | ✓ saas/hr |
| Event trigger `threshold` (metric/above/below/for) | ✓ | ✓ | ✓ saas |
| `for` ↔ `for_periods` alias | ✓ alias | ✓ doc'd | ✓ both used |
| Column type `id` | ✓ | ✓ column-types.md | ✓ all |
| Column type `ref.{dim}` | ✓ | ✓ | ✓ all |
| Column type `metric.{name}` | ✓ | ✓ | ✓ all |
| Column type `faker.{kind}` | ✓ | ✓ | ✓ all |
| Column type `static.{value}` | ✓ | ✓ | ✓ saas |
| Column type `pool.{attr}` | ✓ (M122) | ✓ | ✓ saas |
| Column type `segment.count` | ✓ (M117) | ✓ | ✓ saas |
| Column type `timestamp` | ✓ | ✓ | ✓ |
| Column type `flag` (events) | ✓ | ✓ | ✓ saas |
| Column type `bucket` | ✓ | ✓ | ✗ |
| Column type `scd` (SCD Type 2) | ✓ | ✓ | ✓ saas/hr |
| Column dtype-only `date` / `int` / `string` / `float` (dim_date) | ✓ | ✓ | ✓ all |

### Seasonality (M119)

| Capability | Builder | Docs | Template |
|---|:-:|:-:|:-:|
| Global `seasonality` effects (months + strength) | ✓ `SeasonalEffectInput` | ✓ seasonality.md | ✗ no template |
| Multiple overlapping effects | ✓ | ✓ | ✗ |
| Per-segment `seasonal_sensitivity` | ✓ `SegmentInput.seasonal_sensitivity` | ✓ seasonality.md | ✗ |
| Per-metric `seasonal_sensitivity` | ✓ `MetricInput.seasonal_sensitivity` | ✓ seasonality.md | ✗ |

### Bridges / Holdout / Entity Features (M122)

| Capability | Builder | Docs | Template |
|---|:-:|:-:|:-:|
| Bridges (`left` + `right` + `cardinality`) | ✓ `BridgeInput` | ✓ config-reference | ✗ no template |
| Bridge `driver` metric | ✓ | ✓ | ✗ |
| Bridge columns (`metric.X` / `static.X` / `faker.X`) | ✓ `BridgeColumnInput` | ✓ | ✗ |
| Holdout (`target` + `periods` + `min_training_periods`) | ✓ `HoldoutInput` | ✓ config-reference | ✗ |
| Entity features `true` shorthand | ✓ `_coerce_entity_features` | ✓ | ✗ |
| Entity features `metrics` filter | ✓ `EntityFeaturesInput.metrics` | ✓ | ✗ |
| Entity features `include_labels` | ✓ | ✓ | ✗ |

### Engine-only knobs (no builder surface, by design)

| Capability | Builder | Docs | Template |
|---|:-:|:-:|:-:|
| Per-Entity `cross_dim_fks` weighting | ✗ | ✓ engine-direct | N/A |
| Per-Entity `inflection_month` override | ✗ | ✓ engine-direct | N/A |
| Per-Archetype `metric_overrides.curve` | ✗ | ✓ engine-direct | N/A |
| `generation_mode` (`serial`/`vectorized`/`auto`) | ✗ builder forces `auto` | ✓ engine-internals | N/A |

---

## 2. Gaps Blocking Launch

### A. Builder field gaps (engine feature with no builder surface)

These are the operator's flagged five plus closely related coverage:

1. **Noise presets** — engine ships `PERFECTLY_CLEAN`, `SLIGHTLY_MESSY`, `REALISTIC`, `DIRTY` at [config.py:1574-1584](plotsim/config.py#L1574-L1584); `NOISE_PRESETS` dict at [config.py:1579](plotsim/config.py#L1579). UserInput has no `noise` field. Fix: add `noise: str | NoiseInput | None`.

2. **Output format** — engine `OutputConfig` at [config.py:1555](plotsim/config.py#L1555) supports `csv`/`parquet`. Builder hardcodes csv at [interpreter.py:180](plotsim/builder/interpreter.py#L180). Fix: add `output: str | OutputInput | None`.

3. **Faker locale** — engine `PlotsimConfig.locale` at [config.py:1801](plotsim/config.py#L1801). Builder never sets it. Fix: add `locale: str | list[str]` to UserInput.

4. **Arbitrary correlation coefficient** — engine `CorrelationPair.coefficient: float` at [config.py:1425](plotsim/config.py#L1425) accepts any r ∈ [-1,1]. Builder pins to 9 named words via `RELATIONSHIP_RECIPES` at [recipes.py:99-109](plotsim/builder/recipes.py#L99-L109). Fix: extend `ConnectionInput` to accept a numeric coefficient (alongside / instead of the relationship word).

5. **Stage ordering** — engine `StageSequence.enforce_order` and `downgrade_delay` at [config.py:1648-1649](plotsim/config.py#L1648-L1649). Builder always emits `enforce_order=False` at [interpreter.py:453](plotsim/builder/interpreter.py#L453). Fix: add both fields to `LifecycleInput`.

### B. Doc inconsistencies (docs reference builder fields/patterns that mislead users)

6. **Cookbook examples use `model_copy` on engine internals.** Per operator directive: cookbook code examples should use `create()` / `create_from_yaml()`, not `model_copy` on PlotsimConfig. Audit `cookbook/data-engineers.md` and `cookbook/data-scientists.md`.

7. **Cookbook scope language uses role titles.** Operator directive: rename scope from "data engineers" / "data scientists" to domain titles (e.g. "warehouse modelling" / "feature engineering"). Filenames stay for URL stability.

8. **`stages` alias is undocumented.** `UserInput.lifecycle` accepts both `lifecycle:` and `stages:` keys ([input.py:688-691](plotsim/builder/input.py#L688-L691)). config-reference.md only documents `lifecycle:`.

9. **Amount-distribution branching threshold is undocumented.** Interpreter switches between lognorm and beta at `(max/min) >= 10.0` or `min == 0` ([interpreter.py:276](plotsim/builder/interpreter.py#L276)). Users with custom amount ranges have no way to predict which they get.

### C. Template gaps (features documented but never demonstrated)

10. **No template demonstrates seasonality** — every template omits the `seasonality:` block.
11. **No template demonstrates bridges** — many-to-many feature has no example.
12. **No template demonstrates quality issues** — corruption injection has no example.
13. **No template demonstrates holdout** — ML train/holdout split has no example.
14. **No template demonstrates entity features** — flat feature emission has no example.
15. **`saas_template.yaml` includes a schema legend in comments (lines ~254-341)** that is reference material rather than template content. Same legend was previously identified as belonging in config-reference; carrying it on the template adds noise.
16. **Only `saas_template` has a `.py` companion.** `bare_minimum`, `hr`, `education`, `retail`, `marketing` have YAML only.
17. **No template uses parquet output.** With format unexposed in builder this is a chicken-and-egg gap; closes once §A.2 lands.
18. **No template uses noise presets.** Same dependency on §A.1.
19. **No template uses an arbitrary correlation coefficient.** Depends on §A.4.
20. **No template demonstrates `enforce_order: true` lifecycle.** Depends on §A.5.

### D. Missing public-API documentation

21. **`plotsim.inspect.trace_metric_cell`** referenced in api-reference.md but the `plotsim.inspect` module is not listed in the quick-map at top of file. Verify and add.
22. **NOISE_PRESETS dict and constants** are exported from `plotsim` but not documented at the public-API level.

---

## 3. Gaps for Later (deferred — not blocking launch)

- **Notebook tutorials** — operator explicitly asked these be deferred. Existing `notebooks/evaluation.ipynb`, `notebooks/whitebox.ipynb` not reviewed.
- **Cookbook expansion** — end-to-end ML pipeline example (saas → holdout → entity_features → train/eval).
- **Per-entity overrides cookbook** — `cross_dim_fks` and `inflection_month` engine-direct knobs need a recipe.
- **Generator RNG threading recipe** — multi-run sequencing from one seed stream.
- **Manifest JSON schema export** — sibling `manifest-schema.json` for IDE validation.
- **Engine curves not exposed via builder DSL** — `sigmoid`, `step`, `logistic`, `plateau`, `oscillating`, `compound`, `sawtooth` only available through pre-baked shape words. Not blocking; the DSL is intentional.
- **Faker parameterized providers** — engine grammar supports `faker.date_between:start_date:2020-01-01:...` but builder column-type vocabulary doesn't. Decide whether to surface or document as engine-direct only.
- **Holdout × entity_features interaction recipe** — interaction rule documented inline; would benefit from worked example.
- **`saas_template.yaml` legend block** — cleanup task; move legend rows out of the template.
- **In-body M-tag cleanup** — separately tracked as `[127a→docs/in-body-mtags]`.

---

## 4. Inconsistencies (doc says X, code does Y)

- **`stages` vs `lifecycle` alias** — both accepted; only `lifecycle` documented.
- **`for` vs `for_periods` alias on events** — both documented in some places but YAML templates use `for:` while the .py template uses `for_periods:`. Document both forms once and standardise template usage.
- **`saas_template.{py,yaml}` provenance** — flagged in state.md as `[126→docs/saas-template-404]`; internal docs reference these via 404'd GitHub URLs. The files exist locally but were never committed. Closing this gap will resolve when this mission's commit lands.
- **Cookbooks contain `cfg.model_copy(update={...})` patterns** referring to engine-internal fields (e.g. `noise`, `output`). Once §A fixes land these become unnecessary; cookbooks should switch to builder-native syntax.

---

## 5. Operator Verification (the five flagged gaps)

| Gap | Confirmed missing? | Engine reference | Builder evidence |
|---|---|---|---|
| Noise presets | **Yes** | `NOISE_PRESETS` at [config.py:1579](plotsim/config.py#L1579) | No `noise` field in `UserInput` |
| Output format | **Yes** | `OutputConfig.format: Literal["csv","parquet"]` at [config.py:1570](plotsim/config.py#L1570) | Hardcoded csv at [interpreter.py:180](plotsim/builder/interpreter.py#L180) |
| Arbitrary correlation coefficients | **Yes** | `CorrelationPair.coefficient: float` at [config.py:1425](plotsim/config.py#L1425) | Only 9 fixed words via [recipes.py:99-109](plotsim/builder/recipes.py#L99-L109) |
| Stage ordering | **Yes** | `enforce_order` + `downgrade_delay` at [config.py:1648-1649](plotsim/config.py#L1648-L1649) | Hardcoded `enforce_order=False` at [interpreter.py:453](plotsim/builder/interpreter.py#L453) |
| Faker locale | **Yes** | `PlotsimConfig.locale` at [config.py:1801](plotsim/config.py#L1801) | No `locale` field in `UserInput`; never set in interpreter |

All five are real gaps. Fixes proceed.
