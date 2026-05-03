# Changelog

All notable changes to plotsim are documented here.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning: [SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Builder API** — `plotsim.create()` and `plotsim.create_from_yaml()`
  are now the documented public front door. High-level shape (`about`,
  `unit`, `window`, `metrics`, `segments`, plus optional `connections`,
  `lifecycle`, `dimensions`, `facts`, `events`) returns a fully validated
  config. Engine-direct `load_config()` / `dump_config()` is preserved
  as the advanced surface.
- **Builder feature parity** — `create()` now exposes everything
  available in YAML: per-entity attribute pools, M:N bridges, quality
  injection, train/test holdouts, per-entity aggregate tables.
- **Template discovery** — `plotsim.list_templates()` returns the
  bundled builder template names; `plotsim.load_template(name)` loads
  one and returns a `PlotsimConfig`. Tab-completable.
- **Builder templates** — `bare_minimum`, `saas`, `hr`, `education`,
  `retail`, `marketing`. Copy any out with `plotsim template <name>`.
- **`plotsim.inspect`** — public introspection surface for trajectory
  positions, archetype assignments, and tolerance constants.
- **PoolSource + per-archetype `value_range`** — cross-dimension
  attribute draws via `pool:<dim_table>.<column>`, and per-archetype
  re-scaling of bounded distributions.
- **Higham PSD projection** — non-positive-definite correlation
  matrices are projected to the nearest PSD matrix instead of being
  rejected at load. The pre-projection matrix is preserved on the
  manifest for audit.
- **Global seasonal modulation** — a `seasonality` block lets a single
  periodic signal modulate all metric draws (holiday lift, fiscal
  cycles) without per-metric oscillation segments.
- **Correlation pre-compensation** — when configured pairwise
  correlations conflict with the trajectory-driven covariance the
  archetype imposes, the engine subtracts the trajectory contribution
  from the target before sampling. Configured negative correlations on
  growth-and-decline mixes recover their target sign.
- **Vectorized generation path** — per-(entity, period) sampling is now
  an `(E, P, M)` tensor op for the pure-MetricSource fast path.
  End-to-end wall-clock improves 20–35% on the bundled templates.
- **Streaming Parquet writer** — `output.format: parquet` with
  `streaming: true` writes row-group-by-row-group via PyArrow, capping
  peak memory at the largest single chunk. Requires the `[parquet]`
  extra.
- **`segment.count` expansion** — `segments[*].count > 1` now expands
  into per-row `Entity` instances, so sub-entity dim cardinality and
  fact-row counts match user intuition.
- **Tutorial notebooks** — eight Jupyter walkthroughs at
  `docs/tutorial-notebooks/` covering the builder API surface.
- **Builder docs** —
  `docs/builder-{quickstart,reference,errors}.md`: annotated
  walkthroughs, full vocabulary reference, validation-error catalog.
- **Property-based test layer** — `hypothesis>=6.0` in `[test]` /
  `[dev]` extras. Four randomized properties: determinism,
  trajectory-first, FK integrity, correlation accuracy under
  randomized declaration order.
- **Brand pack and docs site** — `mohossam01.github.io/plotsim`,
  README banner, brand assets bundled with the wheel.

### Changed

- **Templates folder is now `plotsim/configs/templates/`** (was
  `plotsim/configs/new/`). The folder is a real Python subpackage —
  templates are importable via `from plotsim.configs.templates.bare_minimum
  import config` and addressable via `importlib.resources`. The 0.5.0
  wheel did not actually include any builder templates due to a
  package-data glob that only covered top-level YAML; this release
  ships all 12 template files (6 `.py` + 6 `.yaml`).
- **Gaussian copula reformulated.** Now applied in textbook order:
  `rng.standard_normal(M) → L @ → family-grouped transform → clip`.
  Cell values change for any config that declares correlations. Same
  `(config, seed)` still produces byte-identical CSVs within this
  release.
- **Correlations no longer attenuate on degenerate metric pairs.**
  Cells whose pre-release center triggered an independent-sample
  bypass (lognorm scale ≈ 0, poisson λ ≈ 0, etc.) now produce
  copula-correlated values. The new family transforms handle
  degenerate centers natively.
- **Builder is the documented entry point** — README, package
  docstring, and `docs/getting-started.md` lead with `create()`.
- **`StageSequence.enforce_order` defaults to `False`** (free-mode
  stages). Strict-monotonic stage walk is opt-in. Bundled templates
  that relied on the prior default carry an explicit `enforce_order:
  true`.
- **Cookbook pages renamed** — `docs/site/cookbook/data-engineers.md`
  → `data-engineering.md`, `data-scientists.md` → `data-science.md`.
- **GitHub URLs migrated to `mohossam01/plotsim`** — project metadata,
  mkdocs config, contributing guide, and 12 docs files updated.

### Fixed

- **Configured correlations match observed values.** Within ±0.10 for
  9 of 10 measured distribution pairings; `lognorm × lognorm` widens
  to ±0.15 at high magnitudes (|coef| ≥ 0.7) due to heavy-tail
  asymmetry of the Gaussian copula on twin lognormals.
- **Causal lags compose across chains.** `A → B(lag=2) → C(lag=3)`
  produces a `C` series that reads `A`'s trajectory from 5 periods
  ago.
- **Cholesky indexing realigned with topological metric order.**
  Configured correlations are no longer applied to a different pair
  of metrics when declaration order differs from toposort order.
- **`validation_report.txt` is byte-identical across runs** — wall-
  clock stamp removed from the library default; CLI still passes one.
  Same `(config, seed)` → byte-identical for every file `write_tables`
  emits.
- **`write_tables` no longer mutates the caller's DataFrame.** A
  shallow copy is taken before in-place dtype promotion.
- **Vectorized fact-builder respects `dtype: int` / `dtype: boolean`.**
  In-memory dtypes from `generate_tables` now match on-disk dtypes.
- **Sub-entity FK collapse on threshold events.** Threshold-event
  tables that FK into a sub-entity dim now distribute their FKs across
  the parent's candidate sub-entities instead of always picking the
  first row.
- **Bridge cardinality validator** — used `sum(Entity.size)` instead
  of `len(entities)` for per-entity dim row counts; engine-direct
  configs with `size > 1` saw spurious warnings.
- **Bridge auto-resolution** — interpreter auto-resolves bridge-
  referenced `dim_{unit}` and `dim_date` tables when the user omits
  them.
- **Proportional-event float cast** — driver values cast to `float64`
  before `np.isnan`, fixing a count-driver bug on proportional events.
- **Duplicate correlation entries raise** instead of silently last-
  write-wins. The pair is treated as unordered (`(a,b) == (b,a)`).
- **`StageSequence.threshold_exit` is wired in.** Hysteresis mode
  (`threshold_exit ≤ threshold_enter`) demotes when value drops below
  exit; legacy mode (`threshold_exit > threshold_enter`) keeps non-
  overlap upper-bound semantics. Was previously decorative.
- **Static-source date validation** — `dtype: date` paired with a
  malformed ISO date (`static:not-a-date`) now rejects at config load
  instead of silently leaving a string in a date-typed column.
- **`Entity.overrides` accepts only known keys.** Now a typed
  `EntityOverrides` model with `extra="forbid"`; unknown keys raise
  at load.
- **`dtype: boolean` paired with `metric:` / `lag:` source rejected
  at load** — the cell value `bool(continuous)` was structurally
  near-constant for any positive-skewed distribution.
- **`causal_lag.lag_periods` cap is now granularity-aware** —
  120 (monthly) / 520 (weekly) / 3650 (daily), each ≈10 years.
- **Silent dispatch sites in `tables.py` raise** when handed an
  unsupported source type, instead of producing a column of `None`.
- **Test-tooling fixes** — coverage instrumentation no longer
  triggers numpy ufunc dispatch corruption in two test sites.

### Removed

- **`requirements.txt`** — `pyproject.toml` is the single source of
  dependency truth for a library.
- **Optional `groq` dependency** — was unused.
- **`_bypass_mask_batch` and bypass-counter plumbing** (~170 lines).
  The new copula handles degenerate centers natively. Manifest field
  `bypass_fallback_counts` is preserved as `{}` for backward-compat.

### Migration

- **Output values shift for any config with correlations.** The
  copula reformulation is intentional and bounded to one version.
  Same `(config, seed)` is byte-identical within this release.
- **`apply_correlations` callers must pass `rng=`.** Internal API;
  most users go through `generate_tables` and are unaffected.
- **Configs with `stages` and no explicit `enforce_order`** now run
  in free mode. Add `enforce_order: true` to keep strict-monotonic
  behavior.
- **`dtype: boolean` on a `metric:` or `lag:` source** is now
  rejected. Switch to `dtype: float` (continuous) / `dtype: int`
  (poisson), or use a `threshold:` source if a boolean indicator was
  intended.

## [0.4.0] — 2026-04-23

Correctness and hardening. Configured correlations and causal lags
match their observed values; malformed configs raise at load instead
of mid-generation.

**Output values change** for any config using correlations or
`causal_lag`. Same `(config, seed)` is byte-identical within 0.4.0.

### Added

- **`CausalLag.blend_weight`** — per-lag float in `[0.0, 1.0]`,
  default `1.0`. At default, lag is a pure period shift, so a metric
  with `lag_periods: N` reads the driver's value from exactly N
  periods ago. Set `blend_weight: 0.6` for the pre-0.4.0 blend.
- **`RedundantCorrelationWarning`** — emitted at load for
  `correlations` entries with `coefficient: 0.0`.
- **CLI `--allow-absolute-output`** — escape hatch for the new sandbox
  on `plotsim run`.
- **SQL-safe identifier validation** on `Table.name` and
  `Column.name`.
- **Faker method allowlist** — 53 methods permitted, 11 denied
  (seeding, provider mutation, `binary`, etc.). 4096-character cap on
  length-like kwargs.
- **Numeric and list-length caps** — entity, metric, table, period,
  and correlation-count limits enforced at load.
- **Config-time cell-count estimator** — one-line stderr summary;
  warns above 500k cells, raises above 2M.
- **Reference fixtures for all 5 shipped templates** at
  `tests/fixtures/layer4_reference/<template>/`.

### Changed

- **Configured correlations are delivered exactly.** Gaussian copula
  replaces the pre-0.4.0 residual-transform. Observed coefficient
  matches configured within ±0.10 (±0.15 for poisson pairings).
- **Causal lags compose across chains.** A 3-metric chain
  `A → B(lag=2) → C(lag=3)` produces `C` reading `A` from 5 periods
  ago. Pre-0.4.0 the `driver` field was vestigial.
- **`assign_stages` vectorized** — `np.searchsorted` +
  `np.maximum.accumulate`. 35–46× speedup across representative
  shapes.
- **Cholesky factor computed once per generation** and threaded
  through the inner loop.
- **Fact-path vectorization.** Per-(entity, period) construction
  materializes an `(E, P, M)` ndarray; scalar fallback preserved for
  `FakerSource` and boolean metric columns.
- **Full-suite wall clock 48s → 31s** post-vectorization.

### Fixed

- **Non-PSD correlation matrices raise at `load_config`** rather than
  at `generate_tables`.
- **Empty `entities: []` configs raise at load** instead of silently
  producing zero-row tables.
- **Cholesky indexing realigned with topological metric order.** Pairs
  whose declaration index differed from their toposort index were
  applied to a different pair of metrics. `saas` and `hr` reference
  fixtures regenerated.

### Removed

- **`plotsim.metrics.LAG_BLEND_WEIGHT` module constant.** Per-lag
  `CausalLag.blend_weight` replaces it.

### Dependencies

- **`scipy>=1.11`** pinned in `pyproject.toml`.

### Migration

- **Configs with correlations** see shifted output values; the new
  values match what was configured. Reference fixtures regenerated
  for all 5 templates.
- **Configs with `causal_lag`** see shifted output. Set
  `blend_weight: 0.6` to recover pre-0.4.0 blend.
- **Non-PSD correlation matrices** must be tightened so all
  eigenvalues are strictly positive.
- **Table and column names** must match `[A-Za-z_][A-Za-z0-9_]{0,127}`.
- **CLI default writes to cwd.** Absolute output paths require
  `--allow-absolute-output`.

## [0.3.0] — 2026-04-22

Post-launch hardening. Two correctness fixes and seven quality
improvements.

### Added

- **Parameterized Faker grammar** — `generated:faker.<provider>[:k:v]*`
  enables e.g. `faker.date_between:start:2022-01-01:end:2024-12-31`.
- **`PlotsimConfig.locale`** — threaded through Faker so `faker.name`,
  `faker.company`, etc. honor the configured locale.
- **`Entity.cross_dim_fks`** + **`Column.distribution`** +
  **`FKDistribution`** — cross-dim FKs draw per-entity with explicit
  uniform / weighted / fixed distributions.
- **`StageSequence.downgrade_delay`** — relaxes strict-monotonic
  progression after N consecutive lower-stage periods.
- **`Column.allow_outside_window`** — opt-out for the new temporal
  coherence validator.
- **`Column.pii_note`** — field-level PII documentation surfaced via
  schema introspection.
- **Path sandbox on `write_tables(base_dir=...)`** — absolute paths
  and `..` traversal rejected.
- **New validators** — `validate_empty_event_tables`,
  `validate_temporal_coherence`,
  `validate_cross_dim_fk_cardinality`.
- **Six archetype distinguishability tests** — KMeans on
  (mean, slope, last-first, std) of each entity's primary continuous
  metric; asserts `adjusted_rand_score > 0.5` vs ground truth.

### Changed

- **`apply_correlations` non-PSD is a hard raise.** Silent fallback
  to independent samples is gone.
- **`assign_stages` and `_entity_groups` vectorized** — 35–46×
  speedup.
- **CLI `info` daily-granularity period estimate** — uses
  `calendar.monthrange` to include the last day of the end month.
- **`sample_hr.yaml hire_date`** switched from unbounded `faker.date`
  to bounded `faker.date_between`.

### Fixed

- **Cross-dim FK collapse to parent row 0** — fact tables now sample
  from the parent dim per-entity using the configured distribution.
- **`hire_date` temporal incoherence** — was landing outside
  `time_window`.
- **`validate_null_policy` isinstance tuple** widened to include
  `FakerSource`.

### Performance

- `assign_stages` 35–46× speedup.

### Dependencies

- **`scikit-learn>=1.3`** added to `[dev]` and a new `[test]` extra.
  Core runtime dependencies unchanged.

### Migration

- Configs whose non-PSD matrices previously generated anyway now
  raise. Tighten the correlation triangle.
- `generated:faker.date` configs that produce out-of-window values
  now warn via `validate_temporal_coherence`. Either switch to
  `faker.date_between`, or set `Column.allow_outside_window: true`.
- `write_tables(base_dir=...)` rejects absolute paths and `..`.

## [0.2.0] — 2026-04-22

### Added

- **`archetypes[].metric_overrides` is wired into generation.**
  Per-archetype overrides of `distribution` and `params` now take
  effect when sampling. Schema previously accepted the field but the
  generator ignored it.
- **`py.typed` marker** ships with the package; mypy / pyright
  recognize plotsim as typed.

### Removed (schema-breaking)

- `Metric.default_curve` — dead field.
- `MetricOverride.curve` — dead field.
- `noise.temporal_jitter_days` — never read by `apply_noise`.
- `noise.duplicate_rate` — never read by `apply_noise`.
- `per_subentity_per_period` grain — used by no table or sample.
- `plotsim/scaffold.py` — empty stub.

### Changed

- **`NOISE_PRESETS`** entries collapsed to the three fields that
  actually apply (`gaussian_sigma`, `outlier_rate`, `mcar_rate`).
- All five bundled samples swept to drop the removed noise fields.
- README gained a `model_json_schema()` snippet so an LLM can author
  a custom-domain config from the live schema.

### Migration

A 0.1.0 config that sets any of the removed fields, or uses the
`per_subentity_per_period` grain, is now rejected at load (Pydantic
`extra="forbid"`). Remove those fields. `metric_overrides` authors
whose configs round-tripped through 0.1.0 without effect should
verify the overrides produce the intended sampling shift under
0.2.0.

## [0.1.0] — 2026-04

Initial public release on PyPI.

- Trajectory-first multi-table generator driven by behavioral
  archetypes.
- YAML-configured domains; 5 bundled templates (saas, hr, ecommerce,
  education, healthcare).
- Curve registry: sigmoid, exp_decay, step, logistic, plateau,
  oscillating, compound, sawtooth.
- Distributions: lognorm, gamma, poisson, beta, normal, weibull.
- Six validation checks: correlation PSD, PK uniqueness, FK
  integrity, date spine, causal coherence, null policy.
- CLI: `run`, `validate`, `info`, `list-templates`, `template`.
- 424 tests.
