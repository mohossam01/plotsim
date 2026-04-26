# Changelog

All notable changes to plotsim are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **F5 (validation_report.txt determinism).** `output._format_report`
  injected `datetime.now()` into the `Generated:` header line, so two
  invocations of `write_tables` with the same `(config, seed)` produced
  byte-different `validation_report.txt` files. CSV output was already
  deterministic; the validation report alone broke the project's
  "same config + same seed → byte-identical output" invariant.

  `write_tables` and `write_validation_report` now accept an optional
  `generated_at: datetime` parameter. When omitted (the library default),
  the `Generated:` line renders a deterministic identifier — a 16-character
  SHA-256 prefix of the JSON-serialized config dump (`config-sha256[:16]`).
  CLI's `cmd_run` passes `generated_at=datetime.now()` so operators still
  see the wall-clock stamp; library callers get determinism by default.

  Same config + same seed now produces byte-identical `validation_report.txt`
  across runs, completing determinism for every file `write_tables` emits.
  Layer4 reference fixtures were regenerated to the new deterministic
  format (one-time fixture commit). Verified by
  `tests/test_validation_report_determinism.py` (byte-identical across
  runs, fingerprint present by default, explicit `generated_at` honored).

- **F4 (`write_tables` mutated the caller's DataFrame).** `write_tables`
  → `write_single_table` used to call `output._coerce_integer_columns`
  directly on the user's DataFrame, reassigning each `dtype:int` column
  in place via `df[col] = series.astype("Int64")`. Even when the dtype
  was already correct after F3, the astype call replaced the Series
  *object*, silently invalidating the user's references to the column.
  `write_single_table` now takes a shallow copy of the DataFrame
  (column-selected `.loc[:, ordered].copy(deep=False)`) before any
  promotion runs. Memory overhead is one Series wrapper per column —
  the underlying float arrays stay shared.

  User-facing impact: `tables[name]` is now untouched by `write_tables`.
  Series objects, dtypes, and any user-added columns survive the call.
  The most natural CSV round-trip (`pd.read_csv(written_path)` with
  default backend) recovers dtypes that match the in-memory ones under
  documented equivalences (`Int64` with no nulls ≡ `int64`; `Int64`
  with nulls ≡ `float64`; `BooleanDtype` with no nulls ≡ `bool`;
  `BooleanDtype` with nulls ≡ `object`). Verified by
  `tests/test_dataframe_mutation.py` (mutation-identity assertion,
  custom-column preservation, default-backend round-trip with
  normalization helper).

- **F3 (vectorized fact-builder dtype on `dtype:int` / `dtype:boolean`
  metric columns).** The vectorized fact-builder used to assign the raw
  float64 slice from `metrics_3d` straight into MetricSource and
  LagSource columns, ignoring the declared `Column.dtype`. Library
  callers consuming the dict from `generate_tables` got float64 where
  they had declared int (and would have got float64 where they
  declared boolean). The CSV path was rescued downstream by
  `output._coerce_integer_columns` at write-time, so on-disk was
  correct but in-memory and on-disk diverged.

  A new helper `_coerce_array_for_dtype(arr, dtype)` now applies the
  scalar `_coerce_metric_value` semantics to whole-column arrays —
  NaN-safe round to nullable `Int64` for `dtype:int`, mask-aware
  cast to `BooleanDtype` for `dtype:boolean`, pass-through otherwise.
  Applied in the MetricSource and LagSource branches of
  `_vectorized_per_entity_per_period_fact`. The `has_bool_metric`
  forced-scalar-fallback gate is removed (boolean metric columns now
  flow through the vectorized path correctly).

  CSV bytes on disk are byte-identical pre/post-fix (the
  write-time rescue was already producing the same shape) — all five
  bundled-template `test_e2e_template[*]` and
  `test_layer4_reference_fixtures_match[*]` tests pass without
  fixture regeneration. In-memory dtype now matches on-disk dtype
  recovered via `pd.read_csv(..., dtype_backend='numpy_nullable')` —
  the contract `tests/test_dataframe_mutation.py` locks down from
  both ends.

  Companion change in `plotsim/validation.py`: `_is_nullish` now uses
  `pd.isna()` so it recognizes `pd.NA` (newly present in-memory via
  F3's nullable extension dtypes) alongside Python `None` and
  `float('nan')`.

- **F2 (correlation attenuation under bypass).** When one metric's
  distribution degenerated mid-period (poisson λ ≈ 0, lognorm
  scale ≈ 0, gamma shape → 0, etc.), `apply_correlations` zeroed
  the bypassed slot in the Gaussian residual vector but kept the
  full Cholesky factor. The matmul `corr_z = L @ z` then mixed the
  forced 0 into every active metric whose Cholesky row had a
  non-zero bypass-column entry, structurally attenuating cross-pair
  correlations during bypass periods.

  Now, when `any(bypass)` is True, the function slices the
  correlation matrix to the active rows/columns (recovered as
  `L @ L.T` — exact for the full-rank Cholesky the load-time PSD
  gate guarantees), Cholesky-factors the principal submatrix, and
  applies that to the active z-values only. Bypass slots stay at 0
  in `corr_z` but are skipped by the per-metric output loop, so the
  zeros never leak into output. An `all(bypass)` short-circuit
  returns the independent draw unchanged.

  **Output values change** for any config whose bypass-prone metric
  is configured-correlated with other metrics, *including* the
  bundled `saas` and `ecommerce` templates (whose negative-polarity
  poisson metrics drive center to ≈ 0 during the second-half of
  several archetype trajectories). Same config + same seed remains
  byte-identical within 0.4.x post-fix, but cross-version
  byte-matching against pre-F2 0.4.x is not preserved. `hr`,
  `education`, and `healthcare` produce byte-identical output
  (none triggers bypass). Verified by
  `tests/test_correlation_bypass.py` (regression + no-bypass
  control across 8 seeds with median-pair-Pearson assertion) and
  the regenerated `test_layer4_reference_fixtures_match` baseline.

- **F1 (sub-entity FK collapse on threshold events).** Threshold-event
  tables that FK into a sub-entity dim now distribute their FKs across
  the parent's candidate sub-entities instead of always picking the
  first row. Pre-fix, `_build_threshold_event` called `_resolve_event_row`
  with `rng=None`, which made the sub-entity FK branch fall back to
  `candidates.iloc[0]` — silently attributing every threshold event for
  a given parent to the same sub-entity record. None of the five
  bundled templates triggered the bug (saas's `evt_churn` only FKs into
  `dim_company`, not `dim_user`), so on-disk output for shipped configs
  is byte-identical. User configs that declare a sub-entity FK on a
  threshold-event table got cardinality-1 joins; those will now see
  proper distribution. Verified by
  `tests/test_threshold_event_subentity.py`.

- **F15 (test tooling).** Replaced `np.polyfit` in `tests/test_integration.py`
  with a manual ordinary-least-squares slope formula in two call sites
  (`test_revenue_follows_trajectory_for_steady_grower` and the
  `_distinguishability_ari` helper used by 6 archetype-distinguishability
  tests). Under coverage.py instrumentation, numpy gets reloaded mid-suite
  (a pandas import warning makes this visible), which corrupts numpy's
  ufunc dispatch table. After that, `np.polyfit`'s internal call to
  `np.linalg.lstsq` crashed with
  `_UFuncNoLoopError(Float64DType, StrDType)` on lstsq's deprecated-default
  check (`if rcond == "warn":`). Result: 7 integration tests that pass
  under `pytest -q` failed under `pytest --cov`. The OLS replacement uses
  pure numpy reductions (no `lstsq` path) and produces the same slope to
  numerical precision. No `plotsim/` source change. Verified against the
  full M101 invocation: 667 passed / 1 xfailed under `--cov=plotsim.tables`,
  matching the `--no-cov` baseline.

## [0.4.0] — 2026-04-23

Correctness and hardening release. Configured correlations and causal
lags now match their observed values in the generated output; a
round of validation and resource-bound work catches malformed configs
at load time rather than mid-generation.

**Output values change** for any config that uses correlations or
`causal_lag`. Same config + same seed remains byte-identical within
0.4.0. Cross-version byte-matching against 0.3.x is not preserved.
See Migration at the bottom.

### Added

- **`CausalLag.blend_weight`** — per-lag float in `[0.0, 1.0]`,
  default `1.0`. Controls the blend between the metric's current
  trajectory position and the driver's past position. At the default
  the lag is a pure period shift, so a metric configured with
  `lag_periods: N` reads the driver's value from exactly N periods
  ago and cross-correlation peaks at that offset. Setting
  `blend_weight: 0.6` recovers the pre-0.4.0 blend.
- **`RedundantCorrelationWarning`** — emitted at `load_config` for
  any `correlations` entry with `coefficient: 0.0`, which is
  structurally valid but operationally a no-op.
- **CLI `--allow-absolute-output` flag** — escape hatch for the new
  `plotsim run` sandbox. Without the flag, `plotsim run` resolves
  the output directory relative to the current working directory
  and rejects `..` traversal.
- **SQL-safe identifier validation** on `Table.name` and `Column.name`
  (pattern `[A-Za-z_][A-Za-z0-9_]{0,127}`, enforced at `load_config`).
- **Faker method allowlist** — 53 methods permitted, 11 denied
  (seeding, provider mutation, `binary`, `format`, `parse`,
  `pystr_format`). 4096-character cap on length-like kwargs.
- **Numeric and list-length caps** on the config model:
  `Entity.size <= 5000`, `ProportionalSource.scale <= 100`,
  `NoiseConfig.gaussian_sigma <= 5.0`,
  `CausalLag.lag_periods <= 120`,
  `StageSequence.downgrade_delay <= 120`;
  `PlotsimConfig.metrics <= 50 / archetypes <= 20 / entities <= 100 /
  tables <= 50 / correlations <= 1225`,
  `Table.columns <= 100`, `Archetype.curve_segments <= 10`,
  `StageSequence.sequence <= 10`.
- **Time-span caps** — 360 monthly / 1560 weekly / 3650 daily
  periods, enforced on `TimeWindow.period_count()`.
- **Total-entity cap** — `sum(Entity.size) > 100_000` rejected at
  load.
- **Config-time cell-count estimator** — prints a one-line stderr
  summary at load (entities × periods, metrics, tables, estimated
  peak memory). Warns above 500k cells; raises above 2M.
- **Reference fixtures for all 5 shipped templates** at
  `tests/fixtures/layer4_reference/<template>/` — the YAML and
  validation report are tracked; regenerated CSVs remain gitignored.
  A metadata canary catches generation-output regressions.

### Changed

- **Configured correlations are now delivered exactly.** A Gaussian
  copula replaces the pre-0.4.0 residual-transform. Each metric's
  marginal distribution is preserved by a CDF round-trip; the
  Cholesky factor delivers the configured correlation in Gaussian
  space, so the observed coefficient matches the configured one
  within ±0.10 for continuous pairings and ±0.15 for pairings that
  include `poisson` (inherent discretization noise).
- **Causal lags compose across chains.** The lag buffer now stores
  effective (post-blend) positions, and metrics are processed in
  topological driver → target order. A three-metric chain
  `A → B(lag=2) → C(lag=3)` produces a `C` series that reads `A`'s
  trajectory from 5 periods ago, matching the declared DAG. Before
  0.4.0 the `driver` field was effectively vestigial and lags did
  not compose.
- **`causal_coherence` validator threshold** relaxed from strict
  `|lagged| > |unlagged|` to a 50% ratio. Under the new pure-shift
  blend, the Iman-Conover same-period correlation can inflate
  `|unlagged|` above `|lagged|` on slow-varying trajectories even
  when the lag is implemented correctly. The ratio still catches
  flagrantly broken lags.
- **`assign_stages` vectorized** — `np.searchsorted` +
  `np.maximum.accumulate` for the strict-monotonic common case;
  per-entity numpy walk preserves state for the `downgrade_delay`
  branch. Measured 35–46× speedup across 85×365 / 500×365 /
  2000×365 shapes. Byte-identical output is locked by parity tests.
- **Cholesky factor computed once per generation** and threaded
  through `build_fact_tables → generate_entity_metrics →
  generate_metrics_for_period` so the per-(entity, period) inner
  loop no longer rebuilds the correlation matrix.
- **Fact-path vectorization.** Per-entity-per-period fact
  construction materializes an `(E, P, M)` float ndarray and
  dispatches between a vectorized path (shipped templates) and a
  preserved scalar path (for `FakerSource` / `boolean` metric
  columns, which need byte-identical RNG consumption).
- **Event-path hybrid vectorization.** Deterministic event columns
  go through `np.repeat` over per-row counts; stochastic columns
  keep the per-row loop.
- **Full-suite wall clock 48s → 31s** post-vectorization.

### Fixed

- **Non-positive-definite correlation matrices are rejected at
  `load_config`** rather than at `generate_tables`. The
  `generate_tables` gate is retained as defense-in-depth for
  callers that construct `PlotsimConfig` programmatically.
- **Empty `entities: []` configs raise at load.** Previously they
  silently produced zero-row dim and fact tables.
- **Cholesky indexing realigned with topological metric order.**
  The Cholesky factor is now built from the topologically sorted
  metric list, so its rows and columns match the z-vector the
  correlation step produces. Before this fix, any pair whose
  declaration index differed from its toposort index got its
  configured coefficient applied to a different pair of metrics.
  Among shipped templates, `saas` and `hr` were affected because
  they use both `correlations` and `causal_lag`; ecommerce,
  education, and healthcare were unaffected (no lag chains, so
  toposort is a no-op). Reference fixtures for `saas` and `hr`
  regenerated.

### Removed

- **`plotsim.metrics.LAG_BLEND_WEIGHT`** module constant. Per-lag
  `CausalLag.blend_weight` replaces it. External imports of the
  constant will raise `ImportError`.

### Dependencies

- **`scipy>=1.11`** pinned in `pyproject.toml` (was implicit before).
  All six shipped distributions (`lognorm`, `gamma`, `poisson`,
  `beta`, `normal`, `weibull`) ship in scipy 1.11.

### Migration

- **Output values shift for configs with correlations.** The
  Gaussian copula reconstructs configured correlations faithfully;
  previously-attenuated coefficients now land where you configured
  them. Reference fixtures for all 5 bundled templates regenerated.
- **Output values shift for configs with `causal_lag`.** The new
  `blend_weight=1.0` default is a pure period shift rather than a
  60/40 blend. Two bundled templates are affected (`sample_saas.yaml`
  `support_tickets` lag=2; `sample_hr.yaml` `absence_rate` lag=1).
  Set `blend_weight: 0.6` explicitly to recover the pre-0.4.0 blend.
- **Output values shift for configs with both correlations AND
  `causal_lag`.** The Cholesky-indexing fix corrects which pair each
  configured correlation applies to. Only `saas` and `hr` among
  shipped templates have both features. There is no way to recover
  pre-fix values from config — the pre-fix values were wrong.
- **Non-PSD correlation matrices now raise at `load_config`**
  instead of at `generate_tables`. Tighten the correlation triangle
  so all eigenvalues are strictly positive.
- **Configs with `entities: []` now raise at load.** Populate the
  list with at least one entity.
- **CLI default writes to cwd.** Pre-0.4.0 callers that relied on
  absolute `output.directory` paths must add
  `--allow-absolute-output` or switch to a relative path.
- **Table and column names must be SQL-safe identifiers.**
  Whitespace, punctuation, and path separators are rejected.
- **Faker methods outside the allowlist raise at load.** The
  denylist covers seeding, provider mutation, and `binary`.
- **Configs above the new numeric / list / span / total-entity
  limits raise at load.** Most hand-authored configs sit well
  under these caps.
- **Cell counts above 2M raise at load; 500k–2M warn.** A one-line
  summary always prints to stderr.
- **`CausalLag.blend_weight` is a new field on a frozen model.**
  0.3.x configs without the field inherit the new default `1.0` on
  load. Round-tripped configs (`dump_config` → YAML) now carry the
  explicit `blend_weight: 1.0` under every `causal_lag`.
- **Metric processing order inside `generate_entity_metrics` is
  now topological** (driver → target). For configs without
  `causal_lag` chains this is a stable permutation and RNG
  consumption is unchanged; configs with chains will see different
  per-period RNG consumption.

## [0.3.0] — 2026-04-22

Post-launch hardening pass. Two correctness fixes and seven quality
improvements sourced from a read-only package audit.

### Added
- **Parameterized Faker source grammar.** `generated:faker.<provider>[:k:v]*`
  with a dedicated `FakerSource` typed parse split out of `GeneratedSource`.
  Enables e.g. `generated:faker.date_between:start:2022-01-01:end:2024-12-31`
  for temporally-bounded date generation instead of `faker.date`'s
  unbounded range.
- **`PlotsimConfig.locale`** (default `"en_US"`) — threaded through the
  dim and fact/event Faker instantiations so providers like `faker.name`
  and `faker.company` honor the configured locale.
- **`Entity.cross_dim_fks`** + **`Column.distribution`** +
  **`FKDistribution`** schema — cross-dimension FKs can now be drawn
  per-entity from a parent dim with explicit uniform / weighted / fixed
  distributions instead of collapsing to parent row 0.
- **`StageSequence.downgrade_delay: int | None`** — relaxes
  strict-monotonic stage progression under `enforce_order=True` after
  N consecutive lower-stage periods. `None` keeps the prior monotonic
  behavior.
- **`Column.allow_outside_window: bool`** — opt-out for the new temporal
  coherence validator.
- **`Column.pii_note: str | None`** — field-level PII documentation carried
  through schema introspection and the README.
- **Path sandbox on `write_tables(base_dir=...)`** — absolute paths and
  `..` traversal are rejected with a clear error.
- **`validate_empty_event_tables`** — warns when an event table produces
  zero rows (likely a threshold misconfiguration or mismatched polarity).
- **`validate_temporal_coherence`** — warns when `faker.date` and other
  date generators produce values outside the config's `time_window`
  (suppressed by `Column.allow_outside_window=True`).
- **`validate_cross_dim_fk_cardinality`** — warns when a cross-dim FK
  distribution references a dim with fewer rows than the assigned weights.
- **Six archetype distinguishability tests.** One per shipped
  template plus a graceful-degradation test. Projects each entity's
  primary continuous metric into (mean, slope, last-first, std),
  clusters with KMeans, asserts `adjusted_rand_score > 0.5` vs.
  ground truth. Catches regressions where curves, noise, or
  correlations silently destroy archetype separability.

### Changed
- **`apply_correlations` non-PSD is now a hard raise.** The previous
  silent fallback to independent samples is gone; a non-PD
  correlation matrix raises at generation time, and
  `generate_tables` gates on `validate_correlation_psd` before
  sampling so the error surfaces at the config boundary rather than
  mid-generation. **Behavior break** for any 0.1.0 / 0.2.0 config
  whose correlations quietly degenerated.
- **`assign_stages` and `_entity_groups` vectorized.**
  `np.searchsorted` + `np.maximum.accumulate` handle the
  strict-monotonic case fully vectorized; a per-entity numpy walk
  preserves state for the `downgrade_delay` branch. **35–46×
  speedup** benchmarked across 85×365 / 500×365 / 2000×365 shapes.
- **CLI `info` daily-granularity period estimate.** Uses
  `calendar.monthrange(end.year, end.month)[1]` to include the
  last-day-of-end-month; previously undercounted daily granularity
  by (days-in-end-month − 1).
- **`apply_correlations` near-zero-center bypass** kept but no longer
  load-bearing (the pre-generation PSD gate means the Cholesky path
  never sees a non-PD matrix).
- **`sample_hr.yaml hire_date`** switched from unbounded `faker.date`
  to `faker.date_between:start:...:end:...` within the config's
  time window.
- **README** gains the `PlotsimConfig.locale` bullet and the cross-dim
  FK distribution documentation.

### Fixed
- **Cross-dim FK collapse to parent row 0** — structural fix. Fact
  tables now sample from the parent dim per-entity using the
  configured `FKDistribution` instead of always returning row 0.
  Invisible on shipped 1-row reference dims; realism-breaking the
  moment a user expands `dim_plan` or `dim_department`.
- **`hire_date` temporal incoherence** — HR sample's `hire_date`
  could land outside `time_window`. The parameterized Faker grammar
  with bounded `faker.date_between` produces in-window dates; the
  new `validate_temporal_coherence` check catches the class of bug
  for future configs.
- **`validate_null_policy` isinstance tuple** widened to include
  `FakerSource`.
- **Stale `Metric.default_curve`-era constructor sites** swept across
  tests (no runtime path; call-site cleanup only).

### Performance
- `assign_stages` 35–46× speedup across representative shapes.

### Dependencies
- **`scikit-learn>=1.3`** added to `[dev]` and a new `[test]`
  optional-dependencies group for the archetype distinguishability
  tests. Core runtime `dependencies` block unchanged — the shipped
  library keeps its numpy / scipy / pandas / pyyaml / pydantic /
  faker footprint.

### Migration
- Configs relying on `apply_correlations`' silent independent-sample
  fallback (non-PD matrices that previously generated anyway) will now
  raise. Fix is one-shot: tighten the correlation triangle so all
  eigenvalues are strictly positive. The five shipped samples are
  already PD.
- Configs using `generated:faker.date` with implicit open-ended date
  ranges still work but will warn via `validate_temporal_coherence` if
  generated values land outside `time_window`. Opt into the warning-free
  path by switching to `faker.date_between:start:Y-M-D:end:Y-M-D` or
  setting `Column.allow_outside_window: true`.
- `write_tables(base_dir=...)` now rejects absolute paths and `..`
  traversal. Callers passing absolute output directories should switch
  to relative or use `output_dir=Path(...)` with `write_tables` directly.

## [0.2.0] — 2026-04-22

### Added
- `archetypes[].metric_overrides` is now wired into generation. Per-archetype
  overrides of `distribution` and `params` take effect when sampling metric
  values (previously the schema accepted the field but the generator silently
  ignored it). Threaded through `generate_metrics_for_period` and
  `generate_entity_metrics` via `Metric.model_copy(update=...)`.
- `py.typed` marker shipped with the package so downstream type-checkers
  (mypy, pyright) recognize plotsim as typed.

### Removed (schema-breaking)
- `Metric.default_curve` — dead field; curves come from archetype segments,
  never from the metric.
- `MetricOverride.curve` — dead field; archetype segments own curve shape,
  not per-metric overrides.
- `noise.temporal_jitter_days` — schema accepted it, `apply_noise` never
  read it.
- `noise.duplicate_rate` — schema accepted it, `apply_noise` never read it.
- `per_subentity_per_period` grain — present in the enum, used by no table
  or sample. Sub-entity dims are routed via `grain: variable` + FK instead.
- `plotsim/scaffold.py` — docstring-only stub with no symbols, referenced
  by no module.

### Changed
- `NOISE_PRESETS` entries collapsed to the three fields that actually apply
  (`gaussian_sigma`, `outlier_rate`, `mcar_rate`).
- All five bundled sample configs (`saas`, `hr`, `ecommerce`, `education`,
  `healthcare`) swept to drop the removed noise fields.
- `FEATURE_REPORT.md` refreshed to match the trimmed surface area.
- README gained a schema-extraction snippet
  (`json.dumps(PlotsimConfig.model_json_schema(), indent=2)`) so an LLM can
  author a custom-domain config from the live schema.

### Migration

A 0.1.0 config that sets any of `default_curve`, `temporal_jitter_days`,
`duplicate_rate`, or uses the `per_subentity_per_period` grain will now be
rejected by `load_config` (Pydantic `extra="forbid"`). Remove those fields;
behavior of the remaining schema is unchanged. `metric_overrides` authors
whose configs round-tripped through 0.1.0 without effect should verify the
overrides produce the intended sampling shift under 0.2.0.

## [0.1.0] — 2026-04

Initial public release on PyPI.

- Trajectory-first multi-table generator driven by behavioral archetypes.
- YAML-configured domains; 5 bundled templates (saas, hr, ecommerce,
  education, healthcare).
- Curve registry: sigmoid, exp_decay, step, logistic, plateau, oscillating,
  compound, sawtooth.
- Distributions: lognorm, gamma, poisson, beta, normal, weibull.
- Six validation checks: correlation PSD, PK uniqueness, FK integrity,
  date spine, causal coherence, null policy.
- CLI: `run`, `validate`, `info`, `list-templates`, `template`.
- 424 tests.
