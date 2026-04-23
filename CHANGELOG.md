# Changelog

All notable changes to plotsim are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] — 2026-04-22

Post-launch hardening pass sourced from the 2026-04-22 read-only
package audit (2 Must Fix, 9 Should Fix). Cumulative fix bundle from
five commit groups (`bb6feb2` → `0af6b27` → `0aed106` → `2e63bac` →
`78788cd`).

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
- **Six archetype distinguishability tests** (FIX-09, SF-7). One per
  shipped template plus a graceful-degradation test. Projects each
  entity's primary continuous metric into (mean, slope, last-first, std),
  clusters with KMeans, asserts `adjusted_rand_score > 0.5` vs. ground
  truth. Catches regressions where curves/noise/correlations silently
  destroy archetype separability.

### Changed
- **`apply_correlations` non-PSD is now a hard raise** (FIX-01, SF-1).
  The M004-era silent fallback to independent samples is gone; a non-PD
  correlation matrix raises at generation time. `generate_tables` gates
  on `validate_correlation_psd` before sampling so the error surfaces
  at the config boundary rather than mid-generation. **Behavior break**
  for any 0.1.0/0.2.0 config whose correlations quietly degenerated.
- **`assign_stages` + `_entity_groups` vectorized** (FIX-07, SF-5).
  `np.searchsorted` + `np.maximum.accumulate` handle the strict-monotonic
  case fully vectorized; a per-entity numpy walk preserves state for the
  `downgrade_delay` branch. **35–46× speedup** benchmarked across
  85×365 / 500×365 / 2000×365 shapes (mission target was ~5×).
- **CLI `info` daily-granularity period estimate** (FIX-02, SF-6).
  `_estimate_periods` now uses `calendar.monthrange(end.year, end.month)[1]`
  to include the last-day-of-end-month; previously undercounted daily
  granularity by (days-in-end-month − 1).
- **`apply_correlations` near-zero-center bypass** kept but no longer
  load-bearing (the pre-generation PSD gate means the Cholesky path
  never sees a non-PD matrix).
- **`sample_hr.yaml hire_date`** switched from unbounded `faker.date`
  to `faker.date_between:start:...:end:...` within the config's
  time window.
- **README** gains the `PlotsimConfig.locale` bullet and the cross-dim
  FK distribution documentation.

### Fixed
- **Cross-dim FK collapse to parent row 0** (FIX-04, MF-1) — structural
  fix. Fact tables now sample from the parent dim per-entity using the
  configured `FKDistribution` instead of always returning row 0. Invisible
  on shipped 1-row reference dims; realism-breaking the moment a user
  expands `dim_plan` or `dim_department`.
- **`hire_date` temporal incoherence** (FIX-05, MF-2) — HR sample's
  `hire_date` could land outside `time_window`. The parameterized Faker
  grammar + bounded `faker.date_between` produces in-window dates; the
  new `validate_temporal_coherence` check catches the class of bug for
  future configs.
- **`validate_null_policy` isinstance tuple** widened to include
  `FakerSource` (discovered during Group 3 via a regression in the
  existing null-policy test).
- **Stale `Metric.default_curve`-era constructor sites** swept across
  tests (no runtime path, just call-site cleanup).

### Performance
- `assign_stages` 35–46× speedup across representative shapes (see
  Changed / FIX-07 above).

### Dependencies
- **`scikit-learn>=1.3`** added to `[dev]` and a new `[test]` optional-
  dependencies group for the FIX-09 distinguishability test suite. Core
  runtime `dependencies` block unchanged — the shipped library keeps
  its numpy / scipy / pandas / pyyaml / pydantic / faker footprint.

### Migration
- Configs relying on `apply_correlations`' silent independent-sample
  fallback (non-PD matrices that previously generated anyway) will now
  raise. Fix is one-shot: tighten the correlation triangle so all
  eigenvalues are strictly positive. The five shipped samples are
  already PD (007a fix).
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
