# Changelog

All notable changes to plotsim are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
