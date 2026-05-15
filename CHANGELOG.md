# Changelog

All notable changes to plotsim are documented here.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning: [SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Heteroscedastic gaussian noise.** Optional `scale_with_trajectory`
  flag on `NoiseConfig` (mirror on the builder's `NoiseInput`). When
  `true`, each cell's gaussian standard deviation becomes
  `gaussian_sigma × trajectory_position` instead of
  `gaussian_sigma × |value|` — position-zero cells receive zero
  gaussian noise, position-one cells receive the full σ. Outlier and
  MCAR rates are unaffected. Default `false` keeps engine output
  byte-identical to the magnitude-scaled lane. Manifest schema bumps
  1.6 → 1.7 with a new optional `noise_config` field populated only
  when the flag is enabled.

- **`pool.<attr>` source on per_entity_per_period facts.** Widens the
  per-entity value-pool surface to the most common fact grain (one
  row per entity per period). Two new dispatch handlers —
  `_fact_scalar_pool` and `_fact_vec_pool` — register against
  `BuilderKind.PER_ENTITY_PER_PERIOD_FACT_{SCALAR,VECTORIZED}` and
  draw uniformly from the row's entity's pool list. Pool sources
  remain rejected on per_period facts (no per-row entity binding),
  reference dims, and sub-entity dims. Pairs naturally with `cdc:
  true` on the same fact, so a column like `payment_type:
  pool.payment_method` now works alongside SCD2 and CDC on a single
  transactional table.

- **Parent/child fact grain + sibling-fact references.** Three
  composable patterns for multi-fact stars:
  - **Header / detail** — a `per_parent_row` child fact fans out
    deterministically from each parent row. The parent FK column on the
    child is auto-synthesized from `parent_table` (bridge precedent:
    user declares the relationship once, engine emits the FK column).
    Models orders + line items, claims + claim lines, etc.
  - **Variable-grain parent** — a `variable`-grain fact whose row count
    is trajectory-driven via a `row_count_driver` metric. Reads the
    driver directly from the metric layer; no intermediate driver-host
    fact required (2-table minimum: parent + child).
  - **Sibling-fact reference** — a second variable-grain fact carries a
    `ref.<other_fact>` column; the engine resolves it via same-entity-
    filtered stochastic draw from the referenced fact's PK column.
    Models orders + returns, orders + selectively-shipped orders, etc.
  Builder vocabulary: `row_count_driver` / `row_count_scale` (variable
  parent or sibling), `parent_table` / `children_per_row` (header /
  detail child). Topological build order across fact dependencies
  (parent_table edges + `fk:fct_*` column edges). Bundled template
  `orders` demonstrates all three patterns (`fct_orders` parent,
  `fct_order_items` child, `fct_returns` sibling). Manifest schema
  bumps 1.5 → 1.6 with a new `parent_child_relations` list (one record
  per declared header / detail edge, carrying actual row counts).
- **Broader feature coverage in bundled domain templates.** The five
  domain templates (`saas`, `hr`, `retail`, `education`, `marketing`)
  now demonstrate audit, quality, treatment, and nested-output
  surfaces that previously lived only in dedicated single-feature
  templates. `saas` opts into denormalized output (`output.denormalized:
  true`) and ships a syslog `log_format` on `evt_login`;
  `retail.fct_purchases` flips `cdc: true` with a paired
  `null_injection` quality block on `cart_value`;
  `retail.dim_product_category` adds a nested `struct` column for
  JSON-flattening exercises; `retail.fct_sessions` carries a 50%
  `volume_anomaly` spike at period 18 for data-observability training;
  `marketing.awareness_builder` adds a 50/50 A/B treatment cohort with
  a 0.5 log-odds lift. All five domain templates gain a top-level
  `quality:` block with two issue types each.
- **`crm_billing_overlap` listed in the bundled-templates index.** The
  multi-source overlap template (shipped previously) was missing from
  the bundled-templates catalog in `docs/site/feature-reference.md` —
  documentation parity catch-up.
- **Section 7 of `feature-reference.md` extended.** "Audit + downstream-
  pipeline outputs" now covers denormalization, the log-file writer,
  multi-source overlap mode, and nested struct / array column types,
  with each row citing its demonstrating bundled template.

### Changed

- **`narrative_reviews` lexicon: cross-segment shared opener pool.**
  Each band's per-segment opener phrases are concatenated with a
  shared cross-segment pool of four band-keyed phrases. Bag-of-words
  classifier accuracy on the held-out entity split drops from 0.90
  (trivially separable) into the 0.65–0.85 range — preserves the
  per-archetype speaking style while introducing controlled overlap
  so the classification challenge isn't a one-line solve.

## [0.6.1] — 2026-05-08

### Added

- **Contributor release automation.** A new `release.yml`
  `workflow_dispatch` runs the full release end-to-end: validates
  source-file versions and the dated `CHANGELOG.md` entry, runs the
  test matrix on Python 3.10–3.13, builds the wheel + sdist, creates
  the annotated tag and GitHub Release, and uploads to PyPI via OIDC
  Trusted Publishing (paused at the `pypi` environment for
  required-reviewer approval). Full process in `RELEASE.md`.
- **Configurable cell-count budget with tiered messaging.** The
  load-time cell-count gate now reads two environment variables:
  `PLOTSIM_CELL_BUDGET=N` raises (or `0` disables) the soft cap that
  defaults to 2,000,000; `PLOTSIM_ALLOW_LARGE_DATASET=1` opts a single
  run into above-soft-budget generation. The `plotsim run`,
  `plotsim validate`, and `plotsim info` commands gain a matching
  `--allow-large-dataset` flag. Output above 500,000 cells now prints
  a stderr advisory recommending `output.format: parquet` and
  `generation_mode: auto`. A new non-configurable hard ceiling at
  50,000,000 cells rejects unreachably-large configs regardless of
  opt-in. The "Limits and performance gates" docs section covers the
  full ladder.
- **Binder and Colab one-click run for tutorial notebooks.** Each of
  the ten tutorial notebooks under `docs/site/tutorial-notebooks/` now
  opens directly in Binder or Colab via badges in the first markdown
  cell. A new `binder/requirements.txt` pre-installs plotsim plus the
  visualization deps the notebooks use (matplotlib, scikit-learn) so
  users can run cells without local setup.

### Changed

- **`generation_mode` default flipped from `"serial"` to `"auto"`.** Configs
  that don't pin `generation_mode` now resolve to vectorized when the
  entity count crosses the auto threshold, serial below. Same `(config,
  seed)` produces statistically equivalent but **byte-different** output
  vs. the previous serial default — the two modes consume RNG in
  different orders. Integration reference fixtures regenerated under the new
  default. Pin `generation_mode: "serial"` explicitly to preserve
  pre-flip bytes.
- **Auto-mode threshold keys on archetype batch size, not total entity
  count.** `_resolve_generation_mode` now selects vectorized when the
  largest single-archetype entity group reaches the threshold (50),
  rather than when total entities reach it. Catches the thin-archetype
  case (e.g. 60 entities × 12 archetypes, avg group size 5) where the
  old heuristic flipped to vectorized and paid setup overhead with no
  per-batch amortization win. Among the bundled templates, retail and
  marketing now resolve to serial under auto (largest groups of 30);
  saas / hr / education stay vectorized.
- **Tutorial notebooks render as static pages on the docs site.** The
  ten notebooks moved from `docs/tutorial-notebooks/` to
  `docs/site/tutorial-notebooks/` and now render via `mkdocs-jupyter`
  with download-source buttons. The `Tutorials` nav was reorganized
  into Getting started / Feature surfaces / Workflows / Use cases.
  The previous standalone `tutorials.md` index page has been removed;
  cross-links from user-guide pages now point at the relevant specific
  notebook.
- **README adds a "See it" output sample plus two diagrams.** A
  trajectory-first plot at the top of "See it"
  (`docs/site/assets/trajectory-first.png`, regenerable from
  `examples/render_trajectory_plot.py`) shows one customer's
  trajectory and the four metrics derived from it, with positive-
  polarity metrics rising and negative-polarity metrics falling as
  the trajectory rises. A Mermaid pipeline diagram at the top of
  "How it works" maps config → validation → trajectory engine →
  per-metric derivation → schema assembly → output. The README
  also gains the side-by-side Faker-style vs. trajectory-correlated
  comparison tables and the full output-folder file inventory so
  readers can see what `plotsim run` produces without leaving the
  README.

### Security

- **`schema` and `template` CLI subcommands sandbox `--output` to the
  cwd by default.** Absolute paths and `..` traversal are rejected
  unless `--allow-absolute-output` is passed explicitly, matching the
  behavior `run` already had. Closes a CWE-22 path-traversal exposure
  on those subcommands surfaced by the post-0.6.0 free-text field
  audit.

## [0.6.0] — 2026-05-04

A reworked public API around a new builder, plus correctness and quality
improvements throughout.

### Added

- **Builder API.** `plotsim.create()` and `plotsim.create_from_yaml()`
  are the new way to author a config. Pass a high-level shape (about,
  unit, time window, metrics, segments, plus optional connections,
  lifecycle, dimensions, facts, events) and get back a fully validated
  config.
- **Template discovery.** `plotsim.list_templates()` returns the names
  of bundled templates; `plotsim.load_template(name)` loads one and
  returns a config you can edit before generating.
- **Two new bundled templates** — `retail` and `marketing`, alongside
  `bare_minimum`, `saas`, `hr`, and `education`.
- **`plotsim.inspect`.** Read each entity's archetype assignment and
  trajectory position from a generated dataset, without re-deriving
  them from the tables.
- **Cross-dimension attribute pools.** `pool:<dim_table>.<column>`
  draws an entity attribute from another dimension table.
- **Per-archetype `value_range` overrides.** Re-scale a metric's
  bounded distribution within a single archetype.
- **Automatic correlation-matrix repair.** Non-positive-definite
  correlation matrices are projected to the nearest valid matrix at
  load instead of being rejected. The original matrix and the
  adjustment are recorded in the manifest for audit.
- **Global seasonality.** A `seasonality` block lets one periodic
  signal modulate every metric (holiday lift, fiscal cycles) without
  declaring oscillation per-metric.
- **Correlation pre-compensation.** When a configured correlation
  conflicts with the trajectory shape an archetype already imposes,
  the engine compensates before sampling. Negative correlations on
  growth-and-decline mixes now recover their target sign instead of
  clamping toward zero.
- **20–35 % faster generation** on the bundled templates.
- **Streaming Parquet writer.** `output.format: parquet` with
  `streaming: true` writes one row group at a time, keeping memory
  bounded on large outputs. Requires `pip install plotsim[parquet]`.
- **Tutorial notebooks.** Eight walkthroughs at
  `docs/tutorial-notebooks/` — one per feature surface.
- **Builder documentation** at
  `docs/builder-{quickstart,reference,errors}.md` covering the full
  vocabulary and a catalog of validation errors.
- **Documentation site** at
  [mohossam01.github.io/plotsim](https://mohossam01.github.io/plotsim/).

### Changed

- **The builder is the documented entry point.** README and
  getting-started docs lead with `create()` / `create_from_yaml()`.
- **Bundled templates now ship inside the installed wheel.**
  `pip install plotsim` followed by `plotsim list-templates` returns
  the full list. Earlier wheels were silently missing templates.
- **`StageSequence.enforce_order` now defaults to `False`.** Stages
  are free-mode by default; set `enforce_order: true` to keep the
  strict-monotonic stage walk.
- **Configured correlations now match observed values within ±0.10**
  for most distribution pairings, ±0.15 for `lognorm × lognorm` at
  high magnitudes. Output values shift for any config that declares
  correlations; same `(config, seed)` is byte-identical within this
  release.
- **Causal lags compose across chains.** A chain like
  `A → B (lag 2) → C (lag 3)` produces a `C` series that reads
  `A`'s trajectory from 5 periods ago.
- **GitHub repository moved to `mohossam01/plotsim`.** Project URLs,
  issue tracker, and docs site updated.

### Fixed

- **`write_tables()` no longer mutates the caller's DataFrame.** Your
  in-memory tables are untouched after writing to disk; column
  references survive the call.
- **In-memory tables and on-disk CSVs now report the same dtype** for
  `dtype: int` and `dtype: boolean` columns. Previously the in-memory
  tables saw float values where the on-disk file had been corrected.
- **`validation_report.txt` is byte-identical across runs** with the
  same config and seed. The wall-clock stamp that previously broke
  byte equality was removed from the library default; the CLI still
  emits one for operators.
- **Threshold events on a sub-entity dimension now distribute their
  FKs across candidate sub-entities** instead of always picking the
  first one.
- **Bridge tables are auto-resolved.** When a config references a
  bridge into a `dim_{unit}` or `dim_date` table that wasn't declared
  explicitly, the builder adds it automatically.
- **Duplicate `correlations` entries raise** with a clear error
  instead of silently keeping whichever entry happened to land last.
  The pair is treated as unordered, so `(a, b)` and `(b, a)` collide.
- **`StageSequence.threshold_exit` is wired in.** When
  `threshold_exit ≤ threshold_enter`, the value acts as a hysteresis
  lower bound for demoting back to the previous stage. Previously
  the field was accepted but had no effect.
- **Malformed dates in `static:` sources reject at config load** with
  a clear error, instead of silently leaving a string in a date-typed
  column.
- **`Entity.overrides` only accepts known keys.** A typo of
  `inflection_month`, for example, now raises at load instead of
  loading silently and being ignored.
- **`dtype: boolean` paired with a `metric:` or `lag:` source rejects
  at load.** The resulting column was effectively always-True for
  any distribution producing positive values. Use `threshold:` if a
  boolean indicator was intended.
- **`causal_lag.lag_periods` cap is now granularity-aware** — 120
  for monthly, 520 for weekly, 3650 for daily (each ≈ 10 years).
  Daily configs that wanted a multi-month lag previously hit the
  monthly-scaled cap.
- **Bridge cardinality validator no longer warns spuriously** on
  bridges into per-entity dimensions when entities have `size > 1`.

### Removed

- **`ecommerce` and `healthcare` templates** are gone. Use `retail`
  in place of `ecommerce`; a healthcare template is planned for a
  future release.

### Migration

- **Configs with `stages` that relied on the old strict-monotonic
  default** must add `enforce_order: true` explicitly.
- **`dtype: boolean` on a `metric:` or `lag:` column** now rejects.
  Switch to `dtype: float` (continuous) or `dtype: int` (count), or
  use `threshold:` if a boolean indicator was intended.
- **Unknown keys in `Entity.overrides`** now raise at load. Remove
  any typos or non-supported keys; only `inflection_month` is
  accepted.

## [0.4.0] — 2026-04-23

- Configured correlations and causal lags now match observed values
  in generated output.
- New per-lag `CausalLag.blend_weight` controls how a lag blends the
  current trajectory with the driver's past (default `1.0` = pure
  shift).
- 35–46× faster stage assignment; faster fact generation overall.
- Configs above new resource caps (entities, metrics, time-window
  span, total cells) now reject at load with a clear error.
- CLI default writes to the current working directory; absolute
  output paths require `--allow-absolute-output`.

## [0.3.0] — 2026-04-22

- Parameterized Faker grammar — e.g.,
  `generated:faker.date_between:start:2022-01-01:end:2024-12-31`.
- `PlotsimConfig.locale` for non-English Faker output.
- Cross-dimension FKs draw per-entity from a parent dim with explicit
  uniform / weighted / fixed distributions instead of always
  collapsing to row 0.
- `StageSequence.downgrade_delay` relaxes strict-monotonic stage
  progression after N consecutive lower-stage periods.
- `write_tables(base_dir=...)` rejects absolute paths and `..`
  traversal.

## [0.2.0] — 2026-04-22

- `archetypes[].metric_overrides` now actually affects sampling
  (the field was previously accepted but silently ignored).
- `py.typed` marker — mypy / pyright now recognize plotsim as typed.
- Several dead schema fields removed; configs setting them now
  reject at load.

## [0.1.0] — 2026-04

Initial public release on PyPI.

- Trajectory-first multi-table generator driven by behavioral
  archetypes.
- Five bundled templates (saas, hr, ecommerce, education,
  healthcare).
- Six distributions (`lognorm`, `gamma`, `poisson`, `beta`,
  `normal`, `weibull`) and eight curve shapes.
- CLI: `run`, `validate`, `info`, `list-templates`, `template`.
