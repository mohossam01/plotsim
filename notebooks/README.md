# notebooks/

Jupyter notebooks that exercise the bundled-template surface end-to-end.
Companion to the docs site (`docs/`) — these show *what plotsim does* rather
than *how the API works*. Both notebooks are intended to be re-runnable from
a clean checkout (no absolute paths, no machine-specific config) and review-
ready via GitHub or nbviewer rendering without execution.

## What's here

* **`acceptance_test.ipynb`** — phase-by-phase verification of the engine
  pipeline against configured intent. Fixed point: `sample_saas.yaml` at
  seed 42. 17 sections (§0–§15 plus §13.5) covering config parse,
  correlation infrastructure with M111 Higham projection, dimensions,
  trajectories, metric series with `plotsim.inspect.trace_metric_cell`
  end-to-end, fact assembly with the load-bearing 72-cell traceback
  assertion, events, stages, SCD Type 2, bridges (in-memory extension),
  manifest reconciliation, fidelity audit, output stage (quality + holdout,
  in-memory extension), determinism + cross-seed, validation report. Final
  summary code cell tallies pass / pre-registered breach / warn /
  unexpected fail across all sections and asserts `n_unexpected == 0`.

* **`template_qualifier.ipynb`** — parameterized template inspection.
  Compares **baseline** vs **varied** runs to surface whether a config knob
  actually moves the dataset in the direction its name implies. Default:
  saas, vary `noise.gaussian_sigma` (KNOB_VARIATION=2.0 doubles the value).
  v1 implements `gaussian_sigma`; six other knobs
  (`archetype_assignment`, `correlation_magnitude`, `archetype_mix`,
  `entity_count`, `time_window`, `metric_distribution_params`) are
  scaffolded with `NotImplementedError` so the extension surface is
  explicit. Future missions implement additional knobs by replacing the
  TODO bodies in §0's dispatch table.

* **`_helpers.py`** — shared introspection helpers and the eleven
  tolerance constants. All reconstruction logic exceeding 5 lines lives
  here, not in cells.
  * `load_fixed_point() -> (PlotsimConfig, int)` — saas + seed 42
  * `setup_plot_style() -> None` — matplotlib style (called once per nb)
  * `manual_rng_replay(seed, n_draws, distribution, params) -> ndarray` —
    external single-threaded RNG replay; verifies engine RNG-order against
    the pre-correlation `independent_draw` field on `TraceResult`
  * `archetype_curve_eval(archetype, n_periods) -> ndarray` — expected
    trajectory shape with no entity-side stochasticity (used by §4 / §13
    shape recovery)
  * `archetype_color(name) -> Optional[str]` — consistent color per
    archetype across both notebooks (`ARCHETYPE_COLORS` dict)
  * Tolerance constants: see the table below

## How to run

From the repo root:

```bash
pip install -e ".[dev]"        # if not already installed
pip install jupyter matplotlib # notebooks-only deps
jupyter lab notebooks/acceptance_test.ipynb
```

Both notebooks are deterministic — same checkout + same seed → byte-
identical output cells across consecutive runs. Outputs are committed so
the notebooks can be reviewed via GitHub or nbviewer rendering without
execution. The acceptance notebook executes in roughly 6–10 minutes wall
time on a current laptop; the qualifier in roughly 1–2 minutes.

## Tolerance constants

Pass / warn / outlier thresholds defined in `_helpers.py`. Naming: `_PASS`
is the audit-fail threshold; `_WARN` is stricter (surfaces concern without
failing); `_OUTLIER` is looser (catches extreme deviations that pass
quietly because `_PASS` is intentionally relaxed for structural reasons).

| Constant | Value | Direction | Provenance |
|---|---|---|---|
| `MONOTONIC_ARCHETYPE_PEARSON_PASS` | 0.45 | floor | engine-fidelity-check.md §1; mean range 0.640–0.908, min 0.237 |
| `OSCILLATING_ARCHETYPE_PEARSON_PASS` | 0.30 | floor | engine-fidelity-check.md §1; mean range 0.408–0.553 |
| `OSCILLATING_ARCHETYPE_PEARSON_WARN` | 0.60 | floor (stricter) | tighter than pass |
| `MARGINAL_MEAN_REL_PASS` | 0.30 | ceiling | engine-fidelity-check.md §3; 23/31 metrics > 10% |
| `MARGINAL_STD_REL_PASS` | 1.50 | ceiling | engine-fidelity-check.md §3; median +119% |
| `MARGINAL_STD_REL_OUTLIER` | 3.00 | ceiling (looser) | marketing scale-amplified outliers |
| `CORRELATION_DEVIATION_PASS` | 0.50 | ceiling | engine-fidelity-check.md §2; max observed 0.48 |
| `CORRELATION_DEVIATION_WARN` | 0.30 | ceiling (stricter) | tighter than pass |
| `CORRELATION_HIGHAM_DELTA_PASS` | 0.05 | ceiling | M111 + M112 marketing baseline (max 0.023) |
| `DETERMINISM_BYTE_PASS` | 0 | ceiling | byte-identical contract |
| `CHOLESKY_RECONSTRUCTION_ULP_PASS` | 1e-12 | ceiling | theoretical ULP bound |

`_PASS` thresholds fail the audit. `_WARN` and `_OUTLIER` thresholds surface
in the §13 report without failing — both are visibility tools for
known-issue tracking. The operator adjusts these in one place if engine
work tightens any bound.

## What passing looks like

* **`acceptance_test.ipynb`** runs top-to-bottom with no exceptions raised.
  Every numerical assertion passes. The final summary code cell reports
  `UNEXPECTED FAIL: 0`. Pre-registered threshold breaches surface (see
  *Known issues surfaced* below) — they trace to `engine-fidelity-check.md`
  and require operator review, not engine fix.
* **`template_qualifier.ipynb`** runs top-to-bottom with the §3.1
  trajectory-invariance assertion passing, the §3.3
  `_correlation_adjustments` byte-identity assertion passing, and the §4
  directional assertion passing — possibly with `⚠ DIRECTIONAL VARIATION`
  structural flags on bounded-distribution metrics whose realized values
  cluster near the bounds.
* **`tests/test_inspect.py` and `tests/test_notebook_helpers.py`** pass
  under `python -m pytest tests/`.

## Known issues surfaced

These breaches and warnings surface during normal acceptance runs. They
are **pre-registered against `project/research/engine-fidelity-check.md`**
and listed here so reviewers see them once and recognize them on every
re-run.

### Acceptance threshold breaches (3, all pre-registered)

* **§2 Higham projection** — `CORRELATION_HIGHAM_DELTA_PASS = 0.05`
  breached by all 3 saas-correlated pairs (max |Δ| = 0.117 on
  engagement↔churn_risk; engagement↔mrr 0.087; support_tickets↔churn_risk
  0.054). Operator decision: raise the constant to 0.15, accept saas's
  delta as a known engine-level issue, or tighten the engine.
* **§5 / §13 marginal mean** — `MARGINAL_MEAN_REL_PASS = 0.30` breached
  by `support_tickets` Δmean = 0.459 (45.9%). Poisson distribution at
  mid-trajectory; the 30% pass threshold is empirical, the 45.9%
  realization is structural for the configured λ.
* **§13 shape recovery** — `MONOTONIC_ARCHETYPE_PEARSON_PASS = 0.45`
  breached by `globex_cohort/mrr` Pearson = 0.237.

### Acceptance warns (1, pre-registered)

* **§13 correlation deviation** — `engagement ↔ churn_risk` empirical-vs-
  achieved deviation 0.31 lands between `CORRELATION_DEVIATION_WARN = 0.30`
  and `CORRELATION_DEVIATION_PASS = 0.50`. Raw-output Pearson is
  structurally inflated above the Gaussian-space achieved correlation.

### Saas template-coverage caveats

* **Archetype coverage gap.** Saas declares 6 archetypes but only
  exercises 3 (`rocket_then_cliff`, `steady_grower`, `zombie_account`).
  The other three (`slow_death`, `seasonal_spiker`, `expansion_champion`)
  gain coverage via the qualifier notebook — switch `TEMPLATE_NAME` to
  `marketing` or `retail` to bring oscillating archetypes
  (`deal_seeker`, `bargain_hunter`, `holiday_surge`) into scope.
* **Marketing scale-amplified outliers.** `impressions`, `ad_spend`, and
  `average_order_value` show |Δstd| > `MARGINAL_STD_REL_OUTLIER = 3.00`
  in §13 audits when `TEMPLATE_NAME = "marketing"`. Surfaced as outlier
  flags rather than audit failures because the underlying lognormal /
  poisson distributions have wide native scales that amplify trajectory
  variance non-linearly.

### Qualifier directional flags (template-dependent)

* **Bounded-distribution non-rise on `gaussian_sigma`.** On saas at
  `KNOB_VARIATION=2.0`, `feature_adoption` (beta) is flagged with
  `⚠ DIRECTIONAL VARIATION` because realized std slightly *decreases*
  (−0.3%) instead of rising. Mechanism: `gaussian_sigma` applies
  multiplicative jitter `v + N(0, σ·|v|)`, then [`metrics.py:715`](../plotsim/metrics.py#L715)
  clamps to `value_range` AFTER noise. When realized values cluster near
  the bounds, the jitter that pushes them past gets clipped, which can
  flatten or slightly shrink realized std even though noise injection
  itself worked. Structural — not a knob bug. The §4 assertion only
  fails on **unexpected** non-rises (non-bounded distribution + no
  structural reason); structural cases pass with a flag.

## File layout

```
notebooks/
├── README.md                  ← this file
├── _helpers.py                ← shared helpers + tolerance constants
├── acceptance_test.ipynb      ← single-template fixed-point audit
└── template_qualifier.ipynb   ← parameterized knob inspection
```

The acceptance notebook is the primary deliverable — it pins every claim
in the engine-fidelity report to a runnable assertion. The qualifier is
the parameterized companion that complements the saas fixed point
(oscillating archetype coverage, scale-amplified outlier surface,
knob-direction sanity).
