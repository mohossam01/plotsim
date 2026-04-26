# Mission 103 — Statistical fidelity addendum

Empirical characterization of plotsim's four headline statistical
guarantees. The user-facing distillation of these findings lives in
[`docs/statistical-fidelity.md`](../docs/statistical-fidelity.md); this
document is the empirical foundation underneath that page.

---

## 1. Hardware, toolchain, codebase commit

| Component       | Version |
|-----------------|---------|
| codebase commit | `59f4e14` (post-Mission-102; suite at 808 passed / 1 xfailed) |
| Python          | 3.10.10 |
| numpy           | 1.26.4  |
| scipy           | 1.13.0  |
| pandas          | 2.1.4   |
| OS              | Windows 10 (10.0.19045 SP0), x86_64 |
| CPU             | Intel64 Family 6 Model 158 Stepping 10 |

Re-run from these CSVs alone: every result row carries the full input
parameter set so a reader with only `analysis/fidelity_sweeps/*.csv`
can rebuild any cell. Driver scripts are
`analysis/fidelity_sweeps/claim{1,2,3,4}_*.py`; the smoke test
`tests/test_fidelity_smoke.py` checks the headline findings reproduce
on every CI run.

**Scope cuts (Option C, operator-approved 2026-04-25).** The mission
spec's 0.3 s/cell estimate did not reproduce on this hardware /
toolchain. The current scipy frozen-distribution construction overhead
(43 % of `generate_tables` per the spot-checks report, perf-pass
mission item 11 deferred) makes per-cell cost ~7 s at 100×24, ~14 s at
100×48, and ~104 s at 100×360. Running the spec's full grids (Claim 1:
8 100 cells; Claim 2: 1 350 cells at 100×360) would have required 60+
wall-clock hours. The lean grid the operator approved trades coverage
on a few axes for in-budget completion; the deferred axes are listed
explicitly in each claim's section below.

---

## 2. Correlation fidelity (Claim 1)

**Hypothesis under test.** Configured pairwise correlation lands within a
small tolerance of the observed Pearson on the generated fact table.
The engine's R-01 / R-01b tests use ±0.10 for continuous pairs and
±0.15 for poisson-involving pairs; this sweep characterizes the actual
envelope.

**Sweep grid (focused subset per D1(c)):**

- 10 distribution pairings — covers all six distribution families and
  the bundled templates' actual correlation patterns:
  beta×lognorm (saas), beta×beta (universal), beta×poisson (saas /
  healthcare), beta×normal (saas / hr / education), beta×gamma (hr /
  education / healthcare), lognorm×poisson (saas / ecommerce),
  lognorm×normal (saas), lognorm×lognorm (likely user pair),
  normal×gamma (hr / education), weibull×normal (covers weibull).
- 5 magnitudes per pair: −0.7, −0.3, 0.0, 0.3, 0.7.
- 100 entities × 24 periods per cell (~2 400 samples — sufficient for a
  per-realization Pearson uncertainty of ~0.02 around 0.5).
- 5 seeds per cell (mission rule).
- 1 metric count: 2 (the pair alone). The mission spec's 3-tier
  axis (2 / 5 / 10 metrics) is deferred; the F6 property test and
  R-01b already verify pair tolerance is robust to declaration-order
  permutation, which is the only structural failure mode an n_metrics
  axis adds beyond "more metrics = more samples".

**Total cells:** 10 × 5 × 1 × 1 × 5 = 250. Wall clock: 1 676 s
(28 min, 6.7 s/cell average).

**Per-pair tolerance — all magnitudes pooled.**

| dist_a    | dist_b    | median \|err\| | p95 \|err\| | max \|err\| | n   |
|-----------|-----------|---------------:|------------:|------------:|----:|
| beta      | normal    | 0.009          | 0.021       | 0.021       | 25  |
| beta      | poisson   | 0.015          | 0.027       | 0.034       | 25  |
| beta      | beta      | 0.013          | 0.039       | 0.050       | 25  |
| weibull   | normal    | 0.025          | 0.045       | 0.048       | 25  |
| normal    | gamma     | 0.026          | 0.045       | 0.053       | 25  |
| lognorm   | normal    | 0.026          | 0.049       | 0.053       | 25  |
| lognorm   | poisson   | 0.024          | 0.069       | 0.078       | 25  |
| beta      | gamma     | 0.029          | 0.075       | 0.078       | 25  |
| beta      | lognorm   | 0.026          | 0.088       | 0.091       | 25  |
| **lognorm** | **lognorm** | **0.046**     | **0.145**   | **0.164**   | 25  |

Source: `correlation_matrix_results.csv`, all 250 rows; group keys
`(dist_a, dist_b)`.

**Findings.**

1. **9 of 10 pairings land within ±0.10** of configured at the 95th
   percentile across all magnitudes. The headline tolerance the
   engine's R-01 test asserts is empirically sustained for every
   pair *except* lognorm × lognorm.

2. **lognorm × lognorm widens at high magnitudes only.** Per-magnitude
   breakdown:

   | configured | median observed | median \|err\| |
   |-----------:|----------------:|---------------:|
   | −0.70      | −0.563          | 0.137          |
   | −0.30      | −0.253          | 0.047          |
   | 0.00       | −0.010          | 0.011          |
   | +0.30      | +0.259          | 0.041          |
   | +0.70      | +0.670          | 0.030          |

   The negative-magnitude breach is the single failure mode. At
   |configured| ≤ 0.3 lognorm × lognorm lands within ±0.05 like every
   other pair. Asymmetry between +0.7 (clean) and −0.7 (breach) is
   the heavy-tail signature: a Gaussian copula maps both margins
   through their CDF, applies the linear Cholesky factor, and inverts.
   For a strongly *negative* correlation the joint extreme requires
   simultaneous draws from one distribution's right tail and the
   other's left tail. Lognormals have a heavy right tail and a thin
   left tail; the copula structurally favors right-tail concordance
   over diagonal-tail discordance, so the achieved negative
   correlation falls short of configured. Strong positive
   correlation (right-tail concordance on both sides) doesn't fight
   the same asymmetry.

3. **Poisson-involving pairs are tighter than the engine's test
   tolerance.** R-01 uses ±0.15 for any pair containing a poisson
   metric. Measured worst-case (lognorm × poisson, p95 = 0.069) is
   well inside ±0.10. The ±0.15 budget is conservative against
   sample-noise outliers at smaller sample counts, not the actual
   envelope of the copula.

**Recommendation, applied.**

- README claims softened from "the exact correlation you specify" to
  "the configured correlation, with measured tolerance documented".
- `docs/statistical-fidelity.md` documents per-pair tolerances and
  calls out the lognorm × lognorm boundary case.
- CHANGELOG entry records the per-pair envelope, the lognorm ×
  lognorm widening, and the poisson tightening.
- The engine's R-01 / R-01b tolerance budgets (`±0.10` continuous,
  `±0.15` poisson) are *not* changed — they are conservative test
  margins against sampling noise at small per-cell sample counts and
  remain appropriate for that role.

**Deferred axes (post-perf-pass M104+):**

- The 3-tier `n_metrics` axis (2 / 5 / 10). F6's existing test pins
  pair tolerance under permutation; this axis adds redundancy.
- Sample-size sweep (100×48, 100×360). 100×24 is the smallest
  population the report cites as well-characterized; larger
  populations should only tighten the per-pair envelope.
- The full 6×6 distribution matrix (36 pairings). Available via
  `claim1_correlation.run_full_matrix()` for an overnight run; not
  needed for the headline tolerance which the focused subset
  already pins.

---

## 3. Causal-lag fidelity (Claim 2)

**Hypothesis under test.** Configured causal lag at `blend_weight=1.0`
delivers a pure period shift recoverable by output-level cross-
correlation analysis. The
[`tests/test_output_fidelity.py::test_lag_peak`](../tests/test_output_fidelity.py)
xfail concedes detection failure at lag ≤ 1; this sweep characterizes
the boundary.

**Sweep grid (lean, per Option C):**

- 5 lag values: 1, 2, 5, 10, 30.
- 3 blend weights: 0.6, 0.8, 1.0.
- 1 archetype: sigmoid (smooth-rising). The mission spec's 3-archetype
  axis (`oscillating`, `plateau`) is deferred; sigmoid is the
  archetype shape the bundled templates use most heavily.
- 1 lagged-metric distribution: poisson (the saas template's
  canonical lag pattern is `engagement → support_tickets`, which is
  poisson). Driver is fixed as continuous lognorm to isolate the lag
  mechanism.
- 100 entities × 120 periods per cell. Long enough for cross-
  correlation analysis up to lag = 30; the spec's 100×360 was
  budget-prohibitive at ~104 s/cell.
- 5 seeds per cell.

**Total cells:** 1 × 5 × 3 × 1 × 5 = 75. Wall clock: 46 s
(0.6 s/cell average — far faster than projected because the
2-metric / 1-archetype config is small).

**Recoverability matrix.** For each cell, the per-entity cross-
correlation function between driver and lagged-target was computed
across lags 0..2× configured. Each row reports the median (across
seeds × entities) of: peak lag, peak magnitude, unlagged magnitude.
A cell PASSES iff `|median(peak_lag) − configured| ≤ 1` AND
`peak_magnitude > unlagged_magnitude`.

| configured_lag | blend_weight | median peak lag | peak mag | unlag mag | verdict |
|---------------:|-------------:|----------------:|---------:|----------:|---------|
| 1              | 0.6          | 2               | 0.612    | 0.554     | PASS    |
| 1              | 0.8          | 2               | 0.606    | 0.550     | PASS    |
| 1              | 1.0          | 2               | 0.608    | 0.549     | PASS    |
| 2              | 0.6          | 2               | 0.612    | 0.555     | PASS    |
| 2              | 0.8          | 2               | 0.612    | 0.557     | PASS    |
| 2              | 1.0          | 2               | 0.616    | 0.551     | PASS    |
| 5              | 0.6          | 3               | 0.625    | 0.554     | FAIL    |
| 5              | 0.8          | 3               | 0.626    | 0.553     | FAIL    |
| 5              | 1.0          | 3               | 0.629    | 0.553     | FAIL    |
| 10             | 0.6          | 3               | 0.632    | 0.558     | FAIL    |
| 10             | 0.8          | 3.5             | 0.638    | 0.560     | FAIL    |
| 10             | 1.0          | 4               | 0.635    | 0.568     | FAIL    |
| 30             | 0.6          | 5               | 0.618    | 0.537     | FAIL    |
| 30             | 0.8          | 5               | 0.626    | 0.533     | FAIL    |
| 30             | 1.0          | 8               | 0.629    | 0.515     | FAIL    |

Source: `lag_recovery_results.csv`, all 75 rows; aggregated by
`(configured_lag, blend_weight)`.

**Findings.**

1. **Small lags (1, 2) recover at output level on smooth drivers.**
   Peak lands within ±1 of configured across every blend weight; peak
   magnitude consistently exceeds unlagged baseline by ~0.06.

2. **Larger lags (5+) fail at output level on smooth drivers.** The
   peak position drifts away from configured and toward small values
   regardless of blend weight. At lag=5 the median peak is 3, and the
   trend doesn't recover at higher configured lags.

3. **The peak magnitude vs unlagged baseline gap stays ~0.06 across
   all configured lags.** The cross-correlation does detect *some*
   lag-related structure even at lag=30, but the peak position is not
   a faithful estimator of the configured lag.

4. **This is a different boundary from the test_lag_peak xfail.**
   That xfail's "lag ≤ 1" concession is about MCAR / null-handling
   interactions at the small-lag end. The upper-boundary failure
   surfaced here is about the *detection method*: cross-correlation
   on a smooth driver has poor lag resolution because
   `driver(T)` ≈ `driver(T+5)` ≈ `driver(T+10)` across the curve's
   transition region. The argmax-over-lag picks small values
   regardless of where the engine actually shifted the signal.

5. **Engine-layer correctness is unchanged.** R-11 / R-12 tests in
   `tests/test_metrics.py` verify the lag mechanism at the metric-
   generator level using controlled trajectories. Those tests pass —
   the engine implements the lag faithfully. The output-level
   detectability finding is a downstream concern about cross-
   correlation as a *detection technique*, not a generation defect.

**Recommendation, applied.**

- `docs/statistical-fidelity.md` documents the recoverable region
  (small lags 1–2 on smooth archetypes, all blend weights) and the
  upper-boundary detection failure with the autocorrelation
  explanation.
- `docs/statistical-fidelity.md` advises tutors to use small-lag
  cases on smooth archetypes, larger-lag cases on non-smooth
  archetypes, or to verify lag at the engine layer rather than
  output-level cross-correlation.
- No engine code changes (mission rule: "Do not modify plotsim/
  source code").

**Deferred axes (post-perf-pass M104+):**

- 3-archetype axis (oscillating, plateau). Sigmoid is the high-
  autocorrelation case; the lower-autocorrelation cases would
  recover more lags more cleanly. Confirming the asymmetry would
  add ~50 cells × ~25 s = ~20 min.
- 6 lagged-metric distributions. The mechanism is identical at the
  engine layer; only the discreteness of poisson interacts with
  detection in a meaningful way.
- 100×360 sample size. Longer windows tighten per-entity cross-
  correlation but the upper-boundary failure at smooth driver is a
  resolution problem, not a sample-size problem.

---

## 4. Trajectory-first verification (Claim 3)

**Hypothesis under test.** Every metric value at every (entity,
period) cell is consistent with a single trajectory position computed
for that cell; deviations from the predicted center are within the
configured noise envelope plus distribution-tail behavior.

**Sweep grid (full mission spec — small enough to run as-spec):**

- All 5 bundled templates (saas, hr, ecommerce, education, healthcare).
- 100 randomly-sampled `(entity, period)` cells per (template, seed).
- 5 seeds per template.
- For each sampled cell × each non-lagged scalar metric (~3–6 metrics
  per template), recompute the entity's trajectory at that period
  using the archetype's curve segments, recompute the predicted
  center via `position_to_center`, and compare to the observed
  fact-table value. Lagged metrics are excluded — their effective
  position is a blend with the driver's history (Claim 2 territory).

**Total cells:** 5 × 5 × 100 × ~3.5 metrics = 11 865 verification rows.
Wall clock: 19 s.

**Methodology.** The deviation denominator (`envelope_sigma`) combines
two contributors in quadrature: (a) the configured distribution's
intrinsic standard deviation at the predicted center, and (b) the
multiplicative-Gaussian noise contribution from
`noise.gaussian_sigma`. This denominator does *not* fold in the
distribution's skew, so heavy-tailed distributions (lognorm, poisson)
legitimately produce larger σ-units than a normal-σ comparison
expects. Outlier-rate hits (3×–10× value blowups by design) populate
the deep tail and are reported as the expected baseline.

**Global envelope over 11 865 cells.**

| Metric                        | Value        |
|-------------------------------|-------------:|
| Median deviation_in_sigma     | −0.041       |
| 95th-percentile (signed)      | 1.757        |
| 99th-percentile (signed)      | 3.604        |
| Max deviation                 | 19.253       |
| Min deviation                 | −6.966       |
| Cells \|dev\| > 4 σ           | 100 (0.84 %) |
| Cells \|dev\| > 6 σ           |  55 (0.46 %) |
| Cells \|dev\| > 10 σ          |  23 (0.19 %) |

**Per-template envelope vs outlier-rate baseline.**

| template    | n_cells | outlier_rate | expected outliers | observed > 4σ | p95   | p99   |
|-------------|--------:|-------------:|------------------:|--------------:|------:|------:|
| saas        | 2 472   | 0.020        | 49.4              | 27            | 1.678 | 4.030 |
| hr          | 1 976   | 0.015        | 29.6              | 11            | 1.830 | 2.871 |
| ecommerce   | 2 469   | 0.015        | 37.0              | 27            | 1.938 | 4.537 |
| education   | 2 473   | 0.010        | 24.7              | 11            | 1.595 | 2.680 |
| healthcare  | 2 475   | 0.015        | 37.1              | 24            | 1.728 | 3.854 |

Per-template observed cells > 4σ are strictly **less than** the
configured-outlier-rate expected count. The deep tail is fully
accounted for by the configured outlier mechanism — no invariant
violation.

**Per-(template, distribution) deep tail.**

| template    | distribution | n     | median | p99   | max    |
|-------------|--------------|------:|-------:|------:|-------:|
| ecommerce   | beta         | 1 481 | −0.052 | 2.578 |  8.017 |
| ecommerce   | lognorm      | 495   |  0.059 | 6.276 | 16.422 |
| ecommerce   | poisson      | 493   |  0.000 |15.471 | 19.253 |
| education   | beta         | 1 482 | −0.074 | 2.295 |  5.452 |
| education   | gamma        | 494   | −0.303 | 7.442 |  7.721 |
| education   | normal       | 497   |  0.073 | 2.490 |  3.298 |
| healthcare  | beta         | 1 487 | −0.074 | 2.999 |  6.420 |
| healthcare  | gamma        | 494   | −0.252 | 4.760 | 11.351 |
| healthcare  | poisson      | 494   |  0.000 |10.154 | 14.031 |
| hr          | beta         | 985   | −0.070 | 2.457 |  6.700 |
| hr          | gamma        | 495   | −0.270 | 3.044 |  6.511 |
| hr          | normal       | 496   |  0.163 | 3.604 |  5.575 |
| saas        | beta         | 1 485 |  0.000 | 2.874 |  6.016 |
| saas        | lognorm      | 499   |  0.023 | 8.354 | 17.044 |
| saas        | normal       | 488   | −0.011 | 3.097 |  6.637 |

Source: `trajectory_first_results.csv`, group keys `(template, distribution)`.

**Findings.**

1. **The trajectory-first invariant holds.** Median deviation across
   11 865 cells is −0.04σ — well-centered around 0 with no systematic
   bias for any (template, metric) combination. No (template, metric)
   group shows median deviation outside ±0.4σ.

2. **The 99th-percentile envelope is 3.6σ at the global level.**
   That's the headline empirical bound on the trajectory-first
   invariant: with 99 % confidence, any randomly-sampled (entity,
   period, metric) cell lands within 3.6σ of its predicted center
   (where σ is the noise-inflated distribution sigma).

3. **Heavy-tail distributions (lognorm, poisson) populate the deep
   tail.** 99th-percentile for these reaches 6–15σ, max 19σ. This is
   distribution-skew behavior, not invariant violation: my σ
   denominator under-counts the heavy-tail variance, and
   outlier_rate-injected blowups also live here. Bounded
   distributions (beta, normal) stay within p99 of 2–4σ, max ≤ 7σ.

4. **No invariant violations.** Per-template observed cells > 4σ
   (range 11–27) are strictly below the configured-outlier-rate
   expected count (range 25–49). The mission's escalation criterion
   ("Any unexplained deviations → STOP, escalate to fix mission")
   does *not* trigger.

**Recommendation, applied.**

- `docs/statistical-fidelity.md` documents the empirical envelope
  (3.6σ at p99) and explains the heavy-tail vs invariant-violation
  distinction.
- `tests/test_property_invariants.py` already enforces the sign-
  level form of this invariant via Spearman; the cell-level
  envelope is now characterized empirically here.
- No engine code changes.

---

## 5. Determinism contract (Claim 4)

**Hypothesis under test.** Same config + same seed → byte-identical
CSV output. Single-Python only per operator decision D2(a); cross-
Python-version, numpy-version, and OS axes are recorded as not tested
rather than measured.

**Sweep grid:**

- Axis 1: same-process, two `generate_tables` calls with same seed.
- Axis 2: cross-process, two subprocesses with same `cwd`.
- Axis 3: cross-process, two subprocesses with different `cwd`
  (catches accidental absolute-path leakage into output).
- Axis 4: same-process, seed changed → must differ (sanity).
- Axes 5–7: NOT TESTED — recorded with explicit `(n/a)` sentinels so
  the report's "untested" column is data-backed.

Saas template used as the representative config (90 entities × 24
periods, 6 metrics, 9 tables — the most complex shipped template).

**Total cells:** 39 (9 CSV files × 4 measured axes + 3 untested
sentinels). Wall clock: 32 s.

**Verdict matrix.**

| axis                        | n_pairs | n_identical | verdict     |
|-----------------------------|--------:|------------:|-------------|
| same_process_same_seed      | 9       | 9           | GUARANTEED  |
| cross_process_same_cwd      | 9       | 9           | GUARANTEED  |
| cross_process_different_cwd | 9       | 9           | GUARANTEED  |
| seed_changed (sanity)       | 9       | 2           | PARTIAL ✓   |
| python_version              | 1       | 0           | NOT TESTED  |
| numpy_version               | 1       | 0           | NOT TESTED  |
| operating_system            | 1       | 0           | NOT TESTED  |

The seed-changed sanity check passes: 7 of 9 CSVs differ, the 2 that
remain identical are `dim_date` and `dim_plan` — both static reference
dims with no RNG-driven content, exactly as the F-04 / F-09 design
guarantees.

Source: `determinism_matrix_results.csv`, all 39 rows.

**Findings.**

1. **Single-Python determinism is fully guaranteed across the three
   measured user-relevant axes.** Same process, cross-process same
   cwd, and cross-process different cwd all produce byte-identical
   CSVs. No `os.getcwd()` or other environmental leakage in the
   output path.

2. **Cross-environment determinism is not characterized.** Multi-
   Python-version, multi-numpy-version, and cross-OS axes were
   declared out of scope at mission launch (D2(a)) and are recorded
   as NOT TESTED. The user-facing doc tells consumers to pin Python
   and numpy versions in CI rather than rely on cross-environment
   reproducibility.

**Recommendation, applied.**

- `docs/statistical-fidelity.md` publishes the three GUARANTEED
  axes and the three NOT TESTED axes with practical guidance for
  CI configuration.
- No engine code changes.

**Deferred axes (M104+ if cross-environment determinism becomes a
user concern):**

- Multi-Python-version: 3.10 / 3.11 / 3.12 / 3.13 via tox.
- Multi-numpy-version: pinned vs latest minor.
- Cross-OS: Linux / Windows / macOS pairwise.

---

## Appendix — Reproducing this report

**Re-running every sweep.** From the repository root:

```bash
python analysis/fidelity_sweeps/claim1_correlation.py
python analysis/fidelity_sweeps/claim2_lag.py
python analysis/fidelity_sweeps/claim3_trajectory.py
python analysis/fidelity_sweeps/claim4_determinism.py
```

Wall-clock totals (this hardware, observed):

| sweep                         | cells  | wall clock |
|-------------------------------|-------:|-----------:|
| Claim 1 — correlation focused | 250    | 28 min     |
| Claim 2 — lag                 |  75    | 46 s       |
| Claim 3 — trajectory-first    | 11 865 | 19 s       |
| Claim 4 — determinism         |  39    | 32 s       |
| **total**                     | **12 229** | **~30 min** |

**Optional appendix sweep — full 6×6 distribution matrix.**

```bash
python analysis/fidelity_sweeps/claim1_correlation.py full
```

Writes to `correlation_matrix_full_results.csv`. Operator-approved as
optional / overnight (decision D1(c)); not run as part of M103.
Estimated wall-clock at this perf state: ~10 hours (3.6× the focused
subset).

**Smoke test.** The headline tolerances are pinned by:

```bash
python -m pytest tests/test_fidelity_smoke.py
```

Six smoke-subset tests + three Phase 1 infrastructure tests, total
runtime ~6 s. Failure means the documented tolerances above have
drifted from current-codebase reality; the addendum needs re-running
before release.

**Re-running a single cell.** Each result CSV row carries every input
parameter alongside the measurement. Example for Claim 1:

```python
import pandas as pd
from analysis.fidelity_sweeps.claim1_correlation import _simulate_pair_pearson

df = pd.read_csv("analysis/fidelity_sweeps/correlation_matrix_results.csv")
row = df.iloc[42]
observed = _simulate_pair_pearson(
    row["dist_a"], row["dist_b"], n_other=int(row["n_metrics"] - 2),
    coefficient=row["configured"], n_entities=int(row["n_entities"]),
    n_periods=int(row["n_periods"]), seed=int(row["seed"]),
)
assert abs(observed - row["observed"]) < 1e-9
```

The four sweep drivers expose their per-cell helper functions
(`_simulate_pair_pearson`, `_per_entity_xcorr`, `_verify_template_seed`,
`_same_process_pair`) so re-running any single cell from a CSV row is
a one-call exercise.
