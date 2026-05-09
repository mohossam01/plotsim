"""Claim 1 — correlation fidelity.

Per D1(c) operator decision: a focused subset of 10 pairings produces the
headline numbers, with the full 6x6 (36 pairings) reserved as an optional
appendix sweep that the operator can launch overnight via run_full_matrix().

Methodology mirrors test_metrics.py::_simulate_correlated_pair (the canonical
R-01 fixture). Use a flat plateau trajectory (constant 0.5) so the
trajectory signal is held fixed and the only correlation source is the
configured Cholesky-injection mechanism. Generate via
plotsim.metrics.generate_entity_metrics (not the full generate_tables) so
the per-cell cost stays sub-second; this path is what the engine actually
runs for fact-table population.

Each cell measures the Pearson correlation of the configured pair across
n_entities * n_periods samples. The result CSV row carries every input
parameter alongside the observed value, so a reader with only the CSV can
reconstruct any cell.

Out of scope here (reported but not measured):
- Noise envelope effects (gaussian_sigma > 0). Configured noise is 0 in
  these cells so the measurement isolates the copula. This matches R-01's
  methodology and lets the report cite a clean tolerance for the copula
  itself; a separate sentence in statistical-fidelity.md notes that
  configured noise widens the observed Pearson tolerance.
- Causal lag interactions. Pair metrics never have causal_lag here.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from plotsim.config import (
    CorrelationPair,
    Metric,
    NoiseConfig,
    ValueRange,
)
from plotsim.metrics import generate_entity_metrics


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_CSV = REPO_ROOT / "analysis" / "fidelity_sweeps" / "correlation_matrix_results.csv"
APPENDIX_CSV = REPO_ROOT / "analysis" / "fidelity_sweeps" / "correlation_matrix_full_results.csv"


# --- Per-distribution params + value_range patterns -------------------------
# Each entry returns a (params_dict, ValueRange|None) calibrated so the
# distribution's center sits in a stable, non-degenerate range. Values are
# chosen to match patterns the bundled templates use so the measurement
# tolerance generalizes to user configs.


def _params_for(dist: str) -> tuple[dict[str, Any], ValueRange | None]:
    if dist == "lognorm":
        return {"s": 0.5, "loc": 0.0, "scale": 1.0}, None
    if dist == "gamma":
        return {"shape": 2.0, "scale": 1.0}, None
    if dist == "poisson":
        return {"lambda": 5.0}, None
    if dist == "beta":
        return {"alpha": 2.0, "beta": 5.0}, ValueRange(min=0.0, max=1.0)
    if dist == "normal":
        return {"mu": 100.0, "sigma": 10.0}, None
    if dist == "weibull":
        return {"shape": 1.5, "scale": 1.0}, None
    raise ValueError(f"unsupported distribution {dist!r}")


def _make_metric(name: str, dist: str) -> Metric:
    params, vr = _params_for(dist)
    return Metric(
        name=name,
        label=name,
        distribution=dist,
        params=params,
        polarity="positive",
        value_range=vr,
    )


# --- Focused subset and full matrix definitions ----------------------------

# Focused subset (10 pairs): bundled-template pairings + likely-user pairings,
# covering all 6 distribution families at least once.
FOCUSED_PAIRS: list[tuple[str, str]] = [
    ("beta", "lognorm"),  # saas: engagement x mrr
    ("beta", "beta"),  # universal: engagement x churn_risk
    ("beta", "poisson"),  # saas/healthcare: bounded x discrete
    ("beta", "normal"),  # saas/hr/education: engagement x rate
    ("beta", "gamma"),  # hr/education/healthcare: engagement x duration
    ("lognorm", "poisson"),  # saas/ecommerce: continuous x discrete
    ("lognorm", "normal"),  # saas: revenue x rate
    ("lognorm", "lognorm"),  # likely user: revenue x cost
    ("normal", "gamma"),  # hr/education
    ("weibull", "normal"),  # covers weibull (no template uses it)
]

DISTRIBUTIONS = ["lognorm", "gamma", "poisson", "beta", "normal", "weibull"]

MAGNITUDES = [-0.7, -0.3, 0.0, 0.3, 0.7]
# Option C lean grid (operator-approved 2026-04-25): n_metrics axis collapsed
# to the canonical pair-only baseline, sample-size axis collapsed to (100, 24).
# Full axes deferred to a post-perf-pass M104+ when the per-cell cost no longer
# blows the session budget (currently ~7 s/cell at 100×24 vs the mission spec's
# 0.3 s assumption — see project/notes/spot-checks-report.md fix item 11).
N_METRIC_OPTIONS = [2]
SAMPLE_SIZE_OPTIONS = [(100, 24)]
SEEDS_PER_CELL = 5
SEED_BASE = 7_000


def _simulate_pair_pearson(
    pair_a: str,
    pair_b: str,
    n_other: int,
    coefficient: float,
    n_entities: int,
    n_periods: int,
    seed: int,
) -> float:
    """Generate the configured pair (plus n_other uncorrelated metrics) and
    return observed Pearson on (m_a, m_b).

    The pair is always named (m_a, m_b). The n_other other metrics are named
    m_o0, m_o1, ... and cycle through the 6 distributions so the larger
    configurations exercise broader Cholesky matrices.
    """
    metrics: list[Metric] = [
        _make_metric("m_a", pair_a),
        _make_metric("m_b", pair_b),
    ]
    for k in range(n_other):
        dist = DISTRIBUTIONS[k % len(DISTRIBUTIONS)]
        metrics.append(_make_metric(f"m_o{k}", dist))

    correlations: list[CorrelationPair] = [
        CorrelationPair(metric_a="m_a", metric_b="m_b", coefficient=coefficient),
    ]
    rng = np.random.default_rng(seed)

    a_pool: list[float] = []
    b_pool: list[float] = []
    for _ in range(n_entities):
        out = generate_entity_metrics(
            trajectory=np.full(n_periods, 0.5),
            metrics=metrics,
            correlations=correlations,
            noise=NoiseConfig(gaussian_sigma=0.0, outlier_rate=0.0, mcar_rate=0.0),
            rng=rng,
        )
        a_pool.extend(float(x) for x in out["m_a"])
        b_pool.extend(float(x) for x in out["m_b"])

    arr_a = np.asarray(a_pool, dtype=float)
    arr_b = np.asarray(b_pool, dtype=float)
    if np.std(arr_a) == 0.0 or np.std(arr_b) == 0.0:
        return float("nan")
    return float(np.corrcoef(arr_a, arr_b)[0, 1])


def _build_cells(pairs: list[tuple[str, str]]) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for a, b in pairs:
        for mag in MAGNITUDES:
            for n_metrics in N_METRIC_OPTIONS:
                for n_entities, n_periods in SAMPLE_SIZE_OPTIONS:
                    for seed_offset in range(SEEDS_PER_CELL):
                        cells.append(
                            {
                                "dist_a": a,
                                "dist_b": b,
                                "configured": float(mag),
                                "n_metrics": int(n_metrics),
                                "n_entities": int(n_entities),
                                "n_periods": int(n_periods),
                                "seed": SEED_BASE + seed_offset,
                            }
                        )
    return cells


def run_correlation_sweep(
    pairs: list[tuple[str, str]],
    out_csv: Path,
    *,
    flush_every: int = 100,
    label: str = "claim1",
) -> int:
    """Drive the cells, write rows incrementally."""
    cells = _build_cells(pairs)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "configured",
        "dist_a",
        "dist_b",
        "n_entities",
        "n_metrics",
        "n_periods",
        "seed",
        "observed",
        "error",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        import csv

        csv.DictWriter(fh, fieldnames=fieldnames).writeheader()

    buffered: list[dict[str, Any]] = []
    n_written = 0
    t0 = time.monotonic()

    for i, cell in enumerate(cells):
        cell_t0 = time.monotonic()
        n_other = cell["n_metrics"] - 2
        observed = _simulate_pair_pearson(
            cell["dist_a"],
            cell["dist_b"],
            n_other,
            cell["configured"],
            cell["n_entities"],
            cell["n_periods"],
            cell["seed"],
        )
        error = float("nan") if not np.isfinite(observed) else abs(observed - cell["configured"])
        row = {**cell, "observed": observed, "error": error}
        buffered.append({k: row.get(k) for k in fieldnames})
        n_written += 1

        if (i + 1) % flush_every == 0 or i == len(cells) - 1:
            with out_csv.open("a", encoding="utf-8", newline="") as fh:
                import csv

                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                for r in buffered:
                    writer.writerow(r)
            buffered = []
            elapsed = time.monotonic() - t0
            sys.stderr.write(
                f"[{label}] {n_written}/{len(cells)} "
                f"({elapsed:.1f}s, last cell {time.monotonic() - cell_t0:.2f}s)\n"
            )
            sys.stderr.flush()

    return n_written


def run_focused() -> int:
    """Headline focused-subset sweep (per D1(c))."""
    return run_correlation_sweep(FOCUSED_PAIRS, RESULT_CSV, label="claim1-focused")


def run_full_matrix() -> int:
    """Optional appendix sweep — full 6x6 (36 pairings).

    Not run inline by default; operator can launch separately if appendix
    coverage is desired. Wall-clock estimate ~40 minutes.
    """
    pairs = [(a, b) for a in DISTRIBUTIONS for b in DISTRIBUTIONS]
    return run_correlation_sweep(pairs, APPENDIX_CSV, label="claim1-full")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "full":
        run_full_matrix()
    else:
        run_focused()
