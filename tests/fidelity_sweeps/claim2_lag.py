"""Claim 2 — causal-lag fidelity at output level.

Per Option C lean grid (operator-approved 2026-04-25): the dist axis on the
lagged metric collapses to poisson (the saas template's canonical lag
target — engagement → support_tickets), the archetype axis collapses to
sigmoid (the smooth-rising default), and the time window is 100 entities ×
120 periods (long enough for cross-correlation analysis up to lag 30, shy
of the spec's 100×360 which would be 3× the wall-clock cost).

Full mission grid (6 dists × 5 lags × 3 weights × 3 archetypes × 5 seeds at
100×360) deferred to a post-perf-pass M104+ when the per-cell cost no
longer blows the session budget. The lean grid still characterizes the
xfail's stated lag ≤ 1 boundary, the blend_weight axis, and the lag-30
upper bound.

Methodology mirrors test_output_fidelity.py::TestLagBoundaries::test_lag_peak.
The driver metric is a continuous lognorm forced through a sigmoid
trajectory; the lagged metric inherits the driver's effective position
(blended at blend_weight) shifted by lag_periods. Cross-correlation across
lags 0..2× configured is computed per entity; the median peak lag and peak
magnitude characterize whether the lag is recoverable at output level.
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from plotsim import generate_tables
from plotsim.config import (
    Archetype,
    CausalLag,
    Column,
    CurveSegment,
    Domain,
    Entity,
    Metric,
    NoiseConfig,
    OutputConfig,
    PlotsimConfig,
    SurrogateKeyWarning,
    Table,
    TimeWindow,
    ValueRange,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_CSV = REPO_ROOT / "analysis" / "fidelity_sweeps" / "lag_recovery_results.csv"

LAGS = [1, 2, 5, 10, 30]
BLEND_WEIGHTS = [0.6, 0.8, 1.0]
ARCHETYPES = ["sigmoid"]  # lean grid; 'oscillating', 'plateau' deferred
LAGGED_METRIC_DISTS = ["poisson"]  # lean grid; full set deferred
N_ENTITIES = 100
N_PERIODS = 120
SEEDS_PER_CELL = 5
SEED_BASE = 9_000


def _archetype_segments(name: str) -> list[CurveSegment]:
    if name == "sigmoid":
        return [
            CurveSegment(
                curve="sigmoid",
                params={"midpoint": 0.5, "steepness": 6.0},
                start_pct=0.0,
                end_pct=1.0,
            )
        ]
    if name == "oscillating":
        return [
            CurveSegment(
                curve="oscillating",
                params={"period": 0.25, "amplitude": 0.4},
                start_pct=0.0,
                end_pct=1.0,
            )
        ]
    if name == "plateau":
        return [CurveSegment(curve="plateau", params={"level": 0.5}, start_pct=0.0, end_pct=1.0)]
    raise ValueError(f"unsupported archetype name {name!r}")


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


def _build_cfg(
    archetype_name: str,
    lagged_dist: str,
    lag: int,
    blend: float,
    n_entities: int,
    n_periods: int,
    seed: int,
) -> PlotsimConfig:
    arch = Archetype(
        name="a",
        label="a",
        description="lag-test archetype",
        curve_segments=_archetype_segments(archetype_name),
    )
    driver_params, _ = _params_for("lognorm")
    driver = Metric(
        name="driver",
        label="driver",
        distribution="lognorm",
        params=driver_params,
        polarity="positive",
    )
    target_params, target_vr = _params_for(lagged_dist)
    target = Metric(
        name="target",
        label="target",
        distribution=lagged_dist,
        params=target_params,
        polarity="positive",
        value_range=target_vr,
        causal_lag=CausalLag(driver="driver", lag_periods=lag, blend_weight=blend),
    )
    end_year = 2024 + (n_periods - 1) // 12
    end_month = (n_periods - 1) % 12 + 1
    fct_target_dtype = "int" if lagged_dist == "poisson" else "float"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SurrogateKeyWarning)
        return PlotsimConfig(
            domain=Domain(name="lag", description="lag", entity_type="e", entity_label="E"),
            time_window=TimeWindow(
                start="2024-01",
                end=f"{end_year}-{end_month:02d}",
                granularity="monthly",
            ),
            seed=seed,
            metrics=[driver, target],
            archetypes=[arch],
            entities=[Entity(name=f"e{i:03d}", archetype="a", size=1) for i in range(n_entities)],
            tables=[
                Table(
                    name="dim_date",
                    type="dim",
                    grain="per_period",
                    primary_key="date_key",
                    columns=[
                        Column(name="date_key", dtype="id", source="pk"),
                        Column(name="date", dtype="date", source="generated:date_key"),
                    ],
                ),
                Table(
                    name="dim_entity",
                    type="dim",
                    grain="per_entity",
                    primary_key="entity_pk",
                    columns=[
                        Column(name="entity_pk", dtype="id", source="pk"),
                        Column(name="nm", dtype="string", source="static:e"),
                    ],
                ),
                Table(
                    name="fct",
                    type="fact",
                    grain="per_entity_per_period",
                    primary_key=["date_key", "entity_pk"],
                    foreign_keys=["dim_date.date_key", "dim_entity.entity_pk"],
                    columns=[
                        Column(name="date_key", dtype="id", source="fk:dim_date.date_key"),
                        Column(name="entity_pk", dtype="id", source="fk:dim_entity.entity_pk"),
                        Column(name="driver_v", dtype="float", source="metric:driver"),
                        Column(name="target_v", dtype=fct_target_dtype, source="metric:target"),
                    ],
                ),
            ],
            correlations=[],
            noise=NoiseConfig(gaussian_sigma=0.0, outlier_rate=0.0, mcar_rate=0.0),
            output=OutputConfig(format="csv", directory="out/lag"),
        )


def _per_entity_xcorr(
    driver: np.ndarray, target: np.ndarray, max_lag: int
) -> tuple[int, float, float]:
    """Cross-correlation of driver and target across lags 0..max_lag.

    Returns (peak_lag, peak_magnitude, unlagged_magnitude).
    Both inputs are length n_periods; entity-aware caller does this per row.
    """
    n = len(driver)
    if n < max_lag + 5:
        return 0, float("nan"), float("nan")
    # Standardize once for stable comparison.
    d = (driver - driver.mean()) / (driver.std() + 1e-12)
    t = (target - target.mean()) / (target.std() + 1e-12)
    mags = np.empty(max_lag + 1, dtype=float)
    for k in range(max_lag + 1):
        d_k = d[: n - k]
        t_k = t[k:]
        mags[k] = float(np.mean(d_k * t_k))
    peak = int(np.argmax(mags))
    return peak, float(mags[peak]), float(mags[0])


def run_claim2(out_csv: Path = RESULT_CSV) -> int:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    t0 = time.monotonic()
    cells_total = (
        len(LAGGED_METRIC_DISTS) * len(LAGS) * len(BLEND_WEIGHTS) * len(ARCHETYPES) * SEEDS_PER_CELL
    )
    cell_count = 0

    for arch_name in ARCHETYPES:
        for lagged_dist in LAGGED_METRIC_DISTS:
            for lag in LAGS:
                for blend in BLEND_WEIGHTS:
                    for seed_offset in range(SEEDS_PER_CELL):
                        cell_count += 1
                        seed = SEED_BASE + seed_offset
                        cell_t0 = time.monotonic()
                        cfg = _build_cfg(
                            arch_name,
                            lagged_dist,
                            lag,
                            blend,
                            N_ENTITIES,
                            N_PERIODS,
                            seed,
                        )
                        rng = np.random.default_rng(cfg.seed)
                        tables = generate_tables(cfg, rng)
                        fct = tables["fct"].sort_values(["entity_pk", "date_key"])
                        peaks, peak_mags, unlagged_mags = [], [], []
                        max_lag_window = max(lag * 2, lag + 3)
                        for ent_pk, group in fct.groupby("entity_pk", sort=True):
                            d = group["driver_v"].astype(float).to_numpy()
                            t = group["target_v"].astype(float).to_numpy()
                            peak, mag, unlag = _per_entity_xcorr(
                                d,
                                t,
                                max_lag_window,
                            )
                            if not np.isfinite(mag):
                                continue
                            peaks.append(peak)
                            peak_mags.append(mag)
                            unlagged_mags.append(unlag)
                        rows.append(
                            {
                                "archetype": arch_name,
                                "metric_dist": lagged_dist,
                                "configured_lag": int(lag),
                                "blend_weight": float(blend),
                                "n_entities": N_ENTITIES,
                                "n_periods": N_PERIODS,
                                "seed": seed,
                                "peak_lag_per_entity_median": (
                                    float(np.median(peaks)) if peaks else float("nan")
                                ),
                                "peak_magnitude": (
                                    float(np.median(peak_mags)) if peak_mags else float("nan")
                                ),
                                "unlagged_magnitude": (
                                    float(np.median(unlagged_mags))
                                    if unlagged_mags
                                    else float("nan")
                                ),
                            }
                        )
                        sys.stderr.write(
                            f"[claim2] {cell_count}/{cells_total} "
                            f"arch={arch_name} dist={lagged_dist} "
                            f"lag={lag} w={blend:.1f} "
                            f"seed={seed} ({time.monotonic() - cell_t0:.1f}s)\n"
                        )
                        sys.stderr.flush()
                        if cell_count % 10 == 0 or cell_count == cells_total:
                            pd.DataFrame(rows).to_csv(
                                out_csv,
                                index=False,
                                encoding="utf-8",
                            )

    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8")
    sys.stderr.write(
        f"[claim2] wrote {len(rows)} rows to {out_csv} in " f"{time.monotonic() - t0:.1f}s\n"
    )
    return len(rows)


if __name__ == "__main__":
    run_claim2()
