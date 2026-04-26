"""Read result CSVs, compute summaries, emit markdown for the fidelity report.

Each claim's analysis function takes the path to its result CSV and returns a
dict of section name -> markdown table. The report writer (claim driver
scripts and Phase 4) concatenates these into ``analysis/fidelity-report.md``.

Summaries always come back as median + IQR (25th–75th percentile spread)
across the seed axis at minimum; some claims add additional grouping axes
(distribution pair, lag, archetype). The mission rule "results reported as
median + IQR, not single-seed point estimates" is enforced in this module
rather than in each claim driver, so a future claim that grows the harness
gets the same statistical treatment by default.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def _quantile_str(values: np.ndarray, q: float) -> str:
    """Format a quantile so the report's tables align cleanly. Empty -> '—'."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return "—"
    return f"{np.quantile(arr, q):.4f}"


def summarize_grouped(
    df: pd.DataFrame,
    group_keys: list[str],
    measurement_cols: list[str],
    *,
    tolerance_col: str | None = None,
    tolerance: float | None = None,
) -> pd.DataFrame:
    """Group ``df`` by ``group_keys`` and return median + IQR per measurement.

    If ``tolerance_col`` is provided alongside ``tolerance``, a ``flag`` column
    marks groups where ``|median(measurement) - target| > tolerance``. ``target``
    is read from ``tolerance_col`` per group. Used by the Claim 1 analyzer to
    flag distribution pairings that exceed the headline tolerance.
    """
    if df.empty:
        return df.copy()

    rows: list[dict] = []
    for keys, group in df.groupby(group_keys, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row: dict = dict(zip(group_keys, keys))
        for col in measurement_cols:
            values = group[col].to_numpy(dtype=float, na_value=np.nan)
            values = values[np.isfinite(values)]
            if values.size == 0:
                row[f"{col}_median"] = np.nan
                row[f"{col}_iqr_lo"] = np.nan
                row[f"{col}_iqr_hi"] = np.nan
                row[f"{col}_n"] = 0
                continue
            row[f"{col}_median"] = float(np.median(values))
            row[f"{col}_iqr_lo"] = float(np.quantile(values, 0.25))
            row[f"{col}_iqr_hi"] = float(np.quantile(values, 0.75))
            row[f"{col}_n"] = int(values.size)
        if tolerance_col is not None and tolerance is not None:
            target = group[tolerance_col].iloc[0]
            primary = measurement_cols[0]
            median = row[f"{primary}_median"]
            row["flag"] = (
                "" if not np.isfinite(median)
                else ("OUT" if abs(median - float(target)) > tolerance else "")
            )
        rows.append(row)
    return pd.DataFrame(rows)


def render_markdown_table(
    summary: pd.DataFrame,
    *,
    columns: list[str] | None = None,
    title: str | None = None,
    fmt_overrides: dict[str, str] | None = None,
) -> str:
    """Convert a summary DataFrame to a markdown table string.

    ``columns`` restricts and orders the output. ``fmt_overrides`` maps column
    name to a printf-style format string; default formatting is ``.4f`` for
    floats, ``str`` for everything else.
    """
    if summary.empty:
        return f"_(no rows in summary)_"

    cols = columns if columns is not None else list(summary.columns)
    fmt_overrides = fmt_overrides or {}

    def _fmt(col: str, val) -> str:
        if pd.isna(val):
            return "—"
        if col in fmt_overrides:
            return fmt_overrides[col] % val
        if isinstance(val, float):
            return f"{val:.4f}"
        return str(val)

    lines: list[str] = []
    if title:
        lines.append(f"**{title}**")
        lines.append("")
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join("---" for _ in cols) + " |")
    for _, row in summary.iterrows():
        lines.append("| " + " | ".join(_fmt(c, row.get(c)) for c in cols) + " |")
    return "\n".join(lines)


# --- Claim-specific analyzers ----------------------------------------------


def analyze_correlation_csv(
    csv_path: Path, *, tolerance: float = 0.10, poisson_tolerance: float = 0.15,
) -> dict[str, str]:
    """Claim 1 analysis. Returns a dict of section markdown blocks.

    Sections:
        ``per_pair``: median observed Pearson per (dist_a, dist_b, configured)
            with IQR and pass/fail flag against the appropriate tolerance.
        ``95th_error_heatmap``: 6×6 (or smaller for the focused subset) table
            of 95th-percentile |observed - configured| per pair, across all
            magnitudes/seeds/sample sizes.
        ``exceeders``: the configurations where the headline tolerance fails,
            with the measured tolerance the report should adopt instead.
    """
    df = pd.read_csv(csv_path)
    out: dict[str, str] = {}

    df["error"] = (df["observed"] - df["configured"]).abs()
    df["uses_poisson"] = (df["dist_a"] == "poisson") | (df["dist_b"] == "poisson")
    df["effective_tol"] = np.where(df["uses_poisson"], poisson_tolerance, tolerance)

    per_pair = summarize_grouped(
        df, group_keys=["dist_a", "dist_b", "configured"],
        measurement_cols=["observed", "error"],
        tolerance_col="configured", tolerance=tolerance,
    )
    out["per_pair"] = render_markdown_table(
        per_pair,
        columns=["dist_a", "dist_b", "configured",
                 "observed_median", "observed_iqr_lo", "observed_iqr_hi",
                 "error_median", "error_iqr_hi", "observed_n", "flag"],
        title="Correlation fidelity — median observed Pearson per (dist_a, dist_b, configured)",
    )

    # 95th-percentile error per (dist_a, dist_b) across magnitudes.
    pair_keys = ["dist_a", "dist_b"]
    rows: list[dict] = []
    for keys, group in df.groupby(pair_keys, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rows.append({
            "dist_a": keys[0], "dist_b": keys[1],
            "p95_error": float(np.quantile(group["error"], 0.95)),
            "max_error": float(group["error"].max()),
            "n_cells": int(len(group)),
        })
    p95 = pd.DataFrame(rows)
    out["p95_error"] = render_markdown_table(
        p95,
        columns=["dist_a", "dist_b", "p95_error", "max_error", "n_cells"],
        title="95th-percentile |observed - configured| per pair",
    )

    # Exceeders: per-pair, do any configurations breach the effective tolerance?
    breaches: list[dict] = []
    for keys, group in df.groupby(["dist_a", "dist_b", "configured"], dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        median_obs = float(group["observed"].median())
        eff_tol = float(group["effective_tol"].iloc[0])
        if abs(median_obs - float(keys[2])) > eff_tol:
            breaches.append({
                "dist_a": keys[0], "dist_b": keys[1],
                "configured": float(keys[2]),
                "median_observed": median_obs,
                "effective_tol": eff_tol,
                "observed_minus_target": median_obs - float(keys[2]),
                "p95_error": float(np.quantile(group["error"], 0.95)),
            })
    if breaches:
        out["exceeders"] = render_markdown_table(
            pd.DataFrame(breaches),
            columns=["dist_a", "dist_b", "configured",
                     "median_observed", "effective_tol",
                     "observed_minus_target", "p95_error"],
            title=("Configurations where median observed exceeds the "
                   "effective tolerance"),
        )
    else:
        out["exceeders"] = (
            "_All measured (dist_a, dist_b, configured) cells land within the "
            "effective tolerance._"
        )

    return out


def analyze_lag_csv(csv_path: Path, *, lag_window: int = 1) -> dict[str, str]:
    """Claim 2 analysis: where is configured lag recoverable at output level?

    A cell passes when ``|median(peak_lag) - configured_lag| <= lag_window``
    AND ``peak_magnitude > unlagged_magnitude``. The breakdown shows which
    (configured_lag, blend_weight, archetype, metric_dist) cells pass and which
    don't.
    """
    df = pd.read_csv(csv_path)
    if df.empty:
        return {"matrix": "_(no lag-sweep rows)_"}

    keys = ["configured_lag", "blend_weight", "archetype", "metric_dist"]
    rows: list[dict] = []
    for k_vals, group in df.groupby(keys, dropna=False):
        if not isinstance(k_vals, tuple):
            k_vals = (k_vals,)
        median_peak = float(group["peak_lag_per_entity_median"].median())
        median_peak_mag = float(group["peak_magnitude"].median())
        median_unlagged = float(group["unlagged_magnitude"].median())
        configured = float(k_vals[0])
        within_window = abs(median_peak - configured) <= lag_window
        beats_unlagged = median_peak_mag > median_unlagged
        rows.append({
            "configured_lag": int(configured),
            "blend_weight": float(k_vals[1]),
            "archetype": k_vals[2],
            "metric_dist": k_vals[3],
            "median_peak_lag": median_peak,
            "median_peak_mag": median_peak_mag,
            "median_unlagged_mag": median_unlagged,
            "verdict": "PASS" if (within_window and beats_unlagged) else "FAIL",
        })
    summary = pd.DataFrame(rows)
    return {"matrix": render_markdown_table(
        summary,
        columns=["configured_lag", "blend_weight", "archetype",
                 "metric_dist", "median_peak_lag", "median_peak_mag",
                 "median_unlagged_mag", "verdict"],
        title="Lag fidelity — recoverable cells (PASS) vs boundary cases (FAIL)",
    )}


def analyze_trajectory_csv(csv_path: Path) -> dict[str, str]:
    """Claim 3 analysis: deviation in sigmas at the cell level."""
    df = pd.read_csv(csv_path)
    if df.empty:
        return {"per_template": "_(no trajectory-sweep rows)_"}

    rows: list[dict] = []
    for (template, metric), group in df.groupby(["template", "metric"], dropna=False):
        dev = group["deviation_in_sigma"].astype(float).to_numpy()
        dev = dev[np.isfinite(dev)]
        if dev.size == 0:
            continue
        rows.append({
            "template": template, "metric": metric,
            "median_dev_sigma": float(np.median(dev)),
            "p99_dev_sigma": float(np.quantile(dev, 0.99)),
            "max_dev_sigma": float(dev.max()),
            "cells_over_4sigma": int((dev > 4.0).sum()),
            "cells_total": int(dev.size),
        })
    summary = pd.DataFrame(rows)
    return {"per_template": render_markdown_table(
        summary,
        columns=["template", "metric", "median_dev_sigma", "p99_dev_sigma",
                 "max_dev_sigma", "cells_over_4sigma", "cells_total"],
        title=("Trajectory-first cell-level deviation per (template, metric); "
               "deviation in standard deviations of the configured noise envelope"),
    )}


def analyze_determinism_csv(csv_path: Path) -> dict[str, str]:
    """Claim 4 analysis: which axes guarantee byte-identical output?"""
    df = pd.read_csv(csv_path)
    if df.empty:
        return {"contract": "_(no determinism-sweep rows)_"}

    rows: list[dict] = []
    for axis, group in df.groupby("test_dimension", dropna=False):
        # NOT-TESTED rows carry empty hash_a/hash_b sentinels; their identical
        # column is False but the verdict should reflect "we didn't measure",
        # not "we measured and it differs".
        not_tested = (
            (group["hash_a"].fillna("") == "")
            & (group["hash_b"].fillna("") == "")
        ).all()
        n = len(group)
        n_identical = int(group["identical"].astype(int).sum())
        if not_tested:
            verdict = "NOT TESTED"
        elif n_identical == n:
            verdict = "GUARANTEED"
        elif n_identical == 0:
            verdict = "DIFFERS"
        else:
            verdict = "PARTIAL"
        rows.append({
            "test_dimension": axis,
            "n_pairs": n,
            "n_identical": n_identical,
            "verdict": verdict,
        })
    summary = pd.DataFrame(rows)
    return {"contract": render_markdown_table(
        summary,
        columns=["test_dimension", "n_pairs", "n_identical", "verdict"],
        title="Determinism — byte-identical CSV output across each tested axis",
    )}


def iter_summary_rows(csv_path: Path, group_keys: Iterable[str], measurement: str):
    """Convenience iterator used by smoke tests. Yields (key_tuple, median).

    Materialises the median per group without going through the markdown
    renderer — keeps assert-side comparisons clean.
    """
    df = pd.read_csv(csv_path)
    keys = list(group_keys)
    for k_vals, group in df.groupby(keys, dropna=False):
        if not isinstance(k_vals, tuple):
            k_vals = (k_vals,)
        values = group[measurement].astype(float).to_numpy()
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        yield k_vals, float(np.median(values))


__all__ = [
    "summarize_grouped",
    "render_markdown_table",
    "analyze_correlation_csv",
    "analyze_lag_csv",
    "analyze_trajectory_csv",
    "analyze_determinism_csv",
    "iter_summary_rows",
]
