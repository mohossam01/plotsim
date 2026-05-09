"""F2 regression — bypass-aware correlation in apply_correlations (M102).

Pre-fix: when one metric's center degenerated mid-window (poisson λ ≈ 0 here),
``apply_correlations`` set its z-slot to 0 but kept the full Cholesky factor.
The matmul ``corr_z = L @ z`` then mixed that forced 0 into every other
metric's transformed Gaussian, structurally attenuating the configured
correlations between the *non-bypass* pairs during the bypass window.

Post-fix: when ``any(bypass)`` is True, the function slices the correlation
matrix to the active rows/columns, Cholesky-factors the submatrix, and
applies that to the active z-values only — leaving the bypass slots inert
(they are skipped by the per-metric output loop anyway).

This file has two tests:

* ``test_correlation_bypass_attenuation_resolved`` — the regression. A
  4-metric config (3 normals + 1 poisson) drops to position 0 in the
  second half of a 24-month window. Poisson center → 0 → bypass; the
  normals' configured 0.7 pairwise correlations must hold during the
  bypass window after the fix. Pre-fix, the late-window observed
  Pearson on each (normal, normal) pair is measurably attenuated below
  0.6.
* ``test_no_bypass_path_unchanged_control`` — the control. Same metrics
  and configured correlations, but the trajectory plateaus at a level
  that never triggers bypass. Confirms F2 does not regress the
  existing ``L @ z`` fast path.

Both tests parametrize over multiple seeds and assert on the median
observed Pearson per pair, so single-seed sampling noise can't flip
the verdict.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pytest

from plotsim import generate_tables, load_config
from plotsim.config import SurrogateKeyWarning


SEEDS = list(range(8))
PAIRS = (("x", "y"), ("x", "z"), ("y", "z"))
CONFIGURED = 0.7
TOLERANCE = 0.10
EARLY_END_KEY = 20250101  # cut at 2025-01-01: months 1-12 vs 13-24


_BYPASS_YAML = """\
domain:
  name: "F2 bypass test"
  description: "Drop to 0 in second half — triggers poisson bypass"
  entity_type: "user"
  entity_label: "Users"

time_window:
  start: "2024-01"
  end: "2025-12"
  granularity: "monthly"

seed: 0

metrics:
  # poisson FIRST in declaration order so it occupies column 0 of the
  # lower-triangular Cholesky factor. The bug requires the bypass metric
  # to be at a column index k such that L[i, k] is non-zero for some
  # downstream i — only achievable when the bypass metric is at or
  # before any affected metric. With poisson last, L[*, last] = 0 in
  # every preceding row by construction, and the bug cannot manifest.
  - name: "p"
    label: "Poisson p"
    distribution: "poisson"
    params: {lambda: 5.0}
    polarity: "positive"
  - name: "x"
    label: "Normal x"
    distribution: "normal"
    params: {mu: 10.0, sigma: 2.0}
    polarity: "positive"
  - name: "y"
    label: "Normal y"
    distribution: "normal"
    params: {mu: 10.0, sigma: 2.0}
    polarity: "positive"
  - name: "z"
    label: "Normal z"
    distribution: "normal"
    params: {mu: 10.0, sigma: 2.0}
    polarity: "positive"

archetypes:
  - name: "drops_at_half"
    label: "Plateau then drops to zero"
    description: "0.7 for first half, 0.0 for second half — bypass trigger for poisson"
    curve_segments:
      - curve: "plateau"
        params: {level: 0.7}
        start_pct: 0.0
        end_pct: 0.5
      - curve: "plateau"
        params: {level: 0.0}
        start_pct: 0.5
        end_pct: 1.0

entities:
  - {name: "u01", archetype: "drops_at_half", size: 1}
  - {name: "u02", archetype: "drops_at_half", size: 1}
  - {name: "u03", archetype: "drops_at_half", size: 1}
  - {name: "u04", archetype: "drops_at_half", size: 1}
  - {name: "u05", archetype: "drops_at_half", size: 1}
  - {name: "u06", archetype: "drops_at_half", size: 1}
  - {name: "u07", archetype: "drops_at_half", size: 1}
  - {name: "u08", archetype: "drops_at_half", size: 1}
  - {name: "u09", archetype: "drops_at_half", size: 1}
  - {name: "u10", archetype: "drops_at_half", size: 1}
  - {name: "u11", archetype: "drops_at_half", size: 1}
  - {name: "u12", archetype: "drops_at_half", size: 1}
  - {name: "u13", archetype: "drops_at_half", size: 1}
  - {name: "u14", archetype: "drops_at_half", size: 1}
  - {name: "u15", archetype: "drops_at_half", size: 1}
  - {name: "u16", archetype: "drops_at_half", size: 1}
  - {name: "u17", archetype: "drops_at_half", size: 1}
  - {name: "u18", archetype: "drops_at_half", size: 1}
  - {name: "u19", archetype: "drops_at_half", size: 1}
  - {name: "u20", archetype: "drops_at_half", size: 1}

tables:
  - name: "dim_date"
    type: "dim"
    grain: "per_period"
    columns:
      - {name: "date_key", dtype: "id", source: "pk"}
      - {name: "date", dtype: "date", source: "generated:date_key"}
      - {name: "year", dtype: "int", source: "generated:date_key"}
      - {name: "month", dtype: "int", source: "generated:date_key"}
    primary_key: "date_key"

  - name: "dim_user"
    type: "dim"
    grain: "per_entity"
    columns:
      - {name: "user_id", dtype: "id", source: "pk"}
      - {name: "user_name", dtype: "string", source: "generated:faker.name"}
    primary_key: "user_id"

  - name: "fct_metrics"
    type: "fact"
    grain: "per_entity_per_period"
    columns:
      - {name: "date_key", dtype: "id", source: "fk:dim_date.date_key"}
      - {name: "user_id", dtype: "id", source: "fk:dim_user.user_id"}
      - {name: "x", dtype: "float", source: "metric:x"}
      - {name: "y", dtype: "float", source: "metric:y"}
      - {name: "z", dtype: "float", source: "metric:z"}
      - {name: "p", dtype: "int", source: "metric:p"}
    primary_key: ["date_key", "user_id"]
    foreign_keys: ["dim_date.date_key", "dim_user.user_id"]

correlations:
  # 0.7 between every (normal, normal) pair — the assertion target.
  - {metric_a: "x", metric_b: "y", coefficient: 0.7}
  - {metric_a: "x", metric_b: "z", coefficient: 0.7}
  - {metric_a: "y", metric_b: "z", coefficient: 0.7}
  # Non-zero (poisson, normal) coefficients are required for the bug
  # to manifest at all — they populate L[*, 0], so when z[0]=0 from
  # the poisson bypass it contaminates corr_z[1..3]. Pre-fix this
  # attenuates the (normal, normal) correlations the test asserts on.
  - {metric_a: "p", metric_b: "x", coefficient: 0.6}
  - {metric_a: "p", metric_b: "y", coefficient: 0.6}
  - {metric_a: "p", metric_b: "z", coefficient: 0.6}

output:
  format: "csv"
  directory: "out/f2_bypass_test"
"""


# Same fields as the bypass yaml, only the archetype curve differs:
# plateau at 0.5 throughout — never zero, poisson never bypasses.
_NO_BYPASS_YAML = _BYPASS_YAML.replace(
    """\
  - name: "drops_at_half"
    label: "Plateau then drops to zero"
    description: "0.7 for first half, 0.0 for second half — bypass trigger for poisson"
    curve_segments:
      - curve: "plateau"
        params: {level: 0.7}
        start_pct: 0.0
        end_pct: 0.5
      - curve: "plateau"
        params: {level: 0.0}
        start_pct: 0.5
        end_pct: 1.0""",
    """\
  - name: "drops_at_half"
    label: "Flat 0.5 plateau (no bypass control)"
    description: "Constant 0.5 — poisson lam = 5 * 0.5 = 2.5, well above 1e-9"
    curve_segments:
      - curve: "plateau"
        params: {level: 0.5}
        start_pct: 0.0
        end_pct: 1.0""",
)


def _load_inline(yaml_str: str, tmp_path: Path, name: str):
    path = tmp_path / name
    path.write_text(yaml_str, encoding="utf-8")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SurrogateKeyWarning)
        return load_config(path)


@pytest.fixture(scope="module")
def bypass_config(tmp_path_factory):
    return _load_inline(
        _BYPASS_YAML,
        tmp_path_factory.mktemp("f2_bypass"),
        "bypass.yaml",
    )


@pytest.fixture(scope="module")
def no_bypass_config(tmp_path_factory):
    return _load_inline(
        _NO_BYPASS_YAML,
        tmp_path_factory.mktemp("f2_no_bypass"),
        "no_bypass.yaml",
    )


def _generate_observed_pearson(
    cfg,
    seeds: Iterable[int],
    pairs: tuple[tuple[str, str], ...],
) -> tuple[dict, dict]:
    """Generate at each seed; return (early_observed, late_observed) dicts
    keyed by pair, with one observed-Pearson per seed in each list.
    """
    early: dict = {p: [] for p in pairs}
    late: dict = {p: [] for p in pairs}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        tables = generate_tables(cfg, rng)
        fct: pd.DataFrame = tables["fct_metrics"]
        early_df = fct[fct["date_key"] < EARLY_END_KEY]
        late_df = fct[fct["date_key"] >= EARLY_END_KEY]
        for a, b in pairs:
            early[(a, b)].append(float(np.corrcoef(early_df[a], early_df[b])[0, 1]))
            late[(a, b)].append(float(np.corrcoef(late_df[a], late_df[b])[0, 1]))
    return early, late


def test_correlation_bypass_attenuation_resolved(bypass_config):
    """F2 regression — late-window correlations between non-bypass metrics
    must hold within ±0.10 of configured even when one metric (poisson)
    bypasses every period in that window.

    Pre-fix expectation: median late-window Pearson on each (normal, normal)
    pair lands materially below 0.6 — the test fails.
    Post-fix expectation: median late-window Pearson lands within ±0.10 of
    the configured 0.7.
    """
    _, late = _generate_observed_pearson(bypass_config, SEEDS, PAIRS)
    for a, b in PAIRS:
        median_late = float(np.median(late[(a, b)]))
        assert abs(median_late - CONFIGURED) < TOLERANCE, (
            f"F2 regression: late-period (poisson-bypass) median observed "
            f"Pearson on ({a}, {b}) = {median_late:.3f}, expected within "
            f"±{TOLERANCE} of configured {CONFIGURED}. "
            f"Per-seed: {[round(v, 3) for v in late[(a, b)]]}"
        )


def test_no_bypass_path_unchanged_control(no_bypass_config):
    """F2 control — when no metric ever bypasses, the no-bypass branch is
    the existing ``corr_z = L @ z`` path. Both early and late windows
    should hit the configured correlations within ±0.10. Confirms F2's
    new ``any(bypass)`` slicing branch does not regress the existing
    fast path.
    """
    early, late = _generate_observed_pearson(no_bypass_config, SEEDS, PAIRS)
    for a, b in PAIRS:
        median_early = float(np.median(early[(a, b)]))
        median_late = float(np.median(late[(a, b)]))
        assert abs(median_early - CONFIGURED) < TOLERANCE, (
            f"F2 control: early-period median Pearson on ({a}, {b}) = "
            f"{median_early:.3f}, expected within ±{TOLERANCE} of "
            f"{CONFIGURED}. Per-seed: {[round(v, 3) for v in early[(a, b)]]}"
        )
        assert abs(median_late - CONFIGURED) < TOLERANCE, (
            f"F2 control: late-period median Pearson on ({a}, {b}) = "
            f"{median_late:.3f}, expected within ±{TOLERANCE} of "
            f"{CONFIGURED}. Per-seed: {[round(v, 3) for v in late[(a, b)]]}"
        )
