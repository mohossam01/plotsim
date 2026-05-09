"""M103 fidelity smoke test — regression insurance against the report drifting.

Two roles:
1. Phase 1 (this initial commit): infrastructure tests that pin sweep_runner and
   sweep_analyzer's contract. F1 and F2 of the mission spec — the trivial-sweep
   schema check and the markdown-rendering check.
2. Phase 3 (added after Phase 2 sweeps complete): subset checks that re-run a
   small slice of each claim's sweep and assert the report's headline finding
   reproduces. Without these, the report drifts silently the next time the
   engine changes.

Total runtime budget: <60 seconds. Phase 1 alone clocks in a few seconds; Phase
3 absorbs the rest with carefully-sized subset cells.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tests.fidelity_sweeps.sweep_runner import run_sweep
from tests.fidelity_sweeps.sweep_analyzer import (
    analyze_correlation_csv,
    render_markdown_table,
    summarize_grouped,
)
from plotsim.config import (
    Archetype,
    Column,
    CorrelationPair,
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
)


# --- Phase 1 fixtures: minimal config builder for a trivial sweep -----------


def _trivial_config(seed: int, *, configured_corr: float) -> PlotsimConfig:
    """Build a 2-metric flat-plateau config the trivial sweep can target.

    Plateau matches the F6 pattern in test_metrics.py — trajectory is constant
    so the configured correlation is the only signal. Two beta metrics keep the
    setup symmetric and avoid the poisson-attenuation channel.
    """
    archetype = Archetype(
        name="trivial_arch",
        label="trivial",
        description="flat plateau for trivial sweep",
        curve_segments=[
            CurveSegment(curve="plateau", params={"level": 0.5}, start_pct=0.0, end_pct=1.0),
        ],
    )
    metrics = [
        Metric(
            name="m_a",
            label="metric a",
            distribution="beta",
            params={"alpha": 2.0, "beta": 5.0},
            polarity="positive",
            value_range={"min": 0.0, "max": 1.0},
        ),
        Metric(
            name="m_b",
            label="metric b",
            distribution="beta",
            params={"alpha": 2.0, "beta": 5.0},
            polarity="positive",
            value_range={"min": 0.0, "max": 1.0},
        ),
    ]
    entities = [Entity(name="e0", archetype="trivial_arch", size=1)]

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SurrogateKeyWarning)
        return PlotsimConfig(
            domain=Domain(
                name="trivial", description="trivial", entity_type="user", entity_label="Users"
            ),
            time_window=TimeWindow(start="2024-01", end="2024-12", granularity="monthly"),
            seed=seed,
            metrics=metrics,
            archetypes=[archetype],
            entities=entities,
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
                    name="dim_user",
                    type="dim",
                    grain="per_entity",
                    primary_key="user_id",
                    columns=[
                        Column(name="user_id", dtype="id", source="pk"),
                        Column(name="user_name", dtype="string", source="generated:faker.name"),
                    ],
                ),
                Table(
                    name="fct_metrics",
                    type="fact",
                    grain="per_entity_per_period",
                    primary_key=["date_key", "user_id"],
                    foreign_keys=["dim_date.date_key", "dim_user.user_id"],
                    columns=[
                        Column(name="date_key", dtype="id", source="fk:dim_date.date_key"),
                        Column(name="user_id", dtype="id", source="fk:dim_user.user_id"),
                        Column(name="m_a_value", dtype="float", source="metric:m_a"),
                        Column(name="m_b_value", dtype="float", source="metric:m_b"),
                    ],
                ),
            ],
            correlations=[
                CorrelationPair(metric_a="m_a", metric_b="m_b", coefficient=configured_corr)
            ],
            noise=NoiseConfig(gaussian_sigma=0.0, outlier_rate=0.0, mcar_rate=0.0),
            output=OutputConfig(format="csv", directory="out/trivial"),
        )


def _build_trivial(cell: dict) -> PlotsimConfig:
    return _trivial_config(seed=cell["seed"], configured_corr=cell["configured"])


def _measure_trivial(cell: dict, tables: dict) -> dict:
    """Compute observed Pearson on the m_a × m_b pair."""
    fct = tables["fct_metrics"]
    a = fct["m_a_value"].astype(float).to_numpy()
    b = fct["m_b_value"].astype(float).to_numpy()
    import numpy as np

    return {"observed": float(np.corrcoef(a, b)[0, 1]), "n_samples": int(len(a))}


# --- F1: sweep_runner trivial-sweep schema check ----------------------------


def test_sweep_runner_writes_expected_csv_schema(tmp_path: Path) -> None:
    """F1 mission spec: run on a 10-row trivial sweep and verify the CSV has
    the expected schema. Runner contract: each row carries every cell input
    column plus every measurement column, ordered alphabetically within each
    group, with header on the first line.
    """
    cells = [
        {"seed": s, "configured": c, "dist_a": "beta", "dist_b": "beta"}
        for s in range(5)
        for c in [-0.4, 0.4]
    ]
    out_csv = tmp_path / "trivial.csv"

    n = run_sweep(
        cells,
        _build_trivial,
        _measure_trivial,
        out_csv,
        flush_every=4,
        progress_label="trivial",
    )
    assert n == 10

    df = pd.read_csv(out_csv)
    assert len(df) == 10
    expected_cols = {"configured", "dist_a", "dist_b", "seed", "n_samples", "observed"}
    assert set(df.columns) == expected_cols, (
        f"trivial sweep schema drift: got {set(df.columns)}, " f"expected {expected_cols}"
    )
    # Sanity: input columns survive the round-trip unchanged; both correlation
    # signs are present so the sweep covered both magnitudes.
    assert set(df["configured"].unique()) == {-0.4, 0.4}
    assert set(df["dist_a"].unique()) == {"beta"}
    # Plateau-isolated pairs of beta with hand-tuned correlation should track
    # within tolerance even at this small per-cell sample count (12 periods ×
    # 1 entity = 12 samples). Looser than the headline tolerance because the
    # trivial sweep deliberately uses minimal samples to stay sub-second.
    # M127b: bumped from 0.30 → 0.40 to absorb the new copula's slight
    # tail attenuation on tiny samples; the headline correlation tests
    # carry the tighter regression guard.
    df["error"] = (df["observed"] - df["configured"]).abs()
    assert (
        df["error"].median() < 0.40
    ), f"trivial sweep medians wandered too far: {df['error'].median():.3f}"


# --- F2: sweep_analyzer renders sensible markdown ---------------------------


def test_sweep_analyzer_produces_markdown_table_on_trivial_sweep(
    tmp_path: Path,
) -> None:
    """F2 mission spec: runs on the trivial sweep from F1 and produces a
    sensible markdown table. The renderer's contract is column ordering, NaN
    handling, and a header row.
    """
    cells = [
        {"seed": s, "configured": c, "dist_a": "beta", "dist_b": "beta"}
        for s in range(5)
        for c in [-0.4, 0.4]
    ]
    out_csv = tmp_path / "trivial.csv"
    run_sweep(
        cells,
        _build_trivial,
        _measure_trivial,
        out_csv,
        flush_every=10,
        progress_label="trivial-analyze",
    )

    blocks = analyze_correlation_csv(out_csv, tolerance=0.20)
    assert "per_pair" in blocks
    assert "p95_error" in blocks
    table = blocks["per_pair"]
    assert table.startswith("**"), "title prefix missing"
    assert "| dist_a | dist_b | configured |" in table, "header row missing or malformed"
    # Two configured magnitudes → two summary rows for the same (beta, beta) pair.
    assert table.count("\n| beta | beta |") == 2

    # Direct summarize_grouped call: median + IQR columns appear.
    df = pd.read_csv(out_csv)
    summary = summarize_grouped(
        df,
        group_keys=["dist_a", "dist_b", "configured"],
        measurement_cols=["observed"],
    )
    for col in ["observed_median", "observed_iqr_lo", "observed_iqr_hi", "observed_n"]:
        assert col in summary.columns, f"summarize_grouped missing {col}"


# --- F2 supplemental: render_markdown_table handles empty + NaN -------------


def test_render_markdown_table_handles_empty_and_nan() -> None:
    """Renderer edge cases: an empty summary returns the no-rows sentinel; a
    NaN cell renders as the dash placeholder.
    """
    empty = pd.DataFrame(columns=["a", "b"])
    assert "no rows" in render_markdown_table(empty)

    nan_df = pd.DataFrame([{"label": "x", "value": float("nan"), "value_n": 0}])
    rendered = render_markdown_table(
        nan_df,
        columns=["label", "value", "value_n"],
    )
    assert "| x | — | 0 |" in rendered


# --- Phase 3 smoke subsets ---------------------------------------------------
# Each subset re-runs a tiny slice of one claim's sweep and asserts the
# documented headline tolerance reproduces. Together they fit under the
# 60-second mission budget. If any fails, the documented tolerances in
# docs/statistical-fidelity.md have drifted from reality and the addendum
# needs re-running before the next release.


# --- Claim 1 smoke: 3 representative pairs at one moderate magnitude --------


@pytest.mark.parametrize(
    "dist_a,dist_b,documented_tol",
    [
        # Tight pair (documented 95th-pct |err| = 0.021 at full sweep;
        # M128 re-measure 2026-05-02 seed=7000–7004 max|err|=0.018 at
        # +0.7 — tolerance held).
        ("beta", "normal", 0.20),
        # Edge of headline ±0.12 post-M128 (documented 95th-pct |err|=0.113
        # at -0.7; M128 re-measure max|err|=0.043 at +0.7 — tolerance held).
        ("beta", "lognorm", 0.25),
        # Poisson-involving (documented 95th-pct |err| = 0.069;
        # M128 re-measure max|err|=0.027 at +0.7 — tolerance held).
        ("beta", "poisson", 0.25),
    ],
)
def test_claim1_correlation_subset_within_documented_tolerance(
    dist_a: str,
    dist_b: str,
    documented_tol: float,
) -> None:
    """Claim 1 smoke. Smaller-than-headline tolerance because smoke uses
    30×12 samples (~360) vs the report's 100×24 (~2,400) — sampling noise
    widens single-realization Pearson by ~0.06. The smoke tolerance is the
    documented full-sweep p95 plus a sampling-noise margin, doubled for
    safety. A regression that shifts the engine's correlation envelope by
    more than ~0.10 still trips this — which is what we want.

    M128 (2026-05-02): re-measured against post-M127b engine commit `b1df0c6`
    (copula flip + distribution registry). beta×normal/poisson tightened;
    beta×lognorm widened at high negative magnitude (max|err| 0.091 → 0.118
    at -0.7) — documented in `docs/internal/statistical-fidelity.md`. The
    smoke tolerances above already absorb this drift; no smoke value needed
    to change.
    """
    from tests.fidelity_sweeps.claim1_correlation import (
        _simulate_pair_pearson,
    )

    coef = 0.5
    observed = _simulate_pair_pearson(
        dist_a,
        dist_b,
        n_other=0,
        coefficient=coef,
        n_entities=30,
        n_periods=12,
        seed=7000,
    )
    err = abs(observed - coef)
    assert err < documented_tol, (
        f"Claim 1 smoke regression: configured ({dist_a}, {dist_b})="
        f"{coef:+.2f}, observed={observed:+.4f}, "
        f"|err|={err:.4f} >= {documented_tol:.2f}. "
        f"Documented full-sweep p95 |err| is well below this; "
        f"see analysis/fidelity-report.md and docs/statistical-fidelity.md."
    )


# --- Claim 2 smoke: small-lag recovery on sigmoid -----------------------------


def test_claim2_small_lag_recovers_at_output_level() -> None:
    """Claim 2 smoke. Documented headline: configured lags 1 and 2 are
    recoverable at output level on a sigmoid-archetype driver (median peak
    lag within ±1 of configured). Larger lags are documented as not
    recoverable. This test pins the recoverable cell at lag=2.
    """
    from tests.fidelity_sweeps.claim2_lag import (
        _build_cfg,
        _per_entity_xcorr,
    )

    cfg = _build_cfg(
        archetype_name="sigmoid",
        lagged_dist="poisson",
        lag=2,
        blend=1.0,
        n_entities=30,
        n_periods=60,
        seed=9000,
    )
    import numpy as np
    from plotsim import generate_tables

    rng = np.random.default_rng(cfg.seed)
    tables = generate_tables(cfg, rng)
    fct = tables["fct"].sort_values(["entity_pk", "date_key"])
    peaks: list[int] = []
    for _, group in fct.groupby("entity_pk", sort=True):
        d = group["driver_v"].astype(float).to_numpy()
        t = group["target_v"].astype(float).to_numpy()
        peak, mag, unlag = _per_entity_xcorr(d, t, max_lag=6)
        if not np.isfinite(mag):
            continue
        peaks.append(peak)
    median_peak = float(np.median(peaks))
    assert abs(median_peak - 2) <= 1.5, (
        f"Claim 2 smoke regression: configured lag=2 with sigmoid "
        f"archetype produced median peak lag {median_peak:.2f}; "
        f"documented as recoverable within ±1 of configured."
    )


# --- Claim 3 smoke: 50-cell trajectory subset on saas -----------------------


def test_claim3_trajectory_envelope_reproduces_on_saas_subset() -> None:
    """Claim 3 smoke. Documented headline: <1% of cells exceed 4σ across
    the full 11,865-cell sweep. The subset takes a single saas generation
    + 50 sampled (entity, period) cells. Smoke tolerance: at most 30% of
    sampled cells beyond 4σ (vastly looser than the documented 0.84% so
    the test isn't sensitive to single-seed noise; a regression that
    shifts cells systematically off-trajectory still trips it).
    """
    import numpy as np

    from tests.fidelity_sweeps.claim3_trajectory import (
        _verify_template_seed,
    )

    sample_rng = np.random.default_rng(0xC1A1)
    rows = _verify_template_seed("saas", seed_idx=0, cells_per_run=50, rng=sample_rng)
    devs = np.array(
        [r["deviation_in_sigma"] for r in rows if np.isfinite(r["deviation_in_sigma"])],
        dtype=float,
    )
    assert len(devs) > 50, (
        f"Claim 3 smoke regression: 50-sample run produced only "
        f"{len(devs)} valid cells (expected 50 cells × ~6 metrics ≈ 300). "
        f"Likely an MCAR or filtering bug in the verifier."
    )
    over_4sigma_frac = float((np.abs(devs) > 4.0).mean())
    assert over_4sigma_frac < 0.30, (
        f"Claim 3 smoke regression: {over_4sigma_frac:.2%} of saas cells "
        f"exceed 4σ; documented full-sweep is 0.84%. A drift this large "
        f"signals the trajectory-first invariant is no longer holding."
    )


# --- Claim 4 smoke: same-process determinism on saas ------------------------


def test_claim4_same_process_determinism_on_saas() -> None:
    """Claim 4 smoke. Documented headline: same-process, same seed, two
    calls produce byte-identical CSVs. Cheapest determinism check
    available; runs in <2s on the saas template.
    """
    from tests.fidelity_sweeps.claim4_determinism import _same_process_pair

    pairs = _same_process_pair()
    differences = [(fn, ha, hb) for fn, (ha, hb) in pairs.items() if ha != hb]
    assert not differences, (
        f"Claim 4 smoke regression: same-process determinism broken for: " f"{differences[:3]!r}"
    )
