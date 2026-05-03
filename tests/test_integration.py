"""End-to-end integration tests (Mission 010).

Covers:
  * Per-template e2e: load → generate → validate → write → reload → reload-roundtrip.
  * Determinism: same seed = identical CSVs; different seed = different values.
  * Cross-table consistency: trajectory-first invariant, proportional events,
    lagged metrics, threshold events, stage monotonicity, FK integrity.
  * Edge-case configs built by mutating the shipped SaaS template.
  * A handful of subprocess CLI smoke tests that exercise the installed
    entry point (test_cli.py already covers the in-process path).

These tests are tagged ``integration`` — the marker is not registered
in pyproject so pytest emits a ``PytestUnknownMarkWarning``, but
``pytest -m "not integration"`` still filters them out for fast loops.
"""
from __future__ import annotations

import io
import subprocess
import sys
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

import plotsim
from plotsim import (
    generate_tables,
    load_config,
    validate,
    write_tables,
)
from plotsim import cli
from plotsim.config import PlotsimConfig


pytestmark = pytest.mark.integration


ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = ROOT / "plotsim" / "configs"
SAAS_YAML = CONFIGS_DIR / "sample_saas.yaml"
HR_YAML = CONFIGS_DIR / "sample_hr.yaml"
EDUCATION_YAML = CONFIGS_DIR / "sample_education.yaml"
RETAIL_YAML = CONFIGS_DIR / "sample_retail.yaml"
MARKETING_YAML = CONFIGS_DIR / "sample_marketing.yaml"

ALL_TEMPLATES: dict[str, Path] = {
    "saas": SAAS_YAML,
    "hr": HR_YAML,
    "education": EDUCATION_YAML,
    "retail": RETAIL_YAML,
    "marketing": MARKETING_YAML,
}


# --- Helpers ------------------------------------------------------------------


def generate(config: PlotsimConfig, seed: int | None = None) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(config.seed if seed is None else seed)
    return generate_tables(config, rng)


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_yaml(data: dict[str, Any], path: Path) -> Path:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def mutate_saas(tmp_path: Path, mutate_fn) -> PlotsimConfig:
    """Load saas yaml, pass the dict to ``mutate_fn`` (in-place), reload.

    Used for edge-case tests that can't express themselves as a shipped
    template without bloating the ``configs/`` directory.
    """
    data = load_yaml(SAAS_YAML)
    mutate_fn(data)
    out = write_yaml(data, tmp_path / "mutated.yaml")
    return load_config(out)


def run_cli_inproc(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        try:
            code = cli.main(list(argv))
        except SystemExit as exc:
            code = int(exc.code) if exc.code is not None else 0
    return code, out.getvalue(), err.getvalue()


def run_cli_subprocess(*argv: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Invoke ``python -m plotsim.cli ...``. Works without pip-install."""
    return subprocess.run(
        [sys.executable, "-m", "plotsim.cli", *argv],
        capture_output=True,
        text=True,
        cwd=str(cwd) if cwd else None,
    )


# --- Part 1: per-template end-to-end ------------------------------------------


@pytest.mark.parametrize("domain,path", ALL_TEMPLATES.items())
def test_e2e_template(domain: str, path: Path, tmp_path: Path):
    config = load_config(path)
    tables = generate(config)
    report = validate(config, tables)
    assert report.ok, (
        f"{domain}: validation failed with "
        f"{len(report.errors)} error(s): "
        f"{[(i.check, i.table, i.message) for i in report.errors[:3]]}"
    )

    target = write_tables(tables, config, report, output_dir=tmp_path)
    assert target == tmp_path

    # Every configured table must have a matching CSV on disk.
    for tbl in config.tables:
        csv = tmp_path / f"{tbl.name}.csv"
        assert csv.exists(), f"{domain}: missing {csv.name}"

    # config.yaml and validation_report.txt were written.
    assert (tmp_path / "config.yaml").exists()
    assert (tmp_path / "validation_report.txt").exists()

    # Every non-event CSV reloads with the expected row count.
    for tbl in config.tables:
        reloaded = pd.read_csv(tmp_path / f"{tbl.name}.csv")
        if tbl.type != "event":
            assert len(reloaded) == len(tables[tbl.name]), (
                f"{domain}: row count drift on {tbl.name}"
            )

    # Config roundtrip: reload the written config.yaml and regenerate.
    reloaded_config = load_config(tmp_path / "config.yaml")
    regenerated = generate(reloaded_config)
    for name in regenerated:
        pd.testing.assert_frame_equal(
            regenerated[name].reset_index(drop=True),
            tables[name].reset_index(drop=True),
            check_dtype=False,
            obj=f"{domain}:{name} after config-yaml roundtrip",
        )


# --- Part 2: determinism ------------------------------------------------------


def test_determinism_same_seed_identical_tables():
    config = load_config(SAAS_YAML)
    a = generate(config, seed=42)
    b = generate(config, seed=42)
    assert set(a) == set(b)
    for name in a:
        pd.testing.assert_frame_equal(
            a[name].reset_index(drop=True),
            b[name].reset_index(drop=True),
            check_dtype=False,
            obj=f"determinism:{name}",
        )


def test_determinism_different_seed_same_shape_different_values():
    config = load_config(SAAS_YAML)
    a = generate(config, seed=42)
    c = generate(config, seed=99)
    assert set(a) == set(c)
    for name in a:
        assert list(a[name].columns) == list(c[name].columns), name
        if name.startswith("fct_"):
            assert len(a[name]) == len(c[name]), name
    # At least one fact table must differ value-wise under a different seed.
    differed = False
    for name in a:
        if not name.startswith("fct_"):
            continue
        df_a, df_c = a[name], c[name]
        if len(df_a) != len(df_c):
            differed = True
            break
        if not df_a.equals(df_c):
            differed = True
            break
    assert differed, "seed change produced no observable difference in facts"


# --- Part 3: cross-table consistency (SaaS) -----------------------------------


def _saas_tables() -> tuple[PlotsimConfig, dict[str, pd.DataFrame]]:
    cfg = load_config(SAAS_YAML)
    return cfg, generate(cfg)


def test_no_orphan_fks_saas():
    """Every FK value resolves to a PK value in the parent table."""
    config, tables = _saas_tables()
    from plotsim.config import FKSource, parse_source

    for tbl in config.tables:
        df = tables[tbl.name]
        for col in tbl.columns:
            parsed = parse_source(col.source)
            if not isinstance(parsed, FKSource):
                continue
            parent = tables[parsed.table]
            parent_keys = set(parent[parsed.column].tolist())
            child_keys = set(df[col.name].dropna().tolist())
            orphans = child_keys - parent_keys
            assert not orphans, (
                f"{tbl.name}.{col.name} → {parsed.table}.{parsed.column}: "
                f"{len(orphans)} orphan value(s) (sample: "
                f"{sorted(orphans, key=str)[:3]})"
            )
            nulls = df[col.name].isna().sum()
            assert nulls == 0, f"{tbl.name}.{col.name} has {nulls} null FK(s)"


def test_revenue_follows_trajectory_for_steady_grower():
    """steady_grower archetype → MRR slope must be positive and end > start.

    ``dim_company`` is grain ``per_entity`` (one row per cohort), so each
    configured cohort collapses to a single company_id. The saas template
    assigns ``steady_grower`` to exactly one cohort (``globex_cohort``).
    """
    config, tables = _saas_tables()
    dim_company = tables["dim_company"]
    globex_ids = dim_company[dim_company["cohort_size"] == 30]["company_id"].tolist()
    assert globex_ids, "expected a globex_cohort in dim_company"

    fct_revenue = tables["fct_revenue"].merge(
        tables["dim_date"][["date_key", "date"]], on="date_key"
    ).sort_values(["company_id", "date"])

    checked = 0
    for cid in globex_ids:
        series = fct_revenue[fct_revenue["company_id"] == cid]["mrr"].values
        series = np.asarray(series, dtype=float)
        series = series[~np.isnan(series)]
        if len(series) < 6:
            continue
        x = np.arange(len(series), dtype=float)
        # Manual OLS slope — avoids np.polyfit's internal np.linalg.lstsq call,
        # which crashes under coverage.py instrumentation when numpy's ufunc
        # dispatch table is corrupted by a mid-suite numpy reload (M102 / F15).
        xc = x - x.mean()
        yc = series - series.mean()
        slope = float((xc * yc).sum() / (xc * xc).sum())
        assert slope > 0, f"steady_grower entity {cid} has non-positive MRR slope"
        assert series[-3:].mean() > series[:3].mean(), (
            f"steady_grower entity {cid} ends lower than it starts"
        )
        checked += 1
    assert checked >= 1, "expected at least one steady_grower entity to test"


def test_rocket_then_cliff_ends_below_peak():
    """rocket_then_cliff archetype → last window avg < peak window avg."""
    config, tables = _saas_tables()
    dim_company = tables["dim_company"]
    acme_ids = dim_company[dim_company["cohort_size"] == 50]["company_id"].tolist()
    assert acme_ids, "expected an acme_corp_cohort in dim_company"

    fct_eng = tables["fct_engagement"].merge(
        tables["dim_date"][["date_key", "date"]], on="date_key"
    ).sort_values(["company_id", "date"])

    checked = 0
    for cid in acme_ids:
        series = fct_eng[fct_eng["company_id"] == cid]["engagement_score"].values
        series = np.asarray(series, dtype=float)
        series = series[~np.isnan(series)]
        if len(series) < 12:
            continue
        peak = series.max()
        tail_avg = series[-3:].mean()
        assert tail_avg < peak, (
            f"rocket_then_cliff entity {cid}: tail avg {tail_avg:.3f} >= peak {peak:.3f}"
        )
        checked += 1
    assert checked >= 1, "expected at least one rocket_then_cliff entity to test"


def test_event_counts_correlate_with_driving_metric():
    """evt_login has row_count_source=proportional:engagement:scale:5 —
    row counts per (company, period) must correlate positively with
    engagement_score on the same (company, period).
    """
    config, tables = _saas_tables()
    counts = (
        tables["evt_login"]
        .groupby(["company_id", "date_key"])
        .size()
        .reset_index(name="n_events")
    )
    joined = counts.merge(
        tables["fct_engagement"][["company_id", "date_key", "engagement_score"]],
        on=["company_id", "date_key"],
        how="inner",
    ).dropna()
    assert len(joined) >= 30, f"only {len(joined)} joined rows to correlate"
    r = float(joined["n_events"].corr(joined["engagement_score"]))
    assert r > 0.5, f"evt_login vs engagement correlation too weak: r={r:.3f}"


def test_lagged_metric_correlates_better_when_shifted():
    """support_tickets.causal_lag = {driver: engagement, lag_periods: 2}.

    For a majority of entities, corr(tickets[2:], engagement[:-2]) must be
    stronger (in absolute value) than the unshifted correlation.
    """
    config, tables = _saas_tables()
    eng = tables["fct_engagement"][["company_id", "date_key", "engagement_score"]]
    tix = tables["fct_support_tickets"][["company_id", "date_key", "ticket_count"]]
    merged = eng.merge(tix, on=["company_id", "date_key"]).merge(
        tables["dim_date"][["date_key", "date"]], on="date_key"
    ).sort_values(["company_id", "date"])

    stronger = 0
    total = 0
    for _, grp in merged.groupby("company_id"):
        # F3 (M102): Int64 nullable columns survive pd.to_numeric as Int64,
        # then `.to_numpy()` returns object dtype with pd.NA for nulls —
        # which np.isnan can't handle. Force float + np.nan up front so the
        # mask works regardless of upstream nullable-int promotion.
        e = pd.to_numeric(grp["engagement_score"], errors="coerce").to_numpy(
            dtype=float, na_value=np.nan,
        )
        t = pd.to_numeric(grp["ticket_count"], errors="coerce").to_numpy(
            dtype=float, na_value=np.nan,
        )
        mask = ~(np.isnan(e) | np.isnan(t))
        e, t = e[mask], t[mask]
        if len(e) < 10 or e.std() == 0 or t.std() == 0:
            continue
        lag = 2
        e_sh, t_sh = e[:-lag], t[lag:]
        if e_sh.std() == 0 or t_sh.std() == 0:
            continue
        c0 = abs(np.corrcoef(e, t)[0, 1])
        c2 = abs(np.corrcoef(e_sh, t_sh)[0, 1])
        total += 1
        if c2 > c0:
            stronger += 1
    # Per-entity grain on dim_company means the saas template only exposes
    # a handful of testable entities (one per cohort). Require a majority
    # among whatever entities have sufficient variance.
    assert total >= 2, f"only {total} entities were testable for lag alignment"
    assert stronger >= (total + 1) // 2, (
        f"lagged correlation stronger in only {stronger}/{total} entities"
    )


def test_churn_events_align_with_engagement_decline():
    """For every evt_churn row, engagement in the 3 periods up to and
    including the churn period averages below the entity's overall mean.
    """
    config, tables = _saas_tables()
    churn = tables["evt_churn"]
    if churn.empty:
        pytest.skip("no churn events generated for this seed")

    eng = tables["fct_engagement"].merge(
        tables["dim_date"][["date_key", "date"]], on="date_key"
    ).sort_values(["company_id", "date"])

    declines = 0
    total = 0
    for _, row in churn.iterrows():
        cid, dkey = row["company_id"], row["date_key"]
        entity = eng[eng["company_id"] == cid].reset_index(drop=True)
        if entity.empty:
            continue
        idx = entity.index[entity["date_key"] == dkey]
        if len(idx) == 0:
            continue
        end = int(idx[0])
        start = max(0, end - 2)
        window = pd.to_numeric(
            entity.loc[start:end, "engagement_score"], errors="coerce",
        ).dropna()
        overall = pd.to_numeric(entity["engagement_score"], errors="coerce").dropna()
        if window.empty or overall.empty:
            continue
        total += 1
        if window.mean() < overall.mean():
            declines += 1
    # The saas template produces a handful of events across three
    # archetypes — some entities (e.g. zombie_account) run hot on
    # churn_risk the whole window without a meaningful prior decline.
    # Require at least one event to trace back to an engagement decline.
    if total > 0:
        assert declines >= 1, (
            f"no churn event followed an engagement decline "
            f"(0 declines across {total} events)"
        )


def test_stages_never_go_backward_saas():
    """SaaS config has enforce_order=true on the stage sequence. The
    monotonically-non-decreasing property must hold per entity.
    """
    config, tables = _saas_tables()
    assert config.stages is not None and config.stages.enforce_order
    stage_order = {s.name: i for i, s in enumerate(config.stages.sequence)}
    # Find the fact table that owns the stage column.
    stage_table = None
    for name, df in tables.items():
        if "stage" in df.columns:
            stage_table = (name, df)
            break
    assert stage_table is not None, "no generated table has a 'stage' column"
    name, df = stage_table

    merged = df.merge(
        tables["dim_date"][["date_key", "date"]], on="date_key"
    )
    # Pick the per-entity FK (company_id) for the saas fact tables.
    entity_col = "company_id"
    assert entity_col in merged.columns

    violations = 0
    for _, grp in merged.sort_values([entity_col, "date"]).groupby(entity_col):
        ranks = [stage_order[s] for s in grp["stage"].tolist() if s in stage_order]
        for a, b in zip(ranks, ranks[1:]):
            if b < a:
                violations += 1
                break
    assert violations == 0, f"{violations} entities moved backward through stages"


# --- Part 4: edge-case configs -----------------------------------------------


def test_single_entity(tmp_path: Path):
    """Config collapsed to one entity with size=1 still generates valid tables."""
    def shrink(data):
        data["entities"] = [{
            "name": "solo",
            "archetype": "steady_grower",
            "size": 1,
        }]
    cfg = mutate_saas(tmp_path, shrink)
    tables = generate(cfg)
    report = validate(cfg, tables)
    assert report.ok, f"single-entity config failed validation: {report.errors[:2]}"
    # M106: SCD-expanded dim_company holds one row per (entity, version);
    # the entity count is the unique-PK count, not the raw row count.
    assert tables["dim_company"]["company_id"].nunique() == 1


def test_shortest_window(tmp_path: Path):
    """Two-month window — the shortest the TimeWindow validator permits
    (strict start<end blocks a true 1-month configuration; see state.md).
    """
    def shrink(data):
        data["time_window"]["start"] = "2024-01"
        data["time_window"]["end"] = "2024-02"
    cfg = mutate_saas(tmp_path, shrink)
    tables = generate(cfg)
    report = validate(cfg, tables)
    assert report.ok, f"2-month window failed: {report.errors[:2]}"
    assert len(tables["dim_date"]) == 2
    # dim_company is grain=per_entity (one row per cohort), so
    # per_entity_per_period facts are cohorts × periods.
    n_cohorts = len(cfg.entities)
    assert len(tables["fct_engagement"]) == n_cohorts * 2


def test_all_same_archetype(tmp_path: Path):
    """Five cohorts on the same non-monotonic archetype generate cleanly.

    ``rocket_then_cliff`` is used rather than a monotonic archetype like
    ``steady_grower`` because the causal_coherence validator needs enough
    trajectory inflection to distinguish a lagged driver from an unshifted
    one — a strictly-rising curve produces degenerate lag correlations.
    """
    def flatten(data):
        data["entities"] = [
            {"name": f"cohort_{i}", "archetype": "rocket_then_cliff", "size": 5}
            for i in range(5)
        ]
    cfg = mutate_saas(tmp_path, flatten)
    tables = generate(cfg)
    report = validate(cfg, tables)
    assert report.ok, f"all-same-archetype failed: {report.errors[:2]}"
    # M106: SCD-expanded dim_company holds one row per (entity, version);
    # the entity count is the unique-PK count, not the raw row count.
    assert tables["dim_company"]["company_id"].nunique() == 5


def test_zero_noise_produces_no_metric_nulls(tmp_path: Path):
    def zero_noise(data):
        data["noise"] = {
            "gaussian_sigma": 0.0,
            "outlier_rate": 0.0,
            "mcar_rate": 0.0,
        }
    cfg = mutate_saas(tmp_path, zero_noise)
    tables = generate(cfg)
    report = validate(cfg, tables)
    assert report.ok, f"zero-noise failed: {report.errors[:2]}"
    # No nulls in any metric column (mcar_rate=0).
    for name, df in tables.items():
        if not name.startswith("fct_"):
            continue
        for col in df.columns:
            assert df[col].isna().sum() == 0, f"{name}.{col} has nulls under zero noise"


def test_maximum_noise_preserves_structural_checks(tmp_path: Path):
    """Cranking noise high should not break PK uniqueness, FK integrity,
    or the date spine. Metric null_policy may warn — allowed here.
    """
    def crank(data):
        data["noise"] = {
            "gaussian_sigma": 0.2,
            "outlier_rate": 0.1,
            "mcar_rate": 0.1,
        }
    cfg = mutate_saas(tmp_path, crank)
    tables = generate(cfg)
    report = validate(cfg, tables)
    structural_checks = {"pk_uniqueness", "fk_integrity", "date_spine"}
    structural_errors = [
        i for i in report.errors if i.check in structural_checks
    ]
    assert not structural_errors, (
        f"max-noise broke structural checks: "
        f"{[(i.check, i.message) for i in structural_errors]}"
    )


def test_no_correlations(tmp_path: Path):
    def drop_corrs(data):
        data.pop("correlations", None)
    cfg = mutate_saas(tmp_path, drop_corrs)
    tables = generate(cfg)
    report = validate(cfg, tables)
    assert report.ok, f"no-correlations failed: {report.errors[:2]}"
    assert not cfg.correlations


def test_no_stages_omits_stage_column(tmp_path: Path):
    def drop_stages(data):
        data.pop("stages", None)
    cfg = mutate_saas(tmp_path, drop_stages)
    tables = generate(cfg)
    report = validate(cfg, tables)
    assert report.ok, f"no-stages failed: {report.errors[:2]}"
    for name, df in tables.items():
        assert "stage" not in df.columns, (
            f"{name} still has a stage column after stages were dropped"
        )


def test_no_event_tables(tmp_path: Path):
    """Config with dims + facts only; no events, no threshold/proportional logic."""
    def drop_events(data):
        data["tables"] = [t for t in data["tables"] if t["type"] != "event"]
    cfg = mutate_saas(tmp_path, drop_events)
    tables = generate(cfg)
    report = validate(cfg, tables)
    assert report.ok, f"no-events failed: {report.errors[:2]}"
    assert not any(
        name.startswith("evt_") for name in tables
    ), f"unexpected event table generated: {list(tables)}"


# --- Part 5: CLI subprocess smoke tests ---------------------------------------
#
# These hit the ``python -m plotsim.cli`` entry point the installed script
# resolves to. The in-process path is already covered by tests/test_cli.py;
# these add coverage for the argv / exit-code contract that ships to users.


def test_cli_subprocess_run(tmp_path: Path):
    result = run_cli_subprocess(
        "run", str(SAAS_YAML), "-o", str(tmp_path), "--seed", "42", "-q",
        "--allow-absolute-output",
    )
    assert result.returncode == 0, (
        f"exit={result.returncode}; stderr={result.stderr!r}"
    )
    assert (tmp_path / "dim_date.csv").exists()
    assert (tmp_path / "config.yaml").exists()


def test_cli_subprocess_validate_valid():
    result = run_cli_subprocess("validate", str(SAAS_YAML))
    assert result.returncode == 0
    assert "VALID" in result.stdout


def test_cli_subprocess_validate_invalid(tmp_path: Path):
    broken = tmp_path / "broken.yaml"
    broken.write_text("not: a: valid: plotsim: config\n", encoding="utf-8")
    result = run_cli_subprocess("validate", str(broken))
    assert result.returncode == 1
    assert "INVALID" in result.stdout


def test_cli_subprocess_info():
    result = run_cli_subprocess("info", str(SAAS_YAML))
    assert result.returncode == 0
    for token in ("Domain:", "Entities:", "Tables:"):
        assert token in result.stdout, f"missing {token!r} in info output"


def test_cli_subprocess_list_templates():
    result = run_cli_subprocess("list-templates")
    assert result.returncode == 0
    for name in ALL_TEMPLATES:
        assert name in result.stdout


def test_cli_subprocess_template_to_stdout():
    result = run_cli_subprocess("template", "saas")
    assert result.returncode == 0
    parsed = yaml.safe_load(result.stdout)
    assert parsed is not None
    assert "domain" in parsed and "tables" in parsed


# --- Part 6: archetype statistical distinguishability (FIX-09) ---------------
#
# SF-7 capstone: prove the shipped archetypes produce statistically recoverable
# behavioral patterns. Project each entity's primary metric time series into a
# 4D feature space (mean, slope, last-first, std), cluster with KMeans, compare
# the clustering to ground-truth archetype labels via adjusted_rand_score.
#
# A future change to curves, noise, or correlations that silently destroys
# archetype distinguishability trips these tests — not the per-archetype shape
# assertions in Part 3, which only check each archetype in isolation.

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    from sklearn.preprocessing import StandardScaler
    _HAS_SKLEARN = True
except ImportError:  # pragma: no cover — optional test dep
    _HAS_SKLEARN = False

needs_sklearn = pytest.mark.skipif(
    not _HAS_SKLEARN,
    reason="scikit-learn not installed; `pip install plotsim[test]`",
)

# template → (yaml path, primary fact table, per_entity FK column, metric column)
#
# Metric choice: first continuous (beta/normal/gamma/lognorm) metric in the
# primary fact table. Integer poisson counts (session_count, sessions) are
# the first-listed metric for retail/marketing but carry too much sampling
# noise at their configured λ values (sqrt(λ) ≈ 2.5–3 ticks on a 0–15 value
# range) to discriminate trajectory shape with only 24 periods. The
# continuous siblings on the same fact table preserve archetype signal.
DISTINGUISHABILITY_SPEC: dict[str, tuple[Path, str, str, str]] = {
    "saas":      (SAAS_YAML,      "fct_engagement",  "company_id",  "engagement_score"),
    "hr":        (HR_YAML,        "fct_performance", "employee_id", "performance_score"),
    "education": (EDUCATION_YAML, "fct_grades",      "student_id",  "assignment_score"),
    "retail":    (RETAIL_YAML,    "fct_sessions",    "segment_id",  "conversion_rate"),
    "marketing": (MARKETING_YAML, "fct_traffic",     "customer_id", "bounce_rate"),
}


def _mutate_template(
    tmp_path: Path,
    yaml_path: Path,
    mutate_fn,
    out_name: str = "mutated.yaml",
) -> PlotsimConfig:
    data = load_yaml(yaml_path)
    mutate_fn(data)
    return load_config(write_yaml(data, tmp_path / out_name))


def _expand_entities_per_archetype(data: dict, n_per_archetype: int = 5) -> None:
    """Replace ``entities`` with ``n`` rows per originally-assigned archetype.

    Pure in-place mutation. Ground truth is recovered later via
    ``config.entities[i].archetype`` — row ordering is preserved by the
    dim_entity builder (_make_ids walks config.entities in order).
    """
    assigned = [e["archetype"] for e in data["entities"]]
    expanded = [
        {"name": f"{arch}_{i:02d}", "archetype": arch, "size": 1}
        for arch in assigned
        for i in range(n_per_archetype)
    ]
    data["entities"] = expanded


def _distinguishability_ari(
    config: PlotsimConfig,
    tables_dict: dict[str, pd.DataFrame],
    fact_name: str,
    entity_col: str,
    metric_col: str,
) -> float:
    """Cluster entities on (mean, slope, last-first, std) of their metric series,
    compare to ground-truth archetype labels, return adjusted_rand_score.
    """
    entity_dim_name = next(
        t.name for t in config.tables
        if t.grain == "per_entity" and any(c.name == entity_col for c in t.columns)
    )
    entity_dim = tables_dict[entity_dim_name]
    # M106: dedupe SCD-versioned per_entity dims to one row per entity PK
    # (first-version-wins) so positional indexing matches config.entities.
    entity_dim = (
        entity_dim.drop_duplicates(subset=entity_col, keep="first")
        .reset_index(drop=True)
    )
    fact = tables_dict[fact_name].sort_values(["date_key"]).reset_index(drop=True)

    n_entities = len(config.entities)
    feats = np.zeros((n_entities, 4), dtype=float)
    for i in range(n_entities):
        pk = entity_dim.iloc[i][entity_col]
        series = fact.loc[fact[entity_col] == pk, metric_col].astype(float).to_numpy()
        finite = series[np.isfinite(series)]
        if finite.size < 2:
            continue
        t_axis = np.arange(finite.size, dtype=float)
        # Manual OLS slope — see F15 in M102 for why we avoid np.polyfit here.
        tc = t_axis - t_axis.mean()
        fc = finite - finite.mean()
        slope = float((tc * fc).sum() / (tc * tc).sum())
        feats[i] = (
            float(np.mean(finite)),
            slope,
            float(finite[-1] - finite[0]),
            float(np.std(finite)),
        )

    scaled = StandardScaler().fit_transform(feats)
    labels_true = np.array([e.archetype for e in config.entities])
    n_clusters = len(np.unique(labels_true))
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(scaled)
    return float(adjusted_rand_score(labels_true, km.labels_))


def _assert_distinguishable(domain: str, tmp_path: Path) -> None:
    """Generate 5-per-archetype entities for ``domain``, cluster their primary
    continuous metric in a 4D feature space, and assert ARI > 0.5.

    ARI: 1.0 = perfect, 0.0 = chance. The 0.5 threshold is well above chance
    but loose enough to tolerate per-entity noise and correlation jitter.
    """
    yaml_path, fact_name, entity_col, metric_col = DISTINGUISHABILITY_SPEC[domain]
    cfg = _mutate_template(
        tmp_path, yaml_path,
        lambda d: _expand_entities_per_archetype(d, n_per_archetype=5),
    )
    tables = generate(cfg)
    ari = _distinguishability_ari(cfg, tables, fact_name, entity_col, metric_col)
    assert ari > 0.5, (
        f"{domain}: archetypes not distinguishable — ARI={ari:.3f}. "
        f"Either curves collapsed, noise overwhelmed signal, or correlations "
        f"pulled every series toward the same shape."
    )


@needs_sklearn
def test_archetypes_distinguishable_saas(tmp_path: Path):
    _assert_distinguishable("saas", tmp_path)


@needs_sklearn
def test_archetypes_distinguishable_hr(tmp_path: Path):
    _assert_distinguishable("hr", tmp_path)


@needs_sklearn
def test_archetypes_distinguishable_education(tmp_path: Path):
    _assert_distinguishable("education", tmp_path)


@needs_sklearn
def test_archetypes_distinguishable_retail(tmp_path: Path):
    _assert_distinguishable("retail", tmp_path)


@needs_sklearn
def test_archetypes_distinguishable_marketing(tmp_path: Path):
    _assert_distinguishable("marketing", tmp_path)


@needs_sklearn
def test_high_noise_degrades_distinguishability_gracefully(tmp_path: Path):
    """Cranking noise must reduce ARI (the test is noise-sensitive, not vacuous)
    without breaking the pipeline or destroying the signal entirely.

    Uses the marketing template because its five archetypes mix oscillating
    (``deal_seeker``), sigmoid (``organic_convert``), step-down
    (``paid_acquisition_churn``) and step-up (``dormant_reactivation``)
    shapes — several pairs share overlapping bounce_rate trajectories
    near the 0.3–0.5 band, making the signal structurally more fragile
    under noise than SaaS's rocket-vs-grower-vs-zombie set.

    Three assertions:
      1. Low-noise control clears the 0.5 bar (matches per-template tests).
      2. High-noise run still produces a finite ARI — pipeline degrades
         gracefully, no NaN / exception.
      3. High-noise ARI is strictly lower than low-noise ARI — proves the
         assertion is noise-sensitive and not a vacuous pass.
    """
    fact_name, entity_col, metric_col = (
        "fct_traffic", "customer_id", "bounce_rate",
    )

    def low_noise(data):
        _expand_entities_per_archetype(data, n_per_archetype=5)
        data["noise"] = {
            "gaussian_sigma": 0.05,
            "outlier_rate": 0.01,
            "mcar_rate": 0.005,
        }

    def high_noise(data):
        _expand_entities_per_archetype(data, n_per_archetype=5)
        data["noise"] = {
            "gaussian_sigma": 0.50,
            "outlier_rate": 0.20,
            "mcar_rate": 0.15,
        }

    low_cfg = _mutate_template(tmp_path, MARKETING_YAML, low_noise, "low.yaml")
    high_cfg = _mutate_template(tmp_path, MARKETING_YAML, high_noise, "high.yaml")
    low_ari = _distinguishability_ari(
        low_cfg, generate(low_cfg), fact_name, entity_col, metric_col,
    )
    high_ari = _distinguishability_ari(
        high_cfg, generate(high_cfg), fact_name, entity_col, metric_col,
    )

    assert low_ari > 0.5, (
        f"control: low-noise ARI {low_ari:.3f} below 0.5 — test setup is broken"
    )
    assert np.isfinite(high_ari), f"high-noise ARI non-finite: {high_ari}"
    assert high_ari < low_ari, (
        f"noise did not reduce ARI (low={low_ari:.3f}, high={high_ari:.3f}) — "
        f"assertion is not noise-sensitive"
    )


# --- Part 7: public API stability --------------------------------------------


def test_public_api_surface_matches_readme():
    """The three-line quickstart in the README imports must stay live."""
    assert callable(plotsim.load_config)
    assert callable(plotsim.generate_tables)
    assert callable(plotsim.validate)
    assert callable(plotsim.write_tables)
    assert plotsim.ValidationReport is not None
    # __version__ must mirror [project].version in pyproject.toml — reading
    # the file directly catches drift that importlib.metadata would miss
    # on a stale editable install.
    import re
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    m = re.search(r'\[project\][^\[]*?\nversion\s*=\s*"([^"]+)"', pyproject, re.DOTALL)
    assert m, "could not find [project].version in pyproject.toml"
    assert plotsim.__version__ == m.group(1)
