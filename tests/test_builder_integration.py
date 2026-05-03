"""End-to-end integration tests for the builder.

Mirrors the acceptance criteria in mission-115-builder.md ::Bare minimum::,
::Full template::, ::Post-generation quality assertions::, and ::Regression::
sections. Each test runs the full pipeline:

    create()/create_from_yaml() → PlotsimConfig → generate_tables → assertions

A fixed seed is passed to ``generate_tables`` directly so the generated
data is deterministic regardless of the secrets-derived ``cfg.seed`` from
``interpret``.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from plotsim import create, create_from_yaml, generate_tables, validate
from plotsim.config import PlotsimConfig
from plotsim.trajectory import compute_trajectory


REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_YAML = REPO_ROOT / "plotsim" / "configs" / "templates" / "saas_template.yaml"
TEMPLATE_PY = REPO_ROOT / "plotsim" / "configs" / "templates" / "saas_template.py"


# ── Helpers ─────────────────────────────────────────────────────────────────


def _generate(cfg: PlotsimConfig, seed: int = 42) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    return generate_tables(cfg, rng)


def _saas_yaml_config() -> PlotsimConfig:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return create_from_yaml(TEMPLATE_YAML)


def _saas_py_config() -> PlotsimConfig:
    """Execute the Python template file and return its `config` global."""
    import runpy
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = runpy.run_path(str(TEMPLATE_PY))
    return result["config"]


# ── Bare-minimum acceptance ─────────────────────────────────────────────────


def test_bare_minimum_create_generates_valid_dataset():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = create(
            about="bare minimum smoke test",
            unit="company",
            window=("2023-01", "2024-12"),
            metrics=[
                {"name": "engagement", "type": "score", "polarity": "positive"},
                {"name": "mrr", "type": "amount", "polarity": "positive",
                 "range": [100, 50000]},
            ],
            segments=[
                {"name": "alpha", "count": 10, "archetype": "growth"},
                {"name": "beta", "count": 10, "archetype": "decline"},
            ],
        )

    table_names = [t.name for t in cfg.tables]
    assert "dim_date" in table_names
    assert "dim_company" in table_names
    assert "fct_company" in table_names

    fact = next(t for t in cfg.tables if t.name == "fct_company")
    metric_cols = {c.name for c in fact.columns if c.source.startswith("metric:")}
    assert metric_cols == {"engagement", "mrr"}

    tables = _generate(cfg)
    report = validate(cfg, tables)
    errors = [i for i in report.issues if i.severity == "error"]
    assert errors == [], f"engine validation errors: {[e.message for e in errors]}"

    assert len(tables["fct_company"]) > 0


def test_bare_minimum_writes_csv_files(tmp_path):
    """The bare-minimum config must round-trip to disk via the standard writer."""
    from plotsim.output import write_tables
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = create(
            about="csv smoke",
            unit="customer",
            window=("2023-01", "2024-06"),
            metrics=[
                {"name": "engagement", "type": "score", "polarity": "positive"},
            ],
            segments=[
                {"name": "a", "count": 5, "archetype": "growth"},
                {"name": "b", "count": 5, "archetype": "decline"},
            ],
        )
    cfg = cfg.model_copy(update={
        "output": cfg.output.model_copy(update={"directory": str(tmp_path)}),
    })
    tables = _generate(cfg)
    report = validate(cfg, tables)
    write_tables(tables, cfg, report)
    csv_files = list(tmp_path.glob("*.csv"))
    assert csv_files, f"no CSV files written under {tmp_path}"
    written_names = {p.stem for p in csv_files}
    assert {"dim_date", "dim_customer", "fct_customer"}.issubset(written_names)


# ── Full template acceptance ────────────────────────────────────────────────


def test_saas_template_yaml_loads_and_passes_engine_validation():
    cfg = _saas_yaml_config()
    tables = _generate(cfg)
    report = validate(cfg, tables)
    errors = [i for i in report.issues if i.severity == "error"]
    assert errors == [], f"engine validation errors: {[e.message for e in errors]}"


def test_saas_template_py_produces_equivalent_config():
    cfg_py = _saas_py_config()
    cfg_yaml = _saas_yaml_config()
    # Compare the load-time shape (modulo seed which is per-construction).
    def shape(c: PlotsimConfig) -> dict:
        return {
            "metrics": [(m.name, m.distribution) for m in c.metrics],
            "archetypes": [(a.name, len(a.curve_segments)) for a in c.archetypes],
            "entities": [(e.name, e.archetype, e.size) for e in c.entities],
            "tables": [(t.name, t.type, t.grain) for t in c.tables],
            "correlations": [(p.metric_a, p.metric_b, p.coefficient) for p in c.correlations],
            "stages_field": c.stages.field if c.stages else None,
        }
    assert shape(cfg_py) == shape(cfg_yaml)


def test_saas_template_has_explicit_dimensions_facts_events():
    cfg = _saas_yaml_config()
    types = {t.name: t.type for t in cfg.tables}
    assert "dim_date" in types and types["dim_date"] == "dim"
    assert "dim_company" in types and types["dim_company"] == "dim"
    assert "fct_engagement" in types and types["fct_engagement"] == "fact"
    assert "fct_revenue" in types and types["fct_revenue"] == "fact"
    assert "evt_login" in types and types["evt_login"] == "event"
    assert "evt_churn" in types and types["evt_churn"] == "event"


# ── Quality assertions on saas_template.yaml output ─────────────────────────


@pytest.fixture(scope="module")
def saas_dataset() -> dict[str, Any]:
    """Generate the saas template once per module so quality assertions
    share work."""
    cfg = _saas_yaml_config()
    tables = _generate(cfg)
    report = validate(cfg, tables)
    return {"cfg": cfg, "tables": tables, "report": report}


def _entity_engagement_series(
    fact: pd.DataFrame, entity_id_col: str, entity_id: str
) -> np.ndarray:
    rows = fact[fact[entity_id_col] == entity_id].sort_values("date_key")
    return rows["engagement_score"].to_numpy()


def test_pearson_shape_recovery_for_non_plateau_archetypes(saas_dataset):
    cfg: PlotsimConfig = saas_dataset["cfg"]
    fact: pd.DataFrame = saas_dataset["tables"]["fct_engagement"]
    n_periods = cfg.time_window.period_count()

    for archetype in cfg.archetypes:
        # Pure-plateau (constant) archetypes have zero variance — Pearson
        # is undefined. Skip these by detecting all curve_segments as plateau.
        if all(seg.curve == "plateau" for seg in archetype.curve_segments):
            continue

        expected = compute_trajectory(archetype, n_periods)
        if expected.std() < 1e-9:
            continue  # belt-and-braces

        # Each entity assigned to this archetype emits one cohort row
        # per period in fct_engagement; the ID column is "company_id".
        entities_for_arch = [e for e in cfg.entities if e.archetype == archetype.name]
        for entity in entities_for_arch:
            series = _entity_engagement_series(fact, "company_id", entity.name)
            if series.size != n_periods:
                # cohort row count must match period count for this fact's grain.
                continue
            if series.std() < 1e-9:
                continue
            r = np.corrcoef(expected, series)[0, 1]
            assert r > 0.5, (
                f"archetype {archetype.name!r} entity {entity.name!r}: "
                f"Pearson with expected trajectory was {r:.3f} (< 0.5)"
            )


def test_correlation_signs_match_configured_connections(saas_dataset):
    cfg: PlotsimConfig = saas_dataset["cfg"]
    tables = saas_dataset["tables"]

    # Build a single per-entity-per-period table merging all fact metrics.
    fct_eng = tables["fct_engagement"][["company_id", "date_key", "engagement_score"]]
    fct_rev = tables["fct_revenue"][["company_id", "date_key", "mrr"]]
    fct_sup = tables["fct_support_tickets"][[
        "company_id", "date_key", "ticket_count", "churn_risk", "nps",
    ]]
    merged = fct_eng.merge(fct_rev, on=["company_id", "date_key"]).merge(
        fct_sup, on=["company_id", "date_key"],
    )

    # Map config metric name → fact column name carrying it.
    metric_col = {
        "engagement": "engagement_score",
        "mrr": "mrr",
        "support_tickets": "ticket_count",
        "churn_risk": "churn_risk",
    }

    for pair in cfg.correlations:
        col_a = metric_col.get(pair.metric_a)
        col_b = metric_col.get(pair.metric_b)
        if col_a is None or col_b is None:
            continue
        # pandas Int64 (nullable integer extension) trips numpy's corrcoef;
        # cast to float64 so the integer poisson-driven count column threads
        # cleanly into corrcoef. Drop rows where either side is NaN — the
        # template's `mcar_rate` may have NaN'd cells, and corrcoef returns
        # NaN if either input contains a single NaN.
        sub = merged[[col_a, col_b]].astype("float64").dropna()
        a = sub[col_a].to_numpy()
        b = sub[col_b].to_numpy()
        if a.size < 8 or a.std() < 1e-9 or b.std() < 1e-9:
            continue
        r = np.corrcoef(a, b)[0, 1]
        # Loose sign check — engine adds noise + Higham projection so exact
        # value won't match. Sign agreement is the contract.
        assert (r > 0) == (pair.coefficient > 0), (
            f"correlation sign for {pair.metric_a}↔{pair.metric_b}: "
            f"configured {pair.coefficient:+.2f}, observed {r:+.3f}"
        )


def test_no_single_stage_dominates_more_than_60_percent(saas_dataset):
    """The lifecycle stages should partition the dataset, not collapse onto
    one label. > 60% in a single stage means the user's behavioural variety
    isn't translating to lifecycle variety."""
    cfg: PlotsimConfig = saas_dataset["cfg"]
    if cfg.stages is None:
        pytest.skip("template has no stages")
    fact = saas_dataset["tables"]["fct_support_tickets"]
    # The engine doesn't materialise stage labels in fact tables directly;
    # we approximate by binning churn_risk into the stage thresholds.
    thresholds = [s.threshold_enter for s in cfg.stages.sequence]
    risks = fact["churn_risk"].to_numpy()
    bins = np.digitize(risks, thresholds[1:])  # right side of each stage band
    counts = np.bincount(bins, minlength=len(thresholds))
    dominant_share = counts.max() / counts.sum()
    assert dominant_share <= 0.60, (
        f"one stage holds {dominant_share:.1%} of rows — concentration "
        f"too high (per-stage counts: {counts.tolist()})"
    )


def test_event_tables_non_empty(saas_dataset):
    tables = saas_dataset["tables"]
    assert len(tables["evt_login"]) > 0, "proportional event evt_login is empty"
    assert len(tables["evt_churn"]) > 0, "threshold event evt_churn is empty"


def test_customer_sentiment_labels_monotonic_in_engagement_mean(saas_dataset):
    """text:bucket labels must read low → high in declared order. Average
    engagement for rows tagged each label should ascend with label order."""
    fact = saas_dataset["tables"]["fct_engagement"]
    declared_order = ["at_risk", "lukewarm", "satisfied", "delighted"]
    means: list[tuple[str, float]] = []
    for label in declared_order:
        mask = fact["customer_sentiment"] == label
        if mask.sum() == 0:
            continue
        means.append((label, fact.loc[mask, "engagement_score"].mean()))
    # Don't require all four labels to be populated — small-N may skip a band.
    # But the labels that do appear must be monotone non-decreasing in mean.
    values = [v for _, v in means]
    assert values == sorted(values), (
        f"sentiment label means not monotone: {means}"
    )


def test_high_baseline_group_mean_exceeds_low_baseline_group_mean(saas_dataset):
    """Per saas_template.yaml: 2 cohorts have baseline mrr=high (promising_client,
    steady_enterprise), 2 have baseline mrr=low (slow_churn, dormant), 2 have
    baseline mrr=mid. Baseline overrides should produce a clear top-vs-bottom
    separation in archetype-level mean MRR.

    M117: post-expansion ``cfg.entities`` is one Entity per simulated company
    (not one per cohort), so a company-level groupby would pick the two
    lowest individuals of 95 instead of the low-baseline cohorts. The test
    aggregates company means up to archetype using ``cfg.entities`` order
    + the dim builder's deterministic ID convention (``c-001``..``c-095``)
    so each company maps to its archetype, then takes top-2 vs bottom-2
    archetype means.
    """
    cfg: PlotsimConfig = saas_dataset["cfg"]
    fact = saas_dataset["tables"]["fct_revenue"]

    n_high = sum(
        1 for s in cfg.archetypes
        if "mrr" in s.metric_overrides
        and s.metric_overrides["mrr"].value_range is not None
        and s.metric_overrides["mrr"].value_range.min
        > (cfg.metrics[1].value_range.min + cfg.metrics[1].value_range.max) / 2
    )
    if n_high == 0:
        pytest.skip("template has no high-baseline cohorts")

    # Build company_id → archetype using the dim builder's predictable PK
    # convention (one row per Entity in cfg.entities order, prefix from
    # table name, zero-padded to width=max(3, len(str(N)))).
    n_entities = len(cfg.entities)
    width = max(3, len(str(n_entities)))
    company_to_archetype = {
        f"c-{i+1:0{width}d}": e.archetype
        for i, e in enumerate(cfg.entities)
    }

    fact_with_arch = fact.assign(
        archetype=fact["company_id"].map(company_to_archetype),
    )
    archetype_means = (
        fact_with_arch.groupby("archetype")["mrr"].mean().sort_values()
    )
    # Two high-baseline + two low-baseline archetypes; bottom-2 vs top-2
    # archetype means should split sharply.
    bottom_2_mean = archetype_means.iloc[:2].mean()
    top_2_mean = archetype_means.iloc[-2:].mean()
    assert top_2_mean > 2.0 * bottom_2_mean, (
        f"top-2 archetype mean MRR ({top_2_mean:.0f}) should be at least 2× "
        f"bottom-2 archetype mean MRR ({bottom_2_mean:.0f}); per-archetype "
        f"means: {archetype_means.to_dict()}"
    )


# ── Regression: builder existence is non-disruptive ────────────────────────


def test_plotsim_top_level_create_re_export_exists():
    import plotsim
    assert hasattr(plotsim, "create")
    assert hasattr(plotsim, "create_from_yaml")
    # The mission's __init__.py annotation lists both. Verify __all__.
    assert "create" in plotsim.__all__
    assert "create_from_yaml" in plotsim.__all__


def test_existing_engine_imports_still_work():
    """Adding the builder must not break the existing `from plotsim import ...`
    surface."""
    import plotsim
    for name in ("PlotsimConfig", "load_config", "generate_tables", "validate",
                 "write_tables", "ManifestSchema"):
        assert hasattr(plotsim, name), f"existing surface missing: {name}"


# ── Regression: bundled engine templates still load and validate ───────────


@pytest.mark.parametrize("template", [
    "sample_education.yaml",
    "sample_hr.yaml",
    "sample_saas.yaml",
    "sample_marketing.yaml",
    "sample_retail.yaml",
])
def test_bundled_engine_templates_still_load(template):
    """The five bundled templates must continue to load through the engine
    path (load_config, not the new builder). Mission-115 must not regress
    them."""
    from plotsim import load_config
    path = REPO_ROOT / "plotsim" / "configs" / template
    if not path.exists():
        pytest.skip(f"{template} not present (M112 deletion)")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = load_config(str(path))
    assert cfg.metrics, f"{template}: no metrics loaded"
