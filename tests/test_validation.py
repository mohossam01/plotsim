"""Tests for plotsim.validation — Mission 007 acceptance.

Covers:
  - validate_correlation_psd: sample-config matrices pass; a hand-crafted
    indefinite matrix is caught.
  - validate_pk_uniqueness: single-column + composite PKs clean on both
    sample domains; injected duplicates fire errors.
  - validate_fk_integrity: clean on both sample domains; injected orphan
    and null FK values fire error and warning respectively.
  - validate_date_spine: clean on both sample domains; injected gap + missing
    fact date_keys fire errors.
  - validate_causal_coherence: SaaS config (support_tickets lags engagement)
    passes; threshold events fire only at periods satisfying the condition;
    a hand-injected event row at a below-threshold period fires an error.
  - validate_null_policy: clean on both sample domains under the default
    noise config; injected extra nulls over the 3σ bound fire an error.
  - validate_tables orchestrator: deterministic issue list; clean run on
    both sample domains.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from plotsim import load_config
from plotsim.config import (
    Archetype,
    Column,
    CorrelationPair,
    CurveSegment,
    Domain,
    Entity,
    Metric,
    PlotsimConfig,
    NoiseConfig,
    OutputConfig,
    SurrogateKeyWarning,
    Table,
    TimeWindow,
)
from plotsim.tables import generate_tables
from plotsim.validation import (
    ALL_CHECKS,
    CHECK_CROSS_DIM_FK_CARDINALITY,
    CHECK_EMPTY_EVENT_TABLE,
    CHECK_FK_INTEGRITY,
    CHECK_TEMPORAL_COHERENCE,
    ValidationReport,
    validate_causal_coherence,
    validate_correlation_psd,
    validate_cross_dim_fk_cardinality,
    validate_date_spine,
    validate_empty_event_tables,
    validate_fk_integrity,
    validate_null_policy,
    validate_pk_uniqueness,
    validate_tables,
    validate_temporal_coherence,
)


ROOT = Path(__file__).resolve().parent.parent
SAAS_YAML = ROOT / "plotsim" / "configs" / "sample_saas.yaml"
HR_YAML = ROOT / "plotsim" / "configs" / "sample_hr.yaml"


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


@pytest.fixture(scope="module")
def saas_cfg():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SurrogateKeyWarning)
        return load_config(SAAS_YAML)


@pytest.fixture(scope="module")
def hr_cfg():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SurrogateKeyWarning)
        return load_config(HR_YAML)


@pytest.fixture
def saas_tables(saas_cfg):
    return generate_tables(saas_cfg, _rng(saas_cfg.seed))


@pytest.fixture
def hr_tables(hr_cfg):
    return generate_tables(hr_cfg, _rng(hr_cfg.seed))


# --- correlation_psd ---------------------------------------------------------


def test_psd_sample_saas_passes(saas_cfg):
    # Post-M007a: SaaS correlation matrix is strictly PD under the tightened
    # coefficients (0.72, -0.55, 0.55). See mission 007a for the original
    # non-PD state and why it was caught.
    assert validate_correlation_psd(saas_cfg) == []


def test_psd_sample_hr_passes(hr_cfg):
    # Post-M007a: HR correlation matrix is strictly PD under the tightened
    # coefficients (-0.65, -0.50, 0.55).
    assert validate_correlation_psd(hr_cfg) == []


def test_psd_passes_on_identity_only():
    cfg = _minimal_config(correlations=[])
    assert validate_correlation_psd(cfg) == []


def test_psd_passes_on_mild_correlations():
    cfg = _minimal_config(
        correlations=[
            CorrelationPair(metric_a="a", metric_b="b", coefficient=0.3),
        ],
    )
    assert validate_correlation_psd(cfg) == []


def test_psd_no_correlations_is_noop():
    cfg = _minimal_config(correlations=[])
    assert validate_correlation_psd(cfg) == []


def test_psd_projects_indefinite_matrix():
    # M111: an indefinite matrix (three metrics with 0.99 / 0.99 / -0.99 —
    # transitivity violation) is auto-corrected via Higham projection.
    # Pre-FIX-F04 this returned an error issue from ``validate_correlation_psd``
    # and load-time PlotsimConfig construction raised. Under M111 the
    # validator returns ``[]`` for any matrix that successfully projects;
    # an issue only appears if Higham + eigenvalue-clipping fallback
    # both fail (impossible for symmetric input). ``skip_validation=True``
    # is retained so we exercise the validator on the bare config —
    # going through PlotsimConfig.__init__ would also fire the load-time
    # warning, which we cover separately in test_correlation_projection.
    cfg = _minimal_config(
        correlations=[
            CorrelationPair(metric_a="a", metric_b="b", coefficient=0.99),
            CorrelationPair(metric_a="b", metric_b="c", coefficient=0.99),
            CorrelationPair(metric_a="a", metric_b="c", coefficient=-0.99),
        ],
        skip_validation=True,
    )
    issues = validate_correlation_psd(cfg)
    assert issues == []


# --- pk_uniqueness -----------------------------------------------------------


def test_pk_uniqueness_saas_clean(saas_cfg, saas_tables):
    assert validate_pk_uniqueness(saas_cfg, saas_tables) == []


def test_pk_uniqueness_hr_clean(hr_cfg, hr_tables):
    assert validate_pk_uniqueness(hr_cfg, hr_tables) == []


def test_pk_uniqueness_flags_single_col_duplicate(saas_cfg, saas_tables):
    # Duplicate the first event_id onto the second row of evt_login.
    evt = saas_tables["evt_login"].copy()
    assert len(evt) >= 2
    evt.loc[evt.index[1], "event_id"] = evt.loc[evt.index[0], "event_id"]
    broken = {**saas_tables, "evt_login": evt}
    issues = validate_pk_uniqueness(saas_cfg, broken)
    assert any(
        i.table == "evt_login" and i.severity == "error"
        for i in issues
    ), issues


def test_pk_uniqueness_flags_composite_duplicate(saas_cfg, saas_tables):
    # fct_engagement PK = [date_key, company_id]. Duplicate row 0 onto row 1.
    fct = saas_tables["fct_engagement"].copy()
    assert len(fct) >= 2
    fct.loc[fct.index[1], "date_key"] = fct.loc[fct.index[0], "date_key"]
    fct.loc[fct.index[1], "company_id"] = fct.loc[fct.index[0], "company_id"]
    broken = {**saas_tables, "fct_engagement": fct}
    issues = validate_pk_uniqueness(saas_cfg, broken)
    fct_issues = [i for i in issues if i.table == "fct_engagement"]
    assert fct_issues, issues
    assert fct_issues[0].details["duplicate_count"] >= 2


# --- fk_integrity ------------------------------------------------------------


def test_fk_integrity_saas_clean(saas_cfg, saas_tables):
    errors = [i for i in validate_fk_integrity(saas_cfg, saas_tables) if i.severity == "error"]
    assert errors == []


def test_fk_integrity_hr_clean(hr_cfg, hr_tables):
    errors = [i for i in validate_fk_integrity(hr_cfg, hr_tables) if i.severity == "error"]
    assert errors == []


def test_fk_integrity_flags_orphan_value(saas_cfg, saas_tables):
    fct = saas_tables["fct_engagement"].copy()
    fct.loc[fct.index[0], "company_id"] = "ghost-company-999"
    broken = {**saas_tables, "fct_engagement": fct}
    issues = validate_fk_integrity(saas_cfg, broken)
    orphan_errors = [
        i for i in issues
        if i.severity == "error"
        and i.table == "fct_engagement"
        and "orphan" in i.message
    ]
    assert orphan_errors
    assert "ghost-company-999" in orphan_errors[0].details["orphans_sample"]


def test_fk_integrity_null_fk_is_warning(saas_cfg, saas_tables):
    fct = saas_tables["fct_revenue"].copy()
    # plan_id is an FK from fct_revenue → dim_plan
    fct.loc[fct.index[0], "plan_id"] = None
    broken = {**saas_tables, "fct_revenue": fct}
    issues = validate_fk_integrity(saas_cfg, broken)
    warns = [
        i for i in issues
        if i.severity == "warning"
        and i.table == "fct_revenue"
        and i.details.get("column") == "plan_id"
    ]
    assert warns, issues


# --- date_spine --------------------------------------------------------------


def test_date_spine_saas_clean(saas_cfg, saas_tables):
    assert validate_date_spine(saas_cfg, saas_tables) == []


def test_date_spine_hr_clean(hr_cfg, hr_tables):
    assert validate_date_spine(hr_cfg, hr_tables) == []


def test_date_spine_flags_dim_date_missing(saas_cfg, saas_tables):
    broken = {k: v for k, v in saas_tables.items() if k != "dim_date"}
    issues = validate_date_spine(saas_cfg, broken)
    assert any(
        i.table == "dim_date" and "missing or empty" in i.message
        for i in issues
    )


def test_date_spine_flags_monthly_gap(saas_cfg, saas_tables):
    # Delete the 3rd month from dim_date → creates a gap.
    dd = saas_tables["dim_date"].drop(index=saas_tables["dim_date"].index[2]).reset_index(drop=True)
    broken = {**saas_tables, "dim_date": dd}
    issues = validate_date_spine(saas_cfg, broken)
    assert any(
        i.table == "dim_date" and "month gap" in i.message
        for i in issues
    ), issues


def test_date_spine_flags_fact_date_key_not_in_dim(saas_cfg, saas_tables):
    fct = saas_tables["fct_engagement"].copy()
    fct.loc[fct.index[0], "date_key"] = 99990101
    broken = {**saas_tables, "fct_engagement": fct}
    issues = validate_date_spine(saas_cfg, broken)
    assert any(
        i.table == "fct_engagement" and "not present in dim_date" in i.message
        for i in issues
    )


# --- causal_coherence --------------------------------------------------------


def test_causal_coherence_saas_passes(saas_cfg, saas_tables):
    # SaaS has support_tickets → causal_lag(engagement, 2), and evt_churn
    # with a threshold:churn_risk:above:0.7:for:3 column. Both should pass
    # on a fresh generation.
    issues = validate_causal_coherence(saas_cfg, saas_tables)
    assert issues == [], issues


def test_causal_coherence_hr_passes(hr_cfg, hr_tables):
    issues = validate_causal_coherence(hr_cfg, hr_tables)
    assert issues == [], issues


def test_causal_coherence_flags_fake_threshold_event(saas_cfg, saas_tables):
    # Take fct_support_tickets' first row (period 0). churn_risk at row 0 is
    # very unlikely to have satisfied a 3-period above-0.7 streak (the streak
    # can't start before the series). Point evt_churn's first event there.
    evt = saas_tables["evt_churn"].copy()
    if evt.empty:
        # Construct a single-row evt_churn at the earliest fct_support_tickets
        # row so we always have something to invalidate.
        fct = saas_tables["fct_support_tickets"]
        evt = pd.DataFrame([{
            "event_id": "e-0001",
            "date_key": fct.iloc[0]["date_key"],
            "company_id": fct.iloc[0]["company_id"],
            "churn_reason": "synthetic",
            "churn_flag": True,
        }])
    else:
        fct = saas_tables["fct_support_tickets"]
        evt.loc[evt.index[0], "date_key"] = fct.iloc[0]["date_key"]
        evt.loc[evt.index[0], "company_id"] = fct.iloc[0]["company_id"]
    broken = {**saas_tables, "evt_churn": evt}
    issues = validate_causal_coherence(saas_cfg, broken)
    assert any(
        i.table == "evt_churn"
        and i.severity == "error"
        and "threshold event column" in i.message
        for i in issues
    ), issues


# --- null_policy -------------------------------------------------------------


def test_null_policy_saas_clean(saas_cfg, saas_tables):
    issues = validate_null_policy(saas_cfg, saas_tables)
    assert issues == [], issues


def test_null_policy_hr_clean(hr_cfg, hr_tables):
    issues = validate_null_policy(hr_cfg, hr_tables)
    assert issues == [], issues


def test_null_policy_flags_non_metric_null(saas_cfg, saas_tables):
    # Null a generated column on dim_company (company_name = faker).
    dc = saas_tables["dim_company"].copy()
    dc.loc[dc.index[0], "company_name"] = None
    broken = {**saas_tables, "dim_company": dc}
    issues = validate_null_policy(saas_cfg, broken)
    assert any(
        i.table == "dim_company"
        and i.severity == "error"
        and "non-metric column" in i.message
        for i in issues
    ), issues


def test_null_policy_flags_metric_over_bound(saas_cfg, saas_tables):
    # mcar_rate=0.01, n=72 rows per metric column → upper bound ≈ 4 nulls.
    # Inject 40 nulls into engagement_score, far over the bound.
    fct = saas_tables["fct_engagement"].copy()
    fct["engagement_score"] = fct["engagement_score"].astype("object")
    fct.loc[fct.index[:40], "engagement_score"] = None
    broken = {**saas_tables, "fct_engagement": fct}
    issues = validate_null_policy(saas_cfg, broken)
    metric_errors = [
        i for i in issues
        if i.table == "fct_engagement"
        and i.details.get("column") == "engagement_score"
    ]
    assert metric_errors, issues
    # >= 40 because mcar_rate=0.01 may have produced 0-1 nulls in the
    # original column before injection. The validator's job is detecting
    # the over-bound condition, not asserting the original column was
    # clean.
    assert metric_errors[0].details["null_count"] >= 40


# --- Orchestrator + report ---------------------------------------------------


def test_validate_tables_saas_is_ok(saas_cfg, saas_tables):
    report = validate_tables(saas_cfg, saas_tables)
    assert isinstance(report, ValidationReport)
    assert report.ok, [f"{i.check}/{i.table}: {i.message}" for i in report.errors]


def test_validate_tables_hr_is_ok(hr_cfg, hr_tables):
    report = validate_tables(hr_cfg, hr_tables)
    assert report.ok, [f"{i.check}/{i.table}: {i.message}" for i in report.errors]


def test_validate_tables_deterministic(saas_cfg):
    # Same (config, seed) → same issue set twice in a row. We also inject a
    # single deterministic break so the issue list is non-empty.
    t1 = generate_tables(saas_cfg, _rng(saas_cfg.seed))
    t2 = generate_tables(saas_cfg, _rng(saas_cfg.seed))
    t1["fct_engagement"] = t1["fct_engagement"].copy()
    t2["fct_engagement"] = t2["fct_engagement"].copy()
    t1["fct_engagement"].loc[0, "company_id"] = "orphan-1"
    t2["fct_engagement"].loc[0, "company_id"] = "orphan-1"
    r1 = validate_tables(saas_cfg, t1)
    r2 = validate_tables(saas_cfg, t2)
    assert [(i.check, i.table, i.message) for i in r1.issues] == \
           [(i.check, i.table, i.message) for i in r2.issues]


def test_validate_tables_report_accessors(saas_cfg, saas_tables):
    # Sanity-check .errors / .warnings / .by_check on a report with a single
    # injected warning (null FK) and a single injected error (orphan FK).
    fct = saas_tables["fct_revenue"].copy()
    fct.loc[fct.index[0], "plan_id"] = None
    fct.loc[fct.index[1], "plan_id"] = "ghost-plan"
    broken = {**saas_tables, "fct_revenue": fct}
    report = validate_tables(saas_cfg, broken)
    fk_issues = report.by_check(CHECK_FK_INTEGRITY)
    assert any(i.severity == "warning" for i in fk_issues)
    assert any(i.severity == "error" for i in fk_issues)
    assert not report.ok
    assert set(i.check for i in report.issues).issubset(set(ALL_CHECKS))


# --- FIX-04 acceptance: cross-dim FK cardinality warning ---------------------


def test_validation_warns_on_collapsed_fk(saas_cfg, saas_tables):
    """FIX-04 / MF-1: regression guard. If a future change re-introduces
    the row-0 collapse on a multi-row parent dim, this validator must
    surface a WARNING. We synthesize the broken state by mutating
    fct_revenue.plan_id to a single value and inflating dim_plan to
    multiple rows.
    """
    plan = saas_tables["dim_plan"].copy()
    extra_row = plan.iloc[0].to_dict()
    extra_row["plan_id"] = "p-002"
    plan = pd.concat([plan, pd.DataFrame([extra_row])], ignore_index=True)

    fct = saas_tables["fct_revenue"].copy()
    fct["plan_id"] = plan["plan_id"].iloc[0]

    broken = {**saas_tables, "dim_plan": plan, "fct_revenue": fct}
    issues = validate_cross_dim_fk_cardinality(saas_cfg, broken)
    fct_issues = [
        i for i in issues
        if i.table == "fct_revenue" and i.details.get("column") == "plan_id"
    ]
    assert len(fct_issues) == 1
    assert fct_issues[0].severity == "warning"
    assert fct_issues[0].check == CHECK_CROSS_DIM_FK_CARDINALITY


def test_cross_dim_fk_skipped_for_single_row_parent(saas_cfg, saas_tables):
    """FIX-04: shipped SaaS has dim_plan with 1 row; collapse is the only
    correct outcome, so no warning fires."""
    issues = validate_cross_dim_fk_cardinality(saas_cfg, saas_tables)
    assert all(
        i.details.get("parent", "").split(".")[0] != "dim_plan"
        for i in issues
    )


# --- FIX-03 acceptance: empty event tables surface as warnings ---------------


def test_driverless_event_table_produces_validation_warning(hr_cfg, hr_tables):
    """FIX-03 / SF-9: HR's evt_attrition declares no driver and emits 0 rows.
    Generation contract preserved (table exists with correct schema), but the
    validator now flags it as a WARNING so the user knows it's silent.
    """
    issues = validate_empty_event_tables(hr_cfg, hr_tables)
    assert len(issues) == 1
    assert issues[0].table == "evt_attrition"
    assert issues[0].severity == "warning"
    assert issues[0].check == CHECK_EMPTY_EVENT_TABLE
    # The DataFrame still exists, with the right columns, just empty.
    df = hr_tables["evt_attrition"]
    assert len(df) == 0
    assert set(df.columns) == {"event_id", "date_key", "employee_id", "reason"}


def test_driven_event_table_produces_no_warning(saas_cfg, saas_tables):
    """FIX-03: SaaS templates' evt_login (proportional) and evt_churn (threshold)
    have configured drivers, so no empty-event warning fires for them.
    """
    issues = validate_empty_event_tables(saas_cfg, saas_tables)
    # Either zero issues, or only issues for tables that legitimately have a
    # driver but produced no rows (which the validator skips). Assert that
    # neither evt_login nor evt_churn show up.
    flagged = {i.table for i in issues}
    assert "evt_login" not in flagged
    assert "evt_churn" not in flagged


# --- M111: defense-in-depth gate at generate_tables under project-and-warn ---


def test_non_psd_matrix_no_longer_raises_at_generation_time():
    """M111: the defense-in-depth ``validate_correlation_psd`` call at the top
    of ``generate_tables`` no longer raises on non-PD matrices.

    Pre-M111 (FIX-01 + FIX-F04) it raised ``ValueError("positive
    semi-definite")``. Under M111, ``validate_correlation_psd`` returns
    ``[]`` for any matrix that successfully projects via Higham (which
    includes every realistic non-PD input on symmetric data). The gate
    is now an assertion that should never fire, since both the load-time
    pydantic validator and the gate use the same projection logic.

    This test exercises the ``skip_validation=True`` programmatic path
    (which bypasses the load-time validator) and confirms the gate
    accepts the config without raising. Generation may still fail for
    unrelated reasons in this stripped-down config (it only has
    ``dim_date``), so we exercise the gate's check directly.
    """
    cfg = _minimal_config(
        correlations=[
            CorrelationPair(metric_a="a", metric_b="b", coefficient=0.99),
            CorrelationPair(metric_a="b", metric_b="c", coefficient=0.99),
            CorrelationPair(metric_a="a", metric_b="c", coefficient=-0.99),
        ],
        skip_validation=True,
    )
    # The gate's check function returns [] for projectable inputs.
    assert validate_correlation_psd(cfg) == []


def _delivered_coefficient(cfg, metric_a: str, metric_b: str) -> float:
    """Return the coefficient the engine actually delivers for a pair.

    Post-M111, non-PD correlation matrices are projected at load time and
    the per-pair achieved value is recorded on ``cfg._correlation_adjustments``
    (a list of dicts: ``metric_a``, ``metric_b``, ``requested``, ``achieved``,
    ``adjustment``). For pairs that survived projection unchanged (or PD
    configs), fall back to the raw YAML coefficient on ``cfg.correlations``.
    """
    pair = {metric_a, metric_b}
    if cfg._correlation_adjustments is not None:
        for adj in cfg._correlation_adjustments:
            if {adj["metric_a"], adj["metric_b"]} == pair:
                return float(adj["achieved"])
    return float(next(
        c.coefficient for c in cfg.correlations
        if {c.metric_a, c.metric_b} == pair
    ))


def test_valid_correlation_matrix_still_works(saas_cfg, saas_tables):
    """FIX-01 negative case: a valid PSD matrix produces correlated output.

    Uses the shipped SaaS template and verifies that the engagement/mrr
    correlation in fct_engagement+fct_revenue lands within a generous ±0.30
    band of the *delivered* coefficient. Post-M112 the saas YAML's
    correlations are non-PD (intentionally — restoring the original
    intended values) and Higham projection adjusts them at load time, so
    "delivered" means the projected coefficient (when projection fired)
    rather than the raw YAML value. The bound stays generous: per-entity
    sample size is small and we're computing across pooled rows — this
    test asserts "correlation is applied", not "the coefficient lands
    exactly".
    """
    # SaaS schema: metric "engagement" lives in fct_engagement.engagement_score;
    # metric "mrr" lives in fct_revenue.mrr. MCAR may introduce None values,
    # so drop nulls before computing the pairwise correlation.
    eng = saas_tables["fct_engagement"][["date_key", "company_id", "engagement_score"]]
    rev = saas_tables["fct_revenue"][["date_key", "company_id", "mrr"]]
    joined = eng.merge(rev, on=["date_key", "company_id"], how="inner")
    e = pd.to_numeric(joined["engagement_score"], errors="coerce")
    m = pd.to_numeric(joined["mrr"], errors="coerce")
    mask = e.notna() & m.notna()
    obs_corr = float(np.corrcoef(e[mask].values, m[mask].values)[0, 1])
    delivered = _delivered_coefficient(saas_cfg, "engagement", "mrr")
    assert abs(obs_corr - delivered) < 0.30, (
        f"observed engagement/mrr correlation {obs_corr:.3f} too far from "
        f"delivered {delivered:.3f}"
    )


def test_empty_correlations_skips_psd_check():
    """FIX-01: a config with correlations=[] generates without invoking PSD.

    Empty correlations is a legitimate "metrics are independent" config; the
    PSD gate must be a no-op in that case (no spurious raise on degenerate
    matrix shapes).
    """
    cfg = _minimal_config(correlations=[])
    tables = generate_tables(cfg, _rng(0))
    assert "dim_date" in tables  # generation completed at least to dims


# --- Helper: minimal config for PSD tests ------------------------------------


def _minimal_config(correlations, *, skip_validation: bool = False):
    # skip_validation=True builds via ``model_construct`` to bypass the
    # load-time PSD validator (FIX-F04). Used by the two tests that must
    # construct a known-bad matrix to exercise ``validate_correlation_psd``
    # and the redundant gate at the top of ``generate_tables`` directly.
    kwargs = dict(
        domain=Domain(name="n", description="d", entity_type="e", entity_label="E"),
        time_window=TimeWindow(start="2024-01", end="2024-06", granularity="monthly"),
        seed=1,
        metrics=[
            Metric(
                name="a", label="A", distribution="normal",
                params={"mu": 0.0, "sigma": 1.0},
                polarity="positive",
            ),
            Metric(
                name="b", label="B", distribution="normal",
                params={"mu": 0.0, "sigma": 1.0},
                polarity="positive",
            ),
            Metric(
                name="c", label="C", distribution="normal",
                params={"mu": 0.0, "sigma": 1.0},
                polarity="positive",
            ),
        ],
        archetypes=[
            Archetype(
                name="flat", label="Flat", description="-",
                curve_segments=[
                    CurveSegment(curve="plateau", params={"level": 0.5},
                                 start_pct=0.0, end_pct=1.0),
                ],
            ),
        ],
        entities=[Entity(name="e1", archetype="flat", size=1)],
        tables=[
            Table(
                name="dim_date", type="dim", grain="per_period",
                columns=[Column(name="date_key", dtype="id", source="pk")],
                primary_key="date_key",
            ),
        ],
        correlations=correlations,
        noise=NoiseConfig(),
        output=OutputConfig(format="csv", directory="out"),
    )
    if skip_validation:
        return PlotsimConfig.model_construct(**kwargs)
    return PlotsimConfig(**kwargs)


# --- FIX-05 / MF-2: temporal coherence ---------------------------------------


def _temporal_cfg(
    allow_outside_window: bool = False,
) -> PlotsimConfig:
    """Minimal config with a dim_employee carrying a hire_date column."""
    return PlotsimConfig(
        domain=Domain(name="t", description="t", entity_type="e", entity_label="Es"),
        time_window=TimeWindow(start="2023-01", end="2023-12", granularity="monthly"),
        seed=0,
        metrics=[
            Metric(
                name="m", label="m", distribution="beta",
                params={"alpha": 2.0, "beta": 2.0}, polarity="positive",
                value_range={"min": 0.0, "max": 1.0},
            ),
        ],
        archetypes=[
            Archetype(
                name="x", label="x", description="x",
                curve_segments=[
                    CurveSegment(curve="plateau", params={"level": 0.5},
                                 start_pct=0.0, end_pct=1.0),
                ],
            ),
        ],
        entities=[Entity(name="e1", archetype="x", size=1)],
        tables=[
            Table(
                name="dim_date", type="dim", grain="per_period",
                columns=[Column(name="date_key", dtype="id", source="pk")],
                primary_key="date_key",
            ),
            Table(
                name="dim_employee", type="dim", grain="per_entity",
                columns=[
                    Column(name="employee_id", dtype="id", source="pk"),
                    Column(
                        name="hire_date", dtype="date",
                        source="generated:faker.date",
                        allow_outside_window=allow_outside_window,
                    ),
                ],
                primary_key="employee_id",
            ),
        ],
        noise=NoiseConfig(),
        output=OutputConfig(format="csv", directory="out"),
    )


def test_temporal_coherence_validator_warns_on_out_of_range():
    """FIX-05 / MF-2: hire dates outside time_window → warning."""
    import datetime as _dt
    cfg = _temporal_cfg(allow_outside_window=False)
    dim_employee = pd.DataFrame({
        "employee_id": ["e-001", "e-002", "e-003"],
        # First is inside the 2023 window, others are well outside.
        "hire_date": [
            _dt.date(2023, 6, 1),
            _dt.date(1995, 3, 4),
            _dt.date(2030, 11, 7),
        ],
    })
    tables = {"dim_date": pd.DataFrame({"date_key": []}), "dim_employee": dim_employee}
    issues = validate_temporal_coherence(cfg, tables)
    assert len(issues) == 1
    issue = issues[0]
    assert issue.check == CHECK_TEMPORAL_COHERENCE
    assert issue.severity == "warning"
    assert issue.table == "dim_employee"
    assert issue.details["out_of_range_count"] == 2


def test_temporal_coherence_allows_outside_window_when_marked():
    """FIX-05 / MF-2: allow_outside_window=true suppresses the warning."""
    import datetime as _dt
    cfg = _temporal_cfg(allow_outside_window=True)
    dim_employee = pd.DataFrame({
        "employee_id": ["e-001", "e-002"],
        "hire_date": [_dt.date(1995, 3, 4), _dt.date(2030, 11, 7)],
    })
    tables = {"dim_date": pd.DataFrame({"date_key": []}), "dim_employee": dim_employee}
    assert validate_temporal_coherence(cfg, tables) == []


def test_temporal_coherence_clean_on_hr_sample(hr_cfg, hr_tables):
    """FIX-05: the shipped HR template (post-fix) has every hire_date in-window."""
    issues = validate_temporal_coherence(hr_cfg, hr_tables)
    assert issues == [], f"HR template should be temporally coherent, got {issues}"
