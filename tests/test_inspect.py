"""Tests for plotsim.inspect — single-cell pipeline trace.

Coverage matches the M113 Phase 1 mission spec:

    - trace_metric_cell returns a populated TraceResult on the saas fixed point
    - result.realized_cell equals the fact-table cell to floating-point equality
      (the load-bearing assertion)
    - independent_draw != correlated_draw for in-matrix metrics; equal for out
    - mcar_fired=True ⇒ noised_value None, clamped_value None, realized_cell NaN
    - causal_lag_driver populated for support_tickets, None for mrr
    - exception types raised on bad input with clear messages
    - determinism across consecutive calls
    - no RNG side effects on subsequent generate_tables_with_state calls
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from plotsim.config import load_config
from plotsim.inspect import (
    EntityNotFound,
    MetricNotFound,
    PeriodOutOfRange,
    TraceResult,
    trace_metric_cell,
)
from plotsim.tables import generate_tables_with_state


REPO_ROOT = Path(__file__).resolve().parent.parent
SAAS_CONFIG = REPO_ROOT / "plotsim" / "configs" / "sample_saas.yaml"


@pytest.fixture(scope="module")
def saas_cfg():
    # Higham projection fires on saas (engagement↔churn_risk |Δ| ≈ 0.117) and
    # raises a UserWarning at config load. Suppress here so test output stays
    # focused on assertions; M111's projection behavior is its own test surface.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return load_config(SAAS_CONFIG)


def test_trace_returns_populated_traceresult(saas_cfg):
    result = trace_metric_cell(saas_cfg, "acme_corp_cohort", 12, "mrr", seed=42)
    assert isinstance(result, TraceResult)
    assert result.entity_name == "acme_corp_cohort"
    assert result.archetype_name == "rocket_then_cliff"
    assert result.period_index == 12
    assert result.metric_name == "mrr"
    assert 0.0 <= result.trajectory_position <= 1.0
    assert 0.0 <= result.effective_position <= 1.0
    assert result.polarity == "positive"
    assert result.distribution_family == "lognorm"
    assert result.distribution_center > 0.0
    assert result.independent_draw == result.independent_draw  # not NaN
    assert result.correlated_draw == result.correlated_draw
    assert result.realized_cell is not None  # period 12 mrr is not MCAR


def test_realized_cell_matches_fact_table_floating_point_exact(saas_cfg):
    """Load-bearing assertion: trace_metric_cell.realized_cell must equal the
    corresponding cell in the generated fact table to bit-exact float equality.

    This is the §7 traceback assertion in the acceptance notebook — if it
    fails, every other audit downstream is meaningless because the trace
    isn't actually tracing the cell that landed in the dataset.
    """
    result = trace_metric_cell(saas_cfg, "acme_corp_cohort", 12, "mrr", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        tables, _ = generate_tables_with_state(
            saas_cfg, np.random.default_rng(42),
        )
    fct_revenue = tables["fct_revenue"]
    n_periods = 24
    # acme_corp_cohort is config.entities[0]; period 12; entity-major row order
    flat_idx = 0 * n_periods + 12
    expected = float(fct_revenue.iloc[flat_idx]["mrr"])
    assert result.realized_cell == expected


def test_clamped_equals_realized_across_random_cells(saas_cfg):
    """Stronger sanity: clamped_value == realized_cell for every non-MCAR cell
    on the saas grid. Catches replay desync that the single-cell test misses.
    """
    entities = [e.name for e in saas_cfg.entities]
    metrics = [m.name for m in saas_cfg.metrics]
    # Sample sparsely to keep the test fast — one period per entity per metric
    # is enough to surface any per-(entity, period) drift.
    for ent in entities:
        for p in (0, 7, 17, 23):
            for m in metrics:
                r = trace_metric_cell(saas_cfg, ent, p, m, seed=42)
                if r.mcar_fired:
                    assert r.realized_cell is None
                    continue
                assert r.realized_cell is not None
                assert r.clamped_value == r.realized_cell, (
                    f"replay desync at {ent}/{p}/{m}: "
                    f"clamped={r.clamped_value} realized={r.realized_cell}"
                )


@pytest.mark.parametrize("metric", ["mrr", "churn_risk", "support_tickets"])
def test_in_matrix_metrics_produce_finite_cells(saas_cfg, metric):
    """Metrics in the saas correlation matrix produce finite, non-bypass
    cells under the M127b copula.

    M127b version-boundary update: the pre-M127b assertion that
    ``independent_draw != correlated_draw`` no longer holds — the new
    copula draws Gaussians directly and there is no separate per-metric
    "independent" sample. ``independent_draw`` and ``correlated_draw``
    on the dataclass are now the same value on the correlation-active
    path (both the marginal value the new pipeline produced). This test
    keeps the bypass/finiteness regression guard the original test
    carried.
    """
    r = trace_metric_cell(saas_cfg, "acme_corp_cohort", 4, metric, seed=42)
    assert r.bypass_in_copula is False
    assert r.correlated_draw is not None
    assert math.isfinite(r.correlated_draw), (
        f"{metric}: copula produced non-finite cell {r.correlated_draw!r}"
    )


def test_engagement_passes_through_copula_as_first_in_toposort(saas_cfg):
    """engagement is the causal_lag driver for support_tickets, so it sits at
    toposort position 0. Under M127b's flip, the dataclass surfaces
    ``independent_draw == correlated_draw`` for every metric on the
    correlation-active path (single source draw); the test is kept as a
    regression guard against the dataclass shape changing.
    """
    r = trace_metric_cell(saas_cfg, "acme_corp_cohort", 4, "engagement", seed=42)
    assert math.isclose(
        r.independent_draw, r.correlated_draw, abs_tol=1e-9,
    )


@pytest.mark.parametrize("metric", ["feature_adoption", "nps"])
def test_non_pair_metrics_produce_finite_cells(saas_cfg, metric):
    """Metrics not in any correlation pair still produce finite cells.

    M127b version-boundary update: the pre-M127b assertion that out-of-
    matrix metrics pass through the copula as the float-precision
    identity (``independent_draw ≈ correlated_draw``) no longer applies —
    the new pipeline draws Gaussians for every metric and runs the same
    family transform on each. The realized value is still a valid sample
    from the metric's marginal; that's the surviving contract.
    """
    r = trace_metric_cell(saas_cfg, "acme_corp_cohort", 12, metric, seed=42)
    assert r.correlated_draw is not None
    assert math.isfinite(r.correlated_draw), (
        f"{metric}: copula produced non-finite cell {r.correlated_draw!r}"
    )


def test_mcar_fired_nullifies_downstream(saas_cfg):
    """When MCAR fires, noised_value and clamped_value must be None and the
    fact-table cell must be NaN.

    M127b version-boundary update: the cell discovered via grid scan
    moved with the copula reformulation. Pre-M127b: acme_corp_cohort
    period 17 metric engagement. Post-M127b: globex_cohort period 2
    metric engagement under seed 42 is an MCAR-fired cell.
    """
    r = trace_metric_cell(saas_cfg, "globex_cohort", 2, "engagement", seed=42)
    assert r.mcar_fired is True
    assert r.noised_value is None
    assert r.clamped_value is None
    assert r.realized_cell is None  # _resolve_realized_cell returns None on NaN

    # Cross-check: the actual fact-table cell at this position is NaN.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        tables, _ = generate_tables_with_state(
            saas_cfg, np.random.default_rng(42),
        )
    fct_engagement = tables["fct_engagement"]
    cell = fct_engagement.iloc[1 * 24 + 2]["engagement_score"]
    assert pd.isna(cell)


def test_causal_lag_driver_populated_for_support_tickets(saas_cfg):
    r = trace_metric_cell(
        saas_cfg, "acme_corp_cohort", 12, "support_tickets", seed=42,
    )
    assert r.causal_lag_driver == "engagement"
    assert r.causal_lag_blend_weight is not None
    assert 0.0 < r.causal_lag_blend_weight <= 1.0


def test_causal_lag_driver_none_for_mrr(saas_cfg):
    r = trace_metric_cell(saas_cfg, "acme_corp_cohort", 12, "mrr", seed=42)
    assert r.causal_lag_driver is None
    assert r.causal_lag_blend_weight is None


def test_entity_not_found_raises(saas_cfg):
    with pytest.raises(EntityNotFound, match="not in config.entities"):
        trace_metric_cell(saas_cfg, "no_such_cohort", 0, "mrr", seed=42)


def test_period_out_of_range_high_raises(saas_cfg):
    with pytest.raises(PeriodOutOfRange, match="outside"):
        trace_metric_cell(saas_cfg, "acme_corp_cohort", 999, "mrr", seed=42)


def test_period_out_of_range_negative_raises(saas_cfg):
    with pytest.raises(PeriodOutOfRange, match="outside"):
        trace_metric_cell(saas_cfg, "acme_corp_cohort", -1, "mrr", seed=42)


def test_metric_not_found_raises(saas_cfg):
    with pytest.raises(MetricNotFound, match="not in config.metrics"):
        trace_metric_cell(saas_cfg, "acme_corp_cohort", 0, "no_such_metric", seed=42)


def test_determinism_across_consecutive_calls(saas_cfg):
    a = trace_metric_cell(saas_cfg, "acme_corp_cohort", 12, "mrr", seed=42)
    b = trace_metric_cell(saas_cfg, "acme_corp_cohort", 12, "mrr", seed=42)
    assert a == b


def test_no_rng_side_effects_on_subsequent_engine_run(saas_cfg):
    """trace_metric_cell must not perturb a subsequent generate_tables_with_state
    call's output. Pass 1 inside trace creates its own RNG from ``effective_seed``
    rather than consuming from a caller-provided RNG, so the engine state is
    untouched. This test pins that contract.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        tables_before, _ = generate_tables_with_state(
            saas_cfg, np.random.default_rng(42),
        )
        _ = trace_metric_cell(saas_cfg, "acme_corp_cohort", 12, "mrr", seed=42)
        tables_after, _ = generate_tables_with_state(
            saas_cfg, np.random.default_rng(42),
        )
    pd.testing.assert_frame_equal(
        tables_before["fct_revenue"], tables_after["fct_revenue"],
    )


def test_seed_default_uses_config_seed(saas_cfg):
    """seed=None must reproduce the engine's default-seeded run."""
    r_default = trace_metric_cell(saas_cfg, "acme_corp_cohort", 12, "mrr")
    r_explicit = trace_metric_cell(
        saas_cfg, "acme_corp_cohort", 12, "mrr", seed=saas_cfg.seed,
    )
    assert r_default == r_explicit


def test_seed_override_changes_output(saas_cfg):
    """Different seed ⇒ same trajectory (deterministic from archetype) but
    different metric values. Verifies the seed parameter is wired through.
    """
    r42 = trace_metric_cell(saas_cfg, "acme_corp_cohort", 12, "mrr", seed=42)
    r43 = trace_metric_cell(saas_cfg, "acme_corp_cohort", 12, "mrr", seed=43)
    assert r42.trajectory_position == r43.trajectory_position
    assert r42.realized_cell != r43.realized_cell
