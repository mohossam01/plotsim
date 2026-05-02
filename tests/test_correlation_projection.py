"""M111 — Higham nearest-PD correlation projection.

Covers the project-and-warn flow that replaced FIX-F04's hard raise on
non-PD correlation matrices:

  * Pure algorithm (`_higham_nearest_pd`, `_eigvalue_clip_to_pd`,
    `project_correlation_matrix`) — PD passthrough, projection
    properties, fidelity vs. baselines, convergence on adversarial
    inputs, fallback path.
  * Adjustment record + warning text format — determinism + spec format.
  * Load-time validator integration — warning fires once on
    YAML/programmatic construction, `PlotsimConfig._correlation_adjustments`
    is populated.
  * Manifest integration — `correlation_adjustments` populated on
    projection runs, `None` on already-PD runs.
  * End-to-end determinism — same seed + same projected config →
    byte-identical fact-table values.
  * Bundled-template byte-identity guarantee — every shipped template
    has a PD matrix, so `correlation_adjustments` stays `None` and
    output stays byte-identical to pre-M111.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

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
    Table,
    TimeWindow,
    load_config,
)
from plotsim.manifest import build_manifest
from plotsim.metrics import (
    _correlation_adjustment_records,
    _eigvalue_clip_to_pd,
    _ensure_pd_margin,
    _format_correlation_adjustment_warning,
    _higham_nearest_pd,
    project_correlation_matrix,
)
from plotsim.tables import generate_tables
from plotsim.validation import (
    project_correlation_or_issue,
    validate_correlation_psd,
)


SAAS_YAML = "plotsim/configs/sample_saas.yaml"
HR_YAML = "plotsim/configs/sample_hr.yaml"
EDU_YAML = "plotsim/configs/sample_education.yaml"
RETAIL_YAML = "plotsim/configs/sample_retail.yaml"
MARKETING_YAML = "plotsim/configs/sample_marketing.yaml"
ALL_TEMPLATES = [SAAS_YAML, HR_YAML, EDU_YAML, RETAIL_YAML, MARKETING_YAML]


def _three_cycle_pairs() -> list[CorrelationPair]:
    """Classic non-PD 3-cycle: (a,b)=0.9, (b,c)=0.9, (a,c)=-0.9.

    The Frobenius-nearest correlation matrix for this input is
    [[1, 0.5, -0.5], [0.5, 1, 0.5], [-0.5, 0.5, 1]] — a textbook
    Higham worked example.
    """
    return [
        CorrelationPair(metric_a="a", metric_b="b", coefficient=0.9),
        CorrelationPair(metric_a="b", metric_b="c", coefficient=0.9),
        CorrelationPair(metric_a="a", metric_b="c", coefficient=-0.9),
    ]


def _three_metric_config(
    correlations: list[CorrelationPair],
    *,
    skip_validation: bool = False,
) -> PlotsimConfig:
    kwargs = dict(
        domain=Domain(name="n", description="d", entity_type="e", entity_label="E"),
        time_window=TimeWindow(start="2024-01", end="2024-06", granularity="monthly"),
        seed=1,
        metrics=[
            Metric(name="a", label="A", distribution="normal",
                   params={"mu": 0.0, "sigma": 1.0}, polarity="positive"),
            Metric(name="b", label="B", distribution="normal",
                   params={"mu": 0.0, "sigma": 1.0}, polarity="positive"),
            Metric(name="c", label="C", distribution="normal",
                   params={"mu": 0.0, "sigma": 1.0}, polarity="positive"),
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


# --- Pure algorithm ---------------------------------------------------------


class TestPDPassthrough:
    """Identity / already-PD inputs return unchanged — byte-identical."""

    def test_identity_matrix_passes_through_unchanged(self):
        mat = np.eye(5)
        out, used, fallback = project_correlation_matrix(mat)
        assert used is False
        assert fallback is False
        assert np.array_equal(out, mat)

    def test_pd_matrix_passes_through_unchanged(self):
        # Construct a guaranteed-PD 4×4 by random-symmetric-positive-definite:
        # B^T @ B with random B is always PSD, +epsilon*I makes it PD.
        rng = np.random.default_rng(0)
        B = rng.standard_normal((4, 4))
        cov = B.T @ B + 0.5 * np.eye(4)
        d = np.sqrt(np.diag(cov))
        mat = cov / np.outer(d, d)
        np.fill_diagonal(mat, 1.0)
        np.linalg.cholesky(mat)  # confirm PD
        out, used, fallback = project_correlation_matrix(mat)
        assert used is False
        assert fallback is False
        # Byte-identical: the pre-M111 contract for already-PD inputs.
        assert np.array_equal(out, mat)


class TestProjectionProperties:
    """Properties every projected matrix must satisfy."""

    def test_three_cycle_projects_to_textbook_solution(self):
        # Higham's classic worked example: 0.9 / 0.9 / -0.9 on a 3-cycle
        # has Frobenius-nearest correlation matrix at exactly 0.5 / 0.5 / -0.5.
        mat = np.array([
            [1.0, 0.9, -0.9],
            [0.9, 1.0, 0.9],
            [-0.9, 0.9, 1.0],
        ])
        out, used, fallback = project_correlation_matrix(mat)
        assert used is True
        assert fallback is False
        # Match to ~1e-8; the alternating projections converges to a known
        # closed-form here.
        np.testing.assert_allclose(
            out,
            [[1.0, 0.5, -0.5], [0.5, 1.0, 0.5], [-0.5, 0.5, 1.0]],
            atol=1e-6,
        )

    def test_projected_matrix_is_pd_with_margin(self):
        mat = np.array([
            [1.0, 0.9, -0.9],
            [0.9, 1.0, 0.9],
            [-0.9, 0.9, 1.0],
        ])
        out, _used, _fallback = project_correlation_matrix(mat)
        min_eig = float(np.linalg.eigvalsh(out).min())
        assert min_eig > 1e-10

    def test_projected_matrix_has_unit_diagonal(self):
        mat = np.array([
            [1.0, 0.9, -0.9],
            [0.9, 1.0, 0.9],
            [-0.9, 0.9, 1.0],
        ])
        out, _used, _fallback = project_correlation_matrix(mat)
        np.testing.assert_array_equal(np.diag(out), [1.0, 1.0, 1.0])

    def test_projected_matrix_is_symmetric(self):
        mat = np.array([
            [1.0, 0.9, -0.9],
            [0.9, 1.0, 0.9],
            [-0.9, 0.9, 1.0],
        ])
        out, _used, _fallback = project_correlation_matrix(mat)
        np.testing.assert_allclose(out, out.T, atol=1e-12)

    def test_projected_off_diagonals_in_open_unit_interval(self):
        mat = np.array([
            [1.0, 0.9, -0.9],
            [0.9, 1.0, 0.9],
            [-0.9, 0.9, 1.0],
        ])
        out, _used, _fallback = project_correlation_matrix(mat)
        n = out.shape[0]
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                assert -1.0 < out[i, j] < 1.0

    def test_cholesky_succeeds_on_projected_matrix(self):
        # Adversarial input: Higham must produce a Cholesky-able output
        # for a matrix where vanilla Cholesky fails.
        mat = np.array([
            [1.0, 0.99, 0.99, -0.99],
            [0.99, 1.0, 0.99, -0.99],
            [0.99, 0.99, 1.0, 0.99],
            [-0.99, -0.99, 0.99, 1.0],
        ])
        with pytest.raises(np.linalg.LinAlgError):
            np.linalg.cholesky(mat)
        out, used, _fallback = project_correlation_matrix(mat)
        assert used is True
        np.linalg.cholesky(out)  # must not raise


class TestFrobeniusOptimality:
    """Higham is the Frobenius-nearest correlation matrix.

    No tighter PD correlation matrix exists than the one Higham returns.
    Verified empirically by comparing against eigenvalue-clipping (a
    cruder, non-optimal projection).
    """

    def test_higham_beats_eigenvalue_clipping_on_three_cycle(self):
        # Compare Higham's iterate to eigenvalue-clipping at the same
        # numerical precision. ``project_correlation_matrix`` post-
        # processes Higham's result via ``_ensure_pd_margin`` to push
        # the minimum eigenvalue strictly above tol; we compare the raw
        # Higham iterate here so the test measures Frobenius optimality
        # of the projection itself, not the margin-lift post-process.
        mat = np.array([
            [1.0, 0.9, -0.9],
            [0.9, 1.0, 0.9],
            [-0.9, 0.9, 1.0],
        ])
        higham, converged = _higham_nearest_pd(mat)
        assert converged is True
        clipped = _eigvalue_clip_to_pd(mat, tol=1e-10)
        higham_dist = float(np.linalg.norm(higham - mat, ord="fro"))
        clipped_dist = float(np.linalg.norm(clipped - mat, ord="fro"))
        # Higham's Frobenius distance must be ≤ eigenvalue-clipping's.
        # Equality is rare but allowed numerically; strict less-than-or-equal
        # is the spec.
        assert higham_dist <= clipped_dist + 1e-10


class TestFidelityOnSpecCases:
    """Mission-spec fidelity bounds."""

    def test_pre_tuning_saas_coefs_mean_adjustment_under_010(self):
        # Mission spec acceptance: pre-tuning saas coefs (0.82, -0.75, 0.68)
        # → mean per-pair adjustment < 0.10 after Higham projection.
        # The actual saas template structure is a sparse 4×4 chain (not
        # a dense 3-cycle): engagement↔mrr, engagement↔churn_risk,
        # support_tickets↔churn_risk. Off-diagonals not on those edges
        # are 0.0. Reconstruct that sparse pre-tuning matrix here.
        names = ["engagement", "mrr", "churn_risk", "support_tickets"]
        idx = {n: i for i, n in enumerate(names)}
        mat = np.eye(4)
        for a, b, c in [
            ("engagement", "mrr", 0.82),
            ("engagement", "churn_risk", -0.75),
            ("support_tickets", "churn_risk", 0.68),
        ]:
            i, j = idx[a], idx[b]
            mat[i, j] = c
            mat[j, i] = c
        # If this matrix is already PD, the test premise is wrong.
        try:
            np.linalg.cholesky(mat)
            already_pd = True
        except np.linalg.LinAlgError:
            already_pd = False
        assert not already_pd, "pre-tuning saas matrix should be non-PD"
        out, used, _fallback = project_correlation_matrix(mat)
        assert used is True
        deltas = [
            abs(out[idx["engagement"], idx["mrr"]] - 0.82),
            abs(out[idx["engagement"], idx["churn_risk"]] - (-0.75)),
            abs(out[idx["support_tickets"], idx["churn_risk"]] - 0.68),
        ]
        mean_delta = sum(deltas) / 3
        assert mean_delta < 0.10, f"mean per-pair adjustment {mean_delta:.4f} ≥ 0.10"

    def test_twenty_metric_moderate_pairs_mean_adjustment_under_005(self):
        # Mission spec: 20-metric matrix with 12 moderate (≈0.4) pairs
        # → mean per-pair adjustment < 0.05. Construct deterministically.
        n = 20
        mat = np.eye(n)
        rng = np.random.default_rng(42)
        idx_pairs = []
        attempts = 0
        while len(idx_pairs) < 12 and attempts < 200:
            i, j = sorted(rng.integers(0, n, size=2).tolist())
            if i == j or (i, j) in idx_pairs:
                attempts += 1
                continue
            mat[i, j] = 0.4
            mat[j, i] = 0.4
            idx_pairs.append((i, j))
            attempts += 1
        # Whether PD or not, run projection and verify spec bound.
        out, _used, _fallback = project_correlation_matrix(mat)
        deltas = [abs(out[i, j] - mat[i, j]) for (i, j) in idx_pairs]
        mean_delta = sum(deltas) / len(deltas)
        assert mean_delta < 0.05, f"mean per-pair adjustment {mean_delta:.4f} ≥ 0.05"


class TestConvergence:
    """Algorithm converges within 200 iterations for matrices ≤ 50×50.

    The mission's adversarial test: 50×50 with all off-diagonals at 0.95.
    """

    def test_converges_on_50x50_adversarial(self):
        # Mission's adversarial spec was "all off-diagonals at 0.95", but a
        # 50×50 with that structure has min eigenvalue 0.05 (PD by far —
        # rank-1 update of identity). To genuinely stress the algorithm,
        # construct a random asymmetric 50×50 with extreme correlations
        # forced into many cells, then symmetrize. This produces a
        # genuinely non-PD matrix that exercises Higham's iteration count.
        rng = np.random.default_rng(0)
        n = 50
        off = rng.uniform(-0.95, 0.95, size=(n, n))
        # Inject conflicts: force triangle violations in 5 random triplets.
        for _ in range(5):
            i, j, k = rng.choice(n, size=3, replace=False)
            off[i, j] = off[j, i] = 0.95
            off[j, k] = off[k, j] = 0.95
            off[i, k] = off[k, i] = -0.95
        mat = (off + off.T) / 2.0
        np.fill_diagonal(mat, 1.0)
        # Sanity: matrix must be non-PD (the test premise).
        with pytest.raises(np.linalg.LinAlgError):
            np.linalg.cholesky(mat)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)  # any fallback warning fails the test
            out, used, fallback = project_correlation_matrix(
                mat, max_iter=200, tol=1e-10,
            )
        assert used is True
        assert fallback is False
        # Result is PD with margin.
        assert float(np.linalg.eigvalsh(out).min()) > 1e-10

    def test_eigenvalue_clipping_fallback_emits_warning_on_pathological(self):
        # Pathological input that forces the fallback: construct a
        # matrix that won't iterate-stabilize within 5 steps. We do this
        # via direct injection by passing `max_iter=1`.
        mat = np.array([
            [1.0, 0.9, -0.9],
            [0.9, 1.0, 0.9],
            [-0.9, 0.9, 1.0],
        ])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out, used, fallback = project_correlation_matrix(
                mat, max_iter=1, tol=1e-12,
            )
        assert used is True
        assert fallback is True
        # Cholesky succeeds.
        np.linalg.cholesky(out)
        # Warning fires.
        msgs = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
        assert any("eigenvalue-clipping" in m for m in msgs)


class TestDeterminism:
    """Algorithm is deterministic — same input → same output, every time."""

    def test_same_input_same_output(self):
        mat = np.array([
            [1.0, 0.9, -0.9],
            [0.9, 1.0, 0.9],
            [-0.9, 0.9, 1.0],
        ])
        a, _, _ = project_correlation_matrix(mat)
        b, _, _ = project_correlation_matrix(mat)
        c, _, _ = project_correlation_matrix(mat)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(b, c)

    def test_higham_iteration_deterministic(self):
        rng = np.random.default_rng(7)
        n = 6
        # Random symmetric off-diagonals in [-0.9, 0.9].
        off = rng.uniform(-0.9, 0.9, size=(n, n))
        mat = (off + off.T) / 2.0
        np.fill_diagonal(mat, 1.0)
        a, _ = _higham_nearest_pd(mat, max_iter=100)
        b, _ = _higham_nearest_pd(mat, max_iter=100)
        np.testing.assert_array_equal(a, b)


class TestMarginNudge:
    """`_ensure_pd_margin` lifts boundary projections above tol cleanly."""

    def test_margin_nudge_no_op_when_already_pd(self):
        mat = np.eye(3)
        out = _ensure_pd_margin(mat, tol=1e-10)
        # Identity has min eig 1.0 — well above tol, returned unchanged.
        np.testing.assert_array_equal(out, mat)

    def test_margin_nudge_lifts_below_tol_eigenvalue(self):
        # A matrix with a near-zero eigenvalue.
        mat = np.array([
            [1.0, 1.0],
            [1.0, 1.0],
        ])
        # Eigenvalues: 0 and 2. After nudge, min eig should exceed 1e-10.
        out = _ensure_pd_margin(mat, tol=1e-8)
        assert float(np.linalg.eigvalsh(out).min()) >= 1e-8 - 1e-12
        # Diagonal stays at 1 (correlation matrix invariant).
        np.testing.assert_array_almost_equal(np.diag(out), [1.0, 1.0])


# --- Adjustment records + warning text ---------------------------------------


class TestAdjustmentRecords:

    def test_records_dropped_below_noise_floor(self):
        # PD passthrough → empty records.
        mat = np.eye(2)
        records = _correlation_adjustment_records(
            mat, mat,
            metrics=[
                Metric(name="a", label="A", distribution="normal",
                       params={"mu": 0.0, "sigma": 1.0}, polarity="positive"),
                Metric(name="b", label="B", distribution="normal",
                       params={"mu": 0.0, "sigma": 1.0}, polarity="positive"),
            ],
            correlations=[
                CorrelationPair(metric_a="a", metric_b="b", coefficient=0.0),
            ],
        )
        assert records == []

    def test_records_sorted_deterministically(self):
        # Build a minimal config, project, check record order.
        cfg = _three_metric_config(_three_cycle_pairs())
        records = cfg._correlation_adjustments
        assert records is not None
        names = [(r["metric_a"], r["metric_b"]) for r in records]
        assert names == sorted(names)

    def test_records_only_for_requested_pairs(self):
        cfg = _three_metric_config(_three_cycle_pairs())
        records = cfg._correlation_adjustments
        assert records is not None
        pair_keys = {(r["metric_a"], r["metric_b"]) for r in records}
        # Three user-specified pairs, all adjusted.
        assert pair_keys == {("a", "b"), ("a", "c"), ("b", "c")}


class TestWarningFormat:
    """Warning text format pinned by mission spec."""

    def test_warning_text_includes_every_adjusted_pair(self):
        records = [
            {"metric_a": "a", "metric_b": "b",
             "requested": 0.9, "achieved": 0.5, "adjustment": 0.4},
            {"metric_a": "b", "metric_b": "c",
             "requested": 0.9, "achieved": 0.5, "adjustment": 0.4},
        ]
        text = _format_correlation_adjustment_warning(records)
        assert "a ↔ b" in text
        assert "b ↔ c" in text
        assert "Adjusted 2 pairs" in text
        assert "0.9000 → 0.5000" in text
        assert "Δ0.4000" in text

    def test_warning_text_deterministic(self):
        records = [
            {"metric_a": "a", "metric_b": "b",
             "requested": 0.9, "achieved": 0.5, "adjustment": 0.4},
        ]
        a = _format_correlation_adjustment_warning(records)
        b = _format_correlation_adjustment_warning(records)
        assert a == b


# --- Load-time validator integration -----------------------------------------


class TestLoadTimeValidator:

    def test_pd_config_loads_without_warning(self):
        # Education has a PD correlation matrix post-M112 (the saas YAML's
        # correlations got reverted to non-PD originals so Higham now fires
        # on saas — pick a known-PD template for this test).
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cfg = load_config(EDU_YAML)
        m111_warnings = [
            w for w in caught
            if issubclass(w.category, UserWarning)
            and "Correlation matrix was not positive definite" in str(w.message)
        ]
        assert m111_warnings == []
        assert cfg._correlation_adjustments is None

    def test_non_pd_config_load_warns_and_records(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cfg = _three_metric_config(_three_cycle_pairs())
        m111_warnings = [
            w for w in caught
            if issubclass(w.category, UserWarning)
            and "Correlation matrix was not positive definite" in str(w.message)
        ]
        assert len(m111_warnings) == 1
        assert cfg._correlation_adjustments is not None
        assert len(cfg._correlation_adjustments) == 3

    def test_validate_correlation_psd_returns_no_issues_on_projection(self):
        # The post-generation diagnostic shouldn't flag a successfully-
        # projected matrix as a config defect.
        cfg = _three_metric_config(_three_cycle_pairs())
        issues = validate_correlation_psd(cfg)
        assert issues == []

    def test_validate_correlation_psd_returns_no_issues_on_pd(self):
        cfg = load_config(SAAS_YAML)
        assert validate_correlation_psd(cfg) == []

    def test_no_correlations_returns_no_issues(self):
        cfg = _three_metric_config([])
        assert validate_correlation_psd(cfg) == []
        assert cfg._correlation_adjustments is None

    def test_empty_correlations_no_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cfg = _three_metric_config([])
        m111_warnings = [
            w for w in caught
            if issubclass(w.category, UserWarning)
            and "Correlation matrix" in str(w.message)
        ]
        assert m111_warnings == []
        assert cfg._correlation_adjustments is None


# --- project_correlation_or_issue ------------------------------------------


class TestProjectOrIssue:

    def test_pd_returns_no_issues_no_records(self):
        # Education has a PD matrix post-M112 (saas reverted to non-PD).
        cfg = load_config(EDU_YAML)
        issues, adjustments, projected = project_correlation_or_issue(cfg)
        assert issues == []
        assert adjustments is None
        assert projected is None

    def test_non_pd_returns_records_and_matrix(self):
        # Build via skip_validation so the load-time validator doesn't
        # consume the projection state for us.
        cfg = _three_metric_config(_three_cycle_pairs(), skip_validation=True)
        issues, adjustments, projected = project_correlation_or_issue(cfg)
        assert issues == []
        assert adjustments is not None
        assert len(adjustments) == 3
        assert projected is not None
        assert projected.shape == (3, 3)
        assert float(np.linalg.eigvalsh(projected).min()) > 1e-10


# --- Manifest integration ---------------------------------------------------


class TestManifestIntegration:

    def test_manifest_correlation_adjustments_none_for_pd_template(self):
        # Education is the canonical PD bundled template post-M112.
        cfg = load_config(EDU_YAML)
        # Manifest doesn't need real trajectories for this assertion;
        # zeros suffice.
        n_periods = cfg.time_window.period_count()
        trajectories = {e.name: np.zeros(n_periods) for e in cfg.entities}
        mfst = build_manifest(cfg, trajectories, {})
        assert mfst.correlation_adjustments is None

    def test_manifest_correlation_adjustments_populated_on_projection(self):
        cfg = _three_metric_config(_three_cycle_pairs())
        n_periods = cfg.time_window.period_count()
        trajectories = {e.name: np.zeros(n_periods) for e in cfg.entities}
        mfst = build_manifest(cfg, trajectories, {})
        assert mfst.correlation_adjustments is not None
        assert len(mfst.correlation_adjustments) == 3
        for adj in mfst.correlation_adjustments:
            assert adj.metric_a in {"a", "b", "c"}
            assert adj.metric_b in {"a", "b", "c"}
            assert adj.adjustment > 0


# --- End-to-end determinism + bundled-template byte identity ---------------


class TestEndToEndDeterminism:

    def test_projection_run_is_byte_deterministic_across_runs(self):
        # Same config + same seed → byte-identical fact tables, even when
        # projection runs. Use saas template mutated to a non-PD matrix on
        # the first three metrics; the three-metric helper-config has only
        # dim_date and no fact tables to compare against.
        import yaml
        with open(SAAS_YAML) as f:
            data = yaml.safe_load(f)
        names = [m["name"] for m in data["metrics"]][:3]
        data["correlations"] = [
            {"metric_a": names[0], "metric_b": names[1], "coefficient": 0.9},
            {"metric_a": names[1], "metric_b": names[2], "coefficient": 0.9},
            {"metric_a": names[0], "metric_b": names[2], "coefficient": -0.9},
        ]
        # Suppress projection warnings — irrelevant to this test.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            cfg1 = PlotsimConfig(**data)
            cfg2 = PlotsimConfig(**data)
            t1 = generate_tables(cfg1)
            t2 = generate_tables(cfg2)
        # Compare every fact-table column.
        fact_names = [
            t.name for t in cfg1.tables
            if t.type == "fact"
        ]
        assert fact_names, "saas template should have fact tables"
        for name in fact_names:
            df1, df2 = t1[name], t2[name]
            assert df1.equals(df2), (
                f"non-byte-identical fact table {name} across runs with same "
                f"seed"
            )

    @pytest.mark.parametrize("path", [EDU_YAML, RETAIL_YAML])
    def test_pd_bundled_templates_have_no_correlation_adjustments(self, path):
        # Education and retail are PD by construction post-M112. Higham must
        # not fire on them — projection passthrough is byte-identical.
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            cfg = load_config(path)
        assert cfg._correlation_adjustments is None

    @pytest.mark.parametrize("path", [SAAS_YAML, HR_YAML, MARKETING_YAML])
    def test_non_pd_bundled_templates_record_correlation_adjustments(self, path):
        # M112 reverted saas/hr correlations to original intended values
        # (now slightly non-PD) and introduced marketing with five
        # intentionally non-PD pairs. Higham projects all three at load
        # time and stashes the per-pair record on ``_correlation_adjustments``
        # (a list of dicts, mirroring what surfaces in manifest.json).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            cfg = load_config(path)
        assert cfg._correlation_adjustments is not None
        assert len(cfg._correlation_adjustments) >= 1
        for adj in cfg._correlation_adjustments:
            assert adj["adjustment"] > 0
            assert -1.0 <= adj["achieved"] <= 1.0
