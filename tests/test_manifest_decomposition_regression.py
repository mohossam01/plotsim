"""Manifest enrichment: seasonal-decomposition snapshot + per-pair regression.

Schema 1.10 adds three additive sections to ``ManifestSchema``:

* ``seasonal_decomposition`` — the global per-period strength array plus
  per-metric and per-entity sensitivity dicts. Configs without
  ``seasonal_effects`` emit the empty-sentinel shape (empty list / empty
  dicts) — the engine short-circuits on those configs and recording
  the inert multipliers would just be noise.
* ``regression_pairs_global`` — pair-wise OLS β + intercept (both
  directions), r², per-direction residual variance, and finite-cell
  count for every declared correlation pair, pooled across all
  entities.
* ``regression_pairs_by_archetype`` — the same OLS surface restricted
  to the entity subset of each archetype. Archetypes that contribute
  no finite observations are omitted entirely.

This module locks in:

* Seasonal factors match ``_build_seasonal_factors`` output value-for-
  value, and the sensitivity dicts mirror the config.
* No-seasonality configs emit the empty-sentinel shape.
* Pooled OLS coefficients match a manual ``np.linalg.lstsq`` fit on the
  same observations (rtol=1e-6).
* Both directions of β are emitted for every declared pair.
* Cold-start NaN cells are filtered out of the regression — fits still
  succeed with the surviving observations, and ``n_observations``
  reports the surviving count.
* Configs without correlations / seasonality produce manifests
  byte-equivalent to pre-1.10 modulo the schema version string and the
  three new empty-sentinel containers.
* The schema version pin is ``"1.10"``.
"""

from __future__ import annotations

import warnings

import numpy as np

from plotsim import generate_tables_with_state
from plotsim.config import (
    Archetype,
    Column,
    CorrelationPair,
    CurveSegment,
    Domain,
    Entity,
    Metric,
    OutputConfig,
    PlotsimConfig,
    SeasonalEffect,
    SurrogateKeyWarning,
    Table,
    TimeWindow,
)
from plotsim.manifest import (
    MANIFEST_SCHEMA_VERSION,
    RegressionPair,
    SeasonalDecomposition,
    build_manifest,
)
from plotsim.tables import _build_seasonal_factors


# --- Fixtures --------------------------------------------------------------


def _two_metric_config(
    *,
    entities: list[Entity] | None = None,
    archetypes: list[Archetype] | None = None,
    correlations: list[CorrelationPair] | None = None,
    seasonal_effects: list[SeasonalEffect] | None = None,
    metric_seasonal_sensitivity: tuple[float, float] = (1.0, 1.0),
) -> PlotsimConfig:
    """Two-metric engine config with one flat archetype.

    Returns realized cell values in [0, 1] (beta distribution) so a
    pooled OLS fit on entity_metrics has well-conditioned inputs.
    ``correlations`` defaults to empty (no copula at draw time); pass a
    non-empty list to engage the manifest's regression sections.
    """
    if archetypes is None:
        archetypes = [
            Archetype(
                name="flat",
                label="flat",
                description="constant 0.5 plateau",
                curve_segments=[
                    CurveSegment(
                        curve="plateau",
                        params={"level": 0.5},
                        start_pct=0.0,
                        end_pct=1.0,
                    ),
                ],
            ),
        ]
    m1 = Metric(
        name="m1",
        label="m1",
        distribution="beta",
        params={"alpha": 2.0, "beta": 5.0},
        polarity="positive",
        seasonal_sensitivity=metric_seasonal_sensitivity[0],
    )
    m2 = Metric(
        name="m2",
        label="m2",
        distribution="beta",
        params={"alpha": 2.0, "beta": 5.0},
        polarity="positive",
        seasonal_sensitivity=metric_seasonal_sensitivity[1],
    )
    fct = Table(
        name="fct_m",
        type="fact",
        grain="per_entity_per_period",
        primary_key=["date_key", "entity_id"],
        foreign_keys=["dim_date.date_key", "dim_entity.entity_id"],
        columns=[
            Column(name="date_key", dtype="id", source="fk:dim_date.date_key"),
            Column(name="entity_id", dtype="id", source="fk:dim_entity.entity_id"),
            Column(name="m1", dtype="float", source="metric:m1"),
            Column(name="m2", dtype="float", source="metric:m2"),
        ],
    )
    dim_date = Table(
        name="dim_date",
        type="dim",
        grain="per_period",
        primary_key="date_key",
        columns=[
            Column(name="date_key", dtype="id", source="pk"),
            Column(name="date", dtype="date", source="generated:date_key"),
        ],
    )
    dim_entity = Table(
        name="dim_entity",
        type="dim",
        grain="per_entity",
        primary_key="entity_id",
        columns=[
            Column(name="entity_id", dtype="id", source="pk"),
        ],
    )
    if entities is None:
        entities = [Entity(name=f"e_{i}", archetype="flat", size=1) for i in range(10)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SurrogateKeyWarning)
        return PlotsimConfig(
            domain=Domain(
                name="t",
                description="t",
                entity_type="entity",
                entity_label="Entities",
            ),
            time_window=TimeWindow(
                start="2024-01",
                end="2024-12",
                granularity="monthly",
            ),
            seed=0,
            metrics=[m1, m2],
            archetypes=archetypes,
            entities=entities,
            tables=[dim_date, dim_entity, fct],
            correlations=correlations or [],
            seasonal_effects=seasonal_effects or [],
            output=OutputConfig(format="csv", directory="out/m25"),
        )


# --- Schema version pin ----------------------------------------------------


def test_schema_version_is_1_10():
    """Lock the schema version at ``"1.10"`` so downstream readers
    that pin on the constant catch the bump as a typed import-time
    failure rather than an at-runtime field-shape surprise.
    """
    assert MANIFEST_SCHEMA_VERSION == "1.10"


# --- Seasonal decomposition ------------------------------------------------


def test_seasonal_decomposition_matches_engine_factors():
    """The manifest's ``seasonal_factors`` array is value-for-value the
    output of ``_build_seasonal_factors`` for the same config — the
    helper is the engine's own source of truth, so the manifest snapshot
    must agree at the float level rather than at a normalized summary.
    """
    seasonal = [
        SeasonalEffect(months=(12, 1, 2), strength=0.8),
        SeasonalEffect(months=(6, 7), strength=-0.3),
    ]
    cfg = _two_metric_config(
        seasonal_effects=seasonal,
        metric_seasonal_sensitivity=(0.5, 1.5),
    )
    rng = np.random.default_rng(cfg.seed)
    tables, state = generate_tables_with_state(cfg, rng)
    manifest = build_manifest(
        cfg,
        state.trajectories,
        tables,
        scd_state=state.scd,
        bridge_state=state.bridges,
        entity_metrics=state.entity_metrics,
    )
    decomp = manifest.seasonal_decomposition
    assert isinstance(decomp, SeasonalDecomposition)
    expected = _build_seasonal_factors(cfg, n_periods=len(tables["dim_date"]))
    assert expected is not None
    np.testing.assert_allclose(
        np.asarray(decomp.seasonal_factors, dtype=np.float64),
        expected,
        rtol=0.0,
        atol=0.0,
    )
    assert decomp.metric_seasonal_sensitivities == {"m1": 0.5, "m2": 1.5}
    assert set(decomp.entity_seasonal_sensitivities.keys()) == {e.name for e in cfg.entities}
    # Per-entity sensitivities default to 1.0 — they were not overridden
    # in this fixture, so every value should equal 1.0.
    for v in decomp.entity_seasonal_sensitivities.values():
        assert v == 1.0


def test_seasonal_decomposition_empty_when_no_effects():
    """Configs without seasonality emit the empty-sentinel shape: an
    empty list for factors and empty dicts for both sensitivity maps.
    Anchors D3 — D3 picked empty-containers over a null sentinel so
    downstream consumers don't need a None-check before iterating.
    """
    cfg = _two_metric_config()
    rng = np.random.default_rng(cfg.seed)
    tables, state = generate_tables_with_state(cfg, rng)
    manifest = build_manifest(
        cfg,
        state.trajectories,
        tables,
        scd_state=state.scd,
        bridge_state=state.bridges,
        entity_metrics=state.entity_metrics,
    )
    decomp = manifest.seasonal_decomposition
    assert decomp.seasonal_factors == []
    assert decomp.metric_seasonal_sensitivities == {}
    assert decomp.entity_seasonal_sensitivities == {}


# --- Regression pair correctness -------------------------------------------


def _manual_ols(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Fit ``b = beta * a + intercept`` via ``np.linalg.lstsq``.

    Used as the cross-check oracle in
    ``test_regression_beta_matches_numpy_lstsq``. Returns
    ``(beta, intercept)`` so the assertion is direct.
    """
    design = np.column_stack([a, np.ones_like(a)])
    coeffs, *_ = np.linalg.lstsq(design, b, rcond=None)
    return float(coeffs[0]), float(coeffs[1])


def test_regression_beta_matches_numpy_lstsq():
    """Acceptance #3: the manifest's pooled β for a declared correlation
    pair matches a manual ``np.linalg.lstsq`` fit on the same
    ``entity_metrics`` arrays, both directions, within rtol=1e-6. This
    is the cross-check that ``_ols_pair``'s closed-form is right.
    """
    entities = [Entity(name=f"e_{i}", archetype="flat", size=1) for i in range(40)]
    cfg = _two_metric_config(
        entities=entities,
        correlations=[CorrelationPair(metric_a="m1", metric_b="m2", coefficient=0.7)],
    )
    rng = np.random.default_rng(cfg.seed)
    tables, state = generate_tables_with_state(cfg, rng)
    manifest = build_manifest(
        cfg,
        state.trajectories,
        tables,
        scd_state=state.scd,
        bridge_state=state.bridges,
        entity_metrics=state.entity_metrics,
    )
    assert len(manifest.regression_pairs_global) == 1
    rec = manifest.regression_pairs_global[0]
    assert (rec.metric_a, rec.metric_b) == ("m1", "m2")

    pool_a = np.concatenate([state.entity_metrics[e.name]["m1"] for e in cfg.entities])
    pool_b = np.concatenate([state.entity_metrics[e.name]["m2"] for e in cfg.entities])
    mask = np.isfinite(pool_a) & np.isfinite(pool_b)
    pool_a = pool_a[mask]
    pool_b = pool_b[mask]
    expected_beta_a_to_b, expected_int_a_to_b = _manual_ols(pool_a, pool_b)
    expected_beta_b_to_a, expected_int_b_to_a = _manual_ols(pool_b, pool_a)
    assert np.isclose(rec.beta_a_to_b, expected_beta_a_to_b, rtol=1e-6, atol=1e-9)
    assert np.isclose(rec.intercept_a_to_b, expected_int_a_to_b, rtol=1e-6, atol=1e-9)
    assert np.isclose(rec.beta_b_to_a, expected_beta_b_to_a, rtol=1e-6, atol=1e-9)
    assert np.isclose(rec.intercept_b_to_a, expected_int_b_to_a, rtol=1e-6, atol=1e-9)
    expected_r2 = float(np.corrcoef(pool_a, pool_b)[0, 1] ** 2)
    assert np.isclose(rec.r_squared, expected_r2, rtol=1e-6, atol=1e-9)
    assert rec.n_observations == int(pool_a.size)


def test_regression_emits_both_directions():
    """Both ``β_{a→b}`` and ``β_{b→a}`` are emitted, and (given non-
    degenerate variance) they're related by ``β_{a→b} * β_{b→a} ==
    r²``. The mathematical identity is the right invariant to pin —
    it would catch a copy-paste error that reused the same direction's
    β for both fields.
    """
    entities = [Entity(name=f"e_{i}", archetype="flat", size=1) for i in range(40)]
    cfg = _two_metric_config(
        entities=entities,
        correlations=[CorrelationPair(metric_a="m1", metric_b="m2", coefficient=0.7)],
    )
    rng = np.random.default_rng(cfg.seed)
    tables, state = generate_tables_with_state(cfg, rng)
    manifest = build_manifest(
        cfg,
        state.trajectories,
        tables,
        scd_state=state.scd,
        bridge_state=state.bridges,
        entity_metrics=state.entity_metrics,
    )
    rec = manifest.regression_pairs_global[0]
    product = rec.beta_a_to_b * rec.beta_b_to_a
    assert np.isclose(product, rec.r_squared, rtol=1e-6, atol=1e-9)


def test_regression_by_archetype_emits_per_archetype():
    """Acceptance #4: with two archetypes and a declared correlation
    pair, ``regression_pairs_by_archetype`` has one entry per archetype
    and each entry's pooled β is computed over only that archetype's
    entities.
    """
    archetypes = [
        Archetype(
            name="flat_low",
            label="flat_low",
            description="constant 0.3",
            curve_segments=[
                CurveSegment(
                    curve="plateau",
                    params={"level": 0.3},
                    start_pct=0.0,
                    end_pct=1.0,
                ),
            ],
        ),
        Archetype(
            name="flat_high",
            label="flat_high",
            description="constant 0.7",
            curve_segments=[
                CurveSegment(
                    curve="plateau",
                    params={"level": 0.7},
                    start_pct=0.0,
                    end_pct=1.0,
                ),
            ],
        ),
    ]
    entities = [Entity(name=f"lo_{i}", archetype="flat_low", size=1) for i in range(20)] + [
        Entity(name=f"hi_{i}", archetype="flat_high", size=1) for i in range(20)
    ]
    cfg = _two_metric_config(
        entities=entities,
        archetypes=archetypes,
        correlations=[CorrelationPair(metric_a="m1", metric_b="m2", coefficient=0.6)],
    )
    rng = np.random.default_rng(cfg.seed)
    tables, state = generate_tables_with_state(cfg, rng)
    manifest = build_manifest(
        cfg,
        state.trajectories,
        tables,
        scd_state=state.scd,
        bridge_state=state.bridges,
        entity_metrics=state.entity_metrics,
    )
    by_arch = manifest.regression_pairs_by_archetype
    assert set(by_arch.keys()) == {"flat_low", "flat_high"}
    for archetype_name in ("flat_low", "flat_high"):
        recs = by_arch[archetype_name]
        assert len(recs) == 1
        rec = recs[0]
        names = [e.name for e in cfg.entities if e.archetype == archetype_name]
        pool_a = np.concatenate([state.entity_metrics[n]["m1"] for n in names])
        pool_b = np.concatenate([state.entity_metrics[n]["m2"] for n in names])
        mask = np.isfinite(pool_a) & np.isfinite(pool_b)
        pool_a = pool_a[mask]
        pool_b = pool_b[mask]
        expected_beta, expected_int = _manual_ols(pool_a, pool_b)
        assert np.isclose(rec.beta_a_to_b, expected_beta, rtol=1e-6, atol=1e-9)
        assert np.isclose(rec.intercept_a_to_b, expected_int, rtol=1e-6, atol=1e-9)


def test_regression_pairs_empty_without_correlations():
    """Acceptance #5: configs without ``correlations`` emit empty
    regression sections. Mirrors the existing ``correlations`` section's
    contract — undeclared = no record.
    """
    cfg = _two_metric_config()
    rng = np.random.default_rng(cfg.seed)
    tables, state = generate_tables_with_state(cfg, rng)
    manifest = build_manifest(
        cfg,
        state.trajectories,
        tables,
        scd_state=state.scd,
        bridge_state=state.bridges,
        entity_metrics=state.entity_metrics,
    )
    assert manifest.regression_pairs_global == []
    assert manifest.regression_pairs_by_archetype == {}


def test_regression_nan_cells_are_skipped():
    """``n_observations`` reflects only finite ``(a, b)`` cells.
    Synthetic NaN injection into the realized ``entity_metrics`` (cells
    a downstream cold-start contract would produce as NaN) is masked
    out by ``_ols_pair`` — the fit still succeeds on the surviving
    observations and the count drops by the masked-cell count.
    """
    entities = [Entity(name=f"e_{i}", archetype="flat", size=1) for i in range(20)]
    cfg = _two_metric_config(
        entities=entities,
        correlations=[CorrelationPair(metric_a="m1", metric_b="m2", coefficient=0.5)],
    )
    rng = np.random.default_rng(cfg.seed)
    tables, state = generate_tables_with_state(cfg, rng)
    # Drop the first three periods on one entity to NaN — emulates a
    # cold-start lead-in pattern the regression layer must tolerate.
    contaminated = {
        ename: {metric: arr.copy() for metric, arr in per_metric.items()}
        for ename, per_metric in state.entity_metrics.items()
    }
    contaminated["e_0"]["m1"][:3] = np.nan
    n_periods = len(tables["dim_date"])
    expected_finite = len(entities) * n_periods - 3
    manifest = build_manifest(
        cfg,
        state.trajectories,
        tables,
        scd_state=state.scd,
        bridge_state=state.bridges,
        entity_metrics=contaminated,
    )
    rec = manifest.regression_pairs_global[0]
    assert rec.n_observations == expected_finite
    # Fit still produces a finite β under the mask.
    assert np.isfinite(rec.beta_a_to_b)
    assert np.isfinite(rec.beta_b_to_a)


# --- Byte-equivalence guard ------------------------------------------------


def test_byte_equivalent_to_pre_1_10_modulo_new_fields():
    """Configs without correlations or seasonality emit a 1.10 manifest
    whose serialized payload equals the equivalent pre-1.10 payload
    modulo (a) ``schema_version``, (b) the three new sentinel-shaped
    fields. Anchors the additive-only promise: 1.9 readers parsing a
    1.10 manifest see only the new fields' defaults — no existing
    field's value or ordering changed.
    """
    cfg = _two_metric_config()
    rng = np.random.default_rng(cfg.seed)
    tables, state = generate_tables_with_state(cfg, rng)
    manifest = build_manifest(
        cfg,
        state.trajectories,
        tables,
        scd_state=state.scd,
        bridge_state=state.bridges,
        entity_metrics=state.entity_metrics,
    )
    payload = manifest.model_dump(mode="json")
    # Strip the new and version fields; everything else must be empty
    # or a default produced by the pre-1.10 build path.
    assert payload.pop("schema_version") == "1.10"
    seasonal = payload.pop("seasonal_decomposition")
    assert seasonal == {
        "seasonal_factors": [],
        "metric_seasonal_sensitivities": {},
        "entity_seasonal_sensitivities": {},
    }
    assert payload.pop("regression_pairs_global") == []
    assert payload.pop("regression_pairs_by_archetype") == {}
    # The remaining keys are exactly the pre-1.10 field set. We don't
    # snapshot the full dict (that's what schema-pin tests cover); we
    # just assert the new fields are the only delta from the legacy
    # shape by checking nothing pre-1.10 went missing.
    for legacy_key in (
        "seed",
        "config_sha256",
        "archetype_assignments",
        "trajectory_samples",
        "event_firings",
        "causal_graph",
        "correlations",
        "noise_config",
    ):
        assert legacy_key in payload, f"legacy field missing from 1.10 payload: {legacy_key}"


# --- RegressionPair shape --------------------------------------------------


def test_regression_pair_carries_required_fields():
    """``RegressionPair`` exposes the ten fields the schema documents.
    Pins the model surface independently of the engine path — would
    catch a partial rename or a dropped field on a refactor.
    """
    rec = RegressionPair(
        metric_a="x",
        metric_b="y",
        beta_a_to_b=1.0,
        intercept_a_to_b=0.0,
        beta_b_to_a=1.0,
        intercept_b_to_a=0.0,
        r_squared=1.0,
        residual_variance_a_to_b=0.0,
        residual_variance_b_to_a=0.0,
        n_observations=10,
    )
    payload = rec.model_dump(mode="json")
    assert set(payload.keys()) == {
        "metric_a",
        "metric_b",
        "beta_a_to_b",
        "intercept_a_to_b",
        "beta_b_to_a",
        "intercept_b_to_a",
        "r_squared",
        "residual_variance_a_to_b",
        "residual_variance_b_to_a",
        "n_observations",
    }
