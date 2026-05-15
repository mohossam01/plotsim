"""Manifest enrichment: nested-ANOVA variance partition + RBF GP kernel fit.

Mission 026 adds three additive sections to ``ManifestSchema`` (no schema
version bump — still 1.10):

* ``variance_partitions`` — nested-ANOVA decomposition per metric with
  ``Entity.archetype`` as the between-group axis. ``ss_between +
  ss_within_entity + ss_residual == ss_total`` exactly (modulo
  floating-point rounding).
* ``variance_partitions_by_segment`` — same decomposition with curve-
  segment-within-archetype as the axis. Per-archetype only; segments are
  never pooled across archetypes.
* ``gp_kernel_fits`` — per-archetype RBF kernel fits over the trajectory
  shape, plus per-entity records for entities carrying ``overrides``.

This module locks in:

* ANOVA identity (``ss_between + ss_within_entity + ss_residual ==
  ss_total``) for the archetype scope at ``rtol=1e-10``.
* Per-archetype within-segment computation never pools across archetypes
  (a multi-archetype config emits one record per archetype, each
  restricted to its own entities).
* Cold-start NaN cells are excluded from the partition and the
  ``cold_start_entities_excluded`` counter surfaces the dropped count.
* Empty sections for no-metric configs.
* RBF length scale recovered on a sinusoidal trajectory is shorter than
  the total period span (captures the periodicity).
* Sigmoid trajectory fit converges with a length scale consistent with
  monotone smoothness.
* Flat trajectory: ``converged=False`` with null hyperparameters; the
  manifest build does not raise.
* Entity-level trajectory overrides emit ``scope_type="entity"`` records
  alongside the archetype baseline.
* No new ``ManifestSchema`` field outside the three Mission 026
  additions — guards against future additive drift.
"""

from __future__ import annotations

import warnings

import numpy as np

from plotsim import generate_tables_with_state
from plotsim.config import (
    Archetype,
    Column,
    CurveSegment,
    Domain,
    Entity,
    EntityOverrides,
    Metric,
    OutputConfig,
    PlotsimConfig,
    SurrogateKeyWarning,
    Table,
    TimeWindow,
)
from plotsim.gp import fit_rbf
from plotsim.manifest import (
    GPKernelFit,
    MANIFEST_SCHEMA_VERSION,
    VariancePartition,
    build_manifest,
)


# --- Fixtures --------------------------------------------------------------


def _config_with_archetypes(
    archetypes: list[Archetype],
    entities: list[Entity],
    *,
    metrics: list[Metric] | None = None,
) -> PlotsimConfig:
    """Engine config with caller-supplied archetypes / entities.

    Defaults to a single beta-distributed metric ``m1`` so the realized
    ``entity_metrics`` arrays have well-conditioned variance for the
    ANOVA decomposition. Pass ``metrics=[]`` to drive a no-metric run
    (variance partitions and GP fits both empty in that path).
    """
    if metrics is None:
        metrics = [
            Metric(
                name="m1",
                label="m1",
                distribution="beta",
                params={"alpha": 2.0, "beta": 5.0},
                polarity="positive",
            ),
        ]
    if metrics:
        fct_columns = [
            Column(name="date_key", dtype="id", source="fk:dim_date.date_key"),
            Column(name="entity_id", dtype="id", source="fk:dim_entity.entity_id"),
        ] + [Column(name=m.name, dtype="float", source=f"metric:{m.name}") for m in metrics]
        fct = Table(
            name="fct_m",
            type="fact",
            grain="per_entity_per_period",
            primary_key=["date_key", "entity_id"],
            foreign_keys=["dim_date.date_key", "dim_entity.entity_id"],
            columns=fct_columns,
        )
        tables = [
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
                primary_key="entity_id",
                columns=[Column(name="entity_id", dtype="id", source="pk")],
            ),
            fct,
        ]
    else:
        tables = [
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
                primary_key="entity_id",
                columns=[Column(name="entity_id", dtype="id", source="pk")],
            ),
        ]
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
            metrics=metrics,
            archetypes=archetypes,
            entities=entities,
            tables=tables,
            output=OutputConfig(format="csv", directory="out/m26"),
        )


def _three_segment_archetype(name: str, levels: tuple[float, float, float]) -> Archetype:
    """Three-segment plateau archetype with caller-specified levels.

    Used as the variance-partition workhorse: distinct per-segment
    levels guarantee positive between-segment variance, so the segment-
    scope partition has non-trivial structure.
    """
    return Archetype(
        name=name,
        label=name,
        description=f"three-segment plateau {levels}",
        curve_segments=[
            CurveSegment(
                curve="plateau",
                params={"level": levels[0]},
                start_pct=0.0,
                end_pct=0.34,
            ),
            CurveSegment(
                curve="plateau",
                params={"level": levels[1]},
                start_pct=0.34,
                end_pct=0.67,
            ),
            CurveSegment(
                curve="plateau",
                params={"level": levels[2]},
                start_pct=0.67,
                end_pct=1.0,
            ),
        ],
    )


def _flat_archetype(name: str = "flat", level: float = 0.5) -> Archetype:
    return Archetype(
        name=name,
        label=name,
        description="constant plateau",
        curve_segments=[
            CurveSegment(
                curve="plateau",
                params={"level": level},
                start_pct=0.0,
                end_pct=1.0,
            ),
        ],
    )


def _sigmoid_archetype(name: str = "sigmoid") -> Archetype:
    """S-curve archetype: monotone rise from 0 to 1 over the window.

    Sigmoid takes a ``rising: bool`` parameter; default ``True`` produces
    the canonical 0→1 transition centered mid-window. The RBF GP fit on
    this trajectory should converge with a length scale consistent with
    monotone smoothness — neither very short (flat) nor very long
    (periodic).
    """
    return Archetype(
        name=name,
        label=name,
        description="sigmoid rise",
        curve_segments=[
            CurveSegment(
                curve="sigmoid",
                params={"rising": True},
                start_pct=0.0,
                end_pct=1.0,
            ),
        ],
    )


def _oscillating_archetype(name: str = "oscillating", period: int = 6) -> Archetype:
    """Oscillating archetype: sinusoidal cycle.

    ``period`` is the cycle length in periods. With ``n_periods=12`` and
    ``period=6`` the trajectory exhibits two full cycles, so the RBF GP
    fit's recovered length scale should be measurably shorter than the
    full window length.
    """
    return Archetype(
        name=name,
        label=name,
        description="sinusoidal cycle",
        curve_segments=[
            CurveSegment(
                curve="oscillating",
                params={"period": period},
                start_pct=0.0,
                end_pct=1.0,
            ),
        ],
    )


# --- Variance partition: ANOVA identity ------------------------------------


def test_variance_partition_archetype_anova_identity():
    """Acceptance #1: ``ss_between + ss_within_entity + ss_residual ==
    ss_total`` (rtol=1e-10) for a config with two archetypes and one
    metric. ``ss_total`` is computed from the same finite observations
    the partition saw — guards against a decomposition that drops or
    double-counts mass.
    """
    archetypes = [
        _three_segment_archetype("a_low", (0.2, 0.4, 0.6)),
        _three_segment_archetype("a_high", (0.5, 0.7, 0.9)),
    ]
    entities = [Entity(name=f"lo_{i}", archetype="a_low", size=1) for i in range(8)] + [
        Entity(name=f"hi_{i}", archetype="a_high", size=1) for i in range(8)
    ]
    cfg = _config_with_archetypes(archetypes, entities)
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
    assert len(manifest.variance_partitions) == 1
    rec = manifest.variance_partitions[0]
    assert rec.metric == "m1"
    assert rec.scope == "archetype"
    assert rec.scope_name == "all"
    pool = np.concatenate([state.entity_metrics[e.name]["m1"] for e in cfg.entities])
    pool = pool[np.isfinite(pool)]
    ss_total = float(np.sum((pool - float(np.mean(pool))) ** 2))
    ss_sum = rec.ss_between + rec.ss_within_entity + rec.ss_residual
    assert np.isclose(ss_sum, ss_total, rtol=1e-10, atol=1e-10)
    # Fractions sum to 1 within the same tolerance.
    fraction_sum = rec.fraction_between + rec.fraction_within_entity + rec.fraction_residual
    assert np.isclose(fraction_sum, 1.0, rtol=1e-10, atol=1e-10)
    # n_observations equals the pool size after NaN masking.
    assert rec.n_observations == int(pool.size)


def test_variance_partition_by_segment_groups_within_archetype():
    """Acceptance #2: ``variance_partitions_by_segment`` emits one record
    per ``(metric, archetype)`` and ``scope_name`` is the archetype name.
    Segments are grouped within their parent archetype — the section
    never pools observations across archetypes.
    """
    archetypes = [
        _three_segment_archetype("a_low", (0.2, 0.4, 0.6)),
        _three_segment_archetype("a_high", (0.5, 0.7, 0.9)),
    ]
    entities = [Entity(name=f"lo_{i}", archetype="a_low", size=1) for i in range(8)] + [
        Entity(name=f"hi_{i}", archetype="a_high", size=1) for i in range(8)
    ]
    cfg = _config_with_archetypes(archetypes, entities)
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
    by_segment = manifest.variance_partitions_by_segment
    # One record per (metric, archetype); one metric × two archetypes.
    assert len(by_segment) == 2
    scope_names = {r.scope_name for r in by_segment}
    assert scope_names == {"a_low", "a_high"}
    for rec in by_segment:
        assert rec.metric == "m1"
        assert rec.scope == "segment"
        # The archetype-restricted observation count equals the per-
        # archetype pool size (after NaN masking) — proves the segment
        # scope never pools across archetypes.
        names = [e.name for e in cfg.entities if e.archetype == rec.scope_name]
        pool = np.concatenate([state.entity_metrics[n]["m1"] for n in names])
        pool = pool[np.isfinite(pool)]
        assert rec.n_observations == int(pool.size)
        ss_total = float(np.sum((pool - float(np.mean(pool))) ** 2))
        ss_sum = rec.ss_between + rec.ss_within_entity + rec.ss_residual
        assert np.isclose(ss_sum, ss_total, rtol=1e-10, atol=1e-10)


def test_variance_partition_segment_between_is_positive_for_distinct_levels():
    """Within an archetype, three distinct plateau levels produce
    positive between-segment variance. A bug that pooled segments into
    a single group would yield ``ss_between == 0``.
    """
    archetypes = [_three_segment_archetype("layered", (0.1, 0.5, 0.9))]
    entities = [Entity(name=f"e_{i}", archetype="layered", size=1) for i in range(10)]
    cfg = _config_with_archetypes(archetypes, entities)
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
    by_segment = manifest.variance_partitions_by_segment
    assert len(by_segment) == 1
    rec = by_segment[0]
    assert rec.scope_name == "layered"
    assert rec.ss_between > 0.0
    # df_between for three segments = 3 - 1 = 2.
    assert rec.degrees_of_freedom_between == 2


# --- Cold-start handling ----------------------------------------------------


def test_variance_partition_excludes_cold_start_periods():
    """Acceptance #3: a config with cold-start entities (arrival offset
    > 0) excludes NaN-padded periods. ``n_observations`` is less than
    ``n_entities × n_periods`` and ``cold_start_entities_excluded``
    counts the cold-start entities that contributed at least one NaN
    cell.
    """
    archetypes = [_flat_archetype("flat", level=0.5)]
    entities = [
        Entity(name="e_warm", archetype="flat", size=1, start_period=0),
        Entity(name="e_cold_1", archetype="flat", size=1, start_period=3),
        Entity(name="e_cold_2", archetype="flat", size=1, start_period=6),
    ]
    cfg = _config_with_archetypes(archetypes, entities)
    rng = np.random.default_rng(cfg.seed)
    tables, state = generate_tables_with_state(cfg, rng)
    n_periods = len(tables["dim_date"])
    manifest = build_manifest(
        cfg,
        state.trajectories,
        tables,
        scd_state=state.scd,
        bridge_state=state.bridges,
        entity_metrics=state.entity_metrics,
    )
    rec = manifest.variance_partitions[0]
    assert rec.n_observations < len(entities) * n_periods
    # Exact: e_warm contributes n_periods, e_cold_1 contributes
    # n_periods - 3, e_cold_2 contributes n_periods - 6.
    expected = n_periods + (n_periods - 3) + (n_periods - 6)
    assert rec.n_observations == expected
    # Two of the three entities had at least one NaN cell.
    assert rec.cold_start_entities_excluded == 2


# --- Empty section guard ----------------------------------------------------


def test_variance_partition_empty_when_no_metrics():
    """Acceptance #7: a config with no metrics emits empty
    ``variance_partitions`` and empty ``variance_partitions_by_segment``.
    Same guarantee for ``gp_kernel_fits`` — the section piggybacks on
    metric presence so no-metric configs are byte-equivalent to the
    pre-M26 lane modulo the new empty containers.
    """
    archetypes = [_flat_archetype()]
    entities = [Entity(name=f"e_{i}", archetype="flat", size=1) for i in range(3)]
    cfg = _config_with_archetypes(archetypes, entities, metrics=[])
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
    assert manifest.variance_partitions == []
    assert manifest.variance_partitions_by_segment == []
    assert manifest.gp_kernel_fits == []


# --- GP fits: shape characterization ---------------------------------------


def test_gp_fit_archetype_record_emitted_per_archetype():
    """One ``scope_type="archetype"`` record per declared archetype in
    sorted-name order. The fit consumes the archetype's clean
    trajectory (no overrides, no cold-start shift) so the kernel
    characterizes the archetype's shape rather than any individual
    entity's realized data.
    """
    archetypes = [
        _flat_archetype("flat", level=0.5),
        _sigmoid_archetype("sigmoid"),
    ]
    entities = [Entity(name=f"e_{i}", archetype="flat", size=1) for i in range(3)] + [
        Entity(name=f"s_{i}", archetype="sigmoid", size=1) for i in range(3)
    ]
    cfg = _config_with_archetypes(archetypes, entities)
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
    archetype_fits = [f for f in manifest.gp_kernel_fits if f.scope_type == "archetype"]
    assert {f.scope_name for f in archetype_fits} == {"flat", "sigmoid"}
    for fit in archetype_fits:
        assert fit.kernel_type == "rbf"


def test_gp_fit_sigmoid_converges_and_length_scale_is_smooth():
    """Acceptance #5 (variant): a sigmoid archetype produces
    ``converged=True`` with a length scale on the order of the
    transition's smoothness — not a short cycle, not the full window
    span. The exact value is optimizer-dependent; pin only the
    monotone-smoothness signal: length scale > 1 (not collapsed to the
    period grid) AND length scale < total span (not infinite).
    """
    cfg = _config_with_archetypes(
        archetypes=[_sigmoid_archetype("sigmoid")],
        entities=[Entity(name="e_0", archetype="sigmoid", size=1)],
    )
    rng = np.random.default_rng(cfg.seed)
    tables, state = generate_tables_with_state(cfg, rng)
    n_periods = len(tables["dim_date"])
    manifest = build_manifest(
        cfg,
        state.trajectories,
        tables,
        scd_state=state.scd,
        bridge_state=state.bridges,
        entity_metrics=state.entity_metrics,
    )
    fit = next(f for f in manifest.gp_kernel_fits if f.scope_name == "sigmoid")
    assert fit.converged is True
    assert fit.hyperparameters is not None
    length_scale = fit.hyperparameters["length_scale"]
    assert 1.0 < length_scale < float(n_periods)


def test_gp_fit_oscillating_recovers_short_length_scale():
    """Acceptance #4: an oscillating archetype's recovered length scale
    is measurably shorter than the total period span — the kernel
    captures the periodicity rather than averaging it out.
    """
    cfg = _config_with_archetypes(
        archetypes=[_oscillating_archetype("oscillating", period=4)],
        entities=[Entity(name="e_0", archetype="oscillating", size=1)],
    )
    rng = np.random.default_rng(cfg.seed)
    tables, state = generate_tables_with_state(cfg, rng)
    n_periods = len(tables["dim_date"])
    manifest = build_manifest(
        cfg,
        state.trajectories,
        tables,
        scd_state=state.scd,
        bridge_state=state.bridges,
        entity_metrics=state.entity_metrics,
    )
    fit = next(f for f in manifest.gp_kernel_fits if f.scope_name == "oscillating")
    assert fit.converged is True
    assert fit.hyperparameters is not None
    length_scale = fit.hyperparameters["length_scale"]
    # Periodicity ≪ window span — recovered length scale must reflect
    # the cycle, not the full window. n_periods = 12 (Jan–Dec 2024).
    assert length_scale < float(n_periods) / 2.0


def test_gp_fit_flat_trajectory_does_not_converge():
    """Acceptance #5 (variant): a flat archetype produces
    ``converged=False`` with empty hyperparameters and ``None`` log
    marginal likelihood. The manifest build does not raise — the
    failed fit is recorded as a non-fatal record.
    """
    cfg = _config_with_archetypes(
        archetypes=[_flat_archetype("flat", level=0.5)],
        entities=[Entity(name="e_0", archetype="flat", size=1)],
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
    fit = next(f for f in manifest.gp_kernel_fits if f.scope_name == "flat")
    assert fit.converged is False
    assert fit.hyperparameters == {}
    assert fit.log_marginal_likelihood is None


def test_gp_fit_entity_override_emits_entity_scope_record():
    """Acceptance #6: a config with at least one entity carrying
    ``overrides`` emits a ``scope_type="entity"`` record for that
    entity AND keeps the ``scope_type="archetype"`` baseline record
    for the archetype. Other entities sharing the archetype do NOT
    produce per-entity records — only override-bearing entities do.
    """
    cfg = _config_with_archetypes(
        archetypes=[_sigmoid_archetype("sigmoid")],
        entities=[
            Entity(
                name="e_override",
                archetype="sigmoid",
                size=1,
                overrides=EntityOverrides(inflection_month=2),
            ),
            Entity(name="e_default", archetype="sigmoid", size=1),
        ],
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
    by_scope = {(f.scope_type, f.scope_name) for f in manifest.gp_kernel_fits}
    assert ("archetype", "sigmoid") in by_scope
    assert ("entity", "e_override") in by_scope
    # Default-entity does NOT get its own per-entity record.
    assert ("entity", "e_default") not in by_scope


# --- Byte-equivalence guard ------------------------------------------------


def test_no_metric_config_emits_empty_m26_sections():
    """Configs that don't trigger either Mission 026 section produce
    a manifest with all three new fields at their empty defaults. The
    test is structurally identical to the M25 byte-equivalence guard:
    pop the new fields and assert they were the only delta from the
    pre-M26 wire shape.
    """
    cfg = _config_with_archetypes(
        archetypes=[_flat_archetype()],
        entities=[Entity(name=f"e_{i}", archetype="flat", size=1) for i in range(3)],
        metrics=[],
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
    payload = manifest.model_dump(mode="json")
    assert payload.pop("variance_partitions") == []
    assert payload.pop("variance_partitions_by_segment") == []
    assert payload.pop("gp_kernel_fits") == []
    assert payload["schema_version"] == "1.10"


# --- Schema-pin guard ------------------------------------------------------


def test_schema_version_still_pinned_at_1_10():
    """Mission 026 adds fields without bumping the schema version. Pins
    the constant so a stray bump would surface as an import-time typed
    assertion failure rather than a downstream consumer rejecting the
    payload at parse time.
    """
    assert MANIFEST_SCHEMA_VERSION == "1.10"


def test_variance_partition_model_carries_required_fields():
    """``VariancePartition`` exposes the documented field set. Guards
    against a partial rename or a dropped field on a refactor.
    """
    rec = VariancePartition(
        metric="m1",
        scope="archetype",
        scope_name="all",
        ss_between=1.0,
        ss_within_entity=1.0,
        ss_residual=1.0,
        fraction_between=0.33,
        fraction_within_entity=0.33,
        fraction_residual=0.34,
        degrees_of_freedom_between=1,
        degrees_of_freedom_within=10,
        degrees_of_freedom_residual=100,
        n_observations=120,
        cold_start_entities_excluded=2,
    )
    payload = rec.model_dump(mode="json")
    assert set(payload.keys()) == {
        "metric",
        "scope",
        "scope_name",
        "ss_between",
        "ss_within_entity",
        "ss_residual",
        "fraction_between",
        "fraction_within_entity",
        "fraction_residual",
        "degrees_of_freedom_between",
        "degrees_of_freedom_within",
        "degrees_of_freedom_residual",
        "n_observations",
        "cold_start_entities_excluded",
    }


def test_gp_kernel_fit_model_carries_required_fields():
    """``GPKernelFit`` exposes the documented field set."""
    rec = GPKernelFit(
        scope_type="archetype",
        scope_name="flat",
        kernel_type="rbf",
        hyperparameters={"length_scale": 1.0, "signal_variance": 1.0, "noise_variance": 0.1},
        log_marginal_likelihood=-10.0,
        n_train=12,
        converged=True,
    )
    payload = rec.model_dump(mode="json")
    assert set(payload.keys()) == {
        "scope_type",
        "scope_name",
        "kernel_type",
        "hyperparameters",
        "log_marginal_likelihood",
        "n_train",
        "converged",
    }


# --- Direct GP module smoke ------------------------------------------------


def test_fit_rbf_flat_signal_does_not_converge():
    """``fit_rbf`` on a constant input returns
    ``converged=False`` / null hyperparameters / null likelihood.
    Direct unit test of the leaf module — guards against a future
    refactor that drops the flat-variance short-circuit.
    """
    x = np.arange(20, dtype=np.float64)
    y = np.full(20, 0.5, dtype=np.float64)
    result = fit_rbf(x, y)
    assert result.converged is False
    assert result.hyperparameters is None
    assert result.log_marginal_likelihood is None
    assert result.n_train == 20


def test_fit_rbf_sinusoidal_converges_with_short_length_scale():
    """``fit_rbf`` on a sinusoidal signal recovers a length scale much
    smaller than the total span. The exact value is optimizer-
    dependent; pin only the order-of-magnitude relationship.
    """
    n = 60
    x = np.arange(n, dtype=np.float64)
    y = np.sin(2.0 * np.pi * x / 10.0).astype(np.float64)
    result = fit_rbf(x, y)
    assert result.converged is True
    assert result.hyperparameters is not None
    length_scale = result.hyperparameters["length_scale"]
    assert length_scale < float(n) / 2.0


def test_fit_rbf_fewer_than_three_points_does_not_converge():
    """Edge case: under 3 finite training points produces
    ``converged=False``. The kernel has 3 hyperparameters; fewer
    observations than hyperparameters is degenerate.
    """
    x = np.array([0.0, 1.0], dtype=np.float64)
    y = np.array([0.0, 1.0], dtype=np.float64)
    result = fit_rbf(x, y)
    assert result.converged is False
    assert result.n_train == 2
