"""0.6-M24: per-metric treatment effects.

Extends the M8c treatment surface so a configured
``treatment_lift_log_odds`` can target a single named metric instead of
applying trajectory-wide. The per-metric gate lives in
``generate_metrics_for_period``: when ``treatment_target_metric`` is set,
only the metric whose name matches receives the shift; every other
metric sees ``treatment_shift=0.0`` and is byte-identical to its
control-arm draw.

Locks in:

* Default behaviour (``treatment_target_metric=None``) preserves
  pre-M24 output byte-for-byte. Every metric sees the lift, the same
  trajectory-wide behaviour M8c shipped.
* Per-metric targeting: with ``target_metric="m1"`` and a positive
  lift, only ``m1``'s realized values shift in the treatment cohort;
  ``m2`` stays at the control distribution. Under
  ``correlations=[]`` + zero-noise, the non-targeted metric is
  byte-identical between treatment and control arms (the strongest
  pin — no residual leakage at the sample-draw level).
* Correlation pipeline does not leak the lift to a correlated
  non-targeted metric. With ``target_metric="m1"`` and a strong
  baseline correlation ``m1 ↔ m2``, ``m2``'s post-treatment mean
  remains within statistical noise of its control mean — the copula
  operates on residuals around each metric's own (un-shifted) center,
  so the lift on ``m1`` does not propagate.
* Validator rejects ``target_metric`` values that do not match any
  declared metric name — same silent-dead-weight failure mode that
  M8c's ``treatment_start_period >= n_periods`` gate closes.
* Manifest emits ``target_metric`` per entity AND per cohort. Schema
  bumped 1.8 → 1.9 (additive; defaults to ``None``, so pre-M24 readers
  parse 1.9 manifests cleanly except for the new field).
* Builder ``TreatmentConfig.target_metric`` propagates to every
  expanded entity in the segment (treatment AND control arms — the
  field is harmless on control entities because they have no lift to
  gate, and carrying it on both arms preserves ground-truth symmetry).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from pydantic import ValidationError

from plotsim import create, generate_tables_with_state
from plotsim.builder.input import TreatmentConfig
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
    SurrogateKeyWarning,
    Table,
    TimeWindow,
)
from plotsim.manifest import MANIFEST_SCHEMA_VERSION, build_manifest


# --- Helpers ---------------------------------------------------------------


def _two_metric_engine_config(
    entities: list[Entity],
    *,
    correlations: list[CorrelationPair] | None = None,
) -> PlotsimConfig:
    """Engine-direct config with two metrics on a flat archetype.

    Mirrors ``_engine_config`` from ``test_treatment_control.py`` but
    declares two metrics so the per-metric gate has something to gate
    against. Both metrics use a beta distribution so the realized
    values lie in [0,1] and the logit shift's effect is cleanly
    recoverable via a difference of means.
    """
    arch = Archetype(
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
    )
    m1 = Metric(
        name="m1",
        label="m1",
        distribution="beta",
        params={"alpha": 2.0, "beta": 5.0},
        polarity="positive",
    )
    m2 = Metric(
        name="m2",
        label="m2",
        distribution="beta",
        params={"alpha": 2.0, "beta": 5.0},
        polarity="positive",
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
            archetypes=[arch],
            entities=entities,
            tables=[dim_date, dim_entity, fct],
            correlations=correlations or [],
            output=OutputConfig(format="csv", directory="out/m24"),
        )


def _split_cohort_means(
    tables: dict, ctrl_pks: set, trt_pks: set
) -> dict[str, tuple[float, float]]:
    """Return ``{metric_name: (ctrl_mean, trt_mean)}`` over the fact table."""
    fct = tables["fct_m"]
    ctrl_rows = fct[fct["entity_id"].isin(ctrl_pks)]
    trt_rows = fct[fct["entity_id"].isin(trt_pks)]
    return {
        col: (float(ctrl_rows[col].mean()), float(trt_rows[col].mean())) for col in ("m1", "m2")
    }


# --- Default behaviour ------------------------------------------------------


def test_entity_default_target_metric_is_none():
    """Sanity: the new field is opt-in. An ``Entity`` constructed
    without ``treatment_target_metric`` carries ``None`` — the no-op
    default that drops the entity into the pre-M24 trajectory-wide
    lane.
    """
    e = Entity(name="x", archetype="flat", size=1)
    assert e.treatment_target_metric is None


def test_target_metric_none_lifts_every_metric():
    """The pre-M24 contract: with ``treatment_target_metric=None`` and
    a positive lift, every metric in the treatment cohort shifts
    upward. This pins the no-op-default behaviour at the table level —
    if the gate ever started defaulting to ``"first_metric"`` or
    similar, this test would catch it.
    """
    ctrl = [
        Entity(name=f"c_{i}", archetype="flat", size=1, treatment_group="control")
        for i in range(30)
    ]
    trt = [
        Entity(
            name=f"t_{i}",
            archetype="flat",
            size=1,
            treatment_group="treatment",
            treatment_lift_log_odds=1.5,
            treatment_start_period=0,
            treatment_target_metric=None,
        )
        for i in range(30)
    ]
    cfg = _two_metric_engine_config(ctrl + trt)
    rng = np.random.default_rng(cfg.seed)
    tables, _state = generate_tables_with_state(cfg, rng)
    de = tables["dim_entity"]
    ctrl_pks = set(de.iloc[:30]["entity_id"])
    trt_pks = set(de.iloc[30:]["entity_id"])
    means = _split_cohort_means(tables, ctrl_pks, trt_pks)
    for metric in ("m1", "m2"):
        ctrl_mean, trt_mean = means[metric]
        assert trt_mean - ctrl_mean > 0.05, (
            f"target_metric=None failed to lift {metric}: "
            f"ctrl_mean={ctrl_mean}, trt_mean={trt_mean}"
        )


# --- Per-metric targeting --------------------------------------------------


def test_target_metric_shifts_only_named_metric():
    """AC #1: with ``treatment_target_metric="m1"`` and a positive lift,
    the treatment cohort's ``m1`` mean shifts UP versus control; its
    ``m2`` mean stays within statistical noise of control (no
    propagation). 30+30 sample with no correlations and no noise so
    the per-metric gate is the only thing the test pins.
    """
    ctrl = [
        Entity(name=f"c_{i}", archetype="flat", size=1, treatment_group="control")
        for i in range(30)
    ]
    trt = [
        Entity(
            name=f"t_{i}",
            archetype="flat",
            size=1,
            treatment_group="treatment",
            treatment_lift_log_odds=1.5,
            treatment_start_period=0,
            treatment_target_metric="m1",
        )
        for i in range(30)
    ]
    cfg = _two_metric_engine_config(ctrl + trt)
    rng = np.random.default_rng(cfg.seed)
    tables, _state = generate_tables_with_state(cfg, rng)
    de = tables["dim_entity"]
    ctrl_pks = set(de.iloc[:30]["entity_id"])
    trt_pks = set(de.iloc[30:]["entity_id"])
    means = _split_cohort_means(tables, ctrl_pks, trt_pks)
    m1_ctrl, m1_trt = means["m1"]
    m2_ctrl, m2_trt = means["m2"]
    assert (
        m1_trt - m1_ctrl > 0.05
    ), f"target_metric='m1' failed to lift m1: ctrl={m1_ctrl}, trt={m1_trt}"
    # AC #1 envelope: non-targeted metric must stay within 5% of
    # control mean. Picked over a t-test to keep the assertion
    # threshold concrete and seed-independent.
    rel_delta = abs(m2_trt - m2_ctrl) / max(m2_ctrl, 1e-9)
    assert rel_delta < 0.05, (
        f"target_metric='m1' leaked into m2: ctrl={m2_ctrl}, trt={m2_trt}, "
        f"relative delta={rel_delta}"
    )


def test_target_metric_named_byte_identical_non_targeted_under_no_correlations():
    """Strongest pin on the gate: under ``correlations=[]`` and
    zero-noise, the non-targeted metric must be BYTE-identical between
    a config that targets ``m1`` and an otherwise-identical config
    with no treatment at all. Both runs walk the per-entity RNG
    forward through the same number of bytes per cell (distribution
    draws consume a fixed number of bytes per call regardless of
    loc/scale), so if the gate correctly zeros the shift on ``m2``,
    its series is bit-equal across runs.

    If the gate ever leaked the shift to ``m2`` — even by a single
    logit unit — this test fails immediately.
    """
    e_trt = Entity(
        name="solo",
        archetype="flat",
        size=1,
        treatment_lift_log_odds=2.0,
        treatment_start_period=0,
        treatment_target_metric="m1",
    )
    e_none = Entity(name="solo", archetype="flat", size=1)
    cfg_trt = _two_metric_engine_config([e_trt])
    cfg_none = _two_metric_engine_config([e_none])
    rng_trt = np.random.default_rng(cfg_trt.seed)
    rng_none = np.random.default_rng(cfg_none.seed)
    tables_trt, _ = generate_tables_with_state(cfg_trt, rng_trt)
    tables_none, _ = generate_tables_with_state(cfg_none, rng_none)
    m2_trt = tables_trt["fct_m"]["m2"].to_numpy()
    m2_none = tables_none["fct_m"]["m2"].to_numpy()
    assert np.array_equal(m2_trt, m2_none), (
        "non-targeted metric m2 diverged between target_metric='m1' and "
        "no-treatment runs — the per-metric gate leaked the shift"
    )


def test_target_metric_named_lifts_targeted_under_no_correlations():
    """Symmetric pin: ``m1`` IS shifted when targeted. Run the same
    pair of configs as the byte-identity test and assert the ``m1``
    arrays DIFFER — guards against a wholesale-gate bug where the
    shift is zeroed for every metric, not just the non-targeted ones.
    """
    e_trt = Entity(
        name="solo",
        archetype="flat",
        size=1,
        treatment_lift_log_odds=2.0,
        treatment_start_period=0,
        treatment_target_metric="m1",
    )
    e_none = Entity(name="solo", archetype="flat", size=1)
    cfg_trt = _two_metric_engine_config([e_trt])
    cfg_none = _two_metric_engine_config([e_none])
    rng_trt = np.random.default_rng(cfg_trt.seed)
    rng_none = np.random.default_rng(cfg_none.seed)
    tables_trt, _ = generate_tables_with_state(cfg_trt, rng_trt)
    tables_none, _ = generate_tables_with_state(cfg_none, rng_none)
    m1_trt = tables_trt["fct_m"]["m1"].to_numpy()
    m1_none = tables_none["fct_m"]["m1"].to_numpy()
    assert not np.array_equal(m1_trt, m1_none), (
        "targeted metric m1 was NOT shifted under target_metric='m1' — "
        "the per-metric gate also zeroed the shift on the named metric"
    )
    assert m1_trt.mean() > m1_none.mean(), (
        f"targeted metric m1 shifted in the wrong direction: "
        f"trt_mean={m1_trt.mean()} vs none_mean={m1_none.mean()}"
    )


# --- Correlation leakage probe ---------------------------------------------


def test_target_metric_no_leakage_through_correlated_pair():
    """Dispatch Decision 3: with a strong baseline correlation
    ``m1 ↔ m2`` and ``target_metric="m1"``, does the copula propagate
    the m1 lift to m2 at the population mean? The copula at
    ``apply_correlations`` operates on residuals around each metric's
    own (un-shifted) center, so m2's center stays unchanged and the
    correlated-residual transform preserves the mean. This test pins
    that analytic claim empirically.
    """
    ctrl = [
        Entity(name=f"c_{i}", archetype="flat", size=1, treatment_group="control")
        for i in range(60)
    ]
    trt = [
        Entity(
            name=f"t_{i}",
            archetype="flat",
            size=1,
            treatment_group="treatment",
            treatment_lift_log_odds=1.5,
            treatment_start_period=0,
            treatment_target_metric="m1",
        )
        for i in range(60)
    ]
    cfg = _two_metric_engine_config(
        ctrl + trt,
        correlations=[CorrelationPair(metric_a="m1", metric_b="m2", coefficient=0.8)],
    )
    rng = np.random.default_rng(cfg.seed)
    tables, _state = generate_tables_with_state(cfg, rng)
    de = tables["dim_entity"]
    ctrl_pks = set(de.iloc[:60]["entity_id"])
    trt_pks = set(de.iloc[60:]["entity_id"])
    means = _split_cohort_means(tables, ctrl_pks, trt_pks)
    m1_ctrl, m1_trt = means["m1"]
    m2_ctrl, m2_trt = means["m2"]
    # m1 should still shift even with the copula active.
    assert m1_trt - m1_ctrl > 0.03, (
        f"correlated m1 didn't shift under target_metric: " f"ctrl={m1_ctrl}, trt={m1_trt}"
    )
    # m2 should NOT shift via correlation leakage. Tolerance is wider
    # than the no-correlations pin because the copula's residual
    # transform introduces a small sample-mean drift even at lift=0,
    # but it should stay well below the m1 shift.
    rel_leak = abs(m2_trt - m2_ctrl) / max(m2_ctrl, 1e-9)
    m1_shift = (m1_trt - m1_ctrl) / max(m1_ctrl, 1e-9)
    assert rel_leak < 0.10, (
        f"m2 leaked through correlation: ctrl={m2_ctrl}, trt={m2_trt}, "
        f"relative delta={rel_leak} (>10%)"
    )
    assert rel_leak < m1_shift, (
        f"m2 leakage ({rel_leak}) exceeded m1 shift ({m1_shift}) — "
        f"correlation propagation broke the per-metric gate's intent"
    )


# --- Validator -------------------------------------------------------------


def test_validator_rejects_unknown_target_metric():
    """AC #3: a config naming a metric that doesn't exist must raise at
    load time. Catches typos and stale references that would otherwise
    silently fall through the per-metric gate (no metric matches → the
    lift is never applied) and produce a dataset where the treatment
    is invisible.
    """
    entity = Entity(
        name="t",
        archetype="flat",
        size=1,
        treatment_group="treatment",
        treatment_lift_log_odds=1.0,
        treatment_target_metric="not_a_metric",
    )
    with pytest.raises(ValidationError) as excinfo:
        _two_metric_engine_config([entity])
    msg = str(excinfo.value)
    assert "not_a_metric" in msg, f"validator did not name the offending metric: {msg}"
    assert "treatment_target_metric" in msg, f"validator did not name the offending field: {msg}"


def test_validator_accepts_known_target_metric():
    """Sanity: a config naming a declared metric loads cleanly."""
    entity = Entity(
        name="t",
        archetype="flat",
        size=1,
        treatment_group="treatment",
        treatment_lift_log_odds=1.0,
        treatment_target_metric="m1",
    )
    cfg = _two_metric_engine_config([entity])
    # Round-trips through the validator without raising.
    assert cfg.entities[0].treatment_target_metric == "m1"


def test_validator_skips_when_no_treatment_fields():
    """The no-op-default skip predicate must include
    ``treatment_target_metric`` — an entity with EVERY treatment field
    unset (including the M24 addition) must still bypass the gate so
    pre-M24 configs remain validator-invisible.
    """
    entity = Entity(name="t", archetype="flat", size=1)
    cfg = _two_metric_engine_config([entity])
    assert cfg.entities[0].treatment_target_metric is None
    assert cfg.entities[0].treatment_lift_log_odds is None


# --- Manifest --------------------------------------------------------------


def test_manifest_records_target_metric_per_entity():
    """AC #4: the per-entity manifest record carries ``target_metric``.
    Configs that don't use the M24 surface continue to emit
    ``target_metric=None`` so 1.8 readers parse cleanly.
    """
    entities = [
        Entity(name="c", archetype="flat", size=1, treatment_group="control"),
        Entity(
            name="t",
            archetype="flat",
            size=1,
            treatment_group="treatment",
            treatment_lift_log_odds=1.0,
            treatment_target_metric="m1",
        ),
    ]
    cfg = _two_metric_engine_config(entities)
    rng = np.random.default_rng(cfg.seed)
    tables, state = generate_tables_with_state(cfg, rng)
    manifest = build_manifest(cfg, state.trajectories, tables)
    by_entity = {a.entity: a for a in manifest.archetype_assignments}
    assert by_entity["c"].treatment is not None
    assert by_entity["c"].treatment.target_metric is None
    assert by_entity["t"].treatment is not None
    assert by_entity["t"].treatment.target_metric == "m1"


def test_manifest_records_target_metric_per_cohort():
    """AC #4: the per-cohort manifest record carries ``target_metric``.
    Homogeneous cohort (every entity in the cohort shares the same
    target metric, which is the canonical segment-driven shape) reports
    that value directly.
    """
    entities = [
        Entity(name=f"c_{i}", archetype="flat", size=1, treatment_group="control") for i in range(5)
    ] + [
        Entity(
            name=f"t_{i}",
            archetype="flat",
            size=1,
            treatment_group="treatment",
            treatment_lift_log_odds=1.0,
            treatment_target_metric="m2",
        )
        for i in range(5)
    ]
    cfg = _two_metric_engine_config(entities)
    rng = np.random.default_rng(cfg.seed)
    tables, state = generate_tables_with_state(cfg, rng)
    manifest = build_manifest(cfg, state.trajectories, tables)
    by_label = {c.label: c for c in manifest.treatment_cohorts}
    assert by_label["control"].target_metric is None
    assert by_label["treatment"].target_metric == "m2"


def test_manifest_schema_version_bumped_for_m24():
    """Schema version pin: M24's additive field bumps the manifest
    schema. Pre-M24 readers see a 1.9 manifest's new ``target_metric``
    field default to ``None`` so they parse cleanly — but the schema
    string itself must advance to signal that the new field exists.
    """
    assert MANIFEST_SCHEMA_VERSION == "1.9"


# --- Builder propagation ---------------------------------------------------


def test_builder_treatment_config_target_metric_propagates_to_entities():
    """``TreatmentConfig.target_metric`` set on a segment lands on
    every entity expanded from that segment — both treatment AND
    control arms. The field is harmless on control entities (they
    have no lift to gate), but carrying it on both arms preserves
    ground-truth symmetry so a downstream analyst can recover the
    full experiment design from a single ``Entity`` record.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = create(
            about="m24 test",
            unit="company",
            window=("2024-01", "2024-12"),
            metrics=[
                {"name": "engagement", "type": "score", "polarity": "positive"},
                {"name": "satisfaction", "type": "score", "polarity": "positive"},
            ],
            segments=[
                {
                    "name": "s",
                    "count": 10,
                    "archetype": "flat",
                    "treatment": TreatmentConfig(
                        fraction=0.5,
                        lift_log_odds=1.0,
                        target_metric="engagement",
                    ),
                },
            ],
            seed=42,
        )
    for e in cfg.entities:
        assert (
            e.treatment_target_metric == "engagement"
        ), f"entity {e.name!r} did not inherit target_metric from segment"


def test_builder_treatment_config_target_metric_defaults_to_none():
    """A ``TreatmentConfig`` constructed without ``target_metric`` (the
    pre-M24 shape) leaves every expanded entity with
    ``treatment_target_metric=None`` — preserves the trajectory-wide
    behaviour every existing builder-using template depends on.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = create(
            about="m24 default test",
            unit="company",
            window=("2024-01", "2024-12"),
            metrics=[{"name": "engagement", "type": "score", "polarity": "positive"}],
            segments=[
                {
                    "name": "s",
                    "count": 10,
                    "archetype": "flat",
                    "treatment": TreatmentConfig(fraction=0.5, lift_log_odds=1.0),
                },
            ],
            seed=42,
        )
    for e in cfg.entities:
        assert e.treatment_target_metric is None
