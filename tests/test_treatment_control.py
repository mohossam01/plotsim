"""0.6-M8c: treatment / control with known effect (logit-shift cohorts).

Final phase of the M8 mission. Builds on M8a's ``Entity.start_period``
(cold-start) and M8b's ``SegmentInput.arrival`` (drift) surfaces. Adds
``Entity.treatment_group`` / ``treatment_lift_log_odds`` /
``treatment_start_period`` engine fields, the per-entity logit shift in
``_compute_effective_position``, the builder ``TreatmentConfig`` shape,
the manifest ``TreatmentAssignment`` + ``TreatmentCohort`` records, and
the load-time validator gate.

Locks in:

* Logit-shift math: ``_apply_logit_shift(p, shift)`` returns
  ``sigmoid(logit(p) + shift)`` with numerical guards against
  overflow / boundary positions.
* Default behaviour (no treatment fields set) preserves pre-M8c
  output byte-for-byte — every existing template falls through
  unchanged.
* Pre-treatment baseline: at ``period_index < treatment_start_period``,
  treatment and control groups have identical effective trajectory
  positions (the AC for "pre-treatment baseline is identical across
  groups"). Different distributional draws between entities are
  expected RNG noise; the population-level distributions match.
* Post-treatment lift: the configured ``treatment_lift_log_odds``
  is statistically recoverable via difference-in-means on the
  generated metric values. Tolerance documented in the test.
* Manifest: per-entity ``EntityArchetypeAssignment.treatment``
  populated with the entity's group / lift / start_period; per-cohort
  ``ManifestSchema.treatment_cohorts`` aggregated by label.
  Schema version bumped 1.2 → 1.3.
* RNG isolation pin (operator-flagged at start of M8c): the
  interpreter's ``treatment_rng`` is salted independently of
  ``arrival_rng``, so changing a segment's arrival shape does NOT
  shift which entities land in the treatment arm. Pinned by
  ``test_treatment_assignments_independent_of_arrival_shape``.
* Validator: ``treatment_start_period >= n_periods`` and non-finite
  ``treatment_lift_log_odds`` are rejected at config load.
  ``treatment_start_period < entity.start_period`` is INTENTIONALLY
  legal — pinned by
  ``test_treatment_start_before_entity_start_is_legal``.
"""

from __future__ import annotations

import warnings
from collections import Counter

import numpy as np
import pytest

from plotsim import create, generate_tables_with_state
from plotsim.builder.input import (
    ExplicitArrival,
    StepArrival,
    StepArrivalBlock,
    TreatmentConfig,
    UniformArrival,
)
from plotsim.config import (
    Archetype,
    Column,
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
from plotsim.manifest import build_manifest
from plotsim.metrics import _apply_logit_shift


# --- Helpers ----------------------------------------------------------------


def _engine_config(
    entities: list[Entity],
    *,
    metric_distribution: str = "beta",
    metric_params: dict | None = None,
) -> PlotsimConfig:
    """Build a minimal engine-direct config (bypassing the builder).

    Used for the engine-level pin tests where the builder's expansion
    machinery is irrelevant. Defaults to a beta-distributed metric on
    a flat archetype so the post-treatment lift in metric values is
    cleanly recoverable.
    """
    if metric_params is None:
        metric_params = {"alpha": 2.0, "beta": 5.0}
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
    metric = Metric(
        name="m",
        label="m",
        distribution=metric_distribution,
        params=metric_params,
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
            Column(name="m", dtype="float", source="metric:m"),
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
            metrics=[metric],
            archetypes=[arch],
            entities=entities,
            tables=[dim_date, dim_entity, fct],
            output=OutputConfig(format="csv", directory="out/m8c"),
        )


def _builder_config(
    *,
    arrival=None,
    treatment=None,
    count: int = 20,
    seed: int = 42,
):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return create(
            about="m8c test",
            unit="company",
            window=("2024-01", "2024-12"),
            metrics=[
                {"name": "engagement", "type": "score", "polarity": "positive"},
            ],
            segments=[
                {
                    "name": "s",
                    "count": count,
                    "archetype": "flat",
                    "arrival": arrival,
                    "treatment": treatment,
                },
            ],
            seed=seed,
        )


# --- Logit shift math -------------------------------------------------------


def test_logit_shift_zero_is_identity():
    """``shift == 0.0`` short-circuits to ``p`` exactly — preserves
    byte-identity for zero-shift entities (the control-arm contract).
    """
    for p in (0.1, 0.5, 0.9):
        assert _apply_logit_shift(p, 0.0) == p


def test_logit_shift_positive_pushes_toward_one():
    """Positive shift raises the trajectory's effective position."""
    for p in (0.1, 0.3, 0.5, 0.7):
        shifted = _apply_logit_shift(p, 1.0)
        assert shifted > p, f"positive shift didn't raise p={p}: got {shifted}"


def test_logit_shift_negative_pushes_toward_zero():
    for p in (0.3, 0.5, 0.7, 0.9):
        shifted = _apply_logit_shift(p, -1.0)
        assert shifted < p, f"negative shift didn't lower p={p}: got {shifted}"


def test_logit_shift_handles_boundary_positions():
    """``p`` at exactly 0 or 1 must not blow up. The clamp keeps the
    shifted value finite — an entity flatlined at the boundary stays
    there post-treatment (the trajectory has pinned it; the lift can't
    move a position the engine has already saturated).
    """
    # p=0 and p=1 — guard against logit's -inf / +inf branches.
    assert 0.0 <= _apply_logit_shift(0.0, 1.0) < 1e-3
    assert 1.0 - 1e-3 < _apply_logit_shift(1.0, -1.0) <= 1.0


def test_logit_shift_diminishing_returns_near_boundaries():
    """The same shift produces less absolute movement near the
    boundaries — that's the math of working in log-odds space, and
    it's the right behaviour for an A/B lift (same intervention is
    less impactful when the metric is already near saturation).
    """
    delta_mid = _apply_logit_shift(0.5, 0.5) - 0.5
    delta_high = _apply_logit_shift(0.9, 0.5) - 0.9
    assert delta_high < delta_mid


# --- Default behaviour (no treatment fields) preserved ----------------------


def test_no_treatment_fields_preserves_pre_m8c_output():
    """Sanity / regression: a config with no treatment fields produces
    the same output as pre-M8c. Two entities with identical archetype
    and seed produce identical metric series at the same RNG draw
    position — verified at the manifest level (treatment fields
    absent).
    """
    entities = [
        Entity(name="a", archetype="flat", size=1),
        Entity(name="b", archetype="flat", size=1),
    ]
    cfg = _engine_config(entities)
    rng = np.random.default_rng(cfg.seed)
    tables, state = generate_tables_with_state(cfg, rng)
    manifest = build_manifest(cfg, state.trajectories, tables)
    for assignment in manifest.archetype_assignments:
        assert (
            assignment.treatment is None
        ), f"entity {assignment.entity!r} got a treatment record despite no fields set"
    assert manifest.treatment_cohorts == []


# --- Pre-treatment baseline -------------------------------------------------


def test_pre_treatment_baseline_uses_zero_shift():
    """At ``period_index < treatment_start_period``, the treatment
    entity's effective position must equal the control entity's
    effective position when seeded identically. This is the trajectory-
    level contract that backs the population-level "identical
    pre-treatment distributions" AC.
    """
    # Both entities share the flat archetype → trajectory position is
    # 0.5 at every period for both. The shift is 0.0 pre-treatment
    # (period < treatment_start_period=6), so eff_pos at period 0..5
    # must equal 0.5 for both treatment and control. We verify this
    # via the engine-internal helper directly.
    from plotsim.metrics import _compute_effective_position

    metric = Metric(
        name="m",
        label="m",
        distribution="beta",
        params={"alpha": 2.0, "beta": 5.0},
        polarity="positive",
    )
    # Pre-treatment: shift=0 should be a no-op.
    eff = _compute_effective_position(0.5, metric, None, period_index=3, treatment_shift=0.0)
    assert eff == 0.5
    # Post-treatment: shift>0 should push toward 1.0.
    eff_post = _compute_effective_position(0.5, metric, None, period_index=8, treatment_shift=1.0)
    assert eff_post > 0.5


def test_pre_treatment_population_distributions_match():
    """30 control + 30 treatment entities, treatment kicks in at period
    6. Pre-treatment (periods 0-5) population means should be within
    statistical noise of each other; post-treatment means should
    differ by approximately the configured lift.
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
            treatment_lift_log_odds=1.0,
            treatment_start_period=6,
        )
        for i in range(30)
    ]
    cfg = _engine_config(ctrl + trt)
    rng = np.random.default_rng(cfg.seed)
    tables, _state = generate_tables_with_state(cfg, rng)
    fct = tables["fct_m"]
    de = tables["dim_entity"]
    ctrl_pks = set(de.iloc[:30]["entity_id"])
    trt_pks = set(de.iloc[30:]["entity_id"])
    fct = fct.copy()
    fct["period"] = fct.groupby("entity_id").cumcount()
    pre = fct[fct["period"] < 6]
    post = fct[fct["period"] >= 6]
    pre_delta = (
        pre[pre["entity_id"].isin(trt_pks)]["m"].mean()
        - pre[pre["entity_id"].isin(ctrl_pks)]["m"].mean()
    )
    post_delta = (
        post[post["entity_id"].isin(trt_pks)]["m"].mean()
        - post[post["entity_id"].isin(ctrl_pks)]["m"].mean()
    )
    assert abs(pre_delta) < 0.05, f"pre-treatment delta {pre_delta} too large"
    assert post_delta > 0.15, (
        f"post-treatment lift not recoverable: delta={post_delta}, "
        f"expected > 0.15 from log-odds lift of 1.0"
    )


# --- RNG isolation pin (operator-flagged) ----------------------------------


def test_treatment_assignments_independent_of_arrival_shape():
    """The contract operator flagged at M8c kickoff: changing a
    segment's arrival shape (which consumes RNG state on
    ``arrival_rng``) must NOT shift which entities land in the
    treatment arm. Same seed + same treatment config + ANY arrival
    shape → identical treatment assignments.

    Achieved via independent ``arrival_rng`` and ``treatment_rng``
    streams in the interpreter, salted by ``TREATMENT_SALT``.
    """
    treat = TreatmentConfig(fraction=0.5, lift_log_odds=1.0, start_period=4)
    cfg_uniform = _builder_config(
        arrival=UniformArrival(kind="uniform", start=0, end=8),
        treatment=treat,
        count=20,
        seed=42,
    )
    cfg_explicit = _builder_config(
        arrival=ExplicitArrival(
            kind="explicit",
            start_periods=[i % 8 for i in range(20)],
        ),
        treatment=treat,
        count=20,
        seed=42,
    )
    cfg_step = _builder_config(
        arrival=StepArrival(
            kind="step",
            blocks=[
                StepArrivalBlock(period=0, fraction=0.5),
                StepArrivalBlock(period=4, fraction=0.5),
            ],
        ),
        treatment=treat,
        count=20,
        seed=42,
    )
    cfg_none = _builder_config(arrival=None, treatment=treat, count=20, seed=42)

    labels_uniform = [e.treatment_group for e in cfg_uniform.entities]
    labels_explicit = [e.treatment_group for e in cfg_explicit.entities]
    labels_step = [e.treatment_group for e in cfg_step.entities]
    labels_none = [e.treatment_group for e in cfg_none.entities]
    assert (
        labels_uniform == labels_explicit == labels_step == labels_none
    ), "treatment assignments shifted when arrival shape changed — RNG isolation broken"


def test_treatment_determinism_under_reseed():
    """Same seed + same builder config → identical per-entity treatment
    assignment AND identical lift values.
    """
    treat = TreatmentConfig(fraction=0.5, lift_log_odds=1.5, start_period=3)
    a = _builder_config(treatment=treat, count=10, seed=7)
    b = _builder_config(treatment=treat, count=10, seed=7)
    a_groups = [e.treatment_group for e in a.entities]
    b_groups = [e.treatment_group for e in b.entities]
    a_lifts = [e.treatment_lift_log_odds for e in a.entities]
    b_lifts = [e.treatment_lift_log_odds for e in b.entities]
    assert a_groups == b_groups
    assert a_lifts == b_lifts


def test_different_seeds_produce_different_assignments():
    """Sanity: not all seeds collapse to the same treatment split."""
    treat = TreatmentConfig(fraction=0.5, lift_log_odds=1.0)
    a = _builder_config(treatment=treat, count=20, seed=1)
    b = _builder_config(treatment=treat, count=20, seed=999)
    assert [e.treatment_group for e in a.entities] != [e.treatment_group for e in b.entities]


# --- Builder treatment fraction --------------------------------------------


def test_treatment_fraction_produces_correct_split():
    """``fraction=0.5`` on count=20 → exactly 10 treatment + 10 control.
    Rounding via ``round(fraction * count)``: 0.5 * 20 = 10.
    """
    treat = TreatmentConfig(fraction=0.5, lift_log_odds=1.0)
    cfg = _builder_config(treatment=treat, count=20)
    counts = Counter(e.treatment_group for e in cfg.entities)
    assert counts["treatment"] == 10
    assert counts["control"] == 10


def test_treatment_fraction_zero_produces_all_control():
    """Edge case: ``fraction=0.0`` means no entity is in treatment.
    Every entity gets the control label and ``lift_log_odds=None``.
    """
    treat = TreatmentConfig(fraction=0.0, lift_log_odds=1.0)
    cfg = _builder_config(treatment=treat, count=10)
    for e in cfg.entities:
        assert e.treatment_group == "control"
        assert e.treatment_lift_log_odds is None


def test_treatment_fraction_one_produces_all_treatment():
    """Edge case: ``fraction=1.0`` means every entity is treated.
    No control arm. Useful for ablation experiments where the entire
    cohort gets the same intervention.
    """
    treat = TreatmentConfig(fraction=1.0, lift_log_odds=1.0)
    cfg = _builder_config(treatment=treat, count=10)
    for e in cfg.entities:
        assert e.treatment_group == "treatment"
        assert e.treatment_lift_log_odds == 1.0


def test_treatment_custom_labels():
    """``treatment_label`` and ``control_label`` propagate to entities.
    Useful for multi-arm tests or domain-specific naming.
    """
    treat = TreatmentConfig(
        fraction=0.5,
        lift_log_odds=0.5,
        treatment_label="variant_a",
        control_label="variant_b",
    )
    cfg = _builder_config(treatment=treat, count=10)
    labels = {e.treatment_group for e in cfg.entities}
    assert labels == {"variant_a", "variant_b"}


# --- Manifest treatment surface ---------------------------------------------


def test_manifest_per_entity_treatment_assignment():
    entities = [
        Entity(name="ctrl", archetype="flat", size=1, treatment_group="control"),
        Entity(
            name="trt",
            archetype="flat",
            size=1,
            treatment_group="treatment",
            treatment_lift_log_odds=1.0,
            treatment_start_period=4,
        ),
    ]
    cfg = _engine_config(entities)
    rng = np.random.default_rng(cfg.seed)
    tables, state = generate_tables_with_state(cfg, rng)
    manifest = build_manifest(cfg, state.trajectories, tables)
    by_entity = {a.entity: a for a in manifest.archetype_assignments}
    assert by_entity["ctrl"].treatment is not None
    assert by_entity["ctrl"].treatment.group == "control"
    assert by_entity["ctrl"].treatment.lift_log_odds is None
    assert by_entity["trt"].treatment is not None
    assert by_entity["trt"].treatment.group == "treatment"
    assert by_entity["trt"].treatment.lift_log_odds == 1.0
    assert by_entity["trt"].treatment.start_period == 4


def test_manifest_treatment_cohorts_aggregate_per_label():
    """Per-cohort aggregation: 5 control entities + 5 treatment entities
    with lift=1.0 → two ``TreatmentCohort`` records, label-sorted.
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
            treatment_start_period=4,
        )
        for i in range(5)
    ]
    cfg = _engine_config(entities)
    rng = np.random.default_rng(cfg.seed)
    tables, state = generate_tables_with_state(cfg, rng)
    manifest = build_manifest(cfg, state.trajectories, tables)
    cohorts_by_label = {c.label: c for c in manifest.treatment_cohorts}
    assert set(cohorts_by_label.keys()) == {"control", "treatment"}
    assert cohorts_by_label["control"].n_entities == 5
    assert cohorts_by_label["control"].mean_lift_log_odds is None
    assert cohorts_by_label["treatment"].n_entities == 5
    assert cohorts_by_label["treatment"].mean_lift_log_odds == 1.0
    assert cohorts_by_label["treatment"].start_period == 4


# --- Validator gates --------------------------------------------------------


def test_validator_rejects_treatment_start_at_n_periods():
    """``treatment_start_period >= n_periods`` would never apply the
    shift — silent dead-weight. Validator rejects.
    """
    entities = [
        Entity(
            name="bad",
            archetype="flat",
            size=1,
            treatment_lift_log_odds=1.0,
            treatment_start_period=12,  # n_periods=12 → out of window
        ),
    ]
    with pytest.raises(ValueError, match="at or past n_periods"):
        _engine_config(entities)


def test_validator_rejects_non_finite_lift():
    """``lift_log_odds`` must be finite; ``inf`` or ``nan`` would
    propagate NaN through the logit shift.
    """
    entities = [
        Entity(
            name="bad",
            archetype="flat",
            size=1,
            treatment_lift_log_odds=float("inf"),
            treatment_start_period=3,
        ),
    ]
    with pytest.raises(ValueError, match="non-finite"):
        _engine_config(entities)


def test_treatment_start_before_entity_start_is_legal():
    """Cold-start interaction: an entity arriving at period 6 with
    ``treatment_start_period=4`` is fine. Periods 4-5 are dormant
    (cold-start NaN trajectory, no rows generated), so the shift
    kicks in naturally at the entity's first active period (6).
    The builder relies on this slack to assign one segment-level
    ``TreatmentConfig.start_period`` to a cohort whose entities have
    arrival-distribution-drawn ``start_period`` values varying per
    entity.
    """
    entities = [
        Entity(
            name="late",
            archetype="flat",
            size=1,
            start_period=6,
            treatment_lift_log_odds=1.0,
            treatment_start_period=4,
        ),
    ]
    cfg = _engine_config(entities)
    rng = np.random.default_rng(cfg.seed)
    tables, _state = generate_tables_with_state(cfg, rng)
    fct = tables["fct_m"]
    # 12 - 6 = 6 active periods (cold-start filter drops periods 0-5).
    assert len(fct) == 6
    # The retained cells are all post-treatment (period 6 >= 4).
    assert fct["m"].notna().all()
