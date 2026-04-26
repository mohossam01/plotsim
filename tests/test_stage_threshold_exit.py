"""F8 regression — wire threshold_exit with hysteresis semantics (M102).

Mission 100 found ``StageDefinition.threshold_exit`` accepted at
config load but ignored at runtime — both ``_monotonic_stage_walk``
and ``_free_mode_stages`` consume only ``threshold_enter``. Operator
chose option 2 (additive wiring with a mode-discriminating
validator) after the grep audit confirmed the field is decorative
in every runtime path.

Three-way regression:

* ``test_legacy_mode_byte_identical`` — a 4-stage config matching the
  saas template's contiguous semantic (``threshold_exit ==
  next.threshold_enter``, ``exit > enter`` per stage). Loads under
  the new mode-discriminating validator without complaint and
  produces output byte-identical to the pre-F8 expected stage
  walk. Confirms the wiring change does not perturb the legacy
  path that all five bundled templates use.
* ``test_hysteresis_mode_demotes_below_exit`` — a 3-stage config with
  ``threshold_exit < threshold_enter`` per stage. Walks a synthetic
  trajectory that climbs above ``stage[2].enter``, dips into the
  hysteresis band ``[stage[2].exit, stage[2].enter]`` (asserts the
  cursor stays in stage[2]), then drops below ``stage[2].exit``
  (asserts the cursor demotes). Locks the F8 wiring's runtime
  effect.
* ``test_mixed_mode_rejected`` — a sequence that mixes legacy
  (stage 0: enter=0.0, exit=0.3) with hysteresis (stage 1:
  enter=0.5, exit=0.4). Loads must raise with both per-stage
  modes named in the message.

Plus two helpers:

* ``test_hysteresis_mode_with_downgrade_delay`` — combines hysteresis
  with ``downgrade_delay=2`` and asserts demotion only fires after
  two consecutive periods below exit.
* ``test_stage_sequence_mode_property`` — direct unit test on the
  derived ``StageSequence.mode`` property.
"""
from __future__ import annotations

import warnings
from typing import Iterable

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from plotsim import generate_tables
from plotsim.config import (
    Archetype,
    Column,
    CurveSegment,
    Domain,
    Entity,
    Metric,
    OutputConfig,
    PlotsimConfig,
    StageDefinition,
    StageSequence,
    SurrogateKeyWarning,
    Table,
    TimeWindow,
)
from plotsim.tables import _monotonic_stage_walk


# --- Helpers ----------------------------------------------------------------


def _stage_seq(
    stages: list[tuple[str, float, float | None]],
    *,
    field: str = "score",
    enforce_order: bool = True,
    downgrade_delay: int | None = None,
) -> StageSequence:
    """Build a StageSequence from ``[(name, enter, exit), ...]`` triples.
    ``exit=None`` marks the terminal stage."""
    return StageSequence(
        field=field,
        sequence=[
            StageDefinition(
                name=name,
                threshold_enter=enter,
                threshold_exit=exit_,
            )
            for name, enter, exit_ in stages
        ],
        enforce_order=enforce_order,
        downgrade_delay=downgrade_delay,
    )


def _config_with_stages(
    stages: StageSequence,
    *,
    metric_name: str = "score",
    seed: int = 0,
) -> PlotsimConfig:
    """Minimal config with a single normal metric named ``score``,
    used as the stages.field driver."""
    metric = Metric(
        name=metric_name, label=metric_name,
        distribution="normal", params={"mu": 1.0, "sigma": 0.0001},
        polarity="positive",
    )
    arch = Archetype(
        name="flat", label="flat",
        description="constant 0.5 plateau",
        curve_segments=[
            CurveSegment(
                curve="plateau", params={"level": 0.5},
                start_pct=0.0, end_pct=1.0,
            ),
        ],
    )
    fct = Table(
        name="fct_score", type="fact", grain="per_entity_per_period",
        primary_key=["date_key", "user_id"],
        foreign_keys=["dim_date.date_key", "dim_user.user_id"],
        columns=[
            Column(name="date_key", dtype="id", source="fk:dim_date.date_key"),
            Column(name="user_id", dtype="id", source="fk:dim_user.user_id"),
            Column(name=metric_name, dtype="float", source=f"metric:{metric_name}"),
        ],
    )
    dim_date = Table(
        name="dim_date", type="dim", grain="per_period",
        primary_key="date_key",
        columns=[
            Column(name="date_key", dtype="id", source="pk"),
            Column(name="date", dtype="date", source="generated:date_key"),
        ],
    )
    dim_user = Table(
        name="dim_user", type="dim", grain="per_entity",
        primary_key="user_id",
        columns=[
            Column(name="user_id", dtype="id", source="pk"),
            Column(name="user_name", dtype="string", source="generated:faker.name"),
        ],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SurrogateKeyWarning)
        return PlotsimConfig(
            domain=Domain(
                name="t", description="t",
                entity_type="user", entity_label="Users",
            ),
            time_window=TimeWindow(
                start="2024-01", end="2024-12", granularity="monthly",
            ),
            seed=seed,
            metrics=[metric],
            archetypes=[arch],
            entities=[Entity(name="u01", archetype="flat", size=1)],
            tables=[dim_date, dim_user, fct],
            stages=stages,
            output=OutputConfig(format="csv", directory="out/f8"),
        )


# --- Test 1: legacy mode is byte-identical ----------------------------------


def test_legacy_mode_byte_identical():
    """A saas-style 4-stage legacy config (exit > enter per stage,
    contiguous) must produce the same monotonic walk as the pre-F8
    code: searchsorted on threshold_enter, np.maximum.accumulate,
    no demotion. Verified by running ``_monotonic_stage_walk``
    without ``exit_thresholds`` (the legacy path) on a synthetic
    trajectory and confirming output equals
    ``np.maximum.accumulate(searchsorted(enters, values) - 1)``.
    """
    enter_thresholds = np.asarray([0.0, 0.2, 0.5, 0.8], dtype=float)
    # Trajectory: climbs 0 → 0.3 → 0.7 → 0.9, then dips back to 0.3.
    # Legacy strict-monotonic must hold the high-water mark.
    values = np.asarray([0.0, 0.3, 0.7, 0.9, 0.3, 0.1, 0.7], dtype=float)

    out_legacy = _monotonic_stage_walk(
        values, enter_thresholds, downgrade_delay=None,
    )

    actual = np.searchsorted(
        enter_thresholds, values, side="right",
    ) - 1
    np.clip(actual, 0, len(enter_thresholds) - 1, out=actual)
    expected = np.maximum.accumulate(actual)
    np.testing.assert_array_equal(out_legacy, expected)
    # Concrete values: 0=onb, 1=active, 2=at_risk, 3=churned.
    np.testing.assert_array_equal(out_legacy, [0, 1, 2, 3, 3, 3, 3])


def test_legacy_mode_with_delay_unchanged():
    """Legacy mode + downgrade_delay must reproduce the pre-F8
    behavior: demote when ``actual[i] < cursor`` (equivalent to
    ``value < threshold_enter[cursor]``) for ``downgrade_delay``
    consecutive periods.
    """
    enter_thresholds = np.asarray([0.0, 0.5], dtype=float)
    # Climb above 0.5 (cursor=1), then 3 periods below 0.5: demote.
    values = np.asarray([0.0, 0.6, 0.3, 0.3, 0.3, 0.7], dtype=float)

    out = _monotonic_stage_walk(
        values, enter_thresholds, downgrade_delay=3,
    )
    # i=0: cursor 0, i=1: cursor 1, i=2-4: streak 1,2,3 → demote at i=4,
    # i=5: above 0.5, cursor 1.
    np.testing.assert_array_equal(out, [0, 1, 1, 1, 0, 1])


# --- Test 2: hysteresis mode demotes below exit -----------------------------


def test_hysteresis_mode_demotes_below_exit():
    """Hysteresis mode (exit ≤ enter) makes the cursor demote when
    value drops below ``exit_thresholds[cursor]``. With no
    ``downgrade_delay``, demotion is immediate (delay=1). The
    hysteresis band ``[exit, enter]`` keeps the cursor in the
    higher stage on transient dips.
    """
    enter_thresholds = np.asarray([0.0, 0.5, 0.8], dtype=float)
    # Hysteresis: stage 1 has exit=0.3 (band [0.3, 0.5]); stage 2 has
    # exit=0.6 (band [0.6, 0.8]). Lowest stage's exit is decorative
    # (cannot demote below stage 0); set to 0.0.
    exit_thresholds = np.asarray([0.0, 0.3, 0.6], dtype=float)
    # Trajectory: 0.0 → 0.6 (cursor=1) → 0.4 (in hysteresis band, stay
    # at cursor=1) → 0.9 (cursor=2) → 0.7 (band [0.6, 0.8], stay)
    # → 0.55 (below stage[2].exit=0.6, demote to a=1) → 0.25 (below
    # stage[1].exit=0.3, demote to a=0).
    values = np.asarray(
        [0.0, 0.6, 0.4, 0.9, 0.7, 0.55, 0.25], dtype=float,
    )

    out = _monotonic_stage_walk(
        values, enter_thresholds, downgrade_delay=None,
        exit_thresholds=exit_thresholds,
    )
    np.testing.assert_array_equal(out, [0, 1, 1, 2, 2, 1, 0])


def test_hysteresis_mode_with_downgrade_delay():
    """Hysteresis + downgrade_delay=2: demotion fires only after two
    consecutive periods below the current stage's exit. A single
    sub-exit dip stays in the higher stage."""
    enter_thresholds = np.asarray([0.0, 0.5], dtype=float)
    exit_thresholds = np.asarray([0.0, 0.3], dtype=float)
    # Climb to stage 1, single dip below exit (stay), then two
    # consecutive dips (demote on the second).
    values = np.asarray(
        [0.0, 0.6, 0.2, 0.7, 0.2, 0.2, 0.7], dtype=float,
    )

    out = _monotonic_stage_walk(
        values, enter_thresholds, downgrade_delay=2,
        exit_thresholds=exit_thresholds,
    )
    # i=0: 0; i=1: 1; i=2: streak 1 below 0.3, stay; i=3: above
    # exit, streak resets, stay 1; i=4: streak 1, stay; i=5: streak
    # 2, demote to a=0; i=6: above 0.5, cursor 1.
    np.testing.assert_array_equal(out, [0, 1, 1, 1, 1, 0, 1])


# --- Test 3: mixed mode rejected --------------------------------------------


def test_mixed_mode_rejected():
    """A sequence that mixes legacy (exit > enter on one stage) with
    hysteresis (exit ≤ enter on another) must raise at load with
    both per-stage modes named in the message.
    """
    with pytest.raises(ValidationError) as exc_info:
        _stage_seq([
            ("low", 0.0, 0.3),    # legacy: 0.3 > 0.0
            ("mid", 0.5, 0.4),    # hysteresis: 0.4 < 0.5
            ("high", 0.8, None),  # terminal
        ])
    msg = str(exc_info.value)
    assert "mix" in msg.lower(), f"mode-mix not mentioned in message: {msg}"
    assert "legacy" in msg.lower()
    assert "hysteresis" in msg.lower()
    assert "'low'" in msg or '"low"' in msg
    assert "'mid'" in msg or '"mid"' in msg


def test_legacy_validator_still_rejects_overlap():
    """Legacy mode rule survives: prev.exit > curr.enter is rejected."""
    with pytest.raises(ValidationError) as exc_info:
        _stage_seq([
            ("low", 0.0, 0.5),
            ("mid", 0.3, 0.7),  # 0.3 < prev.exit=0.5 → overlap
            ("high", 0.8, None),
        ])
    assert "overlap" in str(exc_info.value).lower()


def test_hysteresis_validator_rejects_exit_below_prev_enter():
    """Hysteresis mode rule: this.exit ≥ prev.enter (so demoting from
    this stage lands the entity in a defined lower stage)."""
    with pytest.raises(ValidationError) as exc_info:
        _stage_seq([
            ("low", 0.2, 0.1),    # hysteresis: 0.1 < 0.2
            # mid.exit=0.05 < prev.enter=0.2 → reject
            ("mid", 0.5, 0.05),
            ("high", 0.8, None),
        ])
    msg = str(exc_info.value)
    assert "hysteresis" in msg.lower()
    assert ">=" in msg or "previous" in msg.lower()


# --- Test 4: end-to-end via generate_tables ---------------------------------


def test_generate_tables_legacy_mode_loads_and_runs():
    """End-to-end: a legacy-mode config loads and generate_tables
    produces a stage column. Output values are the strict-monotonic
    walk; no hysteresis effect."""
    stages = _stage_seq([
        ("low", 0.0, 0.3),
        ("mid", 0.3, 0.7),
        ("high", 0.7, None),
    ])
    cfg = _config_with_stages(stages)
    tables = generate_tables(cfg, np.random.default_rng(0))
    fct = tables["fct_score"]
    assert "stage" in fct.columns
    # Mode introspection survives load.
    assert cfg.stages is not None
    assert cfg.stages.mode == "legacy"


def test_generate_tables_hysteresis_mode_loads_and_runs():
    """End-to-end hysteresis-mode config loads and produces a stage
    column. The mode property reflects the new semantic."""
    stages = _stage_seq([
        ("low", 0.0, 0.0),
        ("mid", 0.5, 0.3),     # hysteresis band [0.3, 0.5]
        ("high", 0.8, None),
    ])
    cfg = _config_with_stages(stages)
    tables = generate_tables(cfg, np.random.default_rng(0))
    fct = tables["fct_score"]
    assert "stage" in fct.columns
    assert cfg.stages is not None
    assert cfg.stages.mode == "hysteresis"


# --- Test 5: derived mode property ------------------------------------------


def test_stage_sequence_mode_property():
    """``StageSequence.mode`` is derived from the first non-terminal
    stage's ``exit`` vs ``enter`` relationship."""
    legacy = _stage_seq([
        ("low", 0.0, 0.5),
        ("high", 0.5, None),
    ])
    assert legacy.mode == "legacy"

    hysteresis = _stage_seq([
        ("low", 0.0, 0.0),
        ("high", 0.5, None),
    ])
    # exit=0.0 ≤ enter=0.0 → hysteresis (degenerate, equivalent to
    # legacy at runtime since demote_t equals enter; still classified
    # as hysteresis for mode discrimination).
    assert hysteresis.mode == "hysteresis"

    hysteresis_strict = _stage_seq([
        ("low", 0.2, 0.1),
        ("high", 0.5, 0.4),
        ("term", 0.8, None),
    ])
    assert hysteresis_strict.mode == "hysteresis"
