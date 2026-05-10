"""0.6-M8b: segment proportion drift (cohort-mix evolution).

Builds on M8a's ``Entity.start_period`` cold-start surface. Adds
``SegmentInput.arrival`` — a discriminated union over four arrival
distributions — and a load-time gate that rejects entities born too
late to satisfy ``MIN_ACTIVE_PERIODS``.

Locks in:

* Four arrival shapes — ``UniformArrival`` / ``LinearArrival`` /
  ``StepArrival`` / ``ExplicitArrival`` — discriminated by ``kind``.
* The interpreter draws per-entity ``Entity.start_period`` values
  deterministically from a seed-derived RNG. Same seed + same
  ``UserInput`` → same per-entity arrival schedule.
* RNG isolation: ``step`` and ``explicit`` shapes consume zero RNG
  draws, so adding either between two random shapes does not shift
  the random shapes' draws.
* Default behaviour (omit ``arrival``) preserves pre-M8b output —
  every entity gets ``start_period=0`` and existing templates are
  byte-identical.
* Load-time validator (``validate_cold_start_active_periods``)
  rejects configs where any entity has fewer than
  ``MIN_ACTIVE_PERIODS`` (= 2) active periods.
* The arrival models are builder-internal — not exported on the
  ``plotsim.*`` namespace; engine-direct configs use
  ``Entity.start_period`` directly.
"""

from __future__ import annotations

import warnings
from collections import Counter

import numpy as np
import pytest
from pydantic import ValidationError

from plotsim import create, generate_tables_with_state
from plotsim.builder.input import (
    ExplicitArrival,
    LinearArrival,
    StepArrival,
    StepArrivalBlock,
    UniformArrival,
)
from plotsim.validation import MIN_ACTIVE_PERIODS, validate_cold_start_active_periods


# --- Helpers ----------------------------------------------------------------


def _build(arrival, *, count: int = 5, seed: int = 42, window=("2024-01", "2024-12")):
    """Build a one-segment config with the given arrival distribution.

    Returns the ``PlotsimConfig``. Suppress ``UserWarning`` (single-segment
    cohort warning) since these tests are about arrival behaviour, not
    archetype mix.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return create(
            about="m8b test",
            unit="company",
            window=window,
            metrics=[{"name": "engagement", "type": "score", "polarity": "positive"}],
            segments=[
                {
                    "name": "s",
                    "count": count,
                    "archetype": "growth",
                    "arrival": arrival,
                },
            ],
            seed=seed,
        )


# --- Default (no arrival) preserves pre-M8b behaviour -----------------------


def test_omitting_arrival_leaves_all_entities_at_period_zero():
    """Sanity / regression: a segment without ``arrival`` produces every
    entity at ``start_period=0`` — byte-identical to pre-M8b output for
    existing templates.
    """
    cfg = _build(arrival=None, count=5)
    assert all(e.start_period == 0 for e in cfg.entities)


# --- UniformArrival ---------------------------------------------------------


def test_uniform_draws_within_bounds():
    cfg = _build(UniformArrival(kind="uniform", start=2, end=8), count=20)
    starts = [e.start_period for e in cfg.entities]
    assert all(2 <= s < 8 for s in starts), f"start_periods out of bounds: {starts}"


def test_uniform_default_end_uses_n_periods_minus_min_active_periods():
    """``end=None`` defaults to ``n_periods - MIN_ACTIVE_PERIODS`` so
    every drawn entity automatically has at least ``MIN_ACTIVE_PERIODS``
    active periods. The validator wouldn't catch a draw past the
    deadline (validator runs AFTER interpretation), so the default must
    pre-bound it.
    """
    cfg = _build(UniformArrival(kind="uniform"), count=20)
    n_periods = cfg.time_window.period_count()
    deadline = n_periods - MIN_ACTIVE_PERIODS
    starts = [e.start_period for e in cfg.entities]
    assert all(
        0 <= s <= deadline for s in starts
    ), f"draw past default deadline {deadline}: {starts}"


def test_uniform_spreads_arrivals_across_window():
    """50 entities + uniform(0, 10) should hit at least 5 distinct
    start_periods (sanity that the draw isn't degenerate, e.g. all
    entities collapsed to one period). Probabilistic but the chance of
    fewer than 5 distinct values is vanishingly small at this size.
    """
    cfg = _build(UniformArrival(kind="uniform", start=0, end=10), count=50)
    distinct = len({e.start_period for e in cfg.entities})
    assert distinct >= 5, f"uniform should spread arrivals; got {distinct} distinct"


def test_uniform_rejects_end_le_start():
    """Interpreter raises if ``end <= start``. Caught at interpret time
    rather than at the field validator because ``end=None`` is a valid
    field state (the default-fill happens during interpretation).
    """
    with pytest.raises(ValueError, match="end .* must be > start"):
        _build(UniformArrival(kind="uniform", start=8, end=8))


# --- LinearArrival ----------------------------------------------------------


def test_linear_increasing_back_loads_arrivals():
    """``direction='increasing'`` density rises with period — most
    entities arrive late. With 100 entities on [0, 10), the median
    start_period should sit in the upper half of the range.
    """
    cfg = _build(
        LinearArrival(kind="linear", start=0, end=10, direction="increasing"),
        count=100,
    )
    starts = sorted(e.start_period for e in cfg.entities)
    median = starts[len(starts) // 2]
    assert median >= 5, f"increasing linear should back-load; median={median}"


def test_linear_decreasing_front_loads_arrivals():
    """``direction='decreasing'`` density falls with period — most
    entities arrive early. Median should sit in the lower half.
    """
    cfg = _build(
        LinearArrival(kind="linear", start=0, end=10, direction="decreasing"),
        count=100,
    )
    starts = sorted(e.start_period for e in cfg.entities)
    median = starts[len(starts) // 2]
    assert median <= 4, f"decreasing linear should front-load; median={median}"


def test_linear_draws_within_bounds():
    cfg = _build(
        LinearArrival(kind="linear", start=2, end=8, direction="increasing"),
        count=50,
    )
    assert all(2 <= e.start_period < 8 for e in cfg.entities)


# --- StepArrival ------------------------------------------------------------


def test_step_assigns_each_block_in_order():
    """Step is deterministic — no RNG. Counts come from
    ``round(fraction * count)`` with the last block absorbing rounding
    remainder. With count=10 and blocks [(0, 0.5), (6, 0.3), (12, 0.2)],
    counts should be (5, 3, 2) and entity assignment should be
    period-grouped in declaration order.
    """
    cfg = _build(
        StepArrival(
            kind="step",
            blocks=[
                StepArrivalBlock(period=0, fraction=0.5),
                StepArrivalBlock(period=6, fraction=0.3),
                StepArrivalBlock(period=10, fraction=0.2),
            ],
        ),
        count=10,
    )
    starts = [e.start_period for e in cfg.entities]
    counts = Counter(starts)
    assert counts == {0: 5, 6: 3, 10: 2}


def test_step_rounding_remainder_absorbed_by_last_block():
    """count=7 + blocks 0.5/0.3/0.2 should not produce 7 entities lost
    to rounding. Last block absorbs remainder so the total exactly
    equals count.
    """
    cfg = _build(
        StepArrival(
            kind="step",
            blocks=[
                StepArrivalBlock(period=0, fraction=0.5),
                StepArrivalBlock(period=6, fraction=0.3),
                StepArrivalBlock(period=10, fraction=0.2),
            ],
        ),
        count=7,
    )
    assert len(cfg.entities) == 7
    counts = Counter(e.start_period for e in cfg.entities)
    # 0.5*7=3.5→4, 0.3*7=2.1→2, last absorbs remainder = 7-4-2=1.
    assert counts == {0: 4, 6: 2, 10: 1}


def test_step_rejects_fractions_not_summing_to_one():
    """Step model_validator catches blocks whose fractions don't sum to
    1.0 (±0.001 tolerance).
    """
    with pytest.raises(ValidationError, match="fractions must sum to 1.0"):
        StepArrival(
            kind="step",
            blocks=[
                StepArrivalBlock(period=0, fraction=0.5),
                StepArrivalBlock(period=6, fraction=0.3),
            ],
        )


# --- ExplicitArrival --------------------------------------------------------


def test_explicit_passes_through_user_provided_periods():
    """No RNG, no rounding — user-supplied list maps 1:1 onto entities."""
    cfg = _build(
        ExplicitArrival(kind="explicit", start_periods=[0, 3, 5, 7, 9]),
        count=5,
    )
    starts = [e.start_period for e in cfg.entities]
    assert starts == [0, 3, 5, 7, 9]


def test_explicit_length_must_match_segment_count():
    """SegmentInput model_validator rejects the mismatch — count=5 with
    a 3-element start_periods list shouldn't reach interpretation.
    """
    with pytest.raises(ValidationError, match="lengths must match"):
        _build(
            ExplicitArrival(kind="explicit", start_periods=[0, 3, 5]),
            count=5,
        )


def test_explicit_rejects_negative_start_period():
    with pytest.raises(ValidationError, match="must be >= 0"):
        ExplicitArrival(kind="explicit", start_periods=[0, -1, 3])


# --- Load-time validator ----------------------------------------------------


def test_validator_rejects_entity_past_min_active_deadline():
    """Entity at start_period=11 in a 12-period window leaves 1 active
    period — below MIN_ACTIVE_PERIODS=2. The PlotsimConfig
    model_validator (``_cold_start_active_periods_gate``) raises.
    """
    with pytest.raises(ValueError, match="MIN_ACTIVE_PERIODS"):
        _build(
            ExplicitArrival(kind="explicit", start_periods=[0, 5, 11]),
            count=3,
        )


def test_validator_accepts_entity_at_exact_deadline():
    """Boundary case: start_period=10 in a 12-period window leaves
    exactly 2 active periods (= MIN_ACTIVE_PERIODS). Should be accepted
    — the gate is ``start_period <= n_periods - MIN_ACTIVE_PERIODS``,
    not strictly less than.
    """
    cfg = _build(
        ExplicitArrival(kind="explicit", start_periods=[0, 5, 10]),
        count=3,
    )
    assert cfg.entities[2].start_period == 10


def test_validator_accepts_default_only_config():
    """Configs with no cold-start entities pass cleanly (no false
    positives from the new validator).
    """
    cfg = _build(arrival=None, count=5)
    assert validate_cold_start_active_periods(cfg) == []


# --- Determinism ------------------------------------------------------------


def test_same_seed_produces_same_arrival_schedule():
    cfg_a = _build(UniformArrival(kind="uniform", start=0, end=8), count=20, seed=42)
    cfg_b = _build(UniformArrival(kind="uniform", start=0, end=8), count=20, seed=42)
    a = [e.start_period for e in cfg_a.entities]
    b = [e.start_period for e in cfg_b.entities]
    assert a == b


def test_different_seeds_produce_different_schedules():
    """Sanity: not all seeds collapse to the same draw."""
    a = [
        e.start_period
        for e in _build(UniformArrival(kind="uniform", start=0, end=10), count=20, seed=1).entities
    ]
    b = [
        e.start_period
        for e in _build(
            UniformArrival(kind="uniform", start=0, end=10), count=20, seed=999
        ).entities
    ]
    assert a != b


def test_step_and_explicit_consume_zero_rng_draws():
    """Adding a deterministic-shape segment (step or explicit) between
    two random-shape segments must NOT shift the random shapes' draws.
    Pin: build with [uniform, uniform], then [uniform, step, uniform].
    The first uniform's draws must be identical; the second uniform's
    draws must also be identical (because the step segment consumes
    zero RNG state).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        def _two_uniforms():
            cfg = create(
                about="rng iso",
                unit="company",
                window=("2024-01", "2024-12"),
                metrics=[{"name": "m", "type": "score", "polarity": "positive"}],
                segments=[
                    {
                        "name": "a",
                        "count": 5,
                        "archetype": "growth",
                        "arrival": UniformArrival(kind="uniform", start=0, end=8),
                    },
                    {
                        "name": "b",
                        "count": 5,
                        "archetype": "growth",
                        "arrival": UniformArrival(kind="uniform", start=0, end=8),
                    },
                ],
                seed=7,
            )
            return [e.start_period for e in cfg.entities]

        def _uniform_step_uniform():
            cfg = create(
                about="rng iso",
                unit="company",
                window=("2024-01", "2024-12"),
                metrics=[{"name": "m", "type": "score", "polarity": "positive"}],
                segments=[
                    {
                        "name": "a",
                        "count": 5,
                        "archetype": "growth",
                        "arrival": UniformArrival(kind="uniform", start=0, end=8),
                    },
                    {
                        "name": "step",
                        "count": 4,
                        "archetype": "growth",
                        "arrival": StepArrival(
                            kind="step",
                            blocks=[
                                StepArrivalBlock(period=0, fraction=0.5),
                                StepArrivalBlock(period=4, fraction=0.5),
                            ],
                        ),
                    },
                    {
                        "name": "b",
                        "count": 5,
                        "archetype": "growth",
                        "arrival": UniformArrival(kind="uniform", start=0, end=8),
                    },
                ],
                seed=7,
            )
            # Slice out the two uniform segments (positions 0–4 and 9–13).
            starts = [e.start_period for e in cfg.entities]
            return starts[:5] + starts[9:14]

    pair = _two_uniforms()
    triple = _uniform_step_uniform()
    assert pair == triple, (
        "step segment shifted RNG state — uniform draws diverged when a "
        "step segment was inserted between them"
    )


# --- End-to-end with engine generation --------------------------------------


def test_arrival_distribution_flows_through_to_fact_row_count():
    """Cohort-mix evolution must reach the engine output: a uniform
    arrival across 100 entities produces fact tables where the total
    row count is below ``100 * n_periods`` because cold-start entities
    drop their pre-arrival rows (M8a's row filter).
    """
    cfg = _build(UniformArrival(kind="uniform", start=0, end=8), count=20, seed=42)
    n_periods = cfg.time_window.period_count()
    rng = np.random.default_rng(cfg.seed)
    tables, _state = generate_tables_with_state(cfg, rng)
    fct = next(df for name, df in tables.items() if name.startswith("fct_"))
    # Total rows = sum over entities of (n_periods - start_period).
    expected = sum(n_periods - e.start_period for e in cfg.entities)
    assert len(fct) == expected
    # And it's strictly less than the no-cold-start baseline.
    assert len(fct) < 20 * n_periods


def test_arrival_distribution_visible_in_manifest_active_windows():
    """0.6-M8a's manifest ``active_window`` field carries the per-entity
    arrival period the M8b interpreter draws. This is the bridge the two
    phases share.
    """
    from plotsim.manifest import build_manifest

    cfg = _build(
        ExplicitArrival(kind="explicit", start_periods=[0, 4, 7]),
        count=3,
    )
    rng = np.random.default_rng(cfg.seed)
    tables, state = generate_tables_with_state(cfg, rng)
    manifest = build_manifest(cfg, state.trajectories, tables)
    by_entity = {a.entity: a.active_window for a in manifest.archetype_assignments}
    n_periods = cfg.time_window.period_count()
    assert by_entity["s_0000"].start == 0
    assert by_entity["s_0000"].end == n_periods
    assert by_entity["s_0001"].start == 4
    assert by_entity["s_0002"].start == 7
