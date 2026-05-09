"""Composite archetype DSL parser tests.

Mirrors the acceptance criteria in mission-115-builder.md ::Composite
archetype parser:: section, plus float-contiguity and engine-roundtrip
checks.
"""

from __future__ import annotations

import pytest

from plotsim.builder.parser import ArchetypeParseError, parse_archetype
from plotsim.builder.recipes import VALID_SHAPE_WORDS
from plotsim.config import Archetype


# ── Single shape ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("shape", sorted(VALID_SHAPE_WORDS))
def test_single_shape_covers_full_window(shape):
    segments = parse_archetype(shape, n_periods=24)
    assert segments, f"single-shape spec {shape!r} produced no segments"
    assert segments[0].start_pct == 0.0
    assert segments[-1].end_pct == 1.0


def test_single_shape_leading_and_trailing_whitespace_tolerated():
    a = parse_archetype("growth", n_periods=24)
    b = parse_archetype("   growth   ", n_periods=24)
    assert len(a) == len(b)
    assert a[0].curve == b[0].curve


# ── Two-phase composition ───────────────────────────────────────────────────


def test_two_phase_at_midpoint_splits_window_evenly():
    # mission acceptance: "flat > decline @ 12" in 24-period window
    # → two segment groups at [0.0, 0.5] and [0.5, 1.0]
    segs = parse_archetype("flat > decline @ 12", n_periods=24)
    assert len(segs) == 2
    assert segs[0].start_pct == 0.0
    assert segs[0].end_pct == 0.5
    assert segs[0].curve == "plateau"  # flat → plateau
    assert segs[1].start_pct == 0.5
    assert segs[1].end_pct == 1.0
    assert segs[1].curve == "exp_decay"  # decline → exp_decay


def test_two_phase_off_centre_period():
    segs = parse_archetype("growth > flat @ 6", n_periods=24)
    assert len(segs) == 2
    assert segs[0].end_pct == pytest.approx(6 / 24)
    assert segs[1].start_pct == pytest.approx(6 / 24)


# ── Three-phase composition with multi-segment middle ───────────────────────


def test_three_phase_with_multi_segment_middle_rescales_correctly():
    # mission acceptance: "growth > spike_then_crash > flat @ 8 @ 16"
    # → 5 segments (1 + 3 + 1) — spike_then_crash rescaled into [1/3, 2/3]
    segs = parse_archetype("growth > spike_then_crash > flat @ 8 @ 16", n_periods=24)
    assert len(segs) == 5

    # Phase 1: growth (sigmoid) over [0.0, 1/3]
    assert segs[0].curve == "sigmoid"
    assert segs[0].start_pct == 0.0
    assert segs[0].end_pct == pytest.approx(8 / 24)

    # Phase 2: spike_then_crash sub-segments rescaled into [1/3, 2/3]
    # Sub-segments are sigmoid (rel 0.0–0.55), step (0.55–0.65), plateau (0.65–1.0)
    phase2_start = 8 / 24
    phase2_end = 16 / 24
    width2 = phase2_end - phase2_start
    assert segs[1].curve == "sigmoid"
    assert segs[1].start_pct == phase2_start
    assert segs[1].end_pct == pytest.approx(phase2_start + 0.55 * width2)
    assert segs[2].curve == "step"
    assert segs[2].start_pct == pytest.approx(phase2_start + 0.55 * width2)
    assert segs[2].end_pct == pytest.approx(phase2_start + 0.65 * width2)
    assert segs[3].curve == "plateau"
    assert segs[3].start_pct == pytest.approx(phase2_start + 0.65 * width2)
    assert segs[3].end_pct == phase2_end

    # Phase 3: flat over [2/3, 1.0]
    assert segs[4].curve == "plateau"
    assert segs[4].start_pct == phase2_end
    assert segs[4].end_pct == 1.0


# ── Engine-roundtrip: parser output passes Archetype validation ─────────────


@pytest.mark.parametrize(
    "spec,n_periods",
    [
        ("growth", 24),
        ("decline", 12),
        ("seasonal", 36),
        ("flat", 24),
        ("spike_then_crash", 24),
        ("accelerating", 24),
        ("flat > decline @ 12", 24),
        ("growth > seasonal @ 6", 24),
        ("decline > flat > growth @ 6 @ 14", 24),
        ("growth > spike_then_crash > flat @ 8 @ 16", 24),
    ],
)
def test_parser_output_passes_archetype_contiguity_validator(spec, n_periods):
    segments = parse_archetype(spec, n_periods=n_periods)
    # Archetype._segments_cover_full_range raises on gap/overlap or
    # missing 0.0/1.0 boundaries.
    Archetype(
        name="test",
        label="test",
        description="test",
        curve_segments=segments,
    )


# ── Float-drift safety on long compositions ─────────────────────────────────


def test_three_phase_boundaries_are_bitwise_contiguous():
    # Without the rel_start==0.0 / rel_end==1.0 pinning, ps + 1.0*(pe-ps)
    # would not always equal pe in IEEE 754. Archetype validator uses ==.
    segs = parse_archetype("growth > spike_then_crash > flat @ 8 @ 16", n_periods=24)
    for prev, curr in zip(segs, segs[1:]):
        assert (
            prev.end_pct == curr.start_pct
        ), f"non-contiguous boundary: {prev.end_pct!r} vs {curr.start_pct!r}"


def test_phase_boundary_at_non_terminating_fraction_holds():
    # 8 / 24 is non-terminating in binary; this is the canonical drift case.
    segs = parse_archetype("growth > flat @ 8", n_periods=24)
    assert segs[0].end_pct == segs[1].start_pct


# ── '+' rejection ───────────────────────────────────────────────────────────


def test_layered_plus_operator_rejected_with_future_release_message():
    with pytest.raises(ArchetypeParseError) as exc:
        parse_archetype("growth + decline", n_periods=24)
    assert "future release" in str(exc.value).lower()
    assert ">" in str(exc.value)


def test_plus_inside_otherwise_valid_spec_still_rejected():
    with pytest.raises(ArchetypeParseError, match="future release"):
        parse_archetype("growth > flat + decline @ 12", n_periods=24)


# ── '@' / '>' count mismatch ────────────────────────────────────────────────


def test_two_shapes_no_period_rejected():
    with pytest.raises(ArchetypeParseError, match=r"transition\(s\) for 2 shape"):
        parse_archetype("growth > decline", n_periods=24)


def test_one_shape_with_period_rejected():
    with pytest.raises(ArchetypeParseError, match=r"transition\(s\) for 1 shape"):
        parse_archetype("growth @ 12", n_periods=24)


def test_three_shapes_only_one_period_rejected():
    with pytest.raises(ArchetypeParseError, match=r"transition\(s\) for 3 shape"):
        parse_archetype("growth > flat > decline @ 12", n_periods=24)


# ── Period range and ordering ───────────────────────────────────────────────


def test_period_zero_rejected():
    with pytest.raises(ArchetypeParseError, match="out of"):
        parse_archetype("growth > decline @ 0", n_periods=24)


def test_period_at_or_beyond_window_end_rejected():
    with pytest.raises(ArchetypeParseError, match="out of"):
        parse_archetype("growth > decline @ 24", n_periods=24)
    with pytest.raises(ArchetypeParseError, match="out of"):
        parse_archetype("growth > decline @ 100", n_periods=24)


def test_periods_not_strictly_ascending_rejected():
    with pytest.raises(ArchetypeParseError, match="ascending"):
        parse_archetype("growth > spike_then_crash > flat @ 16 @ 8", n_periods=24)


def test_duplicate_periods_rejected():
    with pytest.raises(ArchetypeParseError, match="ascending"):
        parse_archetype("growth > spike_then_crash > flat @ 8 @ 8", n_periods=24)


# ── Unknown vocabulary ──────────────────────────────────────────────────────


def test_unknown_shape_rejected_with_vocabulary_listing():
    with pytest.raises(ArchetypeParseError) as exc:
        parse_archetype("rocketship", n_periods=24)
    msg = str(exc.value)
    assert "rocketship" in msg
    assert "growth" in msg  # vocabulary suggestion


def test_typo_in_one_of_many_shapes_caught():
    with pytest.raises(ArchetypeParseError, match="declien"):
        parse_archetype("growth > declien @ 12", n_periods=24)


# ── Malformed input ─────────────────────────────────────────────────────────


def test_empty_spec_rejected():
    with pytest.raises(ArchetypeParseError, match="empty"):
        parse_archetype("", n_periods=24)


def test_whitespace_only_spec_rejected():
    with pytest.raises(ArchetypeParseError, match="empty"):
        parse_archetype("   ", n_periods=24)


def test_trailing_arrow_rejected():
    with pytest.raises(ArchetypeParseError, match="empty shape"):
        parse_archetype("growth >", n_periods=24)


def test_leading_arrow_rejected():
    with pytest.raises(ArchetypeParseError, match="empty shape"):
        parse_archetype("> growth", n_periods=24)


def test_doubled_arrow_rejected():
    with pytest.raises(ArchetypeParseError, match="empty shape"):
        parse_archetype("growth >> decline @ 12", n_periods=24)


def test_non_integer_period_rejected():
    with pytest.raises(ArchetypeParseError, match="integer"):
        parse_archetype("growth > decline @ 12.5", n_periods=24)


def test_empty_period_rejected():
    with pytest.raises(ArchetypeParseError, match="empty"):
        parse_archetype("growth > decline @ ", n_periods=24)


def test_n_periods_below_two_rejected():
    with pytest.raises(ArchetypeParseError, match="n_periods"):
        parse_archetype("growth", n_periods=1)


def test_non_string_spec_rejected():
    with pytest.raises(ArchetypeParseError, match="string"):
        parse_archetype(123, n_periods=24)  # type: ignore[arg-type]


# ── Coverage: every shape parses both standalone and in composition ─────────


@pytest.mark.parametrize("shape", sorted(VALID_SHAPE_WORDS))
def test_every_shape_works_in_composition(shape):
    # Each shape paired with growth at the midpoint of a 24-period window.
    if shape == "growth":
        spec = f"flat > {shape} @ 12"
    else:
        spec = f"growth > {shape} @ 12"
    segs = parse_archetype(spec, n_periods=24)
    Archetype(
        name="t",
        label="t",
        description="t",
        curve_segments=segs,
    )
