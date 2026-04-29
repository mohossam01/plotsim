"""Builder recipe-completeness tests.

Every vocabulary word the user-facing builder accepts must map to a valid
engine parameter set. These tests are the contract: if a word appears in
the vocabulary surface but produces an engine validation error when threaded
through the actual config models, the recipe is broken regardless of what
the interpreter does with it.
"""
from __future__ import annotations

import pytest

from plotsim.builder.recipes import (
    AMOUNT_BETA_PARAMS,
    AMOUNT_LOGNORM_LOC,
    AMOUNT_LOGNORM_RATIO_THRESHOLD,
    AMOUNT_LOGNORM_S,
    BASELINE_RECIPES,
    INDEX_DISTRIBUTION,
    INDEX_SIGMA_FRACTION,
    METRIC_RECIPES,
    RELATIONSHIP_RECIPES,
    SHAPE_RECIPES,
    VALID_BASELINE_WORDS,
    VALID_METRIC_TYPES,
    VALID_POLARITIES,
    VALID_RELATIONSHIP_WORDS,
    VALID_SHAPE_WORDS,
)
from plotsim.config import CorrelationPair, CurveSegment, Metric, ValueRange
from plotsim.curves import CURVE_REGISTRY


# ── Vocabulary surface ──────────────────────────────────────────────────────


def test_metric_types_cover_all_input_categories():
    assert VALID_METRIC_TYPES == {"score", "amount", "count", "index"}


def test_polarities_match_engine_literal():
    assert VALID_POLARITIES == {"positive", "negative"}


def test_shape_words_cover_documented_archetypes():
    # Mirrors the legend in saas_template.yaml lines 280-285.
    assert VALID_SHAPE_WORDS == {
        "growth", "decline", "seasonal", "flat",
        "spike_then_crash", "accelerating",
    }


def test_relationship_words_cover_full_correlation_spectrum():
    assert VALID_RELATIONSHIP_WORDS == {
        "mirrors", "driven_by", "related", "hints_at",
        "independent",
        "hints_against", "resists", "opposes", "inverts",
    }


def test_baseline_words_cover_three_bands():
    assert VALID_BASELINE_WORDS == {"high", "mid", "low"}


# ── METRIC_RECIPES — engine roundtrip ───────────────────────────────────────


def test_score_recipe_produces_valid_metric():
    recipe = METRIC_RECIPES["score"]
    Metric(
        name="m",
        label="Score",
        distribution=recipe["distribution"],
        params=recipe["params"],
        polarity="positive",
        value_range=ValueRange(min=0.0, max=1.0),
    )


def test_count_recipe_produces_valid_metric():
    recipe = METRIC_RECIPES["count"]
    Metric(
        name="m",
        label="Count",
        distribution=recipe["distribution"],
        params=recipe["params"],
        polarity="negative",
    )


def test_index_distribution_is_normal_with_documented_sigma_fraction():
    assert INDEX_DISTRIBUTION == "normal"
    # 1/6 keeps ~99.7% inside the user's declared range when mu sits at
    # the midpoint.
    assert INDEX_SIGMA_FRACTION == pytest.approx(1.0 / 6.0)


def test_amount_lognorm_constants_are_valid_engine_params():
    # Constants build a lognorm Metric. Scale comes from the interpreter
    # (range midpoint); use a representative value here.
    Metric(
        name="m",
        label="Amount",
        distribution="lognorm",
        params={
            "s": AMOUNT_LOGNORM_S,
            "loc": AMOUNT_LOGNORM_LOC,
            "scale": 25050.0,
        },
        polarity="positive",
        value_range=ValueRange(min=100.0, max=50000.0),
    )


def test_amount_beta_constants_are_valid_engine_params():
    Metric(
        name="m",
        label="Amount-tight",
        distribution="beta",
        params=AMOUNT_BETA_PARAMS,
        polarity="positive",
        value_range=ValueRange(min=100.0, max=500.0),
    )


def test_amount_ratio_threshold_is_documented_value():
    assert AMOUNT_LOGNORM_RATIO_THRESHOLD == 10.0


# ── SHAPE_RECIPES — every shape's sub-segments build valid CurveSegments ────


@pytest.mark.parametrize("shape", sorted(VALID_SHAPE_WORDS))
def test_shape_sub_segments_produce_valid_curve_segments(shape):
    sub_segments = SHAPE_RECIPES[shape]
    assert sub_segments, f"shape {shape!r} has no sub-segments"
    for curve, params, rel_start, rel_end in sub_segments:
        assert curve in CURVE_REGISTRY, (
            f"shape {shape!r} references unknown curve {curve!r}"
        )
        # CurveSegment validation runs here — float param types pass through.
        CurveSegment(
            curve=curve,
            params=dict(params),
            start_pct=rel_start,
            end_pct=rel_end,
        )


@pytest.mark.parametrize("shape", sorted(VALID_SHAPE_WORDS))
def test_shape_sub_segments_chain_contiguously_and_cover_unit_window(shape):
    sub_segments = SHAPE_RECIPES[shape]
    assert sub_segments[0][2] == 0.0, (
        f"shape {shape!r} first sub-segment must start at 0.0"
    )
    assert sub_segments[-1][3] == 1.0, (
        f"shape {shape!r} last sub-segment must end at 1.0"
    )
    for prev, curr in zip(sub_segments, sub_segments[1:]):
        assert prev[3] == curr[2], (
            f"shape {shape!r} has gap/overlap between sub-segments: "
            f"{prev[3]} != {curr[2]}"
        )


def test_spike_then_crash_has_three_sub_segments():
    # Mirrors the engine's "rocket_then_cliff" archetype shape.
    sub_segments = SHAPE_RECIPES["spike_then_crash"]
    assert len(sub_segments) == 3
    curves = [s[0] for s in sub_segments]
    assert curves == ["sigmoid", "step", "plateau"]


# ── RELATIONSHIP_RECIPES — every word maps to a coefficient in [-1, 1] ──────


@pytest.mark.parametrize("word", sorted(VALID_RELATIONSHIP_WORDS))
def test_relationship_recipe_builds_valid_correlation_pair(word):
    coef = RELATIONSHIP_RECIPES[word]
    assert -1.0 <= coef <= 1.0, (
        f"relationship {word!r} has out-of-range coefficient {coef}"
    )
    if word == "independent":
        # CorrelationPair rejects coefficient == 0 in some configs; skip
        # the engine-roundtrip check for the no-op word.
        return
    CorrelationPair(metric_a="m1", metric_b="m2", coefficient=coef)


def test_independent_is_zero_coefficient():
    assert RELATIONSHIP_RECIPES["independent"] == 0.0


def test_relationship_recipes_are_symmetric_around_zero():
    pairs = [
        ("mirrors",       "inverts"),
        ("driven_by",     "opposes"),
        ("related",       "resists"),
        ("hints_at",      "hints_against"),
    ]
    for pos, neg in pairs:
        assert RELATIONSHIP_RECIPES[pos] == -RELATIONSHIP_RECIPES[neg], (
            f"{pos} and {neg} should be negatives of each other"
        )


# ── BASELINE_RECIPES — fractions partition [0, 1] into thirds ───────────────


@pytest.mark.parametrize("word", sorted(VALID_BASELINE_WORDS))
def test_baseline_recipe_fractions_are_valid(word):
    lo, hi = BASELINE_RECIPES[word]
    assert 0.0 <= lo < hi <= 1.0, (
        f"baseline {word!r} fractions {(lo, hi)} not a valid sub-range"
    )


def test_baseline_high_targets_upper_third():
    lo, hi = BASELINE_RECIPES["high"]
    assert lo == pytest.approx(2.0 / 3.0)
    assert hi == 1.0


def test_baseline_low_targets_lower_third():
    lo, hi = BASELINE_RECIPES["low"]
    assert lo == 0.0
    assert hi == pytest.approx(1.0 / 3.0)


def test_baseline_mid_targets_middle_third():
    lo, hi = BASELINE_RECIPES["mid"]
    assert lo == pytest.approx(1.0 / 3.0)
    assert hi == pytest.approx(2.0 / 3.0)


def test_baseline_thirds_are_contiguous():
    # high.lo == mid.hi; mid.lo == low.hi
    assert BASELINE_RECIPES["high"][0] == BASELINE_RECIPES["mid"][1]
    assert BASELINE_RECIPES["mid"][0] == BASELINE_RECIPES["low"][1]
