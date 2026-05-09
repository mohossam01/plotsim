"""Pure-data recipes mapping plain-language vocabulary to engine parameters.

The builder layer (input.py / parser.py / interpreter.py) consumes these
constants. The only plotsim import here is ``plotsim._types`` — a typing-
only module providing the Literal aliases (``CurveType``, ``Distribution``)
so the recipe data can be statically verified against the engine's
vocabulary without depending on engine code or pydantic.

Four recipe families:

- METRIC_RECIPES       — metric type → engine distribution + default params.
                          ``amount`` and ``index`` are range-conditional and
                          handled by the interpreter (constants below).
- SHAPE_RECIPES        — archetype shape word → list of curve sub-segments
                          covering [0.0, 1.0]. The DSL parser rescales these
                          into composite phase windows.
- RELATIONSHIP_RECIPES — connection word → correlation coefficient.
- BASELINE_RECIPES     — high/mid/low → (lo_fraction, hi_fraction) of the
                          metric's value range. The interpreter uses these
                          to build ``MetricOverride.value_range`` per segment.
"""

from __future__ import annotations

from plotsim._types import CurveType, Distribution


# ── Metric recipes ──────────────────────────────────────────────────────────

# Deterministic recipes (no range dependency). ``score`` and ``count`` map
# directly. ``index`` distribution is fixed (normal); its mu/sigma are
# computed by the interpreter from the user-declared range. ``amount`` is
# branched into lognorm vs beta by the interpreter using the ratio
# threshold below — see ``AMOUNT_LOGNORM_RATIO_THRESHOLD``.
METRIC_RECIPES: dict[str, dict] = {
    "score": {
        "distribution": "beta",
        "params": {"alpha": 2.0, "beta": 5.0},
    },
    "count": {
        "distribution": "poisson",
        "params": {"lambda": 5.0},
    },
}

# ``index`` — normal centered on the midpoint of the user-declared range.
INDEX_DISTRIBUTION: Distribution = "normal"
# sigma = (max - min) * INDEX_SIGMA_FRACTION. 1/6 keeps ~99.7% inside the
# declared range when mu sits at the midpoint (3-sigma rule).
INDEX_SIGMA_FRACTION = 1.0 / 6.0

# ``amount`` — lognorm if min == 0 OR (max / min) >= threshold; else beta.
AMOUNT_LOGNORM_RATIO_THRESHOLD = 10.0
AMOUNT_LOGNORM_S = 0.85
AMOUNT_LOGNORM_LOC = 0.0
# scale defaults to the midpoint of the declared range so the lognorm
# median lands roughly mid-range. The interpreter overrides this when
# constructing the params dict; the constant is exposed for tests.
AMOUNT_LOGNORM_SCALE_AT_MIDPOINT = True

AMOUNT_BETA_PARAMS = {"alpha": 2.0, "beta": 5.0}


# ── Shape recipes ───────────────────────────────────────────────────────────

# Each entry is a list of (curve, params, rel_start, rel_end) tuples.
# rel_start and rel_end are within the shape's own [0.0, 1.0] window; the
# parser rescales them into the global trajectory window. Sub-segments must
# chain contiguously (one's rel_end == next's rel_start) and cover [0.0,
# 1.0] exactly — Archetype._segments_cover_full_range enforces this on the
# global side after rescaling.
ShapeSegment = tuple[CurveType, dict, float, float]

SHAPE_RECIPES: dict[str, list[ShapeSegment]] = {
    "growth": [
        ("sigmoid", {"midpoint": 0.5, "steepness": 6.0, "rising": True}, 0.0, 1.0),
    ],
    "decline": [
        ("exp_decay", {"rate": 2.0}, 0.0, 1.0),
    ],
    "seasonal": [
        ("oscillating", {"period": 2.0, "amplitude": 0.4, "center": 0.5}, 0.0, 1.0),
    ],
    "flat": [
        ("plateau", {"level": 0.15}, 0.0, 1.0),
    ],
    "spike_then_crash": [
        ("sigmoid", {"midpoint": 0.3, "steepness": 10.0, "rising": True}, 0.0, 0.55),
        ("step", {"threshold": 0.5, "before": 1.0, "after": 0.2}, 0.55, 0.65),
        ("plateau", {"level": 0.2}, 0.65, 1.0),
    ],
    "accelerating": [
        ("compound", {"base_rate": 0.05, "acceleration": 0.02}, 0.0, 1.0),
    ],
}


# ── Relationship recipes ────────────────────────────────────────────────────

# Nine connection words spanning -0.75 to +0.75 in 0.20 (or 0.15) steps.
# The vocabulary is symmetric around ``independent`` so reversing a pair
# (``a opposes b`` ↔ ``b opposes a``) yields the same coefficient.
RELATIONSHIP_RECIPES: dict[str, float] = {
    "mirrors": 0.75,
    "driven_by": 0.55,
    "related": 0.40,
    "hints_at": 0.20,
    "independent": 0.00,
    "hints_against": -0.20,
    "resists": -0.40,
    "opposes": -0.55,
    "inverts": -0.75,
}


# ── Baseline recipes ────────────────────────────────────────────────────────

# (lo_fraction, hi_fraction) of the metric's [min, max] value range.
# ``high`` baseline produces a value_range override restricted to the upper
# third; ``low`` to the lower third; ``mid`` to the middle third. The
# interpreter applies these fractions via:
#     override_min = vmin + lo * (vmax - vmin)
#     override_max = vmin + hi * (vmax - vmin)
BASELINE_RECIPES: dict[str, tuple[float, float]] = {
    "high": (2.0 / 3.0, 1.0),
    "mid": (1.0 / 3.0, 2.0 / 3.0),
    "low": (0.0, 1.0 / 3.0),
}


# ── Vocabulary surface (used by validators) ─────────────────────────────────

VALID_METRIC_TYPES = frozenset({"score", "amount", "count", "index"})
VALID_POLARITIES = frozenset({"positive", "negative"})
VALID_SHAPE_WORDS = frozenset(SHAPE_RECIPES.keys())
VALID_RELATIONSHIP_WORDS = frozenset(RELATIONSHIP_RECIPES.keys())
VALID_BASELINE_WORDS = frozenset(BASELINE_RECIPES.keys())
