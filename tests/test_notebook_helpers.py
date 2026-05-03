"""Tests for notebooks/_helpers.py — shared introspection helpers.

Coverage matches the M113 Phase 1 mission spec:

    - load_fixed_point() returns saas config + seed 42
    - manual_rng_replay produces deterministic draws for the documented
      distributions and parameters
    - archetype_curve_eval produces correct shape for the bundled archetypes
    - tolerance constants are positive and obey the documented direction-rule
      semantics (warn stricter than pass for floors; outlier looser than pass
      for ceilings; warn stricter than pass for ceilings)
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

# notebooks/_helpers.py isn't in a package — add its directory to the path
# so the helper module can be imported by the test runner. Mirrors how the
# notebooks themselves import it (`import _helpers`).
_NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent / "notebooks"
sys.path.insert(0, str(_NOTEBOOKS_DIR))

# notebooks/ is gitignored — skip the whole module in CI where it doesn't ship.
_helpers = pytest.importorskip(
    "_helpers", reason="notebooks/_helpers.py not available in CI"
)


def test_load_fixed_point_returns_saas_and_seed_42():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        cfg, seed = _helpers.load_fixed_point()
    assert seed == 42
    assert cfg.seed == 42
    # Saas declares three cohort entities; pin the canonical name as the
    # acceptance notebook's recommended trace target.
    entity_names = [e.name for e in cfg.entities]
    assert "acme_corp_cohort" in entity_names


def test_manual_rng_replay_lognorm_deterministic():
    a = _helpers.manual_rng_replay(
        seed=42, n_draws=10, distribution="lognorm",
        params={"s": 0.85, "scale": 1200.0},
    )
    b = _helpers.manual_rng_replay(
        seed=42, n_draws=10, distribution="lognorm",
        params={"s": 0.85, "scale": 1200.0},
    )
    assert a.shape == (10,)
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("distribution,params", [
    ("lognorm", {"s": 0.85, "scale": 1200.0}),
    ("gamma", {"shape": 2.0, "scale": 1.5}),
    ("poisson", {"lambda": 5.0}),
    ("beta", {"alpha": 2.0, "beta": 5.0}),
    ("normal", {"loc": 0.0, "sigma": 1.0}),
    ("weibull", {"shape": 1.5, "scale": 2.0}),
])
def test_manual_rng_replay_supports_documented_distributions(distribution, params):
    arr = _helpers.manual_rng_replay(
        seed=42, n_draws=5, distribution=distribution, params=params,
    )
    assert arr.shape == (5,)
    assert np.all(np.isfinite(arr))


def test_manual_rng_replay_unsupported_distribution_raises():
    with pytest.raises(ValueError, match="unsupported distribution"):
        _helpers.manual_rng_replay(
            seed=42, n_draws=1, distribution="cauchy", params={},
        )


def test_archetype_curve_eval_returns_in_unit_range():
    """Every bundled archetype's evaluated curve must stay in [0, 1] — the
    trajectory-first contract. Covers sigmoid, compound, exp_decay,
    oscillating, plateau, step shapes via the saas archetype set.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        cfg, _ = _helpers.load_fixed_point()
    n_periods = 24
    for arch in cfg.archetypes:
        traj = _helpers.archetype_curve_eval(arch, n_periods)
        assert traj.shape == (n_periods,)
        assert np.all(traj >= 0.0), f"{arch.name}: trajectory dips below 0"
        assert np.all(traj <= 1.0), f"{arch.name}: trajectory exceeds 1"


def test_archetype_curve_eval_deterministic():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        cfg, _ = _helpers.load_fixed_point()
    arch = cfg.archetypes[0]
    a = _helpers.archetype_curve_eval(arch, 24)
    b = _helpers.archetype_curve_eval(arch, 24)
    np.testing.assert_array_equal(a, b)


def test_archetype_color_lookup():
    # Saas archetypes are pinned in ARCHETYPE_COLORS.
    assert _helpers.archetype_color("rocket_then_cliff") == "#d62728"
    assert _helpers.archetype_color("steady_grower") == "#2ca02c"
    # Unknown archetypes fall through to None so callers can let matplotlib
    # assign from its default cycle.
    assert _helpers.archetype_color("nonexistent_archetype") is None


# --- Tolerance constants: positive + direction-rule semantics --------------

def test_all_tolerance_constants_non_negative():
    """Every constant must be >= 0. DETERMINISM_BYTE_PASS = 0 is the only
    legal zero; the rest are strictly positive.
    """
    for name in (
        "MONOTONIC_ARCHETYPE_PEARSON_PASS",
        "OSCILLATING_ARCHETYPE_PEARSON_PASS",
        "OSCILLATING_ARCHETYPE_PEARSON_WARN",
        "MARGINAL_MEAN_REL_PASS",
        "MARGINAL_STD_REL_PASS",
        "MARGINAL_STD_REL_OUTLIER",
        "CORRELATION_DEVIATION_PASS",
        "CORRELATION_DEVIATION_WARN",
        "CORRELATION_HIGHAM_DELTA_PASS",
        "CHOLESKY_RECONSTRUCTION_ULP_PASS",
    ):
        v = getattr(_helpers, name)
        assert v > 0, f"{name} = {v}; expected strictly positive"
    assert _helpers.DETERMINISM_BYTE_PASS == 0


def test_oscillating_pearson_warn_stricter_than_pass():
    """Pearson is a floor metric (higher better) ⇒ ``WARN > PASS``."""
    assert (
        _helpers.OSCILLATING_ARCHETYPE_PEARSON_WARN
        > _helpers.OSCILLATING_ARCHETYPE_PEARSON_PASS
    )


def test_marginal_std_outlier_looser_than_pass():
    """Δstd is a ceiling metric (lower better) ⇒ ``OUTLIER > PASS``."""
    assert (
        _helpers.MARGINAL_STD_REL_OUTLIER
        > _helpers.MARGINAL_STD_REL_PASS
    )


def test_correlation_deviation_warn_stricter_than_pass():
    """|Δ correlation| is a ceiling metric (lower better) ⇒ ``WARN < PASS``."""
    assert (
        _helpers.CORRELATION_DEVIATION_WARN
        < _helpers.CORRELATION_DEVIATION_PASS
    )


def test_archetype_colors_are_valid_hex():
    for name, color in _helpers.ARCHETYPE_COLORS.items():
        assert color.startswith("#"), f"{name}: color {color!r} not hex"
        assert len(color) == 7, f"{name}: color {color!r} not 6-digit hex"
        int(color[1:], 16)  # raises if not valid hex
