"""Tests for plotsim.trajectory — Mission 003 acceptance criteria.

Covers: compute_time_steps labels for all three granularities; single-,
two-, three-, and six-segment trajectory stitching; discontinuity
preservation; rounding remainder absorbed by the last segment; entity
inflection overrides; all-entity batch computation; determinism; and
end-to-end validation against both sample YAML configs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from plotsim import load_config
from plotsim.config import (
    Archetype,
    CurveSegment,
    Entity,
    TimeWindow,
)
from plotsim.trajectory import (
    _segment_boundaries,
    compute_all_trajectories,
    compute_time_steps,
    compute_trajectory,
)

ROOT = Path(__file__).resolve().parent.parent
SAAS_YAML = ROOT / "plotsim" / "configs" / "sample_saas.yaml"
HR_YAML = ROOT / "plotsim" / "configs" / "sample_hr.yaml"

EPS = 1e-9


# --- Fixtures ----------------------------------------------------------------


def _arch(segments: list[CurveSegment], name: str = "test_arch") -> Archetype:
    return Archetype(
        name=name,
        label="Test archetype",
        description="Synthetic archetype for unit tests",
        curve_segments=segments,
    )


def _one_segment_plateau(level: float = 0.5) -> Archetype:
    return _arch([
        CurveSegment(curve="plateau", params={"level": level},
                     start_pct=0.0, end_pct=1.0),
    ])


def _two_segment_step() -> Archetype:
    return _arch([
        CurveSegment(curve="plateau", params={"level": 0.9},
                     start_pct=0.0, end_pct=0.5),
        CurveSegment(curve="plateau", params={"level": 0.1},
                     start_pct=0.5, end_pct=1.0),
    ])


def _three_segment_rocket() -> Archetype:
    """Matches sample_saas rocket_then_cliff structure."""
    return _arch([
        CurveSegment(curve="sigmoid", params={"midpoint": 0.3, "steepness": 10.0},
                     start_pct=0.0, end_pct=0.55),
        CurveSegment(curve="step", params={"threshold": 0.5, "before": 1.0, "after": 0.2},
                     start_pct=0.55, end_pct=0.65),
        CurveSegment(curve="plateau", params={"level": 0.2},
                     start_pct=0.65, end_pct=1.0),
    ])


def _six_segment_arch() -> Archetype:
    """Six plateau segments at descending levels — validates many-segment stitching."""
    levels = [0.9, 0.75, 0.6, 0.45, 0.3, 0.15]
    boundaries = [0.0, 0.1, 0.3, 0.5, 0.7, 0.85, 1.0]
    segs = [
        CurveSegment(
            curve="plateau",
            params={"level": lvl},
            start_pct=boundaries[i],
            end_pct=boundaries[i + 1],
        )
        for i, lvl in enumerate(levels)
    ]
    return _arch(segs, name="six_seg")


# --- compute_time_steps ------------------------------------------------------


def test_time_steps_monthly_labels():
    tw = TimeWindow(start="2023-01", end="2023-03", granularity="monthly")
    out = compute_time_steps(tw)
    assert list(out) == ["2023-01", "2023-02", "2023-03"]


def test_time_steps_monthly_spans_year_boundary():
    tw = TimeWindow(start="2023-11", end="2024-02", granularity="monthly")
    out = compute_time_steps(tw)
    assert list(out) == ["2023-11", "2023-12", "2024-01", "2024-02"]


def test_time_steps_monthly_full_window_count():
    # sample_saas: 2023-01 to 2024-12 → 24 months
    tw = TimeWindow(start="2023-01", end="2024-12", granularity="monthly")
    out = compute_time_steps(tw)
    assert len(out) == 24
    assert out[0] == "2023-01"
    assert out[-1] == "2024-12"


def test_time_steps_first_label_matches_start():
    # The acceptance criterion: start month's label is the first entry.
    tw = TimeWindow(start="2023-05", end="2023-08", granularity="monthly")
    out = compute_time_steps(tw)
    assert out[0] == "2023-05"


def test_time_steps_daily_labels():
    # TimeWindow enforces start < end, so use Jan–Feb for a short daily sweep.
    tw = TimeWindow(start="2023-01", end="2023-02", granularity="daily")
    out = compute_time_steps(tw)
    assert len(out) == 59  # Jan (31) + Feb 2023 (28)
    assert out[0] == "2023-01-01"
    assert out[30] == "2023-01-31"
    assert out[31] == "2023-02-01"
    assert out[-1] == "2023-02-28"


def test_time_steps_daily_spans_month_with_varied_lengths():
    # Jan (31) + Feb 2023 (28) = 59 days
    tw = TimeWindow(start="2023-01", end="2023-02", granularity="daily")
    out = compute_time_steps(tw)
    assert len(out) == 59
    assert out[0] == "2023-01-01"
    assert out[-1] == "2023-02-28"


def test_time_steps_weekly_label_format():
    tw = TimeWindow(start="2023-01", end="2023-03", granularity="weekly")
    out = compute_time_steps(tw)
    # Every label matches ISO "YYYY-Www"
    for label in out:
        assert len(label) == 8 and label[4] == "-" and label[5] == "W"
        int(label[:4])
        int(label[6:])
    # Labels are ordered and unique
    assert len(set(out)) == len(out)


def test_time_steps_weekly_covers_window():
    # 2023-01-01 is a Sunday → ISO week 2022-W52. 2023-W01 begins Mon 2023-01-02.
    tw = TimeWindow(start="2023-01", end="2023-02", granularity="weekly")
    out = compute_time_steps(tw)
    assert out[0] == "2022-W52"
    # Last day of Feb 2023 is Tuesday → ISO week 2023-W09
    assert out[-1] == "2023-W09"


# --- compute_trajectory: shape, bounds ---------------------------------------


@pytest.mark.parametrize("n_periods", [1, 2, 5, 12, 24, 100])
def test_trajectory_shape(n_periods):
    out = compute_trajectory(_three_segment_rocket(), n_periods)
    assert out.shape == (n_periods,)


@pytest.mark.parametrize("n_periods", [1, 5, 24, 100])
def test_trajectory_values_in_unit_interval(n_periods):
    out = compute_trajectory(_three_segment_rocket(), n_periods)
    assert out.min() >= 0.0 - EPS
    assert out.max() <= 1.0 + EPS


def test_trajectory_rejects_zero_periods():
    with pytest.raises(ValueError):
        compute_trajectory(_three_segment_rocket(), 0)


# --- Segment boundary math ---------------------------------------------------


def test_boundaries_cover_exactly_n_periods():
    arch = _three_segment_rocket()
    for n in (1, 2, 7, 12, 24, 37, 100):
        b = _segment_boundaries(arch, n)
        assert b[0] == 0
        assert b[-1] == n
        # Monotonic non-decreasing
        assert all(b[i] <= b[i + 1] for i in range(len(b) - 1))
        # Lengths sum to n
        lengths = [b[i + 1] - b[i] for i in range(len(arch.curve_segments))]
        assert sum(lengths) == n


def test_boundaries_last_segment_absorbs_remainder():
    # rocket_then_cliff n=24: 0.55*24=13.2, 0.65*24=15.6
    # → boundaries [0, 13, 15, 24]; last segment length = 9 = remainder.
    arch = _three_segment_rocket()
    b = _segment_boundaries(arch, 24)
    assert b == [0, 13, 15, 24]


def test_boundaries_no_gaps_or_overlaps():
    arch = _three_segment_rocket()
    for n in (3, 10, 24, 77, 1000):
        b = _segment_boundaries(arch, n)
        # Adjacent segments share an endpoint (end of i == start of i+1)
        for i in range(len(b) - 1):
            assert b[i] <= b[i + 1]
        # Covered periods are contiguous — union of [b[i], b[i+1]) == [0, n)
        covered = set()
        for i in range(len(arch.curve_segments)):
            for p in range(b[i], b[i + 1]):
                covered.add(p)
        assert covered == set(range(n))


# --- Discontinuity preservation ----------------------------------------------


def test_discontinuity_preserved_at_boundary():
    """Two plateau segments at very different levels produce an abrupt jump.

    The engine must not smooth this.
    """
    arch = _two_segment_step()  # plateau 0.9 → plateau 0.1 at mid
    out = compute_trajectory(arch, n_periods=10)
    # First half ~= 0.9, second half ~= 0.1. Jump at index 5.
    assert abs(out[4] - 0.9) < EPS
    assert abs(out[5] - 0.1) < EPS
    # Confirm the jump magnitude is preserved (no interpolation across boundary).
    assert abs(out[4] - out[5]) > 0.7


def test_rocket_then_cliff_has_sharp_drop():
    """Ensures the three-segment archetype reaches its low plateau post-step.

    The sigmoid segment peaks near 1.0; the step segment itself (length 2)
    emits `before, after` in its own local axis; the final plateau fully
    settles at level=0.2. Tolerating the step segment's transient lets us
    still assert the archetype is on the low plateau by segment 3.
    """
    arch = _three_segment_rocket()
    out = compute_trajectory(arch, n_periods=24)
    # Boundaries are [0, 13, 15, 24]. Seg 1 peak near 1.0, seg 3 = plateau(0.2).
    assert out[:13].max() > 0.7
    assert np.allclose(out[15:], 0.2)
    # Overall drop from rising-peak to terminal plateau is substantial.
    assert out[:13].max() - out[15:].max() > 0.5


# --- Segment-count variations ------------------------------------------------


def test_single_segment_archetype():
    arch = _one_segment_plateau(level=0.42)
    out = compute_trajectory(arch, n_periods=15)
    assert out.shape == (15,)
    assert np.allclose(out, 0.42)


def test_two_segment_archetype():
    arch = _two_segment_step()
    out = compute_trajectory(arch, n_periods=20)
    assert out.shape == (20,)
    # First half at 0.9
    assert np.allclose(out[:10], 0.9)
    # Second half at 0.1
    assert np.allclose(out[10:], 0.1)


def test_six_segment_archetype():
    arch = _six_segment_arch()
    out = compute_trajectory(arch, n_periods=100)
    assert out.shape == (100,)
    # Values should all be one of the defined plateau levels
    expected_levels = {0.9, 0.75, 0.6, 0.45, 0.3, 0.15}
    assert set(np.round(out, 4).tolist()).issubset(
        {round(v, 4) for v in expected_levels}
    )
    # Should be monotonically non-increasing (each plateau lower than prev)
    assert np.all(np.diff(out) <= EPS)


# --- compute_all_trajectories ------------------------------------------------


def test_all_trajectories_one_per_entity_saas():
    config = load_config(SAAS_YAML)
    n_periods = 24
    trajectories = compute_all_trajectories(config, n_periods)
    entity_names = {e.name for e in config.entities}
    assert set(trajectories) == entity_names
    for name, traj in trajectories.items():
        assert traj.shape == (n_periods,)
        assert traj.min() >= 0.0 - EPS
        assert traj.max() <= 1.0 + EPS


def test_all_trajectories_one_per_entity_hr():
    config = load_config(HR_YAML)
    # HR window is 36 months (2022-01 to 2024-12)
    n_periods = 36
    trajectories = compute_all_trajectories(config, n_periods)
    entity_names = {e.name for e in config.entities}
    assert set(trajectories) == entity_names
    for traj in trajectories.values():
        assert traj.shape == (n_periods,)
        assert traj.min() >= 0.0 - EPS
        assert traj.max() <= 1.0 + EPS


def test_all_trajectories_raises_on_missing_archetype():
    """Defence-in-depth: validator should prevent this, but the fn checks too."""
    config = load_config(SAAS_YAML)
    # Construct a rogue entity referencing a non-existent archetype.
    # config is frozen, so build a shallow-hacked version via model_copy with
    # updated entities. Using a new Entity with an unknown archetype bypasses
    # PlotsimConfig's cross-ref check (which ran at construction) — we skip
    # that by building a fresh dict payload with an extra entity.
    bad_entity = Entity(name="ghost", archetype="does_not_exist", size=1)
    # model_copy with update= creates a new frozen instance; cross-ref
    # validation only fires on initial construction, so this works for the
    # defensive-code path we want to exercise.
    hacked = config.model_copy(update={
        "entities": list(config.entities) + [bad_entity],
    })
    with pytest.raises(KeyError):
        compute_all_trajectories(hacked, 12)


# --- Entity inflection override ----------------------------------------------


def test_inflection_override_shifts_boundaries_earlier():
    arch = _three_segment_rocket()  # default inflection at 0.55
    n = 24
    # Default boundaries: [0, 13, 15, 24]
    # Override inflection_month=6 → shift = 6/24 - 0.55 = -0.3
    # Shifted end_pcts: 0.55 → 0.25, 0.65 → 0.35
    # New boundaries: [0, floor(0.25*24)=6, floor(0.35*24)=8, 24]
    default_traj = compute_trajectory(arch, n)
    shifted_traj = compute_trajectory(arch, n, overrides={"inflection_month": 6})
    assert default_traj.shape == shifted_traj.shape
    # The trajectories differ — the cliff lands earlier.
    assert not np.allclose(default_traj, shifted_traj)
    # Specifically, at period 10 the default is still high (pre-cliff) but
    # shifted is already in the low plateau (post-cliff at 0.2).
    assert default_traj[10] > shifted_traj[10]
    # Confirm the inflection landed near month 6: values 7-23 should be
    # near 0.2 (post-cliff plateau).
    assert np.allclose(shifted_traj[8:], 0.2, atol=0.01)


def test_inflection_override_shifts_boundaries_later():
    arch = _three_segment_rocket()
    n = 24
    # Override inflection_month=18 → shift = 18/24 - 0.55 = 0.2
    # Shifted end_pcts: 0.75, 0.85 → boundaries [0, 18, 20, 24]
    traj = compute_trajectory(arch, n, overrides={"inflection_month": 18})
    # Pre-cliff (indices 0-17) should show the rising sigmoid.
    assert traj[17] > traj[0]
    # Post-cliff plateau starts around index 20.
    assert np.allclose(traj[20:], 0.2, atol=0.01)


def test_no_overrides_uses_archetype_unchanged():
    arch = _three_segment_rocket()
    n = 24
    a = compute_trajectory(arch, n)
    b = compute_trajectory(arch, n, overrides=None)
    c = compute_trajectory(arch, n, overrides={})
    # An overrides dict without "inflection_month" should also be a no-op.
    d = compute_trajectory(arch, n, overrides={"unused_key": 99})
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a, c)
    np.testing.assert_array_equal(a, d)


def test_inflection_override_clamped_to_bounds():
    """An inflection_month past n_periods shouldn't produce invalid boundaries."""
    arch = _three_segment_rocket()
    n = 12
    # inflection_month way out of range — shift > 1.0, boundaries clamp to n.
    traj = compute_trajectory(arch, n, overrides={"inflection_month": 100})
    assert traj.shape == (n,)
    assert traj.min() >= 0.0 - EPS
    assert traj.max() <= 1.0 + EPS


# --- Sample-config sweep -----------------------------------------------------


@pytest.mark.parametrize("yaml_path", [SAAS_YAML, HR_YAML])
def test_all_sample_archetypes_produce_valid_trajectories(yaml_path):
    config = load_config(yaml_path)
    n_periods = 24
    for arch in config.archetypes:
        traj = compute_trajectory(arch, n_periods)
        assert traj.shape == (n_periods,), f"{arch.name}: wrong shape"
        assert traj.min() >= 0.0 - EPS, f"{arch.name}: values below 0"
        assert traj.max() <= 1.0 + EPS, f"{arch.name}: values above 1"
        assert not np.any(np.isnan(traj)), f"{arch.name}: NaN in trajectory"


# --- Determinism -------------------------------------------------------------


def test_trajectory_is_deterministic():
    arch = _three_segment_rocket()
    a = compute_trajectory(arch, 24)
    b = compute_trajectory(arch, 24)
    np.testing.assert_array_equal(a, b)


def test_all_trajectories_deterministic_across_loads():
    n = 24
    t1 = compute_all_trajectories(load_config(SAAS_YAML), n)
    t2 = compute_all_trajectories(load_config(SAAS_YAML), n)
    assert set(t1) == set(t2)
    for name in t1:
        np.testing.assert_array_equal(t1[name], t2[name])
