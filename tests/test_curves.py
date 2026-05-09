"""Tests for plotsim.curves — Mission 002 acceptance criteria."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from plotsim import load_config
from plotsim.curves import (
    CURVE_REGISTRY,
    compound,
    evaluate_segment,
    exp_decay,
    logistic,
    oscillating,
    plateau,
    sawtooth,
    sigmoid,
    step,
)

ROOT = Path(__file__).resolve().parent.parent
SAAS_YAML = ROOT / "plotsim" / "configs" / "sample_saas.yaml"
HR_YAML = ROOT / "plotsim" / "configs" / "sample_hr.yaml"

CURVE_FNS = [sigmoid, exp_decay, step, logistic, plateau, oscillating, compound, sawtooth]
EPS = 1e-9


# --- Registry ---


def test_registry_has_all_eight_curves():
    assert set(CURVE_REGISTRY) == {
        "sigmoid",
        "exp_decay",
        "step",
        "logistic",
        "plateau",
        "oscillating",
        "compound",
        "sawtooth",
    }


def test_registry_values_are_callables():
    for name, fn in CURVE_REGISTRY.items():
        assert callable(fn), f"{name} is not callable"


# --- Shape & bounds (acceptance: outputs clamped to [0,1]) ---


@pytest.mark.parametrize("fn", CURVE_FNS)
def test_returns_ndarray_and_bounded(fn):
    t = np.linspace(0.0, 1.0, 100)
    out = fn(t)
    assert isinstance(out, np.ndarray)
    assert out.shape == t.shape
    assert out.min() >= 0.0 - EPS
    assert out.max() <= 1.0 + EPS


# --- Edge cases ---


@pytest.mark.parametrize("fn", CURVE_FNS)
def test_empty_input_returns_empty(fn):
    t = np.array([], dtype=float)
    out = fn(t)
    assert isinstance(out, np.ndarray)
    assert out.size == 0


@pytest.mark.parametrize("fn", CURVE_FNS)
def test_single_element(fn):
    t = np.array([0.5])
    out = fn(t)
    assert out.shape == (1,)
    assert 0.0 - EPS <= out[0] <= 1.0 + EPS


@pytest.mark.parametrize("fn", CURVE_FNS)
def test_all_zeros(fn):
    t = np.zeros(10)
    out = fn(t)
    assert out.shape == (10,)
    assert np.all((out >= 0.0 - EPS) & (out <= 1.0 + EPS))


@pytest.mark.parametrize("fn", CURVE_FNS)
def test_all_ones(fn):
    t = np.ones(10)
    out = fn(t)
    assert out.shape == (10,)
    assert np.all((out >= 0.0 - EPS) & (out <= 1.0 + EPS))


# --- Per-curve shape properties ---


def test_sigmoid_rising():
    t = np.linspace(0.0, 1.0, 100)
    out = sigmoid(t, rising=True)
    assert out[-1] > out[0]


def test_sigmoid_falling():
    t = np.linspace(0.0, 1.0, 100)
    out = sigmoid(t, rising=False)
    assert out[-1] < out[0]


def test_exp_decay_monotonically_decreasing():
    t = np.linspace(0.0, 1.0, 100)
    out = exp_decay(t)
    assert np.all(np.diff(out) <= EPS)


def test_step_two_distinct_values():
    t = np.linspace(0.0, 1.0, 100)
    out = step(t)
    assert len(np.unique(out)) == 2


def test_step_respects_threshold():
    t = np.array([0.0, 0.3, 0.7, 1.0])
    out = step(t, threshold=0.5, before=1.0, after=0.0)
    np.testing.assert_array_equal(out, np.array([1.0, 1.0, 0.0, 0.0]))


def test_logistic_monotonically_increasing():
    t = np.linspace(0.0, 1.0, 100)
    out = logistic(t)
    assert np.all(np.diff(out) >= -EPS)


def test_logistic_bounded_by_ceiling():
    t = np.linspace(0.0, 1.0, 100)
    out = logistic(t, ceiling=0.7)
    assert out.max() <= 0.7 + EPS


def test_plateau_constant():
    t = np.linspace(0.0, 1.0, 50)
    out = plateau(t, level=0.42)
    assert np.allclose(out, 0.42)


def test_oscillating_bounds():
    t = np.linspace(0.0, 1.0, 200)
    out = oscillating(t, period=3.0, amplitude=0.3, center=0.5)
    assert out.min() >= 0.5 - 0.3 - EPS
    assert out.max() <= 0.5 + 0.3 + EPS


def test_compound_monotonically_increasing():
    t = np.linspace(0.0, 1.0, 100)
    out = compound(t)
    assert np.all(np.diff(out) >= -EPS)


def test_sawtooth_has_at_least_period_local_maxima():
    t = np.linspace(0.0, 1.0, 300)
    out = sawtooth(t, period=4.0, amplitude=0.8, base=0.1)
    peaks = int(np.sum((out[1:-1] > out[:-2]) & (out[1:-1] > out[2:])))
    assert peaks >= 4


# --- Determinism ---


@pytest.mark.parametrize("fn", CURVE_FNS)
def test_deterministic(fn):
    t = np.linspace(0.0, 1.0, 50)
    np.testing.assert_array_equal(fn(t), fn(t))


# --- Segment evaluator ---


def test_evaluate_segment_dispatches_to_each_curve():
    t = np.linspace(0.0, 1.0, 50)
    for name in CURVE_REGISTRY:
        out = evaluate_segment(t, name, {})
        assert out.shape == t.shape
        assert out.min() >= 0.0 - EPS
        assert out.max() <= 1.0 + EPS


def test_evaluate_segment_unknown_raises():
    with pytest.raises(ValueError, match="unknown curve type"):
        evaluate_segment(np.array([0.0, 0.5, 1.0]), "not_a_curve", {})


def test_evaluate_segment_passes_params_through():
    t = np.linspace(0.0, 1.0, 100)
    flat_02 = evaluate_segment(t, "plateau", {"level": 0.2})
    flat_08 = evaluate_segment(t, "plateau", {"level": 0.8})
    assert np.allclose(flat_02, 0.2)
    assert np.allclose(flat_08, 0.8)


def test_evaluate_segment_none_params_uses_defaults():
    t = np.linspace(0.0, 1.0, 50)
    out = evaluate_segment(t, "sigmoid", None)
    assert out.shape == t.shape
    assert out[-1] > out[0]


def test_evaluate_segment_clamps_to_unit_interval():
    t = np.linspace(0.0, 1.0, 100)
    out = evaluate_segment(t, "oscillating", {"period": 2.0, "amplitude": 0.9, "center": 0.5})
    assert out.min() >= 0.0
    assert out.max() <= 1.0


# --- Real-world validation against sample YAML archetypes ---


@pytest.mark.parametrize("yaml_path", [SAAS_YAML, HR_YAML])
def test_sample_archetypes_evaluate_without_errors(yaml_path):
    cfg = load_config(yaml_path)
    t = np.linspace(0.0, 1.0, 100)
    for arch in cfg.archetypes:
        for seg in arch.curve_segments:
            out = evaluate_segment(t, seg.curve, dict(seg.params))
            assert out.shape == t.shape
            assert out.min() >= 0.0 - EPS
            assert out.max() <= 1.0 + EPS
