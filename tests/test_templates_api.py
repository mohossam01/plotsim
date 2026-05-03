"""Tests for the public template-discovery API: ``list_templates`` and
``load_template`` exported from ``plotsim`` (M129).

Scope is the public surface only — the underlying YAML content is covered
by ``test_templates_m112.py``.
"""
from __future__ import annotations

import pytest

import plotsim
from plotsim.config import PlotsimConfig


EXPECTED_TEMPLATES = {
    "bare_minimum",
    "education",
    "hr",
    "marketing",
    "retail",
    "saas",
}


def test_list_templates_returns_expected_names():
    names = plotsim.list_templates()
    assert set(names) == EXPECTED_TEMPLATES


def test_list_templates_is_sorted():
    names = plotsim.list_templates()
    assert names == sorted(names)


def test_list_templates_returns_strings():
    for name in plotsim.list_templates():
        assert isinstance(name, str)
        assert "/" not in name
        assert "\\" not in name
        assert not name.endswith(".yaml")
        assert not name.endswith("_template")


@pytest.mark.parametrize("name", sorted(EXPECTED_TEMPLATES))
def test_load_template_returns_plotsim_config(name):
    cfg = plotsim.load_template(name)
    assert isinstance(cfg, PlotsimConfig)


def test_load_template_unknown_name_raises_value_error():
    with pytest.raises(ValueError, match="Unknown template"):
        plotsim.load_template("not_a_real_template")


def test_load_template_error_message_lists_available():
    with pytest.raises(ValueError) as exc_info:
        plotsim.load_template("xyz")
    msg = str(exc_info.value)
    for name in EXPECTED_TEMPLATES:
        assert name in msg


def test_public_api_exports():
    assert "list_templates" in plotsim.__all__
    assert "load_template" in plotsim.__all__
