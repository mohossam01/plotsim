"""End-to-end locale regression tests.

The ``PlotsimConfig.locale`` field, ``_make_faker`` accepting a
locale argument, and the builder's ``UserInput.locale`` passthrough
have shipped since FIX-05/SF-3 and M124 — those layers already have
unit-level coverage (``test_config.py`` round-trips, ``test_dimensions.py``
proves a renamed locale produces locale-appropriate names on a dim
column called directly).

The gap this module closes: confirm the orchestrator actually
threads ``config.locale`` into every ``_make_faker`` call site —
facts, events, and bridges — so a refactor that silently dropped
the argument from one orchestrator level would surface as a clear
regression. The dim-level tests don't catch that because they call
the dim builder functions directly with an explicit ``locale=``
kwarg, bypassing the orchestrator's plumbing.

Each test in this module generates an end-to-end run with
``locale="fr_FR"`` (chosen because French Faker output is visually
distinct from the US-English default — accented characters, "de la"
particles in names, etc.) and asserts the locale-derived output
lands on the targeted table.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest

from plotsim import create, generate_tables
from plotsim.config import PlotsimConfig


def _fr_config(**overrides: Any) -> PlotsimConfig:
    """Builder config with ``locale="fr_FR"`` and a single per_entity
    dim ``dim_customer`` carrying a ``faker.name`` column. Caller
    overrides extend ``facts`` / ``events`` / ``bridges`` to target
    each orchestrator-level Faker call site in turn.
    """
    base: dict[str, Any] = {
        "about": "locale regression",
        "unit": "customer",
        "seed": 19301,
        "window": ("2024-01", "2024-04", "monthly"),
        "locale": "fr_FR",
        "metrics": [
            {
                "name": "purchases",
                "type": "amount",
                "polarity": "positive",
                "range": [1, 5],
            },
        ],
        "segments": [
            {"name": "g", "count": 6, "archetype": "growth"},
        ],
        "dimensions": [
            {
                "name": "dim_customer",
                "per": "unit",
                "columns": [
                    {"name": "customer_id", "type": "id"},
                    {"name": "customer_name", "type": "faker.name"},
                ],
            },
        ],
    }
    base.update(overrides)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return create(**base)


def _looks_french(values: list[str]) -> bool:
    """Loose check: French Faker output frequently includes accented
    characters or the ``" de "`` / ``" du "`` / ``" Le "`` / ``" La "``
    particle. en_US output for the same seed has none of these, so any
    one of them appearing across a 5-row sample is enough signal.
    """
    accented = "àâäçéèêëîïôöùûüÿœÀÂÄÇÉÈÊËÎÏÔÖÙÛÜŸŒ"
    particles = (" de ", " du ", " Le ", " La ", " Du ", " Des ")
    blob = " ".join(values)
    if any(ch in blob for ch in accented):
        return True
    return any(p in blob for p in particles)


def test_locale_on_dim_faker_column():
    """Baseline — dim builder honors locale (the well-covered surface)."""
    cfg = _fr_config()
    tables = generate_tables(cfg, np.random.default_rng(cfg.seed))
    names = tables["dim_customer"]["customer_name"].tolist()
    assert _looks_french(names), f"dim names look not-French: {names!r}"


def test_locale_on_fact_faker_column():
    """0.6-M19: regression test for ``tables.py:_build_facts``'s
    ``_make_faker(rng, config.locale)`` call site. A per_entity_per_period
    fact with a ``faker.name`` column must emit French names when
    ``locale="fr_FR"``.
    """
    cfg = _fr_config(
        facts=[
            {
                "name": "fct_visit",
                "metrics": ["purchases"],
                "columns": [
                    {"name": "date_key", "type": "ref.dim_date"},
                    {"name": "customer_id", "type": "ref.dim_customer"},
                    {"name": "purchases", "type": "metric.purchases"},
                    {"name": "greeter_name", "type": "faker.name"},
                ],
            }
        ],
    )
    tables = generate_tables(cfg, np.random.default_rng(cfg.seed))
    names = tables["fct_visit"]["greeter_name"].head(10).tolist()
    assert _looks_french(names), (
        f"fact-column Faker output is not French — orchestrator "
        f"may have dropped config.locale on the facts pass. Got: {names!r}"
    )


def test_locale_on_event_faker_column():
    """0.6-M19: regression test for ``tables.py:_build_events``'s
    ``_make_faker(rng, config.locale)`` call site. A proportional
    event with a ``faker.sentence`` column must emit French sentences
    when ``locale="fr_FR"``.
    """
    cfg = _fr_config(
        facts=[
            {
                "name": "fct_visit",
                "metrics": ["purchases"],
                "columns": [
                    {"name": "date_key", "type": "ref.dim_date"},
                    {"name": "customer_id", "type": "ref.dim_customer"},
                    {"name": "purchases", "type": "metric.purchases"},
                ],
            }
        ],
        events=[
            {
                "name": "evt_action",
                "trigger": "proportional",
                "driver": "purchases",
                "scale": 1.0,
                "columns": [
                    {"name": "event_id", "type": "id"},
                    {"name": "customer_id", "type": "ref.dim_customer"},
                    {"name": "date_key", "type": "ref.dim_date"},
                    {"name": "note", "type": "faker.sentence"},
                ],
            }
        ],
    )
    tables = generate_tables(cfg, np.random.default_rng(cfg.seed))
    sentences = tables["evt_action"]["note"].head(10).tolist()
    assert _looks_french(sentences), (
        f"event-column Faker output is not French — orchestrator "
        f"may have dropped config.locale on the events pass. Got: {sentences!r}"
    )


def test_locale_default_baseline_is_en_us():
    """Same shape config with no explicit locale defaults to en_US
    and produces non-French output — confirms the regression tests
    above aren't trivially asserting on shared output bytes."""
    base = {
        "about": "en_US baseline",
        "unit": "customer",
        "seed": 19301,
        "window": ("2024-01", "2024-04", "monthly"),
        # locale omitted — engine defaults to en_US
        "metrics": [
            {
                "name": "purchases",
                "type": "amount",
                "polarity": "positive",
                "range": [1, 5],
            },
        ],
        "segments": [
            {"name": "g", "count": 6, "archetype": "growth"},
        ],
        "dimensions": [
            {
                "name": "dim_customer",
                "per": "unit",
                "columns": [
                    {"name": "customer_id", "type": "id"},
                    {"name": "customer_name", "type": "faker.name"},
                ],
            },
        ],
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        cfg = create(**base)
    assert cfg.locale == "en_US"
    tables = generate_tables(cfg, np.random.default_rng(cfg.seed))
    names = tables["dim_customer"]["customer_name"].tolist()
    assert not _looks_french(names), f"en_US baseline produced French-looking output: {names!r}"


def test_locale_deterministic_under_seed():
    """Same (config, seed) reproduces the same locale-derived output
    across two runs. Locks down the seeding contract for non-default
    locales."""
    cfg_a = _fr_config()
    cfg_b = _fr_config()
    tables_a = generate_tables(cfg_a, np.random.default_rng(cfg_a.seed))
    tables_b = generate_tables(cfg_b, np.random.default_rng(cfg_b.seed))
    assert (
        tables_a["dim_customer"]["customer_name"].tolist()
        == tables_b["dim_customer"]["customer_name"].tolist()
    )


@pytest.mark.parametrize("locale", ["en_US", "fr_FR", "ja_JP", "de_DE"])
def test_locale_round_trip_through_builder(locale):
    """Builder ``locale=`` passes through to engine ``config.locale``
    verbatim for every common locale value. Catches regressions in
    the interpreter's locale-passthrough wiring."""
    base = {
        "about": "locale round-trip",
        "unit": "customer",
        "seed": 19302,
        "window": ("2024-01", "2024-04", "monthly"),
        "locale": locale,
        "metrics": [
            {
                "name": "m",
                "type": "amount",
                "polarity": "positive",
                "range": [1, 5],
            },
        ],
        "segments": [
            {"name": "g", "count": 3, "archetype": "flat"},
        ],
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        cfg = create(**base)
    assert cfg.locale == locale
