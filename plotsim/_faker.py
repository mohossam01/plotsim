"""plotsim._faker — shared deterministic Faker construction.

Both ``plotsim.tables`` and ``plotsim.dimensions`` need a Faker instance
seeded from the engine's numpy ``Generator``. The previous arrangement
defined ``_make_faker`` independently in each module, which left two
near-identical 7-line functions to drift over time. This module is the
single source of truth: importers consume ``_make_faker`` and the seed
helper from here.

The seed derivation consumes one draw from the supplied rng so that
two callers asking for a faker at the same orchestrator step land on
distinct seeds, while ``(config, seed)`` reproduction stays exact.
"""

from __future__ import annotations

import numpy as np
from faker import Faker


def _faker_seed_from_rng(rng: np.random.Generator) -> int:
    """Derive a stable 32-bit seed from ``rng`` and consume one draw to do it."""
    return int(rng.integers(0, 2**31 - 1))


def _make_faker(
    rng: np.random.Generator,
    locale: str | list[str] = "en_US",
) -> Faker:
    """Return a Faker seeded deterministically from ``rng``.

    Consumes one ``rng.integers`` draw. ``locale`` mirrors Faker's own
    constructor argument: ``"en_US"`` or a list of locale strings for
    multi-locale mixing.
    """
    fake = Faker(locale)
    fake.seed_instance(_faker_seed_from_rng(rng))
    return fake
