"""Typing-only aliases and TypeGuards used across the engine and builder.

Why this module exists
----------------------
Several engine string fields are restricted to a fixed vocabulary
(``CurveType``, ``Distribution``, ``Dtype``, ``Grain``, etc.). The Literal
aliases originally lived in ``plotsim.config`` next to the pydantic models
that consume them, but ``plotsim.builder.recipes`` — which holds the
hardcoded vocabulary data the rest of the engine dispatches on — is
documented as engine-import-free. Lifting the aliases here lets both
``recipes.py`` and ``config.py`` import the same Literal types without
``recipes.py`` having to depend on pydantic.

This file imports ONLY from ``typing`` / stdlib. No engine code, no pydantic.
``plotsim.config`` re-exports every alias defined here so existing imports
of ``from plotsim.config import Dtype`` etc. continue to work.

TypeGuards
----------
For string-typed values that originate from user input and pass through
an ``if x in (literals): ...`` runtime check, ``cast(LiteralType, x)`` at
the consumer is unverified. A ``TypeGuard`` lets mypy narrow through the
predicate, so the cast disappears and the runtime check IS the proof:

    if not is_threshold_direction(s):
        raise ValueError(...)
    # mypy now sees s as Literal["above", "below"]

Add a new TypeGuard here whenever a parser narrows a string into one of
the engine's Literal aliases.
"""

from __future__ import annotations

from typing import Literal, TypeGuard

# ── Engine Literal aliases ──────────────────────────────────────────────────
#
# These are re-exported from ``plotsim.config`` for backwards compatibility.
# New code can import from either location; ``_types`` is preferred when
# the importing module wants to avoid the pydantic transitive import.

CurveType = Literal[
    "sigmoid",
    "exp_decay",
    "step",
    "logistic",
    "plateau",
    "oscillating",
    "compound",
    "sawtooth",
]

Distribution = Literal["lognorm", "gamma", "poisson", "beta", "normal", "weibull"]

Polarity = Literal["positive", "negative"]

TableType = Literal["dim", "fact", "event"]

Grain = Literal[
    "per_entity",
    "per_period",
    "per_reference",
    "per_entity_per_period",
    "variable",
]

Dtype = Literal["int", "float", "string", "date", "boolean", "id", "struct", "array"]

# Primitive element types valid inside a ``struct`` field map or as the
# ``array_element_type`` of an ``array`` column. Excludes nested types
# (no struct-of-struct or array-of-struct in V1) and engine-internal
# dtypes (``id``, ``date`` carry semantic constraints that don't compose
# inside a one-level nested cell).
NestedPrimitiveType = Literal["int", "float", "string", "boolean"]


def is_nested_primitive(s: str) -> TypeGuard[NestedPrimitiveType]:
    """True when ``s`` is a valid type word inside a struct/array column."""
    return s in ("int", "float", "string", "boolean")


Granularity = Literal["monthly", "weekly", "daily"]

# ``dim_date`` columns accept a strict subset of ``Dtype`` — boolean and id
# aren't meaningful for date-key columns. The interpreter dispatches on
# this subset; ``is_dim_date_dtype`` narrows to it.
DimDateDtype = Literal["int", "float", "string", "date"]

# Threshold-source direction is parsed out of a config string of the form
# ``threshold:<metric>:<above|below>:<value>:for:<consecutive>``.
ThresholdDirection = Literal["above", "below"]


# ── TypeGuards ──────────────────────────────────────────────────────────────


def is_dim_date_dtype(s: str) -> TypeGuard[DimDateDtype]:
    """True when ``s`` is one of the four dtype words valid on dim_date columns."""
    return s in ("int", "float", "string", "date")


def is_threshold_direction(s: str) -> TypeGuard[ThresholdDirection]:
    """True when ``s`` is a valid ``ThresholdSource.direction`` value."""
    return s in ("above", "below")
