"""plotsim._column_dispatch — shared source-type dispatcher for column builders.

The per-row column-resolution logic lives in five builder sites:

  * ``plotsim.tables._vectorized_per_entity_per_period_fact``
  * ``plotsim.tables._resolve_fact_cell`` (scalar fact cell)
  * ``plotsim.tables._build_per_period_fact``
  * ``plotsim.tables._build_proportional_event``
  * ``plotsim.tables._resolve_event_row``
  * ``plotsim.dimensions._column_value_for_entity`` (per_entity dim)
  * ``plotsim.dimensions.build_dim_subentity``
  * ``plotsim.dimensions.build_dim_reference``

Each site owns an ``isinstance(parsed, ...)`` ladder over the same set of
source types (``PKSource``, ``FKSource``, ``MetricSource``, ``LagSource``,
``GeneratedSource``, ``FakerSource``, ``StaticSource``, ``DerivedSource``,
``ThresholdSource``, ``TextBucketSource``, ``PoolSource``,
``SCDType2Source``). Each site supports a different *subset* of the types,
because the cell shape and available context differ (per_entity dim
columns can't read a metric series, per_period fact cells have no entity
axis to pin, etc.).

This module provides one ``ColumnDispatcher`` class plus the per-site
``BuilderKind`` enum so each ladder consumes the same registry rather than
hand-coding ``isinstance`` chains. Adding a new source type means:

  1. Register a resolver function for each site that supports the new type.
  2. Sites that do NOT support the new type get a single "raise" registration
     (kept verbose so the failure mode is loud, not silent).

**Critical contract** — ``FakerSource`` columns always route through the
scalar (per-row) path, regardless of whether the surrounding builder has a
vectorized branch. Vectorizing ``faker.method()`` would advance Faker's
internal state in a different order than the scalar fallback, breaking the
RNG-consumption-order parity the Layer 4 reference fixtures depend on.
The dispatcher's ``forces_scalar(parsed)`` predicate names this contract
explicitly so callers don't have to remember the rule.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class BuilderKind(Enum):
    """One enum value per builder site that consumes the dispatcher."""

    PER_ENTITY_PER_PERIOD_FACT_SCALAR = "per_entity_per_period_fact_scalar"
    PER_ENTITY_PER_PERIOD_FACT_VECTORIZED = "per_entity_per_period_fact_vectorized"
    PER_PERIOD_FACT = "per_period_fact"
    PROPORTIONAL_EVENT = "proportional_event"
    THRESHOLD_EVENT_ROW = "threshold_event_row"
    PER_ENTITY_DIM = "per_entity_dim"
    SUB_ENTITY_DIM = "sub_entity_dim"
    REFERENCE_DIM = "reference_dim"


# Canonical handler signature: ``(parsed_source, context) -> value_or_array``.
# ``context`` is a builder-specific dict so each site can pass in only what
# its supported source types need; the dispatcher itself is type-erased.
Handler = Callable[[Any, dict], Any]


@dataclass(frozen=True)
class ColumnDispatcher:
    """Registry of (BuilderKind, source_type) → handler.

    ``handlers`` is keyed by a tuple of ``(BuilderKind, source_class)`` so
    the same source class can dispatch differently per site. Lookup falls
    back to the registered "unsupported" handler when no entry exists, so
    the failure surface stays loud (raises with a config-pointing message)
    rather than silently emitting ``None``.
    """

    handlers: dict[tuple[BuilderKind, type], Handler] = field(default_factory=dict)
    unsupported: dict[BuilderKind, Handler] = field(default_factory=dict)

    def register(
        self,
        kind: BuilderKind,
        source_type: type,
        handler: Handler,
    ) -> None:
        """Register a resolver for one ``(BuilderKind, source class)`` pair."""
        self.handlers[(kind, source_type)] = handler

    def register_unsupported(
        self,
        kind: BuilderKind,
        handler: Handler,
    ) -> None:
        """Register the ``raise``-or-fallback handler for unhandled types."""
        self.unsupported[kind] = handler

    def dispatch(self, kind: BuilderKind, parsed: Any, context: dict) -> Any:
        """Resolve ``parsed`` against the ``kind`` site's registered handlers."""
        handler = self.handlers.get((kind, type(parsed)))
        if handler is not None:
            return handler(parsed, context)
        unsupported = self.unsupported.get(kind)
        if unsupported is not None:
            return unsupported(parsed, context)
        raise TypeError(
            f"ColumnDispatcher: no handler registered for {type(parsed).__name__} on {kind.value!r}"
        )

    def supports(self, kind: BuilderKind, source_type: type) -> bool:
        """Return True if a handler is registered for this (kind, type) pair."""
        return (kind, source_type) in self.handlers


def forces_scalar(parsed: Any, faker_source_type: type) -> bool:
    """True when ``parsed`` must run through the scalar/per-row builder path.

    ``FakerSource`` columns always force the scalar path so the per-row
    Faker state advances in the same order as the pre-vectorization
    builder. This protects byte-identical fixture parity for tables whose
    only stochastic columns are Faker-driven.

    Adding a new source type that consumes RNG or stateful generators at
    cell-build time should also force scalar — extend this predicate
    rather than threading the constraint through every call site.
    """
    return isinstance(parsed, faker_source_type)


# --- Module-level singleton ---------------------------------------------------
#
# Builders import this single dispatcher and register their per-site
# handlers at module-load time. Tests that want to introspect the registry
# read from this object directly.
COLUMN_DISPATCH = ColumnDispatcher()
