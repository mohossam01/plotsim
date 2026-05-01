"""plotsim.builder.schema — JSON Schema export for the builder UserInput model.

Mirrors ``plotsim.schema`` (which exports ``PlotsimConfig``). Editor
integrations and UI tooling point at the produced schema for autocomplete
and inline validation on plain-language YAML configs (see
``plotsim/configs/new/saas_template.yaml`` for the canonical shape).

Also exports the five vocabulary lookup dicts so downstream consumers
(UI dropdowns, lint rules, prompt scaffolding) can introspect the
contract without re-deriving the constants from the validator code.
The dict keys mirror the recipe dicts in ``plotsim.builder.recipes`` and
the ``Literal`` enums on ``plotsim.builder.input.UserInput``; the values
are short human-readable hints suitable for tooltips. ``COLUMN_TYPES``
covers the column-type vocabulary handled by the interpreter — there is
no recipe dict for it, so this module is the single declared source.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from plotsim.builder.input import UserInput


SCHEMA_FILENAME = "plotsim-user-input-schema.json"


def generate_user_input_schema() -> dict[str, Any]:
    """Return the JSON Schema dict for ``UserInput``.

    Pydantic v2 emits Draft 2020-12 by default; the dialect identifier is
    set explicitly on the returned dict so the file declares it for
    consumers (jsonschema validators, IDEs).
    """
    schema = UserInput.model_json_schema()
    schema.setdefault("$schema", "https://json-schema.org/draft/2020-12/schema")
    schema.setdefault("title", "UserInput")
    return schema


def write_user_input_schema(path: str | Path) -> Path:
    """Write the JSON Schema to ``path`` as pretty-printed JSON.

    Output is deterministic: ``indent=2``, ``ensure_ascii=False``, trailing
    newline. Same plotsim version produces byte-identical output.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        generate_user_input_schema(), indent=2, ensure_ascii=False
    ) + "\n"
    target.write_text(payload, encoding="utf-8")
    return target


# ── Vocabulary enums (plain dicts; values are short human-readable hints) ──


METRIC_TYPES: dict[str, str] = {
    "score":  "Bounded [0, 1] score (engagement, adoption, risk).",
    "amount": "Currency or bounded quantity (requires `range`).",
    "count":  "Non-negative integer event count (no `range`; poisson).",
    "index":  "Signed centered metric in a declared range (requires `range`).",
}

SHAPE_WORDS: dict[str, str] = {
    "growth":           "Smooth S-curve rise.",
    "decline":          "Exponential fade.",
    "seasonal":         "Two oscillation cycles across the window.",
    "flat":             "Low constant plateau.",
    "spike_then_crash": "Rapid rise, sharp drop, low plateau.",
    "accelerating":     "Compound growth (base + acceleration).",
}

RELATIONSHIP_WORDS: dict[str, float] = {
    "mirrors":        0.75,
    "driven_by":      0.55,
    "related":        0.40,
    "hints_at":       0.20,
    "independent":    0.00,
    "hints_against": -0.20,
    "resists":       -0.40,
    "opposes":       -0.55,
    "inverts":       -0.75,
}

BASELINE_WORDS: dict[str, str] = {
    "high": "Restrict the segment's value range to the upper third.",
    "mid":  "Restrict to the middle third (the default if omitted).",
    "low":  "Restrict to the lower third.",
}

COLUMN_TYPES: dict[str, str] = {
    "id":             "Primary key column, auto-generated.",
    "ref.{dim}":      "Foreign key to the named dimension table.",
    "metric.{name}":  "Populated from the named declared metric.",
    "faker.{kind}":   "Generated text via faker (company, name, sentence, year, ...).",
    "static.{value}": "Fixed literal value for every row (numeric → float, else string).",
    "segment.count":  "Engine fills with the segment's row count.",
    "pool.{attr}":    "Per-entity value pool drawn from the segment's `attributes[attr]` list.",
    "timestamp":      "Datetime sampled within each row's period.",
    "flag":           "Boolean derived from a threshold-trigger event firing.",
    "bucket":         "Categorical label derived from trajectory (requires `labels`).",
    "scd":            "Slowly-changing dim band (requires `tracks`, `tiers`, `at`).",
    "date":           "`date` dtype on `dim_date` columns only.",
    "int":            "`int` dtype on `dim_date` columns only.",
    "string":         "`string` dtype on `dim_date` columns only.",
    "float":          "`float` dtype on `dim_date` columns only.",
}


__all__ = [
    "SCHEMA_FILENAME",
    "generate_user_input_schema",
    "write_user_input_schema",
    "METRIC_TYPES",
    "SHAPE_WORDS",
    "RELATIONSHIP_WORDS",
    "BASELINE_WORDS",
    "COLUMN_TYPES",
]
