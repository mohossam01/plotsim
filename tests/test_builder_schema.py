"""Tests for plotsim.builder.schema — JSON Schema export + vocabulary enums.

Covers M116 acceptance criteria:

- ``generate_user_input_schema()`` returns a valid JSON Schema.
- Required-field set in the exported schema matches the model.
- Optional fields carry defaults consistent with the model.
- Vocabulary enums (METRIC_TYPES / SHAPE_WORDS / RELATIONSHIP_WORDS /
  BASELINE_WORDS) match their corresponding recipe dict keys exactly.
- COLUMN_TYPES covers every column-type stem the interpreter handles.
- Schema round-trip: saas_template.yaml validates against the schema.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from plotsim.builder import create_from_yaml
from plotsim.builder.input import UserInput
from plotsim.builder.recipes import (
    BASELINE_RECIPES,
    RELATIONSHIP_RECIPES,
    SHAPE_RECIPES,
    VALID_METRIC_TYPES,
)
from plotsim.builder.schema import (
    BASELINE_WORDS,
    COLUMN_TYPES,
    METRIC_TYPES,
    RELATIONSHIP_WORDS,
    SCHEMA_FILENAME,
    SHAPE_WORDS,
    generate_user_input_schema,
    write_user_input_schema,
)


SAAS_TEMPLATE = Path("plotsim/configs/new/saas_template.yaml")
BARE_TEMPLATE = Path("plotsim/configs/new/bare_minimum.yaml")


# ── Schema export shape ────────────────────────────────────────────────────


def test_schema_dialect_and_title():
    schema = generate_user_input_schema()
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["title"] == "UserInput"
    assert schema["type"] == "object"


def test_schema_required_fields_match_model():
    """Required set in JSON Schema must match the model's required fields.

    Required fields on UserInput are those without defaults: ``about``,
    ``unit``, ``window``, ``metrics``, ``segments``. Anything else is
    optional and has a default.
    """
    schema = generate_user_input_schema()
    expected = {"about", "unit", "window", "metrics", "segments"}
    assert set(schema["required"]) == expected


def test_schema_optional_fields_omittable_and_default_correctly():
    """Every optional field can be omitted from input and produces the documented default.

    Pydantic does not emit ``"default": []`` for ``default_factory=list``
    fields in JSON Schema (only ``Optional[...]`` with explicit
    ``default=None`` gets ``"default": null``). The semantic check that
    matters — that the model accepts omission and produces the documented
    defaults — is verified directly here.
    """
    import warnings as _warnings
    with _warnings.catch_warnings():
        # Single-segment input intentionally — silence the documented warning.
        _warnings.simplefilter("ignore", UserWarning)
        minimal = UserInput.model_validate({
            "about": "X", "unit": "x",
            "window": {"start": "2024-01", "end": "2024-12"},
            "metrics": [{"name": "m", "type": "score", "polarity": "positive"}],
            "segments": [{"name": "s", "count": 50, "archetype": "growth"}],
        })
    assert minimal.connections == []
    assert minimal.lifecycle is None
    assert minimal.dimensions == []
    assert minimal.facts == []
    assert minimal.events == []
    # Schema must mark only the truly required fields.
    schema = generate_user_input_schema()
    optional = {"connections", "lifecycle", "dimensions", "facts", "events"}
    assert optional.isdisjoint(set(schema["required"]))
    # ``lifecycle`` is the only Optional[...] explicit default — it gets
    # ``default: null`` inline. The list-default fields don't emit a default.
    assert schema["properties"]["lifecycle"]["default"] is None


def test_schema_metric_type_literal_emitted_as_enum():
    """MetricInput.type is Literal[...] → must surface as an enum in the schema."""
    schema = generate_user_input_schema()
    metric_def = schema["$defs"]["MetricInput"]
    assert sorted(metric_def["properties"]["type"]["enum"]) == sorted(
        METRIC_TYPES
    )


def test_schema_polarity_literal_emitted_as_enum():
    schema = generate_user_input_schema()
    metric_def = schema["$defs"]["MetricInput"]
    assert sorted(metric_def["properties"]["polarity"]["enum"]) == [
        "negative", "positive",
    ]


def test_schema_window_every_literal_emitted_as_enum():
    schema = generate_user_input_schema()
    window_def = schema["$defs"]["WindowInput"]
    assert sorted(window_def["properties"]["every"]["enum"]) == [
        "daily", "monthly", "weekly",
    ]


# ── Vocabulary enums ────────────────────────────────────────────────────────


def test_metric_types_match_validator_set():
    assert set(METRIC_TYPES) == set(VALID_METRIC_TYPES)


def test_shape_words_match_recipe_keys():
    assert set(SHAPE_WORDS) == set(SHAPE_RECIPES)


def test_relationship_words_match_recipe_keys_and_values():
    assert set(RELATIONSHIP_WORDS) == set(RELATIONSHIP_RECIPES)
    # Values must round-trip exactly — drift here changes documented
    # correlation coefficients without anyone noticing.
    for word, coef in RELATIONSHIP_RECIPES.items():
        assert RELATIONSHIP_WORDS[word] == coef


def test_baseline_words_match_recipe_keys():
    assert set(BASELINE_WORDS) == set(BASELINE_RECIPES)


def test_column_types_cover_every_interpreter_stem():
    """COLUMN_TYPES must list every column-type stem the interpreter handles.

    The interpreter's _translate_column dispatches on these prefixes /
    literals; if a new one is added there, COLUMN_TYPES must learn it.
    """
    expected = {
        "id", "ref.{dim}", "metric.{name}", "faker.{kind}", "static.{value}",
        "segment.count", "timestamp", "flag", "bucket", "scd",
        "date", "int", "string", "float",
    }
    assert set(COLUMN_TYPES) == expected


# ── Round-trip: real templates must validate against the exported schema ──


def test_saas_template_validates_against_schema():
    """Schema round-trip: model_validate(saas) → model_dump → jsonschema.validate.

    The YAML template uses shorthand forms (window as date scalars,
    lifecycle stages as ``{name: threshold}`` single-key dicts,
    connections as 3-token strings). UserInput's pre-validators coerce
    these to canonical form; the JSON Schema documents the canonical
    form. Round-trip = "what the model accepts, when re-emitted, must
    pass schema validation." That is what tooling needs.
    """
    jsonschema = pytest.importorskip("jsonschema")
    schema = generate_user_input_schema()
    raw = yaml.safe_load(SAAS_TEMPLATE.read_text(encoding="utf-8"))
    if isinstance(raw["window"], dict):
        for k in ("start", "end"):
            v = raw["window"].get(k)
            if v is not None and not isinstance(v, str):
                raw["window"][k] = str(v)
    ui = UserInput.model_validate(raw)
    canonical = ui.model_dump(mode="json", by_alias=True)
    jsonschema.validate(instance=canonical, schema=schema)


def test_bare_minimum_template_validates_against_schema():
    jsonschema = pytest.importorskip("jsonschema")
    schema = generate_user_input_schema()
    raw = yaml.safe_load(BARE_TEMPLATE.read_text(encoding="utf-8"))
    if isinstance(raw["window"], dict):
        for k in ("start", "end"):
            v = raw["window"].get(k)
            if v is not None and not isinstance(v, str):
                raw["window"][k] = str(v)
    jsonschema.validate(instance=raw, schema=schema)


def test_bare_minimum_loads_via_create_from_yaml():
    """bare_minimum.yaml must build a valid PlotsimConfig (no warnings)."""
    import warnings as _warnings
    with _warnings.catch_warnings():
        _warnings.simplefilter("error", UserWarning)
        cfg = create_from_yaml(str(BARE_TEMPLATE))
    assert cfg.domain.entity_type == "customer"
    assert {m.name for m in cfg.metrics} == {"engagement", "payments"}
    # Auto-generated schema: dim_date + dim_<unit> + fct_<unit>.
    assert {t.name for t in cfg.tables} == {
        "dim_date", "dim_customer", "fct_customer",
    }


# ── Round-trip via UserInput.model_validate (no extra dep) ─────────────────


def test_saas_template_model_validates_after_str_coercion():
    """UserInput.model_validate accepts the saas template post-coercion.

    Catches drift where the schema diverges from the model: if the schema
    says a field is required but the model accepts it as optional (or
    vice versa), this test won't catch it — but the
    test_schema_required_fields_match_model test will.
    """
    raw = yaml.safe_load(SAAS_TEMPLATE.read_text(encoding="utf-8"))
    if isinstance(raw["window"], dict):
        for k in ("start", "end"):
            v = raw["window"].get(k)
            if v is not None and not isinstance(v, str):
                raw["window"][k] = str(v)
    UserInput.model_validate(raw)


# ── write helper ────────────────────────────────────────────────────────────


def test_write_user_input_schema(tmp_path):
    out = write_user_input_schema(tmp_path / "schema.json")
    assert out.exists()
    payload = out.read_text(encoding="utf-8")
    assert payload.endswith("\n")
    # Pretty-printed.
    assert '\n  "' in payload


def test_schema_filename_constant():
    assert SCHEMA_FILENAME == "plotsim-user-input-schema.json"
