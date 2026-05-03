"""plotsim builder — translate plain-language input into a PlotsimConfig.

Public entry points:

    create(**kwargs) -> PlotsimConfig
        Build a PlotsimConfig from keyword arguments matching the input
        template at ``plotsim/configs/templates/saas_template.py``.

    create_from_yaml(path) -> PlotsimConfig
        Load a YAML file conforming to ``plotsim/configs/templates/saas_template.yaml``
        and build a PlotsimConfig from it.

Both surfaces share the same input model (``UserInput``) and the same
interpreter (``interpret``). The structural validation surface, vocabulary
recipes, and DSL parser are exposed at module scope for downstream tools
(linters, schema generators) that need to introspect the contract without
constructing a config.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from plotsim.config import PlotsimConfig

from plotsim.builder.input import UserInput
from plotsim.builder.interpreter import interpret
from plotsim.builder.parser import ArchetypeParseError, parse_archetype
from plotsim.builder.recipes import (
    BASELINE_RECIPES,
    METRIC_RECIPES,
    RELATIONSHIP_RECIPES,
    SHAPE_RECIPES,
)


def create(**kwargs: Any) -> PlotsimConfig:
    """Build a PlotsimConfig from keyword arguments.

    The arguments mirror the YAML template shape — see
    ``plotsim/configs/templates/saas_template.py`` for the canonical example.
    Construction validates the user-facing input model first; structural
    problems raise ``pydantic.ValidationError`` with the specific field
    named, before the interpreter runs.
    """
    user_input = UserInput.model_validate(kwargs)
    return interpret(user_input)


def create_from_yaml(path: str | Path) -> PlotsimConfig:
    """Load a YAML config file and build a PlotsimConfig.

    YAML's relaxed scalar parsing turns ``2023-01`` into a ``datetime.date``
    object; we coerce window fields back to strings so they thread through
    the same ``UserInput`` validators that ``create()`` uses.
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(
            f"YAML at {path!r} must be a mapping at the top level, got "
            f"{type(raw).__name__}"
        )
    if "window" in raw and isinstance(raw["window"], dict):
        for k in ("start", "end"):
            v = raw["window"].get(k)
            if v is not None and not isinstance(v, str):
                raw["window"][k] = str(v)
    return create(**raw)


__all__ = [
    "create",
    "create_from_yaml",
    # Validation / parsing surfaces (re-exported for downstream tooling)
    "UserInput",
    "ArchetypeParseError",
    "parse_archetype",
    "BASELINE_RECIPES",
    "METRIC_RECIPES",
    "RELATIONSHIP_RECIPES",
    "SHAPE_RECIPES",
    "interpret",
]
