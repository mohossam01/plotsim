"""plotsim.schema — JSON Schema export for the PlotsimConfig model.

Pydantic v2 emits a JSON Schema directly from the root model via
``model_json_schema()``; this module is a thin wrapper that pins the
output format (Draft 2020-12, pretty-printed with stable key order) and
offers a one-line write helper. Editor integrations (VSCode, JetBrains)
point at the produced ``plotsim-schema.json`` for autocomplete and
inline validation on ``sample_*.yaml`` configs.

Pure metadata module — does not import the generation engine. Calling
``plotsim schema`` from the CLI executes nothing beyond Pydantic's
schema introspection.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from plotsim.config import PlotsimConfig


SCHEMA_FILENAME = "plotsim-schema.json"


def generate_schema() -> dict[str, Any]:
    """Return the JSON Schema dict for ``PlotsimConfig``.

    Pydantic v2 emits Draft 2020-12 by default; the dialect identifier is
    set explicitly on the returned dict so the file declares it for
    consumers (jsonschema validators, IDEs).
    """
    schema = PlotsimConfig.model_json_schema()
    schema.setdefault("$schema", "https://json-schema.org/draft/2020-12/schema")
    schema.setdefault("title", "PlotsimConfig")
    return schema


def write_schema(path: str | Path) -> Path:
    """Write the generated JSON Schema to ``path`` as pretty-printed JSON.

    Output is deterministic: ``indent=2``, ``sort_keys=False`` (Pydantic's
    insertion order is already stable across runs for a given Python /
    Pydantic version), trailing newline. Same plotsim version produces
    byte-identical output.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(generate_schema(), indent=2, ensure_ascii=False) + "\n"
    target.write_text(payload, encoding="utf-8")
    return target
