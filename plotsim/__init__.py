"""plotsim — generate realistic multi-table datasets from behavioral archetypes.

Quick start:
    from plotsim import create, generate_tables, write_tables

    cfg = create(
        about="Subscription customers",
        unit="customer",
        window=("2024-01", "2024-12", "monthly"),
        metrics=[
            {"name": "engagement", "type": "score", "polarity": "positive"},
            {"name": "payments",   "type": "count", "polarity": "positive"},
        ],
        segments=[
            {"name": "active",   "count": 50, "archetype": "growth"},
            {"name": "inactive", "count": 30, "archetype": "decline"},
        ],
    )
    tables = generate_tables(cfg)
    write_tables(tables, cfg)

The CLI offers the same flow against a YAML file:

    plotsim template saas -o config.yaml
    plotsim run config.yaml -o ./output
"""

__version__ = "0.5.0"

import importlib.resources as _resources
from pathlib import Path as _Path

from plotsim import inspect
from plotsim.builder import create, create_from_yaml
from plotsim.config import (
    DIRTY,
    NOISE_PRESETS,
    PERFECTLY_CLEAN,
    REALISTIC,
    SLIGHTLY_MESSY,
    ManifestConfig,
    PlotsimConfig,
    PoolSource,
    SurrogateKeyWarning,
    TextBucketSource,
    dump_config,
    load_config,
)
from plotsim.manifest import (
    EntityArchetypeAssignment,
    EventFiring,
    ManifestSchema,
    TrajectorySample,
    build_manifest,
    write_manifest,
)
from plotsim.output import (
    write_config_copy,
    write_single_table,
    write_tables,
    write_validation_report,
)
from plotsim.tables import GenerationState, generate_tables, generate_tables_with_state
from plotsim.validation import (
    ValidationIssue,
    ValidationReport,
    validate_tables as validate,
)
from plotsim.validation import validate_tables

_TEMPLATE_PACKAGE = "plotsim.configs.templates"


def list_templates() -> list[str]:
    """Return bundled builder template names, sorted alphabetically.

    Names round-trip through ``load_template(name)``: ``bare_minimum``
    keeps its full stem; the domain templates strip the ``_template``
    suffix (so ``saas_template.yaml`` becomes ``saas``).
    """
    out: list[str] = []
    for entry in _resources.files(_TEMPLATE_PACKAGE).iterdir():
        name = entry.name
        if not name.endswith(".yaml"):
            continue
        stem = name[:-5]
        if stem.endswith("_template"):
            stem = stem[: -len("_template")]
        out.append(stem)
    out.sort()
    return out


def load_template(name: str) -> PlotsimConfig:
    """Load a bundled template by name and return a ``PlotsimConfig``.

    ``name`` is one of ``list_templates()``. Equivalent to calling
    ``create_from_yaml`` on the template's bundled path. Raises
    ``ValueError`` if the name is not a bundled template.
    """
    root = _resources.files(_TEMPLATE_PACKAGE)
    for candidate in (f"{name}.yaml", f"{name}_template.yaml"):
        entry = root / candidate
        if entry.is_file():
            return create_from_yaml(_Path(str(entry)))
    raise ValueError(
        f"Unknown template {name!r}. Available: {list_templates()}"
    )


__all__ = [
    # Builder (one-call public API; M115/M122)
    "create",
    "create_from_yaml",
    "list_templates",
    "load_template",
    # Generation + validation
    "generate_tables",
    "validate",
    # Output
    "write_tables",
    "write_single_table",
    "write_config_copy",
    "write_validation_report",
    # Config (advanced / engine-direct)
    "PlotsimConfig",
    "SurrogateKeyWarning",
    "ManifestConfig",
    "TextBucketSource",
    "PoolSource",
    "load_config",
    "dump_config",
    "NOISE_PRESETS",
    "PERFECTLY_CLEAN",
    "SLIGHTLY_MESSY",
    "REALISTIC",
    "DIRTY",
    # Generation extras
    "generate_tables_with_state",
    "GenerationState",
    # Manifest
    "ManifestSchema",
    "EntityArchetypeAssignment",
    "TrajectorySample",
    "EventFiring",
    "build_manifest",
    "write_manifest",
    # Validation extras
    "validate_tables",
    "ValidationReport",
    "ValidationIssue",
    # Introspection
    "inspect",
    # Version
    "__version__",
]
