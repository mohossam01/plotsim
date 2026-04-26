"""F13 regression — dead-schema audit (M102).

Mission 100 review found two instances of "schema field accepted at
load but unused at runtime" (``Entity.overrides`` permissive dict
in F9 / mission 100, ``StageDefinition.threshold_exit`` in F8 /
mission 100). The 0.2.0 cleanup closed the same class for
``Metric.default_curve``, ``MetricOverride.curve``,
``noise.temporal_jitter_days``, and ``noise.duplicate_rate``.

This test guards against the class re-emerging. Mechanically: for
every Pydantic field declared in ``plotsim/config.py``, scan
``plotsim/*.py`` for an attribute read of that field name. A field
with zero attribute reads must either be added to the documented
``ALLOWLIST`` below (with a reason) or removed from the schema.
A new field that escapes both gates fails the test.

Allowlist semantics: the heuristic only counts ``.field_name``
attribute reads. Fields read via dict-key indirection
(``obj.model_dump()`` then ``d.get("field_name")``), via
``model_json_schema()`` introspection, or as documentation-only
display fields surfaced through YAML round-trip are listed here
with the structural reason — these uses are real but the
attribute-read regex doesn't see them.

Audit performed in mission 102 / F13. The 8 entries below are the
full set of fields that didn't have an attribute read at the time
of the audit. Re-running the audit after schema changes is
expected; adding entries should be deliberate, not ambient.
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

from plotsim import config as plotsim_config


PLOTSIM_DIR = Path(__file__).resolve().parent.parent / "plotsim"


# (model_name, field_name) → reason allowed.
# Each entry must be either:
#   * a display / UI field surfaced via model_dump() / YAML round-trip,
#   * a schema-introspection field (e.g., pii_note read via JSON schema),
#   * a Literal-type constraint where the value IS the validation, or
#   * a field consumed via dict-key indirection in the engine
#     (the heuristic only catches attribute reads).
ALLOWLIST: dict[tuple[str, str], str] = {
    ("Archetype", "description"): (
        "display field surfaced via model_dump and YAML serialization"
    ),
    ("Archetype", "label"): (
        "display field surfaced via model_dump and YAML serialization"
    ),
    ("Column", "pii_note"): (
        "0.3.0 field-level PII metadata, surfaced via "
        "PlotsimConfig.model_json_schema() for downstream catalogs"
    ),
    ("Domain", "description"): (
        "display field surfaced via model_dump and YAML serialization"
    ),
    ("Domain", "entity_type"): (
        "domain metadata for downstream UI/scaffolding "
        "(ed. companion to entity_label, which IS read in cli.py)"
    ),
    ("EntityOverrides", "inflection_month"): (
        "consumed via overrides.model_dump() then dict.get() in "
        "trajectory.compute_all_trajectories — F9-introduced "
        "indirection that the attribute-read heuristic doesn't see"
    ),
    ("Metric", "label"): (
        "display field surfaced via model_dump and YAML serialization"
    ),
}


def _collect_pydantic_fields() -> list[tuple[str, str]]:
    """Enumerate every Pydantic field on every PlotsimConfig-related model
    declared in plotsim.config. Returns a sorted list of
    (model_name, field_name) pairs."""
    out: list[tuple[str, str]] = []
    for name, obj in vars(plotsim_config).items():
        if (
            inspect.isclass(obj)
            and hasattr(obj, "model_fields")
            and obj.__module__ == "plotsim.config"
            and name not in {"BaseModel", "_Frozen"}
        ):
            for field_name in obj.model_fields:
                out.append((name, field_name))
    return sorted(out)


def _engine_source() -> str:
    """All plotsim/*.py source concatenated. Excludes config.py-only
    declarations are still in scope — validator and property bodies
    in config.py legitimately consume their own model's fields."""
    return "\n".join(
        p.read_text(encoding="utf-8")
        for p in PLOTSIM_DIR.glob("*.py")
        if p.name != "__init__.py"
    )


def _has_attribute_read(field_name: str, source: str) -> bool:
    """True if any ``.field_name`` token appears in source. Heuristic —
    misses dict-key indirection and ``model_dump()`` consumers; those
    structural uses go on the ALLOWLIST."""
    return bool(re.search(rf"\.{re.escape(field_name)}\b", source))


def test_every_field_is_read_or_allowlisted():
    """Every Pydantic field on every plotsim.config model must either
    have an attribute read in plotsim/*.py or appear in ALLOWLIST.

    Adding a new schema field requires either using it in the engine
    or adding it to ALLOWLIST with a documented reason. A field
    that escapes both gates is dead schema and should be removed.
    """
    source = _engine_source()
    fields = _collect_pydantic_fields()
    unread_unlisted: list[str] = []
    for model_name, field_name in fields:
        if _has_attribute_read(field_name, source):
            continue
        if (model_name, field_name) in ALLOWLIST:
            continue
        unread_unlisted.append(f"{model_name}.{field_name}")
    assert not unread_unlisted, (
        f"Dead schema candidates (no attribute read in plotsim/, not on "
        f"ALLOWLIST): {unread_unlisted}. Either use the field in the "
        f"engine, add it to ALLOWLIST with a reason, or remove it from "
        f"the schema."
    )


def test_allowlist_entries_still_apply():
    """Every ALLOWLIST entry must reference a real
    (model, field) pair currently in the schema. Catches stale
    allowlist entries after refactors that rename or remove fields."""
    fields_set = set(_collect_pydantic_fields())
    stale = [
        (model, field)
        for (model, field) in ALLOWLIST
        if (model, field) not in fields_set
    ]
    assert not stale, (
        f"ALLOWLIST has entries that no longer exist on any model: "
        f"{stale}. Remove the stale entries from ALLOWLIST."
    )


def test_allowlist_entries_are_actually_unread():
    """An ALLOWLIST entry whose field DOES have an attribute read is
    a tautology — it should be removed so the audit's signal stays
    sharp. Prevents the allowlist from accumulating stale exemptions
    over time."""
    source = _engine_source()
    spurious = [
        (model, field)
        for (model, field) in ALLOWLIST
        if _has_attribute_read(field, source)
    ]
    assert not spurious, (
        f"ALLOWLIST entries that ARE read (and thus don't need an "
        f"exemption): {spurious}. Remove these from ALLOWLIST."
    )


@pytest.mark.parametrize("entry", sorted(ALLOWLIST.items()))
def test_allowlist_entry_has_documented_reason(entry):
    """Each ALLOWLIST entry's reason must be a non-empty string.
    Prevents drive-by exemptions with no documented justification."""
    (model_name, field_name), reason = entry
    assert reason and reason.strip(), (
        f"ALLOWLIST entry {model_name}.{field_name} has an empty reason"
    )
