"""M118 — bridge cardinality validator row-count fix.

Regression cover for ``audit-report.md`` §3.1: the bridge cardinality gate
in ``PlotsimConfig._cross_reference_integrity`` previously sized a
``per_entity`` dim as ``sum(e.size for e in entities)`` instead of
``len(entities)``. ``Entity.size`` is a cohort-population value carried
as a ``derived:size`` metadata column on the dim — it is **not** a row
multiplier. ``build_dim_entity`` emits one row per ``Entity`` regardless
of ``size``.

The bug was masked in the builder path (M117 expansion always sets
``size=1``, so ``sum(e.size) == len(entities)``) but engine-direct
configs with ``Entity.size > 1`` and a bridge whose second dim is
``per_entity`` could declare a ``cardinality.max`` that exceeded the
dim's actual row count and the validator would silently accept it.
"""

from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from plotsim.config import PlotsimConfig

from tests.test_bridge_tables import _base_education_dict


def _config_with_per_entity_second_dim(card_max: int) -> dict:
    """Education template + a second ``per_entity`` dim wired into the bridge.

    The bundled education template's bridge points at ``dim_course``
    (``per_reference``), so it doesn't exercise the ``per_entity_dim_table_count``
    code path. This helper adds ``dim_advisor`` (``per_entity``) and
    re-targets the bridge at it so the second-dim row-count check fires.

    Entity sizes stay at the template defaults (60 / 20 / 15), giving
    ``len(entities)=3`` and ``sum(e.size)=95``. Pick ``card_max`` between
    those two to exercise the bug; pick ``card_max <= 3`` for the
    accept path.
    """
    cfg = copy.deepcopy(_base_education_dict())
    cfg["tables"].append(
        {
            "name": "dim_advisor",
            "type": "dim",
            "grain": "per_entity",
            "columns": [
                {"name": "advisor_id", "dtype": "id", "source": "pk"},
                {"name": "advisor_name", "dtype": "string", "source": "generated:faker.name"},
            ],
            "primary_key": "advisor_id",
        }
    )
    cfg["bridges"][0]["connects"] = ["dim_student", "dim_advisor"]
    cfg["bridges"][0]["cardinality"] = {"min": 0, "max": card_max}
    cfg["bridges"][0]["metrics"] = [
        {"name": "grade", "dtype": "float", "source": "metric:assignment_score"},
    ]
    return cfg


def test_bridge_cardinality_rejects_max_above_len_entities_engine_direct():
    """Engine-direct config: 3 entities, sizes 60/20/15 → dim_advisor has 3 rows.

    ``cardinality.max=4`` exceeds the actual row count (3) but is below
    ``sum(e.size)=95``. Pre-M118 the validator computed the second-dim
    count as ``sum(e.size)`` and silently accepted; post-fix it must
    reject.
    """
    cfg = _config_with_per_entity_second_dim(card_max=4)
    with pytest.raises(ValidationError, match=r"cardinality\.max.*exceeds"):
        PlotsimConfig(**cfg)


def test_bridge_cardinality_accepts_max_at_len_entities_engine_direct():
    """Same shape, but ``cardinality.max=3`` matches the actual row count.

    Validator must accept regardless of ``sum(e.size)``.
    """
    cfg = _config_with_per_entity_second_dim(card_max=3)
    PlotsimConfig(**cfg)


def test_bridge_cardinality_uses_len_entities_not_sum_sizes():
    """Direct check that the second-dim count is ``len(entities)``.

    With three entities of sizes 60/20/15, a ``cardinality.max=10``
    must be rejected: it is well below ``sum(e.size)=95`` (which the
    pre-M118 code would have accepted) but above the true dim row
    count of 3.
    """
    cfg = _config_with_per_entity_second_dim(card_max=10)
    with pytest.raises(ValidationError) as exc_info:
        PlotsimConfig(**cfg)
    msg = str(exc_info.value)
    assert "cardinality.max" in msg
    assert "(10)" in msg
    assert "(3)" in msg
