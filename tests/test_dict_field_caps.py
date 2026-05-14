"""Hardening: cap dict-/list-shaped config fields at load.

Five fields previously had no length cap and could pin or thrash a
config-load with adversarially large input:

  * ``FakerSource.kwargs`` — capped at 20 entries.
  * ``NarrativeConfig.template`` — capped at 1000 characters.
  * ``NarrativeConfig.lexicons[arch][slot][band]`` — capped at 100
    phrases per band cell.
  * ``Column.value_pool[entity_name]`` — capped at 1000 entries.
  * ``Column.nested_schema`` — capped at 20 fields.

One test per cap, exercising the boundary and the just-over-boundary
rejection message.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from plotsim.config import (
    Column,
    FakerSource,
    NarrativeConfig,
)


def test_faker_source_kwargs_cap_rejects_above_twenty():
    # 20 entries — accepted.
    ok = FakerSource(method="date_between", kwargs={f"k{i}": str(i) for i in range(20)})
    assert len(ok.kwargs) == 20

    # 21 entries — rejected.
    with pytest.raises(ValidationError, match=r"FakerSource\.kwargs has 21 entries"):
        FakerSource(method="date_between", kwargs={f"k{i}": str(i) for i in range(21)})


def test_narrative_template_length_capped_at_1000():
    placeholder = "{tone} {momentum}"
    pad = " " * (1000 - len(placeholder))
    NarrativeConfig(
        template=placeholder + pad,
        lexicons={
            "growth": {
                "tone": {"low": ["a"], "mid": ["b"], "high": ["c"]},
                "momentum": {"low": ["d"], "mid": ["e"], "high": ["f"]},
            }
        },
    )

    too_long = placeholder + (" " * (1001 - len(placeholder)))
    with pytest.raises(ValidationError, match=r"template"):
        NarrativeConfig(
            template=too_long,
            lexicons={
                "growth": {
                    "tone": {"low": ["a"], "mid": ["b"], "high": ["c"]},
                    "momentum": {"low": ["d"], "mid": ["e"], "high": ["f"]},
                }
            },
        )


def test_narrative_phrase_list_cap_rejects_above_100():
    # 100 phrases in one band cell — accepted.
    NarrativeConfig(
        template="{tone}",
        lexicons={
            "growth": {
                "tone": {
                    "low": [f"phrase_{i}" for i in range(100)],
                    "mid": ["b"],
                    "high": ["c"],
                },
            }
        },
    )

    with pytest.raises(
        ValidationError,
        match=r"lexicons\['growth'\]\['tone'\]\['low'\] has 101 phrases",
    ):
        NarrativeConfig(
            template="{tone}",
            lexicons={
                "growth": {
                    "tone": {
                        "low": [f"phrase_{i}" for i in range(101)],
                        "mid": ["b"],
                        "high": ["c"],
                    },
                }
            },
        )


def test_column_value_pool_cap_rejects_above_1000_per_entity():
    Column(
        name="industry",
        dtype="string",
        source="pool:industry",
        value_pool={"e1": [f"v{i}" for i in range(1000)]},
    )

    with pytest.raises(
        ValidationError,
        match=r"value_pool for entity 'e1' has 1001 entries",
    ):
        Column(
            name="industry",
            dtype="string",
            source="pool:industry",
            value_pool={"e1": [f"v{i}" for i in range(1001)]},
        )


def test_column_nested_schema_cap_rejects_above_20_fields():
    Column(
        name="profile",
        dtype="struct",
        source="nested",
        nested_schema={f"field_{i}": "string" for i in range(20)},
    )

    with pytest.raises(
        ValidationError,
        match=r"nested_schema has 21 fields",
    ):
        Column(
            name="profile",
            dtype="struct",
            source="nested",
            nested_schema={f"field_{i}": "string" for i in range(21)},
        )
