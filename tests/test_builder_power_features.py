"""M122 — bridges, quality injection, holdout, entity_features.

Covers the AC subsets in
``project/missions/mission-122-builder-feature-parity.md``:

  * ``quality`` list translates to ``QualityConfig.quality_issues`` (all
    five issue types; column omitted on duplicate_rows / late_arrival).
  * ``holdout`` dict translates to ``HoldoutConfig(enabled=True)``;
    invalid period count caught by engine load.
  * ``entity_features: true`` shorthand and dict form translate to
    ``EntityFeaturesConfig(enabled=True)``.
  * ``bridges`` list translates to ``BridgeTableConfig`` entries with
    correct cardinality, FK references, driver validation, and column
    translation.
  * Omitting any feature → identical to pre-M122 default config (engine
    sees ``enabled=false`` / empty list).
"""

from __future__ import annotations

import warnings
from typing import Any

import pytest

from plotsim.builder import create
from plotsim.builder.input import UserInput


def _explicit_input(**overrides: Any) -> dict[str, Any]:
    """Return a dict suitable for ``create(**)`` with an explicit schema.

    The explicit schema is needed for bridges (auto-schema only emits
    ``dim_date`` and ``dim_{unit}``; bridges between those would hit
    the engine's per_period rejection).
    """
    base: dict[str, Any] = {
        "about": "M122 power features",
        "unit": "company",
        "window": {"start": "2024-01", "end": "2024-12"},
        "metrics": [
            {"name": "engagement", "type": "score", "polarity": "positive"},
            {"name": "mrr", "type": "amount", "polarity": "positive", "range": [100, 50000]},
        ],
        "segments": [
            {"name": "alpha", "count": 5, "archetype": "growth"},
            {"name": "beta", "count": 5, "archetype": "flat"},
        ],
        "dimensions": [
            {
                "name": "dim_date",
                "per": "period",
                "columns": [
                    {"name": "date_key", "type": "id"},
                    {"name": "date", "type": "date"},
                ],
            },
            {
                "name": "dim_company",
                "per": "unit",
                "columns": [
                    {"name": "company_id", "type": "id"},
                    {"name": "company_name", "type": "faker.company"},
                ],
            },
            {
                "name": "dim_user",
                "per": "unit",
                "columns": [
                    {"name": "user_id", "type": "id"},
                    {"name": "company_id", "type": "ref.dim_company"},
                    {"name": "user_name", "type": "faker.name"},
                ],
            },
        ],
        "facts": [
            {
                "name": "fct_company",
                "metrics": ["engagement", "mrr"],
                "columns": [
                    {"name": "date_key", "type": "ref.dim_date"},
                    {"name": "company_id", "type": "ref.dim_company"},
                    {"name": "engagement", "type": "metric.engagement"},
                    {"name": "mrr", "type": "metric.mrr"},
                ],
            },
        ],
    }
    base.update(overrides)
    return base


def _create(**overrides: Any):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return create(**_explicit_input(**overrides))


# ── Quality injection ──────────────────────────────────────────────────────


def test_quality_list_populates_quality_issues():
    cfg = _create(
        quality=[
            {"table": "fct_company", "column": "mrr", "issue": "null_injection", "rate": 0.05},
            {"table": "fct_company", "issue": "duplicate_rows", "rate": 0.02},
        ]
    )
    issues = cfg.quality.quality_issues
    assert len(issues) == 2
    assert issues[0].type == "null_injection"
    assert issues[0].target_table == "fct_company"
    assert issues[0].target_columns == ["mrr"]
    assert issues[0].rate == 0.05
    assert issues[1].type == "duplicate_rows"
    # column omitted → engine "*" sentinel = "every eligible column"
    assert issues[1].target_columns == ["*"]


@pytest.mark.parametrize(
    "issue_type,needs_column",
    [
        ("null_injection", True),
        ("duplicate_rows", False),
        ("type_mismatch", True),
        ("late_arrival", False),
        ("schema_drift", True),
    ],
)
def test_all_quality_issue_types_translate(issue_type, needs_column):
    spec: dict[str, Any] = {
        "table": "fct_company",
        "issue": issue_type,
        "rate": 0.05,
    }
    if needs_column:
        spec["column"] = "mrr"
    cfg = _create(quality=[spec])
    assert cfg.quality.quality_issues[0].type == issue_type


def test_quality_omitted_yields_empty_default():
    cfg = _create()
    assert cfg.quality.quality_issues == []


def test_quality_invalid_issue_type_raises():
    with pytest.raises(ValueError):
        _create(
            quality=[
                {"table": "fct_company", "column": "mrr", "issue": "not_a_thing", "rate": 0.05},
            ]
        )


def test_quality_null_injection_without_column_rejected():
    """``null_injection`` / ``type_mismatch`` / ``schema_drift`` need an
    explicit column — the corruption is column-level. Reject at builder
    layer with a clear message rather than passing through to the engine.
    """
    with pytest.raises(ValueError, match="requires a `column` name"):
        _create(
            quality=[
                {"table": "fct_company", "issue": "null_injection", "rate": 0.05},
            ]
        )


# ── Holdout ────────────────────────────────────────────────────────────────


def test_holdout_dict_translates_to_enabled_config():
    cfg = _create(holdout={"target": "mrr", "periods": 3})
    assert cfg.holdout.enabled is True
    assert cfg.holdout.target_metric == "mrr"
    assert cfg.holdout.holdout_periods == 3


def test_holdout_omitted_yields_disabled_default():
    cfg = _create()
    assert cfg.holdout.enabled is False
    assert cfg.holdout.target_metric is None


def test_holdout_periods_exceeding_window_rejected():
    """Window is 12 monthly periods. Holdout=11 leaves 1 training period
    < ``min_training_periods=3``; engine load should reject.
    """
    with pytest.raises(ValueError):
        _create(holdout={"target": "mrr", "periods": 11})


# ── Entity features ────────────────────────────────────────────────────────


def test_entity_features_true_enables_with_defaults():
    cfg = _create(entity_features=True)
    assert cfg.entity_features.enabled is True
    assert cfg.entity_features.metrics == []  # all metrics
    assert cfg.entity_features.include_labels is True


def test_entity_features_dict_form_with_metric_filter():
    cfg = _create(
        entity_features={
            "metrics": ["mrr"],
            "include_labels": False,
        }
    )
    assert cfg.entity_features.enabled is True
    assert cfg.entity_features.metrics == ["mrr"]
    assert cfg.entity_features.include_labels is False


def test_entity_features_omitted_yields_disabled_default():
    cfg = _create()
    assert cfg.entity_features.enabled is False


def test_entity_features_false_yields_disabled_default():
    cfg = _create(entity_features=False)
    assert cfg.entity_features.enabled is False


def test_entity_features_unknown_metric_rejected():
    with pytest.raises(ValueError, match="not a declared metric"):
        _create(entity_features={"metrics": ["not_a_metric"]})


# ── Bridges ────────────────────────────────────────────────────────────────


def _bridge_spec(**override: Any) -> dict[str, Any]:
    base = {
        "name": "bridge_co_user",
        "left": "dim_company",
        "right": "dim_user",
        "cardinality": [1, 3],
        "driver": "engagement",
        "columns": [
            {"name": "engagement_share", "type": "metric.engagement"},
        ],
    }
    base.update(override)
    return base


def test_bridge_translates_to_bridge_table_config():
    cfg = _create(bridges=[_bridge_spec()])
    assert len(cfg.bridges) == 1
    b = cfg.bridges[0]
    assert b.name == "bridge_co_user"
    assert b.connects == ["dim_company", "dim_user"]


def test_bridge_cardinality_maps_correctly():
    cfg = _create(bridges=[_bridge_spec(cardinality=[2, 5])])
    b = cfg.bridges[0]
    assert b.cardinality.min == 2
    assert b.cardinality.max == 5


def test_bridge_columns_translate_to_bridge_metrics():
    cfg = _create(
        bridges=[
            _bridge_spec(
                columns=[
                    {"name": "engagement_share", "type": "metric.engagement"},
                    {"name": "tier", "type": "static.gold"},
                    {"name": "company_name", "type": "faker.company"},
                ]
            )
        ]
    )
    metrics = cfg.bridges[0].metrics
    assert len(metrics) == 3
    sources = {m.name: m.source for m in metrics}
    assert sources["engagement_share"] == "metric:engagement"
    assert sources["tier"] == "static:gold"
    assert sources["company_name"] == "generated:faker.company"


def test_bridge_driver_validates_against_declared_metrics():
    with pytest.raises(ValueError, match="not a declared metric"):
        _create(bridges=[_bridge_spec(driver="ghost_metric")])


def test_bridge_invalid_dim_reference_rejected():
    with pytest.raises(ValueError, match="not a declared dimension"):
        _create(bridges=[_bridge_spec(left="dim_nonexistent")])


def test_bridge_unsupported_column_type_rejected():
    """Bridge metrics support metric/static/faker only — bridges have no
    period axis to anchor period-derived sources.
    """
    with pytest.raises(ValueError, match="not supported on bridge rows"):
        _create(
            bridges=[
                _bridge_spec(
                    columns=[
                        {"name": "date_key", "type": "ref.dim_date"},
                    ]
                )
            ]
        )


def test_bridge_omitted_yields_empty_default():
    cfg = _create()
    assert cfg.bridges == []


def test_bridge_self_join_rejected():
    with pytest.raises(ValueError, match="must be distinct"):
        _create(bridges=[_bridge_spec(right="dim_company")])


def test_bridge_inverse_cardinality_rejected():
    with pytest.raises(ValueError, match="min must be <= max"):
        _create(bridges=[_bridge_spec(cardinality=[5, 2])])


# ── Combined: power features compose ───────────────────────────────────────


def test_combined_power_features_all_enabled():
    """Mission AC: every feature must be opt-in independent. Combined use
    must validate clean (modulo engine-side mutex rules — quality with
    entity_features is rejected by the engine, but this test pairs
    bridges + holdout + entity_features without quality).
    """
    cfg = _create(
        bridges=[_bridge_spec()],
        holdout={"target": "mrr", "periods": 3},
        entity_features=True,
    )
    assert len(cfg.bridges) == 1
    assert cfg.holdout.enabled is True
    assert cfg.entity_features.enabled is True


def test_no_power_features_yields_pre_m122_defaults():
    """Omitting every M122 field reproduces the engine's pre-M122 defaults."""
    cfg = _create()
    assert cfg.bridges == []
    assert cfg.quality.quality_issues == []
    assert cfg.holdout.enabled is False
    assert cfg.entity_features.enabled is False


# ── UserInput-level validation surface ─────────────────────────────────────


def test_user_input_rejects_extra_top_level_field():
    """Confirms that ``extra='forbid'`` still applies after the M122
    additions — typos on the new optional fields surface as schema errors.
    """
    with pytest.raises(ValueError):
        UserInput.model_validate(
            {
                **_explicit_input(),
                "qualitee": [],  # typo — must not silently pass
            }
        )
