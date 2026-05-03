"""UserInput validation tests.

Mirrors the acceptance criteria in mission-115-builder.md ::Input
validation:: section. The model rejects structural problems with clear
error messages and emits ``UserWarning`` for semantic concerns.
"""
from __future__ import annotations

import warnings
from typing import Any

import pytest
from pydantic import ValidationError

from plotsim.builder.input import (
    ConnectionInput,
    EventInput,
    LifecycleInput,
    MetricInput,
    SegmentInput,
    UserInput,
    WindowInput,
)


# ── Helpers ─────────────────────────────────────────────────────────────────


def _minimal_input(**overrides: Any) -> dict[str, Any]:
    """A minimal-but-valid UserInput dict, ready for ``model_validate``."""
    base: dict[str, Any] = {
        "about": "test domain",
        "unit": "company",
        "window": {"start": "2023-01", "end": "2024-12", "every": "monthly"},
        "metrics": [
            {"name": "engagement", "type": "score", "polarity": "positive"},
            {"name": "mrr", "type": "amount", "polarity": "positive",
             "range": [100, 50000]},
        ],
        "segments": [
            {"name": "alpha", "count": 10, "archetype": "growth"},
            {"name": "beta", "count": 10, "archetype": "decline"},
        ],
    }
    base.update(overrides)
    return base


# ── Bare-minimum acceptance ─────────────────────────────────────────────────


def test_minimum_valid_input_constructs():
    UserInput.model_validate(_minimal_input())


def test_window_accepts_2_tuple_shorthand():
    data = _minimal_input(window=("2023-01", "2024-12"))
    ui = UserInput.model_validate(data)
    assert ui.window.every == "monthly"


def test_window_accepts_3_tuple_shorthand():
    data = _minimal_input(window=("2023-01", "2024-12", "weekly"))
    ui = UserInput.model_validate(data)
    assert ui.window.every == "weekly"


def test_window_tuple_wrong_arity_rejected():
    with pytest.raises(ValidationError):
        UserInput.model_validate(_minimal_input(window=("2023-01",)))


def test_connection_string_shorthand_parsed():
    data = _minimal_input(connections=["engagement driven_by mrr"])
    ui = UserInput.model_validate(data)
    assert ui.connections[0].metric_a == "engagement"
    assert ui.connections[0].relationship == "driven_by"
    assert ui.connections[0].metric_b == "mrr"


def test_connection_tuple_shorthand_parsed():
    data = _minimal_input(connections=[("engagement", "opposes", "mrr")])
    ui = UserInput.model_validate(data)
    assert ui.connections[0].relationship == "opposes"


def test_connection_string_wrong_token_count_rejected():
    with pytest.raises(ValidationError, match="three"):
        UserInput.model_validate(
            _minimal_input(connections=["engagement driven_by mrr extra"])
        )


# ── Required field enforcement ──────────────────────────────────────────────


def test_missing_about_rejected():
    data = _minimal_input()
    del data["about"]
    with pytest.raises(ValidationError, match="about"):
        UserInput.model_validate(data)


def test_missing_unit_rejected():
    data = _minimal_input()
    del data["unit"]
    with pytest.raises(ValidationError, match="unit"):
        UserInput.model_validate(data)


def test_missing_window_rejected():
    data = _minimal_input()
    del data["window"]
    with pytest.raises(ValidationError, match="window"):
        UserInput.model_validate(data)


def test_empty_metrics_rejected():
    with pytest.raises(ValidationError):
        UserInput.model_validate(_minimal_input(metrics=[]))


def test_empty_segments_rejected():
    with pytest.raises(ValidationError):
        UserInput.model_validate(_minimal_input(segments=[]))


def test_unknown_top_level_field_rejected():
    # extra="forbid" prevents typoed top-level keys.
    with pytest.raises(ValidationError):
        UserInput.model_validate(_minimal_input(metrix=[]))


# ── Metric range conditional rules ──────────────────────────────────────────


def test_score_metric_works_without_range():
    MetricInput.model_validate(
        {"name": "engagement", "type": "score", "polarity": "positive"}
    )


def test_count_metric_works_without_range():
    MetricInput.model_validate(
        {"name": "tickets", "type": "count", "polarity": "negative"}
    )


def test_amount_metric_without_range_rejected():
    with pytest.raises(ValidationError, match="range"):
        MetricInput.model_validate(
            {"name": "mrr", "type": "amount", "polarity": "positive"}
        )


def test_index_metric_without_range_rejected():
    with pytest.raises(ValidationError, match="range"):
        MetricInput.model_validate(
            {"name": "nps", "type": "index", "polarity": "positive"}
        )


def test_count_metric_with_range_rejected():
    with pytest.raises(ValidationError, match="range"):
        MetricInput.model_validate({
            "name": "tickets", "type": "count", "polarity": "negative",
            "range": [0, 100],
        })


def test_invalid_range_min_ge_max_rejected():
    with pytest.raises(ValidationError, match="min.*less than max"):
        MetricInput.model_validate({
            "name": "mrr", "type": "amount", "polarity": "positive",
            "range": [500, 500],
        })


# ── follows / delay coupling ────────────────────────────────────────────────


def test_follows_without_delay_rejected():
    with pytest.raises(ValidationError, match="follows.*delay"):
        MetricInput.model_validate({
            "name": "tickets", "type": "count", "polarity": "negative",
            "follows": "engagement",
        })


def test_delay_without_follows_rejected():
    with pytest.raises(ValidationError, match="follows.*delay"):
        MetricInput.model_validate({
            "name": "tickets", "type": "count", "polarity": "negative",
            "delay": 2,
        })


def test_follows_self_rejected():
    with pytest.raises(ValidationError, match="follow itself"):
        MetricInput.model_validate({
            "name": "tickets", "type": "count", "polarity": "negative",
            "follows": "tickets", "delay": 2,
        })


def test_delay_below_one_rejected():
    with pytest.raises(ValidationError):
        MetricInput.model_validate({
            "name": "tickets", "type": "count", "polarity": "negative",
            "follows": "engagement", "delay": 0,
        })


# ── Cross-reference: orphan and cycle detection ─────────────────────────────


def test_follows_unknown_metric_rejected_with_listing():
    data = _minimal_input(metrics=[
        {"name": "engagement", "type": "score", "polarity": "positive"},
        {"name": "tickets", "type": "count", "polarity": "negative",
         "follows": "ghost", "delay": 2},
    ])
    with pytest.raises(ValidationError, match="ghost"):
        UserInput.model_validate(data)


def test_two_metric_causal_cycle_detected():
    data = _minimal_input(metrics=[
        {"name": "a", "type": "score", "polarity": "positive",
         "follows": "b", "delay": 1},
        {"name": "b", "type": "score", "polarity": "positive",
         "follows": "a", "delay": 1},
    ])
    with pytest.raises(ValidationError, match="cycle"):
        UserInput.model_validate(data)


def test_three_metric_causal_cycle_detected():
    data = _minimal_input(metrics=[
        {"name": "a", "type": "score", "polarity": "positive",
         "follows": "b", "delay": 1},
        {"name": "b", "type": "score", "polarity": "positive",
         "follows": "c", "delay": 1},
        {"name": "c", "type": "score", "polarity": "positive",
         "follows": "a", "delay": 1},
    ])
    with pytest.raises(ValidationError, match="cycle"):
        UserInput.model_validate(data)


def test_follows_chain_without_cycle_accepted():
    # a → b → c chain (no cycle)
    data = _minimal_input(metrics=[
        {"name": "a", "type": "score", "polarity": "positive"},
        {"name": "b", "type": "score", "polarity": "positive",
         "follows": "a", "delay": 1},
        {"name": "c", "type": "score", "polarity": "positive",
         "follows": "b", "delay": 1},
    ])
    UserInput.model_validate(data)


def test_connection_orphan_endpoint_rejected_with_both_sides_named():
    data = _minimal_input(
        connections=["engagement driven_by ghost_metric"],
    )
    with pytest.raises(ValidationError) as exc:
        UserInput.model_validate(data)
    msg = str(exc.value)
    assert "ghost_metric" in msg
    assert "engagement" in msg  # both sides named


def test_baseline_orphan_metric_rejected():
    data = _minimal_input(segments=[
        {"name": "alpha", "count": 10, "archetype": "growth",
         "baseline": {"phantom": "high"}},
        {"name": "beta", "count": 10, "archetype": "decline"},
    ])
    with pytest.raises(ValidationError, match="phantom"):
        UserInput.model_validate(data)


def test_lifecycle_track_orphan_rejected():
    data = _minimal_input(lifecycle={
        "track": "phantom_metric",
        "stages": [
            {"onboarding": 0.0},
            {"active": 0.5},
        ],
    })
    with pytest.raises(ValidationError, match="phantom_metric"):
        UserInput.model_validate(data)


# ── Duplicates ──────────────────────────────────────────────────────────────


def test_duplicate_metric_names_rejected():
    data = _minimal_input(metrics=[
        {"name": "engagement", "type": "score", "polarity": "positive"},
        {"name": "engagement", "type": "score", "polarity": "negative"},
    ])
    with pytest.raises(ValidationError, match="duplicate"):
        UserInput.model_validate(data)


def test_duplicate_segment_names_rejected():
    data = _minimal_input(segments=[
        {"name": "alpha", "count": 10, "archetype": "growth"},
        {"name": "alpha", "count": 10, "archetype": "decline"},
    ])
    with pytest.raises(ValidationError, match="duplicate"):
        UserInput.model_validate(data)


# ── Vocabulary checks ───────────────────────────────────────────────────────


def test_unknown_relationship_word_rejected_with_listing():
    with pytest.raises(ValidationError) as exc:
        ConnectionInput.model_validate({
            "metric_a": "a", "relationship": "wibbles", "metric_b": "b",
        })
    msg = str(exc.value)
    assert "wibbles" in msg
    assert "mirrors" in msg  # listing of valid words


def test_unknown_baseline_word_rejected():
    with pytest.raises(ValidationError, match="baseline"):
        SegmentInput.model_validate({
            "name": "alpha", "count": 10, "archetype": "growth",
            "baseline": {"engagement": "stratospheric"},
        })


def test_unknown_metric_type_rejected():
    with pytest.raises(ValidationError):
        MetricInput.model_validate({
            "name": "x", "type": "voltage", "polarity": "positive",
        })


def test_unknown_polarity_rejected():
    with pytest.raises(ValidationError):
        MetricInput.model_validate({
            "name": "x", "type": "score", "polarity": "neutral",
        })


def test_malformed_archetype_dsl_rejected_with_segment_named():
    data = _minimal_input(segments=[
        {"name": "alpha", "count": 10, "archetype": "rocket > flat @ 12"},
        {"name": "beta", "count": 10, "archetype": "decline"},
    ])
    with pytest.raises(ValidationError) as exc:
        UserInput.model_validate(data)
    assert "alpha" in str(exc.value)
    assert "rocket" in str(exc.value)


def test_archetype_layered_plus_rejected():
    data = _minimal_input(segments=[
        {"name": "alpha", "count": 10, "archetype": "growth + decline"},
        {"name": "beta", "count": 10, "archetype": "decline"},
    ])
    with pytest.raises(ValidationError, match="future release"):
        UserInput.model_validate(data)


# ── Segment count guard ─────────────────────────────────────────────────────


def test_segment_count_below_three_rejected():
    with pytest.raises(ValidationError):
        SegmentInput.model_validate({
            "name": "alpha", "count": 2, "archetype": "growth",
        })


# ── Connection endpoint rules ───────────────────────────────────────────────


def test_connection_endpoints_must_differ():
    with pytest.raises(ValidationError, match="distinct"):
        ConnectionInput.model_validate({
            "metric_a": "x", "relationship": "mirrors", "metric_b": "x",
        })


# ── Lifecycle / stages aliases ──────────────────────────────────────────────


def test_lifecycle_block_accepted_via_lifecycle_keyword():
    data = _minimal_input(metrics=[
        {"name": "engagement", "type": "score", "polarity": "positive"},
        {"name": "churn_risk", "type": "score", "polarity": "negative"},
    ], lifecycle={
        "track": "churn_risk",
        "stages": [
            {"onboarding": 0.0},
            {"active": 0.2},
            {"at_risk": 0.5},
        ],
    })
    ui = UserInput.model_validate(data)
    assert ui.lifecycle is not None
    assert ui.lifecycle.track == "churn_risk"
    assert [s.name for s in ui.lifecycle.stages] == ["onboarding", "active", "at_risk"]


def test_lifecycle_block_accepted_via_stages_alias():
    # mission spec text used 'stages:' as the outer block name; we accept it
    # as an alias for resilience even though templates use 'lifecycle:'.
    data = _minimal_input(metrics=[
        {"name": "engagement", "type": "score", "polarity": "positive"},
        {"name": "churn_risk", "type": "score", "polarity": "negative"},
    ], stages={
        "track": "churn_risk",
        "stages": [
            {"onboarding": 0.0},
            {"active": 0.2},
        ],
    })
    ui = UserInput.model_validate(data)
    assert ui.lifecycle is not None
    assert ui.lifecycle.track == "churn_risk"


def test_lifecycle_stages_accept_tuple_shape():
    LifecycleInput.model_validate({
        "track": "x",
        "stages": [("a", 0.0), ("b", 0.5)],
    })


def test_lifecycle_stages_strictly_ascending_thresholds():
    with pytest.raises(ValidationError, match="ascending"):
        LifecycleInput.model_validate({
            "track": "x",
            "stages": [
                {"a": 0.5},
                {"b": 0.2},  # not ascending
            ],
        })


def test_lifecycle_stages_unique_names():
    with pytest.raises(ValidationError, match="unique"):
        LifecycleInput.model_validate({
            "track": "x",
            "stages": [
                {"a": 0.0},
                {"a": 0.5},
            ],
        })


# ── Event trigger field gating ──────────────────────────────────────────────


def test_proportional_event_requires_driver_and_scale():
    with pytest.raises(ValidationError, match="driver"):
        EventInput.model_validate({
            "name": "evt", "trigger": "proportional",
            "columns": [{"name": "id", "type": "id"}],
        })


def test_threshold_event_requires_metric():
    with pytest.raises(ValidationError, match="metric"):
        EventInput.model_validate({
            "name": "evt", "trigger": "threshold",
            "columns": [{"name": "id", "type": "id"}],
        })


def test_threshold_event_requires_above_or_below():
    with pytest.raises(ValidationError, match="above|below"):
        EventInput.model_validate({
            "name": "evt", "trigger": "threshold", "metric": "x",
            "columns": [{"name": "id", "type": "id"}],
        })


def test_threshold_event_for_keyword_alias_yaml():
    # YAML reference template uses 'for:' — accepted as alias for for_periods.
    evt = EventInput.model_validate({
        "name": "evt", "trigger": "threshold", "metric": "x", "above": 0.7,
        "columns": [{"name": "id", "type": "id"}],
        "for": 3,
    })
    assert evt.for_periods == 3


def test_threshold_event_for_periods_canonical_python():
    evt = EventInput.model_validate({
        "name": "evt", "trigger": "threshold", "metric": "x", "above": 0.7,
        "columns": [{"name": "id", "type": "id"}],
        "for_periods": 3,
    })
    assert evt.for_periods == 3


def test_threshold_event_above_and_below_both_rejected():
    with pytest.raises(ValidationError, match="not both"):
        EventInput.model_validate({
            "name": "evt", "trigger": "threshold", "metric": "x",
            "above": 0.7, "below": 0.2,
            "columns": [{"name": "id", "type": "id"}],
        })


# ── Semantic warnings ───────────────────────────────────────────────────────


def test_short_window_seasonal_emits_warning():
    data = _minimal_input(
        window={"start": "2024-01", "end": "2024-12", "every": "monthly"},
        segments=[
            {"name": "alpha", "count": 10, "archetype": "growth"},
            {"name": "beta", "count": 10, "archetype": "seasonal"},  # 12 < 24
        ],
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        UserInput.model_validate(data)
    assert any(
        "seasonal" in str(w.message) for w in caught
    ), f"expected a seasonal-window warning, got {[str(w.message) for w in caught]}"


def test_seasonal_with_long_window_no_warning():
    data = _minimal_input(
        window={"start": "2022-01", "end": "2024-12", "every": "monthly"},
        segments=[
            {"name": "alpha", "count": 10, "archetype": "growth"},
            {"name": "beta", "count": 10, "archetype": "seasonal"},  # 36 periods
        ],
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        UserInput.model_validate(data)
    assert not any(
        "seasonal" in str(w.message) for w in caught
    )


def test_single_segment_emits_warning():
    data = _minimal_input(segments=[
        {"name": "only", "count": 10, "archetype": "growth"},
    ])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        UserInput.model_validate(data)
    assert any(
        "one segment" in str(w.message) for w in caught
    )


def test_mirrors_with_eight_metrics_emits_warning():
    metrics = [
        {"name": f"m{i}", "type": "score", "polarity": "positive"}
        for i in range(8)
    ]
    data = _minimal_input(
        metrics=metrics,
        connections=["m0 mirrors m1"],
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        UserInput.model_validate(data)
    assert any(
        "mirrors" in str(w.message) or "PSD" in str(w.message)
        for w in caught
    )


def test_modest_correlations_with_eight_metrics_no_warning():
    metrics = [
        {"name": f"m{i}", "type": "score", "polarity": "positive"}
        for i in range(8)
    ]
    data = _minimal_input(
        metrics=metrics,
        connections=["m0 driven_by m1"],  # +0.55, not +0.75
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        UserInput.model_validate(data)
    assert not any("PSD" in str(w.message) for w in caught)


# ── Reference-template smoke test ───────────────────────────────────────────


def test_yaml_reference_template_loads_as_userinput():
    """The saas_template.yaml must validate against UserInput verbatim.

    This test acts as the early canary for Phase 4: if the structural
    surface of UserInput drifts from the reference template, this fails
    long before integration tests catch it.
    """
    import yaml
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    template = repo_root / "plotsim" / "configs" / "templates" / "saas_template.yaml"
    raw = yaml.safe_load(template.read_text(encoding="utf-8"))
    # YAML parses date-like strings (2023-01) into date objects in some
    # codepaths; coerce window strings back to ISO month form for
    # WindowInput's `str` typing.
    if "window" in raw:
        for key in ("start", "end"):
            if key in raw["window"] and not isinstance(raw["window"][key], str):
                raw["window"][key] = str(raw["window"][key])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # template uses seasonal; ignore that warn
        UserInput.model_validate(raw)
