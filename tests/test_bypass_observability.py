"""Tests for M121b bypass-counter manifest surfacing.

The vectorized path's per-cell degenerate-distribution fallback was
silent in M121a — users couldn't tell whether vectorization wasn't
helping because of bypass. M121b threads a per-archetype counter
through ``_apply_correlations_batch`` →
``generate_archetype_batch`` → ``_compute_entity_metrics`` →
``GenerationState`` → ``manifest.bypass_fallback_counts``.

Three states the manifest field encodes:

  * ``None`` — serial mode; bypass never measured (no batched copula
    to fall back from).
  * ``{}`` — vectorized run with zero bypass cells (fast path covered
    every period).
  * ``{archetype: count, ...}`` — vectorized run where one or more
    archetypes had degenerate centers.

A complementary field ``vectorized_threshold_used`` records the
constant ``_VECTORIZED_AUTO_THRESHOLD`` at generation time so old
manifests stay reproducible if the constant changes.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from plotsim.builder import create_from_yaml
from plotsim.config import load_config
from plotsim.manifest import build_manifest
from plotsim.metrics import _VECTORIZED_AUTO_THRESHOLD
from plotsim.tables import generate_tables_with_state


ROOT = Path(__file__).resolve().parent.parent


def _build_manifest_for(cfg) -> object:
    rng = np.random.default_rng(cfg.seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tables, state = generate_tables_with_state(cfg, rng)
        return build_manifest(
            cfg, state.trajectories, tables,
            scd_state=state.scd, bridge_state=state.bridges,
        )


# --- Manifest field shape --------------------------------------------------


class TestManifestFieldShape:
    """``bypass_fallback_counts`` distinguishes serial from vectorized
    runs via the ``None`` vs ``{}`` boundary."""

    def test_serial_run_emits_none(self):
        cfg = load_config(ROOT / "plotsim" / "configs" / "sample_saas.yaml")
        assert cfg.generation_mode == "serial"
        m = _build_manifest_for(cfg)
        assert m.bypass_fallback_counts is None, (
            "serial mode should emit None — there's no batched copula "
            "to measure bypass against"
        )

    def test_vectorized_run_emits_dict(self):
        cfg = load_config(ROOT / "plotsim" / "configs" / "sample_saas.yaml")
        cfg = cfg.model_copy(update={"generation_mode": "vectorized"})
        m = _build_manifest_for(cfg)
        assert isinstance(m.bypass_fallback_counts, dict), (
            "vectorized mode should always emit a dict — empty when no "
            "bypass occurred, populated otherwise"
        )

    def test_keys_are_archetype_names(self):
        cfg = create_from_yaml(
            ROOT / "plotsim" / "configs" / "new" / "saas_template.yaml"
        )
        cfg = cfg.model_copy(update={"generation_mode": "vectorized"})
        m = _build_manifest_for(cfg)
        archetype_names = {e.archetype for e in cfg.entities}
        # Counter keys must be a subset of archetype names — the
        # dispatcher inserts entries on bypass; archetypes with no
        # bypass don't pre-populate.
        assert set(m.bypass_fallback_counts.keys()).issubset(archetype_names)

    def test_counts_are_nonnegative_ints(self):
        cfg = create_from_yaml(
            ROOT / "plotsim" / "configs" / "new" / "saas_template.yaml"
        )
        cfg = cfg.model_copy(update={"generation_mode": "vectorized"})
        m = _build_manifest_for(cfg)
        for arch, count in m.bypass_fallback_counts.items():
            assert isinstance(count, int), arch
            assert count >= 0, arch


# --- Counter increments correctly ------------------------------------------


class TestCounterIncrement:
    """Synthetic config that deliberately constructs a degenerate
    metric should drive the counter above zero on every period the
    degenerate cells appear."""

    def test_zero_in_serial_zero_lognorm_metric_yields_bypass(self):
        """A lognorm metric with very low ``loc`` and ``scale`` keeps
        the center near 0, which trips the lognorm bypass branch
        (``center <= _CENTER_EPS``). Force vectorized mode so the
        counter has somewhere to land."""
        from plotsim.config import (
            Archetype,
            Column,
            CorrelationPair,
            CurveSegment,
            Domain,
            Entity,
            Metric,
            NoiseConfig,
            OutputConfig,
            PlotsimConfig,
            Table,
            TimeWindow,
        )

        # Trajectory: plateau at 0.0 → polarity-positive lognorm
        # centers stay at ~0 the entire run → every cell trips bypass.
        archetype = Archetype(
            name="dead", label="Dead", description="zero plateau",
            curve_segments=[CurveSegment(
                curve="plateau", params={"level": 0.0},
                start_pct=0.0, end_pct=1.0,
            )],
            metric_overrides={},
        )
        m_a = Metric(
            name="a", label="A", distribution="lognorm",
            params={"s": 0.5, "loc": 0.0, "scale": 1.0},
            polarity="positive",
        )
        m_b = Metric(
            name="b", label="B", distribution="lognorm",
            params={"s": 0.5, "loc": 0.0, "scale": 1.0},
            polarity="positive",
        )
        cfg = PlotsimConfig(
            domain=Domain(name="bypass-trip", description="-",
                          entity_type="unit", entity_label="Unit"),
            time_window=TimeWindow(start="2024-01", end="2024-12",
                                   granularity="monthly"),
            seed=42,
            metrics=[m_a, m_b],
            archetypes=[archetype],
            entities=[Entity(name=f"e_{i:03d}", archetype="dead", size=1)
                      for i in range(60)],
            tables=[
                Table(name="dim_date", type="dim", grain="per_period",
                      primary_key="date_key",
                      columns=[Column(name="date_key", dtype="id", source="pk")]),
                Table(name="dim_entity", type="dim", grain="per_entity",
                      primary_key="entity_id",
                      columns=[Column(name="entity_id", dtype="id", source="pk")]),
                Table(name="fct_m", type="fact",
                      grain="per_entity_per_period",
                      primary_key=["entity_id", "date_key"],
                      columns=[
                          Column(name="date_key", source="fk:dim_date.date_key", dtype="int"),
                          Column(name="entity_id", source="fk:dim_entity.entity_id", dtype="id"),
                          Column(name="a", source="metric:a", dtype="float"),
                          Column(name="b", source="metric:b", dtype="float"),
                      ]),
            ],
            correlations=[CorrelationPair(metric_a="a", metric_b="b", coefficient=0.6)],
            noise=NoiseConfig(),
            output=OutputConfig(format="csv", directory="output"),
            generation_mode="vectorized",
        )
        m = _build_manifest_for(cfg)
        # Every cell across every period should trip bypass — the
        # archetype's plateau-at-zero trajectory keeps lognorm centers
        # below the epsilon threshold for the whole window.
        assert m.bypass_fallback_counts.get("dead", 0) > 0, (
            f"expected non-zero bypass for the all-degenerate config; "
            f"got {m.bypass_fallback_counts}"
        )

    def test_counter_zero_for_healthy_centers(self):
        """A config whose trajectory keeps centers comfortably above
        the epsilon threshold should produce zero bypass cells. The
        check guards against the bypass detection misfiring on
        production-shape configs."""
        from plotsim.config import (
            Archetype,
            Column,
            CorrelationPair,
            CurveSegment,
            Domain,
            Entity,
            Metric,
            NoiseConfig,
            OutputConfig,
            PlotsimConfig,
            Table,
            TimeWindow,
            ValueRange,
        )
        archetype = Archetype(
            name="healthy", label="Healthy",
            description="midrange normal centers",
            curve_segments=[CurveSegment(
                curve="plateau", params={"level": 0.5},
                start_pct=0.0, end_pct=1.0,
            )],
            metric_overrides={},
        )
        # Normal distribution with non-degenerate sigma + center at
        # mu*0.5 = 0.5; not degenerate.
        m_a = Metric(
            name="a", label="A", distribution="normal",
            params={"mu": 1.0, "sigma": 0.05},
            polarity="positive", value_range=ValueRange(min=0.0, max=10.0),
        )
        m_b = Metric(
            name="b", label="B", distribution="normal",
            params={"mu": 1.0, "sigma": 0.05},
            polarity="positive", value_range=ValueRange(min=0.0, max=10.0),
        )
        cfg = PlotsimConfig(
            domain=Domain(name="healthy", description="-",
                          entity_type="unit", entity_label="Unit"),
            time_window=TimeWindow(start="2024-01", end="2024-12",
                                   granularity="monthly"),
            seed=42,
            metrics=[m_a, m_b],
            archetypes=[archetype],
            entities=[Entity(name=f"e_{i:03d}", archetype="healthy", size=1)
                      for i in range(60)],
            tables=[
                Table(name="dim_date", type="dim", grain="per_period",
                      primary_key="date_key",
                      columns=[Column(name="date_key", dtype="id", source="pk")]),
                Table(name="dim_entity", type="dim", grain="per_entity",
                      primary_key="entity_id",
                      columns=[Column(name="entity_id", dtype="id", source="pk")]),
                Table(name="fct_m", type="fact",
                      grain="per_entity_per_period",
                      primary_key=["entity_id", "date_key"],
                      columns=[
                          Column(name="date_key", source="fk:dim_date.date_key", dtype="int"),
                          Column(name="entity_id", source="fk:dim_entity.entity_id", dtype="id"),
                          Column(name="a", source="metric:a", dtype="float"),
                          Column(name="b", source="metric:b", dtype="float"),
                      ]),
            ],
            correlations=[CorrelationPair(metric_a="a", metric_b="b", coefficient=0.6)],
            noise=NoiseConfig(),
            output=OutputConfig(format="csv", directory="output"),
            generation_mode="vectorized",
        )
        m = _build_manifest_for(cfg)
        assert m.bypass_fallback_counts.get("healthy", 0) == 0, (
            f"healthy normal centers shouldn't trip bypass; got "
            f"{m.bypass_fallback_counts}"
        )


# --- vectorized_threshold_used ---------------------------------------------


class TestThresholdField:
    """``vectorized_threshold_used`` records the constant value at
    generation time. Always populated (whether mode is serial or
    vectorized) so the manifest is always self-describing."""

    def test_field_matches_constant(self):
        cfg = load_config(ROOT / "plotsim" / "configs" / "sample_saas.yaml")
        m = _build_manifest_for(cfg)
        assert m.vectorized_threshold_used == _VECTORIZED_AUTO_THRESHOLD

    def test_field_present_in_serial(self):
        """Always populated regardless of mode — a serial run that
        crossed the threshold but the operator forced serial still
        records the constant for downstream auditing."""
        cfg = load_config(ROOT / "plotsim" / "configs" / "sample_saas.yaml")
        m = _build_manifest_for(cfg)
        assert m.vectorized_threshold_used is not None

    def test_field_present_in_vectorized(self):
        cfg = load_config(ROOT / "plotsim" / "configs" / "sample_saas.yaml")
        cfg = cfg.model_copy(update={"generation_mode": "vectorized"})
        m = _build_manifest_for(cfg)
        assert m.vectorized_threshold_used == _VECTORIZED_AUTO_THRESHOLD


# --- Round-trip via JSON ---------------------------------------------------


class TestManifestRoundTrip:
    """The new fields must serialize through the manifest's JSON
    pathway and reload identically — backward-compatible with the
    pre-M121b manifest schema (old manifests on disk should load with
    the new model via the field defaults)."""

    def test_round_trip_preserves_bypass_counts(self):
        cfg = create_from_yaml(
            ROOT / "plotsim" / "configs" / "new" / "saas_template.yaml"
        )
        cfg = cfg.model_copy(update={"generation_mode": "vectorized"})
        m = _build_manifest_for(cfg)
        from plotsim.manifest import ManifestSchema
        payload = m.model_dump(mode="json")
        m2 = ManifestSchema.model_validate(payload)
        assert m2.bypass_fallback_counts == m.bypass_fallback_counts
        assert m2.vectorized_threshold_used == m.vectorized_threshold_used

    def test_old_manifest_payload_loads(self):
        """A payload missing the new fields (i.e., a pre-M121b
        manifest on disk) should load with both fields defaulting
        to ``None``."""
        from plotsim.manifest import ManifestSchema
        payload = {
            "schema_version": "1.0",
            "seed": 0,
            "config_sha256": "0" * 64,
            "archetype_assignments": [],
            "trajectory_samples": [],
            "event_firings": [],
        }
        m = ManifestSchema.model_validate(payload)
        assert m.bypass_fallback_counts is None
        assert m.vectorized_threshold_used is None
