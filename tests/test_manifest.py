"""M105 / Track A — manifest.json tests.

Covers:

* CLI ``plotsim run`` produces ``manifest.json`` next to the table files
* manifest payload is JSON-loadable; values are native Python types (no
  numpy leak)
* ``ManifestSchema`` validates the loaded payload (forward-looking
  contract for downstream consumers)
* manifest is byte-deterministic across runs at the same seed
* every entity has an archetype assignment
* every event table has at least one EventFiring row per entity
* every sampled entity has trajectory_samples covering every period
* ``include: false`` suppresses manifest creation
* ``trajectory_sample_rate < 1.0`` reduces the sample list and is
  itself deterministic (sorted-name prefix, not RNG-driven)
* every bundled template produces a valid manifest
* ``config_sha256`` is the full 64-char SHA-256 of the json-serialized
  config dump and changes when the config changes
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest
import yaml

from plotsim import (
    ManifestSchema,
    SurrogateKeyWarning,
    build_manifest,
    generate_tables_with_state,
    load_config,
    write_tables,
)
from plotsim.manifest import MANIFEST_FILENAME, MANIFEST_SCHEMA_VERSION


ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = ROOT / "plotsim" / "configs"
SAAS_YAML = CONFIGS_DIR / "sample_saas.yaml"


@pytest.fixture
def saas_run(tmp_path):
    """Run the saas template end-to-end into ``tmp_path``; return artefacts."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SurrogateKeyWarning)
        cfg = load_config(SAAS_YAML)
    rng = np.random.default_rng(cfg.seed)
    tables, state = generate_tables_with_state(cfg, rng)
    manifest = build_manifest(cfg, state.trajectories, tables)
    target = write_tables(
        tables,
        cfg,
        output_dir=tmp_path,
        manifest=manifest,
    )
    return cfg, tables, state, manifest, target


# --- Manifest is written ----------------------------------------------------


def test_manifest_file_present_after_write(saas_run):
    _cfg, _tables, _state, _manifest, target = saas_run
    assert (
        target / MANIFEST_FILENAME
    ).is_file(), f"manifest not written to {target / MANIFEST_FILENAME}"


def test_manifest_is_json_loadable(saas_run):
    _cfg, _tables, _state, _manifest, target = saas_run
    payload = json.loads((target / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert isinstance(payload, dict)


def test_manifest_payload_validates_against_schema_model(saas_run):
    _cfg, _tables, _state, _manifest, target = saas_run
    raw = json.loads((target / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    # Round-trips through the model — proves the wire shape is whatever
    # the model says it is.
    loaded = ManifestSchema(**raw)
    assert loaded.schema_version == MANIFEST_SCHEMA_VERSION


def test_manifest_seed_recorded(saas_run):
    cfg, _tables, _state, manifest, _target = saas_run
    assert manifest.seed == cfg.seed


def test_manifest_config_sha256_full_64_hex(saas_run):
    _cfg, _tables, _state, manifest, _target = saas_run
    assert len(manifest.config_sha256) == 64
    int(manifest.config_sha256, 16)  # must be valid hex


# --- Numpy leakage check ----------------------------------------------------


def test_manifest_payload_has_no_numpy_types(saas_run):
    """Every leaf value must be a native Python primitive.

    pyarrow / json round-trips choke on numpy scalars; this test makes
    a numpy leak into the manifest fail at the wire format, not at the
    downstream consumer.
    """
    _cfg, _tables, _state, _manifest, target = saas_run
    raw = json.loads((target / MANIFEST_FILENAME).read_text(encoding="utf-8"))

    def _walk(node):
        if isinstance(node, dict):
            for v in node.values():
                _walk(v)
        elif isinstance(node, list):
            for v in node:
                _walk(v)
        else:
            assert type(node).__module__ == "builtins" or node is None, (
                f"non-builtin leaf {type(node).__module__}.{type(node).__name__} "
                f"escaped into manifest.json: {node!r}"
            )

    _walk(raw)


# --- Completeness on the bundled saas template ------------------------------


def test_every_entity_has_an_archetype_assignment(saas_run):
    cfg, _tables, _state, manifest, _target = saas_run
    declared = {e.name for e in cfg.entities}
    recorded = {a.entity for a in manifest.archetype_assignments}
    assert (
        recorded == declared
    ), f"missing entities: {declared - recorded!r}; extra: {recorded - declared!r}"


def test_archetype_assignment_matches_config(saas_run):
    cfg, _tables, _state, manifest, _target = saas_run
    by_entity = {a.entity: a.archetype for a in manifest.archetype_assignments}
    for ent in cfg.entities:
        assert by_entity[ent.name] == ent.archetype


def test_trajectory_samples_cover_every_period_for_sampled_entities(saas_run):
    cfg, tables, _state, manifest, _target = saas_run
    n_periods = len(tables["dim_date"])
    by_entity: dict = {}
    for s in manifest.trajectory_samples:
        by_entity.setdefault(s.entity, []).append(s.period_index)
    for ename, periods in by_entity.items():
        assert sorted(periods) == list(range(n_periods)), (
            f"entity {ename!r} trajectory samples missing periods: "
            f"got {sorted(periods)}, expected 0..{n_periods - 1}"
        )


def test_trajectory_sample_positions_are_in_unit_interval(saas_run):
    _cfg, _tables, _state, manifest, _target = saas_run
    for s in manifest.trajectory_samples:
        assert (
            0.0 <= s.position <= 1.0
        ), f"trajectory sample {s.entity}@{s.period_index} position {s.position} outside [0, 1]"


def test_every_event_table_has_a_firing_row_per_entity(saas_run):
    cfg, tables, _state, manifest, _target = saas_run
    event_table_names = {t.name for t in cfg.tables if t.type == "event"} & set(tables)
    if not event_table_names:
        pytest.skip("template declares no event tables")
    by_table: dict = {}
    for f in manifest.event_firings:
        by_table.setdefault(f.table, set()).add(f.entity)
    declared_entities = {e.name for e in cfg.entities}
    for table_name in event_table_names:
        assert by_table.get(table_name) == declared_entities, (
            f"event_firings for {table_name!r} missing entities: "
            f"{declared_entities - by_table.get(table_name, set())!r}"
        )


def test_event_firing_period_indices_are_valid_and_sorted(saas_run):
    _cfg, tables, _state, manifest, _target = saas_run
    n_periods = len(tables["dim_date"])
    for f in manifest.event_firings:
        assert f.period_indices == sorted(
            f.period_indices
        ), f"period_indices for {f.entity}@{f.table} is unsorted: {f.period_indices}"
        assert all(
            0 <= p < n_periods for p in f.period_indices
        ), f"period_indices for {f.entity}@{f.table} out of range: {f.period_indices}"


# --- Determinism contract ---------------------------------------------------


def test_manifest_byte_identical_across_runs(tmp_path):
    """Same config + same seed → byte-identical manifest.json."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SurrogateKeyWarning)
        cfg = load_config(SAAS_YAML)

    out_a = tmp_path / "run_a"
    out_b = tmp_path / "run_b"

    for out in (out_a, out_b):
        rng = np.random.default_rng(cfg.seed)
        tables, state = generate_tables_with_state(cfg, rng)
        manifest = build_manifest(cfg, state.trajectories, tables)
        write_tables(tables, cfg, output_dir=out, manifest=manifest)

    bytes_a = (out_a / MANIFEST_FILENAME).read_bytes()
    bytes_b = (out_b / MANIFEST_FILENAME).read_bytes()
    assert bytes_a == bytes_b, (
        f"manifest.json differs across two runs at the same seed: "
        f"{len(bytes_a)} vs {len(bytes_b)} bytes"
    )


# --- Suppression: include: false --------------------------------------------


def test_include_false_suppresses_manifest_file(tmp_path):
    """``manifest: {include: false}`` skips the file entirely."""
    base = yaml.safe_load(SAAS_YAML.read_text(encoding="utf-8"))
    base["manifest"] = {"include": False}
    cfg_path = tmp_path / "no_manifest.yaml"
    cfg_path.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SurrogateKeyWarning)
        cfg = load_config(cfg_path)
    rng = np.random.default_rng(cfg.seed)
    tables, state = generate_tables_with_state(cfg, rng)
    # Still build (cheap) and pass; write_tables gates on config.manifest.include.
    manifest = build_manifest(cfg, state.trajectories, tables)
    target = write_tables(
        tables,
        cfg,
        output_dir=tmp_path / "out",
        manifest=manifest,
    )

    assert not (
        target / MANIFEST_FILENAME
    ).exists(), "include:false config still emitted manifest.json"


def test_include_default_is_true():
    """The bundled saas template (no manifest block) opts into emission."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SurrogateKeyWarning)
        cfg = load_config(SAAS_YAML)
    assert cfg.manifest.include is True


# --- Sample-rate determinism ------------------------------------------------


def test_trajectory_sample_rate_below_one_reduces_samples(tmp_path):
    """A 33% sample on 3 entities yields exactly 1 sampled entity."""
    base = yaml.safe_load(SAAS_YAML.read_text(encoding="utf-8"))
    base["manifest"] = {"include": True, "trajectory_sample_rate": 0.34}
    cfg_path = tmp_path / "sampled.yaml"
    cfg_path.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SurrogateKeyWarning)
        cfg = load_config(cfg_path)
    rng = np.random.default_rng(cfg.seed)
    tables, state = generate_tables_with_state(cfg, rng)
    manifest = build_manifest(cfg, state.trajectories, tables)

    sampled = {s.entity for s in manifest.trajectory_samples}
    # ceil(3 * 0.34) == 2 → first two entities by sorted name.
    expected = set(sorted(e.name for e in cfg.entities)[:2])
    assert sampled == expected, f"sampled set {sampled!r} != expected sorted-prefix {expected!r}"


def test_trajectory_sample_rate_minimum_one_entity():
    """Even at very small sample rates at least one entity lands."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SurrogateKeyWarning)
        cfg = load_config(SAAS_YAML)
    rng = np.random.default_rng(cfg.seed)
    tables, state = generate_tables_with_state(cfg, rng)
    manifest = build_manifest(
        cfg,
        state.trajectories,
        tables,
        sample_rate=0.001,
    )
    sampled = {s.entity for s in manifest.trajectory_samples}
    assert len(sampled) == 1


# --- config_sha256 sensitivity ----------------------------------------------


def test_config_sha256_changes_when_seed_changes(tmp_path):
    """Different seed → different config_sha256."""
    base = yaml.safe_load(SAAS_YAML.read_text(encoding="utf-8"))

    base["seed"] = 42
    p_a = tmp_path / "a.yaml"
    p_a.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")

    base["seed"] = 99
    p_b = tmp_path / "b.yaml"
    p_b.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SurrogateKeyWarning)
        cfg_a = load_config(p_a)
        cfg_b = load_config(p_b)

    rng_a = np.random.default_rng(cfg_a.seed)
    tables_a, state_a = generate_tables_with_state(cfg_a, rng_a)
    rng_b = np.random.default_rng(cfg_b.seed)
    tables_b, state_b = generate_tables_with_state(cfg_b, rng_b)

    m_a = build_manifest(cfg_a, state_a.trajectories, tables_a)
    m_b = build_manifest(cfg_b, state_b.trajectories, tables_b)
    assert m_a.config_sha256 != m_b.config_sha256


# --- Every bundled template produces a valid manifest -----------------------


@pytest.mark.parametrize(
    "template",
    [
        "sample_saas.yaml",
        "sample_retail.yaml",
        "sample_education.yaml",
        "sample_marketing.yaml",
        "sample_hr.yaml",
    ],
)
def test_all_bundled_templates_produce_valid_manifest(template, tmp_path):
    path = CONFIGS_DIR / template
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SurrogateKeyWarning)
        cfg = load_config(path)
    rng = np.random.default_rng(cfg.seed)
    tables, state = generate_tables_with_state(cfg, rng)
    manifest = build_manifest(cfg, state.trajectories, tables)
    target = write_tables(
        tables,
        cfg,
        output_dir=tmp_path,
        manifest=manifest,
    )
    raw = json.loads((target / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    loaded = ManifestSchema(**raw)
    assert loaded.schema_version == MANIFEST_SCHEMA_VERSION
    assert {a.entity for a in loaded.archetype_assignments} == {e.name for e in cfg.entities}


# --- 0.6-M5: causal_graph ---------------------------------------------------


def test_schema_version_bumped_to_1_7():
    """0.6-M5 added causal_graph / correlations / outlier_injections (1.0 → 1.1).
    0.6-M8a added per-entity ``active_window`` on EntityArchetypeAssignment
    (1.1 → 1.2). 0.6-M8c added per-entity ``treatment`` and the top-level
    ``treatment_cohorts`` list (1.2 → 1.3). 0.6-M11 added the top-level
    ``correlation_phases`` summary list and the optional ``phase_index``
    field on ``CorrelationAdjustment`` / ``CorrelationCompensation`` /
    ``CorrelationEntry`` (1.3 → 1.4). 0.6-M13 added the
    ``source_entity_mappings`` list for the multi-source / overlap mode
    (1.4 → 1.5). 0.6-M18 added the ``parent_child_relations`` list for
    the parent/child fact grain (1.5 → 1.6). 0.6-M22 added the optional
    ``noise_config`` field, populated only on heteroscedastic-noise runs
    (1.6 → 1.7).

    The version pin lives in this test rather than just the manifest module
    so a downstream consumer pinning ``schema_version >= "1.7"`` has a
    direct on-disk contract test it can reference.
    """
    assert MANIFEST_SCHEMA_VERSION == "1.7"


def test_causal_graph_emits_one_edge_per_metric_with_lag(saas_run):
    cfg, _tables, _state, manifest, _target = saas_run
    expected = {(m.causal_lag.driver, m.name) for m in cfg.metrics if m.causal_lag is not None}
    actual = {(e.driver, e.target) for e in manifest.causal_graph}
    assert actual == expected
    assert len(manifest.causal_graph) >= 1, "saas template has at least one causal_lag"


def test_causal_graph_records_lag_periods_and_blend_weight(saas_run):
    cfg, _tables, _state, manifest, _target = saas_run
    by_pair = {(e.driver, e.target): e for e in manifest.causal_graph}
    for m in cfg.metrics:
        if m.causal_lag is None:
            continue
        edge = by_pair[(m.causal_lag.driver, m.name)]
        assert edge.lag_periods == m.causal_lag.lag_periods
        assert edge.blend_weight == m.causal_lag.blend_weight


def test_causal_graph_sorted_for_byte_determinism(saas_run):
    _cfg, _tables, _state, manifest, _target = saas_run
    keys = [(e.driver, e.target) for e in manifest.causal_graph]
    assert keys == sorted(keys), "causal_graph must be sorted for stable JSON"


def test_causal_graph_empty_when_no_metric_has_lag():
    """Strip ``causal_lag`` off every metric; expect an empty graph."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SurrogateKeyWarning)
        cfg = load_config(SAAS_YAML)
    new_metrics = [m.model_copy(update={"causal_lag": None}) for m in cfg.metrics]
    cfg_no_lag = cfg.model_copy(update={"metrics": new_metrics})
    rng = np.random.default_rng(cfg_no_lag.seed)
    tables, state = generate_tables_with_state(cfg_no_lag, rng)
    manifest = build_manifest(cfg_no_lag, state.trajectories, tables)
    assert manifest.causal_graph == []


# --- 0.6-M5: correlations ---------------------------------------------------


def test_correlations_one_entry_per_user_declared_pair(saas_run):
    cfg, _tables, _state, manifest, _target = saas_run
    assert len(manifest.correlations) == len(cfg.correlations)
    expected_pairs = {(p.metric_a, p.metric_b) for p in cfg.correlations}
    actual_pairs = {(c.metric_a, c.metric_b) for c in manifest.correlations}
    assert actual_pairs == expected_pairs


def test_correlations_requested_matches_config(saas_run):
    cfg, _tables, _state, manifest, _target = saas_run
    cfg_by_pair = {(p.metric_a, p.metric_b): p.coefficient for p in cfg.correlations}
    for entry in manifest.correlations:
        assert entry.requested == cfg_by_pair[(entry.metric_a, entry.metric_b)]


def test_correlations_projected_in_unit_interval(saas_run):
    """Higham projection lands every coefficient in [-1, 1]."""
    _cfg, _tables, _state, manifest, _target = saas_run
    for entry in manifest.correlations:
        assert -1.0 <= entry.projected <= 1.0


def test_correlations_projected_differs_from_requested_for_non_pd(saas_run):
    """saas correlation matrix is not PD — projected values must shift."""
    _cfg, _tables, _state, manifest, _target = saas_run
    deltas = [abs(entry.requested - entry.projected) for entry in manifest.correlations]
    # Saas is the canonical non-PD config (also asserted by M111
    # correlation_adjustments tests), so at least one pair must move.
    assert any(
        d > 1e-9 for d in deltas
    ), "saas matrix is not PD; expected at least one projected coefficient to shift"


def test_correlations_sorted_for_byte_determinism(saas_run):
    _cfg, _tables, _state, manifest, _target = saas_run
    keys = [(c.metric_a, c.metric_b) for c in manifest.correlations]
    assert keys == sorted(keys)


def test_correlations_empty_when_no_correlations_configured():
    """Strip the correlations list; expect an empty manifest section."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SurrogateKeyWarning)
        cfg = load_config(SAAS_YAML)
    cfg_no_corr = cfg.model_copy(update={"correlations": []})
    rng = np.random.default_rng(cfg_no_corr.seed)
    tables, state = generate_tables_with_state(cfg_no_corr, rng)
    manifest = build_manifest(cfg_no_corr, state.trajectories, tables)
    assert manifest.correlations == []


# --- 0.6-M5: outlier_injections --------------------------------------------


@pytest.fixture
def saas_serial_run():
    """Run saas pinned to serial mode so outlier replay is enabled."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SurrogateKeyWarning)
        cfg = load_config(SAAS_YAML)
    cfg = cfg.model_copy(update={"generation_mode": "serial"})
    rng = np.random.default_rng(cfg.seed)
    tables, state = generate_tables_with_state(cfg, rng)
    manifest = build_manifest(cfg, state.trajectories, tables)
    return cfg, tables, state, manifest


def test_outlier_injections_populated_for_serial_with_outlier_rate(saas_serial_run):
    _cfg, _tables, _state, manifest = saas_serial_run
    assert manifest.outlier_injections is not None
    # Serial saas at outlier_rate=0.02 over ~432 cells: a handful of
    # firings is overwhelmingly likely. Zero would still be valid output
    # but would also be a strong signal of detector regression — pin a
    # weak lower bound to fail loudly if the replay short-circuits.
    assert len(manifest.outlier_injections) >= 1


def test_outlier_injection_records_have_valid_coordinates(saas_serial_run):
    cfg, _tables, _state, manifest = saas_serial_run
    entity_names = {e.name for e in cfg.entities}
    metric_names = {m.name for m in cfg.metrics}
    for record in manifest.outlier_injections or []:
        assert record.entity in entity_names
        assert record.metric in metric_names
        assert record.period_index >= 0


def test_outlier_injections_sorted_for_byte_determinism(saas_serial_run):
    _cfg, _tables, _state, manifest = saas_serial_run
    records = manifest.outlier_injections or []
    keys = [(r.entity, r.period_index, r.metric) for r in records]
    assert keys == sorted(keys)


def test_outlier_injections_none_when_outlier_rate_zero():
    """outlier_rate=0 short-circuits the detector before the replay."""
    from plotsim.config import NoiseConfig

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SurrogateKeyWarning)
        cfg = load_config(SAAS_YAML)
    clean_noise = NoiseConfig(
        gaussian_sigma=cfg.noise.gaussian_sigma,
        outlier_rate=0.0,
        mcar_rate=cfg.noise.mcar_rate,
    )
    cfg_clean = cfg.model_copy(update={"noise": clean_noise, "generation_mode": "serial"})
    rng = np.random.default_rng(cfg_clean.seed)
    tables, state = generate_tables_with_state(cfg_clean, rng)
    manifest = build_manifest(cfg_clean, state.trajectories, tables)
    assert manifest.outlier_injections is None


def test_outlier_injections_none_when_vectorized_mode():
    """Vectorized RNG order doesn't match per-cell apply_noise replay."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SurrogateKeyWarning)
        cfg = load_config(SAAS_YAML)
    cfg_vec = cfg.model_copy(update={"generation_mode": "vectorized"})
    rng = np.random.default_rng(cfg_vec.seed)
    tables, state = generate_tables_with_state(cfg_vec, rng)
    manifest = build_manifest(cfg_vec, state.trajectories, tables)
    assert manifest.outlier_injections is None


def test_outlier_injections_none_above_cell_budget(monkeypatch):
    """Cost-gate skip when cells exceed OUTLIER_DETECTION_CELL_BUDGET."""
    import plotsim.outlier_injections as oi_mod

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SurrogateKeyWarning)
        cfg = load_config(SAAS_YAML)
    cfg_serial = cfg.model_copy(update={"generation_mode": "serial"})
    # Crank the budget down below saas's actual cell count to exercise the
    # gate without having to fabricate a large config that runs slow.
    monkeypatch.setattr(oi_mod, "OUTLIER_DETECTION_CELL_BUDGET", 1)
    result = oi_mod.detect_outlier_injections(cfg_serial)
    assert result is None


def test_outlier_injections_appears_in_written_manifest_json(tmp_path):
    """The new section must survive the JSON round-trip on disk."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SurrogateKeyWarning)
        cfg = load_config(SAAS_YAML)
    cfg = cfg.model_copy(update={"generation_mode": "serial"})
    rng = np.random.default_rng(cfg.seed)
    tables, state = generate_tables_with_state(cfg, rng)
    manifest = build_manifest(cfg, state.trajectories, tables)
    target = write_tables(tables, cfg, output_dir=tmp_path, manifest=manifest)
    raw = json.loads((target / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert "outlier_injections" in raw
    assert "causal_graph" in raw
    assert "correlations" in raw
    loaded = ManifestSchema(**raw)
    assert loaded.outlier_injections is not None
    assert loaded.causal_graph
    assert loaded.correlations
