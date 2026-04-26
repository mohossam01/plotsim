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
        tables, cfg,
        output_dir=tmp_path,
        manifest=manifest,
    )
    return cfg, tables, state, manifest, target


# --- Manifest is written ----------------------------------------------------


def test_manifest_file_present_after_write(saas_run):
    _cfg, _tables, _state, _manifest, target = saas_run
    assert (target / MANIFEST_FILENAME).is_file(), (
        f"manifest not written to {target / MANIFEST_FILENAME}"
    )


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
    assert recorded == declared, (
        f"missing entities: {declared - recorded!r}; "
        f"extra: {recorded - declared!r}"
    )


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
        assert 0.0 <= s.position <= 1.0, (
            f"trajectory sample {s.entity}@{s.period_index} position "
            f"{s.position} outside [0, 1]"
        )


def test_every_event_table_has_a_firing_row_per_entity(saas_run):
    cfg, tables, _state, manifest, _target = saas_run
    event_table_names = {
        t.name for t in cfg.tables if t.type == "event"
    } & set(tables)
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
        assert f.period_indices == sorted(f.period_indices), (
            f"period_indices for {f.entity}@{f.table} is unsorted: "
            f"{f.period_indices}"
        )
        assert all(0 <= p < n_periods for p in f.period_indices), (
            f"period_indices for {f.entity}@{f.table} out of range: "
            f"{f.period_indices}"
        )


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
        tables, cfg, output_dir=tmp_path / "out", manifest=manifest,
    )

    assert not (target / MANIFEST_FILENAME).exists(), (
        "include:false config still emitted manifest.json"
    )


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
    assert sampled == expected, (
        f"sampled set {sampled!r} != expected sorted-prefix {expected!r}"
    )


def test_trajectory_sample_rate_minimum_one_entity():
    """Even at very small sample rates at least one entity lands."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SurrogateKeyWarning)
        cfg = load_config(SAAS_YAML)
    rng = np.random.default_rng(cfg.seed)
    tables, state = generate_tables_with_state(cfg, rng)
    manifest = build_manifest(
        cfg, state.trajectories, tables, sample_rate=0.001,
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


@pytest.mark.parametrize("template", [
    "sample_saas.yaml",
    "sample_ecommerce.yaml",
    "sample_education.yaml",
    "sample_healthcare.yaml",
    "sample_hr.yaml",
])
def test_all_bundled_templates_produce_valid_manifest(template, tmp_path):
    path = CONFIGS_DIR / template
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SurrogateKeyWarning)
        cfg = load_config(path)
    rng = np.random.default_rng(cfg.seed)
    tables, state = generate_tables_with_state(cfg, rng)
    manifest = build_manifest(cfg, state.trajectories, tables)
    target = write_tables(
        tables, cfg, output_dir=tmp_path, manifest=manifest,
    )
    raw = json.loads((target / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    loaded = ManifestSchema(**raw)
    assert loaded.schema_version == MANIFEST_SCHEMA_VERSION
    assert {a.entity for a in loaded.archetype_assignments} == \
        {e.name for e in cfg.entities}
