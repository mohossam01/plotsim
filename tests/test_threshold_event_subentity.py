"""F1 regression — threshold-event sub-entity FK collapse (M102).

Pre-fix: ``_build_threshold_event`` passed ``rng=None`` into
``_resolve_event_row``. Inside ``_resolve_event_row``'s sub-entity FK
branch (``tables.py:1115-1119``), a ``None`` rng falls back to
``candidates.iloc[0]`` — collapsing every threshold event for a given
parent entity to the same sub-entity record across seeds.

Post-fix: rng is threaded from ``build_event_tables`` through
``_build_threshold_event`` into ``_resolve_event_row``, so the
sub-entity FK is sampled randomly from the parent's candidates.

Bug surface: any threshold-driven event table that FKs into a
sub-entity dim. None of the five bundled templates exercise this
combination (``evt_churn`` in saas FKs only into ``dim_company``,
not ``dim_user``), so the regression here builds a mutated saas
config that adds a ``user_id`` FK to ``evt_churn``.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from plotsim import generate_tables, load_config
from plotsim.config import SurrogateKeyWarning


ROOT = Path(__file__).resolve().parent.parent
SAAS_YAML = ROOT / "plotsim" / "configs" / "sample_saas.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _write_yaml(data: dict[str, Any], path: Path) -> Path:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def _mutate_evt_churn_to_fk_dim_user(data: dict) -> None:
    """Add a ``user_id`` FK to ``evt_churn``, FKing into ``dim_user``.

    Triggers the F1 bug path. Also flips every entity to
    ``rocket_then_cliff`` so every parent's churn_risk crosses the
    ``above:0.7:for:3`` threshold during the window — guarantees an
    event row per parent per seed for statistical power.
    """
    for tbl in data["tables"]:
        if tbl["name"] == "evt_churn":
            tbl["columns"].insert(3, {
                "name": "user_id",
                "dtype": "id",
                "source": "fk:dim_user.user_id",
            })
            tbl["foreign_keys"] = list(tbl.get("foreign_keys", []))
            tbl["foreign_keys"].insert(1, "dim_user.user_id")
            break
    else:
        raise AssertionError("evt_churn not found in saas template")
    for ent in data["entities"]:
        ent["archetype"] = "rocket_then_cliff"


def test_threshold_event_sub_entity_fk_distributes_uniformly(tmp_path: Path):
    """Each parent's threshold events must spread across its sub-entity
    candidates, not collapse to the first row.

    Construction: 3 parent entities (saas) × 30 seeds = up to 90 events.
    Per parent, observe which user_id was picked at each seed; assert
    distinct picks > 1 (a single distinct value across 30 seeds is the
    pre-fix signature). Chi-squared accepts uniform when
    ``len(unique) > 1`` is hit.
    """
    data = _load_yaml(SAAS_YAML)
    _mutate_evt_churn_to_fk_dim_user(data)
    yaml_path = _write_yaml(data, tmp_path / "saas_subentity_evt_churn.yaml")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SurrogateKeyWarning)
        cfg = load_config(yaml_path)

    # Build the (parent → users) candidate map once.
    rng0 = np.random.default_rng(0)
    tables0 = generate_tables(cfg, rng0)
    users = tables0["dim_user"]
    parent_to_user_ids: dict[str, list[str]] = {
        company_id: users[users["company_id"] == company_id]["user_id"].tolist()
        for company_id in users["company_id"].unique()
    }

    # Per parent, collect which user_id was picked at each seed.
    picks_by_parent: dict[str, list[str]] = {p: [] for p in parent_to_user_ids}

    for seed in range(30):
        rng = np.random.default_rng(seed)
        tables = generate_tables(cfg, rng)
        evt = tables["evt_churn"]
        for _, row in evt.iterrows():
            parent = row["company_id"]
            picks_by_parent[parent].append(row["user_id"])

    # Sanity: at least one parent fired enough events for the test to mean
    # anything. With rocket_then_cliff on every entity, all three parents
    # should fire on every seed.
    eligible = [
        (p, picks) for p, picks in picks_by_parent.items()
        if len(picks) >= 30 and len(parent_to_user_ids[p]) >= 5
    ]
    assert eligible, (
        f"F1 test setup failure: no parent had >=30 events with >=5 candidate "
        f"sub-entities. Picks per parent: "
        f"{ {p: len(picks) for p, picks in picks_by_parent.items()} }; "
        f"candidates per parent: "
        f"{ {p: len(c) for p, c in parent_to_user_ids.items()} }"
    )

    for parent, picks in eligible:
        unique = set(picks)
        assert len(unique) > 1, (
            f"F1 regression: parent {parent!r} picked the same sub-entity "
            f"({next(iter(unique))!r}) for all {len(picks)} threshold events "
            f"across seeds, despite having {len(parent_to_user_ids[parent])} "
            f"candidate users. _build_threshold_event is passing rng=None "
            f"into _resolve_event_row."
        )
