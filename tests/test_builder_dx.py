"""Mission 124 — Builder DX paper-cut regression tests.

One test per acceptance-criterion bullet:

  * `create(seed=N)` produces a config with `seed == N`.
  * `create_from_yaml` reads a top-level `seed:` key.
  * Declaring an explicit dim without `dim_date` no longer raises
    `KeyError: 'dim_date'`; the interpreter auto-prepends the dim.
  * A bridge that references the auto-generated `dim_{unit}` validates
    end-to-end (engine `PlotsimConfig` accepts the resulting tables).
  * Proportional events with a `count`-typed driver no longer raise
    `TypeError: ufunc 'isnan' not supported for input types`.

These tests live alongside the fix; they fail loudly if a future change
re-introduces any of the friction items.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from plotsim.builder import create, create_from_yaml
from plotsim.tables import generate_tables_with_state


# ── Shared fixtures ─────────────────────────────────────────────────────────


def _two_segment_kwargs(**overrides):
    """Minimum-viable two-segment config kwargs for the builder.

    Two segments dodges the "only one segment" warning; growth + decline
    is the smallest mix that exercises the trajectory engine.
    """
    base = dict(
        about="m124 fixture",
        unit="company",
        window=("2024-01", "2024-06"),
        metrics=[
            {"name": "engagement", "type": "score", "polarity": "positive"},
        ],
        segments=[
            {"name": "winners", "count": 3, "archetype": "growth"},
            {"name": "losers", "count": 3, "archetype": "decline"},
        ],
    )
    base.update(overrides)
    return base


# ── Fix 1: explicit seed kwarg ──────────────────────────────────────────────


def test_create_accepts_explicit_seed():
    """create(seed=42) must thread through to PlotsimConfig.seed."""
    cfg = create(**_two_segment_kwargs(seed=42))
    assert cfg.seed == 42


def test_create_seed_is_optional():
    """Omitting `seed` falls back to a random draw — non-zero, in range."""
    cfg = create(**_two_segment_kwargs())
    assert 0 <= cfg.seed < 2**32


def test_create_seed_pin_makes_runs_byte_identical():
    """Same seed in, same trajectories out (the determinism contract)."""
    cfg_a = create(**_two_segment_kwargs(seed=99))
    cfg_b = create(**_two_segment_kwargs(seed=99))
    rng_a = np.random.default_rng(cfg_a.seed)
    rng_b = np.random.default_rng(cfg_b.seed)
    tables_a, _ = generate_tables_with_state(cfg_a, rng_a)
    tables_b, _ = generate_tables_with_state(cfg_b, rng_b)
    # Compare the fact column cell-for-cell — the trajectory contract
    # bottoms out here.
    fct_a = tables_a["fct_company"]["engagement"].to_numpy()
    fct_b = tables_b["fct_company"]["engagement"].to_numpy()
    np.testing.assert_array_equal(fct_a, fct_b)


def test_create_from_yaml_reads_seed_key(tmp_path: Path):
    """A top-level ``seed:`` line in YAML lands on PlotsimConfig.seed."""
    cfg_yaml = """
about: yaml seed test
unit: company
window:
  start: 2024-01
  end: 2024-06
  every: monthly
seed: 1234
metrics:
  - name: engagement
    type: score
    polarity: positive
segments:
  - name: winners
    count: 3
    archetype: growth
  - name: losers
    count: 3
    archetype: decline
"""
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(cfg_yaml, encoding="utf-8")
    cfg = create_from_yaml(yaml_path)
    assert cfg.seed == 1234


def test_create_seed_must_be_non_negative():
    """Negative seeds violate Pydantic ge=0 — surface a clear field error."""
    with pytest.raises(Exception) as exc:  # ValidationError under pydantic
        create(**_two_segment_kwargs(seed=-1))
    assert "seed" in str(exc.value)


# ── Fix 2: dim_date auto-fill on explicit-schema configs ────────────────────


def test_explicit_dims_without_dim_date_does_not_raise():
    """Pre-M124: declaring any dim without dim_date raised KeyError.

    Now the interpreter auto-prepends dim_date. The resulting config has
    dim_date as a real Table, indexable from the engine layer.
    """
    cfg = create(
        **_two_segment_kwargs(
            facts=[{
                "name": "fct_company",
                "columns": [
                    {"name": "date_key", "type": "ref.dim_date"},
                    {"name": "company_id", "type": "ref.dim_company"},
                    {"name": "engagement", "type": "metric.engagement"},
                ],
            }],
            dimensions=[{
                "name": "dim_company",
                "columns": [
                    {"name": "company_id", "type": "id"},
                    {"name": "name", "type": "faker.company"},
                ],
            }],
        )
    )
    table_names = [t.name for t in cfg.tables]
    assert "dim_date" in table_names
    # And it lands FIRST so dim ordering matches the auto-schema branch.
    assert table_names[0] == "dim_date"


def test_explicit_dim_date_is_not_clobbered():
    """If the user declared dim_date themselves, the auto-fill stays out."""
    user_dim_date = {
        "name": "dim_date",
        "columns": [
            {"name": "date_key", "type": "id"},
            {"name": "year", "type": "int"},
        ],
        "per": "period",
    }
    cfg = create(
        **_two_segment_kwargs(
            facts=[{
                "name": "fct_company",
                "columns": [
                    {"name": "date_key", "type": "ref.dim_date"},
                    {"name": "company_id", "type": "ref.dim_company"},
                    {"name": "engagement", "type": "metric.engagement"},
                ],
            }],
            dimensions=[
                user_dim_date,
                {
                    "name": "dim_company",
                    "columns": [
                        {"name": "company_id", "type": "id"},
                        {"name": "name", "type": "faker.company"},
                    ],
                },
            ],
        )
    )
    dim_date_tables = [t for t in cfg.tables if t.name == "dim_date"]
    assert len(dim_date_tables) == 1, "user-declared dim_date should not be duplicated"
    # User declared two columns; the auto-fill version has five.
    assert len(dim_date_tables[0].columns) == 2


# ── Fix 3: bridge references auto-generated dim_{unit} ─────────────────────


def test_bridge_referencing_auto_unit_dim_validates_end_to_end():
    """Pre-M124: PlotsimConfig rejected the bridge because dim_company
    wasn't in tables. Now the interpreter auto-prepends it.
    """
    cfg = create(
        **_two_segment_kwargs(
            dimensions=[{
                "name": "dim_user",
                "columns": [
                    {"name": "user_id", "type": "id"},
                    {"name": "name", "type": "faker.name"},
                ],
            }],
            bridges=[{
                "name": "bridge_user_company",
                "left": "dim_user",
                "right": "dim_company",
                "cardinality": [1, 2],
            }],
        )
    )
    table_names = {t.name for t in cfg.tables}
    assert "dim_company" in table_names
    assert "dim_user" in table_names
    # And the bridge references both.
    assert cfg.bridges[0].connects == ["dim_user", "dim_company"]


def test_bridge_no_unit_reference_skips_auto_unit_dim():
    """Sanity: when no bridge references dim_{unit}, the auto-prepend
    doesn't fire — the user controls whether the unit dim exists.
    """
    cfg = create(
        **_two_segment_kwargs(
            dimensions=[
                {
                    "name": "dim_user",
                    "columns": [
                        {"name": "user_id", "type": "id"},
                        {"name": "name", "type": "faker.name"},
                    ],
                },
                {
                    "name": "dim_team",
                    "columns": [
                        {"name": "team_id", "type": "id"},
                        {"name": "name", "type": "faker.company"},
                    ],
                },
            ],
            bridges=[{
                "name": "bridge_user_team",
                "left": "dim_user",
                "right": "dim_team",
                "cardinality": [1, 2],
            }],
        )
    )
    table_names = {t.name for t in cfg.tables}
    assert "dim_company" not in table_names


# ── Fix 4: count-driver proportional events no longer crash ────────────────


def test_proportional_event_count_driver_does_not_raise():
    """Pre-M124: pd.to_numeric on a count column returns Int64; np.isnan
    on int64 raises TypeError. Now the event builder coerces to float64.
    """
    cfg = create(
        about="count-driver test",
        unit="company",
        window=("2024-01", "2024-06"),
        metrics=[
            {"name": "engagement", "type": "score", "polarity": "positive"},
            {"name": "tickets", "type": "count", "polarity": "negative"},
        ],
        segments=[
            {"name": "winners", "count": 3, "archetype": "growth"},
            {"name": "losers", "count": 3, "archetype": "decline"},
        ],
        facts=[{
            "name": "fct_company",
            "columns": [
                {"name": "date_key", "type": "ref.dim_date"},
                {"name": "company_id", "type": "ref.dim_company"},
                {"name": "engagement", "type": "metric.engagement"},
                {"name": "tickets", "type": "metric.tickets"},
            ],
        }],
        dimensions=[{
            "name": "dim_company",
            "columns": [
                {"name": "company_id", "type": "id"},
                {"name": "name", "type": "faker.company"},
            ],
        }],
        events=[{
            "name": "evt_actions",
            "trigger": "proportional",
            "driver": "tickets",
            "scale": 1.0,
            "columns": [
                {"name": "event_id", "type": "id"},
                {"name": "date_key", "type": "ref.dim_date"},
                {"name": "company_id", "type": "ref.dim_company"},
            ],
        }],
        seed=42,
    )
    rng = np.random.default_rng(cfg.seed)
    tables, _ = generate_tables_with_state(cfg, rng)
    assert "evt_actions" in tables
    # tickets is poisson → fct dtype is integer; the proportional event
    # still emits a positive number of rows.
    assert len(tables["evt_actions"]) > 0
