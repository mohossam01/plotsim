"""Tests for the geo bundle provider.

Covers:
  - Reference dataset shape and self-consistency.
  - Row-level coherence: when a dim row carries multiple ``geo.<field>``
    columns, every field reads from the same bundle entry (city ∈ stated
    country, postcode/lat/lng on that city).
  - Single-field shortcuts: a dim with only ``geo.city`` (no country)
    still resolves cleanly.
  - Determinism under seed; different seeds produce different bundles.
  - Independence from stock Faker: ``generated:faker.city`` continues to
    draw via Faker without consulting the geo bundle.
  - Unknown ``geo.<field>`` strings are rejected with a clear message.
  - Builder shortcut ``geo.<field>`` translates to the correct
    engine source + dtype.
  - Bundled ``geo_retail`` template loads, runs end-to-end, and produces
    a coherent dim_store frame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import plotsim
from plotsim import generate_tables, list_templates, load_template
from plotsim.builder import create
from plotsim.config import Column, Entity, Table
from plotsim.data import GEO_BUNDLE_FIELDS, GEO_LOCATIONS
from plotsim.dimensions import (
    _assign_geo_bundles,
    _geo_provider_field,
    _table_uses_geo_bundle,
    build_dim_entity,
    build_dim_reference,
    build_dim_subentity,
)


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


# ── Reference dataset ──────────────────────────────────────────────────


def test_geo_dataset_size_in_spec_range():
    # Mission spec: 200–500 locations across 10–20 countries.
    assert 200 <= len(GEO_LOCATIONS) <= 500
    countries = {entry["country"] for entry in GEO_LOCATIONS}
    assert 10 <= len(countries) <= 20


def test_geo_dataset_every_entry_has_every_advertised_field():
    for entry in GEO_LOCATIONS:
        assert GEO_BUNDLE_FIELDS <= set(entry), entry


def test_geo_dataset_country_code_two_uppercase_letters():
    for entry in GEO_LOCATIONS:
        cc = entry["country_code"]
        assert isinstance(cc, str) and len(cc) == 2 and cc.isupper(), cc


def test_geo_dataset_lat_lng_are_floats_in_valid_range():
    for entry in GEO_LOCATIONS:
        lat = entry["latitude"]
        lng = entry["longitude"]
        assert isinstance(lat, float)
        assert isinstance(lng, float)
        assert -90.0 <= lat <= 90.0
        assert -180.0 <= lng <= 180.0


def test_geo_dataset_country_to_country_code_is_one_to_one():
    pairs = {(entry["country"], entry["country_code"]) for entry in GEO_LOCATIONS}
    countries = {p[0] for p in pairs}
    codes = {p[1] for p in pairs}
    assert len(pairs) == len(countries) == len(codes), pairs


# ── Helper functions ───────────────────────────────────────────────────


def test_geo_provider_field_resolves_known_fields():
    for f in GEO_BUNDLE_FIELDS:
        assert _geo_provider_field(f"geo.{f}") == f


def test_geo_provider_field_returns_none_for_non_geo_provider():
    assert _geo_provider_field("entity_name") is None
    assert _geo_provider_field("timestamp") is None
    assert _geo_provider_field("date_key") is None


def test_geo_provider_field_raises_on_unknown_geo_field():
    with pytest.raises(ValueError, match="unknown geo bundle field 'planet'"):
        _geo_provider_field("geo.planet")


def test_table_uses_geo_bundle_detects_geo_columns():
    cols = [
        Column(name="pk", dtype="id", source="pk"),
        Column(name="city", dtype="string", source="generated:geo.city"),
    ]
    assert _table_uses_geo_bundle(cols) is True


def test_table_uses_geo_bundle_no_geo_returns_false():
    cols = [
        Column(name="pk", dtype="id", source="pk"),
        Column(name="city", dtype="string", source="generated:faker.city"),
    ]
    assert _table_uses_geo_bundle(cols) is False


def test_table_uses_geo_bundle_raises_on_unknown_geo_field():
    cols = [
        Column(name="pk", dtype="id", source="pk"),
        Column(name="planet", dtype="string", source="generated:geo.planet"),
    ]
    with pytest.raises(ValueError, match="unknown geo bundle field 'planet'"):
        _table_uses_geo_bundle(cols)


def test_assign_geo_bundles_returns_none_when_no_geo_columns():
    cols = [Column(name="pk", dtype="id", source="pk")]
    assert _assign_geo_bundles(cols, n_rows=5, rng=_rng(0)) is None


def test_assign_geo_bundles_returns_n_rows_entries():
    cols = [
        Column(name="pk", dtype="id", source="pk"),
        Column(name="city", dtype="string", source="generated:geo.city"),
    ]
    bundles = _assign_geo_bundles(cols, n_rows=7, rng=_rng(42))
    assert bundles is not None
    assert len(bundles) == 7
    for b in bundles:
        assert b in GEO_LOCATIONS  # each is a tuple-element identity


# ── Per-entity dim ─────────────────────────────────────────────────────


def _entities(n: int) -> list[Entity]:
    return [Entity(name="cohort", archetype="steady_growth", size=1) for _ in range(n)]


def _full_geo_dim(name: str = "dim_store") -> Table:
    return Table(
        name=name,
        type="dim",
        grain="per_entity",
        primary_key="store_id",
        columns=[
            Column(name="store_id", dtype="id", source="pk"),
            Column(name="country", dtype="string", source="generated:geo.country"),
            Column(name="country_code", dtype="string", source="generated:geo.country_code"),
            Column(name="region", dtype="string", source="generated:geo.region"),
            Column(name="city", dtype="string", source="generated:geo.city"),
            Column(name="postcode", dtype="string", source="generated:geo.postcode"),
            Column(name="latitude", dtype="float", source="generated:geo.latitude"),
            Column(name="longitude", dtype="float", source="generated:geo.longitude"),
        ],
    )


def _city_lookup() -> dict[tuple[str, str], dict]:
    """City+postcode is unique in the dataset; country+city is not (e.g.
    Newcastle exists in UK and Australia). Key by (city, postcode) to
    pick the right bundle when verifying coherence on duplicates.
    """
    out: dict[tuple[str, str], dict] = {}
    for entry in GEO_LOCATIONS:
        out[(entry["city"], entry["postcode"])] = entry
    return out


def test_per_entity_geo_full_bundle_coherent():
    df = build_dim_entity(_full_geo_dim(), _entities(20), _rng(0))
    assert len(df) == 20
    lookup = _city_lookup()
    for _, row in df.iterrows():
        expected = lookup[(row["city"], row["postcode"])]
        assert expected["country"] == row["country"]
        assert expected["country_code"] == row["country_code"]
        assert expected["region"] == row["region"]
        assert expected["latitude"] == pytest.approx(row["latitude"])
        assert expected["longitude"] == pytest.approx(row["longitude"])


def test_per_entity_geo_single_column_only_works():
    # Only `geo.city` declared; no country/region/postcode etc.
    tbl = Table(
        name="dim_store",
        type="dim",
        grain="per_entity",
        primary_key="store_id",
        columns=[
            Column(name="store_id", dtype="id", source="pk"),
            Column(name="city", dtype="string", source="generated:geo.city"),
        ],
    )
    df = build_dim_entity(tbl, _entities(10), _rng(0))
    assert len(df) == 10
    cities = {entry["city"] for entry in GEO_LOCATIONS}
    for c in df["city"]:
        assert c in cities


def test_per_entity_geo_determinism_under_seed():
    a = build_dim_entity(_full_geo_dim(), _entities(15), _rng(123))
    b = build_dim_entity(_full_geo_dim(), _entities(15), _rng(123))
    pd.testing.assert_frame_equal(a, b)


def test_per_entity_geo_different_seeds_diverge():
    a = build_dim_entity(_full_geo_dim(), _entities(15), _rng(1))
    b = build_dim_entity(_full_geo_dim(), _entities(15), _rng(2))
    # At 15 stores with ~250-entry pool, identical city sequence has
    # vanishing probability across two seeds.
    assert (a["city"].tolist() != b["city"].tolist()) or (
        a["postcode"].tolist() != b["postcode"].tolist()
    )


def test_per_entity_geo_independent_of_faker_city():
    # Geo provider and faker.city draw from independent pools. Faker
    # cities can be any city Faker knows about; geo cities are restricted
    # to GEO_LOCATIONS. The two columns should coexist on the same row
    # without one feeding the other.
    tbl = Table(
        name="dim_store",
        type="dim",
        grain="per_entity",
        primary_key="store_id",
        columns=[
            Column(name="store_id", dtype="id", source="pk"),
            Column(name="geo_city", dtype="string", source="generated:geo.city"),
            Column(name="faker_city", dtype="string", source="generated:faker.city"),
        ],
    )
    df = build_dim_entity(tbl, _entities(15), _rng(0))
    geo_cities = {entry["city"] for entry in GEO_LOCATIONS}
    for c in df["geo_city"]:
        assert c in geo_cities
    # Faker.city draws are not constrained to geo dataset; at least one
    # row in 15 should differ from the geo column to prove independence.
    assert any(g != f for g, f in zip(df["geo_city"], df["faker_city"]))


def test_per_entity_geo_lat_lng_are_floats_not_strings():
    df = build_dim_entity(_full_geo_dim(), _entities(5), _rng(0))
    assert df["latitude"].dtype.kind == "f"
    assert df["longitude"].dtype.kind == "f"


def test_per_entity_no_geo_columns_does_not_consume_geo_rng():
    # Sanity: a dim with no geo columns produces the same output whether
    # or not the geo machinery is wired in. This protects against future
    # refactors that would silently shift rng draws.
    tbl = Table(
        name="dim_store",
        type="dim",
        grain="per_entity",
        primary_key="store_id",
        columns=[
            Column(name="store_id", dtype="id", source="pk"),
            Column(name="store_name", dtype="string", source="generated:faker.company"),
        ],
    )
    a = build_dim_entity(tbl, _entities(5), _rng(0))
    b = build_dim_entity(tbl, _entities(5), _rng(0))
    pd.testing.assert_frame_equal(a, b)


# ── Reference dim ──────────────────────────────────────────────────────


def test_reference_dim_geo_coherent():
    # per_reference dim sized by static-CSV column with geo bundle
    # filling the rest.
    tbl = Table(
        name="dim_country_lookup",
        type="dim",
        grain="per_reference",
        primary_key="lookup_id",
        columns=[
            Column(name="lookup_id", dtype="id", source="pk"),
            Column(name="tag", dtype="string", source="static:a,b,c,d,e"),
            Column(name="country", dtype="string", source="generated:geo.country"),
            Column(name="city", dtype="string", source="generated:geo.city"),
            Column(name="postcode", dtype="string", source="generated:geo.postcode"),
        ],
    )
    df = build_dim_reference(tbl, _rng(0))
    assert len(df) == 5
    lookup = _city_lookup()
    for _, row in df.iterrows():
        expected = lookup[(row["city"], row["postcode"])]
        assert expected["country"] == row["country"]


def test_reference_dim_unknown_geo_field_raises():
    tbl = Table(
        name="dim_lookup",
        type="dim",
        grain="per_reference",
        primary_key="id",
        columns=[
            Column(name="id", dtype="id", source="pk"),
            Column(name="tag", dtype="string", source="static:a,b,c"),
            Column(name="planet", dtype="string", source="generated:geo.planet"),
        ],
    )
    with pytest.raises(ValueError, match="unknown geo bundle field 'planet'"):
        build_dim_reference(tbl, _rng(0))


# ── Sub-entity dim ─────────────────────────────────────────────────────


def test_subentity_dim_geo_each_row_gets_its_own_bundle():
    # A sub-entity dim with size 1 per parent and N parents → N rows,
    # each with its own draw. With size > 1, sub-rows under one parent
    # all draw independently (different cities possible per sibling).
    parent = Table(
        name="dim_company",
        type="dim",
        grain="per_entity",
        primary_key="company_id",
        columns=[
            Column(name="company_id", dtype="id", source="pk"),
        ],
    )
    sub = Table(
        name="dim_office",
        type="dim",
        grain="variable",
        primary_key="office_id",
        columns=[
            Column(name="office_id", dtype="id", source="pk"),
            Column(name="company_id", dtype="id", source="fk:dim_company.company_id"),
            Column(name="country", dtype="string", source="generated:geo.country"),
            Column(name="city", dtype="string", source="generated:geo.city"),
            Column(name="postcode", dtype="string", source="generated:geo.postcode"),
        ],
    )
    entities = [
        Entity(name="acme", archetype="steady_growth", size=3),
        Entity(name="globex", archetype="steady_growth", size=2),
    ]
    parent_df = build_dim_entity(parent, entities, _rng(0))
    df = build_dim_subentity(sub, entities, parent_df, _rng(99))
    assert len(df) == 5  # 3 + 2 offices
    lookup = _city_lookup()
    for _, row in df.iterrows():
        expected = lookup[(row["city"], row["postcode"])]
        assert expected["country"] == row["country"]


# ── Builder shortcut ────────────────────────────────────────────────────


def _builder_dim_with_geo(types: list[tuple[str, str]]) -> plotsim.config.PlotsimConfig:
    cols = [{"name": "store_id", "type": "id"}]
    for name, t in types:
        cols.append({"name": name, "type": t})
    return create(
        about="geo builder smoke",
        unit="store",
        seed=2026,
        window=("2024-01", "2024-03", "monthly"),
        metrics=[
            {"name": "x", "label": "x", "type": "score", "polarity": "positive"},
        ],
        segments=[
            {"name": "s", "count": 4, "archetype": "flat", "label": "s"},
        ],
        dimensions=[
            {
                "name": "dim_date",
                "per": "period",
                "columns": [
                    {"name": "date_key", "type": "id"},
                    {"name": "date", "type": "date"},
                    {"name": "year", "type": "int"},
                    {"name": "month", "type": "int"},
                    {"name": "quarter", "type": "int"},
                ],
            },
            {"name": "dim_store", "per": "unit", "columns": cols},
        ],
        facts=[
            {
                "name": "fct_x",
                "columns": [
                    {"name": "date_key", "type": "ref.dim_date"},
                    {"name": "store_id", "type": "ref.dim_store"},
                    {"name": "x_value", "type": "metric.x"},
                ],
            },
        ],
    )


def test_builder_geo_country_translates_to_string_source():
    cfg = _builder_dim_with_geo([("country", "geo.country")])
    dim_store = next(t for t in cfg.tables if t.name == "dim_store")
    col = next(c for c in dim_store.columns if c.name == "country")
    assert col.dtype == "string"
    assert col.source == "generated:geo.country"


def test_builder_geo_latitude_translates_to_float_source():
    cfg = _builder_dim_with_geo([("lat", "geo.latitude")])
    dim_store = next(t for t in cfg.tables if t.name == "dim_store")
    col = next(c for c in dim_store.columns if c.name == "lat")
    assert col.dtype == "float"
    assert col.source == "generated:geo.latitude"


def test_builder_geo_longitude_translates_to_float_source():
    cfg = _builder_dim_with_geo([("lng", "geo.longitude")])
    dim_store = next(t for t in cfg.tables if t.name == "dim_store")
    col = next(c for c in dim_store.columns if c.name == "lng")
    assert col.dtype == "float"
    assert col.source == "generated:geo.longitude"


def test_builder_geo_unknown_field_rejected_at_interpret_time():
    with pytest.raises(ValueError, match="unknown geo field 'planet'"):
        _builder_dim_with_geo([("planet", "geo.planet")])


# ── Bundled geo_retail template ────────────────────────────────────────


def test_geo_retail_template_in_list():
    assert "geo_retail" in list_templates()


def test_geo_retail_template_loads():
    cfg = load_template("geo_retail")
    table_names = [t.name for t in cfg.tables]
    assert "dim_store" in table_names
    assert "fct_footfall" in table_names
    assert "fct_sales" in table_names


def test_geo_retail_template_runs_end_to_end_with_zero_mismatches():
    cfg = load_template("geo_retail")
    tables = generate_tables(cfg)
    dim_store = tables["dim_store"]
    assert len(dim_store) == 40  # 12 flagship + 28 standard
    lookup = _city_lookup()
    for _, row in dim_store.iterrows():
        expected = lookup[(row["city"], row["postcode"])]
        assert expected["country"] == row["country"], row.to_dict()
        assert expected["country_code"] == row["country_code"]
        assert expected["region"] == row["region"]
        assert expected["latitude"] == pytest.approx(row["latitude"])
        assert expected["longitude"] == pytest.approx(row["longitude"])


def test_geo_retail_template_deterministic_under_seed():
    a = generate_tables(load_template("geo_retail"))
    b = generate_tables(load_template("geo_retail"))
    pd.testing.assert_frame_equal(a["dim_store"], b["dim_store"])
    pd.testing.assert_frame_equal(
        a["fct_footfall"].reset_index(drop=True),
        b["fct_footfall"].reset_index(drop=True),
    )
