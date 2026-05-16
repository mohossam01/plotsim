"""Multi-region retail chain — Python-shaped geo-hierarchy template.

This is the ``create(**kwargs)`` mirror of ``geo_retail.yaml`` — both
produce identical engine configs given the same seed. Pick whichever
surface fits your workflow:

* ``geo_retail.yaml`` for config-as-data fixtures checked into git
* this file for code-shaped configs that compose with regular Python

Showcase template for the geo bundle provider:

  * ``dim_store`` declares six ``geo.<field>`` columns. Each row's
    country / region / city / postcode / latitude / longitude resolve
    from a single bundle entry drawn from
    ``plotsim.data.GEO_LOCATIONS``, so the city is always in the
    stated country and the lat/lng land on the named city.
  * The two segments (``flagship`` and ``standard``) ride two
    different archetypes against the same store dim. Geo is purely
    locational metadata — metrics do not vary by geography per the
    M9 spec ("geo-aware metric generation" is out of scope).
  * Two facts (footfall + sales) pull from the per-store
    trajectory; pair them with the dim to slice sales by region.

Domain narrative: a multi-region retail chain runs 40 stores across
~15 countries. Flagship stores ride a steady-growth curve as the
brand expands; standard stores ride a flatter curve. Geo metadata is
deterministic under seed: same seed → same country/city bundle
assignments per store.
"""

from plotsim import create


config = create(
    about="Multi-region retail chain — geo hierarchy demo",
    unit="store",
    seed=2026,
    window=("2024-01", "2024-12", "monthly"),
    metrics=[
        {
            "name": "footfall",
            "label": "Visitors per period",
            "type": "count",
            "polarity": "positive",
        },
        {
            "name": "sales",
            "label": "Sales revenue per period",
            "type": "amount",
            "polarity": "positive",
            "range": [0, 500_000],
        },
    ],
    segments=[
        {
            "name": "flagship",
            "count": 12,
            "archetype": "growth",
            "label": "Flagship stores (large urban locations)",
        },
        {
            "name": "standard",
            "count": 28,
            "archetype": "flat",
            "label": "Standard stores (smaller regional outlets)",
        },
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
        {
            "name": "dim_store",
            "per": "unit",
            "columns": [
                {"name": "store_id", "type": "id"},
                {"name": "store_name", "type": "faker.company"},
                {"name": "country", "type": "geo.country"},
                {"name": "country_code", "type": "geo.country_code"},
                {"name": "region", "type": "geo.region"},
                {"name": "city", "type": "geo.city"},
                {"name": "postcode", "type": "geo.postcode"},
                {"name": "latitude", "type": "geo.latitude"},
                {"name": "longitude", "type": "geo.longitude"},
            ],
        },
    ],
    facts=[
        {
            "name": "fct_footfall",
            "columns": [
                {"name": "date_key", "type": "ref.dim_date"},
                {"name": "store_id", "type": "ref.dim_store"},
                {"name": "visitors", "type": "metric.footfall"},
            ],
        },
        {
            "name": "fct_sales",
            "columns": [
                {"name": "date_key", "type": "ref.dim_date"},
                {"name": "store_id", "type": "ref.dim_store"},
                {"name": "revenue", "type": "metric.sales"},
            ],
        },
    ],
)
