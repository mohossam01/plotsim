# Geo hierarchy

Some tables need locations that *make sense*. A row that says
`country=Italy` and `city=San Francisco` will not survive contact
with anyone who looks at the data. plotsim's geo bundle provider
solves this with a curated reference dataset and row-level
coherence: pick once, fill many.

## How it works

The provider exposes six `geo.<field>` column types — `country`,
`country_code`, `region`, `city`, `postcode`, `latitude`,
`longitude` — backed by a hand-curated dataset of 200 locations
across 17 countries. When a dim table declares any
`generated:geo.<field>` column, the engine pre-allocates **one
bundle entry per row** and reads every geo column from that single
entry. So `country` and `city` on the same row are always
consistent: the city actually is in the stated country, the
postcode looks like a real postcode for that country, and the
latitude/longitude land on the named city.

The dataset lives at `plotsim.data.GEO_LOCATIONS` (a tuple of dicts,
checked at import time so a typo can never sneak in). Determinism
is the same as every other random column: same seed → same bundle
assignments. The provider draws via `numpy.random.Generator.integers`,
so a single rng call shapes the whole table's geography.

Geo data is **dim-only**. Fact and event tables intentionally
reject `generated:geo.<field>` — geography is reference metadata
that joins back to a dim, not something that varies per period.
The engine design also keeps geography out of metric generation:
metrics do not vary by country.

## Quickstart — builder

The plain-language column type is `geo.<field>`:

```python
from plotsim import create

config = create(
    about="Multi-region retail demo",
    unit="store",
    seed=2026,
    window=("2024-01", "2024-12", "monthly"),
    metrics=[
        {"name": "footfall", "label": "Visitors", "type": "count", "polarity": "positive"},
    ],
    segments=[
        {"name": "stores", "count": 30, "archetype": "growth", "label": "Stores"},
    ],
    dimensions=[
        {"name": "dim_date", "per": "period", "columns": [
            {"name": "date_key", "type": "id"},
            {"name": "date", "type": "date"},
            {"name": "year", "type": "int"},
            {"name": "month", "type": "int"},
            {"name": "quarter", "type": "int"},
        ]},
        {"name": "dim_store", "per": "unit", "columns": [
            {"name": "store_id", "type": "id"},
            {"name": "store_name", "type": "faker.company"},
            {"name": "country", "type": "geo.country"},
            {"name": "country_code", "type": "geo.country_code"},
            {"name": "region", "type": "geo.region"},
            {"name": "city", "type": "geo.city"},
            {"name": "postcode", "type": "geo.postcode"},
            {"name": "latitude", "type": "geo.latitude"},
            {"name": "longitude", "type": "geo.longitude"},
        ]},
    ],
    facts=[
        {"name": "fct_footfall", "columns": [
            {"name": "date_key", "type": "ref.dim_date"},
            {"name": "store_id", "type": "ref.dim_store"},
            {"name": "visitors", "type": "metric.footfall"},
        ]},
    ],
)
```

Run that and `dim_store` will look like this:

| store_id | store_name | country         | region                | city         | postcode | latitude  | longitude  |
|----------|------------|-----------------|-----------------------|--------------|----------|-----------|------------|
| s-001    | Acme Corp  | United Kingdom  | England               | London       | EC1A 1BB | 51.50735  |   -0.12776 |
| s-002    | Globex Inc | United States   | New York              | Buffalo      | 14202    | 42.88645  |  -78.87837 |
| s-003    | Foo Ltd    | Australia       | New South Wales       | Newcastle    | 2300     | -32.92831 |  151.78228 |
| s-004    | Bar AG     | France          | Auvergne-Rhône-Alpes  | Grenoble     | 38000    | 45.18847  |    5.72452 |

## Quickstart — engine config

In an engine YAML, the source string is `generated:geo.<field>`:

```yaml
- name: dim_store
  type: dim
  grain: per_entity
  primary_key: store_id
  columns:
    - {name: store_id,     dtype: id,     source: pk}
    - {name: country,      dtype: string, source: "generated:geo.country"}
    - {name: country_code, dtype: string, source: "generated:geo.country_code"}
    - {name: region,       dtype: string, source: "generated:geo.region"}
    - {name: city,         dtype: string, source: "generated:geo.city"}
    - {name: postcode,     dtype: string, source: "generated:geo.postcode"}
    - {name: latitude,     dtype: float,  source: "generated:geo.latitude"}
    - {name: longitude,    dtype: float,  source: "generated:geo.longitude"}
```

`latitude` and `longitude` are `dtype: float`; the rest are
`dtype: string`. Postcode is a string because real postcodes
include letters and spaces (UK `EC1A 1BB`, Canada `M5H 2N2`,
Netherlands `1011 AC`).

## A single geo column also works

You don't have to declare all six. A dim that only needs a city
can skip the rest:

```yaml
columns:
  - {name: location_id, dtype: id,     source: pk}
  - {name: city,        dtype: string, source: "generated:geo.city"}
```

The engine still pre-allocates a row-level bundle and reads
`bundle["city"]` on each row. The other fields are simply unused
for that table.

## Independence from `faker.city`

`generated:faker.city` and `generated:geo.city` are independent
draws. Faker pulls from the locale's city pool (every English
village under the sun); the geo provider pulls from
`GEO_LOCATIONS`. The two can coexist on the same row — same dim
row will show a Faker city next to a geo city, and they don't
need to agree, because they answer different questions ("a
plausible-looking city name" vs "a real city we have lat/lng
for").

## Bundled template

`plotsim run geo_retail` generates a 40-store retail chain with
the full geo hierarchy. The template is paired:

- `plotsim/configs/templates/geo_retail.py` — builder surface
- `plotsim/configs/templates/geo_retail.yaml` — engine surface

Both produce identical tables given the same seed.

## Reference dataset

The 200-entry dataset covers 17 countries: United States, Canada,
United Kingdom, Germany, France, Spain, Italy, Netherlands,
Sweden, Australia, Japan, Singapore, Brazil, Mexico, South
Africa, United Arab Emirates, India. Each entry pairs a city with
its country, country_code (ISO 3166-1 alpha-2), region (state /
province / planning area / equivalent), postcode, and decimal
latitude/longitude.

The postcodes were chosen to look like a plausible postcode
*format* for each country. They are not guaranteed to be the
exact code the local post office uses for that street — they are
realistic examples, not survey data. If you need exhaustive
coverage or canonical postcodes, this isn't the right tool;
plotsim deliberately ships a curated demo dataset rather than
calling a geocoding API.

To inspect or filter the bundle in code:

```python
from plotsim.data import GEO_LOCATIONS, GEO_BUNDLE_FIELDS

# All fields a `geo.<field>` column can reference:
print(sorted(GEO_BUNDLE_FIELDS))
# ['city', 'country', 'country_code', 'latitude', 'longitude', 'postcode', 'region']

# All Italian cities in the dataset:
italy = [e for e in GEO_LOCATIONS if e["country_code"] == "IT"]
print(len(italy), "Italian cities")
```

## Gotchas

- **Duplicates within a table are expected.** With ~200 dataset
  entries and (say) 40 dim rows, some city will repeat. Two rows
  pointing at the same city carry identical lat/lng, postcode, and
  region — that's the bundle being coherent, not a bug. If you
  need every row to have a unique location, sample without
  replacement at the dim layer (this is not currently a config
  switch — open an issue if you need it).
- **Geo on facts/events is rejected.** The fact-side dispatchers
  only know `timestamp`, `date_key`, and `period_label` for
  `generated:` columns. If you put `generated:geo.city` on a fact,
  you get a clear `unsupported generated provider 'geo.city' on
  fact/event tables` error. Put geography on a dim and join.
- **Only the listed fields are valid.** `generated:geo.planet`
  raises at config-parse time, not at row-write time, so typos
  surface fast. The valid set is `country`, `country_code`,
  `region`, `city`, `postcode`, `latitude`, `longitude`.
