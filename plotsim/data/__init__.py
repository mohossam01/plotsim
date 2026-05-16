"""plotsim.data — read-only reference datasets bundled with the engine.

Modules here expose static reference data (geographic locations, currency
codes, etc.) used by dimension builders. Nothing in this package generates
random values or holds runtime state; importing it is a load of frozen
constants.
"""

from plotsim.data.geo_locations import (
    GEO_BUNDLE_FIELDS,
    GEO_LOCATIONS,
    GeoEntry,
)

__all__ = ["GEO_BUNDLE_FIELDS", "GEO_LOCATIONS", "GeoEntry"]
