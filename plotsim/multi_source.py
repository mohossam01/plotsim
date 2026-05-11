"""plotsim.multi_source — drift application for multi-source / overlap mode.

What it does:
    Given the canonical ``dim_<entity>`` DataFrame produced by
    ``plotsim.dimensions.build_dim_entity`` and a
    :class:`~plotsim.config.SourceDeclaration`, produce one per-source dim
    DataFrame with three flavors of drift applied:

      * **Name drift** — for ``name_drift_rate`` fraction of entities, one
        ``faker.{name, first_name, last_name, company}`` column gets a typo
        applied (adjacent-character swap, full-string casing flip, or
        initials-style abbreviation).

      * **Key scheme drift** — the canonical PK column is renamed to
        ``<entity_type>_id_<source>`` and repopulated with IDs in the
        source's declared scheme (``prefix_padded``, ``numeric``, or
        ``uuid_short``). The canonical PK is preserved on the canonical
        dim so the manifest's mapping records can bridge the two.

      * **Attribute drift** — for ``attribute_drift_rate`` fraction of
        entities, a single eligible string-typed column (non-PK, non-FK,
        non-name-drift) gets a deterministic conflicting value appended.

    The module is pure: no filesystem access, no global state, all
    randomness flows through the caller-supplied per-source
    ``numpy.random.Generator``. Determinism contract: same canonical
    DataFrame + same RNG state + same source declaration → byte-identical
    drifted DataFrame and identical mapping records.

Input:
    The canonical per_entity dim's DataFrame, its declared columns, its
    PK column name, the active :class:`SourceDeclaration`, the entity-type
    string (used as the prefix for ``prefix_padded`` IDs and the suffix
    for the renamed PK column), and the per-source RNG.

Output:
    A 3-tuple ``(drifted_df, source_pk_column, mapping_records)``:
      * ``drifted_df`` — the new DataFrame to emit as
        ``dim_<entity>_<source>``.
      * ``source_pk_column`` — the renamed PK column on that DataFrame.
      * ``mapping_records`` — one dict per row carrying
        ``canonical_entity_id`` / ``source_entity_id`` / ``drifted_fields``;
        the orchestrator stitches entity-name and source-name onto each
        record before they reach the manifest.

Design note (per-source RNG seeding): callers in ``dimensions.py`` derive
each per-source RNG by drawing one ``integers`` value from the dim-build
RNG in config-declaration order (same pattern as quality-issue
``seed_offset`` handlers). Avoiding ``hash(source.name)`` keeps the seeds
stable across processes — Python's ``hash`` is salted per interpreter
invocation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from plotsim.config import (
    Column,
    FakerSource,
    FKSource,
    PKSource,
    SourceDeclaration,
    parse_source,
)


# Three drift flavors picked uniformly per drifted entity. Order is fixed so
# RNG-derived ``rng.integers(0, 3)`` always indexes the same kind.
_NAME_DRIFT_KINDS: tuple[str, ...] = ("swap", "casing", "abbreviate")

# Faker providers whose output we treat as the entity's natural-language
# "name" — the canonical drift target. First column on the dim whose
# source matches one of these is the name-drift column. If no column
# matches (e.g. the canonical dim only carries faker-generated industry /
# city / etc.), name drift no-ops cleanly.
_NAME_FAKER_METHODS: frozenset[str] = frozenset({"name", "first_name", "last_name", "company"})


def _is_name_column(col: Column) -> bool:
    """True when ``col`` is a ``generated:faker.{name-like}`` column."""
    parsed = parse_source(col.source)
    if not isinstance(parsed, FakerSource):
        return False
    return parsed.method in _NAME_FAKER_METHODS


def _is_attribute_drift_column(col: Column, name_column_names: set[str]) -> bool:
    """True when ``col`` is eligible to carry an attribute conflict.

    Eligible: dtype=string, source is NOT PK or FK, and the column is not
    already the name-drift target. We pick a string column so the
    "appended-suffix" stand-in stays readable; numeric attribute drift
    would need a different rewrite rule and is out of scope for M13.
    """
    if col.dtype != "string":
        return False
    if col.name in name_column_names:
        return False
    parsed = parse_source(col.source)
    if isinstance(parsed, (PKSource, FKSource)):
        return False
    return True


def _apply_name_drift(value: str, kind: str, rng: np.random.Generator) -> str:
    """Apply one of three drift kinds to ``value``.

    Cases:
      * ``swap`` — pick a random index ``i`` in ``[0, len-1)`` and swap
        ``value[i]`` with ``value[i+1]``. Mimics keyboard typos.
      * ``casing`` — invert the case of every alphabetic character.
        Mimics systems that lower-case everything on ingest, or upper-
        case it.
      * ``abbreviate`` — split on whitespace, take the first character of
        each token, join with dots, and append a trailing dot. Mimics
        manual data entry shortcuts ("Acme Industries Inc." → "A.I.I.").

    Edge cases: an empty string or a single-character ``swap`` is a
    no-op — the function returns the input unchanged so an empty/short
    canonical value doesn't crash drift application.
    """
    if not value:
        return value
    if kind == "swap":
        if len(value) < 2:
            return value
        idx = int(rng.integers(0, len(value) - 1))
        chars = list(value)
        chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
        return "".join(chars)
    if kind == "casing":
        return "".join(c.lower() if c.isupper() else c.upper() for c in value)
    if kind == "abbreviate":
        parts = value.split()
        if not parts:
            return value
        return ".".join(part[:1] for part in parts) + "."
    raise ValueError(f"unknown name-drift kind {kind!r}")


def _generate_source_ids(
    scheme: str,
    entity_type: str,
    n_rows: int,
    rng: np.random.Generator,
) -> list[str]:
    """Generate ``n_rows`` per-source IDs in the named ``scheme``.

    * ``prefix_padded`` — ``<UPPER_ENTITY>-NNN`` (e.g. ``COMPANY-001``).
      Width is ``max(3, len(str(n_rows)))`` so a 1,000-row dim widens to
      4 digits cleanly. No RNG consumption.
    * ``numeric`` — sequential integers starting from a per-source random
      offset in ``[1000, 10000)``. Mimics a billing system where the ID
      space starts somewhere arbitrary. Consumes one batched draw.
    * ``uuid_short`` — five-character lowercase hex per row. Consumes
      one ``rng.integers`` draw per row.
    """
    if scheme == "prefix_padded":
        prefix = entity_type.upper()
        width = max(3, len(str(max(n_rows, 1))))
        return [f"{prefix}-{i + 1:0{width}d}" for i in range(n_rows)]
    if scheme == "numeric":
        if n_rows == 0:
            return []
        start = int(rng.integers(1000, 10000))
        return [str(start + i) for i in range(n_rows)]
    if scheme == "uuid_short":
        if n_rows == 0:
            return []
        draws = rng.integers(0, 16**5, size=n_rows)
        return [f"{int(d):05x}" for d in draws]
    raise ValueError(f"unknown key_scheme {scheme!r}")


def _draw_drifted_indices(
    rate: float,
    n_rows: int,
    rng: np.random.Generator,
) -> list[int]:
    """Pick ``round(rate * n_rows)`` row indices uniformly without replacement.

    Returns a sorted list so downstream iteration order is deterministic
    (matters: subsequent per-row RNG draws walk these indices in order,
    and changing the iteration order would shift the RNG stream).
    """
    if rate <= 0.0 or n_rows == 0:
        return []
    k = min(n_rows, int(round(rate * n_rows)))
    if k <= 0:
        return []
    chosen = rng.choice(n_rows, size=k, replace=False)
    return sorted(int(i) for i in chosen)


def apply_source_drift(
    canonical_df: pd.DataFrame,
    canonical_columns: list[Column],
    canonical_pk_column: str,
    source: SourceDeclaration,
    entity_type: str,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, str, list[dict[str, Any]]]:
    """Build the drifted ``dim_<entity>_<source>`` frame plus per-row mappings.

    Returns ``(drifted_df, source_pk_column, mapping_records)``:

      * ``drifted_df`` — deep copy of ``canonical_df`` with name / attribute
        drift applied at the configured rates and the PK column renamed +
        repopulated under the source's ``key_scheme``.
      * ``source_pk_column`` — the new PK column name
        (``<entity_type>_id_<source>``). The canonical PK column has been
        removed from ``drifted_df``.
      * ``mapping_records`` — one dict per row carrying
        ``canonical_entity_id`` (the canonical PK value before renaming),
        ``source_entity_id`` (the new per-source PK), and
        ``drifted_fields`` (the canonical column names that received any
        drift — may be empty for un-drifted rows). The orchestrator adds
        ``entity`` (config entity name) and ``source`` (source name)
        before manifest emission.

    RNG consumption order (load-bearing — changing this changes byte
    output for the same seed):

      1. name-drift index draw (``rng.choice``)
      2. name-drift kind picks (``rng.integers``, batched)
      3. per-row name-drift RNG calls (only for ``swap`` rows)
      4. attribute-drift index draw (``rng.choice``)
      5. attribute-drift suffix draws (``rng.integers``, batched)
      6. ``_generate_source_ids`` draws (zero for prefix_padded; one
         batched draw for numeric; n_rows draws for uuid_short)
    """
    n_rows = len(canonical_df)
    drifted = canonical_df.copy(deep=True)

    name_columns = [c for c in canonical_columns if _is_name_column(c)]
    name_col_name = name_columns[0].name if name_columns else None
    name_column_names = {c.name for c in name_columns}

    attr_columns = [
        c for c in canonical_columns if _is_attribute_drift_column(c, name_column_names)
    ]
    attr_col_name = attr_columns[0].name if attr_columns else None

    drifted_fields_per_row: list[list[str]] = [[] for _ in range(n_rows)]

    # --- Name drift ---------------------------------------------------------
    if name_col_name is not None and source.name_drift_rate > 0.0:
        name_indices = _draw_drifted_indices(source.name_drift_rate, n_rows, rng)
        if name_indices:
            kind_picks = rng.integers(0, len(_NAME_DRIFT_KINDS), size=len(name_indices))
            name_col_loc = drifted.columns.get_loc(name_col_name)
            for row_idx, kind_idx in zip(name_indices, kind_picks):
                kind = _NAME_DRIFT_KINDS[int(kind_idx)]
                original = str(drifted.iat[row_idx, name_col_loc])
                drifted.iat[row_idx, name_col_loc] = _apply_name_drift(original, kind, rng)
                drifted_fields_per_row[row_idx].append(name_col_name)

    # --- Attribute drift ----------------------------------------------------
    if attr_col_name is not None and source.attribute_drift_rate > 0.0:
        attr_indices = _draw_drifted_indices(source.attribute_drift_rate, n_rows, rng)
        if attr_indices:
            suffixes = rng.integers(1, 1000, size=len(attr_indices))
            attr_col_loc = drifted.columns.get_loc(attr_col_name)
            for row_idx, suffix in zip(attr_indices, suffixes):
                original = str(drifted.iat[row_idx, attr_col_loc])
                drifted.iat[row_idx, attr_col_loc] = f"{original} (alt-{int(suffix)})"
                drifted_fields_per_row[row_idx].append(attr_col_name)

    # --- PK rename + key-scheme repopulation --------------------------------
    canonical_pk_values = [str(v) for v in canonical_df[canonical_pk_column].tolist()]
    source_pk_column = f"{entity_type}_id_{source.name}"
    new_ids = _generate_source_ids(source.key_scheme, entity_type, n_rows, rng)
    drifted = drifted.drop(columns=[canonical_pk_column])
    # Insert the new PK at position 0 so the per-source dim's column
    # order mirrors the canonical dim (PK first, everything else after).
    drifted.insert(0, source_pk_column, new_ids)

    mapping_records: list[dict[str, Any]] = []
    for row_idx, (canonical_pk, source_pk) in enumerate(zip(canonical_pk_values, new_ids)):
        mapping_records.append(
            {
                "canonical_entity_id": canonical_pk,
                "source_entity_id": source_pk,
                "drifted_fields": list(drifted_fields_per_row[row_idx]),
            }
        )
    return drifted, source_pk_column, mapping_records


__all__ = [
    "apply_source_drift",
]
