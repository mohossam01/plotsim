"""Composite archetype DSL parser.

Grammar
-------
    spec      ::= shape ( ">" shape )* ( "@" period )*
    shape     ::= one of SHAPE_RECIPES keys
    period    ::= int in [1, n_periods - 1]

Rules
-----
- The number of "@" tokens MUST equal the number of ">" tokens. A spec with N
  shapes therefore carries N-1 transition periods. A single-shape spec carries
  zero "@" tokens and covers [0.0, 1.0].
- Periods are strictly ascending and each lies in [1, n_periods - 1].
- "+" is reserved for layered patterns and is rejected at parse time with a
  future-release message — sequential composition uses ">" only.
- Multi-segment shapes (e.g. ``spike_then_crash``) are rescaled into the phase
  window assigned to the shape, preserving the internal sub-segment ratios
  defined in ``SHAPE_RECIPES``.

Examples
--------
    parse_archetype("growth", 24)
        → 1 segment, sigmoid, start_pct=0.0 end_pct=1.0
    parse_archetype("flat > decline @ 12", 24)
        → 2 segments at [0.0, 0.5] and [0.5, 1.0]
    parse_archetype("growth > spike_then_crash > flat @ 8 @ 16", 24)
        → 5 segments (1 + 3 + 1) — spike_then_crash rescaled into [1/3, 2/3]
"""

from __future__ import annotations

from plotsim.config import CurveSegment

from .recipes import SHAPE_RECIPES, VALID_SHAPE_WORDS


class ArchetypeParseError(ValueError):
    """Raised when a composite archetype spec is malformed.

    Distinct subclass of ``ValueError`` so callers (UserInput validators)
    can catch it specifically without swallowing unrelated value errors.
    """


def parse_archetype(spec: str, n_periods: int) -> list[CurveSegment]:
    """Parse a composite archetype spec into a list of ``CurveSegment``s.

    Args:
        spec: DSL string like ``"growth > spike_then_crash > flat @ 8 @ 16"``.
        n_periods: total trajectory length in periods (used to convert
            absolute period indices in ``@ N`` clauses to fractional positions).

    Returns:
        A list of ``CurveSegment`` covering [0.0, 1.0] contiguously.

    Raises:
        ArchetypeParseError: on any structural or vocabulary violation.
    """
    if not isinstance(spec, str):
        raise ArchetypeParseError(f"archetype spec must be a string, got {type(spec).__name__}")

    raw = spec.strip()
    if not raw:
        raise ArchetypeParseError("archetype spec is empty")

    if "+" in raw:
        raise ArchetypeParseError(
            "Layered patterns ship in a future release. Use > for sequential composition."
        )

    if n_periods < 2:
        raise ArchetypeParseError(f"n_periods must be >= 2 to compose phases, got {n_periods}")

    # Split on "@" first → first chunk is the shape chain, rest are periods.
    parts = [p.strip() for p in raw.split("@")]
    shape_chain = parts[0]
    period_strs = parts[1:]

    shapes = [s.strip() for s in shape_chain.split(">")]
    if any(not s for s in shapes):
        raise ArchetypeParseError(
            f"archetype spec {spec!r} has an empty shape "
            f"(check for trailing '>' or doubled separators)"
        )

    n_shapes = len(shapes)
    n_periods_in_spec = len(period_strs)
    if n_periods_in_spec != n_shapes - 1:
        raise ArchetypeParseError(
            f"archetype spec {spec!r}: expected {n_shapes - 1} '@ N' "
            f"transition(s) for {n_shapes} shape(s) (one '@' between every "
            f"pair of '>'), got {n_periods_in_spec}"
        )

    unknown = [s for s in shapes if s not in VALID_SHAPE_WORDS]
    if unknown:
        raise ArchetypeParseError(
            f"archetype spec {spec!r} uses unknown shape word(s): "
            f"{sorted(set(unknown))}. Valid: {sorted(VALID_SHAPE_WORDS)}"
        )

    periods: list[int] = []
    for period_str in period_strs:
        if not period_str:
            raise ArchetypeParseError(f"archetype spec {spec!r} has an empty '@' value")
        try:
            periods.append(int(period_str))
        except ValueError:
            raise ArchetypeParseError(
                f"archetype spec {spec!r}: '@ {period_str}' is not an integer period"
            ) from None

    for p in periods:
        if not (1 <= p <= n_periods - 1):
            raise ArchetypeParseError(
                f"archetype spec {spec!r}: transition period {p} out of "
                f"range [1, {n_periods - 1}] for an {n_periods}-period window"
            )

    for prev, curr in zip(periods, periods[1:]):
        if curr <= prev:
            raise ArchetypeParseError(
                f"archetype spec {spec!r}: transition periods must be "
                f"strictly ascending, got {periods}"
            )

    # Phase windows in [0.0, 1.0]. n_shapes phases, n_shapes-1 boundaries.
    boundaries = [p / n_periods for p in periods]
    phase_starts = [0.0, *boundaries]
    phase_ends = [*boundaries, 1.0]

    segments: list[CurveSegment] = []
    for shape, ps, pe in zip(shapes, phase_starts, phase_ends):
        sub_segments = SHAPE_RECIPES[shape]
        width = pe - ps
        for curve, params, rel_start, rel_end in sub_segments:
            # Pin sub-segment endpoints to phase boundaries whenever the
            # relative position is exactly 0.0 or 1.0. Without pinning,
            # ``ps + 1.0 * (pe - ps)`` may not be bitwise equal to ``pe``,
            # breaking Archetype._segments_cover_full_range contiguity.
            seg_start = ps if rel_start == 0.0 else ps + rel_start * width
            seg_end = pe if rel_end == 1.0 else ps + rel_end * width
            segments.append(
                CurveSegment(
                    curve=curve,
                    params=dict(params),
                    start_pct=seg_start,
                    end_pct=seg_end,
                )
            )

    return segments
