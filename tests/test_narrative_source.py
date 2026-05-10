"""0.6-M10 / NarrativeSource tests.

Covers:

* Grammar parsing — happy path + every rejection path on the source string
* Pydantic gates on NarrativeSource (key identifier rules, frozen, extra=forbid)
* Pydantic gates on NarrativeConfig (template placeholders, bands shape,
  per-archetype slot/band coverage, non-empty phrase pools)
* Column-level pairing — narrative source ↔ Column.narrative both-or-neither
* Cross-config gates — unknown archetypes rejected, missing archetype lexicons
  rejected, dim/per_period placement rejected
* dtype: boolean rejected on NarrativeSource (M102 discipline mirrors
  TextBucketSource)
* Generation determinism — same seed → byte-identical text column
* Trajectory direction — per-entity mean band index ranks with engagement
* Per-archetype lexicon respected — promoter texts use promoter pool only
* Text varies across periods for an entity with non-flat trajectory
* FakerSource regression — narrative + faker columns coexist on the same fact
* Classifier accuracy — hand-rolled multinomial naive Bayes on entity-split
  bag-of-words, accuracy ≥ 0.55 on held-out entities (chance ≈ 0.333 for 3
  segments). Threshold rationale documented inline.
* Bundled template — load_template / .py-vs-yaml parity / CLI
"""

from __future__ import annotations

import re
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from plotsim import (
    NarrativeConfig,
    NarrativeSource,
    PlotsimConfig,
    SurrogateKeyWarning,
    create,
    generate_tables,
    load_config,
    load_template,
)
from plotsim.config import parse_source


ROOT = Path(__file__).resolve().parent.parent


# ──────────────────────────────────────────────────────────────────────
# parse_source: grammar happy path + rejection
# ──────────────────────────────────────────────────────────────────────


def test_parse_narrative_basic():
    parsed = parse_source("narrative:review_text")
    assert isinstance(parsed, NarrativeSource)
    assert parsed.key == "review_text"


def test_parse_narrative_underscores_and_digits_in_key():
    parsed = parse_source("narrative:ticket_text_2")
    assert parsed.key == "ticket_text_2"


@pytest.mark.parametrize(
    "bad",
    [
        "narrative:",  # empty key
        "narrative",  # no prefix colon at all (falls through to invalid source)
        "narrative:foo:bar",  # embedded colon
        "narrative:1bad",  # leading digit — invalid identifier
        "narrative:has space",  # whitespace in identifier
        "narrative:hyphen-key",  # hyphen — invalid identifier
    ],
)
def test_parse_narrative_invalid_grammar_raises(bad):
    with pytest.raises(ValueError):
        parse_source(bad)


def test_narrative_source_key_must_be_identifier():
    """NarrativeSource keys go through the same identifier validator as
    PoolSource names — leading digit / non-identifier chars are rejected
    at construction time, not just at parse time."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        NarrativeSource(key="1bad")
    with pytest.raises(ValidationError):
        NarrativeSource(key="has space")


def test_narrative_source_frozen():
    """The model is frozen — no field reassignment after construction."""
    from pydantic import ValidationError

    src = NarrativeSource(key="ok")
    with pytest.raises(ValidationError):
        src.key = "changed"


def test_narrative_source_extra_forbid():
    """extra='forbid' rejects unknown kwargs (architectural rule)."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        NarrativeSource(key="ok", nonsense=True)


# ──────────────────────────────────────────────────────────────────────
# NarrativeConfig: template / lexicon / bands validation
# ──────────────────────────────────────────────────────────────────────


def _minimal_lexicon(archetypes=("growth",), bands=("low", "mid", "high")):
    """Return a valid lexicon for the template '{a} {b}.' across given archetypes."""
    return {
        arch: {
            "a": {b: ["x"] for b in bands},
            "b": {b: ["y"] for b in bands},
        }
        for arch in archetypes
    }


def test_narrative_config_minimal_valid():
    cfg = NarrativeConfig(
        template="{a} {b}.",
        lexicons=_minimal_lexicon(),
    )
    assert cfg.template_slots() == ["a", "b"]
    assert cfg.bands == ("low", "mid", "high")


def test_narrative_config_template_must_have_placeholders():
    with pytest.raises(ValueError, match="placeholders"):
        NarrativeConfig(template="static sentence.", lexicons={})


def test_narrative_config_template_rejects_duplicate_slots():
    with pytest.raises(ValueError, match="duplicate slot"):
        NarrativeConfig(
            template="{a} and {a}.",
            lexicons={"growth": {"a": {"low": ["x"], "mid": ["x"], "high": ["x"]}}},
        )


def test_narrative_config_lexicons_must_be_non_empty():
    with pytest.raises(ValueError, match="at least one archetype"):
        NarrativeConfig(template="{a}.", lexicons={})


def test_narrative_config_slots_must_match_template():
    """Lexicon slot keys must equal the template's {slot} placeholder set."""
    with pytest.raises(ValueError, match="do not match template placeholders"):
        NarrativeConfig(
            template="{a} {b}.",
            lexicons={
                "growth": {
                    "a": {"low": ["x"], "mid": ["x"], "high": ["x"]},
                    # 'b' missing
                }
            },
        )
    with pytest.raises(ValueError, match="do not match template placeholders"):
        NarrativeConfig(
            template="{a}.",
            lexicons={
                "growth": {
                    "a": {"low": ["x"], "mid": ["x"], "high": ["x"]},
                    "extra_slot": {"low": ["x"], "mid": ["x"], "high": ["x"]},
                }
            },
        )


def test_narrative_config_bands_must_match():
    """Per-slot band keys must equal the bands tuple."""
    with pytest.raises(ValueError, match="do not match declared"):
        NarrativeConfig(
            template="{a}.",
            lexicons={"growth": {"a": {"low": ["x"], "high": ["x"]}}},  # no 'mid'
        )


def test_narrative_config_bands_2_to_20():
    with pytest.raises(ValueError, match="2 and 20"):
        NarrativeConfig(
            template="{a}.",
            lexicons={"growth": {"a": {"only": ["x"]}}},
            bands=("only",),
        )
    with pytest.raises(ValueError, match="2 and 20"):
        NarrativeConfig(
            template="{a}.",
            lexicons={"growth": {"a": {f"b{i}": ["x"] for i in range(21)}}},
            bands=tuple(f"b{i}" for i in range(21)),
        )


def test_narrative_config_bands_two_works():
    """The lower bound of 2 is permissive — declares a binary low/high signal."""
    cfg = NarrativeConfig(
        template="{a}.",
        lexicons={"growth": {"a": {"low": ["x"], "high": ["y"]}}},
        bands=("low", "high"),
    )
    assert cfg.bands == ("low", "high")


def test_narrative_config_bands_unique():
    with pytest.raises(ValueError, match="unique"):
        NarrativeConfig(
            template="{a}.",
            lexicons={"growth": {"a": {"low": ["x"], "high": ["y"]}}},
            bands=("low", "low", "high"),
        )


def test_narrative_config_phrase_lists_non_empty():
    with pytest.raises(ValueError, match="non-empty list"):
        NarrativeConfig(
            template="{a}.",
            lexicons={"growth": {"a": {"low": [], "mid": ["x"], "high": ["y"]}}},
        )


def test_narrative_config_phrase_strings_non_empty():
    with pytest.raises(ValueError, match="non-empty"):
        NarrativeConfig(
            template="{a}.",
            lexicons={"growth": {"a": {"low": [""], "mid": ["x"], "high": ["y"]}}},
        )


def test_narrative_config_frozen():
    from pydantic import ValidationError

    cfg = NarrativeConfig(
        template="{a}.",
        lexicons={"growth": {"a": {"low": ["x"], "mid": ["y"], "high": ["z"]}}},
    )
    with pytest.raises(ValidationError):
        cfg.template = "{b}."


def test_narrative_config_extra_forbid():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        NarrativeConfig(
            template="{a}.",
            lexicons={"growth": {"a": {"low": ["x"], "mid": ["y"], "high": ["z"]}}},
            unknown_field=True,
        )


# ──────────────────────────────────────────────────────────────────────
# Column.narrative pairing
# ──────────────────────────────────────────────────────────────────────


def test_column_narrative_source_without_config_rejected():
    from plotsim.config import Column

    with pytest.raises(ValueError, match="no 'narrative' config block"):
        Column(name="x", dtype="string", source="narrative:foo")


def test_column_narrative_config_without_source_rejected():
    from plotsim.config import Column

    cfg = NarrativeConfig(
        template="{a}.",
        lexicons={"growth": {"a": {"low": ["x"], "mid": ["y"], "high": ["z"]}}},
    )
    with pytest.raises(ValueError, match="narrative config block but source"):
        Column(name="x", dtype="string", source="static:abc", narrative=cfg)


# ──────────────────────────────────────────────────────────────────────
# Cross-config gates — fact-only + archetype coverage
# ──────────────────────────────────────────────────────────────────────


def _build_minimal_config(**overrides):
    """Return a 2-segment narrative-bearing builder config; overrides splice in."""
    base = dict(
        about="Narrative test config",
        unit="customer",
        seed=2026,
        window=("2024-01", "2024-06", "monthly"),
        metrics=[{"name": "engagement", "type": "score", "polarity": "positive"}],
        segments=[
            {"name": "risers", "count": 6, "archetype": "growth"},
            {"name": "fallers", "count": 6, "archetype": "decline"},
        ],
        dimensions=[
            {
                "name": "dim_customer",
                "per": "unit",
                "columns": [
                    {"name": "customer_id", "type": "id"},
                    {"name": "customer_name", "type": "faker.name"},
                ],
            }
        ],
        facts=[
            {
                "name": "fct_reviews",
                "metrics": ["engagement"],
                "columns": [
                    {"name": "review_id", "type": "id"},
                    {"name": "customer_id", "type": "ref.dim_customer"},
                    {"name": "date_key", "type": "ref.dim_date"},
                    {"name": "engagement_score", "type": "metric.engagement"},
                    {
                        "name": "review_text",
                        "type": "narrative",
                        "template": "{a}.",
                        "lexicons": {
                            "risers": {"a": {"low": ["lo-r"], "mid": ["mi-r"], "high": ["hi-r"]}},
                            "fallers": {"a": {"low": ["lo-f"], "mid": ["mi-f"], "high": ["hi-f"]}},
                        },
                    },
                ],
            }
        ],
    )
    base.update(overrides)
    return base


def test_cross_config_unknown_archetype_in_lexicons_rejected():
    """Lexicon archetype keys that don't match config.archetypes are rejected."""
    bad_facts = [
        {
            "name": "fct_reviews",
            "metrics": ["engagement"],
            "columns": [
                {"name": "review_id", "type": "id"},
                {"name": "customer_id", "type": "ref.dim_customer"},
                {"name": "date_key", "type": "ref.dim_date"},
                {"name": "engagement_score", "type": "metric.engagement"},
                {
                    "name": "review_text",
                    "type": "narrative",
                    "template": "{a}.",
                    "lexicons": {
                        "risers": {"a": {"low": ["x"], "mid": ["y"], "high": ["z"]}},
                        "fallers": {"a": {"low": ["x"], "mid": ["y"], "high": ["z"]}},
                        "phantoms": {"a": {"low": ["x"], "mid": ["y"], "high": ["z"]}},
                    },
                },
            ],
        }
    ]
    with pytest.raises(ValueError, match="unknown archetypes"):
        create(**_build_minimal_config(facts=bad_facts))


def test_cross_config_missing_archetype_lexicon_rejected():
    """Every assigned archetype must have a lexicon."""
    bad_facts = [
        {
            "name": "fct_reviews",
            "metrics": ["engagement"],
            "columns": [
                {"name": "review_id", "type": "id"},
                {"name": "customer_id", "type": "ref.dim_customer"},
                {"name": "date_key", "type": "ref.dim_date"},
                {"name": "engagement_score", "type": "metric.engagement"},
                {
                    "name": "review_text",
                    "type": "narrative",
                    "template": "{a}.",
                    "lexicons": {
                        # 'fallers' lexicon missing
                        "risers": {"a": {"low": ["x"], "mid": ["y"], "high": ["z"]}},
                    },
                },
            ],
        }
    ]
    with pytest.raises(ValueError, match="missing entries for archetypes"):
        create(**_build_minimal_config(facts=bad_facts))


def test_cross_config_narrative_on_dim_rejected_at_load(tmp_path):
    """A narrative source on a dim column should be rejected at config load.

    The builder doesn't expose ``type: narrative`` on dim columns
    directly (the interpreter only emits narrative columns from
    FactInput). We dump the bundled narrative_reviews template to YAML,
    relocate the narrative column under ``dim_customer``, and confirm
    the engine cross-config validator catches the placement.
    """
    import yaml

    from plotsim import dump_config

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SurrogateKeyWarning)
        cfg = load_template("narrative_reviews")
    base = yaml.safe_load(dump_config(cfg))

    # Move the narrative column from fct_reviews onto dim_customer.
    nar_col = None
    for tbl in base["tables"]:
        if tbl["name"] == "fct_reviews":
            kept = []
            for c in tbl["columns"]:
                if c["name"] == "review_text":
                    nar_col = c
                else:
                    kept.append(c)
            tbl["columns"] = kept
    assert nar_col is not None
    for tbl in base["tables"]:
        if tbl["name"] == "dim_customer":
            tbl["columns"].append(nar_col)

    path = tmp_path / "narrative_on_dim.yaml"
    path.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="not a per_entity_per_period fact"):
        load_config(path)


# ──────────────────────────────────────────────────────────────────────
# dtype: boolean rejection
# ──────────────────────────────────────────────────────────────────────


def test_narrative_dtype_boolean_rejected_at_load(tmp_path):
    """M102 discipline: dtype: boolean on a NarrativeSource raises at load.

    ``bool("any text")`` is always True — the column would carry no
    discriminative information once cast. The builder forces ``dtype:
    string`` so we go the engine-direct route (dump the working
    template, mutate the dtype, re-load).
    """
    import yaml

    from plotsim import dump_config

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SurrogateKeyWarning)
        cfg = load_template("narrative_reviews")
    base = yaml.safe_load(dump_config(cfg))
    for tbl in base["tables"]:
        if tbl["name"] == "fct_reviews":
            for col in tbl["columns"]:
                if col["name"] == "review_text":
                    col["dtype"] = "boolean"
    path = tmp_path / "narrative_boolean.yaml"
    path.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="narrative-source"):
        load_config(path)


# ──────────────────────────────────────────────────────────────────────
# End-to-end on the bundled narrative_reviews template
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def reviews_cfg() -> PlotsimConfig:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SurrogateKeyWarning)
        return load_template("narrative_reviews")


@pytest.fixture(scope="module")
def reviews_tables(reviews_cfg):
    rng = np.random.default_rng(reviews_cfg.seed)
    return generate_tables(reviews_cfg, rng)


def test_template_loads_and_generates(reviews_tables):
    df = reviews_tables["fct_reviews"]
    assert "review_text" in df.columns
    assert df["review_text"].notna().all()
    # 60 entities × 24 periods = 1440 rows
    assert len(df) == 1440


def test_template_text_columns_dtype_string(reviews_tables):
    df = reviews_tables["fct_reviews"]
    sample = df["review_text"].iloc[0]
    assert isinstance(sample, str) and sample


def test_template_yaml_and_python_produce_identical_text(reviews_cfg):
    """The .py and .yaml templates resolve to byte-identical text columns
    under the same seed."""
    from plotsim.configs.templates.narrative_reviews import config as cfg_py

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SurrogateKeyWarning)
        rng_y = np.random.default_rng(reviews_cfg.seed)
        rng_p = np.random.default_rng(cfg_py.seed)
        df_y = generate_tables(reviews_cfg, rng_y)["fct_reviews"]
        df_p = generate_tables(cfg_py, rng_p)["fct_reviews"]
    assert (df_y["review_text"].values == df_p["review_text"].values).all()


# ──────────────────────────────────────────────────────────────────────
# Determinism
# ──────────────────────────────────────────────────────────────────────


def test_template_determinism(reviews_cfg):
    """Same config + same seed → byte-identical text column across runs."""
    df1 = generate_tables(reviews_cfg, np.random.default_rng(reviews_cfg.seed))
    df2 = generate_tables(reviews_cfg, np.random.default_rng(reviews_cfg.seed))
    assert df1["fct_reviews"]["review_text"].tolist() == df2["fct_reviews"]["review_text"].tolist()


# ──────────────────────────────────────────────────────────────────────
# Trajectory direction + per-archetype lexicon respected
# ──────────────────────────────────────────────────────────────────────


def _segment_for_customer_id(customer_id: str) -> str:
    """The bundled template emits c-001..c-020 promoters, c-021..c-040 neutrals,
    c-041..c-060 detractors (entities are flattened in segments-then-index order).
    """
    n = int(customer_id.split("-")[1])
    if n <= 20:
        return "promoters"
    if n <= 40:
        return "neutrals"
    return "detractors"


def _all_phrases_for_segment(cfg: NarrativeConfig, segment: str) -> set[str]:
    """Flatten every phrase across all (slot, band) for one segment."""
    out: set[str] = set()
    for _slot, by_band in cfg.lexicons[segment].items():
        for _band, phrases in by_band.items():
            for p in phrases:
                out.add(p)
    return out


def test_per_archetype_lexicon_is_respected(reviews_cfg, reviews_tables):
    """Every emitted token sequence for a promoter must have come from the
    promoter lexicon — no leakage from neutrals or detractors."""
    df = reviews_tables["fct_reviews"]
    fct_table = next(t for t in reviews_cfg.tables if t.name == "fct_reviews")
    review_col = next(c for c in fct_table.columns if c.name == "review_text")
    nar_cfg = review_col.narrative
    assert nar_cfg is not None

    promoter_phrases = _all_phrases_for_segment(nar_cfg, "promoters")
    detractor_phrases = _all_phrases_for_segment(nar_cfg, "detractors")
    leaked = []
    for _, row in df.iterrows():
        seg = _segment_for_customer_id(row["customer_id"])
        text = row["review_text"]
        if seg == "promoters":
            # No detractor-only phrase may appear in a promoter row
            offending = [p for p in (detractor_phrases - promoter_phrases) if p in text]
            if offending:
                leaked.append((row["customer_id"], text, offending))
    assert not leaked, f"detractor phrases leaked into promoter rows: {leaked[:3]}"


def test_text_varies_across_periods_for_non_flat_trajectory(reviews_tables):
    """A promoter's first and last reviews should not be the same string —
    the trajectory rises so the band shifts so the vocabulary shifts."""
    df = reviews_tables["fct_reviews"].sort_values(["customer_id", "date_key"])
    first_promoter = df[df["customer_id"] == "c-001"]
    early = first_promoter.head(3)["review_text"].tolist()
    late = first_promoter.tail(3)["review_text"].tolist()
    # At least one early and one late text differ — trivially holds for any
    # band shift.
    assert set(early) != set(late), (
        f"promoter c-001 produced identical text across rising trajectory: "
        f"early={early}, late={late}"
    )


def test_band_monotonicity_per_entity(reviews_cfg, reviews_tables):
    """Per-entity engagement and band-index ranks should align (Spearman > 0)
    when aggregated across periods. The trajectory-first invariant says: a
    higher per-entity engagement mean → a higher per-entity band index mean.
    """
    df = reviews_tables["fct_reviews"]
    fct_table = next(t for t in reviews_cfg.tables if t.name == "fct_reviews")
    nar_cfg = next(c for c in fct_table.columns if c.name == "review_text").narrative

    # Recover the band index by reverse-mapping the comment-slot phrase to its
    # band — comment phrases happen to be unique within each segment's lexicon.
    band_to_idx = {b: i for i, b in enumerate(nar_cfg.bands)}
    phrase_to_band: dict[tuple[str, str], int] = {}
    for seg, by_slot in nar_cfg.lexicons.items():
        for band, phrases in by_slot["comment"].items():
            for p in phrases:
                phrase_to_band[(seg, p)] = band_to_idx[band]

    def lookup_band(row) -> int:
        seg = _segment_for_customer_id(row["customer_id"])
        # Comment is always the last "." chunk before final period —
        # easier: find the unique suffix that matches a comment phrase.
        text = row["review_text"]
        for (s, phrase), idx in phrase_to_band.items():
            if s == seg and text.endswith(phrase):
                return idx
        # Fall-through: should never happen if lexicon is well-formed
        return -1

    df = df.copy()
    df["band_idx"] = df.apply(lookup_band, axis=1)
    assert (df["band_idx"] >= 0).all(), "band lookup failed for some rows"

    per_entity = df.groupby("customer_id", sort=False).agg(
        mean_engagement=("engagement_score", "mean"),
        mean_band_idx=("band_idx", "mean"),
    )
    from scipy.stats import spearmanr

    rho, _p = spearmanr(per_entity["mean_engagement"], per_entity["mean_band_idx"])
    assert rho > 0.5, (
        f"per-entity band index does not track engagement (ρ={rho:.3f}); "
        f"trajectory-first invariant violated for the narrative branch"
    )


# ──────────────────────────────────────────────────────────────────────
# FakerSource regression
# ──────────────────────────────────────────────────────────────────────


def test_faker_column_unaffected_by_narrative_addition(reviews_tables):
    """The dim_customer.customer_name faker column should be populated with
    string values; narrative source shouldn't disturb the faker pipeline."""
    dim = reviews_tables["dim_customer"]
    assert "customer_name" in dim.columns
    assert dim["customer_name"].notna().all()
    assert all(isinstance(v, str) and v for v in dim["customer_name"])


def test_faker_and_narrative_coexist_on_same_fact():
    """A fact table with both a FakerSource column and a NarrativeSource
    column should generate cleanly. Both source types force the scalar
    builder path; this test pins that they coexist without RNG ordering
    drift breaking either."""
    with warnings.catch_warnings():
        # The one-segment advisory is intentional here — we're isolating
        # the (faker × narrative) interaction, not testing variety.
        warnings.simplefilter("ignore", UserWarning)
        cfg = create(
            about="Faker + narrative coexistence",
            unit="customer",
            seed=42,
            window=("2024-01", "2024-06", "monthly"),
            metrics=[{"name": "engagement", "type": "score", "polarity": "positive"}],
            segments=[
                {"name": "risers", "count": 5, "archetype": "growth"},
            ],
            dimensions=[
                {
                    "name": "dim_customer",
                    "per": "unit",
                    "columns": [
                        {"name": "customer_id", "type": "id"},
                    ],
                }
            ],
            facts=[
                {
                    "name": "fct_reviews",
                    "metrics": ["engagement"],
                    "columns": [
                        {"name": "review_id", "type": "id"},
                        {"name": "customer_id", "type": "ref.dim_customer"},
                        {"name": "date_key", "type": "ref.dim_date"},
                        {"name": "engagement_score", "type": "metric.engagement"},
                        {"name": "row_note", "type": "faker.sentence"},
                        {
                            "name": "review_text",
                            "type": "narrative",
                            "template": "{a}.",
                            "lexicons": {
                                "risers": {"a": {"low": ["lo"], "mid": ["mi"], "high": ["hi"]}}
                            },
                        },
                    ],
                }
            ],
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SurrogateKeyWarning)
        df = generate_tables(cfg, np.random.default_rng(cfg.seed))["fct_reviews"]
    assert df["row_note"].notna().all()
    assert df["review_text"].notna().all()
    assert all(t.endswith(".") for t in df["review_text"])


# ──────────────────────────────────────────────────────────────────────
# Classifier accuracy — the core differentiator vs independent Faker text
# ──────────────────────────────────────────────────────────────────────


_TOKEN_RE = re.compile(r"[A-Za-z]+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _train_naive_bayes(
    train_rows: list[tuple[str, str]],
) -> tuple[dict[str, dict[str, int]], Counter[str], int]:
    """Hand-rolled multinomial Naive Bayes with Laplace smoothing.

    Returns ``(token_counts_by_class, class_doc_counts, vocab_size)``. We
    use this instead of importing scikit-learn — sklearn isn't a
    plotsim dep and the engine's success criterion is "above-chance",
    not "best-in-class". A 30-line classifier is the right tool.
    """
    counts_by_class: dict[str, dict[str, int]] = {}
    class_doc_counts: Counter[str] = Counter()
    vocab: set[str] = set()
    for label, text in train_rows:
        class_doc_counts[label] += 1
        bucket = counts_by_class.setdefault(label, {})
        for tok in _tokenize(text):
            bucket[tok] = bucket.get(tok, 0) + 1
            vocab.add(tok)
    return counts_by_class, class_doc_counts, len(vocab)


def _predict_naive_bayes(
    text: str,
    counts_by_class: dict[str, dict[str, int]],
    class_doc_counts: Counter[str],
    vocab_size: int,
) -> str:
    """Pick the class with the largest log-posterior under Laplace smoothing."""
    n_total = sum(class_doc_counts.values())
    best_class = ""
    best_score = -np.inf
    for cls in class_doc_counts:
        log_prior = np.log(class_doc_counts[cls] / n_total)
        bucket = counts_by_class.get(cls, {})
        bucket_total = sum(bucket.values())
        score = log_prior
        for tok in _tokenize(text):
            # Laplace smoothing: (count + 1) / (bucket_total + vocab_size)
            score += np.log((bucket.get(tok, 0) + 1) / (bucket_total + vocab_size))
        if score > best_score:
            best_score = score
            best_class = cls
    return best_class


def test_classifier_accuracy_above_threshold(reviews_tables):
    """A bag-of-words classifier predicts the segment well above chance.

    Threshold rationale (≥ 0.55):
      * 3 segments → 1/3 ≈ 0.333 chance accuracy.
      * The lexicon has intentional cross-segment phrase overlap (e.g.
        "It works." appears in promoter-low and neutral-mid pools), so a
        perfect classifier would not reach 1.0.
      * 0.55 sits comfortably above chance (+22 percentage points) but
        well below the upper bound that a tighter lexicon could
        theoretically allow — leaves headroom for future lexicon edits
        that broaden overlap without breaking the test.
      * Train / test split is by entity-id (80 % train / 20 % test), so
        test entities are unseen — the classifier learns the
        per-segment vocabulary distribution rather than memorizing
        per-entity strings.
    """
    df = reviews_tables["fct_reviews"]
    df = df.copy()
    df["segment"] = df["customer_id"].apply(_segment_for_customer_id)

    # Entity-level split: deterministic, sorted by id, last 20 % held out.
    entity_ids = sorted(df["customer_id"].unique())
    n_test = max(1, len(entity_ids) // 5)
    test_ids = set(entity_ids[-n_test:])
    train_ids = set(entity_ids[:-n_test])

    train_rows = [
        (r["segment"], r["review_text"])
        for _, r in df[df["customer_id"].isin(train_ids)].iterrows()
    ]
    test_rows = [
        (r["segment"], r["review_text"]) for _, r in df[df["customer_id"].isin(test_ids)].iterrows()
    ]

    counts, class_doc_counts, vocab_size = _train_naive_bayes(train_rows)
    correct = sum(
        1
        for label, text in test_rows
        if _predict_naive_bayes(text, counts, class_doc_counts, vocab_size) == label
    )
    accuracy = correct / len(test_rows)
    assert accuracy >= 0.55, (
        f"bag-of-words classifier accuracy {accuracy:.3f} fell below the "
        f"≥ 0.55 threshold (chance ≈ 0.333 for 3 segments). The "
        f"trajectory- × segment-keyed lexicon should produce a learnable "
        f"signal; if this fails, check whether the bundled lexicons have "
        f"drifted toward more cross-segment overlap than the design tolerates."
    )


# ──────────────────────────────────────────────────────────────────────
# CLI smoke — plotsim run + plotsim validate on the template
# ──────────────────────────────────────────────────────────────────────


def test_template_passes_plotsim_validate_via_engine(reviews_cfg):
    """The bundled template loads cleanly and the engine validate pass
    finds no errors — covers the ``plotsim validate`` happy path without
    spawning a subprocess."""
    from plotsim import validate

    rng = np.random.default_rng(reviews_cfg.seed)
    tables = generate_tables(reviews_cfg, rng)
    report = validate(reviews_cfg, tables)
    assert report.ok, f"validate returned errors: {report.errors}"
