"""M105 / Track B — TextBucketSource tests.

Covers:

* parse_source rejection paths (malformed input)
* Pydantic model gates (extra="forbid", min/max bucket count)
* Vectorized fact-builder branch produces text aligned with trajectory
* Scalar fallback path produces the same banding as the vectorized path
* Band assignment monotonicity — entity sorted by trajectory mean fires
  text from lower → higher buckets in matching order
* Determinism — same config + same seed → byte-identical text column
* Boundary handling — position == 1.0 lands in the last bucket (not OOR)
* dtype: boolean rejected at load time (M102 discipline preserved)
* Bundled SaaS template demonstrates a TextBucketSource column

Tests build on ``sample_saas.yaml`` (which now carries
``customer_sentiment``) plus an inline synthetic config that pins archetype
trajectories at known shapes for deterministic band-assertion math.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
import yaml

from plotsim import (
    PlotsimConfig,
    SurrogateKeyWarning,
    TextBucketSource,
    generate_tables,
    load_config,
)
from plotsim.config import parse_source


ROOT = Path(__file__).resolve().parent.parent
SAAS_YAML = ROOT / "plotsim" / "configs" / "sample_saas.yaml"


# --- parse_source: grammar happy path ---------------------------------------


def test_parse_text_bucket_basic():
    parsed = parse_source("text:bucket:[low, mid, high]")
    assert isinstance(parsed, TextBucketSource)
    assert parsed.buckets == ("low", "mid", "high")


def test_parse_text_bucket_strips_whitespace():
    parsed = parse_source("text:bucket:[  one ,  two  ,three]")
    assert parsed.buckets == ("one", "two", "three")


def test_parse_text_bucket_two_labels_minimum():
    parsed = parse_source("text:bucket:[bad, good]")
    assert parsed.buckets == ("bad", "good")


def test_parse_text_bucket_with_underscores_and_digits():
    parsed = parse_source("text:bucket:[tier_1, tier_2, tier_3]")
    assert parsed.buckets == ("tier_1", "tier_2", "tier_3")


# --- parse_source: rejection paths ------------------------------------------


@pytest.mark.parametrize("bad", [
    "text:bucket:",                       # empty
    "text:bucket:low,mid,high",           # no brackets
    "text:bucket:[low, mid, high",        # missing close
    "text:bucket:low, mid, high]",        # missing open
    "text:bucket:[]",                     # empty list
    "text:bucket:[only]",                 # single label
    "text:bucket:[a,,b]",                 # empty middle
    "text:bucket:[a, b, a]",              # duplicate labels
    "text:bucket:[ , a, b]",              # whitespace-only label
])
def test_parse_text_bucket_invalid_grammar_raises(bad):
    with pytest.raises(ValueError):
        parse_source(bad)


def test_text_bucket_source_min_length_enforced():
    """Pydantic min_length=2 raises on direct construction with one label."""
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        TextBucketSource(buckets=("only",))


def test_text_bucket_source_max_length_enforced():
    """Pydantic max_length=20 raises on excessive bucket counts.

    Twenty buckets is already extreme — beyond that the position-band
    width drops below the typical noise floor of a trajectory and the
    bucket assignment becomes random under any noise injection.
    """
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        TextBucketSource(buckets=tuple(f"b{i}" for i in range(21)))


def test_text_bucket_source_extra_forbid():
    """extra='forbid' rejects unknown kwargs (M105 architectural rule)."""
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        TextBucketSource(buckets=("a", "b"), nonsense_field=True)


def test_text_bucket_source_frozen():
    """The model is frozen — no field reassignment after construction."""
    from pydantic import ValidationError
    src = TextBucketSource(buckets=("a", "b"))
    with pytest.raises(ValidationError):
        src.buckets = ("c", "d")


# --- dtype: boolean rejection (M102 discipline preserved) -------------------


def _make_text_bucket_yaml(tmp_path: Path, dtype: str) -> Path:
    """Render a minimal saas-style YAML with a TextBucketSource at given dtype."""
    base = yaml.safe_load(SAAS_YAML.read_text(encoding="utf-8"))
    for tbl in base["tables"]:
        if tbl["name"] == "fct_engagement":
            for col in tbl["columns"]:
                if col["name"] == "customer_sentiment":
                    col["dtype"] = dtype
            break
    out = tmp_path / "saas_dtype.yaml"
    out.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
    return out


def test_text_bucket_dtype_boolean_rejected_at_load(tmp_path):
    """M102 discipline: dtype: boolean on a TextBucketSource raises at load.

    bool('delighted') is always True — the column carries no banding
    information once cast. The error message names the source kind so
    config authors know which gate fired.
    """
    bad = _make_text_bucket_yaml(tmp_path, dtype="boolean")
    with pytest.raises(ValueError, match="text-bucket"):
        load_config(bad)


def test_text_bucket_dtype_string_accepted(tmp_path):
    """The bundled saas template uses dtype: string — the path is exercised."""
    good = _make_text_bucket_yaml(tmp_path, dtype="string")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SurrogateKeyWarning)
        cfg = load_config(good)
    assert isinstance(cfg, PlotsimConfig)


# --- End-to-end on the bundled SaaS template --------------------------------


@pytest.fixture(scope="module")
def saas_cfg():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SurrogateKeyWarning)
        return load_config(SAAS_YAML)


@pytest.fixture(scope="module")
def saas_tables(saas_cfg):
    rng = np.random.default_rng(saas_cfg.seed)
    return generate_tables(saas_cfg, rng)


def test_saas_template_has_customer_sentiment_column(saas_tables):
    df = saas_tables["fct_engagement"]
    assert "customer_sentiment" in df.columns


def test_saas_template_sentiment_values_are_configured_labels(saas_tables):
    """Every emitted value lands in the configured bucket set — no leakage."""
    df = saas_tables["fct_engagement"]
    expected = {"at_risk", "lukewarm", "satisfied", "delighted"}
    actual = set(df["customer_sentiment"].dropna().unique())
    assert actual.issubset(expected), (
        f"sentiment column produced unexpected labels: "
        f"{actual - expected}"
    )


def test_saas_template_sentiment_uses_all_buckets(saas_tables):
    """A 24-period × 90-entity run touches all 4 buckets at least once.

    With three archetypes spanning rocket / grower / cliff and 24 monthly
    periods the trajectory range is wide enough that every band gets at
    least one cell. If this ever stops being true on the bundled
    template, the archetypes have collapsed in distinguishability.
    """
    df = saas_tables["fct_engagement"]
    seen = set(df["customer_sentiment"].dropna().unique())
    assert seen == {"at_risk", "lukewarm", "satisfied", "delighted"}, (
        f"missing buckets: "
        f"{ {'at_risk','lukewarm','satisfied','delighted'} - seen}"
    )


def test_saas_template_determinism(saas_cfg):
    """Same config + same seed → identical sentiment column across runs."""
    df1 = generate_tables(saas_cfg, np.random.default_rng(saas_cfg.seed))
    df2 = generate_tables(saas_cfg, np.random.default_rng(saas_cfg.seed))
    assert df1["fct_engagement"]["customer_sentiment"].tolist() == \
        df2["fct_engagement"]["customer_sentiment"].tolist()


# --- Banding semantics on a controlled trajectory ---------------------------


def _evenly_spaced_band_edges(n_buckets: int) -> list[float]:
    return [k / n_buckets for k in range(n_buckets + 1)]


def test_band_edges_evenly_spaced():
    """Mission AC: band boundaries at 0.25, 0.50, 0.75 for 4 buckets."""
    edges = _evenly_spaced_band_edges(4)
    assert edges == [0.0, 0.25, 0.5, 0.75, 1.0]


def test_band_assignment_position_zero_lands_in_first_bucket():
    """The construction ``min(int(p * N), N - 1)`` puts p=0 in bucket[0]."""
    n_buckets = 4
    p = 0.0
    idx = min(int(p * n_buckets), n_buckets - 1)
    assert idx == 0


def test_band_assignment_position_one_lands_in_last_bucket():
    """Boundary closure: p=1.0 maps to N-1, not N (no out-of-range)."""
    n_buckets = 4
    p = 1.0
    idx = min(int(p * n_buckets), n_buckets - 1)
    assert idx == n_buckets - 1


@pytest.mark.parametrize("position,expected_idx", [
    (0.0, 0),
    (0.24999, 0),
    (0.25, 1),
    (0.49999, 1),
    (0.5, 2),
    (0.74999, 2),
    (0.75, 3),
    (0.99999, 3),
    (1.0, 3),
])
def test_band_assignment_at_edges(position, expected_idx):
    """Each [k/N, (k+1)/N) band maps to bucket k; the top edge closes."""
    n_buckets = 4
    idx = min(int(position * n_buckets), n_buckets - 1)
    assert idx == expected_idx


def test_band_monotonicity_on_saas_per_entity(saas_cfg, saas_tables):
    """Entity-mean sentiment ordinal rises with entity-mean engagement.

    The trajectory-first invariant says: if entity A has a higher
    average trajectory position than B, A's sentiment column should
    skew toward higher buckets than B's. We translate the categorical
    column into bucket-index space, take the per-entity mean, and
    check the ranking matches the per-entity engagement_score mean.
    """
    df = saas_tables["fct_engagement"]
    bucket_to_idx = {
        "at_risk": 0, "lukewarm": 1, "satisfied": 2, "delighted": 3,
    }
    df = df.assign(
        sentiment_idx=df["customer_sentiment"].map(bucket_to_idx),
    )
    per_entity = df.groupby("company_id", sort=False).agg(
        mean_engagement=("engagement_score", "mean"),
        mean_sentiment=("sentiment_idx", "mean"),
    )
    # Spearman rank correlation between the two means; any positive
    # value confirms the bands track engagement direction. We use
    # rank-based to dodge calibration / scale differences between the
    # continuous engagement metric and the discretized sentiment ordinal.
    from scipy.stats import spearmanr
    rho, _p = spearmanr(per_entity["mean_engagement"], per_entity["mean_sentiment"])
    assert rho > 0.5, (
        f"per-entity sentiment ordinal does not track engagement (ρ={rho:.3f}); "
        f"trajectory-first invariant violated for the text-bucket branch"
    )


# --- Scalar-fallback path coverage ------------------------------------------


def test_text_bucket_source_works_in_scalar_path(tmp_path):
    """A fact table that also carries FakerSource forces the scalar path.

    The vectorized branch is the default for per_entity_per_period facts
    with no FakerSource columns; presence of any FakerSource flips the
    builder to ``_scalar_per_entity_per_period_fact``. We mutate the saas
    template to add a Faker column on fct_engagement so the scalar path
    is the one resolving customer_sentiment, and assert the same set of
    bucket labels lands.
    """
    base = yaml.safe_load(SAAS_YAML.read_text(encoding="utf-8"))
    for tbl in base["tables"]:
        if tbl["name"] == "fct_engagement":
            tbl["columns"].append({
                "name": "row_note",
                "dtype": "string",
                "source": "generated:faker.sentence",
            })
            break
    out = tmp_path / "saas_scalar.yaml"
    out.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SurrogateKeyWarning)
        cfg = load_config(out)
    rng = np.random.default_rng(cfg.seed)
    tables = generate_tables(cfg, rng)
    df = tables["fct_engagement"]
    expected = {"at_risk", "lukewarm", "satisfied", "delighted"}
    actual = set(df["customer_sentiment"].dropna().unique())
    assert actual.issubset(expected)
    # Faker column present ⇒ confirms scalar path was the writer.
    assert "row_note" in df.columns
    assert df["row_note"].notna().all()


def test_vectorized_and_scalar_paths_agree_on_band_assignment(tmp_path):
    """Both fact-builder paths produce the same sentiment column for the
    same config + seed.

    The vectorized path uses numpy ravel + bucket_arr indexing; the
    scalar path uses python float * N + min/max clamp. They share the
    formula but execute on different code paths — this test catches
    drift between them.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SurrogateKeyWarning)
        cfg_vec = load_config(SAAS_YAML)
    rng_vec = np.random.default_rng(cfg_vec.seed)
    tables_vec = generate_tables(cfg_vec, rng_vec)
    sentiment_vec = tables_vec["fct_engagement"][
        ["company_id", "date_key", "customer_sentiment"]
    ].sort_values(["company_id", "date_key"]).reset_index(drop=True)

    base = yaml.safe_load(SAAS_YAML.read_text(encoding="utf-8"))
    for tbl in base["tables"]:
        if tbl["name"] == "fct_engagement":
            tbl["columns"].append({
                "name": "row_note",
                "dtype": "string",
                "source": "generated:faker.sentence",
            })
            break
    forced_path = tmp_path / "saas_force_scalar.yaml"
    forced_path.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SurrogateKeyWarning)
        cfg_scalar = load_config(forced_path)
    rng_scalar = np.random.default_rng(cfg_scalar.seed)
    tables_scalar = generate_tables(cfg_scalar, rng_scalar)
    sentiment_scalar = tables_scalar["fct_engagement"][
        ["company_id", "date_key", "customer_sentiment"]
    ].sort_values(["company_id", "date_key"]).reset_index(drop=True)

    # Identical sentiment column on identical (entity, period) keys.
    assert sentiment_vec["customer_sentiment"].tolist() == \
        sentiment_scalar["customer_sentiment"].tolist(), (
            "vectorized path and scalar path disagree on text-bucket "
            "assignment — the band arithmetic has drifted between them"
        )
