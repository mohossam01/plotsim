# Narrative text source

> Trajectory- and archetype-driven sentence generation for fact tables.
> Unlocks NLP-flavored exercises (sentiment analysis, text
> classification, topic modeling) on plotsim output without breaking
> the trajectory-first invariant.

---

## What it does

A `narrative` column emits one sentence per fact row by:

1. Reading the row's trajectory position `p ∈ [0, 1]` from the same
   array that drives every other metric on the row.
2. Mapping `p` into one of N evenly-spaced **bands** (default
   `low / mid / high`) using the same arithmetic
   [text-bucket columns](../column-types.md) use:
   `band_idx = min(int(p * N), N - 1)`.
3. Looking up the entity's archetype to pick which **lexicon** applies.
4. For each `{slot}` placeholder in the **template**, sampling one
   phrase uniformly from `lexicons[archetype][slot][band]` via the
   seeded engine RNG.
5. Returning `template.format(**slot_values)`.

The trajectory-first invariant holds: every cell is a deterministic
function of `(trajectory_position, archetype, RNG_state)`. Same seed →
byte-identical text columns.

---

## When to use it

| Use case | Why narrative fits |
|---|---|
| Sentiment classification | Text vocabulary tracks engagement direction per archetype |
| Text classification by segment | A bag-of-words model can recover the segment with above-chance accuracy |
| Topic modeling demos | Per-archetype lexicons surface as discoverable topics |
| NER / entity extraction | Combine with `faker.*` columns for named-entity slots |

Narrative is **fact-only** and **per_entity_per_period** — text is
expected to vary per entity per period as the trajectory moves. Dim
tables are static (use a `text:bucket:` column on a dim if you want a
single trajectory-aware label per entity); per-period facts and event
tables don't have the per-row trajectory plumbing wired.

---

## How to enable

=== "YAML"

    ```yaml
    facts:
      - name: fct_reviews
        metrics: [engagement]
        columns:
          - { name: review_id,        type: id }
          - { name: customer_id,      type: ref.dim_customer }
          - { name: date_key,         type: ref.dim_date }
          - { name: engagement_score, type: metric.engagement }
          - name: review_text
            type: narrative
            template: "{opener} {object}. {comment}"
            lexicons:
              promoters:
                opener:
                  low:  ["I tried"]
                  mid:  ["I am using"]
                  high: ["I love"]
                object:
                  low:  ["the app"]
                  mid:  ["this product"]
                  high: ["this product"]
                comment:
                  low:  ["Decent start."]
                  mid:  ["Glad we picked it."]
                  high: ["Highly recommend."]
              detractors:
                opener:
                  low:  ["Was hopeful for"]
                  mid:  ["Tried"]
                  high: ["Briefly liked"]
                object:
                  low:  ["this thing"]
                  mid:  ["the service"]
                  high: ["the product"]
                comment:
                  low:  ["Going elsewhere."]
                  mid:  ["Mediocre."]
                  high: ["Some good days."]
    ```

=== "Python"

    ```python
    from plotsim import create

    LEX = {
        "promoters": {
            "opener":  {"low": ["I tried"],         "mid": ["I am using"],     "high": ["I love"]},
            "object":  {"low": ["the app"],         "mid": ["this product"],   "high": ["this product"]},
            "comment": {"low": ["Decent start."],   "mid": ["Glad we picked it."], "high": ["Highly recommend."]},
        },
        "detractors": {
            "opener":  {"low": ["Was hopeful for"], "mid": ["Tried"],          "high": ["Briefly liked"]},
            "object":  {"low": ["this thing"],      "mid": ["the service"],    "high": ["the product"]},
            "comment": {"low": ["Going elsewhere."],"mid": ["Mediocre."],      "high": ["Some good days."]},
        },
    }

    cfg = create(
        # ... about / unit / window / metrics / segments ...
        facts=[
            {
                "name": "fct_reviews",
                "metrics": ["engagement"],
                "columns": [
                    {"name": "review_id",        "type": "id"},
                    {"name": "customer_id",      "type": "ref.dim_customer"},
                    {"name": "date_key",         "type": "ref.dim_date"},
                    {"name": "engagement_score", "type": "metric.engagement"},
                    {
                        "name": "review_text",
                        "type": "narrative",
                        "template": "{opener} {object}. {comment}",
                        "lexicons": LEX,
                    },
                ],
            },
        ],
    )
    ```

The bundled `narrative_reviews` template runs end-to-end:

```bash
plotsim template narrative_reviews -o narrative_reviews.yaml
plotsim run narrative_reviews.yaml -o ./narrative_out
head ./narrative_out/fct_reviews.csv
```

---

## Lexicon design — the signal a classifier learns

Lexicons in plotsim do two jobs at once: they let a classifier recover
the **segment** from the text, and they let a sentiment model recover
the **trajectory band** from the text. Four principles produce that
dual-use signal:

1. **Universal sentiment gradient on bands.** Low-band phrases carry
   negative language, mid-band phrases carry neutral language, and
   high-band phrases carry positive language — consistent across
   *every* archetype. A growth entity's text becomes more positive as
   its trajectory rises; a decline entity's text becomes more negative
   as its trajectory falls. Because sentiment is keyed on the band
   rather than the archetype, the same text is usable for sentiment
   classification (predict band from text) as well as segment
   classification (predict archetype from text).
2. **Each slot is an independent aspect.** The template's slots are
   the seams along which aspect-based sentiment can be recovered. In
   `narrative_reviews` the three slots cover the *experience* aspect
   (`opener`), the *product* aspect (`object`), and the *judgment*
   aspect (`comment`). Each slot independently varies in sentiment
   across bands, so a per-aspect sentiment classifier can be trained
   on top of the same text column without changing any engine config.
3. **Archetype identity concentrates in one slot.** Putting per-
   archetype phrasing on *every* slot makes segment classification
   trivial — a single distinctive token leaks the label. Instead,
   concentrate the archetype-distinct surface in *one* slot (usually
   `opener`) and let the remaining slots share their phrase pools
   across archetypes. The `opener` carries a segment-distinct
   speaking *style* (in `narrative_reviews`: promoters speak in first-
   person emotional voice, neutrals in measured observational voice,
   detractors in terse blunt voice), while `object` and `comment`
   draw from the same band-keyed pool regardless of segment. The
   classifier still recovers the segment — opener tokens are
   distinctive enough — but it cannot cheat its way to 1.0 accuracy.
4. **Multiple phrases per (slot, band) cell.** Ten to fifteen phrases
   per cell gives ≈ 10³ = 1,000 combinations per (segment, band) —
   enough variety that adjacent rows differ even when the band hasn't
   shifted, and enough vocabulary that bag-of-words classifiers learn
   token distributions rather than memorizing per-row strings.

The bundled `narrative_reviews` template hits ≥ 0.55 bag-of-words
classification accuracy on a held-out entity split (chance is 1/3 for
three segments) while keeping sentiment recoverable per band. Lexicons
that ship below the segment-classification threshold either have
collapsed across archetypes (no archetype-distinct slot left) or have
lost intra-band variety — both are test-time signals to revisit the
lexicon design.

---

## Lexicon-key gotcha — segment names, not recipe names

The keys in `lexicons:` must be the **engine archetype names**. In the
builder API, those equal the **segment names** from your `segments:`
list (e.g. `"promoters"`, `"detractors"`) — not the recipe keywords
passed to each segment's `archetype:` field (e.g. `"growth"`,
`"decline"`).

The builder picks a recipe via `archetype:` then names the resulting
archetype after the segment. This means two segments can share the
same recipe but speak with different vocabulary — useful when modeling
two cohorts whose underlying behavior matches but whose surface
language doesn't.

The cross-config validator catches archetype-name mismatches at config
load with a message naming both the unknown lexicon keys and the
declared archetypes:

```
table 'fct_reviews' column 'review_text' narrative lexicons declare
unknown archetypes ['decline', 'growth']; declared archetypes:
['fallers', 'risers']
```

---

## Custom band counts

`bands` defaults to `("low", "mid", "high")`. You can override with
any 2–20 unique non-empty band names. The cell-resolution arithmetic
supports any `N >= 1`; the bound is for lexicon-authoring ergonomics
(more than ~5 bands rarely produces distinguishable per-band
vocabulary in practice).

```yaml
- name: support_ticket_text
  type: narrative
  template: "{description}"
  bands: [crisis, escalating, monitoring, healthy, thriving]
  lexicons:
    promoters:
      description:
        crisis:     ["Login broken"]
        escalating: ["Slow on big queries"]
        monitoring: ["Reviewing usage"]
        healthy:    ["No issues"]
        thriving:   ["Loving the new release"]
    # ... one entry per assigned segment ...
```

---

## Validation gates

| Gate | Where | Catches |
|---|---|---|
| Template has ≥ 1 `{slot}` placeholder | `NarrativeConfig` | Static sentences (use `static:` source) |
| No duplicate `{slot}` placeholders | `NarrativeConfig` | Copy-paste mistakes in templates |
| Lexicon archetype keys cover assigned segments | `validate_narrative_columns` | Missing per-segment vocabulary |
| Lexicon archetype keys are subset of declared archetypes | `validate_narrative_columns` | Stale lexicon entries after segment renames |
| Per-archetype slot keys equal template placeholders | `NarrativeConfig` | Slot/template drift |
| Per-slot band keys equal `bands` tuple | `NarrativeConfig` | Missing or extra bands per slot |
| Each band's phrase list non-empty | `NarrativeConfig` | Empty phrase pools |
| `dtype: boolean` rejected | engine validator | `bool("any text")` collapses to True |
| Source only on `per_entity_per_period` fact | `validate_narrative_columns` | Dim / per-period / event placement |

All checks run at config-load — no narrative misconfiguration will
make it past `load_config` / `create` / `plotsim validate`.

---

## Limits and caveats

- **Fact-only.** Dim tables are static and event tables don't have
  per-row trajectory plumbing in this version.
- **Forces the scalar fact-builder path.** Narrative columns consume
  one RNG draw per slot per row, same constraint as `FakerSource`.
  Expect generation to be ~3-10× slower than a vectorized-only fact
  table for large row counts. Mitigation: keep narrative columns to
  the fact tables that genuinely need text and use vectorized facts
  for high-volume metric-only tables.
- **English lexicons only.** Templates and phrase pools are arbitrary
  strings; nothing forbids non-English content, but no built-in
  translation or locale-driven lexicon switching exists.
- **No metric-value-driven slot selection.** Slot phrase choice is a
  function of `(trajectory_position, archetype, RNG_state)` only.
  Metric values for the row are not consulted directly — but since
  the trajectory drives every metric, "metric-driven" text falls out
  via the trajectory shift.
