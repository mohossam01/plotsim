# Multi-source / overlap mode

Real warehouses are messy because real businesses run *multiple* systems
of record for the same logical entity. A customer in the CRM is the
same customer in the billing platform — except their name has a typo,
their account ID is a different format, and one system thinks they're
in "Manufacturing" while the other has them tagged as "Industrial".
plotsim's multi-source mode generates these divergent views as
ground-truth-backed data so analytics engineers and data engineers can
practice entity resolution, fuzzy matching, MDM, and cross-source
reconciliation against a dataset with a known answer key.

## How it works

When you declare `sources:` on a config, the engine emits the canonical
`dim_<entity>` table **unchanged** plus one additional
`dim_<entity>_<source>` table for every declared source. The per-source
dims carry the same logical entities as the canonical, but with three
configurable flavors of drift applied:

1. **Name drift.** A configurable fraction of entities get a typo on
   their `faker.{name, first_name, last_name, company}` column. The
   engine picks uniformly between three drift kinds — adjacent-character
   swap ("Acme" → "Aceme"), casing flip ("Acme Corp" → "aCME cORP"), and
   initials-style abbreviation ("Acme Industries Inc" → "A.I.I.").
2. **Key scheme drift.** The canonical PK column is renamed to
   `<entity_type>_id_<source>` and repopulated under the source's
   declared key scheme:

    | Scheme           | Example     | Mimics                  |
    |------------------|-------------|-------------------------|
    | `prefix_padded`  | `CUST-001`  | CRM with structured IDs |
    | `numeric`        | `1001`      | billing system          |
    | `uuid_short`     | `c3f9a`     | record-keeping system   |

3. **Attribute drift.** A configurable fraction of entities get a
   conflicting value appended to one eligible string-typed attribute
   column. ("Real Estate" on the canonical dim might become "Real
   Estate (alt-79)" on the CRM dim — the kind of disagreement that
   surfaces when two systems were keyed independently.)

Drift is **dim-only by design**. Facts and events still FK off the
canonical PK, so revenue history, login events, and every other
behavioral series remain consistent regardless of which upstream
system surfaces a given entity. The exercise is "can you tell these
two records refer to the same entity?" — not "are the facts wrong?".

The manifest's `source_entity_mappings` section is the ground-truth
answer key: one record per `(entity, source, dim_table)` carrying the
canonical PK, the source-specific PK, and the list of column names
that drifted on that row. Same seed → identical mappings, so a
correctness-test loop has a stable target.

## Determinism

Per-source RNGs are derived by sequential `integers` draws on the
dim-build RNG in **config declaration order**. So the same `(seed,
sources)` pair always produces byte-identical per-source dims, and
toggling source order shifts each source's drift to a different
deterministic offset without coupling across sources. Adding or
removing a source changes the RNG stream for every source declared
after it (one draw per source, per the design note in
`plotsim/multi_source.py`).

## Quickstart — builder

Declare a `sources:` block alongside your usual segments and schema:

```python
from plotsim import create

config = create(
    about="CRM + billing platform overlap",
    unit="company",
    seed=2026,
    window=("2024-01", "2024-12", "monthly"),
    metrics=[
        {"name": "arr", "type": "amount", "polarity": "positive", "range": [0, 500_000]},
    ],
    segments=[
        {"name": "enterprise", "count": 8, "archetype": "growth"},
        {"name": "smb", "count": 12, "archetype": "spike_then_crash"},
    ],
    sources=[
        {
            "name": "crm",
            "key_scheme": "prefix_padded",
            "name_drift_rate": 0.15,
            "attribute_drift_rate": 0.20,
        },
        {
            "name": "billing",
            "key_scheme": "numeric",
            "name_drift_rate": 0.35,
            "attribute_drift_rate": 0.30,
        },
    ],
    dimensions=[
        {
            "name": "dim_company",
            "per": "unit",
            "columns": [
                {"name": "company_id", "type": "id"},
                {"name": "company_name", "type": "faker.company"},
                {"name": "industry", "type": "faker.industry"},
            ],
        },
    ],
)
```

Output:

```
dim_date.csv
dim_company.csv          # canonical
dim_company_crm.csv      # CRM-shaped, 15% name typos, 20% industry conflicts
dim_company_billing.csv  # billing-shaped, 35% name typos, 30% industry conflicts
fct_revenue.csv
manifest.json            # carries source_entity_mappings
```

YAML form mirrors the keyword form 1:1; see
`plotsim/configs/templates/crm_billing_overlap.yaml` for the bundled
template.

## Entity resolution walkthrough

The canonical use case is teaching record linkage. Load the bundled
template, generate, then write a notebook that joins `dim_company_crm`
to `dim_company_billing` and scores its predictions against the
manifest's `source_entity_mappings`:

```python
import json
import pandas as pd
import plotsim

cfg = plotsim.load_template("crm_billing_overlap")
tables = plotsim.generate_tables(cfg)

crm = tables["dim_company_crm"]
billing = tables["dim_company_billing"]

# Naive blocking + scoring — improve this!
candidates = crm.merge(billing, how="cross", suffixes=("_crm", "_billing"))
candidates["name_match"] = candidates.apply(
    lambda r: r["company_name_crm"].lower() == r["company_name_billing"].lower(),
    axis=1,
)
candidates["industry_match"] = (
    candidates["industry_crm"] == candidates["industry_billing"]
)
predictions = candidates.query("name_match or industry_match")

# Now score against ground truth.
mappings = pd.DataFrame(
    [m.model_dump() for m in plotsim.build_manifest(cfg, tables).source_entity_mappings]
)
crm_ids = mappings.query("source == 'crm'").set_index("entity")["source_entity_id"]
billing_ids = mappings.query("source == 'billing'").set_index("entity")["source_entity_id"]
truth = pd.DataFrame({"crm_id": crm_ids, "billing_id": billing_ids}).reset_index()

# Compute precision, recall, F1 against ``truth``.
```

The interesting part is that this script *fails* — naive case-sensitive
exact-match scoring misses every entity that got casing or typo drift
on the CRM side. The teaching exercise is to layer on fuzzy matching,
phonetic encoding, or learned similarity until the precision/recall
curve climbs. The mappings give the same stable target seed-to-seed,
so iteration is rewarding.

## Field reference

```yaml
sources:
  - name: crm                     # SQL-safe identifier; suffix for dim_<entity>_<name>
    key_scheme: prefix_padded     # one of prefix_padded | numeric | uuid_short
    name_drift_rate: 0.15         # fraction of entities with a name typo on this source (0–1)
    attribute_drift_rate: 0.20    # fraction with one conflicting attribute value (0–1)
```

Rules enforced at load:

* 2–5 sources per config. 1 is degenerate (no overlap); >5 moves out
  of the teaching range called out in the multi-source design spec.
* Source names must be unique within the block.
* Source names must not collide with any existing dim table name
  (the emitted `dim_<entity>_<source>` would shadow a real table).
* `multi_source` requires at least one `per_entity` dim — there's
  nothing to overlay drift on otherwise.

## Out of scope (today)

* **Fact-table drift.** Drift is dim-only; facts are unchanged. A
  future release could optionally divergence-tag fact-side audit
  columns, but the current spec keeps the contract simple.
* **Probabilistic record linkage.** The manifest carries binary
  match / no-match ground truth, not similarity scores. If your
  exercise grades on a continuous metric, derive it from the
  binary truth.
* **More than 5 sources.** The teaching value drops sharply past
  three sources; the cap keeps generated dim sets readable.
* **CDC / SCD interaction.** Each per-source dim is a snapshot;
  there's no per-source SCD history or CDC `_op` column. Future
  work if the demand surfaces.
