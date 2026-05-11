"""CRM + billing overlap — Python-shaped multi-source / overlap template.

This is the ``create(**kwargs)`` mirror of ``crm_billing_overlap.yaml`` —
both produce identical engine configs given the same seed. Pick whichever
surface fits your workflow:

* ``crm_billing_overlap.yaml`` for config-as-data fixtures checked into git
* this file for code-shaped configs that compose with regular Python

Showcase template for the multi-source / overlap mode (0.6-M13):

  * Two upstream systems describe the same 20 customer accounts:

      - ``crm`` mimics a sales CRM with ``CUST-001`` style keys and a
        moderate typo rate (15% of names show drift; 20% of industry
        attributions disagree with the canonical record).
      - ``billing`` mimics a downstream billing platform with numeric
        keys and a heavier drift rate (35% of names, 30% of industries)
        — naming conventions diverge more sharply once data has been
        re-keyed at integration time.

  * The canonical ``dim_company`` is emitted untouched. ``dim_company_crm``
    and ``dim_company_billing`` each carry the drifted copies; the
    manifest's ``source_entity_mappings`` section is the ground-truth
    answer key — exactly which canonical entity corresponds to each
    per-source ID, and which fields drifted on that row.

  * Fact tables (``fct_revenue``) FK off the canonical ``dim_company``
    PK — drift is dim-only by design, so revenue history remains
    consistent across the operator's mental model regardless of which
    upstream system surfaces a given account.

Domain narrative: a B2B SaaS company sells annual contracts to 20
accounts split between an enterprise cohort (steady growth, 8 entities)
and an SMB cohort (volatile, 12 entities). Sales tracks accounts in the
CRM; finance reconciles them in the billing platform. Six months later
an analytics engineer is asked: "how many accounts do we have?" The
correct answer depends on knowing which CRM customer is which billing
customer — the data the manifest's mapping table provides as ground
truth so the operator can build (and grade) an entity-resolution
pipeline against it.

Run: ``plotsim run crm_billing_overlap`` produces CSVs for ``dim_date``,
``dim_company``, ``dim_company_crm``, ``dim_company_billing``, plus
``fct_revenue``. The accompanying ``manifest.json`` carries the
``source_entity_mappings`` answer key.
"""

from plotsim import create


config = create(
    about="CRM + billing platform overlap — multi-source mode demo",
    unit="company",
    seed=2026,
    window=("2024-01", "2024-12", "monthly"),
    metrics=[
        {
            "name": "arr",
            "label": "Annual recurring revenue",
            "type": "amount",
            "polarity": "positive",
            "range": [0, 500_000],
        },
        {
            "name": "seats",
            "label": "Provisioned seats",
            "type": "count",
            "polarity": "positive",
        },
    ],
    segments=[
        {
            "name": "enterprise",
            "count": 8,
            "archetype": "growth",
            "label": "Enterprise accounts (steady expansion)",
        },
        {
            "name": "smb",
            "count": 12,
            "archetype": "spike_then_crash",
            "label": "SMB accounts (volatile, churn-prone)",
        },
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
            "name": "dim_company",
            "per": "unit",
            "columns": [
                {"name": "company_id", "type": "id"},
                {"name": "company_name", "type": "faker.company"},
                {"name": "industry", "type": "faker.industry"},
            ],
        },
    ],
    facts=[
        {
            "name": "fct_revenue",
            "columns": [
                {"name": "date_key", "type": "ref.dim_date"},
                {"name": "company_id", "type": "ref.dim_company"},
                {"name": "arr", "type": "metric.arr"},
                {"name": "seats", "type": "metric.seats"},
            ],
        },
    ],
)
