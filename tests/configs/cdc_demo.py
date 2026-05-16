"""CDC fact-side demo — Python form.

Mirror of ``cdc_demo.yaml``. A subscription-billing fact table with
``cdc: True`` and a 5% null injection on ``mrr`` so the on-disk
``fct_billing.csv`` shows ``_op: "U"`` rows for the cells the
quality layer corrupted.

Run:
    >>> from tests.configs.cdc_demo import config
    >>> from plotsim import generate_tables, write_tables
    >>> tables = generate_tables(config)
    >>> write_tables(tables, config, output_dir="./cdc_demo_output")
"""

from plotsim import create

config = create(
    about="Subscription billing — CDC fact-side audit columns",
    unit="customer",
    seed=42,
    window=("2024-01", "2024-12", "monthly"),
    metrics=[
        {"name": "mrr", "type": "amount", "polarity": "positive", "range": [10, 5000]},
        {"name": "payments", "type": "count", "polarity": "positive"},
    ],
    segments=[
        {"name": "growers", "count": 30, "archetype": "growth"},
        {"name": "decliners", "count": 20, "archetype": "decline"},
    ],
    dimensions=[
        {
            "name": "dim_date",
            "per": "period",
            "columns": [
                {"name": "date_key", "type": "id"},
                {"name": "date", "type": "date"},
            ],
        },
        {
            "name": "dim_customer",
            "per": "unit",
            "columns": [
                {"name": "customer_id", "type": "id"},
                {"name": "customer_name", "type": "faker.name"},
            ],
        },
    ],
    facts=[
        {
            "name": "fct_billing",
            "metrics": ["mrr", "payments"],
            "cdc": True,
            "columns": [
                {"name": "date_key", "type": "ref.dim_date"},
                {"name": "customer_id", "type": "ref.dim_customer"},
                {"name": "mrr", "type": "metric.mrr"},
                {"name": "payments", "type": "metric.payments"},
            ],
        },
    ],
    quality=[
        {"table": "fct_billing", "issue": "null_injection", "rate": 0.05, "column": "mrr"},
    ],
)
