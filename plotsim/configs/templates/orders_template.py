"""Parent/child fact grain — Python form.

Mirror of ``orders_template.yaml``. Demonstrates the 0.6-M18 parent/child
fact pattern: a per_entity_per_period activity fact exposes the
``order_volume`` metric that drives a variable-grain parent
(``fct_orders``), and a per_parent_row child (``fct_order_items``) fans
out 1..5 line items per order.

Run:
    >>> from plotsim.configs.templates.orders_template import config
    >>> from plotsim import generate_tables, write_tables
    >>> tables = generate_tables(config)
    >>> write_tables(tables, config, output_dir="./orders_output")
"""

from plotsim import create

config = create(
    about="Retail orders with line-item detail (parent/child fact grain)",
    unit="customer",
    seed=18180,
    window=("2024-01", "2024-06", "monthly"),
    metrics=[
        {"name": "order_volume", "type": "count", "polarity": "positive"},
        {"name": "loyalty_score", "type": "score", "polarity": "positive"},
        # `return_rate` drives the sibling fct_returns row count.
        # Decliners return more often (negative polarity).
        {"name": "return_rate", "type": "score", "polarity": "negative"},
    ],
    connections=[
        "order_volume driven_by loyalty_score",
        "return_rate opposes loyalty_score",
    ],
    segments=[
        {"name": "regulars", "count": 8, "archetype": "growth"},
        {"name": "occasional", "count": 6, "archetype": "flat"},
        {"name": "churning", "count": 4, "archetype": "decline"},
    ],
    dimensions=[
        {
            "name": "dim_customer",
            "per": "unit",
            "columns": [
                {"name": "customer_id", "type": "id"},
                {"name": "customer_name", "type": "faker.name"},
                {"name": "customer_email", "type": "faker.email"},
            ],
        },
        {
            "name": "dim_product",
            "reference": True,
            "columns": [
                {"name": "product_id", "type": "id"},
                {
                    "name": "product_name",
                    "type": "static.widget,gadget,gizmo,sprocket,lantern,doohickey,thingamajig,whatsit",
                },
                {
                    "name": "category",
                    "type": "static.hardware,hardware,hardware,hardware,outdoor,misc,misc,misc",
                },
            ],
        },
    ],
    facts=[
        {
            "name": "fct_orders",
            "row_count_driver": "order_volume",
            "row_count_scale": 1.2,
            "columns": [
                {"name": "order_id", "type": "id"},
                {"name": "customer_id", "type": "ref.dim_customer"},
                {"name": "order_date", "type": "ref.dim_date"},
                {"name": "payment_method", "type": "faker.company"},
            ],
        },
        {
            "name": "fct_order_items",
            "parent_table": "fct_orders",
            "children_per_row": [1, 5],
            # The parent FK column (named ``order_id``, matching
            # fct_orders' PK) is auto-synthesized by the engine at
            # generation time. Do not declare it here.
            "columns": [
                {"name": "item_id", "type": "id"},
                {"name": "customer_id", "type": "ref.dim_customer"},
                {"name": "order_date", "type": "ref.dim_date"},
                {"name": "product_id", "type": "ref.dim_product"},
                {"name": "quantity", "type": "faker.random_int"},
                {"name": "unit_price", "type": "faker.pyfloat"},
            ],
        },
        # ── sibling-fact reference: returns reference orders ────
        # fct_returns is its own variable-grain fact (one row per
        # return, count driven by `return_rate`). Each row references
        # an order via ``ref.fct_orders``; the engine resolves the FK
        # by drawing same-entity-filtered from the customer's orders.
        {
            "name": "fct_returns",
            "row_count_driver": "return_rate",
            "row_count_scale": 0.6,
            "columns": [
                {"name": "return_id", "type": "id"},
                {"name": "order_id", "type": "ref.fct_orders"},
                {"name": "customer_id", "type": "ref.dim_customer"},
                {"name": "return_date", "type": "ref.dim_date"},
                {"name": "return_reason", "type": "faker.word"},
            ],
        },
    ],
)
