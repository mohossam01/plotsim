"""Banking template — Python form.

Mirror of ``banking.yaml``. Retail banking + credit risk with student-t
noise on transactions, SCD2 credit score band, parent/child loan
applications + documents, M:N customer × product bridge, CDC on loan
disbursements, geo bundle on customer branches, narrative loan-officer
notes, treatment cohort for credit-line increase, and a holdout split
for credit-scoring validation.

Run:
    >>> from plotsim.configs.templates.banking_template import config
    >>> from plotsim import generate_tables
    >>> tables = generate_tables(config)
"""

from plotsim import create


_NAR_BLOCK = {
    "stem": {
        "low": ["Some concerns about", "Watch closely on", "Risk indicators rising on"],
        "mid": ["Account performing as expected for", "Steady profile on", "Routine review for"],
        "high": [
            "Strong performance for",
            "Top-tier credit profile for",
            "Excellent payment history on",
        ],
    },
    "assessment": {
        "low": ["recent utilization spikes", "missed payments in cycle", "behavior outside norms"],
        "mid": ["expected pattern", "consistent usage", "no anomalies"],
        "high": ["disciplined credit use", "low utilization", "consistent on-time payments"],
    },
    "action": {
        "low": [
            "Recommend credit line freeze.",
            "Flag for collections review.",
            "Reduce exposure.",
        ],
        "mid": ["Continue routine monitoring.", "Annual review next cycle.", "No action."],
        "high": [
            "Eligible for credit line increase.",
            "Offer premium products.",
            "Cross-sell opportunity.",
        ],
    },
}


config = create(
    about="Retail banking — accounts, loans, transactions, credit risk",
    unit="customer",
    seed=51231,
    noise={
        "gaussian_sigma": 0.05,
        "outlier_rate": 0.01,
        "mcar_rate": 0.0,
        "noise_family": "student_t",
        "degrees_of_freedom": 4.0,
    },
    window=("2023-01", "2024-12", "monthly"),
    seasonality=[
        {"months": [11, 12], "strength": 0.30},
        {"months": [3, 4], "strength": 0.20},
        {"months": [6, 7], "strength": -0.10},
    ],
    metrics=[
        {
            "name": "account_balance",
            "label": "Account balance",
            "type": "amount",
            "polarity": "positive",
            "range": [0, 250000],
        },
        {
            "name": "transaction_volume",
            "label": "Transactions per period",
            "type": "count",
            "polarity": "positive",
        },
        {
            "name": "credit_utilization",
            "label": "Credit utilization rate",
            "type": "score",
            "polarity": "negative",
        },
        {
            "name": "payment_on_time",
            "label": "On-time payment ratio",
            "type": "score",
            "polarity": "positive",
        },
        {
            "name": "delinquency_risk",
            "label": "Delinquency risk score",
            "type": "score",
            "polarity": "negative",
            "follows": "credit_utilization",
            "delay": 2,
        },
        {
            "name": "loan_volume",
            "label": "New loan applications",
            "type": "count",
            "polarity": "positive",
        },
        {
            "name": "default_risk",
            "label": "Loan default risk",
            "type": "score",
            "polarity": "negative",
        },
    ],
    connections=[
        "delinquency_risk related credit_utilization",
        "payment_on_time opposes delinquency_risk",
        "default_risk 0.55 delinquency_risk",
        "account_balance 0.40 payment_on_time",
        "transaction_volume related account_balance",
    ],
    segments=[
        {
            "name": "prime_borrower",
            "count": 22,
            "archetype": "flat",
            "label": "Stable low-risk borrowers",
            "attributes": {
                "account_type": ["checking", "savings", "credit_card", "mortgage"],
                "employment_status": ["employed_full_time", "self_employed"],
                "income_band": ["80k_120k", "120k_200k", "200k_plus"],
                "product_category": ["checking", "savings", "mortgage", "credit_card"],
            },
            "baseline": {
                "account_balance": "high",
                "payment_on_time": "high",
                "default_risk": "low",
            },
        },
        {
            "name": "subprime_improving",
            "count": 18,
            "archetype": "decline",
            "label": "Subprime customers with declining risk over the window",
            "attributes": {
                "account_type": ["checking", "credit_card"],
                "employment_status": ["employed_full_time", "employed_part_time", "contract"],
                "income_band": ["under_40k", "40k_80k"],
                "product_category": ["checking", "credit_card", "personal_loan"],
            },
            "baseline": {"credit_utilization": "high", "default_risk": "high"},
        },
        {
            "name": "mass_market",
            "count": 24,
            "archetype": "flat",
            "label": "Stable mid-market accounts",
            "attributes": {
                "account_type": ["checking", "savings", "credit_card"],
                "employment_status": ["employed_full_time", "self_employed", "employed_part_time"],
                "income_band": ["40k_80k", "80k_120k"],
                "product_category": ["checking", "savings", "credit_card", "auto_loan"],
            },
            "baseline": {"account_balance": "mid", "transaction_volume": "mid"},
        },
        {
            "name": "deteriorating",
            "count": 12,
            "archetype": "flat > growth > spike_then_crash @ 8 @ 16",
            "label": "Deteriorating credit — risk rising into default",
            "attributes": {
                "account_type": ["credit_card", "personal_loan"],
                "employment_status": ["employed_part_time", "contract", "unemployed"],
                "income_band": ["under_40k", "40k_80k"],
                "product_category": ["credit_card", "personal_loan"],
            },
            "baseline": {"credit_utilization": "high", "delinquency_risk": "high"},
        },
        {
            "name": "hnw",
            "count": 8,
            "archetype": "accelerating",
            "label": "High-net-worth growing balances and product depth",
            "attributes": {
                "account_type": ["savings", "mortgage", "brokerage"],
                "employment_status": ["employed_full_time", "self_employed"],
                "income_band": ["200k_plus"],
                "product_category": ["checking", "savings", "mortgage", "brokerage", "credit_card"],
            },
            "baseline": {"account_balance": "high", "transaction_volume": "high"},
        },
        {
            "name": "new_customer",
            "count": 14,
            "archetype": "flat > growth @ 5",
            "label": "Newly onboarded, building credit history",
            "attributes": {
                "account_type": ["checking", "savings", "credit_card"],
                "employment_status": ["employed_full_time", "employed_part_time"],
                "income_band": ["40k_80k", "80k_120k"],
                "product_category": ["checking", "savings", "credit_card"],
            },
            "baseline": {"account_balance": "low", "payment_on_time": "mid"},
            "treatment": {
                "fraction": 0.5,
                "lift_log_odds": 0.4,
                "start_period": 6,
                "treatment_label": "credit_line_increase",
                "control_label": "standard_credit_line",
            },
        },
    ],
    lifecycle={
        "track": "default_risk",
        "stages": [{"performing": 0.0}, {"watch": 0.3}, {"past_due": 0.55}, {"default": 0.8}],
    },
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
            "name": "dim_customer",
            "per": "unit",
            "columns": [
                {"name": "customer_id", "type": "id"},
                {"name": "customer_name", "type": "faker.name"},
                {"name": "customer_email", "type": "faker.email"},
                {"name": "onboarding_year", "type": "faker.year"},
                {"name": "cohort_size", "type": "segment.count"},
                {"name": "employment_status", "type": "pool.employment_status"},
                {"name": "income_band", "type": "pool.income_band"},
                {"name": "account_type", "type": "pool.account_type"},
                {"name": "branch_country", "type": "geo.country"},
                {"name": "branch_country_code", "type": "geo.country_code"},
                {"name": "branch_region", "type": "geo.region"},
                {"name": "branch_city", "type": "geo.city"},
                {
                    "name": "credit_score_band",
                    "type": "scd",
                    "tracks": "default_risk",
                    "tiers": ["super_prime", "prime", "near_prime", "subprime"],
                    "at": [0.25, 0.55, 0.8],
                },
            ],
        },
        {
            "name": "dim_product",
            "reference": True,
            "columns": [
                {"name": "product_id", "type": "id"},
                {
                    "name": "product_name",
                    "type": "static.checking,savings,credit_card,personal_loan,auto_loan,mortgage,brokerage,heloc",
                },
                {
                    "name": "product_class",
                    "type": "static.deposit,deposit,credit,credit,credit,credit,investment,credit",
                },
            ],
        },
        {
            "name": "dim_merchant_category",
            "reference": True,
            "columns": [
                {"name": "category_id", "type": "id"},
                {
                    "name": "category_name",
                    "type": "static.grocery,fuel,dining,travel,utilities,entertainment,healthcare,retail,subscription,cash_advance",
                },
            ],
        },
    ],
    facts=[
        {
            "name": "fct_account_activity",
            "metrics": [
                "account_balance",
                "transaction_volume",
                "credit_utilization",
                "payment_on_time",
                "delinquency_risk",
                "default_risk",
                "loan_volume",
            ],
            "columns": [
                {"name": "date_key", "type": "ref.dim_date"},
                {"name": "customer_id", "type": "ref.dim_customer"},
                {"name": "account_balance", "type": "metric.account_balance"},
                {"name": "transaction_volume", "type": "metric.transaction_volume"},
                {"name": "credit_utilization", "type": "metric.credit_utilization"},
                {"name": "payment_on_time", "type": "metric.payment_on_time"},
                {"name": "delinquency_risk", "type": "metric.delinquency_risk"},
                {"name": "default_risk", "type": "metric.default_risk"},
                {"name": "loan_volume", "type": "metric.loan_volume"},
                {
                    "name": "loan_officer_notes",
                    "type": "narrative",
                    "template": "{stem} {assessment}. {action}",
                    "lexicons": {
                        "prime_borrower": _NAR_BLOCK,
                        "subprime_improving": _NAR_BLOCK,
                        "mass_market": _NAR_BLOCK,
                        "deteriorating": _NAR_BLOCK,
                        "hnw": _NAR_BLOCK,
                        "new_customer": _NAR_BLOCK,
                    },
                },
            ],
        },
        {
            "name": "fct_loan_applications",
            "row_count_driver": "loan_volume",
            "row_count_scale": 1.0,
            "cdc": True,
            "columns": [
                {"name": "application_id", "type": "id"},
                {"name": "customer_id", "type": "ref.dim_customer"},
                {"name": "application_date", "type": "ref.dim_date"},
                {
                    "name": "loan_purpose",
                    "type": "static.mortgage,auto,education,personal,business,debt_consolidation,home_improvement,medical",
                },
                {"name": "requested_amount", "type": "range", "range": [5000, 750000]},
                {"name": "interest_rate", "type": "range", "range": [3.5, 24.9]},
            ],
        },
        {
            "name": "fct_loan_documents",
            "parent_table": "fct_loan_applications",
            "children_per_row": [1, 3],
            "columns": [
                {"name": "document_id", "type": "id"},
                {"name": "customer_id", "type": "ref.dim_customer"},
                {"name": "application_date", "type": "ref.dim_date"},
                {
                    "name": "document_type",
                    "type": "static.income_proof,id_verification,bank_statement,tax_return,property_appraisal,collateral_doc",
                },
                {
                    "name": "doc_status",
                    "type": "static.received,received,received,pending,verified",
                },
            ],
        },
    ],
    events=[
        {
            "name": "evt_transaction",
            "trigger": "proportional",
            "driver": "transaction_volume",
            "scale": 5.0,
            "columns": [
                {"name": "event_id", "type": "id"},
                {"name": "date_key", "type": "ref.dim_date"},
                {"name": "customer_id", "type": "ref.dim_customer"},
                {"name": "category_id", "type": "ref.dim_merchant_category"},
                {"name": "txn_amount", "type": "range", "range": [1.0, 5000.0]},
                {"name": "event_ts", "type": "timestamp"},
            ],
        },
        {
            "name": "evt_default",
            "trigger": "threshold",
            "metric": "default_risk",
            "above": 0.65,
            "for": 2,
            "columns": [
                {"name": "event_id", "type": "id"},
                {"name": "date_key", "type": "ref.dim_date"},
                {"name": "customer_id", "type": "ref.dim_customer"},
                {"name": "severity", "type": "static.cured,charged_off,bankruptcy,settled"},
                {"name": "voluntary", "type": "flag"},
            ],
        },
    ],
    bridges=[
        {
            "name": "bridge_customer_product",
            "left": "dim_customer",
            "right": "dim_product",
            "cardinality": [1, 5],
            "driver": "account_balance",
        },
    ],
    holdout={"target": "default_risk", "periods": 3},
)
