# CDC fact-side audit columns

> Opt-in `_inserted_at` / `_updated_at` / `_op` columns on fact tables
> for data-engineering workflows that consume change-data-capture (CDC)
> output. Pairs with the existing SCD Type 2 surface on the dim side.

---

## What it does

Flip `cdc: true` on any fact table and plotsim emits three audit
columns at generation time:

| Column | Initial value | Notes |
|---|---|---|
| `_inserted_at` | ISO period string from the row's `date_key` | e.g. `"2024-03"` for monthly grain, `"2024-03-15"` for daily |
| `_updated_at` | same as `_inserted_at` | bumped to the last period's label on rows mutated by a column-level quality issue |
| `_op` | `"I"` | flipped to `"U"` on rows the quality layer corrupts |

CDC-disabled fact tables produce byte-identical output to the pre-CDC
behaviour. The audit columns sit at the end of the column order so
existing user-declared columns stay where they were.

---

## How to enable

=== "YAML"

    ```yaml
    facts:
      - name: fct_billing
        metrics: [mrr, payments]
        cdc: true
        columns:
          - { name: date_key,    type: ref.dim_date }
          - { name: customer_id, type: ref.dim_customer }
          - { name: mrr,         type: metric.mrr }
          - { name: payments,    type: metric.payments }
    ```

=== "Python"

    ```python
    from plotsim import create

    cfg = create(
        # ... about / unit / window / metrics / segments ...
        facts=[
            {
                "name": "fct_billing",
                "metrics": ["mrr", "payments"],
                "cdc": True,
                "columns": [
                    {"name": "date_key",    "type": "ref.dim_date"},
                    {"name": "customer_id", "type": "ref.dim_customer"},
                    {"name": "mrr",         "type": "metric.mrr"},
                    {"name": "payments",    "type": "metric.payments"},
                ],
            },
        ],
    )
    ```

The bundled `cdc_demo` template runs end-to-end:

```bash
plotsim template cdc_demo -o cdc_demo.yaml
plotsim run cdc_demo.yaml -o ./cdc_demo_output
head ./cdc_demo_output/fct_billing.csv
```

---

## Combining CDC with quality issues

When a fact has `cdc: true` AND a column-level quality issue
(`null_injection`, `type_mismatch`, `schema_drift`) targets that
fact, the affected rows flip to:

- `_op = "U"`
- `_updated_at = <last period label>` — semantic: *"the row was
  inserted at its date_key period and then touched again at
  end-of-window when the upstream system corrected itself"*

Row-level issues (`duplicate_rows`, `late_arrival`, `volume_anomaly`)
do **not** flip `_op` — their ground-truth row indices reference the
pre-corruption frame and don't align with the corrupted frame after
the row count shifts. Those quality issues continue to record their
mutations in the manifest's `quality_injections` ground-truth list as
before.

---

## Combining CDC with holdout

CDC and `holdout` are compatible. Both `_train` and `_holdout` files
inherit the audit columns. Note the existing rule: `holdout` requires
`quality.quality_issues == []`, so a holdout-enabled CDC config sees
every row at `_op="I"` (no quality mutations to flip).

---

## Limits and caveats

- CDC is **per-fact-table**. A config can mix CDC-enabled and CDC-disabled
  facts; each fact's columns reflect its own setting.
- Validators reject `cdc: true` on dim, event, and bridge tables. SCD
  Type 2 covers the dim side; events and bridges are out of scope.
- The "last period" `_updated_at` value is a deterministic, simple
  choice — not a configurable mutation timestamp. Downstream consumers
  needing fine-grained mutation timing should derive it from the
  manifest's `quality_injections` records.
- The corrupted CSV on disk has the U-flipped `_op`; the in-memory
  `tables` dict the caller passes to `write_tables` is left clean.

---

## Manifest interaction

The manifest's `quality_injections` field is unaffected by the CDC
flip — it always carries the per-(table, column, row_indices,
clean_values) ground-truth tuples. The CDC columns are an *additional*
surfacing mechanism for the same information, on the row itself.

A consumer that wants to recover the clean values can still read
`quality_injections`; a consumer that just wants to know *which rows
were touched* can scan `_op`.
