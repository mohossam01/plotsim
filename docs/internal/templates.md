# Templates

plotsim ships with six builder templates at `plotsim/configs/new/`. Each is a complete, production-shaped builder YAML you can edit. Pick the one closest to your use case, copy it out with `plotsim template <name> -o <path>`, and modify from there.

```text
$ plotsim list-templates
Builder templates (recommended — front door for new users):
  bare_minimum  Smallest valid builder config — start here
  education     Builder shape: University course enrollment
  hr            Builder shape: HR engagement / attrition
  marketing     Builder shape: Marketing campaigns / attribution
  retail        Builder shape: Retail transactions / loyalty
  saas          Builder shape: B2B SaaS customer success

Engine-direct templates (full PlotsimConfig YAML):
  ...
```

The builder templates load through `create_from_yaml()` and produce a fully validated `PlotsimConfig` after the interpreter fills in defaults. The engine-direct templates (`plotsim/configs/sample_*.yaml`) feed `load_config()` directly and are useful when you need to override a field the builder vocabulary doesn't surface yet.

Quick comparison — the six builder templates:

| Template       | Domain                  | Unit       | Segments | Window  | Stages | Causal lag |
| -------------- | ----------------------- | ---------- | :------: | :-----: | :----: | :--------: |
| `bare_minimum` | Subscription customers  | customer   |    2     | 12 mo   |        |            |
| `saas`         | B2B SaaS customer success | company  |    6     | 24 mo   |   ✓    |     ✓      |
| `hr`           | HR talent and attrition | employee   |    6     | 24 mo   |   ✓    |     ✓      |
| `education`    | University academics    | student    |    6     | 24 mo   |   ✓    |     ✓      |
| `retail`       | Retail purchase + loyalty | customer |    6     | 24 mo   |   ✓    |     ✓      |
| `marketing`    | Campaign performance    | campaign   |    7     | 24 mo   |   ✓    |     ✓      |

`saas` is the canonical reference — it exercises every primitive in the builder vocabulary (segments, archetypes, connections, lifecycle, dimensions, facts, events, SCD Type 2 columns, threshold + proportional events, attribute pools, baselines). The other five are domain twins of comparable depth. `bare_minimum` is the smallest config that loads — start here when you want to learn the surface incrementally.

---

## bare_minimum — start here

**What it models.** Two segments of subscribers, one growing and one declining, across a single year. No correlations, no lifecycle, no schema overrides — the interpreter fills in `dim_date`, `dim_customer`, and a single `fct_customer` table.

```yaml
about: "Subscription customers"
unit: customer

window:
  start: 2024-01
  end: 2024-12
  every: monthly

metrics:
  - {name: engagement, type: score, polarity: positive}
  - {name: payments,   type: count, polarity: positive}

segments:
  - {name: active,   count: 50, archetype: growth}
  - {name: inactive, count: 30, archetype: decline}
```

That's the smallest input the builder accepts. Run `plotsim run bare_minimum.yaml` and you get a date dim, a customer dim with 80 rows, and a per-(customer, period) fact table.

**Why this template.** Best on-ramp for the builder vocabulary. Read this one first to understand which fields are required and which are interpreter-filled.

---

## saas — B2B SaaS customer success

**What it models.** Customer accounts moving through onboarding → active → at-risk → churned. Engagement drives revenue; engagement collapses drive churn. Support ticket volume lags engagement by 2 periods.

**Segments** (95 entities total): `promising_client` (20, `growth > spike_then_crash > flat`), `steady_enterprise` (25, `growth`), `slow_churn` (15, `flat > decline`), `seasonal_accounts` (15, `growth > seasonal`), `dormant` (10, `flat`), `turnaround` (10, `decline > flat > growth`).

**Metrics** (6): `engagement` (score), `mrr` (amount), `support_tickets` (count, negative, lags engagement by 2), `feature_adoption` (score), `churn_risk` (score, negative), `nps` (index, range -100..100).

**Lifecycle.** Four stages keyed on `churn_risk`: `onboarding` → `active` → `at_risk` → `churned`.

**Schema.** 4 dim + 3 fact + 2 event. `dim_company` carries an SCD Type 2 `plan_tier` column tracking `mrr`. `evt_login` is proportional to `engagement`; `evt_churn` is threshold on `churn_risk`.

**Why this template.** Every primitive the builder offers is exercised. Read it end-to-end to learn the full vocabulary, including SCD2, baselines, attribute pools, and both event triggers.

---

## hr — HR talent and attrition analytics

**What it models.** Employees across performance, engagement, training hours, absence, attrition risk, compensation. Engagement leads attrition risk; absence rate lags engagement by 1 month.

**Segments** (95 entities): `new_hire_ramp` (20, `flat > growth @ 6`), `core_team` (30, `flat`), `fast_riser` (12, `accelerating`), `quiet_quitter` (15, `decline`), `burnout_cohort` (8, `growth > spike_then_crash`), `comeback` (10, `decline > flat > growth`).

**Metrics.** Performance, engagement, training_hours, absence_rate (lags engagement, negative), attrition_risk (negative), compensation (amount, range 4000..25000).

**Connections.** `engagement driven_by performance_score`; `engagement opposes attrition_risk`; `absence_rate related attrition_risk`.

**Why this template.** Cleanest example of the `flat > growth @ N` and `decline > flat > growth` archetype shapes. Use it when your domain has tenure semantics or a learning curve.

---

## education — University academics

**What it models.** Student cohorts across assignment scores, attendance, study hours, participation, dropout risk, and stress. Stress level lags study hours by 1 period.

**Segments** (100 entities): `high_achievers` (25, `growth`), `late_bloomers` (20, `flat > growth @ 8`), `early_peakers` (15), `at_risk` (18), `exam_burnout` (10), `seasonal_engagement` (12, `seasonal`).

**Metrics.** Assignment_score, attendance_rate, study_hours (count), participation, dropout_risk (negative), stress_level (negative, lags study_hours).

**Why this template.** The `late_bloomers` (`flat > growth`) and `exam_burnout` (`growth > spike_then_crash`) archetypes mirror each other — the resulting `assignment_score` columns trace clearly opposite curves. Best demo of how archetype shape propagates into output.

---

## retail — Customer purchase and loyalty

**What it models.** Customer segments across sessions, conversion, cart value, returns, loyalty, repeat purchases, NPS. Repeat purchase rate lags loyalty by 1 period.

**Segments** (110 entities): `loyal_climbers` (25, `growth`), `holiday_shoppers` (30, `seasonal`), `cooled_off` (18, `decline`), `one_and_done` (15, `growth > spike_then_crash`), `winback` (12, `decline > flat > growth`), `escalating_basket` (10, `accelerating`).

**Metrics.** Session_count (count), conversion_rate, cart_value (amount), return_rate (negative), loyalty_score, repeat_purchase_rate (lags loyalty), nps (index).

**Why this template.** Best showcase of the `seasonal` archetype on a real domain (Q4 holiday cycles). Use it when your domain has periodic demand.

---

## marketing — Campaign performance and attribution

**What it models.** Marketing campaigns across spend, impressions, click-through, conversion, bounce rate, revenue, ROI, MQLs. Leads_generated lags impressions by 1 period.

**Segments** (93 entities, 7 segments): `awareness_builder` (15, `growth`), `paid_burst` (18, `growth > spike_then_crash`), `seasonal_promo` (20, `seasonal`), `delayed_breakthrough` (12, `flat > growth`), `viral_compound` (10, `accelerating`), `end_of_life` (8, `decline`), `retarget_revival` (10, `decline > flat > growth`).

**Metrics.** Ad_spend (amount), impressions (count), click_through_rate, conversion_rate, bounce_rate (negative), revenue (amount, range 0..250k), roi (index, range -1..5), leads_generated (count, lags impressions).

**Why this template.** The richest connection graph among the bundled set — six causal links chain spend → impressions → CTR → conversion → revenue → ROI. Use it when your domain has a measured funnel.

---

## Adapting a template

Once you've copied a template to a local file:

1. **Rename the domain.** `about` and `unit` are display + dim-table strings — `unit: company` produces `dim_company`, `unit: campaign` produces `dim_campaign`.
2. **Resize segments.** Bump or shrink `segments[*].count`. Hard cap is 5,000 per segment and 100,000 across all segments.
3. **Shift the time window.** Change `window.start` / `end` (YYYY-MM strings). Caps: 360 monthly / 1,560 weekly / 3,650 daily periods.
4. **Reassign archetypes.** Change `segments[*].archetype` to a different recipe — see [`docs/builder-reference.md`](builder-reference.md) for the full catalog.
5. **Add a metric.** Append to `metrics:`. The interpreter auto-routes it to the right fact table by `unit`. Add it to a fact's `columns:` block to project a custom column name.

If LLM-driven authoring is more your speed, paste the YAML into any chat and ask for the change you want — the builder vocabulary is plain English enough that this works well. Run `plotsim validate` on the result before generating.

---

## Where to next

- **[Builder quickstart](builder-quickstart.md)** — annotated walkthrough of the bare-minimum and saas templates
- **[Builder reference](builder-reference.md)** — every keyword in the builder vocabulary
- **[Builder errors](builder-errors.md)** — every validation error, mapped to cause and fix
- **[Config guide](config-guide.md)** — the engine-direct schema for advanced overrides
- **[Reference](reference.md)** — full CLI, full schema, changelog
