# Builder Errors and Warnings

> Every structural error the builder can emit, every conditional-field
> violation, every vocabulary check, every DSL parse failure, and every
> semantic warning — with a minimal triggering input and the fix.
>
> All errors block `plotsim.create()` / `plotsim.create_from_yaml()`.
> All warnings allow construction to succeed; they flag a choice the
> user can defend, not a bug.
>
> Use this page for two purposes: human debugging, and inline error
> rendering in UI tooling. Each entry has a stable shape — message
> pattern, triggering input, fix.

---

## How errors are raised

Structural problems raise `pydantic.ValidationError`. Inside the
ValidationError, each error has a `loc` tuple naming the offending field,
a `msg` (the wording below), and a `type` tag. The builder also defines
`plotsim.builder.ArchetypeParseError` (a `ValueError` subclass) wrapped
into the ValidationError for archetype DSL failures.

Warnings are `UserWarning` emitted via `warnings.warn(...)` with
`stacklevel=2` so the call site (your `create(...)` line) is the warning
origin.

---

## 1. Required fields

Pydantic emits `Field required` for any of these top-level omissions.
The minimum a config needs is `about`, `unit`, `window`, `metrics`,
`segments`. The auto-generation path covers schema and lifecycle.

| What's missing                  | Triggering input                                                                  | Fix                                                                                  |
|---------------------------------|-----------------------------------------------------------------------------------|--------------------------------------------------------------------------------------|
| `about`                         | `create(unit="x", window=("2024-01","2024-12"), metrics=[...], segments=[...])`   | Add `about: "<one-line domain description>"`.                                        |
| `unit`                          | omit `unit`                                                                       | Add `unit: <singular noun>` — drives `dim_{unit}` table naming.                       |
| `window`                        | omit `window`                                                                     | Add `window: {start, end, every}` or a `(start, end, every)` tuple.                  |
| `window.start` / `window.end`   | `window: {every: monthly}`                                                        | Both ends must be present. `every` defaults to `monthly`.                            |
| `metrics`                       | `metrics: []` or omit                                                             | Declare at least one metric.                                                         |
| `segments`                      | `segments: []` or omit                                                            | Declare at least one segment.                                                        |
| `metric[*].name` / `.type` / `.polarity` | `metrics: [{type: score}]`                                               | All three are required per metric.                                                   |
| `segment[*].name` / `.count` / `.archetype` | `segments: [{name: a, count: 50}]`                                  | All three are required per segment.                                                  |

### 1.1 Extra fields are forbidden

Every nested model uses `extra="forbid"`. Typos are caught immediately.

> `Extra inputs are not permitted`

| Triggering input                                                    | Fix                                                       |
|---------------------------------------------------------------------|-----------------------------------------------------------|
| `metrics: [{name: x, typ: score, polarity: positive}]` (typo `typ`) | Rename to `type`. Check the field table in [`builder-reference.md`](./builder-reference.md). |

---

## 2. Naming

Identifier-like fields require alphanumeric + underscores. Display labels
do not.

> `metric name 'X' must be alphanumeric or underscores only`
> `segment name 'X' must be alphanumeric or underscores only`

| Triggering input                                                | Fix                                              |
|-----------------------------------------------------------------|--------------------------------------------------|
| `metrics: [{name: "monthly revenue", type: amount, ...}]`       | Use `monthly_revenue`. Spaces and dashes are forbidden. |
| `segments: [{name: "North America", count: 100, archetype: growth}]` | Use `north_america`.                        |

### 2.1 Duplicate names

> `duplicate metric name(s): ['x']`
> `duplicate segment name(s): ['s']`

| Triggering input                                                                            | Fix                                                |
|---------------------------------------------------------------------------------------------|----------------------------------------------------|
| Two metrics with `name: engagement`                                                         | Rename one or remove the duplicate.                |
| Two segments with `name: enterprise`                                                        | Same.                                              |

---

## 3. References

The builder does cross-reference closure at root validation time. Every
named target must exist among the declared things it points at.

| Error message                                                                                   | Triggering input                                                                                       | Fix                                                                |
|-------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------|
| ``metric 'x': `follows: 'y'` references an unknown metric. Available metrics: [...]``           | Metric `x` has `follows: y`, but `y` is not in `metrics`.                                              | Either declare `y` or remove `follows`/`delay` from `x`.            |
| ``causal-lag cycle detected starting from metric 'a': a → b``                                   | `a` follows `b`, `b` follows `a`.                                                                      | Break the cycle — remove one `follows` link.                       |
| ``connection 'a' driven_by 'b': endpoint 'b' is not a declared metric. Available: [...]``       | Connection references a metric not declared in `metrics`.                                              | Spell-check or add the metric.                                     |
| ``segment 's': baseline references unknown metric 'x'. Available: [...]``                       | A segment's `baseline` dict has a key that's not a declared metric.                                    | Spell-check or remove the entry.                                   |
| ``lifecycle.track='x' is not a declared metric. Available: [...]``                              | `lifecycle.track: x` but `x` is not in `metrics`.                                                      | Use a declared metric name.                                         |

---

## 4. Conditional fields

Some fields are required only in certain contexts. Violations name the
specific triggering combination.

### 4.1 Metric `range`

> ``metric 'x' of type 'amount' requires a `range: [min, max]` (only `score` and `count` may omit it)``
> ``metric 'x' of type 'count' must not declare a range — counts are unbounded integers``
> ``metric 'x' range [a, b] is invalid: min must be strictly less than max``

| Triggering input                                                                  | Fix                                                                            |
|-----------------------------------------------------------------------------------|--------------------------------------------------------------------------------|
| `{name: mrr, type: amount, polarity: positive}` (no range)                        | Add `range: [min, max]`.                                                       |
| `{name: tickets, type: count, polarity: negative, range: [0, 100]}`               | Drop `range`. Counts are unbounded; use a `score` if you want bounded.         |
| `{name: nps, type: index, polarity: positive, range: [100, 100]}`                 | `min` must be strictly less than `max`.                                        |

### 4.2 Metric `follows` + `delay`

> ``metric 'x': `follows` and `delay` must be declared together (got follows=..., delay=...). To remove the lag, omit both.``
> ``metric 'x': `delay` must be >= 1, got 0``
> ``metric 'x' cannot follow itself``

| Triggering input                                                | Fix                                              |
|-----------------------------------------------------------------|--------------------------------------------------|
| `{name: t, type: count, polarity: negative, follows: e}` (no delay) | Add `delay: <int >= 1>`.                     |
| `{name: t, type: count, polarity: negative, delay: 2}` (no follows) | Add `follows: <metric_name>`.                |
| `delay: 0`                                                      | Use `delay: >= 1`.                               |
| `{name: x, ..., follows: x, delay: 1}`                          | A metric cannot lag behind itself.               |

### 4.3 Segment `count`

Pydantic emits `Input should be greater than or equal to 3` and
`Input should be less than or equal to 5000` from the field constraint.
Cohorts smaller than 3 collapse the engine's distribution sampling;
cohorts larger than 5000 are out of scope for the builder.

| Triggering input                          | Fix                                |
|-------------------------------------------|------------------------------------|
| `{name: s, count: 1, archetype: growth}`  | Raise to `count >= 3`.             |
| `{name: s, count: 6000, archetype: growth}` | Lower to `count <= 5000`.        |

### 4.4 Lifecycle stage shape and ordering

Stages may be declared in three shapes (single-key dict, 2-tuple, or
canonical `{name, threshold}`) — see [`builder-reference.md`](./builder-reference.md) §2.5.
The constraints below apply post-normalisation.

> `lifecycle stage names must be unique, got [...]`
> `lifecycle stage thresholds must be strictly ascending, got [...]`

| Triggering input                                              | Fix                                                  |
|---------------------------------------------------------------|------------------------------------------------------|
| `[{onboarding: 0.0}, {active: 0.2}, {active: 0.5}]`           | Rename one of the duplicate stage names.             |
| `[{onboarding: 0.0}, {at_risk: 0.5}, {active: 0.2}]`          | Order by threshold ascending.                         |
| `stages: [{onboarding: 0.0}]`                                 | At least 2 stages required (`min_length=2`).         |

### 4.5 Dim conflict

> ``dimension 'd': `reference: true` and `per` are mutually exclusive``

| Triggering input                                                    | Fix                                                        |
|---------------------------------------------------------------------|------------------------------------------------------------|
| `dimensions: [{name: dim_x, per: unit, reference: true, columns: [...]}]` | Pick one — reference dims are static lookups; `per` is for grain-bearing dims. |

### 4.6 Event trigger fields

> `event 'e': trigger 'proportional' requires a 'driver' metric`
> `event 'e': trigger 'proportional' requires a numeric 'scale'`
> `event 'e': trigger 'threshold' requires a 'metric' to watch`
> `event 'e': trigger 'threshold' requires 'above' or 'below'`
> `event 'e': pick one of 'above' / 'below', not both`

| Triggering input                                                                 | Fix                                                                          |
|----------------------------------------------------------------------------------|------------------------------------------------------------------------------|
| `{name: e, trigger: proportional, columns: [...]}`                               | Add `driver: <metric_name>` and `scale: <float >= 0>`.                        |
| `{name: e, trigger: threshold, columns: [...]}`                                  | Add `metric: <metric_name>` and exactly one of `above` / `below`.            |
| `{name: e, trigger: threshold, metric: x, above: 0.7, below: 0.3, columns: [...]}` | Pick one direction — both is contradictory.                                |

---

## 5. Vocabulary

Vocabulary checks are case-sensitive. The full word lists are in
[`builder-reference.md`](./builder-reference.md) §3.

> `unknown relationship word 'X'. Valid: [...]`
> `baseline values {...} are not in the baseline vocabulary [...]`
> `archetype spec '...' uses unknown shape word(s): [...]. Valid: [...]`

| Triggering input                                                              | Fix                                                            |
|-------------------------------------------------------------------------------|----------------------------------------------------------------|
| `connections: [["a", "linked", "b"]]`                                         | `linked` isn't a vocabulary word — see §3.3 of the reference.  |
| `segments: [{..., baseline: {mrr: huge}}]`                                    | Only `high` / `mid` / `low` accepted.                          |
| `segments: [{..., archetype: "exponential > flat @ 6"}]`                      | `exponential` isn't a shape word; use `growth` or `accelerating`. |

### 5.1 Connection-shape coercion

> `connection string 'X' must have exactly three whitespace-separated tokens: '<metric_a> <relationship> <metric_b>'`
> `connection tuple X must have three elements: (metric_a, relationship, metric_b)`

| Triggering input                                  | Fix                                                       |
|---------------------------------------------------|-----------------------------------------------------------|
| `connections: ["engagement driven_by"]`           | Three tokens — `engagement driven_by mrr`.                |
| `connections: [["a", "b"]]`                       | Three elements — `("a", "driven_by", "b")`.               |

### 5.2 Connection endpoints distinct

> `connection endpoints must be distinct, got 'x' driven_by 'x'`

| Triggering input                              | Fix                              |
|-----------------------------------------------|----------------------------------|
| `engagement driven_by engagement`             | Pick two different metrics.      |

### 5.3 Window tuple shape

> `window tuple must have 2 or 3 elements (start, end, [every]), got N`

| Triggering input                              | Fix                                                          |
|-----------------------------------------------|--------------------------------------------------------------|
| `window: ("2024-01",)`                        | At least `(start, end)`. Optionally `(start, end, every)`.   |

---

## 6. Composite archetype DSL

Errors here come from `plotsim.builder.parser.ArchetypeParseError`,
wrapped into the `UserInput` validation error as
``segment 's' archetype 'spec': <parser message>``.

> `archetype spec is empty`
> `archetype spec must be a string, got <T>`
> `Layered patterns ship in a future release. Use > for sequential composition.`
> `n_periods must be >= 2 to compose phases, got N`
> `archetype spec 'X': has an empty shape (check for trailing '>' or doubled separators)`
> `archetype spec 'X': expected N '@ N' transition(s) for M shape(s) (one '@' between every pair of '>'), got K`
> `archetype spec 'X' uses unknown shape word(s): [...]. Valid: [...]`
> `archetype spec 'X' has an empty '@' value`
> `archetype spec 'X': '@ Y' is not an integer period`
> `archetype spec 'X': transition period P out of range [1, N-1] for an N-period window`
> `archetype spec 'X': transition periods must be strictly ascending, got [...]`

| Triggering input (with 24-period window)         | Fix                                                                   |
|--------------------------------------------------|-----------------------------------------------------------------------|
| `archetype: ""`                                  | Provide at least one shape word.                                       |
| `archetype: "growth + decline"`                  | `+` is reserved for layered patterns. Use `growth > decline @ 12` instead. |
| `archetype: "growth >"`                          | Empty trailing shape — `growth > decline @ 12`.                       |
| `archetype: "growth > decline > flat @ 8"`       | Need 2 transitions for 3 shapes — `growth > decline > flat @ 8 @ 16`. |
| `archetype: "exponential > flat @ 6"`            | `exponential` isn't a vocabulary shape — see §5.                       |
| `archetype: "growth > decline @ "`               | Empty `@` value — provide an integer period.                          |
| `archetype: "growth > decline @ twelve"`         | Use an integer literal — `@ 12`.                                       |
| `archetype: "growth > decline @ 25"`             | Out of range for a 24-period window — `[1, 23]`.                       |
| `archetype: "growth > decline > flat @ 16 @ 8"`  | Periods must ascend — `@ 8 @ 16`.                                     |

---

## 7. Schema-only column-type errors

These come from the interpreter (Phase 3), not the model. They surface
when the schema mis-uses a column type — the structural validators
caught simpler issues before this point.

> `column 'X' in 'T': dtype-only type 'date'/'int'/'string'/'float' is supported on dim_date columns only. Other tables must use a source-bearing type (metric.X, faker.X, static.X, ref.X, etc.)`
> ``column 'X': type 'bucket' requires a non-empty `labels` list``
> ``column 'X': type 'scd' requires `tracks`, `tiers`, and `at` sub-fields``
> `column 'X': scd tracks metric 'M', but no fact table emits that metric`
> `column 'X' in 'T': unknown type 'Y'. Valid types: id, ref.X, metric.X, faker.X, static.X, segment.count, timestamp, date, int, string, float, bucket, scd`
> `event 'E' column 'X': type 'flag' is only valid in threshold-triggered events`

| Triggering input                                                                                            | Fix                                                                              |
|-------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| `dim_company.{name: founded_year, type: int}` (non-`dim_date` table)                                        | Use `faker.year` or `metric.X` — `int` alone is a dim_date dtype declaration.     |
| `{name: bucket_col, type: bucket}` (no `labels`)                                                            | Add `labels: [low, mid, high]` etc.                                              |
| `{name: scd_col, type: scd}` (missing sub-fields)                                                           | Add `tracks: <metric>`, `tiers: [...]`, `at: [...]` (one fewer cut than tier).   |
| `scd ... tracks: nps` but no fact emits `nps`                                                               | The metric must appear in some `fct.columns` as `metric.nps`.                    |
| `{name: x, type: integer}` (typo)                                                                           | Use `int`.                                                                       |
| `evt_login` (proportional) with `flag` column                                                               | `flag` is only valid in `trigger: threshold` events. Use `bool`-derived columns elsewhere. |

---

## 8. Semantic warnings

These do **not** block construction. They flag a choice the user can
defend, not a bug. Promote any of these to errors in your code with
`warnings.simplefilter("error", UserWarning)` if you want to.

> `seasonal pattern declared but the N-period window may be too short to recover two clean cycles (rule of thumb: >= 24 periods)`

| Triggering input                                                                                       | Why it warns                                                       | When to ignore                                                       |
|--------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------|----------------------------------------------------------------------|
| `archetype: seasonal` on a 12-period monthly window                                                    | The `seasonal` shape encodes 2 oscillation cycles (`period=2`). On 12 monthly periods you only get one full cycle of meaningful data; downstream seasonality detection becomes noisy. | Demo / smoke-testing. The data is valid; the seasonality is just under-resolved. |

> `only one segment declared — variation across the dataset will reflect distribution noise, not archetype mix`

| Triggering input                                                                                       | Why it warns                                                       | When to ignore                                                       |
|--------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------|----------------------------------------------------------------------|
| One segment in `segments`                                                                              | Plotsim's value comes from heterogeneous archetype mixes. With one segment, all entities share the same trajectory shape; differences are limited to per-entity distribution noise. | Single-cohort demos or unit tests where you want a clean curve.     |

> `using mirrors/inverts (|0.75|) with N metrics can over-constrain the correlation matrix and force the engine's PSD projection to make large adjustments`

| Triggering input                                                                                       | Why it warns                                                       | When to ignore                                                       |
|--------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------|----------------------------------------------------------------------|
| `>= 8` metrics with at least one `mirrors` or `inverts` connection                                     | A 0.75 magnitude across many pairs can violate positive-semidefiniteness; the engine projects via Higham, which can shift coefficients by O(0.1). | When the deltas the manifest reports under `_correlation_adjustments` are within your tolerance. |

---

## 8a. Engine cross-feature mutexes

Three pairings raise at engine config-load time, not at builder construction.
The builder happily produces the offending `PlotsimConfig`; the engine's
`PlotsimConfig` validator then refuses it. Clear messages, but they surface
later than the field-level errors above.

| Triggering input                                                                                       | Engine-level message                                                                                                                                                | Fix                                                                                                                                |
|--------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------|
| `entity_features` declared (or `entity_features: true`) **and** `quality:` non-empty                  | `entity_features.enabled requires quality.quality_issues to be empty (the entity_features aggregation runs on the pre-corruption fact tables)`                       | Pick one. If you need both, run two configs back-to-back: clean run for entity_features, dirty run for quality.                    |
| `holdout` declared **and** `quality:` non-empty                                                       | `holdout.enabled requires quality.quality_issues to be empty (the train / holdout split operates on the clean fact tables)`                                          | Pick one. The split semantics on corrupted tables are deferred to a future mission — both files would silently mix clean / dirty. |
| `entity_features` declared **and** `manifest.include: false`                                          | `entity_features.enabled requires manifest.include=true (entity feature labels read from the manifest payload)`                                                       | Builder configs always have `manifest.include=true`, so this only affects engine-direct YAML.                                       |

These rules trace to `plotsim.validation.validate_entity_features_config`
and the `HoldoutConfig` docstring on `quality_issues == []`.

## 8b. Distribution × event interaction

| Triggering input                                                                                                                       | Symptom (pre-M124)                                                       | Status / fix                                                                                                                            |
|----------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------|
| `events: [{trigger: proportional, driver: <count metric>}]` with the driver typed as `count` (poisson distribution → integer column)    | `TypeError: ufunc 'isnan' not supported for input types ...`             | **Fixed in M124.** The event-row builder now coerces the driver column to float64 before NaN-masking, so any numeric metric type works as a proportional driver. |

---

## 9. Pydantic-emitted errors not listed above

A few errors come from Pydantic's own validation (type, range, length)
without our custom message:

| Source                      | Example message                                | Triggering input                                       |
|-----------------------------|------------------------------------------------|--------------------------------------------------------|
| `WindowInput.every` Literal | `Input should be 'daily', 'weekly' or 'monthly'` | `window: {start: ..., end: ..., every: yearly}`      |
| `MetricInput.type` Literal  | `Input should be 'score', 'amount', 'count' or 'index'` | `type: percentage`                              |
| `MetricInput.polarity` Literal | `Input should be 'positive' or 'negative'`  | `polarity: neutral`                                   |
| `EventInput.scale` ge=0     | `Input should be greater than or equal to 0`   | `scale: -1`                                           |
| `EventInput.for_periods` ge=1 | `Input should be greater than or equal to 1` | `for_periods: 0`                                      |
| `LifecycleStageInput.threshold` range | `Input should be greater than or equal to 0` / `... less than or equal to 1` | `threshold: 1.5`              |

When in doubt: read the `loc` field of the `ValidationError` — it names
the exact path (e.g. `("metrics", 2, "type")`) that Pydantic flagged.

---

## 10. See also

- [`builder-reference.md`](./builder-reference.md) — vocabulary and field reference.
- [`builder-quickstart.md`](./builder-quickstart.md) — annotated walkthroughs.
- `plotsim/builder/input.py` — the canonical source for every error message above.
- `plotsim/builder/parser.py` — DSL parser with the `+` rejection and shape grammar.
