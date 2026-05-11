# Log-file writer

Most teaching datasets give you tabular event data (`evt_login.csv`,
`evt_purchase.csv`) and call it a day. Real production systems emit
*log lines* — a string per event, formatted to be parsed by Splunk,
Datadog, Logstash, or a custom regex pipeline. The log-file writer
generates those log lines alongside the tabular events, from the same
underlying data, so you can practice log parsing against a dataset
where the parsed-out columns are sitting right next to the log lines
as ground truth.

## How it works

Set `log_format` on any event-typed table in your config. plotsim then
writes one `.log` file per such event table, in addition to the regular
CSV/Parquet event file. Each row in the event table becomes one line
in the log file, with placeholders in the format string resolved
against the row's column values.

```yaml
tables:
  - name: "evt_login"
    type: "event"
    grain: "variable"
    row_count_source: "proportional:engagement:scale:5"
    columns:
      - {name: "event_id", dtype: "id", source: "pk"}
      - {name: "date_key", dtype: "id", source: "fk:dim_date.date_key"}
      - {name: "user_id", dtype: "id", source: "fk:dim_user.user_id"}
      - {name: "company_id", dtype: "id", source: "fk:dim_company.company_id"}
      - {name: "event_ts", dtype: "date", source: "generated:timestamp"}
    primary_key: "event_id"
    foreign_keys: ["dim_date.date_key", "dim_user.user_id", "dim_company.company_id"]
    log_format: "{event_ts} [INFO] user={user_id} company={company_id} action=login event_id={event_id}"
```

After `plotsim run`, the output directory contains both:

```text
out/
  evt_login.csv                ← tabular event data (unchanged)
  evt_login.log                ← NEW: one line per row, formatted
```

A sample line in `evt_login.log` looks like:

```text
2024-03-15 09:42:11 [INFO] user=u-0042 company=c-007 action=login event_id=e-9931
```

## Format string

The `log_format` field is a Python `str.format` template. Placeholders
in `{curly braces}` must match column names on the event table. You
can build whatever shape you like:

| Style | Format string |
|---|---|
| Apache combined | `'{company_id} - {user_id} [{event_ts}] "GET /login HTTP/1.1" 200 0'` |
| Syslog-ish | `"{event_ts} plotsim[12345]: user={user_id} action=login"` |
| Structured JSON | `'{{"ts": "{event_ts}", "user": "{user_id}", "co": "{company_id}", "act": "login"}}'` |
| Pipe-delimited | `"{event_ts}\|{user_id}\|{company_id}\|login"` |

Note the JSON example uses `{{` and `}}` to emit literal braces — that's
Python's standard `str.format` escape.

If a placeholder names a column that doesn't exist on the table,
plotsim fails fast at write time with a `ValueError` listing the
available columns. No silent garbage in your log file.

## Filename

By default the log file is named `<table_name>.log` (so `evt_login` →
`evt_login.log`). Override with `log_filename` if you want a different
name:

```yaml
- name: "evt_login"
  type: "event"
  log_format: "{event_ts} login {user_id}"
  log_filename: "auth.log"      # writes auth.log instead of evt_login.log
```

The filename is sandboxed to the output directory — `..` traversal and
absolute paths are rejected.

## Builder syntax

```python
from plotsim import create

config = create(
    about="B2B SaaS",
    unit="company",
    window=("2023-01", "2024-12", "monthly"),
    metrics=[{"name": "engagement", "type": "score", "polarity": "positive"}],
    segments=[{"name": "alpha", "count": 50, "archetype": "growth"}],
    events=[
        {
            "name": "evt_login",
            "trigger": "proportional",
            "driver": "engagement",
            "scale": 5.0,
            "log_format": "{event_ts} [INFO] user={user_id} co={company_id} login",
            "columns": [
                {"name": "event_id", "type": "id"},
                {"name": "date_key", "type": "id", "fk": "dim_date"},
                {"name": "user_id", "type": "id", "fk": "dim_user"},
                {"name": "company_id", "type": "id", "fk": "dim_company"},
                {"name": "event_ts", "type": "date"},
            ],
        },
    ],
)
```

## Use cases

### Log parsing exercise

Hand a student `evt_login.log` and ask them to write a parser
(regex, grok pattern, structured-log decoder, whatever the lesson
calls for) that extracts `event_ts`, `user_id`, `company_id`, and
`event_id` into a structured table. The ground-truth tabular form
(`evt_login.csv`) is sitting right next to the log file — they can
diff their parser output against the truth without you having to
hand-build an answer key.

### Multi-format log ingestion

Configure each event table with a different `log_format` to simulate
the real warehouse situation where login events come from one system
in JSON, page-view events come from another system in Apache combined
format, and audit events come from a third in syslog. Practice
unifying these into a single observability schema.

### Sample data for log-pipeline tests

Because the log file is deterministic (same seed → byte-identical
log), it's a stable fixture for testing log-pipeline code. Pin the
config + seed; check the parser output against the same expected
values across runs.

## Scope and limitations

* **Event tables only.** Fact tables are state snapshots, not
  discrete events; dim tables are reference data; bridges are M:M
  associations. None of those map cleanly to a log-line shape.
* **No log rotation, no log levels, no structured-log schema
  validation.** The log writer is a thin formatter — operational
  concerns (rotation, retention, level filtering) belong to the
  consumer pipeline, not the synthetic-data generator.
* **One format per event table.** If you need multiple log shapes
  for the same event stream (e.g. a "production" and a "debug" log),
  declare two event tables in the config that share the same trigger
  and FKs but emit different formats.
* **No nested event templating.** Each row is one log line; multi-line
  log entries (Java stack traces, multi-line JSON) aren't generated.
