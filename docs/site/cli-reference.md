# CLI Reference

> Every `plotsim` subcommand and flag. The CLI is a thin shell over the
> Python API — every command here calls a public function also available
> as `from plotsim import ...`.

```bash
plotsim --version
plotsim --help
plotsim <subcommand> --help
```

---

## Subcommands at a glance

| Command | Purpose |
|---|---|
| [`run`](#run) | Generate dataset from a YAML config |
| [`validate`](#validate) | Validate a config without generating |
| [`info`](#info) | Preview what a config would generate (entity / period / table counts) |
| [`list-templates`](#list-templates) | List bundled sample configs |
| [`template`](#template) | Copy a sample config out to disk for editing |
| [`schema`](#schema) | Emit the JSON Schema for `PlotsimConfig` (for editor autocomplete) |

Both *builder-shape* (`about` / `unit` / `segments` keys) and
*engine-direct* (`domain` / `time_window` / `entities` keys) YAML are
accepted by `run`, `validate`, and `info`. The CLI auto-detects which
shape you handed it.

---

## `run`

Generate every dim / fact / event / bridge table from a config and write
to disk.

```bash
plotsim run config.yaml -o ./output --seed 42
plotsim run config.yaml --validate --strict
plotsim run config.yaml -q                # quiet — no stdout chatter
```

| Flag | Type | Default | Purpose |
|---|---|---|---|
| `config` *(positional)* | path | required | Path to a YAML config |
| `-o, --output-dir` | path | inferred from `output.directory` | Directory to write into. Created if missing |
| `-s, --seed` | `int` | `config.seed` | Override the config's seed for one run |
| `-v, --validate` | flag | off | Print the validation report after generation (always run; this just prints) |
| `--strict` | flag | off | Exit with code 1 if validation has any errors. Tables are NOT written |
| `-q, --quiet` | flag | off | Suppress the "Generating..." / "Wrote N rows" lines |
| `--allow-absolute-output` | flag | off | Bypass the cwd path sandbox (SEC-01). Required to write outside the working directory or use `..` segments in `--output-dir` |

**SEC-01 sandbox** — by default, every CLI write must land under the
process's current working directory. A crafted config or a crafted
`-o` flag can't scribble to `/etc/`, your home folder, or any other
absolute location. Pass `--allow-absolute-output` only when you
deliberately need to write elsewhere.

**Exit codes**: `0` on success, `1` on config-load failure or strict
validation failure.

---

## `validate`

Run every load-time validator without generating data. Useful in CI to
fail fast before any table-write effort.

```bash
plotsim validate config.yaml
plotsim validate config.yaml --config-only
```

| Flag | Type | Default | Purpose |
|---|---|---|---|
| `config` *(positional)* | path | required | Path to a YAML config |
| `--config-only` | flag | off (default behavior) | Pin the fast-path contract — no generation, only schema and cross-reference validation. Currently identical to bare `validate`; the flag reserves the bare command for a future deeper-validation mode |

**Exit codes**: `0` on `VALID:`, `1` on `INVALID:` with the error
message printed to stdout.

---

## `info`

Summarize what a config would produce — entity count, period count,
table breakdown — without running generation.

```bash
plotsim info config.yaml
```

| Flag | Type | Default | Purpose |
|---|---|---|---|
| `config` *(positional)* | path | required | Path to a YAML config |

Sample output:

```
Domain: subscription_saas
Entity type: Customer
Entities: 80 across 4 cohort(s) (growers, decliners, seasonal, dormant)
Time window: 2024-01 to 2024-12 (12 months)
Metrics: 4 (engagement, mrr, support_tickets, churn_risk)
Archetypes: 4 defined, 4 in use
Tables: 6 (2 dim, 3 fact, 1 event)
Estimated rows: ~2,000
Seed: 42
```

The cell-count gate (see [Limits](./config-reference.md#limits-and-performance-gates))
is the one to watch on first-time runs of an unfamiliar config.

---

## `list-templates`

List every bundled sample config. plotsim ships two flavors:

- **Builder templates** (`plotsim/configs/templates/`) — the recommended
  front door: `about` / `unit` / `segments` shape, designed for
  hand-editing.
- **Engine-direct templates** (`plotsim/configs/sample_*.yaml`) — full
  `PlotsimConfig` shape, useful when you want every escape hatch.

```bash
plotsim list-templates
```

Both flavors are accepted by `run`, `validate`, and `info`. The
output ends with a copy-pasteable usage hint.

---

## `template`

Copy a bundled template to a path of your choosing (or print to stdout).

```bash
plotsim template saas -o my_config.yaml
plotsim template hr                       # print to stdout
```

| Flag | Type | Default | Purpose |
|---|---|---|---|
| `name` *(positional)* | str | required | Template name (see `list-templates`) |
| `-o, --output` | path | stdout | Destination path. Parent directories are created automatically |

Engine-direct names take precedence on collision; otherwise builder
templates are checked.

---

## `schema`

Emit the JSON Schema for `PlotsimConfig` — used by editor extensions
(VSCode, JetBrains) for autocomplete and inline validation on YAML
configs.

```bash
plotsim schema                            # writes ./plotsim-schema.json
plotsim schema -o /path/to/schema.json    # custom destination
plotsim schema -o - | jq '.properties'    # stdout for piping
```

| Flag | Type | Default | Purpose |
|---|---|---|---|
| `-o, --output` | path | `./plotsim-schema.json` | Destination path. Pass `-` to write to stdout |

The repo's bundled VSCode workspace points at `plotsim-schema.json` at
the project root, so the no-args form is the matching default.

---

## See also

- [Config reference](./config-reference.md) — every YAML field accepted
  by `run` / `validate` / `info`
- [API reference](./api-reference.md) — the Python functions each CLI
  subcommand wraps
- [Cookbook → for data engineers](./cookbook/data-engineering.md) — using
  the CLI as part of an ETL / dbt fixture pipeline
