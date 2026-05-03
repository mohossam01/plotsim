"""plotsim.cli — argparse-driven command-line interface.

Commands:
    plotsim run <config.yaml>            Generate CSVs from a config
    plotsim validate <config.yaml>       Validate a config without generating
    plotsim info <config.yaml>           Summarize what a config would generate
    plotsim list-templates               List bundled sample configs
    plotsim template <name> [--output]   Copy a sample config out for editing
    plotsim schema [--output]            Emit JSON Schema for PlotsimConfig

The CLI is a thin shell over the library. Every command here calls a public
function that's also available as `from plotsim import ...`, so nothing
important lives only in argv-parsing land.
"""
from __future__ import annotations

import argparse
import calendar
import datetime as _dt
import shutil
import sys
from importlib import resources
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from plotsim import __version__
from plotsim.config import PlotsimConfig, load_config
from plotsim.manifest import build_manifest
from plotsim.schema import SCHEMA_FILENAME, write_schema
from plotsim.tables import generate_tables_with_state
from plotsim.validation import validate_tables
from plotsim.output import write_tables


TEMPLATE_PREFIX = "sample_"
TEMPLATE_SUFFIX = ".yaml"
# M124: builder-template directory. Files here are ``UserInput`` shape
# (``about`` / ``unit`` / ``segments`` at the top level) — the front door
# the builder docs walk users through. ``list-templates`` surfaces these
# alongside the engine-direct ``sample_*`` configs.
BUILDER_DIR_NAME = "templates"


# --- Config dispatcher (M124) ------------------------------------------------


def _is_builder_yaml(path: Path) -> bool:
    """Peek the YAML to decide which loader to use.

    Builder YAML is identified by the presence of the top-level ``about``
    key together with ``unit`` and ``segments`` — three required fields on
    ``UserInput`` that engine-direct ``PlotsimConfig`` never carries.
    Engine-direct YAML uses ``domain`` + ``time_window`` + ``entities``.

    The peek is permissive: a malformed YAML or a non-dict top-level
    falls through to ``load_config`` so the engine's existing error
    surface stays the source of truth for engine YAML problems.
    """
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except (OSError, yaml.YAMLError):
        return False
    if not isinstance(data, dict):
        return False
    return (
        "about" in data
        and "unit" in data
        and "segments" in data
        and "domain" not in data
    )


def load_either_config(path: str | Path) -> PlotsimConfig:
    """Load a config file, dispatching to the builder when it looks like one.

    Builder-shape YAML is routed through ``create_from_yaml`` (which runs
    UserInput validation and the interpreter). Everything else falls
    through to ``load_config``. The dispatch keeps every CLI command
    (``run``, ``validate``, ``info``) on a single entry point so users
    can hand either flavour of YAML to any command.
    """
    p = Path(path)
    if _is_builder_yaml(p):
        from plotsim.builder import create_from_yaml
        return create_from_yaml(p)
    return load_config(p)


# --- Template discovery -------------------------------------------------------


def _configs_dir() -> resources.abc.Traversable:
    return resources.files("plotsim") / "configs"


def list_templates() -> list[tuple[str, Path]]:
    """Return ``[(name, path), ...]`` for every bundled sample config.

    Names are the file stem with ``sample_`` stripped (engine-direct) or
    ``_template`` stripped (builder). Sorted alphabetically by name. M124
    surfaces builder templates from ``plotsim/configs/templates/`` alongside the
    engine-direct ``sample_*`` configs — the CLI's ``run`` / ``validate`` /
    ``info`` commands accept either flavour.
    """
    out: list[tuple[str, Path]] = []
    root = _configs_dir()
    for entry in root.iterdir():
        name = entry.name
        if not name.startswith(TEMPLATE_PREFIX) or not name.endswith(TEMPLATE_SUFFIX):
            continue
        stem = name[len(TEMPLATE_PREFIX) : -len(TEMPLATE_SUFFIX)]
        out.append((stem, Path(str(entry))))
    out.sort(key=lambda pair: pair[0])
    return out


def list_builder_templates() -> list[tuple[str, Path]]:
    """Return ``[(name, path), ...]`` for every builder-template YAML.

    Builder templates live in ``plotsim/configs/templates/`` and are
    ``UserInput`` shape — the ``create_from_yaml`` dispatcher route. Names
    are the file stem with the ``_template`` suffix stripped (so
    ``saas_template.yaml`` -> ``saas``); ``bare_minimum.yaml`` is kept
    verbatim. The ``.py`` companions in the directory (annotated kwargs
    examples) are skipped.
    """
    out: list[tuple[str, Path]] = []
    root = _configs_dir() / BUILDER_DIR_NAME
    if not root.is_dir():
        return out
    for entry in root.iterdir():
        name = entry.name
        if not name.endswith(TEMPLATE_SUFFIX):
            continue
        stem = name[: -len(TEMPLATE_SUFFIX)]
        if stem.endswith("_template"):
            stem = stem[: -len("_template")]
        out.append((stem, Path(str(entry))))
    out.sort(key=lambda pair: pair[0])
    return out


def find_template(name: str) -> Optional[Path]:
    """Resolve a template name to a path. Engine-direct names take
    precedence; falls back to builder templates if no engine-direct match.
    """
    for stem, path in list_templates():
        if stem == name:
            return path
    for stem, path in list_builder_templates():
        if stem == name:
            return path
    return None


# --- Info helpers -------------------------------------------------------------


def _estimate_periods(config: PlotsimConfig) -> int:
    """Estimate the row count for dim_date / per_period facts.

    Monthly/weekly arithmetic uses month-1 anchors (the established contract).
    Daily granularity expands ``end`` to the last day of the end-month so
    ``end: "2023-12"`` means "through Dec 31", not "through Dec 1" — otherwise
    the count under-counts by ``(days_in_end_month - 1)``.
    """
    tw = config.time_window
    start = _dt.date.fromisoformat(tw.start + "-01") if len(tw.start) == 7 else _dt.date.fromisoformat(tw.start)
    end = _dt.date.fromisoformat(tw.end + "-01") if len(tw.end) == 7 else _dt.date.fromisoformat(tw.end)
    if tw.granularity == "monthly":
        return (end.year - start.year) * 12 + (end.month - start.month) + 1
    if tw.granularity == "weekly":
        return ((end - start).days // 7) + 1
    if tw.granularity == "daily":
        if len(tw.end) == 7:
            last_day = calendar.monthrange(end.year, end.month)[1]
            end = _dt.date(end.year, end.month, last_day)
        return (end - start).days + 1
    return 0


_GRANULARITY_LABEL = {"monthly": "month", "weekly": "week", "daily": "day"}


def _period_label(granularity: str, n: int) -> str:
    base = _GRANULARITY_LABEL.get(granularity, granularity)
    return f"{base}s" if n != 1 else base


def _estimate_rows(config: PlotsimConfig, n_periods: int) -> int:
    """Rough estimate — dims + per_entity_per_period facts. Events skipped."""
    n_entities = sum(ent.size for ent in config.entities)
    total = n_periods  # dim_date
    total += len(config.entities)  # dim_<entity> (per_entity = one per cohort)
    # Rough event & fact estimate
    for tbl in config.tables:
        if tbl.grain == "per_entity_per_period":
            total += n_entities * n_periods
        elif tbl.grain == "per_period":
            total += n_periods
        elif tbl.grain == "per_reference":
            total += 1  # conservative; real count = longest static list
    return total


# --- Subcommands --------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    try:
        # M124: route builder-shape YAML through the builder pipeline
        # (UserInput → interpret); engine-direct YAML still uses load_config.
        config = load_either_config(args.config)
    except Exception as exc:
        print(f"Error loading config: {exc}", file=sys.stderr)
        return 1

    seed = args.seed if args.seed is not None else config.seed
    rng = np.random.default_rng(seed)

    if not args.quiet:
        print(f"Generating dataset from {args.config} (seed={seed})...")

    # M105: switch to the state-returning entry point so the trajectories
    # used during generation are available for manifest emission. The
    # output dict the CLI hands to ``write_tables`` is identical to the
    # 0.5 ``generate_tables`` return value — no behavior change for runs
    # with ``manifest.include = false``.
    tables, gen_state = generate_tables_with_state(config, rng)
    report = validate_tables(config, tables)

    if args.validate or args.strict:
        _print_validation_report(report)

    if args.strict and not report.ok:
        print(
            f"--strict: validation has {len(report.errors)} error(s); "
            f"aborting before write.",
            file=sys.stderr,
        )
        return 1

    # M105: build the manifest only when the config opts in. Skipping the
    # build (rather than building it then refusing to write) keeps the
    # cost off configs that don't need it — the trajectory sampling alone
    # is O(n_entities × n_periods × sample_rate) and the firing
    # aggregation walks every event row.
    manifest = None
    if config.manifest.include:
        # M106: thread SCD state through so the manifest's scd_events list
        # is populated for configs that use SCD Type 2. Empty for all other
        # configs (build_manifest no-ops on an empty SCDState).
        manifest = build_manifest(
            config, gen_state.trajectories, tables,
            scd_state=gen_state.scd,
            bridge_state=gen_state.bridges,
        )

    output_dir = Path(args.output_dir) if args.output_dir else None
    # SEC-01: sandbox every CLI write under cwd by default so a crafted config
    # (or a crafted -o flag) can't scribble to /etc/, the user's home, or any
    # other absolute location. ``--allow-absolute-output`` is the documented
    # escape hatch for power users who deliberately want to write elsewhere.
    base_dir = None if args.allow_absolute_output else Path.cwd()
    try:
        # F5 (M102): the CLI is the wall-clock-stamp consumer of the
        # validation report; library callers default to generated_at=None
        # and get a deterministic config-fingerprint instead.
        target = write_tables(
            tables, config, report, output_dir=output_dir, base_dir=base_dir,
            generated_at=_dt.datetime.now(),
            manifest=manifest,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        print(
            "Hint: pass --allow-absolute-output to bypass the cwd sandbox.",
            file=sys.stderr,
        )
        return 1

    if not args.quiet:
        total_rows = sum(len(df) for df in tables.values())
        print(
            f"Wrote {len(tables)} table(s), {total_rows} total row(s) to {target}/"
        )
        # FIX-03 / SF-9: surface event tables that emitted zero rows because
        # no driver (row_count_source / threshold column) is configured. The
        # validator owns the warning detection; the CLI mirrors it as a
        # one-line note per offending table so users notice without having
        # to open validation_report.txt.
        from plotsim.validation import CHECK_EMPTY_EVENT_TABLE
        for issue in report.by_check(CHECK_EMPTY_EVENT_TABLE):
            print(f"  ! {issue.table}: 0 rows (no event driver configured)")
    return 0


def cmd_validate_config(args: argparse.Namespace) -> int:
    """Validate a config without running the generation engine.

    ``plotsim validate`` and ``plotsim validate --config-only`` are
    behaviorally identical: both load the YAML, run every Pydantic
    field-level / model-level / cross-reference validator, and exit
    without touching the generation engine. ``--config-only`` is the
    explicit name for that fast-path contract — it makes the intent
    visible to anyone reading a CI script and reserves the bare
    ``validate`` command for a future deeper-validation mode without
    breaking the fast path.

    No output files are written under either invocation.
    """
    try:
        load_either_config(args.config)
    except Exception as exc:
        print(f"INVALID: {args.config}", file=sys.stdout)
        print(f"  {exc}", file=sys.stdout)
        return 1
    print(f"VALID: {args.config}")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    try:
        config = load_either_config(args.config)
    except Exception as exc:
        print(f"Error loading config: {exc}", file=sys.stderr)
        return 1

    n_periods = _estimate_periods(config)
    n_entities = sum(ent.size for ent in config.entities)
    cohorts = [ent.name for ent in config.entities]

    n_dim = sum(1 for t in config.tables if t.type == "dim")
    n_fact = sum(1 for t in config.tables if t.type == "fact")
    n_event = sum(1 for t in config.tables if t.type == "event")
    archetypes_used = sorted({ent.archetype for ent in config.entities})

    lines = [
        f"Domain: {config.domain.name}",
        f"Entity type: {config.domain.entity_label}",
        f"Entities: {n_entities} across {len(cohorts)} cohort(s) "
        f"({', '.join(cohorts)})",
        f"Time window: {config.time_window.start} to {config.time_window.end} "
        f"({n_periods} {_period_label(config.time_window.granularity, n_periods)})",
        f"Metrics: {len(config.metrics)} "
        f"({', '.join(m.name for m in config.metrics)})",
        f"Archetypes: {len(config.archetypes)} defined, "
        f"{len(archetypes_used)} in use",
        f"Tables: {len(config.tables)} "
        f"({n_dim} dim, {n_fact} fact, {n_event} event)",
        f"Estimated rows: ~{_estimate_rows(config, n_periods):,}",
        f"Seed: {config.seed}",
    ]
    print("\n".join(lines))
    return 0


def cmd_list_templates(_args: argparse.Namespace) -> int:
    engine_templates = list_templates()
    builder_templates = list_builder_templates()
    if not engine_templates and not builder_templates:
        print("No templates bundled. (Reinstall plotsim to restore.)")
        return 0

    engine_descriptions = {
        "saas": "B2B SaaS - customer accounts, engagement, revenue, churn",
        "hr": "HR department - employees, performance, training, attrition",
        "ecommerce": "E-commerce - customers, orders, cart abandonment, returns",
        "education": "University - students, courses, grades, enrollment",
        "healthcare": "Clinic - patients, visits, treatments, outcomes",
        "retail": "Retail - stores, transactions, inventory, returns",
        "marketing": "Marketing - campaigns, channels, conversions, attribution",
    }
    builder_descriptions = {
        "bare_minimum": "Smallest valid builder config — start here",
        "saas": "Builder shape: B2B SaaS customer success",
        "hr": "Builder shape: HR engagement / attrition",
        "education": "Builder shape: University course enrollment",
        "retail": "Builder shape: Retail transactions / loyalty",
        "marketing": "Builder shape: Marketing campaigns / attribution",
    }

    all_names = [n for n, _ in engine_templates] + [n for n, _ in builder_templates]
    width = max(len(name) for name in all_names) if all_names else 0

    if builder_templates:
        print("Builder templates (recommended — front door for new users):")
        for name, _path in builder_templates:
            desc = builder_descriptions.get(name, "")
            print(f"  {name.ljust(width)}  {desc}".rstrip())
        print("")
    if engine_templates:
        print("Engine-direct templates (full PlotsimConfig YAML):")
        for name, _path in engine_templates:
            desc = engine_descriptions.get(name, "")
            print(f"  {name.ljust(width)}  {desc}".rstrip())
        print("")
    first_name = (builder_templates or engine_templates)[0][0]
    print(
        f"Usage: plotsim template {first_name} -o my_config.yaml && "
        f"plotsim run my_config.yaml"
    )
    return 0


def cmd_schema(args: argparse.Namespace) -> int:
    """Emit the JSON Schema for ``PlotsimConfig``.

    With no ``--output``, prints the schema as pretty JSON to stdout.
    With ``--output PATH``, writes the schema to that file (default
    target: ``plotsim-schema.json`` at the repo root, the path the
    bundled VSCode workspace points at).
    """
    if args.output is None:
        # Default destination: a top-level ``plotsim-schema.json``. The CLI
        # is expected to run from a checkout's repo root, so writing to
        # ``Path.cwd() / SCHEMA_FILENAME`` matches the in-repo committed
        # path. Operators piping to stdout pass ``-o -``.
        dst = Path.cwd() / SCHEMA_FILENAME
    elif args.output == "-":
        # Stdout sink — useful for ``plotsim schema -o - | jq``.
        from plotsim.schema import generate_schema
        import json as _json
        print(_json.dumps(generate_schema(), indent=2, ensure_ascii=False))
        return 0
    else:
        dst = Path(args.output)
    written = write_schema(dst)
    print(f"Wrote {written}")
    return 0


def cmd_template(args: argparse.Namespace) -> int:
    path = find_template(args.name)
    if path is None:
        available = ", ".join(name for name, _ in list_templates())
        print(
            f"Unknown template: {args.name!r}. Available: {available}",
            file=sys.stderr,
        )
        return 1

    if args.output:
        dst = Path(args.output)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dst)
        print(f"Wrote {dst}")
    else:
        print(path.read_text(encoding="utf-8"), end="")
    return 0


# --- Pretty-printing ----------------------------------------------------------


def _print_validation_report(report) -> None:
    status = "VALID" if report.ok else "INVALID"
    print(
        f"Validation: {status} - "
        f"{len(report.errors)} error(s), {len(report.warnings)} warning(s)"
    )
    for issue in report.issues:
        tag = "ERROR" if issue.severity == "error" else "WARN "
        tbl = issue.table or "-"
        print(f"  [{tag}] {issue.check} ({tbl}) - {issue.message}")


# --- Argparse wiring ----------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="plotsim",
        description=(
            "Generate realistic multi-table datasets from behavioral archetypes."
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"plotsim {__version__}"
    )
    subparsers = parser.add_subparsers(dest="command")

    run_p = subparsers.add_parser("run", help="Generate dataset from config")
    run_p.add_argument("config", help="Path to YAML config file")
    run_p.add_argument("--output-dir", "-o", default=None)
    run_p.add_argument("--seed", "-s", type=int, default=None)
    run_p.add_argument("--validate", "-v", action="store_true",
                       help="Print validation report after generation")
    run_p.add_argument("--strict", action="store_true",
                       help="Exit non-zero if validation has any errors")
    run_p.add_argument("--quiet", "-q", action="store_true")
    run_p.add_argument(
        "--allow-absolute-output", action="store_true",
        help=(
            "Bypass the cwd path sandbox (SEC-01). Permits absolute "
            "output_dir values and ..-segment traversal. Only use this "
            "when you deliberately need to write outside the working "
            "directory."
        ),
    )
    run_p.set_defaults(func=cmd_run)

    val_p = subparsers.add_parser("validate", help="Validate a config file")
    val_p.add_argument("config", help="Path to YAML config file")
    val_p.add_argument(
        "--config-only", action="store_true",
        help=(
            "Run only load-time validators (no generation). Currently the "
            "default — the flag pins the fast-path contract for CI scripts "
            "and reserves the bare command for a future deeper mode."
        ),
    )
    val_p.set_defaults(func=cmd_validate_config)

    info_p = subparsers.add_parser(
        "info", help="Preview what a config would generate"
    )
    info_p.add_argument("config", help="Path to YAML config file")
    info_p.set_defaults(func=cmd_info)

    list_p = subparsers.add_parser(
        "list-templates", help="List available sample configs"
    )
    list_p.set_defaults(func=cmd_list_templates)

    tmpl_p = subparsers.add_parser("template", help="Copy a sample config")
    tmpl_p.add_argument("name", help="Template name (see list-templates)")
    tmpl_p.add_argument("--output", "-o", default=None,
                        help="Destination path (default: stdout)")
    tmpl_p.set_defaults(func=cmd_template)

    schema_p = subparsers.add_parser(
        "schema",
        help="Emit JSON Schema for PlotsimConfig (used for editor autocomplete)",
    )
    schema_p.add_argument(
        "--output", "-o", default=None,
        help=(
            f"Destination path (default: ./{SCHEMA_FILENAME}). "
            f"Pass '-' to write to stdout."
        ),
    )
    schema_p.set_defaults(func=cmd_schema)

    return parser


def _ensure_utf8_stdio() -> None:
    """Force UTF-8 on stdout/stderr so YAML with non-latin1 chars (arrows,
    em-dashes) doesn't crash on Windows' default cp1252.
    """
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8")
            except (OSError, ValueError):
                pass


def main(argv: Optional[list[str]] = None) -> int:
    _ensure_utf8_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
