"""plotsim.cli — argparse-driven command-line interface.

Commands:
    plotsim run <config.yaml>            Generate CSVs from a config
    plotsim validate <config.yaml>       Validate a config without generating
    plotsim info <config.yaml>           Summarize what a config would generate
    plotsim list-templates               List bundled sample configs
    plotsim template <name> [--output]   Copy a sample config out for editing

The CLI is a thin shell over the library. Every command here calls a public
function that's also available as `from plotsim import ...`, so nothing
important lives only in argv-parsing land.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import shutil
import sys
from importlib import resources
from pathlib import Path
from typing import Optional

import numpy as np

from plotsim import __version__
from plotsim.config import PlotsimConfig, load_config
from plotsim.tables import generate_tables
from plotsim.validation import validate_tables
from plotsim.output import write_tables


TEMPLATE_PREFIX = "sample_"
TEMPLATE_SUFFIX = ".yaml"


# --- Template discovery -------------------------------------------------------


def _configs_dir() -> resources.abc.Traversable:
    return resources.files("plotsim") / "configs"


def list_templates() -> list[tuple[str, Path]]:
    """Return ``[(name, path), ...]`` for every bundled sample config.

    Names are the file stem with ``sample_`` stripped. Sorted alphabetically.
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


def find_template(name: str) -> Optional[Path]:
    for stem, path in list_templates():
        if stem == name:
            return path
    return None


# --- Info helpers -------------------------------------------------------------


def _estimate_periods(config: PlotsimConfig) -> int:
    tw = config.time_window
    start = _dt.date.fromisoformat(tw.start + "-01") if len(tw.start) == 7 else _dt.date.fromisoformat(tw.start)
    end = _dt.date.fromisoformat(tw.end + "-01") if len(tw.end) == 7 else _dt.date.fromisoformat(tw.end)
    if tw.granularity == "monthly":
        return (end.year - start.year) * 12 + (end.month - start.month) + 1
    if tw.granularity == "weekly":
        return ((end - start).days // 7) + 1
    if tw.granularity == "daily":
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
        config = load_config(args.config)
    except Exception as exc:
        print(f"Error loading config: {exc}", file=sys.stderr)
        return 1

    seed = args.seed if args.seed is not None else config.seed
    rng = np.random.default_rng(seed)

    if not args.quiet:
        print(f"Generating dataset from {args.config} (seed={seed})...")

    tables = generate_tables(config, rng)
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

    output_dir = Path(args.output_dir) if args.output_dir else None
    target = write_tables(tables, config, report, output_dir=output_dir)

    if not args.quiet:
        total_rows = sum(len(df) for df in tables.values())
        print(
            f"Wrote {len(tables)} table(s), {total_rows} total row(s) to {target}/"
        )
    return 0


def cmd_validate_config(args: argparse.Namespace) -> int:
    try:
        load_config(args.config)
    except Exception as exc:
        print(f"INVALID: {args.config}", file=sys.stdout)
        print(f"  {exc}", file=sys.stdout)
        return 1
    print(f"VALID: {args.config}")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.config)
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
    templates = list_templates()
    if not templates:
        print("No templates bundled. (Reinstall plotsim to restore.)")
        return 0

    descriptions = {
        "saas": "B2B SaaS - customer accounts, engagement, revenue, churn",
        "hr": "HR department - employees, performance, training, attrition",
        "ecommerce": "E-commerce - customers, orders, cart abandonment, returns",
        "education": "University - students, courses, grades, enrollment",
        "healthcare": "Clinic - patients, visits, treatments, outcomes",
    }
    width = max(len(name) for name, _ in templates)
    print("Available templates:")
    for name, _path in templates:
        desc = descriptions.get(name, "")
        print(f"  {name.ljust(width)}  {desc}".rstrip())
    print("")
    first_name = templates[0][0]
    print(
        f"Usage: plotsim template {first_name} -o my_config.yaml && "
        f"plotsim run my_config.yaml"
    )
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
    run_p.set_defaults(func=cmd_run)

    val_p = subparsers.add_parser("validate", help="Validate a config file")
    val_p.add_argument("config", help="Path to YAML config file")
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
