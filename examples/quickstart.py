"""Plotsim quickstart — generate a subscription-customers dataset.

Run from the repo root:

    python examples/quickstart.py

The example mirrors the README and ``plotsim`` package docstring: build a
config with ``create()``, generate tables, validate, and write CSVs. After
installing the package, users typically reach for a bundled template via
the CLI instead:

    plotsim template saas -o my_config.yaml
    plotsim run my_config.yaml -o ./output --validate
"""

from pathlib import Path

from plotsim import create, generate_tables, validate, write_tables


HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "quickstart_output"


def main() -> None:
    cfg = create(
        about="Subscription customers",
        unit="customer",
        window=("2024-01", "2024-12", "monthly"),
        metrics=[
            {"name": "engagement", "type": "score", "polarity": "positive"},
            {"name": "payments", "type": "count", "polarity": "positive"},
        ],
        segments=[
            {"name": "active", "count": 50, "archetype": "growth"},
            {"name": "inactive", "count": 30, "archetype": "decline"},
        ],
    )
    tables = generate_tables(cfg)
    report = validate(cfg, tables)
    print(
        f"Validation: {len(report.errors)} error(s), "
        f"{len(report.warnings)} warning(s), "
        f"{'ok' if report.ok else 'FAILED'}"
    )

    target = write_tables(tables, cfg, report, output_dir=OUTPUT_DIR)
    print(f"Dataset written to {target}")

    for name, df in tables.items():
        print(f"{name}: {len(df)} rows")


if __name__ == "__main__":
    main()
