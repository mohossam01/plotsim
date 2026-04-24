"""Plotsim quickstart — generate a SaaS dataset end-to-end.

Run from the repo root:

    python examples/quickstart.py

After installing the package, users typically start from a template
instead:

    plotsim template saas -o my_config.yaml
    plotsim run my_config.yaml -o ./output --validate
"""
from pathlib import Path

from numpy.random import default_rng

from plotsim import (
    generate_tables,
    load_config,
    validate,
    write_tables,
)


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
SAAS_CONFIG = REPO_ROOT / "plotsim" / "configs" / "sample_saas.yaml"
OUTPUT_DIR = HERE / "quickstart_output"


def main() -> None:
    config = load_config(SAAS_CONFIG)
    tables = generate_tables(config, default_rng(config.seed))
    report = validate(config, tables)
    print(
        f"Validation: {len(report.errors)} error(s), "
        f"{len(report.warnings)} warning(s), "
        f"{'ok' if report.ok else 'FAILED'}"
    )

    target = write_tables(tables, config, report, output_dir=OUTPUT_DIR)
    print(f"Dataset written to {target}")

    eng = tables["fct_engagement"]
    print(f"\nfct_engagement: {len(eng)} rows")
    print(eng.head().to_string(index=False))


if __name__ == "__main__":
    main()
