"""Regenerate Layer 4/5 reference fixtures.

Run this BEFORE landing the Category B Layer 4 (fact vectorization) and Layer 5
(event vectorization) changes — the CSVs it writes are the byte-identical
ground truth that `tests/test_tables.py::test_layer4_reference_fixtures_match`
compares against after vectorization.

Usage (from repo root):

    python tests/fixtures/_generate_layer4_fixtures.py

Produces one subdirectory per bundled template under
`tests/fixtures/layer4_reference/<stem>/`, each containing every CSV that
`generate_tables` emits plus the validation report.

Intentionally not a pytest test. Running it rewrites the fixtures in place,
which is only correct when done deliberately from a known-good plotsim state.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from plotsim.config import load_config
from plotsim.output import write_tables
from plotsim.tables import generate_tables

ROOT = Path(__file__).resolve().parent.parent.parent
CONFIGS = ROOT / "plotsim" / "configs"
FIXTURE_DIR = Path(__file__).resolve().parent / "layer4_reference"


def main() -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    for stem in ("saas", "hr", "ecommerce", "education", "healthcare"):
        cfg = load_config(CONFIGS / f"sample_{stem}.yaml")
        tables = generate_tables(cfg, np.random.default_rng(cfg.seed))
        target = FIXTURE_DIR / stem
        # Drop any previous contents so stale files can't linger.
        if target.exists():
            for child in target.iterdir():
                if child.is_file():
                    child.unlink()
        write_tables(tables, cfg, output_dir=target)
        print(f"wrote {stem} -> {target}")


if __name__ == "__main__":
    main()
