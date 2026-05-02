"""M112 — Bundled-template overhaul tests.

Covers every Acceptance Criteria checkbox from
``project/missions/mission-112-template-overhaul.md`` that's expressible
as Python:

  * Validation: ``plotsim validate --config-only`` passes on each of
    the five domain templates.
  * Generation: ``plotsim run`` succeeds on each domain template; the
    output directory contains every declared table (dims, facts,
    events, bridges, plus ``config.yaml``, ``validation_report.txt``,
    ``manifest.json``).
  * Education upgrade: ``dim_student`` carries ``academic_standing``
    plus the four SCD bookkeeping columns (``dim_row_id``,
    ``valid_from``, ``valid_to``, ``is_current``); ``fct_engagement``
    has the M102 ``stage`` column from the new ``dropout_risk``
    sequence.
  * Hygiene: no domain template contains ``entity_features``,
    ``holdout``, or ``quality`` — neither active nor commented.
  * Recipe files: ``ds_recipes.yaml`` and ``de_recipes.yaml`` are
    valid YAML; each fence-marked snippet uncomments to valid YAML;
    each recipe lists a target value for all five domain templates.
  * Recipe integration: classification recipe merged into a temp
    copy of ``sample_education`` (target ``dropout_risk``) validates;
    pipeline-testing recipe merged into a temp copy of ``sample_saas``
    (default ``fct_revenue.mrr``) validates.
  * Determinism: each domain template produces byte-identical
    CSV/manifest output across two runs with the same seed.
"""
from __future__ import annotations

import io
import re
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pandas as pd
import pytest
import yaml

from plotsim import cli


ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = ROOT / "plotsim" / "configs"

DOMAIN_TEMPLATES = ("saas", "hr", "education", "retail", "marketing")
RECIPE_FILES = ("ds_recipes.yaml", "de_recipes.yaml")

# Per-template expected output tables. Dim_date is universal. The rest
# mirrors each YAML's ``tables:`` + ``bridges:`` declarations.
EXPECTED_TABLES: dict[str, set[str]] = {
    "saas": {
        "dim_date", "dim_company", "dim_user", "dim_plan",
        "fct_engagement", "fct_revenue", "fct_support_tickets",
        "evt_login", "evt_churn",
    },
    "hr": {
        "dim_date", "dim_department", "dim_employee",
        "fct_performance", "fct_training", "fct_attendance",
        "evt_attrition",
    },
    "education": {
        "dim_date", "dim_student", "dim_course",
        "fct_grades", "fct_engagement", "evt_dropout",
        "bridge_enrollment",
    },
    "retail": {
        "dim_date", "dim_customer", "dim_product_category",
        "dim_store_type", "dim_promotion",
        "fct_sessions", "fct_purchases", "evt_cart_abandonment",
        "bridge_customer_category", "bridge_customer_promotion",
    },
    "marketing": {
        "dim_date", "dim_customer", "dim_channel", "dim_campaign",
        "dim_product_category",
        "fct_traffic", "fct_campaigns", "fct_revenue", "evt_churn",
        "bridge_customer_channel", "bridge_customer_campaign",
    },
}


def _template_path(name: str) -> Path:
    return CONFIGS_DIR / f"sample_{name}.yaml"


def _run_cli(*argv: str) -> tuple[int, str, str]:
    out_buf, err_buf = io.StringIO(), io.StringIO()
    with redirect_stdout(out_buf), redirect_stderr(err_buf):
        try:
            code = cli.main(list(argv))
        except SystemExit as exc:
            code = int(exc.code) if exc.code is not None else 0
    return code, out_buf.getvalue(), err_buf.getvalue()


# --- Validation: --config-only passes on all five domain templates ----------


@pytest.mark.parametrize("name", DOMAIN_TEMPLATES)
def test_validate_config_only_passes(name):
    code, out, _err = _run_cli(
        "validate", "--config-only", str(_template_path(name))
    )
    assert code == 0, out
    assert "VALID:" in out


# --- Generation: plotsim run succeeds + output contains every declared table


@pytest.mark.parametrize("name", DOMAIN_TEMPLATES)
def test_run_succeeds_and_emits_expected_tables(name, tmp_path):
    out_dir = tmp_path / name
    code, _stdout, _stderr = _run_cli(
        "run",
        str(_template_path(name)),
        "-o", str(out_dir),
        "-q",
        "--allow-absolute-output",
    )
    assert code == 0

    expected = EXPECTED_TABLES[name]
    actual = {p.stem for p in out_dir.glob("*.csv")}
    assert expected <= actual, f"{name} missing: {expected - actual}"

    # Always-present sidecars.
    assert (out_dir / "config.yaml").exists()
    assert (out_dir / "validation_report.txt").exists()
    assert (out_dir / "manifest.json").exists()


# --- Education-specific: SCD academic_standing + stages ---------------------


def test_education_dim_student_has_scd_columns(tmp_path):
    out_dir = tmp_path / "education"
    code, _o, _e = _run_cli(
        "run", str(_template_path("education")),
        "-o", str(out_dir), "-q", "--allow-absolute-output",
    )
    assert code == 0

    dim_student = pd.read_csv(out_dir / "dim_student.csv")
    expected_cols = {
        "academic_standing",
        "dim_row_id",
        "valid_from",
        "valid_to",
        "is_current",
    }
    missing = expected_cols - set(dim_student.columns)
    assert not missing, f"dim_student missing SCD columns: {missing}"

    # Every entity has exactly one current row.
    current_rows = dim_student[dim_student["is_current"]]
    assert len(current_rows) == current_rows["student_id"].nunique()


def test_education_fct_engagement_has_stage_column(tmp_path):
    out_dir = tmp_path / "education"
    code, _o, _e = _run_cli(
        "run", str(_template_path("education")),
        "-o", str(out_dir), "-q", "--allow-absolute-output",
    )
    assert code == 0

    fct = pd.read_csv(out_dir / "fct_engagement.csv")
    assert "stage" in fct.columns
    declared_stages = {"thriving", "stable", "struggling", "critical"}
    assert set(fct["stage"].dropna().unique()) <= declared_stages


# --- No domain template includes entity_features / holdout / quality --------


@pytest.mark.parametrize("name", DOMAIN_TEMPLATES)
def test_no_ml_or_de_config_in_domain_templates(name):
    """Both as parsed YAML keys AND as commented blocks anywhere in the file."""
    text = _template_path(name).read_text(encoding="utf-8")
    parsed = yaml.safe_load(text)

    for forbidden in ("entity_features", "holdout", "quality"):
        assert forbidden not in parsed, (
            f"{name}.yaml contains active ``{forbidden}:`` key"
        )

    # Commented active blocks would re-enter the file via copy-paste of an
    # earlier mission's template; reject them too. Match a top-level key
    # appearing as a comment header (``# entity_features:``).
    for forbidden in ("entity_features", "holdout", "quality"):
        comment_pattern = re.compile(
            rf"^#\s*{re.escape(forbidden)}\s*:", re.MULTILINE,
        )
        match = comment_pattern.search(text)
        assert match is None, (
            f"{name}.yaml has commented ``{forbidden}:`` block at byte "
            f"{match.start() if match else 0!r}"
        )


# --- Recipe files: parse as YAML, snippets uncomment cleanly ----------------


@pytest.mark.parametrize("recipe_file", RECIPE_FILES)
def test_recipe_file_parses_as_yaml(recipe_file):
    text = (CONFIGS_DIR / recipe_file).read_text(encoding="utf-8")
    yaml.safe_load(text)


def _extract_recipe_snippet(path: Path, name: str) -> str:
    """Pull a fence-marked snippet and strip the leading ``# ``."""
    text = path.read_text(encoding="utf-8")
    pattern = rf">>> RECIPE START: {re.escape(name)}\n(.+?)\n# >>> RECIPE END"
    m = re.search(pattern, text, re.DOTALL)
    assert m, f"Missing recipe ``{name}`` in {path.name}"
    body = m.group(1)
    out_lines: list[str] = []
    for line in body.splitlines():
        if line.startswith("# "):
            out_lines.append(line[2:])
        elif line == "#":
            out_lines.append("")
        else:
            out_lines.append(line)
    return "\n".join(out_lines) + "\n"


RECIPE_SNIPPETS = (
    ("ds_recipes.yaml", "classification"),
    ("ds_recipes.yaml", "clustering"),
    ("ds_recipes.yaml", "forecasting"),
    ("de_recipes.yaml", "pipeline_testing"),
)


@pytest.mark.parametrize("recipe_file,recipe_name", RECIPE_SNIPPETS)
def test_recipe_snippet_uncomments_to_valid_yaml(recipe_file, recipe_name):
    snippet = _extract_recipe_snippet(CONFIGS_DIR / recipe_file, recipe_name)
    parsed = yaml.safe_load(snippet)
    assert isinstance(parsed, dict) and parsed, (
        f"{recipe_name} snippet did not produce a non-empty mapping"
    )


def test_ds_recipes_lists_target_for_all_five_domains():
    """Each recipe with ``<TARGET_METRIC>`` must list a value per domain."""
    text = (CONFIGS_DIR / "ds_recipes.yaml").read_text(encoding="utf-8")
    # Classification + Forecasting both substitute <TARGET_METRIC>.
    for domain in DOMAIN_TEMPLATES:
        # Each domain name appears at least twice (classification + forecasting
        # per-domain tables).
        assert text.count(f"{domain}") >= 2, (
            f"ds_recipes.yaml mentions ``{domain}`` fewer than twice"
        )


def test_de_recipes_lists_target_for_all_five_domains():
    text = (CONFIGS_DIR / "de_recipes.yaml").read_text(encoding="utf-8")
    for domain in DOMAIN_TEMPLATES:
        assert domain in text, (
            f"de_recipes.yaml does not mention ``{domain}``"
        )


# --- Recipe integration: merged config validates ----------------------------


def test_classification_recipe_validates_against_education(tmp_path):
    snippet = _extract_recipe_snippet(
        CONFIGS_DIR / "ds_recipes.yaml", "classification"
    )
    snippet = snippet.replace("<TARGET_METRIC>", "dropout_risk")
    base = _template_path("education").read_text(encoding="utf-8")
    merged = base + "\n" + snippet
    target = tmp_path / "education_classification.yaml"
    target.write_text(merged, encoding="utf-8")

    code, out, _err = _run_cli("validate", "--config-only", str(target))
    assert code == 0, out
    assert "VALID:" in out


def test_pipeline_testing_recipe_validates_against_saas(tmp_path):
    snippet = _extract_recipe_snippet(
        CONFIGS_DIR / "de_recipes.yaml", "pipeline_testing"
    )
    base = _template_path("saas").read_text(encoding="utf-8")
    merged = base + "\n" + snippet
    target = tmp_path / "saas_pipeline_testing.yaml"
    target.write_text(merged, encoding="utf-8")

    code, out, _err = _run_cli("validate", "--config-only", str(target))
    assert code == 0, out
    assert "VALID:" in out


# --- Determinism: same config + same seed → byte-identical output ----------


@pytest.mark.parametrize("name", DOMAIN_TEMPLATES)
def test_template_output_is_byte_identical_across_runs(name, tmp_path):
    out_a = tmp_path / "run_a"
    out_b = tmp_path / "run_b"
    for out_dir in (out_a, out_b):
        code, _o, _e = _run_cli(
            "run", str(_template_path(name)),
            "-o", str(out_dir), "-q", "--allow-absolute-output",
        )
        assert code == 0

    files_a = sorted(p.relative_to(out_a) for p in out_a.glob("*"))
    files_b = sorted(p.relative_to(out_b) for p in out_b.glob("*"))
    assert files_a == files_b, "different file set across runs"

    for rel in files_a:
        # ``validation_report.txt`` carries a wall-clock timestamp by
        # default in the CLI path — not a determinism violation.
        if rel.name == "validation_report.txt":
            continue
        bytes_a = (out_a / rel).read_bytes()
        bytes_b = (out_b / rel).read_bytes()
        assert bytes_a == bytes_b, f"{name}: {rel} differs across runs"
