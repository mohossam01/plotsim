"""Test-vehicle configs.

YAML and Python builder configs that exercise a specific engine feature in
the test suite. These are NOT public builder templates — they live here so
``plotsim.list_templates()`` can keep its catalog scoped to domain templates
while the feature tests retain dedicated, hand-tuned fixtures.

A test loads a YAML vehicle via ``create_from_yaml(CONFIGS_DIR / "<name>.yaml")``
and a ``.py`` vehicle via ``from tests.configs.<name> import config``.
"""

from pathlib import Path

CONFIGS_DIR = Path(__file__).resolve().parent
