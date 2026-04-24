# Contributing to plotsim

## Dev setup

```
git clone https://github.com/mohossam/plotsim
cd plotsim
pip install -e .[dev]
pytest
```

`pytest` with no arguments runs the full unit + integration suite.

## Running tests

```
pytest                            # full suite (unit + integration)
pytest tests/test_curves.py       # a single module
pytest -x                         # stop on first failure
pytest --cov=plotsim             # with coverage
pytest -m "not integration"       # skip slow end-to-end tests
```

The integration tests in `tests/test_integration.py` are tagged with
`@pytest.mark.integration` so fast dev loops can skip them; they cover
all five shipped templates end to end.

## Adding a new domain template

1. Copy an existing config from `plotsim/configs/` to a working file.
2. Edit the `domain`, `metrics`, `archetypes`, `entities`, and `tables`
   sections for the new use case.
3. Validate it: `plotsim validate my_config.yaml`.
4. Run it with validation on: `plotsim run my_config.yaml --validate`.
5. Drop the new config into `plotsim/configs/sample_<name>.yaml` and
   add a description in `plotsim/cli.py::cmd_list_templates` so it
   shows up in `plotsim list-templates`.
6. Open a PR.

## Adding a new curve type

1. Implement the curve function in `plotsim/curves.py` — it must take
   a time array plus keyword params and return values in `[0, 1]`.
2. Register it in `CURVE_REGISTRY`.
3. Add it to the `CURVE_TYPES` / `CurveType` literal in
   `plotsim/config.py` so the schema accepts it.
4. Add unit tests in `tests/test_curves.py`.
5. If the curve needs new parameter types, update the YAML schema
   documentation in the README.

## Code style

```
ruff check .
ruff format .
```

Line length is 100. Target Python version is 3.10.

## Commit conventions

Small, focused commits. Reference the mission or issue when one exists.
Tests should land in the same commit as the behavior they cover.
