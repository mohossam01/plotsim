# Contributing to plotsim

## Dev setup

```
git clone https://github.com/mohossam01/plotsim
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
the bundled templates end to end (13 templates as of this writing —
run `plotsim list-templates` for the current list).

## Adding a new template

Plotsim ships templates in two surfaces — pick the one that matches
your config shape:

1. **Builder-shape templates** (`plotsim/configs/templates/<name>_template.{py,yaml}`)
   are the recommended front door. Author the `.py` file as a
   `create(**kwargs)` call and the `.yaml` file as the equivalent YAML;
   both must round-trip to the same `PlotsimConfig` under the same seed.
2. **Engine-direct templates** (`plotsim/configs/sample_<name>.yaml`)
   are the escape hatch when you need full `PlotsimConfig` shape with
   every knob exposed.

Steps:

1. Copy an existing template (e.g. `saas_template.yaml` +
   `saas_template.py`) as a starting point.
2. Edit metrics, segments / archetypes, dimensions, facts, events, and
   any feature-specific blocks for the new use case.
3. Validate it: `plotsim validate <path>.yaml`.
4. Run it: `plotsim run <path>.yaml --validate`.
5. Add a description for the new name in
   `plotsim/cli.py::cmd_list_templates` so it shows up in
   `plotsim list-templates`.
6. Add the new name to `EXPECTED_TEMPLATES` in
   `tests/test_templates_api.py` — the catalog test fails the moment a
   bundled template lands without a matching entry, so this step is
   structural rather than optional.
7. Open a PR.

## Adding a new curve type

1. Implement the curve function in `plotsim/curves.py` — it must take
   a time array plus keyword params and return values in `[0, 1]`.
2. Register it in `CURVE_REGISTRY`.
3. Add it to the `CURVE_TYPES` / `CurveType` literal in
   `plotsim/config.py` so the schema accepts it.
4. Add unit tests in `tests/test_curves.py`.
5. If the curve needs new parameter types, update the column-type and
   archetype docs under `docs/site/`.

## Code style

```
ruff check .
ruff format .
```

Line length is 100. Target Python version is 3.10.

## Branch strategy

`main` is protected. All work happens on branches off `main` and lands
through a pull request — direct pushes and force-pushes to `main` are
blocked by branch protection.

The repo is configured for **squash-merge only** (merge commits and
rebase merges are disabled in repo settings, linear history is
required, head branches auto-delete on merge). Status checks required
to merge: the `Tests` workflow on Python 3.10–3.13 (all four matrix
legs must be green). One approving review is required.

## Branch naming

Use a short prefix matching the commit type, a slash, then a concise
scope.

| Prefix          | Use for                                                  |
|-----------------|----------------------------------------------------------|
| `feat/…`        | New feature work                                         |
| `fix/…`         | Bug fix                                                  |
| `docs/…`        | Documentation only                                       |
| `chore/…`       | Maintenance, gitignore, dependency bumps                 |
| `ci/…`          | CI / workflow changes                                    |
| `refactor/…`    | Code restructure with no behavior change                 |
| `test/…`        | Test-only change                                         |
| `release/X.Y.Z` | Release prep PR (version bump + CHANGELOG date)          |

## Commit grammar

Conventional Commits. Subject is lowercase, no trailing period, ≤72
chars:

```
<type>(<optional scope>): <imperative summary>
```

| Type       | Use for                                                       |
|------------|---------------------------------------------------------------|
| `feat`     | New user-visible behavior                                     |
| `fix`      | Bug fix                                                       |
| `docs`     | Documentation only (READMEs, CHANGELOG, this file, etc.)      |
| `chore`    | Maintenance, gitignore, dependency bumps, no behavior change  |
| `ci`       | CI / workflow changes                                         |
| `refactor` | Code restructure, no behavior change                          |
| `perf`     | Performance improvement                                       |
| `test`     | Test-only change                                              |

Small, focused commits. Body wraps at ~72 chars and explains *why* —
the diff already shows *what*. Reference an issue when one exists.
Tests should land in the same commit as the behavior they cover.

## Pull request flow

```
git checkout main
git pull --ff-only origin main
git checkout -b <prefix>/<scope>
# … atomic commits …
git push -u origin <prefix>/<scope>
gh pr create
```

`.github/PULL_REQUEST_TEMPLATE.md` populates a sectioned template
(*What this PR does*, *How to test*, checklist, breaking changes).
Fill the relevant sections; flag breaking changes explicitly.

CI runs the 3.10–3.13 matrix on every push to the PR. All four legs
must pass to merge. **Squash-merge** from the GitHub UI. The head
branch auto-deletes on merge.

## Releasing plotsim

Two steps. The full procedure with examples is in
[`RELEASE.md`](RELEASE.md); the summary lives here so it sits adjacent
to the rest of the workflow.

**Step 1 — Prep PR.** Branch from `main` (e.g. `release/0.7.0`). Bump
`__version__` in `plotsim/__init__.py` and `version` in `pyproject.toml`
to the new `X.Y.Z`. In `CHANGELOG.md`, rename `## [Unreleased]` to
`## [X.Y.Z] — YYYY-MM-DD` (em-dash `—` U+2014, single space each side),
seed a fresh empty `## [Unreleased]` above. Open the PR; squash-merge
after CI green.

**Step 2 — Cut release.**

```
gh workflow run release.yml -f version=X.Y.Z
```

(or **Actions → Release → Run workflow**). The dispatched workflow
runs five jobs in sequence: validate the source-file versions and
dated CHANGELOG entry → test matrix on Python 3.10–3.13 → build wheel
+ sdist → create annotated tag and GitHub Release → publish to PyPI.
The publish job pauses at the `pypi` environment for required-reviewer
approval before the OIDC upload. Tests running before the tag means a
test failure aborts the release with no orphan tag.

### Release eligibility

| Bump            | Use for                                                                                    |
|-----------------|--------------------------------------------------------------------------------------------|
| Patch (`0.6.x`) | Bug fixes, code quality, docs, CI/tooling, tests. No new config fields, no output changes. |
| Minor (`0.x.0`) | New features (default-off), new config fields, new templates, manifest schema bumps.       |
| Major (`x.0.0`) | Breaking changes to public API, config schema, or output format.                           |

When unsure, scan `## [Unreleased]` in `CHANGELOG.md`. New config fields
under `### Added` → minor. `### Removed` or `### Migration` notes →
major. Otherwise patch.

## Anti-patterns

The first three are blocked by repo configuration; the rest are
discipline. All of them produce broken or mistrusted artifacts.

- **Direct push to `main`.** Blocked by branch protection. Open a PR.
- **Force-push to `main`.** Blocked by branch protection. If history
  needs rewriting (rare), temporarily disable protection, do the
  rewrite from a backup tag, then re-enable.
- **Merge commits on `main`.** Blocked by the linear-history
  requirement. Squash-merge from the GitHub UI.
- **`twine upload` from a laptop.** PyPI Trusted Publishing is wired
  for this repo + the `publish` job + the `pypi` environment. API
  tokens are not configured and will 403.
- **Manual `git tag v…` then `git push --tags`.** No workflow listens
  on `push: tags` — a manual tag push leaves an orphan tag and
  publishes nothing. Use `gh workflow run release.yml -f version=X.Y.Z`.
- **Version bump without CHANGELOG entry.** `release.yml` refuses to
  dispatch if `## [X.Y.Z] — YYYY-MM-DD` is missing from `CHANGELOG.md`.
- **`--` (ASCII double-hyphen) in place of `—` (em-dash) in the CHANGELOG
  date separator.** The release-workflow regex requires the em-dash.
  Editor autocorrect usually produces it; if not, paste it.
