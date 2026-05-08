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
the diff already shows *what*. Reference a mission file or issue when
one exists. Tests should land in the same commit as the behavior they
cover.

When the commit was AI-assisted, include the trailer:

```
Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

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
validates the source-file versions, the dated CHANGELOG entry, and tag
uniqueness, then creates the annotated tag and the GitHub Release. The
tag push triggers `publish.yml`, which pauses at the `pypi` environment
for required-reviewer approval before the OIDC upload to PyPI.

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
- **Manual `git tag v…` then `git push --tags`.** Triggers `publish.yml`
  but skips `release.yml`'s validation and the GitHub Release creation.
  Use `gh workflow run release.yml -f version=X.Y.Z`.
- **Version bump without CHANGELOG entry.** `release.yml` refuses to
  dispatch if `## [X.Y.Z] — YYYY-MM-DD` is missing from `CHANGELOG.md`.
- **`--` (ASCII double-hyphen) in place of `—` (em-dash) in the CHANGELOG
  date separator.** The release-workflow regex requires the em-dash.
  Editor autocorrect usually produces it; if not, paste it.
