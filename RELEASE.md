# Releasing plotsim

Plotsim ships through a two-step release process: a **prep PR** that
bumps version and dates the changelog, then a **`workflow_dispatch`**
of `release.yml` that runs the full release end-to-end (validate →
test → build → tag → GitHub Release → PyPI upload). Direct `git tag`
or `twine upload` are out of band — both are blocked (branch protection
on `main`, and the `pypi` environment requires reviewer approval before
the upload step).

## Eligibility — patch / minor / major?

| Bump            | Use for                                                                               |
|-----------------|---------------------------------------------------------------------------------------|
| Patch (`0.6.x`) | Bug fixes, code quality, docs, CI/tooling, tests. No new config fields, no output changes. |
| Minor (`0.x.0`) | New features (default-off), new config fields, new templates, manifest schema bumps.  |
| Major (`x.0.0`) | Breaking changes to public API, config schema, or output format.                      |

If unsure, scan `## [Unreleased]` in `CHANGELOG.md`:

- `### Added` of new config fields → minor.
- `### Removed` or a `### Migration` note → major.
- Otherwise → patch.

## Step 1 — Prep PR

1. Branch from `main` (e.g. `release/0.6.1`).
2. Bump the version in **both** source files:
   - `plotsim/__init__.py` — `__version__ = "0.6.1"`
   - `pyproject.toml` — `version = "0.6.1"` (in `[project]`)
3. In `CHANGELOG.md`, rename `## [Unreleased]` to
   `## [0.6.1] — YYYY-MM-DD` (em-dash `—` U+2014, single space on each
   side). Seed a fresh empty `## [Unreleased]` block above it.
4. Open a PR. CI must pass. **Squash-merge** into `main`.

## Step 2 — Cut the release

1. **Actions → Release → Run workflow**, or:
   ```
   gh workflow run release.yml -f version=0.6.1
   ```
2. Enter the version (`0.6.1`, no `v` prefix) and run.
3. `release.yml` runs five jobs in sequence; each must succeed before
   the next starts:
   - **`validate`** — version is `X.Y.Z` (digits and dots only),
     `plotsim/__init__.py` and `pyproject.toml` both contain that
     version on the expected line, `CHANGELOG.md` has a dated
     `## [X.Y.Z] — YYYY-MM-DD` section, and tag `vX.Y.Z` does not
     already exist (local or remote). Extracts the changelog section
     for the GitHub Release body.
   - **`test`** — Python 3.10–3.13 matrix (`ruff check` + full
     `pytest`). A failure here aborts the release before any tag
     gets created — no orphan tag, no broken-code release.
   - **`build`** — `python -m build` produces wheel + sdist, uploaded
     as a workflow artifact.
   - **`tag-and-release`** — annotated tag `vX.Y.Z` is created and
     pushed, then a GitHub Release is created with the changelog
     section as the body.
   - **`publish`** — pauses at the `pypi` environment for
     required-reviewer approval. After a repo admin (`mohossam01`)
     approves in the GitHub UI, the wheel + sdist are uploaded to
     PyPI via OIDC Trusted Publishing (no API tokens).
4. Verify within ~5 minutes:
   ```
   pip install plotsim==0.6.1
   python -c "import plotsim; print(plotsim.__version__)"
   ```

## Anti-patterns

- **Don't push tags manually.** No workflow listens on
  `push: tags` — a manual tag push leaves an orphan tag and
  publishes nothing. If you reach for `git tag` directly, fix
  the missing step (usually the prep PR didn't bump one of the
  files or didn't date the CHANGELOG) and re-dispatch
  `release.yml`.
- **Don't `twine upload` from a laptop.** Trusted Publishing is
  configured for this repo + the `publish` job + the `pypi`
  environment. API-token uploads are not configured and will 403.
- **Don't bump version without a CHANGELOG entry.** `release.yml`
  refuses to dispatch.
- **Don't use `--` (ASCII double-hyphen) in the CHANGELOG date
  separator.** The validation expects an em-dash `—` (U+2014). Editor
  autocorrect normally produces this; if it doesn't, paste it in.
