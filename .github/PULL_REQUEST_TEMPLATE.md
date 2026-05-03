## What this PR does

Brief description of the change and the motivation behind it. Link the issue
this closes (`Closes #123`) if applicable.

## How to test

Steps a reviewer can run locally to verify the change:

```bash
# e.g.
python -m pytest tests/path/to/test_file.py -v
```

Include any config snippets or commands needed to reproduce the new behavior.

## Checklist

- [ ] Tests pass locally (`python -m pytest`)
- [ ] Docs build cleanly (`mkdocs build --strict`) if docs were touched
- [ ] No breaking changes to public API — or, if there are, they're called out below
- [ ] CHANGELOG entry added under `[Unreleased]` if user-visible
- [ ] Trajectory-first invariant preserved (metrics derived from trajectory positions, not generated independently)

## Breaking changes

If this PR changes public behavior, list what breaks and the migration path.
Otherwise: _None._

## Additional notes

Anything reviewers should know — follow-up work deferred, related issues,
performance implications, etc.
