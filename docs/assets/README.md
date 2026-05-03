# Site assets

This directory is reserved for project assets surfaced on the docs site
(logo, favicon, custom illustrations).

Current state: **placeholder**. The site runs on Material for MkDocs
defaults — no custom logo, default favicon. To brand the site, drop:

- `logo.svg` — referenced from `mkdocs.yml` under `theme.logo`
- `favicon.ico` — referenced from `mkdocs.yml` under `theme.favicon`

Then add the references to `mkdocs.yml` and rebuild with `mkdocs build --strict`.

This README is excluded from the published site by the `exclude_docs`
entry in `mkdocs.yml`.
