# plotsim · Brand asset pack

Everything in this folder is ready to drop into a real project. Source-of-truth
files are SVG/CSS/TOML — they're text, edit them freely.

## Folder map

```
assets/
  mark.svg                       Full mark (200×144 default, viewBox-based)
  mark-simplified.svg            For ≤64 px contexts
  mark-mono.svg                  Single-color (uses currentColor)
  favicon.svg                    Rounded-square glyph
  wordmark.svg                   plot.sim typeset
  lockup-horizontal-dark.svg     Mark + word, on ink
  lockup-horizontal-light.svg    Mark + word, on paper
  lockup-stacked-dark.svg        Square contexts, on ink
  lockup-stacked-light.svg       Square contexts, on paper
  tokens.css                     CSS custom properties — Observatory tokens
  manifest.webmanifest           PWA manifest
  head.html                      Drop-in <head> snippet (favicons, og, fonts)

templates/
  README.md                      Repo README starter (uses banner + badges)
  mkdocs.yml                     Material docs config wired to tokens
  .streamlit/config.toml         Streamlit theme

```

## Domain wiring

The brand uses **plotsim.dev** as canonical. Replace `{{DOMAIN}}` in
`assets/head.html` with your full URL. Update `templates/README.md` and
`templates/mkdocs.yml` if you land elsewhere.

The domain shows up on:

- README banner (bottom strip)
- GitHub social card (bottom-right CTA)
- Docs footer
- Slide-deck footers
- Email signatures
- PyPI long description

It does **not** show up in:

- The mark itself
- Favicons or app icons
- Doc-site headers (URL bar already says it)
- Inside the wordmark — `plot.sim` dot is a namespace, not a TLD

## PNG exports

The PNG files referenced by `head.html` (`favicon-32.png`, `favicon-192.png`,
`favicon-512.png`, `apple-touch-icon.png`, `og-image.png`, `readme-banner.png`)
are not generated in this pack. Render them from the SVGs using a tool of your
choice — `rsvg-convert`, ImageMagick, Figma export, or open the brand-kit HTML
in a browser and screenshot the relevant sections.

For og-image and readme-banner specifically: the brand-kit HTML
(`plotsim observatory kit.html`) renders both at 1280×640 — screenshot them
from there for pixel-perfect output.

## Colors at a glance

| Token            | Dark      | Light    | Use                       |
| ---------------- | --------- | -------- | ------------------------- |
| ink              | `#0e1620` |          | Primary dark surface      |
| paper            |           | `#ffffff`| Primary light surface     |
| teal (hero)      | `#3ecfc1` | `#1a9a8c`| Hero signal               |
| amber (counter)  | `#e9b04a` | `#c98a1f`| Counter signal            |
| rose (anomaly)   | `#e26d8a` |          | Warnings, anomalies       |
| muted            | `#5a6b7a` |          | Trailing, structural      |

Light-theme teal/amber are deepened to clear WCAG AA on white.
