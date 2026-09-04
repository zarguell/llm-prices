# AGENTS.md

Notes for anyone (AI agent or human) changing this repo. Read this first.

## What this is

A weekly archive of LLM API prices from models.dev, rendered into a static
GitHub Pages site. Deterministic build, no LLM involved anywhere, and **no
usage telemetry published** — public catalog data only.

## Layout

- `data/snapshots/YYYY-MM-DD.json.gz`: the store. One immutable bundle per
  week (gzip JSON: `{fetched, source, providers}` where `providers` is the
  raw models.dev payload). Date-level timestamps; a same-day re-snapshot
  replaces the file in place rather than forking dates.
- `engine/snapshot.py`: fetch/write a snapshot. `--from-file` seeds from a
  local api.json (used by tests and for bootstrapping).
- `engine/ssg.py`: all build logic. **All markup lives in
  `engine/templates/`**; keep this file logic-only.
- `engine/tracked.json`: `provider/model` ids plotted on the Trends page
  (must exist in a snapshot to be charted; missing ones are skipped).
- Rendered output sits in the repo ROOT and is **gitignored build output**.
  Never hand-edit it; edit templates and rebuild.

## Conventions (shared with carmens-names / tia-n-list)

- Templates: `{{ r }}` prefix for root-relative links (depth-aware).
- Determinism: sort every iteration; no clock timestamps in rendered HTML
  beyond `generated` (date-level) and `data_date` (snapshot date).
- Charts are build-time inline SVG — no JS chart libraries, no CDN assets.
- Prices are USD per million tokens, passed through from models.dev.
  "Priced" = model publishes both input and output rates.
- models.dev sits behind Cloudflare: default python User-Agents get 403 —
  keep the custom UA in snapshot.py.

## Commands

```bash
.venv/bin/python engine/test_snapshot.py   # contract tests (CI gate)
.venv/bin/python engine/test_ssg.py        # contract tests (CI gate)
.venv/bin/python engine/snapshot.py        # fetch this week's snapshot
.venv/bin/python engine/ssg.py             # render site into repo root
python3 -m http.server 8000                # browse localhost:8000
```

## Gotchas

- Don't add client-side chart libraries — the site must stay dependency-free
  and build-identical.
- Don't rewrite past snapshots (even if upstream data changed) — history is
  the product; corrections land in the next week's snapshot.
- `data/latest.json` and all HTML are derived: never edit or commit them.
- The per-provider price tables are big (some providers have hundreds of
  models); keep row markup lean.
