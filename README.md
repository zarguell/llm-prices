# LLM Prices Watch

A deterministic, weekly archive of LLM API prices — built from
[models.dev](https://models.dev) (free, open model catalog) and rendered into
a static GitHub Pages site. **No usage telemetry is published here** — only
public pricing data.

Architecture twin of [carmens-names](https://github.com/zarguell/carmens-names)
and [tia-n-list](https://github.com/zarguell/tia-n-list): `engine/ssg.py`
(logic only) + `engine/templates/` (Jinja2), rendered into the repo root;
Actions tests + build + deploy on push and a weekly cron.

## What you get

- **Prices** — per-provider model tables: input/output/cache rates (USD per
  million tokens), context limits, tool/reasoning flags, release dates
- **Providers** — coverage analytics: priced share, medians, open vs closed weights
- **Trends** — inline-SVG price-trend charts across weekly snapshots (no JS chart libs)
- **Changes** — weekly diff: new models, removed models, price moves
- **Stats** — cheapest / priciest / best-value (tools + ≥128k context)
- **Raw archive** — `data/snapshots/YYYY-MM-DD.json.gz`, one immutable weekly
  bundle, served as-is; machine-readable current prices at `data/latest.json`

The snapshot archive doubles as the local pricing cache for cronman's
`pi-usage` report (which reads the newest snapshot instead of fetching
models.dev itself).

## Layout

- `data/snapshots/YYYY-MM-DD.json.gz` — the data store (committed; one file
  per week, date-level timestamps, never rewritten)
- `engine/snapshot.py` — fetch + write one snapshot (`--from-file` for seeding)
- `engine/ssg.py` — all build logic (markup lives in `engine/templates/`)
- `engine/tracked.json` — models plotted on the Trends page
- Rendered site sits in the repo ROOT and is **gitignored build output**;
  never hand-edit it — edit templates and rebuild.

## Commands

```bash
.venv/bin/python engine/snapshot.py                 # fetch this week's snapshot
.venv/bin/python engine/test_snapshot.py            # contract tests (CI gate)
.venv/bin/python engine/test_ssg.py                 # contract tests (CI gate)
.venv/bin/python engine/ssg.py                      # render site into repo root
python3 -m http.server 8000                         # browse localhost:8000
```

(No jinja2 in system python: `.venv/` carries it — recreate with
`python3 -m venv .venv && .venv/bin/pip install jinja2` if missing.)

## Weekly pipeline

`cronman llm-prices` (Mon 13:30 UTC): pull → snapshot → contract tests →
local build gate → commit `data/` → push. GitHub Actions rebuilds the site
and deploys to Pages. Half an hour later, `cronman pi-usage` prices the
weekly LLM usage report from the fresh snapshot.
