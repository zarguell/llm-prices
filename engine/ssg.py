#!/usr/bin/env python3
"""LLM Prices — static site generator.

Architecture twin of carmens-names / tia-n-list: all markup lives in
engine/templates/, this file is logic only. Reads the snapshot store
(data/snapshots/YYYY-MM-DD.json.gz, newest = current), builds render
contexts, and renders the Jinja2 templates into the repository root
(GitHub Pages serves the root via the site-deploy workflow).

No usage telemetry of any kind is published — only public model pricing
data from models.dev.

Outputs (repo root):
  index.html                    overview: headline stats + provider grid
  prices/index.html             price tables per provider (index)
  prices/<provider>/index.html  full model price table for one provider
  providers/index.html          provider analytics (coverage, weights, modalities)
  trends/index.html             SVG price-trend charts across snapshots
  changes/index.html            diff latest vs previous snapshot
                                (new / removed / price changes)
  stats/index.html              analytics: cheapest, priciest, best value
  about/index.html              methodology
  data/latest.json              machine-readable current price table
  sitemap.xml, robots.txt, 404.html, style.css

Usage: python3 ssg.py   (run from engine/; needs jinja2)
"""
import glob
import gzip
import html
import json
import math
import os
import statistics
from datetime import datetime, timezone

from jinja2 import Environment, FileSystemLoader, select_autoescape

ENGINE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(ENGINE)
TMPL_DIR = os.path.join(ENGINE, "templates")
SNAPSHOTS_DIR = os.path.join(ROOT, "data", "snapshots")
TRACKED_FILE = os.path.join(ENGINE, "tracked.json")
BASE_URL = "https://zarguell.github.io/llm-prices/"
SITE_NAME = "LLM Prices Watch"


# ── snapshot store ──

def snapshot_files(data_dir: str = SNAPSHOTS_DIR) -> list[str]:
    return sorted(glob.glob(os.path.join(data_dir, "????-??-??.json.gz")))


def load_snapshots(data_dir: str = SNAPSHOTS_DIR) -> list[tuple[str, dict]]:
    """[(date_str, providers_dict)] oldest → newest; undecodable files skipped."""
    out = []
    for path in snapshot_files(data_dir):
        try:
            with gzip.open(path, "rt", encoding="utf-8") as f:
                entry = json.load(f)
            date_str = os.path.basename(path)[:10]
            providers = entry.get("providers")
            if isinstance(providers, dict):
                out.append((date_str, providers))
        except Exception:
            continue
    return out


# ── flattening ──

def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def flatten(providers: dict) -> list[dict]:
    """Deterministic row per model: pricing + capability metadata."""
    rows = []
    for prov in sorted(providers):
        pdata = providers[prov] if isinstance(providers[prov], dict) else {}
        prov_name = pdata.get("name") or prov
        for mid in sorted((pdata.get("models") or {})):
            m = (pdata.get("models") or {}).get(mid)
            if not isinstance(m, dict):
                continue
            cost = m.get("cost") if isinstance(m.get("cost"), dict) else {}
            lim = m.get("limit") if isinstance(m.get("limit"), dict) else {}
            modal = m.get("modalities") if isinstance(m.get("modalities"), dict) else {}
            priced = "input" in cost and "output" in cost
            rows.append({
                "provider": prov,
                "provider_name": prov_name,
                "id": mid,
                "key": f"{prov}/{mid}",
                "name": m.get("name") or mid,
                "priced": priced,
                "input": _f(cost.get("input")) if priced else None,
                "output": _f(cost.get("output")) if priced else None,
                "cache_read": _f(cost.get("cache_read")),
                "cache_write": _f(cost.get("cache_write")),
                "context": lim.get("context"),
                "max_output": lim.get("output"),
                "tool_call": bool(m.get("tool_call")),
                "reasoning": bool(m.get("reasoning")),
                "open_weights": bool(m.get("open_weights")),
                "release_date": m.get("release_date") or "",
                "inputs": modal.get("input") or [],
                "outputs": modal.get("output") or [],
            })
    return rows


def by_key(rows: list[dict]) -> dict:
    return {r["key"]: r for r in rows}


# ── diffing ──

def diff(cur_rows: list[dict], prev_rows: list[dict]) -> dict:
    """New / removed models and price changes between two snapshots."""
    cur, prev = by_key(cur_rows), by_key(prev_rows)
    added = sorted(k for k in cur if k not in prev)
    removed = sorted(k for k in prev if k not in cur)
    changed = []
    for k in sorted(set(cur) & set(prev)):
        c, p = cur[k], prev[k]
        if not (c["priced"] and p["priced"]):
            continue
        for field in ("input", "output"):
            if c[field] is None or p[field] is None:
                continue
            if abs(c[field] - p[field]) > 1e-9:
                pct = (c[field] - p[field]) / p[field] * 100.0 if p[field] else None
                changed.append({"key": k, "field": field, "old": p[field],
                                "new": c[field], "pct": pct})
    changed.sort(key=lambda ch: (ch["key"], ch["field"]))
    return {"added": added, "removed": removed, "changed": changed}


# ── analytics ──

def _median(vals):
    vals = [v for v in vals if v is not None]
    return statistics.median(vals) if vals else None


def compute_stats(rows: list[dict]) -> dict:
    priced = [r for r in rows if r["priced"]]
    with_tools_ctx = [r for r in priced if r["tool_call"]
                      and (r["context"] or 0) >= 128_000]
    by_provider = {}
    for r in rows:
        p = by_provider.setdefault(r["provider"], {"name": r["provider_name"],
                                                   "models": 0, "priced": 0,
                                                   "inputs": []})
        p["models"] += 1
        if r["priced"]:
            p["priced"] += 1
            p["inputs"].append(r["input"])
    providers = []
    for prov in sorted(by_provider):
        p = by_provider[prov]
        providers.append({
            "provider": prov, "name": p["name"], "models": p["models"],
            "priced": p["priced"],
            "coverage": round(p["priced"] / p["models"] * 100.0, 1) if p["models"] else 0.0,
            "median_input": _median(p["inputs"]),
        })
    open_m = [r["input"] for r in priced if r["open_weights"]]
    closed_m = [r["input"] for r in priced if not r["open_weights"]]
    return {
        "providers_n": len(by_provider),
        "models_n": len(rows),
        "priced_n": len(priced),
        "coverage": round(len(priced) / len(rows) * 100.0, 1) if rows else 0.0,
        "median_input": _median([r["input"] for r in priced]),
        "cheapest": sorted(priced, key=lambda r: (r["input"], r["output"], r["key"]))[:25],
        "priciest": sorted(priced, key=lambda r: (-r["input"], -r["output"], r["key"]))[:10],
        "best_value": sorted(with_tools_ctx,
                             key=lambda r: (r["input"], r["output"], r["key"]))[:25],
        "open_median": _median(open_m),
        "open_n": len(open_m),
        "closed_median": _median(closed_m),
        "closed_n": len(closed_m),
        "providers": providers,
    }


# ── SVG line chart (deterministic, dependency-free) ──

def svg_line_chart(dates: list[str], series: list[dict], ylog: bool = True,
                   width: int = 880, height: int = 360, unit: str = "/M") -> str:
    """Multi-series line chart as SVG. series: [{"label", "color",
    "points": [value-or-None per date]}]. Log y-scale, <title> tooltips."""
    pad_l, pad_r, pad_t, pad_b = 56, 16, 14, 40
    vals = [v for s in series for v in s["points"] if v is not None and v > 0]
    if not vals or not dates:
        return "<p class='empty'>no data yet</p>"
    tmin, tmax = min(vals), max(vals)
    lo = math.log10(tmin) if ylog and tmin > 0 else 0.0
    hi = math.log10(tmax) if ylog and tmax > 0 else 1.0
    if hi - lo < 1e-9:
        hi = lo + 1.0
    n = len(dates)

    def X(i):
        return pad_l + (width - pad_l - pad_r) * (i / max(n - 1, 1))

    def Y(v):
        lv = math.log10(v) if ylog and v > 0 else (math.log10(min(vals)) if vals else 0)
        lv = min(max(lv, lo), hi)
        return pad_t + (height - pad_t - pad_b) * (1 - (lv - lo) / (hi - lo))

    parts = [f"<svg viewBox='0 0 {width} {height}' class='chart' role='img'>"]
    # y grid at log decades
    step = max(1, int(math.ceil(hi - lo) / 5))
    exp = math.floor(lo)
    while exp <= math.ceil(hi):
        v = 10 ** exp
        if v >= tmin / 3 and v <= tmax * 3:
            y = Y(v)
            parts.append(f"<line x1='{pad_l}' y1='{y:.1f}' x2='{width-pad_r}' "
                         f"y2='{y:.1f}' class='grid'/>")
            label = f"${v:g}" if v >= 1 else f"${v:.2f}"
            parts.append(f"<text x='{pad_l-8}' y='{y+4:.1f}' class='axis' "
                         f"text-anchor='end'>{html.escape(label)}{html.escape(unit)}</text>")
        exp += step
    # x labels (first, last, quarter marks)
    for i in sorted({0, n - 1, n // 4, n // 2, (3 * n) // 4}):
        parts.append(f"<text x='{X(i):.1f}' y='{height-10}' class='axis' "
                     f"text-anchor='middle'>{html.escape(dates[i])}</text>")
    # series
    for s in series:
        pts = [(X(i), Y(v), v, dates[i]) for i, v in enumerate(s["points"])
               if v is not None and v > 0]
        if not pts:
            continue
        if len(pts) > 1:
            d = " ".join(("M" if j == 0 else "L") + f"{x:.1f},{y:.1f}"
                         for j, (x, y, _, _) in enumerate(pts))
            parts.append(f"<path d='{d}' fill='none' stroke='{s['color']}' "
                         f"stroke-width='2'/>")
        for x, y, v, date in pts:
            parts.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='3.5' "
                         f"fill='{s['color']}'><title>{html.escape(s['label'])} — "
                         f"${v:g}{html.escape(unit)} on {html.escape(date)}</title></circle>")
    # legend
    lx = pad_l
    for s in series:
        parts.append(f"<rect x='{lx}' y='2' width='10' height='10' fill='{s['color']}' rx='2'/>")
        parts.append(f"<text x='{lx+14}' y='11' class='legend'>{html.escape(s['label'])}</text>")
        lx += 14 + 6 * len(s["label"]) + 18
    parts.append("</svg>")
    return "".join(parts)


PALETTE = ["#D97757", "#7C9885", "#5B8DB8", "#B8865B", "#9A7CB3",
           "#C25B5B", "#5FA8A0", "#8A8D65"]


def tracked_series(snapshots: list[tuple[str, dict]], tracked: list[str],
                   field: str) -> tuple[list[str], list[dict]]:
    """(dates, series) for tracked 'provider/model' keys across snapshots."""
    dates = [d for d, _ in snapshots]
    index = [{}, ]
    per_date = []
    for _, providers in snapshots:
        rows = by_key(flatten(providers))
        per_date.append(rows)
    series = []
    for t_i, key in enumerate(tracked):
        points = []
        for rows in per_date:
            r = rows.get(key)
            points.append(r.get(field) if r and r.get(field) is not None else None)
        if any(v is not None for v in points):
            series.append({"label": key, "color": PALETTE[t_i % len(PALETTE)],
                           "points": points})
    return dates, series


# ── build ──

def _rel(out_path: str) -> str:
    depth = out_path.count(os.sep) - ROOT.count(os.sep)
    return "../" * depth


def _write(path: str, content: str, manifest: list[str], root: str = ROOT) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    manifest.append(os.path.relpath(path, root))


def build(root: str = ROOT, data_dir: str = SNAPSHOTS_DIR,
          tracked: list[str] | None = None) -> list[str]:
    env = Environment(loader=FileSystemLoader(TMPL_DIR),
                      autoescape=select_autoescape(["html", "xml"]))
    snapshots = load_snapshots(data_dir)
    if not snapshots:
        raise SystemExit("no snapshots found — run engine/snapshot.py first")
    cur_date, cur_providers = snapshots[-1]
    prev_rows = flatten(snapshots[-2][1]) if len(snapshots) > 1 else None
    rows = flatten(cur_providers)
    stats = compute_stats(rows)
    if tracked is None:
        try:
            with open(TRACKED_FILE, "r", encoding="utf-8") as f:
                tracked = [t for t in json.load(f) if isinstance(t, str)]
        except Exception:
            tracked = []
    d = diff(rows, prev_rows) if prev_rows is not None else None
    now = datetime.now(timezone.utc)
    generated = now.strftime("%Y-%m-%d")
    providers_sorted = stats["providers"]
    manifest: list[str] = []

    def ctx(r, **kw):
        kw.update(r=r, site_name=SITE_NAME, base_url=BASE_URL,
                  data_date=cur_date, generated=generated,
                  stats=stats, snapshots_n=len(snapshots))
        return env.get_template(kw.pop("template")).render(**kw)

    # index
    _write(os.path.join(root, "index.html"),
           ctx("index.html", template="index.html", rows=rows, diff=d,
               cheapest=stats["cheapest"][:8]), manifest, root)

    # prices: provider index + per-provider tables
    priced_by_provider = {}
    for row in rows:
        priced_by_provider.setdefault(row["provider"], []).append(row)
    _write(os.path.join(root, "prices", "index.html"),
           ctx("prices_index.html", template="prices_index.html",
               providers=providers_sorted), manifest, root)
    for prov in sorted(priced_by_provider):
        prov_rows = sorted(priced_by_provider[prov],
                           key=lambda r: (not r["priced"], r["input"] or 9e9,
                                          r["output"] or 9e9, r["id"]))
        pname = next((p["name"] for p in providers_sorted
                      if p["provider"] == prov), prov)
        _write(os.path.join(root, "prices", prov, "index.html"),
               ctx("provider.html", template="provider.html", rows=prov_rows,
                   provider=prov, provider_name=pname), manifest, root)

    # providers analytics
    _write(os.path.join(root, "providers", "index.html"),
           ctx("providers.html", template="providers.html"), manifest, root)

    # trends
    in_dates, in_series = tracked_series(snapshots, tracked, "input")
    out_dates, out_series = tracked_series(snapshots, tracked, "output")
    chart_in = svg_line_chart(in_dates, in_series) if in_series else None
    chart_out = svg_line_chart(out_dates, out_series) if out_series else None
    _write(os.path.join(root, "trends", "index.html"),
           ctx("trends.html", template="trends.html", tracked=tracked,
               chart_in_svg=svg_line_chart(in_dates, in_series) if in_series else "",
               chart_out_svg=svg_line_chart(out_dates, out_series) if out_series else "",
               dates=in_dates), manifest, root)

    # changes
    _write(os.path.join(root, "changes", "index.html"),
           ctx("changes.html", template="changes.html", diff=d,
               prev_date=snapshots[-2][0] if len(snapshots) > 1 else None,
               cur_rows=rows, prev_rows=prev_rows), manifest, root)

    # stats
    _write(os.path.join(root, "stats", "index.html"),
           ctx("stats.html", template="stats.html"), manifest, root)

    # about
    _write(os.path.join(root, "about", "index.html"),
           ctx("about.html", template="about.html"), manifest, root)

    # machine-readable current prices
    _write(os.path.join(root, "data", "latest.json"),
           json.dumps({"as_of": cur_date, "source": "models.dev",
                       "models": sorted(rows, key=lambda r: r["key"])},
                      indent=1, sort_keys=True) + "\n", manifest, root)

    # static-ish pages
    _write(os.path.join(root, "style.css"),
           env.get_template("style.css").render(), manifest, root)
    _write(os.path.join(root, "404.html"),
           ctx("404.html", template="404.html"), manifest, root)
    _write(os.path.join(root, "robots.txt"),
           env.get_template("robots.txt").render(base_url=BASE_URL), manifest, root)
    urls = ["", "prices/", "providers/", "trends/", "changes/", "stats/", "about/"]
    for prov in sorted(priced_by_provider):
        urls.append(f"prices/{prov}/")
    _write(os.path.join(root, "sitemap.xml"),
           env.get_template("sitemap.xml").render(
               base_url=BASE_URL, urls=urls, generated=generated), manifest, root)
    return sorted(set(manifest))


def main() -> int:
    manifest = build()
    print(f"BUILT {len(manifest)} files into {ROOT}")
    for m in manifest:
        print(f"  {m}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
