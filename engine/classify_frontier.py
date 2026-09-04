#!/usr/bin/env python3
"""LLM Prices — auto-classify frontier models from OpenRouter benchmarks
+ models.dev pricing data.

Replaces hand-curated frontier.json with a data-driven classification:
1. Pull ALL OpenRouter models with benchmark ELO scores
2. Cross-reference models.dev for pricing, context, release date, capabilities
3. Deduplicate (skip batch/free variants, pick canonical per family)
4. Classify tiers by ELO + price + recency
5. Apply manual overrides from frontier-overrides.json
6. Output engine/frontier.json

Usage: python3 engine/classify_frontier.py [--snapshot PATH] [--or-cache PATH]

Run weekly by the llm-prices cron job before enrich.py. The output is
committed automatically — human review is via git diff.
"""
import argparse
import json
import os
import re
import urllib.request
from datetime import datetime, timezone

ENGINE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(ENGINE)
OVERRIDES_FILE = os.path.join(ENGINE, "frontier-overrides.json")
OUTPUT_FILE = os.path.join(ENGINE, "frontier.json")
SNAPSHOT_DIR = os.path.join(ROOT, "data", "snapshots")
OPENROUTER_URL = "https://openrouter.ai/api/v1/models"
USER_AGENT = "llm-prices-classify/1.0 (weekly frontier classification)"

# Tier thresholds — tweak these to adjust classification
ELO_FRONTIER = 1350          # absolute ELO for frontier
ELO_FRONTIER_PRICE = 1300    # ELO for frontier if price ≥ $2/M
ELO_EFFICIENT = 1200         # absolute ELO for efficient
ELO_EFFICIENT_LAB = 1100     # ELO for efficient from major labs
PRICE_FRONTIER = 2.0         # $/M input price threshold for frontier
PRICE_WORKHORSE = 1.0        # $/M input price threshold for workhorse
MAX_FRONTIER = 8             # max models in frontier tier
MAX_EFFICIENT = 12           # max models in efficient tier
MAX_WORKHORSE = 6            # max models in workhorse tier
MAJOR_LABS = {"openai", "anthropic", "google", "meta-llama", "x-ai",
              "moonshotai", "z-ai", "deepseek", "qwen", "minimax",
              "tencent", "meta"}


def _fetch_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def load_snapshot(path=None):
    """Load the newest models.dev snapshot."""
    import gzip, glob
    if path:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f).get("providers", {})
    files = sorted(glob.glob(os.path.join(SNAPSHOT_DIR, "????-??-??.json.gz")))
    if not files:
        return {}
    with gzip.open(files[-1], "rt", encoding="utf-8") as f:
        return json.load(f).get("providers", {})


def load_openrouter(cache_path=None):
    """Load OpenRouter model list (from cache or live)."""
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "data" in data:
            return {m["id"]: m for m in data["data"]
                    if isinstance(m, dict) and "id" in m}
    data = _fetch_json(OPENROUTER_URL)
    return {m["id"]: m for m in data.get("data", [])
            if isinstance(m, dict) and "id" in m}


def extract_best_elo(or_model):
    """Extract best ELO across all benchmark categories."""
    benchmarks = or_model.get("benchmarks") or {}
    elos = []
    for arena, entries in benchmarks.items():
        if isinstance(entries, list):
            for e in entries:
                if isinstance(e, dict) and "elo" in e:
                    elos.append((e["elo"], e.get("category", "?"), arena))
    if not elos:
        return None, [], 0
    elos.sort(key=lambda x: -x[0])
    return elos[0][0], elos, len(elos)


def deduplicate(or_models):
    """Deduplicate OpenRouter models: skip batch/free variants, pick
    canonical per base model (most benchmark categories wins)."""
    canonical = {}
    for oid, m in or_models.items():
        if ":batch" in oid or ":free" in oid:
            continue
        elo, cats, n_cats = extract_best_elo(m)
        if elo is None:
            continue
        base = oid.split(":")[0]
        if base not in canonical or n_cats > canonical[base]["n_cats"]:
            canonical[base] = {
                "or_id": oid, "elo": elo, "n_cats": n_cats,
                "cats": cats, "or_model": m,
            }
    return canonical


def cross_reference(or_entry, providers):
    """Look up models.dev data for an OpenRouter model."""
    or_id = or_entry["or_id"]
    parts = or_id.split("/", 1)
    if len(parts) != 2:
        return {}
    prov, mid = parts

    # Try exact match
    dev_models = providers.get(prov, {}).get("models") or {}
    dm = dev_models.get(mid)
    if dm:
        return _dev_fields(dm)

    # Try fuzzy: scan all providers for matching model id
    for pid, pdata in providers.items():
        for model_id, dm in (pdata.get("models") or {}).items():
            if model_id == mid:
                return _dev_fields(dm)
    return {}


def _dev_fields(dm):
    c = dm.get("cost") or {}
    l = dm.get("limit") or {}
    return {
        "price_in": c.get("input"),
        "price_out": c.get("output"),
        "context": l.get("context"),
        "max_output": l.get("output"),
        "released": dm.get("release_date"),
        "tools": dm.get("tool_call"),
        "reasoning": dm.get("reasoning"),
        "open_weights": dm.get("open_weights"),
        "family": dm.get("family"),
        "knowledge": dm.get("knowledge"),
        "modalities": dm.get("modalities"),
    }


def classify_tier(elo, price_in, released, tools, reasoning, open_weights, family):
    """Assign tier based on ELO + price + capabilities."""
    has_benchmarks = elo is not None and elo > 0
    is_capable = tools and reasoning

    if has_benchmarks:
        if elo >= ELO_FRONTIER:
            return "frontier"
        if elo >= ELO_FRONTIER_PRICE and price_in is not None and price_in >= PRICE_FRONTIER:
            return "frontier"
        if elo >= ELO_EFFICIENT:
            return "efficient"
        if elo >= ELO_EFFICIENT_LAB:
            return "efficient"

    # No benchmarks — classify by price + capabilities
    if price_in is not None:
        if price_in >= PRICE_FRONTIER and is_capable:
            return "frontier"
        if price_in >= 0.10 and is_capable:
            return "efficient"
        if price_in < PRICE_WORKHORSE:
            return "workhorse"

    # Fallback: open weights = workhorse, else efficient
    if open_weights:
        return "workhorse"
    return "efficient"


def load_overrides():
    """Load manual overrides (include/exclude/tier overrides)."""
    if not os.path.exists(OVERRIDES_FILE):
        return {"include": [], "exclude": [], "tier_override": {}}
    with open(OVERRIDES_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        "include": data.get("include", []),
        "exclude": data.get("exclude", []),
        "tier_override": data.get("tier_override", {}),
    }


def classify(snapshot_path=None, or_cache_path=None):
    """Main classification pipeline. Returns the full frontier dict."""
    providers = load_snapshot(snapshot_path)
    or_models = load_openrouter(or_cache_path)
    canonical = deduplicate(or_models)
    overrides = load_overrides()

    # Build exclude set
    exclude_set = set(overrides.get("exclude", []))
    include_set = set(overrides.get("include", []))
    tier_overrides = overrides.get("tier_override", {})

    models = []
    for base, entry in canonical.items():
        or_id = entry["or_id"]
        if or_id in exclude_set:
            continue

        dev = cross_reference(entry, providers)
        elo = entry["elo"]
        price_in = dev.get("price_in")
        released = dev.get("released")
        tools = dev.get("tools")
        reasoning = dev.get("reasoning")
        open_weights = dev.get("open_weights")
        family = dev.get("family")

        tier = tier_overrides.get(or_id) or classify_tier(
            elo, price_in, released, tools, reasoning, open_weights, family)

        # For models.dev key, use the OR id (enrich.py handles cross-ref)
        key = or_id
        # Try to find the models.dev key if it differs
        parts = or_id.split("/", 1)
        if len(parts) == 2:
            prov, mid = parts
            dev_models = providers.get(prov, {}).get("models") or {}
            if mid in dev_models:
                key = or_id  # same as OR id
            else:
                # try matching across providers
                for pid, pdata in providers.items():
                    if mid in (pdata.get("models") or {}):
                        key = f"{pid}/{mid}"
                        break

        models.append({
            "key": key,
            "or_id": or_id,
            "hf_id": _guess_hf_id(or_id, open_weights),
            "tier": tier,
            "elo": elo,
            "n_cats": entry["n_cats"],
            "price_in": price_in,
            "released": released,
            "params_b": None,
        })

    # Add forced includes not already present
    existing_ids = {m["or_id"] for m in models}
    for inc in include_set:
        if inc in existing_ids:
            continue
        or_m = or_models.get(inc, {})
        elo, _, n_cats = extract_best_elo(or_m)
        dev = cross_reference({"or_id": inc}, providers)
        tier = tier_overrides.get(inc) or classify_tier(
            elo, dev.get("price_in"), dev.get("released"),
            dev.get("tools"), dev.get("reasoning"),
            dev.get("open_weights"), dev.get("family"))
        models.append({
            "key": inc, "or_id": inc,
            "hf_id": _guess_hf_id(inc, dev.get("open_weights")),
            "tier": tier, "elo": elo, "n_cats": n_cats,
            "price_in": dev.get("price_in"),
            "released": dev.get("released"), "params_b": None,
        })

    # Sort by ELO descending within each tier, then cap
    tier_order = {"frontier": 0, "efficient": 1, "workhorse": 2}
    tier_caps = {"frontier": MAX_FRONTIER, "efficient": MAX_EFFICIENT, "workhorse": MAX_WORKHORSE}
    # Separate auto-classified from forced includes
    auto = [m for m in models if m["or_id"] not in include_set]
    forced = [m for m in models if m["or_id"] in include_set]
    # Cap auto-classified, then append forced
    auto.sort(key=lambda m: (tier_order.get(m["tier"], 9), -(m["elo"] or 0)))
    capped = []
    tier_counts = {"frontier": 0, "efficient": 0, "workhorse": 0}
    for m in auto:
        t = m["tier"]
        if tier_counts.get(t, 0) < tier_caps.get(t, 99):
            capped.append(m)
            tier_counts[t] = tier_counts.get(t, 0) + 1
    models = capped + forced

    return {
        "tiers": {
            "frontier": {
                "label": "Frontier",
                "description": "Latest generation, highest capability.",
                "color": "#D97757",
            },
            "efficient": {
                "label": "Efficient",
                "description": "Capable models at a fraction of frontier cost.",
                "color": "#7C9885",
            },
            "workhorse": {
                "label": "Workhorse",
                "description": "Ubiquitous open and mid-range models.",
                "color": "#5B8DB8",
            },
        },
        "models": [{k: v for k, v in m.items()
                    if k in ("key", "or_id", "hf_id", "tier", "params_b", "elo", "n_cats")}
                   for m in models],
        "_meta": {
            "generated": datetime.now(timezone.utc).isoformat(),
            "total": len(models),
            "by_tier": {t: sum(1 for m in models if m["tier"] == t)
                        for t in ("frontier", "efficient", "workhorse")},
            "benchmark_categories": sum(m["n_cats"] for m in models),
        },
    }


def _guess_hf_id(or_id, open_weights):
    """Best-guess HuggingFace id for open-weight models."""
    if not open_weights:
        return None
    parts = or_id.split("/", 1)
    if len(parts) == 2:
        return f"{parts[0]}/{parts[1]}"
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Classify frontier models from data")
    ap.add_argument("--snapshot", help="Path to models.dev snapshot .json.gz")
    ap.add_argument("--or-cache", help="Path to OpenRouter models cache .json")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print classification to stdout, don't write frontier.json")
    args = ap.parse_args(argv)

    result = classify(args.snapshot, args.or_cache)

    if args.dry_run:
        meta = result.pop("_meta")
        print(json.dumps(result, indent=2))
        result["_meta"] = meta
        print(f"\n--- {meta['total']} models classified ---")
        for t in ("frontier", "efficient", "workhorse"):
            count = meta["by_tier"][t]
            models = [m for m in result["models"] if m["tier"] == t]
            names = [m["or_id"].split("/")[-1] for m in models[:5]]
            print(f"  {t}: {count} — {', '.join(names)}{'...' if count > 5 else ''}")
        print(f"  benchmark categories: {meta['benchmark_categories']}")
        return 0

    # Write output
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=False)
    meta = result["_meta"]
    print(f"CLASSIFIED: {meta['total']} models → {OUTPUT_FILE}")
    for t in ("frontier", "efficient", "workhorse"):
        print(f"  {t}: {meta['by_tier'][t]}")
    print(f"  benchmark categories: {meta['benchmark_categories']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
