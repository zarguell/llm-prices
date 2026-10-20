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
import hashlib
import json
import os
import re
import time
import urllib.request
from datetime import datetime, timezone

ENGINE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(ENGINE)
OVERRIDES_FILE = os.path.join(ENGINE, "frontier-overrides.json")
OUTPUT_FILE = os.path.join(ENGINE, "frontier.json")
SNAPSHOT_DIR = os.path.join(ROOT, "data", "snapshots")
OR_CACHE_DIR = os.path.join(ROOT, "data", "or-cache")
HISTORY_DIR = os.path.join(ROOT, "data", "frontier-history")
OPENROUTER_URL = "https://openrouter.ai/api/v1/models"
USER_AGENT = "llm-prices-classify/1.0 (weekly frontier classification)"

# Stability — hysteresis + deterministic caps (2026-10-20: the published list
# flapped daily on OpenRouter ELO jitter of a few points). A demotion or cap
# displacement must clear HYST_ELO points of margin; ties break on or_id so
# equal-ELO models can't coin-flip day to day.
HYST_ELO = 15
# Provider id aliases: same lab, different id on OpenRouter vs models.dev.
# Without this the same model is classified twice (2026-10-20: z-ai/glm-5.3
# AND zai/glm-5.3 in one list).
PROVIDER_ALIASES = {"zai": "z-ai"}
# Live-fetch sanity: OpenRouter serves ~500 models, ~130 with benchmarks.
# Below this the response is partial — fail loudly, don't publish it.
MIN_OR_MODELS = 200
MIN_BENCHMARK_MODELS = 50
FETCH_RETRIES = 3

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


def _fetch_json(url, retries=FETCH_RETRIES):
    """Fetch JSON with retries. Raises RuntimeError (not URLError) so callers
    get one clear failure mode for the cron log."""
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:  # noqa: BLE001 — retry, then raise one error
            last = e
            if attempt + 1 < retries:
                time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"fetch failed after {retries} attempts: {url}: {last}")


def _validate_or_models(or_models):
    """Refuse a partial OpenRouter response instead of classifying it.
    A degraded fetch used to silently re-tier the whole list."""
    with_bench = sum(1 for m in or_models.values()
                      if isinstance(m, dict) and m.get("benchmarks"))
    if len(or_models) < MIN_OR_MODELS or with_bench < MIN_BENCHMARK_MODELS:
        raise RuntimeError(
            f"OpenRouter response looks partial: {len(or_models)} models, "
            f"{with_bench} with benchmarks "
            f"(minimums {MIN_OR_MODELS}/{MIN_BENCHMARK_MODELS}) — refusing to classify")


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
            models = {m["id"]: m for m in data["data"]
                        if isinstance(m, dict) and "id" in m}
            _validate_or_models(models)
            return models
        raise RuntimeError(f"unusable OpenRouter cache (no 'data' list): {cache_path}")
    data = _fetch_json(OPENROUTER_URL)
    models = {m["id"]: m for m in data.get("data", [])
                if isinstance(m, dict) and "id" in m}
    _validate_or_models(models)
    return models


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


def _dedupe_base(oid):
    """Canonical grouping key: variant suffixes (:thinking, :online, …)
    collapse, and aliased provider ids merge (zai/* == z-ai/*)."""
    base = oid.split(":")[0]
    prov, sep, rest = base.partition("/")
    prov = PROVIDER_ALIASES.get(prov, prov)
    return f"{prov}/{rest}" if sep else base


def deduplicate(or_models):
    """Deduplicate OpenRouter models: skip batch/free variants, pick
    canonical per base model (most benchmark categories wins). Ties break
    on (elo, or_id) — NOT on API response order — so equal models can't
    swap identities day to day."""
    canonical = {}
    for oid, m in or_models.items():
        if ":batch" in oid or ":free" in oid:
            continue
        elo, cats, n_cats = extract_best_elo(m)
        if elo is None:
            continue
        base = _dedupe_base(oid)
        cur = canonical.get(base)
        replace = (
            cur is None
            or (n_cats, elo) > (cur["n_cats"], cur["elo"])
            or ((n_cats, elo) == (cur["n_cats"], cur["elo"])
                and oid < cur["or_id"])
        )
        if replace:
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
    prov = PROVIDER_ALIASES.get(prov, prov)

    # Try exact match (alias-resolved)
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


TIER_ORDER = ("frontier", "efficient", "workhorse")


def _lower_edge(tier, price_in):
    """ELO below which a model no longer qualifies for `tier`."""
    if tier == "frontier":
        if price_in is not None and price_in >= PRICE_FRONTIER:
            return ELO_FRONTIER_PRICE
        return ELO_FRONTIER
    if tier == "efficient":
        return ELO_EFFICIENT_LAB
    return None  # workhorse: bottom tier, no lower edge


def _rank_key(row):
    """Total deterministic order: ELO desc (None last), or_id breaks ties."""
    elo = row.get("elo")
    return (elo is None, -(elo or 0), row["or_id"])


def load_previous_tiers(path=None):
    """Incumbent {or_id: tier} from the last published frontier.json.
    Missing/unparsable (first run) → {} → everything is a newcomer."""
    try:
        with open(path or OUTPUT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {m["or_id"]: m["tier"] for m in data.get("models", [])
                if isinstance(m, dict) and m.get("or_id") and m.get("tier")}
    except (OSError, ValueError, AttributeError):
        return {}


def apply_hysteresis(rows, prev_tiers):
    """Hold incumbents at their previous tier unless ELO moved HYST_ELO
    past the tier boundary. Promotions apply immediately; demotions need
    margin. Pinned (override) and ELO-less rows are untouched.
    Returns the number of holds (for logging)."""
    holds = 0
    for r in rows:
        if r.get("pinned") or r.get("elo") is None:
            continue
        prev = prev_tiers.get(r["or_id"])
        if not prev or prev == r["tier"]:
            continue
        if prev not in TIER_ORDER or r["tier"] not in TIER_ORDER:
            continue
        if TIER_ORDER.index(r["tier"]) > TIER_ORDER.index(prev):
            # raw demotion: hold unless clearly below the old tier's edge
            edge = _lower_edge(prev, r.get("price_in"))
            if edge is not None and r["elo"] >= edge - HYST_ELO:
                r["tier"] = prev
                holds += 1
    return holds


def apply_caps(rows, prev_tiers):
    """Cap each auto tier; overflow cascades DOWN one tier instead of
    vanishing from the list (only workhorse overflow is dropped).
    Within a tier incumbents (prev tier == this tier) keep their slots;
    a newcomer displaces only by beating the weakest kept incumbent by
    more than HYST_ELO. All ordering is _rank_key — fully deterministic.
    Returns (kept_rows, dropped_count)."""
    tier_caps = {"frontier": MAX_FRONTIER, "efficient": MAX_EFFICIENT,
                 "workhorse": MAX_WORKHORSE}
    pools = {t: [] for t in TIER_ORDER}
    for r in rows:
        pools.get(r["tier"], pools["workhorse"]).append(r)
    kept = []
    carry = []  # overflow cascaded from the tier above (as newcomers)
    for t in TIER_ORDER:
        # Cascaded rows take the tier they land in (they compete as
        # newcomers; incumbency is judged against prev_tiers, not the label).
        for r in carry:
            r["tier"] = t
        pool = carry + pools[t]
        incumbents = sorted(
            (r for r in pool if prev_tiers.get(r["or_id"]) == t),
            key=_rank_key)
        newcomers = sorted(
            (r for r in pool if prev_tiers.get(r["or_id"]) != t),
            key=_rank_key)
        cap = tier_caps[t]
        keep = incumbents[:cap]
        overflow = incumbents[cap:]
        for n in newcomers:
            if len(keep) < cap:
                keep.append(n)
                continue
            weakest = sorted(keep, key=_rank_key)[-1]
            if (n.get("elo") is not None and weakest.get("elo") is not None
                    and n["elo"] > weakest["elo"] + HYST_ELO):
                keep.remove(weakest)
                keep.append(n)
                overflow.append(weakest)
            else:
                overflow.append(n)
        kept.extend(keep)
        carry = overflow
    return kept, len(carry)


def classify(snapshot_path=None, or_cache_path=None, prev_tiers=None):
    """Main classification pipeline. Returns the full frontier dict."""
    providers = load_snapshot(snapshot_path)
    or_models = load_openrouter(or_cache_path)
    canonical = deduplicate(or_models)
    overrides = load_overrides()

    # Build exclude set
    exclude_set = set(overrides.get("exclude", []))
    include_set = set(overrides.get("include", []))
    tier_overrides = overrides.get("tier_override", {})

    if prev_tiers is None:
        prev_tiers = load_previous_tiers()

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

        pinned = or_id in tier_overrides
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
            "pinned": pinned,
        })

    # Add forced includes not already present (config order, sorted for
    # byte-stable output — set iteration order varies with PYTHONHASHSEED).
    # An include whose alias-normalized base matches an auto-classified
    # model is redundant (e.g. zai/glm-5.3 once z-ai/glm-5.3 grew
    # benchmarks) — skip it instead of listing the model twice.
    existing_ids = {m["or_id"] for m in models}
    existing_bases = {_dedupe_base(m["or_id"]) for m in models}
    for inc in sorted(include_set):
        if inc in existing_ids:
            continue
        if _dedupe_base(inc) in existing_bases:
            print(f"  SKIP include {inc}: already classified under an aliased provider")
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
            "pinned": True,
        })

    # Hysteresis (incumbents held on wobbles) then deterministic capped
    # fill with cascade-down overflow. Forced includes bypass both, as before.
    auto = [m for m in models if m["or_id"] not in include_set]
    forced = sorted((m for m in models if m["or_id"] in include_set),
                    key=lambda m: m["or_id"])
    holds = apply_hysteresis(auto, prev_tiers)
    capped, dropped = apply_caps(auto, prev_tiers)
    # Stable display order: tier, then ELO desc.
    capped.sort(key=lambda m: (TIER_ORDER.index(m["tier"]), _rank_key(m)))
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
            "hysteresis_holds": holds,
            "cap_dropped": dropped,
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


def archive_live_snapshot(today=None):
    """Fetch OpenRouter live (retry + validate), archive the raw response
    to data/or-cache/<date>.json for replay/audit, and return its path."""
    data = _fetch_json(OPENROUTER_URL)
    models = {m["id"]: m for m in data.get("data", [])
                if isinstance(m, dict) and "id" in m}
    _validate_or_models(models)
    day = today or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    os.makedirs(OR_CACHE_DIR, exist_ok=True)
    path = os.path.join(OR_CACHE_DIR, f"{day}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"data": data.get("data", []),
                   "_archived": datetime.now(timezone.utc).isoformat()},
                  f, indent=1, sort_keys=True)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Classify frontier models from data")
    ap.add_argument("--snapshot", help="Path to models.dev snapshot .json.gz")
    ap.add_argument("--or-cache", help="Path to OpenRouter models cache .json "
                    "(replay a previous fetch instead of hitting the live API)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print classification to stdout, don't write frontier.json")
    args = ap.parse_args(argv)

    if args.or_cache:
        or_cache_path = args.or_cache
    elif args.dry_run:
        # Dry runs never touch the network archive or the working tree;
        # classify straight from the live API.
        or_cache_path = None
    else:
        try:
            or_cache_path = archive_live_snapshot()
        except RuntimeError as e:
            print(f"ERROR: {e}")
            return 2
        print(f"ARCHIVED: OpenRouter response → {or_cache_path}")

    try:
        result = classify(args.snapshot, or_cache_path)
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return 2

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

    # Write output (+ dated history copy for audit; the cron job commits both)
    payload = json.dumps(result, indent=2, sort_keys=False) + "\n"
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(payload)
    os.makedirs(HISTORY_DIR, exist_ok=True)
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with open(os.path.join(HISTORY_DIR, f"{day}.json"), "w",
              encoding="utf-8") as f:
        f.write(payload)
    meta = result["_meta"]
    print(f"CLASSIFIED: {meta['total']} models → {OUTPUT_FILE}")
    for t in ("frontier", "efficient", "workhorse"):
        print(f"  {t}: {meta['by_tier'][t]}")
    print(f"  benchmark categories: {meta['benchmark_categories']}")
    print(f"  hysteresis holds: {meta['hysteresis_holds']}, "
          f"cap dropped: {meta['cap_dropped']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
