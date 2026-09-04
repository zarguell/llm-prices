#!/usr/bin/env python3
"""LLM Prices — enrich the curated frontier list with OpenRouter benchmarks
and HuggingFace parameter counts.

Reads engine/frontier.json, fetches OpenRouter's full model list (one call,
~427 models, free, no auth) and HuggingFace's per-model API (one call per
open-weight model with an hf_id, also free/no auth), caches everything to
data/frontier-cache.json with a 7-day TTL.

Usage: python3 engine/enrich.py [--data-dir DIR] [--dry-run]

The cache is consumed by ssg.py at build time — no network at build.
"""
import argparse
import json
import os
import urllib.request
from datetime import datetime, timezone

ENGINE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(ENGINE)
DEFAULT_DATA_DIR = os.path.join(ROOT, "data")
FRONTIER_FILE = os.path.join(ENGINE, "frontier.json")
CACHE_FILE = "frontier-cache.json"
TTL_HOURS = 168  # 7 days
OPENROUTER_URL = "https://openrouter.ai/api/v1/models"
HF_URL = "https://huggingface.co/api/models/{hf_id}"
HF_USER_AGENT = "llm-prices-enrich/1.0 (weekly model metadata; github.com/zarguell/llm-prices)"


def _now():
    return datetime.now(timezone.utc)


def _fetch_json(url, user_agent=None):
    req = urllib.request.Request(url)
    if user_agent:
        req.add_header("User-Agent", user_agent)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def _cache_path(data_dir):
    return os.path.join(data_dir, CACHE_FILE)


def _read_cache(data_dir):
    try:
        with open(_cache_path(data_dir), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _write_cache(data_dir, cache):
    os.makedirs(data_dir, exist_ok=True)
    path = _cache_path(data_dir)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=1, sort_keys=True)
    os.replace(tmp, path)


def fetch_openrouter_index():
    """Fetch the full OpenRouter model list (one call). Returns {or_id: model_dict}."""
    data = _fetch_json(OPENROUTER_URL)
    return {m["id"]: m for m in data.get("data", []) if isinstance(m, dict) and "id" in m}


def fetch_hf_model(hf_id):
    """Fetch a single HuggingFace model's metadata. Returns dict or None."""
    try:
        return _fetch_json(HF_URL.format(hf_id=hf_id), user_agent=HF_USER_AGENT)
    except Exception:
        return None


def extract_benchmarks(or_model):
    """Pull a flattened benchmark summary from OpenRouter's nested structure.

    Returns: {category_path: {elo, rank, win_rate}} or {}.
    Category path = "arena/category" (e.g., "agents/agenticgamedev").
    """
    raw = or_model.get("benchmarks")
    if not isinstance(raw, dict):
        return {}
    out = {}
    for arena_name, entries in raw.items():
        if not isinstance(entries, list):
            continue
        for e in entries:
            if not isinstance(e, dict) or "elo" not in e:
                continue
            path = f"{arena_name}/{e.get('category', '?')}"
            out[path] = {
                "elo": e.get("elo"),
                "rank": e.get("rank"),
                "win_rate": e.get("win_rate"),
            }
    return out


def extract_params_b(hf_model):
    """Extract parameter count in billions from a HuggingFace model API response."""
    st = hf_model.get("safetensors") or {}
    params = st.get("parameters") or {}
    total = params.get("total")
    if total and total > 0:
        return round(total / 1e9, 2)
    return None


def enrich(frontier, or_index, hf_cache):
    """Merge frontier list with OpenRouter + HuggingFace data.

    Returns: list of enriched model dicts (one per frontier entry).
    """
    enriched = []
    for entry in frontier["models"]:
        key = entry["key"]
        or_id = entry.get("or_id")
        hf_id = entry.get("hf_id")
        tier = entry.get("tier", "workhorse")

        # models.dev: (provider, model_id) from key
        parts = key.split("/", 1)
        provider = parts[0] if len(parts) == 2 else "unknown"
        model_id = parts[1] if len(parts) == 2 else parts[0]

        # OpenRouter data
        or_model = or_index.get(or_id, {}) if or_id else {}
        benchmarks = extract_benchmarks(or_model)
        or_arch = or_model.get("architecture") or {}

        # HuggingFace data (use cache)
        hf_data = hf_cache.get(hf_id, {}) if hf_id else {}
        params_b = entry.get("params_b")  # hand-specified fallback
        if params_b is None and hf_data:
            params_b = extract_params_b(hf_data)

        enriched.append({
            "key": key,
            "provider": provider,
            "model_id": model_id,
            "tier": tier,
            "or_id": or_id,
            "hf_id": hf_id,
            "params_b": params_b,
            "context": or_model.get("context_length"),
            "knowledge_cutoff": or_model.get("knowledge_cutoff"),
            "benchmarks": benchmarks,
            "benchmark_count": len(benchmarks),
            "best_elo": max((b["elo"] for b in benchmarks.values() if b.get("elo")), default=None),
            "arch_modality": or_arch.get("modality"),
        })
    return enriched


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Enrich frontier list with OpenRouter benchmarks + HF params")
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    now = _now()
    cache = _read_cache(args.data_dir) or {}
    cache_fresh = False
    if cache.get("fetched"):
        try:
            age = (now - datetime.fromisoformat(cache["fetched"])).total_seconds()
            cache_fresh = age < TTL_HOURS * 3600
        except Exception:
            pass

    if cache_fresh:
        print(f"ENRICH: cache fresh (fetched {cache['fetched']}) — skipping")
        return 0

    with open(FRONTIER_FILE, "r", encoding="utf-8") as f:
        frontier = json.load(f)

    if args.dry_run:
        print(f"DRY RUN: would fetch OpenRouter ({len(frontier['models'])} models to enrich)")
        for e in frontier["models"]:
            print(f"  {e['key']:45s} tier={e['tier']}")
        return 0

    print("ENRICH: fetching OpenRouter model list...")
    or_index = fetch_openrouter_index()
    print(f"  OpenRouter: {len(or_index)} models loaded")

    hf_ids = [e.get("hf_id") for e in frontier["models"] if e.get("hf_id")]
    hf_cache = cache.get("hf", {})
    for hf_id in hf_ids:
        if hf_id in hf_cache:
            continue
        print(f"  HuggingFace: {hf_id}")
        data = fetch_hf_model(hf_id)
        if data:
            hf_cache[hf_id] = {
                "parameters_b": extract_params_b(data),
                "architecture": (data.get("config") or {}).get("architectures", []),
                "license": (data.get("cardData") or {}).get("license"),
            }
        else:
            hf_cache[hf_id] = {"parameters_b": None, "architecture": [], "license": None}

    enriched = enrich(frontier, or_index, hf_cache)
    output = {
        "fetched": now.isoformat(),
        "frontier": frontier,
        "enriched": enriched,
        "hf": hf_cache,
        "or_model_count": len(or_index),
    }
    _write_cache(args.data_dir, output)

    print(f"ENRICH: done — {len(enriched)} models enriched, "
          f"{sum(e['benchmark_count'] for e in enriched)} benchmark categories total")
    best = sorted(enriched, key=lambda e: e.get("best_elo") or 0, reverse=True)[:5]
    for e in best:
        elo = e["best_elo"]
        print(f"  {e['key']:45s}  ELO={elo if elo else '?':>5}  params={e.get('params_b') or '?'}B")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
