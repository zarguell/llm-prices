#!/usr/bin/env python3
"""Canonical {model_key: release_date} fingerprint of models.dev — stdout only.

Zero writes by design: the cronman llm-prices job diffs this against state to
decide whether a full snapshot/classify/enrich/publish run is warranted. Keys
are "provider/model" (same convention as latest.json's `key` field).

Usage: python3 fingerprint.py [--url URL]   (prints one canonical JSON object)
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from snapshot import SOURCE_URL, fetch  # noqa: E402


def build_fingerprint(providers: dict) -> dict:
    """Flatten the models.dev payload to {provider/model: release_date}."""
    out = {}
    for provider_id, pdata in (providers or {}).items():
        models = (pdata or {}).get("models") or {}
        for model_id, m in models.items():
            if isinstance(m, dict):
                out[f"{provider_id}/{model_id}"] = m.get("release_date") or ""
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Print models.dev release fingerprint")
    ap.add_argument("--url", default=SOURCE_URL)
    args = ap.parse_args(argv)
    print(json.dumps(build_fingerprint(fetch(args.url)), sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
