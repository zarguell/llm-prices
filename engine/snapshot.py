#!/usr/bin/env python3
"""LLM Prices — weekly models.dev snapshots (the repo's data store).

Fetches https://models.dev/api.json (free, no auth) and writes one immutable
gzip bundle per day: data/snapshots/YYYY-MM-DD.json.gz. The snapshot archive
is the whole point of this repo: it accumulates price history for the trend
charts AND doubles as the local pricing cache for cronman's pi-usage report
(which reads the newest snapshot instead of hitting models.dev itself).

Determinism: the embedded timestamp is date-level ("fetched":
"YYYY-MM-DDT00:00:00Z"), so a same-day re-snapshot is idempotent and byte-
stable except for upstream's own payload changes. A given date is never
split into two files.

Usage: python3 snapshot.py [--from-file FILE] [--url URL] [--data-dir DIR]
"""
import argparse
import glob
import gzip
import json
import os
import urllib.request
from datetime import datetime, timezone

SOURCE_URL = "https://models.dev/api.json"
USER_AGENT = "llm-prices-snapshot/1.0 (weekly price archive; github.com/zarguell/llm-prices)"

ENGINE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(ENGINE), "data", "snapshots")


def fetch(url: str = SOURCE_URL) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.loads(r.read().decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("unexpected models.dev payload (not a dict)")
    return data


def snapshot_path(data_dir: str, date_str: str) -> str:
    return os.path.join(data_dir, f"{date_str}.json.gz")


def write_snapshot(data_dir: str, providers: dict, now: datetime | None = None,
                   source: str = SOURCE_URL) -> tuple[str, str]:
    """Write today's snapshot; return (path, status) with status in
    {"written", "replaced"}. Date-level timestamp keeps rebuilds stable."""
    now = now or datetime.now(timezone.utc)
    date_str = now.astimezone(timezone.utc).strftime("%Y-%m-%d")
    os.makedirs(data_dir, exist_ok=True)
    path = snapshot_path(data_dir, date_str)
    status = "replaced" if os.path.exists(path) else "written"
    entry = {
        "fetched": date_str + "T00:00:00Z",
        "source": source,
        "providers": providers,
    }
    tmp = path + ".tmp"
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(entry, f)
    os.replace(tmp, path)
    return path, status


def load_snapshot(path: str) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def newest_snapshot(data_dir: str) -> str | None:
    files = sorted(glob.glob(os.path.join(data_dir, "????-??-??.json.gz")))
    return files[-1] if files else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Snapshot models.dev pricing")
    ap.add_argument("--from-file", help="seed from a local api.json instead of fetching")
    ap.add_argument("--url", default=SOURCE_URL)
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    args = ap.parse_args(argv)

    if args.from_file:
        with open(args.from_file, "r", encoding="utf-8") as f:
            providers = json.load(f)
        source = "file:" + args.from_file
    else:
        providers = fetch(args.url)
        source = args.url

    path, status = write_snapshot(args.data_dir, providers, source=source)
    priced = sum(
        1
        for pdata in providers.values()
        for m in (pdata.get("models") or {}).values()
        if isinstance(m, dict) and isinstance(m.get("cost"), dict)
        and "input" in m["cost"] and "output" in m["cost"]
    )
    total = sum(len(p.get("models") or {}) for p in providers.values() if isinstance(p, dict))
    print(f"SNAPSHOT {status}: {path} — {total} models ({priced} priced)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
