#!/usr/bin/env python3
"""Contract tests for the LLM Prices snapshot store.

Run: python3 engine/test_snapshot.py   (standalone; also pytest-collectable)
"""
import glob
import gzip
import json
import os
import sys
import tempfile
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import snapshot  # noqa: E402

PROVIDERS = {
    "prov-a": {"models": {
        "m1": {"name": "Model One", "cost": {"input": 1.0, "output": 2.0}},
        "m2": {"name": "Model Two"},
    }},
    "prov-b": {"models": {
        "m3": {"cost": {"input": 0.5, "output": 1.5}},
    }},
}


def test_write_is_deterministic_per_day():
    with tempfile.TemporaryDirectory() as d:
        now = datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc)
        p1, st1 = snapshot.write_snapshot(d, PROVIDERS, now=now)
        assert st1 == "written"
        entry = snapshot.load_snapshot(p1)
        assert entry["fetched"] == "2026-09-04T00:00:00Z"  # date-level
        assert entry["providers"] == PROVIDERS
        # same-day rewrite: replaced in place, same path, still valid
        p2, st2 = snapshot.write_snapshot(d, PROVIDERS, now=now)
        assert st2 == "replaced" and p1 == p2
        assert snapshot.load_snapshot(p2)["providers"] == PROVIDERS


def test_newest_snapshot_ordering():
    with tempfile.TemporaryDirectory() as d:
        for day in ("2026-08-28", "2026-09-04"):
            now = datetime.fromisoformat(day + "T12:00:00+00:00")
            snapshot.write_snapshot(d, PROVIDERS, now=now)
        files = sorted(glob.glob(os.path.join(d, "*.json.gz")))
        assert len(files) == 2
        assert snapshot.newest_snapshot(d).endswith("2026-09-04.json.gz")


def test_load_roundtrip_gzip():
    with tempfile.TemporaryDirectory() as d:
        p, _ = snapshot.write_snapshot(d, PROVIDERS)
        assert p.endswith(".json.gz")
        with gzip.open(p, "rt", encoding="utf-8") as f:
            raw = json.load(f)
        assert raw["source"] == snapshot.SOURCE_URL
        assert snapshot.load_snapshot(p) == raw


def test_main_from_file():
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "api.json")
        with open(src, "w") as f:
            json.dump(PROVIDERS, f)
        out = os.path.join(d, "snaps")
        rc = snapshot.main(["--from-file", src, "--data-dir", out])
        assert rc == 0
        assert snapshot.newest_snapshot(out) is not None


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  ok {name}")
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"  FAIL {name}: {e}")
    raise SystemExit(1 if failed else 0)
