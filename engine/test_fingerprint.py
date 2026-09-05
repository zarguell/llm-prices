#!/usr/bin/env python3
"""Contract tests for the models.dev fingerprint (daily-gate input).

Run: python3 engine/test_fingerprint.py   (standalone; also pytest-collectable)
"""
import io
import json
import os
import sys
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fingerprint  # noqa: E402

PROVIDERS = {
    "openai": {"models": {
        "gpt-5.6": {"name": "GPT-5.6", "release_date": "2026-07-09"},
        "gpt-4o": {"name": "GPT-4o"},
    }},
    "anthropic": {"models": {
        "claude-sonnet-5": {"name": "Claude Sonnet 5", "release_date": "2026-06-29"},
    }},
    "empty": {},
}


def test_build_fingerprint_keys_and_missing_dates():
    fp = fingerprint.build_fingerprint(PROVIDERS)
    assert fp == {"openai/gpt-5.6": "2026-07-09",
                  "openai/gpt-4o": "",
                  "anthropic/claude-sonnet-5": "2026-06-29"}


def test_key_convention_matches_latest_json():
    fp = fingerprint.build_fingerprint(PROVIDERS)
    assert "openai/gpt-5.6" in fp  # same "provider/model" convention as latest.json keys


def test_main_prints_canonical_sorted_json(monkeypatch=None):
    import fingerprint as fp_mod
    fp_mod.fetch = lambda url="": PROVIDERS  # noqa: ARG001
    buf = io.StringIO()
    with redirect_stdout(buf):
        assert fp_mod.main([]) == 0
    parsed = json.loads(buf.getvalue())
    assert parsed["anthropic/claude-sonnet-5"] == "2026-06-29"
    assert list(parsed) == sorted(parsed)  # canonical: stable bytes for hashing


def main():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
        except Exception as e:  # noqa: BLE001 — report, don't stop
            print(f"FAIL {fn.__name__}: {e}")
            failed += 1
        else:
            print(f"ok {fn.__name__}")
    print(f"{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
