#!/usr/bin/env python3
"""Contract tests for the LLM Prices static site generator.

Run: python3 engine/test_ssg.py   (standalone; also pytest-collectable)
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ssg  # noqa: E402


def _providers():
    return {
        "zai": {"name": "Z.ai", "models": {
            "glm-flash": {"name": "GLM Flash", "tool_call": True,
                          "limit": {"context": 200000, "output": 32000},
                          "release_date": "2026-01-01",
                          "cost": {"input": 0.1, "output": 0.2,
                                   "cache_read": 0.01}},
            "glm-big": {"name": "GLM Big",
                        "cost": {"input": 5.0, "output": 10.0}},
        }},
        "ogo": {"name": "OGO", "models": {
            "mimo": {"name": "MiMo", "tool_call": True, "open_weights": True,
                     "limit": {"context": 1000000},
                     "cost": {"input": 0.14, "output": 0.28}},
            "free-tier": {"name": "Free Tier",
                          "cost": {"input": 0.0, "output": 0.0}},
            "half-free": {"name": "Half Free",
                          "cost": {"input": 0.0, "output": 9.9}},
        }},
    }


def _providers_next_week():
    """Week 2: glm-flash input drops, glm-big removed, new model arrives."""
    p = _providers()
    p["zai"]["models"]["glm-flash"]["cost"]["input"] = 0.05
    p["zai"]["models"]["glm-new"] = {"name": "GLM New",
                                     "cost": {"input": 0.2, "output": 0.4}}
    del p["zai"]["models"]["glm-big"]
    return p


def _write_snaps(data_dir):
    for day, prov in (("2026-08-28", _providers()),
                      ("2026-09-04", _providers_next_week())):
        ssg.os.makedirs(data_dir, exist_ok=True)
        import gzip
        with gzip.open(os.path.join(data_dir, f"{day}.json.gz"), "wt") as f:
            json.dump({"fetched": day + "T00:00:00Z", "providers": prov}, f)


def test_flatten_extracts_pricing_and_capabilities():
    rows = ssg.flatten(_providers())
    assert len(rows) == 5
    by_key = ssg.by_key(rows)
    glm = by_key["zai/glm-flash"]
    assert glm["priced"] and glm["input"] == 0.1 and glm["output"] == 0.2
    assert glm["tool_call"] and glm["context"] == 200000
    assert glm["provider_name"] == "Z.ai"
    # models without full cost data are unpriced
    assert by_key["zai/glm-big"]["priced"] is True
    assert "nope/none" not in by_key


def test_diff_detects_adds_removes_and_moves():
    cur = ssg.flatten(_providers_next_week())
    prev = ssg.flatten(_providers())
    d = ssg.diff(cur, prev)
    assert d["added"] == ["zai/glm-new"]
    assert d["removed"] == ["zai/glm-big"]
    moves = {(c["key"], c["field"]): c for c in d["changed"]}
    assert ("zai/glm-flash", "input") in moves
    mv = moves[("zai/glm-flash", "input")]
    assert mv["old"] == 0.1 and mv["new"] == 0.05
    assert mv["pct"] == -50.0


def test_stats_coverage_best_value_and_medians():
    stats = ssg.compute_stats(ssg.flatten(_providers()))
    assert stats["providers_n"] == 2 and stats["models_n"] == 5
    assert stats["priced_n"] == 5 and stats["coverage"] == 100.0
    # zero-priced models never rank: free-tier (0/0) and half-free (0 in)
    assert stats["zero_priced_n"] == 2
    assert stats["cheapest"][0]["key"] == "zai/glm-flash"
    assert all(r["input"] > 0 for r in stats["cheapest"])
    assert all(r["input"] > 0 for r in stats["best_value"])
    assert stats["priciest"][0]["key"] == "zai/glm-big"
    # best value: tool_call + >=128k context -> glm-flash and mimo, not glm-big
    keys = {r["key"] for r in stats["best_value"]}
    assert keys == {"zai/glm-flash", "ogo/mimo"}
    assert stats["providers"][0]["coverage"] == 100.0  # ogo first (sorted)
    # medians exclude zero rates: ogo [0.14] -> 0.14; zai [0.1, 5.0] -> 2.55
    ogo = next(p for p in stats["providers"] if p["provider"] == "ogo")
    zai = next(p for p in stats["providers"] if p["provider"] == "zai")
    assert ogo["median_input"] == 0.14
    assert zai["median_input"] == 2.55


def test_svg_chart_is_deterministic_svg():
    dates = ["2026-08-28", "2026-09-04"]
    series = [{"label": "zai/glm-flash", "color": "#D97757",
               "points": [0.1, 0.05]}]
    a = ssg.svg_line_chart(dates, series)
    b = ssg.svg_line_chart(dates, series)
    assert a == b and a.startswith("<svg") and a.endswith("</svg>")
    assert "<path" in a and "<circle" in a and "<title>" in a
    # missing points tolerated
    s2 = [{"label": "x", "color": "#fff", "points": [None, 0.05]}]
    assert ssg.svg_line_chart(dates, s2)


def test_build_renders_expected_site():
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = os.path.join(tmp, "data", "snapshots")
        _write_snaps(data_dir)
        out_root = os.path.join(tmp, "root")
        os.makedirs(out_root)
        manifest = ssg.build(root=out_root, data_dir=data_dir,
                             tracked=["zai/glm-flash", "ogo/mimo"])
        for rel in ("index.html", "style.css", "404.html", "robots.txt",
                    "sitemap.xml", "prices/index.html",
                    "prices/zai/index.html", "prices/ogo/index.html",
                    "providers/index.html", "trends/index.html",
                    "changes/index.html", "stats/index.html",
                    "about/index.html", "data/latest.json"):
            assert os.path.exists(os.path.join(out_root, rel)), f"missing {rel}"
            assert rel in manifest
        # index links cheapest model and shows week-over-week section
        idx = open(os.path.join(out_root, "index.html")).read()
        assert "GLM Flash" in idx and "changes/index.html" in idx
        # provider table has rates and sort hooks
        zai = open(os.path.join(out_root, "prices", "zai", "index.html")).read()
        assert "$0.0500" in zai and "sortTable" in zai
        # changes page lists the add/remove/price move
        ch = open(os.path.join(out_root, "changes", "index.html")).read()
        assert "zai/glm-new" in ch and "zai/glm-big" in ch and "-50.0%" in ch
        # trends page has inline SVG charts
        tr = open(os.path.join(out_root, "trends", "index.html")).read()
        assert "<svg" in tr
        # machine-readable dump matches snapshot
        latest = json.load(open(os.path.join(out_root, "data", "latest.json")))
        assert latest["as_of"] == "2026-09-04"
        keys = {m["key"] for m in latest["models"]}
        assert "zai/glm-new" in keys and "zai/glm-big" not in keys


def test_build_link_prefixes_match_depth():
    """The `r` prefix must be depth-correct: "" at root, ../ one level down,
    ../../ two levels down. (Regression: pages rendered with the template
    name as prefix -> href="index.htmlstyle.css" on Pages.)"""
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = os.path.join(tmp, "data", "snapshots")
        _write_snaps(data_dir)
        out_root = os.path.join(tmp, "root")
        os.makedirs(out_root)
        ssg.build(root=out_root, data_dir=data_dir,
                  tracked=["zai/glm-flash"])
        idx = open(os.path.join(out_root, "index.html")).read()
        assert 'href="style.css"' in idx
        assert 'href="prices/index.html"' in idx
        ch = open(os.path.join(out_root, "changes", "index.html")).read()
        assert 'href="../style.css"' in ch
        assert 'href="../prices/index.html"' in ch
        zai = open(os.path.join(out_root, "prices", "zai", "index.html")).read()
        assert 'href="../../style.css"' in zai
        assert "index.htmlstyle" not in idx + ch + zai


def test_build_single_snapshot_baseline():
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = os.path.join(tmp, "data", "snapshots")
        import gzip
        ssg.os.makedirs(data_dir, exist_ok=True)
        with gzip.open(os.path.join(data_dir, "2026-09-04.json.gz"), "wt") as f:
            json.dump({"fetched": "2026-09-04T00:00:00Z",
                       "providers": _providers()}, f)
        out_root = os.path.join(tmp, "root")
        os.makedirs(out_root)
        manifest = ssg.build(root=out_root, data_dir=data_dir,
                             tracked=["zai/glm-flash"])
        ch = open(os.path.join(out_root, "changes", "index.html")).read()
        assert "Baseline snapshot" in ch
        tr = open(os.path.join(out_root, "trends", "index.html")).read()
        assert "<svg" in tr  # single points still render


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
