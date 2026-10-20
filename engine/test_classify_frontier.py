#!/usr/bin/env python3
"""Unit tests for the frontier stability machinery in classify_frontier.py.

Hysteresis (incumbents held unless ELO clears HYST_ELO past the tier edge),
deterministic caps (total order, cascade-down overflow, margin displacement),
provider-alias dedupe, and byte-stable output. All synthetic — no network.

Run: python3 engine/test_classify_frontier.py   (also pytest-collectable)
"""
import gzip
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import classify_frontier as cf  # noqa: E402


def _or_model(elo, n_cats=3):
    cats = [{"elo": elo - i * 2, "category": f"c{i}"} for i in range(n_cats)]
    return {"benchmarks": {"arena": cats}}


def _row(or_id, tier, elo, price_in=None, pinned=False):
    return {"key": or_id, "or_id": or_id, "hf_id": None, "tier": tier,
            "elo": elo, "n_cats": 3, "price_in": price_in,
            "released": None, "params_b": None, "pinned": pinned}


def test_dedupe_alias_merges_providers():
    models = {
        "z-ai/glm-5.3": _or_model(1334, 11),
        "zai/glm-5.3": _or_model(1310, 2),
    }
    canon = cf.deduplicate(models)
    assert len(canon) == 1, canon.keys()
    # more benchmark categories wins regardless of input order
    assert list(canon.values())[0]["or_id"] == "z-ai/glm-5.3"


def test_dedupe_tiebreak_deterministic():
    # identical (n_cats, elo) on the same base model: lexicographically
    # smaller or_id always wins, whichever order the API returned them in
    for order in (["x/model:thinking", "x/model"], ["x/model", "x/model:thinking"]):
        models = {oid: _or_model(1300, 5) for oid in order}
        canon = cf.deduplicate(models)
        assert len(canon) == 1
        assert list(canon.values())[0]["or_id"] == "x/model", order


def test_dedupe_skips_batch_free_and_benchmarkless():
    models = {
        "x/m:batch": _or_model(1500, 9),
        "x/m:free": _or_model(1500, 9),
        "x/plain": {"name": "no benchmarks"},
        "x/real": _or_model(1200, 1),
    }
    canon = cf.deduplicate(models)
    assert set(v["or_id"] for v in canon.values()) == {"x/real"}


def test_cross_reference_uses_alias():
    providers = {"z-ai": {"models": {"glm-5.3": {
        "cost": {"input": 1.4, "output": 4.4},
        "tool_call": True, "reasoning": True}}}}
    dev = cf.cross_reference({"or_id": "zai/glm-5.3"}, providers)
    assert dev["price_in"] == 1.4


def test_hysteresis_holds_wobble():
    # frontier incumbent at 1345: edge 1350, 1345 >= 1350-15 → stays frontier
    rows = [_row("a/m", "efficient", 1345)]
    holds = cf.apply_hysteresis(rows, {"a/m": "frontier"})
    assert holds == 1 and rows[0]["tier"] == "frontier"


def test_hysteresis_releases_real_drop():
    # 1330 < 1350-15 → demotion stands
    rows = [_row("a/m", "efficient", 1330)]
    holds = cf.apply_hysteresis(rows, {"a/m": "frontier"})
    assert holds == 0 and rows[0]["tier"] == "efficient"


def test_hysteresis_price_path_edge():
    # frontier via price path (price >= 2, elo >= 1300): edge is 1300
    rows = [_row("a/m", "efficient", 1290, price_in=3.0)]
    holds = cf.apply_hysteresis(rows, {"a/m": "frontier"})
    assert holds == 1 and rows[0]["tier"] == "frontier"
    rows = [_row("a/m", "efficient", 1280, price_in=3.0)]
    holds = cf.apply_hysteresis(rows, {"a/m": "frontier"})
    assert holds == 0 and rows[0]["tier"] == "efficient"


def test_hysteresis_promotion_immediate_and_boundaries():
    # promotion needs no margin
    rows = [_row("a/m", "frontier", 1351)]
    assert cf.apply_hysteresis(rows, {"a/m": "efficient"}) == 0
    assert rows[0]["tier"] == "frontier"
    # elo-less and pinned rows untouched
    rows = [_row("a/m", "efficient", None), _row("b/m", "efficient", 1000)]
    rows[1]["pinned"] = True
    assert cf.apply_hysteresis(rows, {"a/m": "frontier", "b/m": "frontier"}) == 0
    assert rows[0]["tier"] == "efficient" and rows[1]["tier"] == "efficient"
    # unknown tiers / newcomers pass through
    rows = [_row("a/m", "efficient", 1345)]
    assert cf.apply_hysteresis(rows, {}) == 0


def test_caps_cascade_instead_of_drop():
    # 9 frontier-eligible, cap 8: weakest cascades to efficient, not out
    rows = [_row(f"lab/m{i}", "frontier", 1400 - i * 10) for i in range(9)]
    kept, dropped = cf.apply_caps(rows, {})
    assert dropped == 0
    tiers = {}
    for r in kept:
        tiers.setdefault(r["tier"], []).append(r["or_id"])
    assert len(tiers["frontier"]) == 8
    assert tiers["efficient"] == ["lab/m8"]
    assert len(kept) == 9


def test_caps_incumbent_displacement_needs_margin():
    # 11 efficient incumbents (1320..1330), cap 12: one room for newcomers.
    prev = {f"a/m{i}": "efficient" for i in range(11)}
    incumb = [_row(f"a/m{i}", "efficient", 1330 - i) for i in range(11)]
    # Y (1322, higher) takes the free slot; X (1321) is within margin of
    # the weakest kept (1320) so it must NOT displace — it cascades down.
    rows = incumb + [_row("n/x", "efficient", 1321),
                      _row("n/y", "efficient", 1322)]
    kept, dropped = cf.apply_caps(rows, prev)
    tiers = {r["or_id"]: r["tier"] for r in kept}
    assert tiers["n/y"] == "efficient"  # higher newcomer takes free slot
    assert tiers["n/x"] == "workhorse", tiers["n/x"]  # within margin → down
    assert all(tiers[f"a/m{i}"] == "efficient" for i in range(11))
    assert dropped == 0
    # With a full house of 12 incumbents, Z at 1340 beats the weakest
    # (1320) by 20 > 15 → displaces it; the displaced cascades down.
    prev12 = {f"a/m{i}": "efficient" for i in range(12)}
    incumb12 = [_row(f"a/m{i}", "efficient", 1331 - i) for i in range(12)]
    kept, dropped = cf.apply_caps(incumb12 + [_row("n/z", "efficient", 1340)],
                                   prev12)
    tiers = {r["or_id"]: r["tier"] for r in kept}
    assert tiers["n/z"] == "efficient"
    assert tiers["a/m11"] == "workhorse", tiers["a/m11"]  # 1320 drops
    assert dropped == 0


def test_caps_tiebreak_deterministic_under_shuffle():
    import random
    # 11 models above the tie + the 2026-10-20 pair (both 1310) + nothing
    # below: the pair straddles the efficient cap boundary (rank 12/13).
    # Deterministic order must keep the same one every time — no coin flip.
    base = ([_row("meta/muse-spark-open", "efficient", 1310)] +
            [_row("anthropic/claude-opus-4.8", "efficient", 1310)] +
            [_row(f"n/m{i}", "efficient", 1361 - i * 5) for i in range(11)])
    first = None
    for seed in range(5):
        shuffled = list(base)
        random.Random(seed).shuffle(shuffled)
        kept, _ = cf.apply_caps(shuffled, {})
        tiers = {r["or_id"]: r["tier"] for r in kept}
        first = first or tiers
        assert tiers == first, f"order-dependent result with seed {seed}"
    assert first["anthropic/claude-opus-4.8"] == "efficient"
    assert first["meta/muse-spark-open"] == "workhorse"


def test_forced_include_skipped_when_aliased_model_present():
    # zai/glm-5.3 forced while z-ai/glm-5.3 auto-classified → listed once.
    # gpt-5.6 (no benchmarks anywhere) must still be force-included.
    providers = {"z-ai": {"models": {
        "glm-5.3": {"cost": {"input": 1.4, "output": 4.4},
                      "tool_call": True, "reasoning": True}}}}
    or_models = [_or_entry("z-ai/glm-5.3", 1334)]
    with tempfile.TemporaryDirectory() as tmp:
        snap = _write_snapshot(tmp, providers)
        cache = _write_or_cache(tmp, or_models)
        old_min, old_bench = cf.MIN_OR_MODELS, cf.MIN_BENCHMARK_MODELS
        cf.MIN_OR_MODELS, cf.MIN_BENCHMARK_MODELS = 0, 0
        try:
            ids = [m["or_id"] for m in cf.classify(snap, cache, {})["models"]]
        finally:
            cf.MIN_OR_MODELS, cf.MIN_BENCHMARK_MODELS = old_min, old_bench
    assert "z-ai/glm-5.3" in ids
    assert "zai/glm-5.3" not in ids
    assert "openai/gpt-5.6" in ids  # genuinely missing → backstop kept


def test_load_previous_tiers_missing():
    assert cf.load_previous_tiers("/nonexistent/path.json") == {}


def _write_snapshot(tmp, providers):
    path = os.path.join(tmp, "2026-10-20.json.gz")
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump({"providers": providers}, f)
    return path


def _write_or_cache(tmp, models):
    path = os.path.join(tmp, "or.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"data": models}, f)
    return path


def _or_entry(oid, elo, price_providers=None):
    return {"id": oid,
            "benchmarks": {"arena": [{"elo": elo, "category": "gen"}]}}


def test_classify_end_to_end_deterministic():
    providers = {"lab": {"models": {
        "m1": {"cost": {"input": 5.0, "output": 10.0},
               "tool_call": True, "reasoning": True},
        "m2": {"cost": {"input": 0.2, "output": 0.5},
               "tool_call": True, "reasoning": True},
        "m3": {"cost": {"input": 0.04, "output": 0.10},
               "tool_call": False, "reasoning": False}}}}
    or_models = [_or_entry("lab/m1", 1400), _or_entry("lab/m2", 1250),
                 _or_entry("lab/m3", 1000)]
    with tempfile.TemporaryDirectory() as tmp:
        snap = _write_snapshot(tmp, providers)
        cache = _write_or_cache(tmp, or_models)
        # tiny synthetic fixture: relax the live-fetch minimums (scoped)
        old_min, old_bench = cf.MIN_OR_MODELS, cf.MIN_BENCHMARK_MODELS
        cf.MIN_OR_MODELS, cf.MIN_BENCHMARK_MODELS = 0, 0
        try:
            first = cf.classify(snap, cache, {})["models"]
            # reversed input order must give identical output
            cache2 = _write_or_cache(tmp, list(reversed(or_models)))
            second = cf.classify(snap, cache2, {})["models"]
        finally:
            cf.MIN_OR_MODELS, cf.MIN_BENCHMARK_MODELS = old_min, old_bench
        assert first == second
        tiers = {m["or_id"]: m["tier"] for m in first}
        # forced includes from frontier-overrides.json join in (elo-less);
        # the auto-classified lab models must land exactly here
        assert tiers["lab/m1"] == "frontier"
        assert tiers["lab/m2"] == "efficient"
        assert tiers["lab/m3"] == "workhorse", tiers["lab/m3"]
        # output schema: exactly the ssg/enrich contract fields
        assert set(first[0].keys()) == {"key", "or_id", "hf_id", "tier",
                                        "params_b", "elo", "n_cats"}


def test_validation_rejects_partial_response():
    with tempfile.TemporaryDirectory() as tmp:
        cache = _write_or_cache(tmp, [_or_entry("a/m", 1300)])
        try:
            cf.classify(None, cache, {})
        except RuntimeError as e:
            assert "partial" in str(e)
        else:
            raise AssertionError("expected RuntimeError for partial response")


def run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok {fn.__name__}")
    print(f"{len(fns)} tests passed")


if __name__ == "__main__":
    run_all()
