from __future__ import annotations
import csv

import pytest

from brand_study import config
from brand_study import llm_aggregate as la
from brand_study import llm_judges as lj
from brand_study.labeling import load_llm_overrides
from brand_study.llm_evidence import sha12

J = config.LLM_JUDGE_ORDER


def rec(key, kt="domain", rule_categories=(), ev_extra=""):
    ev = {"key": key, "key_type": kt, "x": ev_extra}
    return {"item_id": f"{kt}:{key}", "key_type": kt, "key": key, "evidence": ev, "evidence_sha": sha12(ev),
            "rule_categories": list(rule_categories), "example_url": f"https://{key}/", "example_title": key,
            "sectors": "coffee", "surfaces": "organic"}


def vote(j, r, cat, conf=0.9, status="ok", sha=None):
    return {"judge": j, "model": config.LLM_JUDGES[j][1], "prompt_version": lj.PROMPT_VERSION,
            "item_id": r["item_id"], "evidence_sha": sha or r["evidence_sha"], "status": status,
            "category": cat if status == "ok" else None, "confidence": conf if status == "ok" else None,
            "reason": f"{j} reason", "error": "" if status == "ok" else "missing", "model_digest": f"sha256:{j}",
            "cli_version": "", "ts": "2026-11-10T00:00:00Z"}


def fill(store, r, cats):
    for j, c in zip(J, cats):
        if c is None:
            continue
        store.append(j, [vote(j, r, c) if c != "abstain" else vote(j, r, None, status="abstain")])


def votes_of(cats):
    return {j: (None if c is None else {"status": "abstain"} if c == "abstain" else {"status": "ok", "category": c})
            for j, c in zip(J, cats)}


@pytest.mark.parametrize("cats,outcome,accepted", [
    (("small_local",) * 3, "unanimous", "small_local"),
    (("small_local", "small_local", "big_brand"), "majority_2of3", None),
    (("small_local", "big_brand", "not_a_brand"), "no_majority", None),
    (("small_local", "small_local", "abstain"), "abstention", None),
    (("unknown",) * 3, "all_unknown", None),
    (("small_local", "small_local", None), "pending", None),
])
def test_outcome_table(cats, outcome, accepted):
    o, cat, src = la.classify(votes_of(cats), [], "unanimous")
    assert (o, cat) == (outcome, accepted)
    if accepted:
        assert src == "llm_unanimous"


def test_majority_policy():
    o, cat, src = la.classify(votes_of(("small_local", "small_local", "big_brand")), [], "majority")
    assert (o, cat, src) == ("majority_2of3", "small_local", "llm_majority")
    assert la.classify(votes_of(("unknown", "unknown", "big_brand")), [], "majority")[1] is None


def test_conflicts_rule():
    v = votes_of(("big_brand",) * 3)
    assert la.classify(v, ["big_brand"]) == ("unanimous", "big_brand", "llm_unanimous_confirm")
    assert la.classify(v, ["small_local"])[0] == "conflicts_rule"
    assert la.classify(v, ["big_brand", "small_local"])[0] == "conflicts_rule"


def _read(path):
    lines = [ln for ln in path.read_text().splitlines() if not ln.startswith("#")]
    return list(csv.DictReader(lines))


def test_aggregate_files(tmp_path, _isolated_llm):
    store = lj.VoteStore(tmp_path / "v")
    a, b, c, d = rec("a.com"), rec("b.com"), rec("illy", "entity", ["big_brand", "small_local"]), rec("solo", "entity",
                                                                                                    ["small_local"])
    fill(store, a, ("small_local",) * 3)
    fill(store, b, ("small_local", "big_brand", "big_brand"))
    fill(store, c, ("big_brand",) * 3)
    fill(store, d, ("small_local",) * 3)
    dis = tmp_path / "dis.csv"
    logs = []
    s = la.aggregate([a, b, c, d], store, disagreements_out=dis, runs=["run1"], log=logs.append)
    assert s["accepted"] == 2 and s["disagreements"] == 2 and s["pending"] == 0 and s["accepted_written"]
    acc = _isolated_llm.read_text()
    assert "GENERATED" in acc and "# runs=run1" in acc and f"# prompt_version={lj.PROMPT_VERSION}" in acc
    rows = _read(_isolated_llm)
    note = next(r["note"] for r in rows if r["key"] == "a.com")
    assert note == ("llm_unanimous; gemma4:31b=small_local(0.90); qwen3.8:27b=small_local(0.90); "
                    "claude-opus-5-5=small_local(0.90)")
    assert next(r["note"] for r in rows if r["key"] == "solo").startswith("llm_unanimous_confirm;")
    drows = {r["key"]: r for r in _read(dis)}
    assert "category" not in drows["b.com"] and drows["b.com"]["outcome"] == "majority_2of3"
    assert drows["b.com"]["majority_category"] == "big_brand" and drows["b.com"]["gemma_reason"] == "gemma reason"
    assert drows["illy"]["outcome"] == "conflicts_rule" and drows["illy"]["rule_categories"] == "big_brand|small_local"
    assert any("label_overrides.csv" in m for m in logs)
    # round trip through the loader, and idempotence
    ov, meta = load_llm_overrides()
    assert ov == {("domain", "a.com"): ("small_local", "llm_unanimous"), ("entity", "solo"): ("small_local",
                                                                                            "llm_unanimous")}
    assert meta["runs"] == "run1"
    before = _isolated_llm.read_bytes(), dis.read_bytes()
    la.aggregate([a, b, c, d], store, disagreements_out=dis, runs=["run1"], log=lambda m: None)
    assert (_isolated_llm.read_bytes(), dis.read_bytes()) == before


def test_incomplete_or_limited_keeps_accepted_file(tmp_path, _isolated_llm):
    store = lj.VoteStore(tmp_path / "v")
    a, b = rec("a.com"), rec("b.com")
    fill(store, a, ("small_local",) * 3)
    la.aggregate([a], store, disagreements_out=tmp_path / "d.csv", runs=["run1"], log=lambda m: None)
    original = _isolated_llm.read_bytes()
    fill(store, b, ("small_local", "small_local", None))
    logs = []
    s = la.aggregate([a, b], store, disagreements_out=tmp_path / "d.csv", runs=["run1"], log=logs.append)
    assert s["pending"] == 1 and not s["accepted_written"] and _isolated_llm.read_bytes() == original
    assert any("aggregation incomplete: 1 pending" in m for m in logs)
    s2 = la.aggregate([a], store, limited=True, disagreements_out=tmp_path / "d2.csv", log=lambda m: None)
    assert not s2["accepted_written"] and not (tmp_path / "d2.csv").exists()


def test_pending_after_evidence_change_and_stale_reuse(tmp_path, _isolated_llm):
    store = lj.VoteStore(tmp_path / "v")
    a = rec("a.com")
    fill(store, a, ("small_local",) * 3)
    a2 = rec("a.com", ev_extra="new snippet")
    logs = []
    s = la.aggregate([a2], store, disagreements_out=tmp_path / "d.csv", log=logs.append)
    assert s["pending"] == 1 and s["pending_changed_evidence"] == 1
    assert any("1 because evidence changed" in m for m in logs)
    s2 = la.aggregate([a2], store, reuse_stale=True, disagreements_out=tmp_path / "d.csv", log=lambda m: None)
    assert s2["accepted"] == 1 and s2["stale_reused"] == 1
    text = _isolated_llm.read_text()
    assert "; stale_evidence" in text and "# stale_evidence=1" in text


def test_warn_mixed_digests(tmp_path):
    store = lj.VoteStore(tmp_path)
    a, b = rec("a.com"), rec("b.com")
    v1, v2 = vote("gemma", a, "small_local"), vote("gemma", b, "small_local")
    v2["model_digest"] = "sha256:other"
    store.append("gemma", [v1, v2])
    logs = []
    lj.warn_mixed(store, ["gemma"], logs.append)
    assert any("2 different model digests" in m for m in logs)
