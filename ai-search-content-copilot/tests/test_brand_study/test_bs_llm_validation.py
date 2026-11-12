from __future__ import annotations
import csv
import json
import math

import pytest

from brand_study import config
from brand_study import llm_judges as lj
from brand_study import llm_validation as lv
from brand_study.config import ConfigError
from brand_study.labeling import load_overrides
from brand_study.llm_evidence import sha12

J = config.LLM_JUDGE_ORDER
SPEC = {"hypothesis": "one_sided_big_brand", "alpha": 0.05, "min_discordant": 5, "statistic": "weighted_net_shift",
        "bootstrap_reps": 4000, "bootstrap_seed": 20261110}


# --- kappa, confusion -------------------------------------------------------------------

def test_cohen_kappa_hand_example():
    # 20 items, marginals 10/10 for both raters, 14 agreements: po = 0.7, pe = 0.5 -> kappa = 0.4
    a = ["y"] * 7 + ["y"] * 3 + ["n"] * 3 + ["n"] * 7
    b = ["y"] * 7 + ["n"] * 3 + ["y"] * 3 + ["n"] * 7
    assert lv.cohen_kappa(a, b) == pytest.approx(0.4)
    assert math.isnan(lv.cohen_kappa(["x"] * 4, ["x"] * 4))  # pe = 1
    assert math.isnan(lv.cohen_kappa([], []))


def test_fleiss_kappa_hand_example():
    # 6 items x 3 raters: p_A = 10/18, p_B = 8/18 -> Pe = 41/81; P_i = 1,1,1,1,1/3,1/3 -> P-bar = 7/9
    # kappa = (7/9 - 41/81) / (1 - 41/81) = 22/40
    table = [[3, 0], [3, 0], [0, 3], [0, 3], [2, 1], [2, 1]]
    assert lv.fleiss_kappa_counts(table) == pytest.approx(22 / 40)
    ratings = [["a"] * 3, ["a"] * 3, ["b"] * 3, ["b"] * 3, ["a", "a", "b"], ["a", "b", "a"]]
    assert lv.fleiss_kappa(ratings) == pytest.approx(22 / 40)
    wiki = [[0, 0, 0, 0, 14], [0, 2, 6, 4, 2], [0, 0, 3, 5, 6], [0, 3, 9, 2, 0], [2, 2, 8, 1, 1], [7, 7, 0, 0, 0],
            [3, 2, 6, 3, 0], [2, 5, 3, 2, 2], [6, 5, 2, 1, 0], [0, 2, 2, 3, 7]]
    assert lv.fleiss_kappa_counts(wiki) == pytest.approx(0.210, abs=1e-3)
    assert math.isnan(lv.fleiss_kappa([["a"] * 3] * 4))


def test_confusion_counts():
    df = lv.confusion(["a", "a", "b"], ["a", "b", "b"], ["a", "b"], ["a", "b"]).set_index("human \\ judge")
    assert df.loc["a", "a"] == 1 and df.loc["a", "b"] == 1 and df.loc["b", "b"] == 1 and df.loc["b", "a"] == 0


# --- exact tests (descriptive) ----------------------------------------------------------

def test_binomial_and_mcnemar():
    assert lv.binom_two_sided(0, 6) == pytest.approx(2 / 64)
    assert lv.mcnemar_exact(6, 0)["one_sided"] == pytest.approx(1 / 64)
    assert lv.mcnemar_exact(5, 0)["one_sided"] == pytest.approx(1 / 32)
    assert lv.mcnemar_exact(5, 1)["two_sided"] == pytest.approx(14 / 64)
    assert lv.mcnemar_exact(9, 2)["one_sided"] == pytest.approx(67 / 2048)
    assert lv.mcnemar_exact(4, 4)["two_sided"] == pytest.approx(1.0)
    assert all(math.isnan(v) for v in lv.mcnemar_exact(0, 0).values())


# --- weighted statistics and the verdict ----------------------------------------------------

def rows_for(stratum, w, pairs):
    """pairs: list of (judge, human, count)."""
    out = []
    for judge, human, n in pairs:
        out += [{"stratum": stratum, "w": w, "j": judge, "human": human} for _ in range(n)]
    return out


def test_weighted_net_shift_hand_example():
    rows = rows_for("A", 2.0, [("big_brand", "small_local", 1), ("big_brand", "big_brand", 1)]) + \
        rows_for("B", 5.0, [("small_local", "big_brand", 1), ("small_local", "small_local", 3)])
    # (2*1 - 5*1) / (2*2 + 5*4) = -3/24
    assert lv.weighted_net_shift(rows, "j") == pytest.approx(-3 / 24)


def reviewer_rows():
    # population: 40 judge-big (20 human-small), 200 judge-small (20 human-big); sampled 18 + 18
    return rows_for("judge_big", 40 / 18, [("big_brand", "small_local", 9), ("big_brand", "big_brand", 9)]) + \
        rows_for("judge_small", 200 / 18, [("small_local", "big_brand", 2), ("small_local", "small_local", 16)])


def test_unequal_sampling_no_bias():
    rows = reviewer_rows()
    t = lv.bias_test(rows, "j", SPEC)
    assert (t["b"], t["c"]) == (9, 2) and t["mcnemar"]["one_sided"] < 0.05
    assert t["delta"] == pytest.approx((20 - 200 / 9) / 240)
    assert t["kind"] != "bias" and "big-brand bias (pre-specified" not in t["verdict"]
    lines, _ = lv._bias_section(rows, "j", SPEC, "x")
    assert any(lv.MCNEMAR_HEADING in ln and "b = 9" in ln for ln in lines)


def test_hidden_bias_found_by_weights():
    rows = rows_for("judge_big", 200 / 18, [("big_brand", "small_local", 4), ("big_brand", "big_brand", 14)]) + \
        rows_for("judge_small", 30 / 18, [("small_local", "big_brand", 5), ("small_local", "small_local", 13)])
    t = lv.bias_test(rows, "j", SPEC)
    assert (t["b"], t["c"]) == (4, 5) and t["mcnemar"]["one_sided"] > 0.05
    assert t["kind"] == "bias" and t["verdict"].startswith("evidence of big-brand bias")


@pytest.mark.parametrize("pairs,kind", [
    ([("big_brand", "small_local", 4), ("small_local", "small_local", 16)], "inconclusive"),
    ([("big_brand", "small_local", 5), ("small_local", "small_local", 15)], "bias"),
    ([("small_local", "big_brand", 8), ("big_brand", "big_brand", 12)], "reverse"),
    ([("big_brand", "small_local", 3), ("small_local", "big_brand", 3), ("big_brand", "big_brand", 14)], "none"),
])
def test_verdict_thresholds(pairs, kind):
    t = lv.bias_test(rows_for("s", 1.0, pairs), "j", SPEC)
    assert t["kind"] == kind
    if kind == "inconclusive":
        assert "b+c=4" in t["verdict"] and "minimum 5" in t["verdict"]
    if kind == "none":
        assert t["lower_bound"] < 0 < t["ci"][1]


def test_bootstrap_properties():
    rows = rows_for("A", 1.0, [("big_brand", "small_local", 3), ("big_brand", "big_brand", 2)]) + \
        rows_for("B", 3.0, [("small_local", "small_local", 7)])
    r1 = lv.stratified_bootstrap(lambda r: lv.weighted_net_shift(r, "j"), rows, 200, 7)
    r2 = lv.stratified_bootstrap(lambda r: lv.weighted_net_shift(r, "j"), rows, 200, 7)
    assert r1 == r2
    for rep in lv.bootstrap_replicates(rows, 50, 1):
        assert sum(1 for x in rep if x["stratum"] == "A") == 5 and sum(1 for x in rep if x["stratum"] == "B") == 7
    t = lv.bias_test(rows, "j", SPEC)
    assert t["ci"][0] <= t["delta"] <= t["ci"][1]


# --- sampling ----------------------------------------------------------------------------

def _rec(i, stratum_tag):
    ev = {"key": f"d{stratum_tag}{i}.com", "key_type": "domain", "queries": [{"text": "best coffee"}],
          "snippets": [{"title": f"T{i}", "snippet": "S", "source": "x"}],
          "local_pack": [{"title": "P", "type": "Cafe", "address": "A", "rating": 4, "reviews": 3, "city": "Tampa"}]
          if i % 2 else [], "homepage": {"title": f"Home {i}", "meta_description": "desc"}}
    return {"item_id": f"domain:{ev['key']}", "key_type": "domain", "key": ev["key"], "evidence": ev,
            "evidence_sha": sha12(ev), "rule_categories": [], "example_url": f"https://{ev['key']}/",
            "example_title": f"T{i}", "sectors": "coffee", "surfaces": "organic"}


PATTERNS = {"bigU": ("big_brand",) * 3, "bigN": ("big_brand", "big_brand", "small_local"),
            "smallU": ("small_local",) * 3, "smallN": ("small_local", "small_local", "big_brand"),
            "otherU": ("directory_media_other",) * 3,
            "otherN": ("directory_media_other", "marketplace_retailer", "not_a_brand")}


def population(store, counts):
    recs, per_judge = [], {j: [] for j in J}
    for tag, n in counts.items():
        for i in range(n):
            r = _rec(i, tag)
            recs.append(r)
            for j, c in zip(J, PATTERNS[tag]):
                per_judge[j].append({"judge": j, "model": config.LLM_JUDGES[j][1], "prompt_version": lj.PROMPT_VERSION,
                                     "item_id": r["item_id"], "evidence_sha": r["evidence_sha"], "status": "ok",
                                     "category": c, "confidence": 0.8, "reason": "r", "model_digest": "",
                                     "cli_version": "", "ts": "t"})
    for j, votes in per_judge.items():
        store.append(j, votes)
    return recs


COUNTS = {"bigU": 20, "bigN": 12, "smallU": 20, "smallN": 12, "otherU": 10, "otherN": 10}


def _paths(tmp_path):
    return tmp_path / "sample.csv", tmp_path / "snap.json"


def _sample(tmp_path, counts=COUNTS, **kw):
    store = lj.VoteStore(tmp_path / "votes")
    recs = population(store, counts)
    sp, sn = _paths(tmp_path)
    snap = lv.make_sample(recs, store, sample_out=sp, snapshot_out=sn, runs=["run1"], log=lambda m: None, **kw)
    return store, recs, snap


def test_allocation_default_and_redistribution():
    full = {s: 100 for s in lv.STRATA}
    a = lv.allocate(50, full)
    assert [a[s] for s in lv.STRATA] == [9, 9, 9, 9, 7, 7]
    short = dict(full, **{"big_brand|unanimous": 3})
    a2 = lv.allocate(50, short)
    assert a2["big_brand|unanimous"] == 3 and a2["big_brand|non_unanimous"] == 15 and sum(a2.values()) == 50
    short2 = dict(full, **{"big_brand|unanimous": 3, "big_brand|non_unanimous": 4})
    a3 = lv.allocate(50, short2)
    assert sum(a3.values()) == 50 and a3["small_local|unanimous"] + a3["small_local|non_unanimous"] > 18
    assert lv.allocate(10, {s: 1 for s in lv.STRATA}) == {s: 1 for s in lv.STRATA}


def test_make_sample(tmp_path):
    store, recs, snap = _sample(tmp_path)
    sp, sn = _paths(tmp_path)
    assert snap["n"] == 50 and [snap["strata"][s]["n"] for s in lv.STRATA] == [9, 9, 9, 9, 7, 7]
    it = next(i for i in snap["items"] if i["stratum"] == "big_brand|unanimous")
    assert it["weight"] == pytest.approx(20 / 9)
    assert snap["bias_test"] == {"hypothesis": "one_sided_big_brand", "alpha": 0.05, "min_discordant": 5,
                                 "statistic": "weighted_net_shift", "bootstrap_reps": config.BOOTSTRAP_REPS,
                                 "bootstrap_seed": config.BOOTSTRAP_SEED}
    text = sp.read_text()
    assert f"snapshot_sha={snap['snapshot_sha']}" in text
    for leak in ("big_brand|", "gemma", "qwen", "claude", "confidence", "0.8"):
        assert leak not in text.split("sample_id,")[1]
    rows, _ = lv.read_sample(sp)
    assert len(rows) == 50 and rows[0][1]["human_category"] == "" and rows[0][1]["homepage_title"].startswith("Home")
    # deterministic
    _, _, snap2 = _sample(tmp_path / "again")
    assert [i["item_id"] for i in snap2["items"]] == [i["item_id"] for i in snap["items"]]
    with pytest.raises(ConfigError, match="--extend"):
        lv.make_sample(recs, store, sample_out=sp, snapshot_out=sn, log=lambda m: None)


def test_make_sample_requires_complete_votes(tmp_path):
    store = lj.VoteStore(tmp_path / "votes")
    recs = population(store, {"bigU": 3})
    recs.append(_rec(99, "x"))
    sp, sn = _paths(tmp_path)
    with pytest.raises(ConfigError, match="do not have votes"):
        lv.make_sample(recs, store, sample_out=sp, snapshot_out=sn, log=lambda m: None)
    snap = lv.make_sample(recs, store, sample_out=sp, snapshot_out=sn, allow_pending=True, log=lambda m: None)
    assert snap["n"] == 3 and snap["pending_excluded"] == 1


def _label(sp, fn):
    rows, sha = lv.read_sample(sp)
    head = [ln for ln in sp.read_text().splitlines() if ln.startswith("#")]
    out = []
    for _, r in rows:
        r = dict(r)
        r["human_category"], r["human_note"] = fn(r)
        out.append(r)
    with sp.open("w", newline="") as f:
        f.write("\n".join(head) + "\n")
        w = csv.DictWriter(f, fieldnames=lv.SAMPLE_COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(out)


def test_extend(tmp_path):
    store, recs, snap = _sample(tmp_path)
    sp, sn = _paths(tmp_path)
    _label(sp, lambda r: ("small_local", "mine"))
    before = {r["sample_id"]: r for _, r in lv.read_sample(sp)[0]}
    snap2 = lv.extend_sample(10, recs, sample_in=sp, snapshot_in=sn, log=lambda m: None)
    rows = {r["sample_id"]: r for _, r in lv.read_sample(sp)[0]}
    assert len(rows) == 60 and all(rows[k] == v for k, v in before.items())
    assert all(rows[f"S{i:03d}"]["human_category"] == "" for i in range(51, 61))
    assert snap2["snapshot_sha"] != snap["snapshot_sha"] and snap2["bias_test"] == snap["bias_test"]
    ids = [i["item_id"] for i in snap2["items"]]
    assert len(ids) == len(set(ids)) == 60
    s = snap2["strata"]["big_brand|unanimous"]
    it = next(i for i in snap2["items"] if i["stratum"] == "big_brand|unanimous")
    assert it["weight"] == pytest.approx(s["N"] / s["n"]) and s["n"] > 9
    assert snap2["extensions"][0]["added"] == 10


def test_evaluate_full(tmp_path, monkeypatch):
    store, recs, snap = _sample(tmp_path)
    sp, sn = _paths(tmp_path)
    by_sid = {i["sample_id"]: i for i in snap["items"]}

    def human(r):
        s = by_sid[r["sample_id"]]["stratum"]
        if r["sample_id"] == "S001":
            return "unknown", ""
        if s.startswith("big_brand|non"):
            return "small_local", "independent"
        return {"big_brand": "big_brand", "small_local": "small_local"}.get(s.split("|")[0],
                                                                             "directory_media_other"), ""
    _label(sp, human)
    monkeypatch.setattr(config, "BIAS_MIN_DISCORDANT", 1000)  # changed after sampling: must be ignored
    out, snip = tmp_path / "val.md", tmp_path / "snip.csv"
    (tmp_path / "votes").rename(tmp_path / "votes_gone")  # evaluation reads the snapshot only
    res = lv.evaluate(sp, sn, out, snip, log=lambda m: None)
    text = out.read_text()
    for sec in ("## 1. Per-judge accuracy", "## 2. Ensemble", "## 3. Agreement between judges", "Fleiss",
                "## 4. Confusion matrices", "## 5. Big-brand bias check", "## 6. Metadata", "**Summary:**",
                lv.MCNEMAR_HEADING, "Design note"):
        assert sec in text
    assert "minimum 1000" not in text and '"min_discordant":5' in text
    assert res["n"] == 49 and "human `unknown` excluded: 1" in text
    # accepted (unanimous) items all correct -> 1.00
    assert "accepted-label accuracy 1.00" in res["summary"] and "policy unanimous" in res["summary"]
    assert lv.validation_summary_line(out).startswith("accepted-label accuracy 1.00")
    ov = load_overrides(snip)
    assert len(ov) == 49
    first = list(csv.DictReader(snip.read_text().splitlines()))[0]
    assert first["note"].startswith("human validation")
    assert res["contradictions"] == 0


def test_evaluate_snippet_contradictions_first(tmp_path):
    store, recs, snap = _sample(tmp_path)
    sp, sn = _paths(tmp_path)
    by_sid = {i["sample_id"]: i for i in snap["items"]}
    target = next(i["sample_id"] for i in snap["items"] if i["stratum"] == "small_local|unanimous")
    _label(sp, lambda r: ("big_brand" if r["sample_id"] == target else
                          "small_local" if by_sid[r["sample_id"]]["stratum"].startswith("small") else "", ""))
    snip = tmp_path / "snip.csv"
    res = lv.evaluate(sp, sn, tmp_path / "v.md", snip, log=lambda m: None)
    rows = list(csv.DictReader(snip.read_text().splitlines()))
    assert res["contradictions"] == 1 and rows[0]["category"] == "big_brand"
    assert "contradicts accepted LLM label small_local" in rows[0]["note"]
    assert not config.DEFAULT_OVERRIDES.read_text().count(target)  # human file never written


def test_accepted_flag_excludes_conflicts_rule(tmp_path):
    """A unanimous vote that aggregation sends to the human (conflicts_rule) is not an accepted label."""
    store = lj.VoteStore(tmp_path / "votes")
    recs = population(store, COUNTS)
    conflicted = [r for r in recs if r["key"].startswith("dsmallU")]
    for r in conflicted:
        r["rule_categories"] = ["big_brand"]  # candidate rule says big; judges unanimously say small
    sp, sn = _paths(tmp_path)
    snap = lv.make_sample(recs, store, sample_out=sp, snapshot_out=sn, runs=["run1"], log=lambda m: None)
    t = snap["full_votes"][conflicted[0]["item_id"]]
    assert t["stratum"] == "small_local|unanimous" and t["outcome"] == "conflicts_rule" and t["accepted"] is None
    big = next(i for i in snap["items"] if i["stratum"] == "big_brand|unanimous")
    assert big["accepted"] == "big_brand" and big["accepted_source"] == "llm_unanimous"
    sm = [i for i in snap["items"] if i["stratum"] == "small_local|unanimous"]
    assert sm and all(i["accepted"] is None for i in sm)
    # human says big for every small-unanimous item: no accepted label is contradicted
    by_sid = {i["sample_id"]: i for i in snap["items"]}
    _label(sp, lambda r: ("big_brand" if by_sid[r["sample_id"]]["stratum"] in ("small_local|unanimous",
                                                                               "big_brand|unanimous") else "", ""))
    out, snip = tmp_path / "v.md", tmp_path / "snip.csv"
    res = lv.evaluate(sp, sn, out, snip, log=lambda m: None)
    assert res["contradictions"] == 0
    assert res["summary"].startswith("**Summary:** accepted-label accuracy 1.00 (n=9")
    text = out.read_text()
    assert "unanimous-vote items (3 identical votes, accepted or not)" in text


def test_accepted_flag_majority_policy(tmp_path):
    store, recs, snap = _sample(tmp_path, policy="majority")
    two = next(i for i in snap["items"] if i["stratum"] == "big_brand|non_unanimous")
    assert two["accepted"] == "big_brand" and two["accepted_source"] == "llm_majority"
    other_n = next(i for i in snap["items"] if i["stratum"] == "other|non_unanimous")
    assert other_n["accepted"] is None  # three different votes: no majority


def test_evaluate_input_validation(tmp_path):
    store, recs, snap = _sample(tmp_path)
    sp, sn = _paths(tmp_path)
    _label(sp, lambda r: ("huge", ""))
    with pytest.raises(ConfigError, match="invalid human_category 'huge'"):
        lv.evaluate(sp, sn, tmp_path / "v.md", tmp_path / "s.csv", log=lambda m: None)
    data = json.loads(sn.read_text())
    data["seed"] = 1
    sn.write_text(json.dumps(data))
    with pytest.raises(ConfigError, match="snapshot_sha mismatch"):
        lv.evaluate(sp, sn, tmp_path / "v.md", tmp_path / "s.csv", log=lambda m: None)


def test_dropped_stratum_reported(tmp_path):
    store, recs, snap = _sample(tmp_path)
    sp, sn = _paths(tmp_path)
    by_sid = {i["sample_id"]: i for i in snap["items"]}
    _label(sp, lambda r: ("unknown" if by_sid[r["sample_id"]]["stratum"] == "other|unanimous" else "small_local", ""))
    out = tmp_path / "v.md"
    lv.evaluate(sp, sn, out, tmp_path / "s.csv", log=lambda m: None)
    assert "Strata with no usable human labels" in out.read_text() and "other|unanimous" in out.read_text()


# --- spreadsheet round trip, extension after overrides, stray rows -----------------------------

def _spreadsheet_save(sp, encoding="utf-8-sig"):
    """Mimic Excel/LibreOffice saving the sample: BOM, CRLF, every row padded to the full column count, comment
    lines split at their commas (so they gain commas) and one comment cell quoted."""
    width = len(lv.SAMPLE_COLUMNS)
    out = []
    for ln in sp.read_text(encoding="utf-8").splitlines():
        if ln.startswith("#"):
            cells = ln.split(",")
            if len(out) == 1:
                cells = ['"' + cells[0] + '"'] + cells[1:]
            ln = ",".join(cells + [""] * max(0, width - len(cells)))
        out.append(ln)
    out.append("," * (width - 1))  # an empty trailing row
    sp.write_bytes(("\r\n".join(out) + "\r\n").encode(encoding))


@pytest.mark.parametrize("encoding", ["utf-8-sig", "cp1252"])
def test_sample_survives_spreadsheet_save(tmp_path, encoding):
    store, recs, snap = _sample(tmp_path)
    sp, sn = _paths(tmp_path)
    _label(sp, lambda r: ("Small_local" if r["sample_id"] == "S002" else "small_local", "café"))
    _spreadsheet_save(sp, encoding)
    raw = sp.read_bytes()
    assert b"\r\n" in raw and (raw.startswith(b"\xef\xbb\xbf") == (encoding == "utf-8-sig"))
    rows, sha = lv.read_sample(sp)
    assert sha == snap["snapshot_sha"] and rows[0][1]["sample_id"] == "S001"
    logs = []
    res = lv.evaluate(sp, sn, tmp_path / "v.md", tmp_path / "s.csv", log=logs.append)
    assert res["n"] == 50 and not any("WARNING" in m for m in logs)
    lv.extend_sample(5, recs, sample_in=sp, snapshot_in=sn, log=lambda m: None)
    assert len(lv.read_sample(sp)[0]) == 55


def test_extend_keeps_items_that_left_the_item_set(tmp_path):
    store, recs, snap = _sample(tmp_path, counts={"bigU": 20, "bigN": 12, "smallU": 20, "smallN": 12,
                                                  "otherU": 10, "otherN": 10})
    sp, sn = _paths(tmp_path)
    taken = {i["item_id"] for i in snap["items"]}
    # every unsampled big_brand|non_unanimous item got a human override, so it is gone from the current item set
    gone = {r["item_id"] for r in recs if "dbigN" in r["key"] and r["item_id"] not in taken}
    assert gone
    current = [r for r in recs if r["item_id"] not in gone]
    with pytest.raises(ConfigError, match="no longer in the item set"):
        lv.extend_sample(30, current, sample_in=sp, snapshot_in=sn, fallback_records=[], log=lambda m: None)
    assert lv.load_snapshot(sn)["snapshot_sha"] == snap["snapshot_sha"]  # nothing written
    logs = []
    snap2 = lv.extend_sample(30, current, sample_in=sp, snapshot_in=sn, fallback_records=recs, log=logs.append)
    added = {i["item_id"] for i in snap2["items"]} - taken
    assert added & gone and any("left the item set" in m for m in logs)
    # default fallback is items.jsonl
    from brand_study import llm_evidence as ev
    ev.write_items(recs)
    store2, recs2, snap3 = _sample(tmp_path / "b")
    sp2, sn2 = _paths(tmp_path / "b")
    snap4 = lv.extend_sample(30, [r for r in recs2 if r["item_id"] not in gone], sample_in=sp2, snapshot_in=sn2,
                             log=lambda m: None)
    assert snap4["n"] == 80


def test_extend_requires_matching_sha(tmp_path):
    store, recs, snap = _sample(tmp_path)
    sp, sn = _paths(tmp_path)
    sp.write_text("\n".join(ln for ln in sp.read_text().splitlines() if "snapshot_sha=" not in ln) + "\n")
    with pytest.raises(ConfigError, match="snapshot_sha missing"):
        lv.extend_sample(5, recs, sample_in=sp, snapshot_in=sn, log=lambda m: None)


def test_evaluate_warns_on_stray_sample_ids(tmp_path):
    store, recs, snap = _sample(tmp_path)
    sp, sn = _paths(tmp_path)
    _label(sp, lambda r: ("small_local", ""))
    with sp.open("a") as f:
        f.write("S999,domain,extra.com" + "," * (len(lv.SAMPLE_COLUMNS) - 3) + "\n")
    logs = []
    res = lv.evaluate(sp, sn, tmp_path / "v.md", tmp_path / "s.csv", log=logs.append)
    assert res["n"] == 50 and any("not in the snapshot" in m and "S999" in m for m in logs)


# --- review round 3: blank rows are missing data, not a smaller stratum ---------------------------

def _mini_snap(n_a, n_b):
    """Stratum A (N=40): judges say big, human small. Stratum B (N=200): judges and human say small."""
    items = []
    for s, n, cat in (("big_brand|unanimous", n_a, "big_brand"), ("small_local|unanimous", n_b, "small_local")):
        for _ in range(n):
            k = len(items) + 1
            items.append({"sample_id": f"S{k:03d}", "item_id": f"domain:d{k}.com", "key_type": "domain",
                          "key": f"d{k}.com", "stratum": s, "accepted": cat,
                          "votes": {j: {"status": "ok", "category": cat} for j in J}})
    snap = {"judges": {j: {} for j in J}, "items": items,
            "strata": {"big_brand|unanimous": {"N": 40, "n": 0}, "small_local|unanimous": {"N": 200, "n": 0}}}
    lv._set_weights(snap)
    return snap


def test_blank_rows_do_not_change_weighted_net_shift():
    snap = _mini_snap(4, 4)
    full = {it["sample_id"]: "small_local" for it in snap["items"]}
    d_full = lv.weighted_net_shift(lv._rows_for_eval(snap, full), "majority")
    assert d_full == pytest.approx(40 / 240)
    # blank half of stratum B (as right after --extend, or a half-labelled sample)
    part = dict(full)
    for sid in ("S007", "S008"):
        part[sid] = ""
    rows = lv._rows_for_eval(snap, part)
    assert lv.weighted_net_shift(rows, "majority") == pytest.approx(d_full)
    assert {r["stratum"]: r["w"] for r in rows} == {"big_brand|unanimous": 10.0, "small_local|unanimous": 100.0}
    # human `unknown` keeps its place in the denominator: B's weight stays 200/4, so Σw covers the decidable part
    unk = dict(full, S007="unknown", S008="unknown")
    w = {r["stratum"]: r["w"] for r in lv._rows_for_eval(snap, unk)}
    assert w["small_local|unanimous"] == pytest.approx(50.0)
    assert math.isnan(lv.eval_weights(snap, {"S001": "small_local"})["small_local|unanimous"])  # no answers


def test_evaluate_after_extend_warns_and_reweights(tmp_path):
    store, recs, snap = _sample(tmp_path)
    sp, sn = _paths(tmp_path)
    by_sid = {i["sample_id"]: i for i in snap["items"]}
    _label(sp, lambda r: ("small_local" if by_sid[r["sample_id"]]["stratum"].startswith("big_brand|non")
                          else by_sid[r["sample_id"]]["stratum"].split("|")[0].replace("other", "not_a_brand"), ""))
    out1 = tmp_path / "v1.md"
    lv.evaluate(sp, sn, out1, tmp_path / "s1.csv", log=lambda m: None)
    lv.extend_sample(30, recs, sample_in=sp, snapshot_in=sn, log=lambda m: None)
    logs = []
    out2 = tmp_path / "v2.md"
    res = lv.evaluate(sp, sn, out2, tmp_path / "s2.csv", log=logs.append)
    assert res["n"] == 50 and any("have no human_category yet" in m for m in logs)
    t1, t2 = out1.read_text(), out2.read_text()
    assert "sampled rows are still blank" in t2 and "sampled rows are still blank" not in t1

    def delta_line(t):
        return next(ln for ln in t.splitlines() if "weighted net shift" in ln.lower() and "Δ" in ln)
    # the 30 blank rows must not move the estimate: same answered rows, same per-stratum N_h / answered_h
    assert delta_line(t1) == delta_line(t2)


def test_extend_warns_on_stray_sample_ids(tmp_path):
    store, recs, snap = _sample(tmp_path)
    sp, sn = _paths(tmp_path)
    with sp.open("a") as f:
        f.write("S051,domain,extra.com" + "," * (len(lv.SAMPLE_COLUMNS) - 3) + "\n")
    logs = []
    lv.extend_sample(5, recs, sample_in=sp, snapshot_in=sn, log=logs.append)
    assert any("not in the snapshot" in m and "S051" in m for m in logs)
