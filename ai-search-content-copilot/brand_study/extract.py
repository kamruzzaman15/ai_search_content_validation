from __future__ import annotations
import re
from pathlib import Path

from brand_study import cache
from brand_study.domains import host, normalize_url, root_domain
from brand_study.entities import AliasIndex, entity_norm, find_aliases

LIST_TITLE_STOPLIST = {
    "pros", "cons", "key features", "features", "best for", "price", "pricing", "overview", "summary",
    "considerations", "tips", "bottom line", "why", "ingredients", "key ingredients", "location", "locations",
    "hours", "ambiance", "atmosphere", "menu", "benefits", "how to use", "what to look for", "other options",
    "other popular options", "factors to consider", "things to consider", "recommendations", "popular choices",
    "highlights", "specialty", "vibe", "address", "reviews", "rating", "budget friendly", "drugstore options",
    "high end options", "affordable options", "skin type", "texture", "application", "note", "important",
}
CATEGORY_HINT_RE = re.compile(r"^(for|best|top|other|more|budget|drugstore|luxury|high end|affordable)\b")


def _list(v) -> list:
    return v if isinstance(v, list) else []


def _dict(v) -> dict:
    return v if isinstance(v, dict) else {}


def _str(v) -> str:
    return v if isinstance(v, str) else ""


# --- Loading -------------------------------------------------------------------

def load_run(run: str, root: Path | None = None) -> dict[str, dict]:
    """query_id -> {"search": wrapper|None, "ai_overview": wrapper|None}."""
    out: dict[str, dict] = {}
    for p in cache.iter_run_files(run, root):
        parts = p.name.split(".")
        if len(parts) != 3 or parts[1] not in cache.KINDS:
            continue
        out.setdefault(parts[0], {"search": None, "ai_overview": None})[parts[1]] = cache.read_cached(p)
    return out


# --- AIO content -----------------------------------------------------------------

def aio_content(search_wrap: dict | None, aio_wrap: dict | None) -> dict | None:
    """The AI Overview object with text_blocks (inline or follow-up), else None."""
    body = _dict((search_wrap or {}).get("response"))
    ai = _dict(body.get("ai_overview"))
    if _list(ai.get("text_blocks")):
        return ai
    if ai.get("page_token") and aio_wrap and _dict(aio_wrap.get("_meta")).get("outcome") == "ok":
        fb = _dict(aio_wrap.get("response"))
        inner = _dict(fb.get("ai_overview"))
        if _list(inner.get("text_blocks")):
            return inner
        if _list(fb.get("text_blocks")):
            return fb
    return None


def aio_status(search_wrap: dict | None, aio_wrap: dict | None, ledger_error: str | None = None) -> tuple[str, str]:
    if search_wrap is None:
        return ("search_failed", ledger_error) if ledger_error else ("not_collected", "")
    meta = _dict(search_wrap.get("_meta"))
    if meta.get("outcome") == "no_results":
        return "absent", "no_results"
    body = _dict(search_wrap.get("response"))
    ai = body.get("ai_overview")
    if not isinstance(ai, dict):
        return "absent", ""
    if _list(ai.get("text_blocks")):
        return "present_inline", ""
    if ai.get("page_token"):
        if aio_wrap is None:
            return "followup_failed", "stale_token"
        am = _dict(aio_wrap.get("_meta"))
        if am.get("outcome") == "ok" and aio_content(search_wrap, aio_wrap):
            return "present_followup", ""
        return "followup_failed", am.get("error_kind") or "empty"
    if ai.get("error"):
        return "error", _str(ai.get("error"))[:200]
    return "absent", "empty_ai_overview"


def _walk_blocks(blocks, block_type: str, segs: list, depth: int = 0):
    if depth > 8:
        return
    for b in _list(blocks):
        if isinstance(b, str):
            segs.append((b, "sentence", block_type))
            continue
        if not isinstance(b, dict):
            continue
        btype = _str(b.get("type")) or block_type
        kind = "heading" if btype == "heading" else "sentence"
        if _str(b.get("title")):
            segs.append((b["title"], "list_item" if block_type == "list" else "heading", btype))
        if _str(b.get("snippet")):
            segs.append((b["snippet"], "list_item" if block_type == "list" else kind, btype))
        if _str(b.get("text")):
            segs.append((b["text"], kind, btype))
        shop = _str(_dict(b.get("shopping_result")).get("title"))
        if shop:
            segs.append((shop, "list_item", btype))
        for item in _list(b.get("list")):
            _walk_blocks([item], "list", segs, depth + 1)
        _walk_table(b.get("table"), btype, segs)
        for row in _list(b.get("comparison")):
            if isinstance(row, dict):
                if _str(row.get("feature")):
                    segs.append((row["feature"], "cell", btype))
                for v in _list(row.get("values")):
                    if isinstance(v, str):
                        segs.append((v, "cell", btype))
        _walk_blocks(b.get("text_blocks"), btype, segs, depth + 1)


def _walk_table(table, btype: str, segs: list):
    if isinstance(table, dict):
        table = _list(table.get("rows")) + [table.get("headers")]
    for row in _list(table):
        cells = row if isinstance(row, list) else (list(row.values()) if isinstance(row, dict) else [])
        for c in cells:
            if isinstance(c, str):
                segs.append((c, "cell", btype))


def aio_segments(ai: dict | None) -> list[tuple[str, str, str]]:
    """(segment_text, segment_kind, block_type) for every text piece of the AI Overview."""
    segs: list = []
    if not ai:
        return segs
    _walk_blocks(ai.get("text_blocks"), "paragraph", segs)
    return [(t.strip(), k, bt) for t, k, bt in segs if t and t.strip()]


def aio_text(ai: dict | None) -> list[tuple[str, str]]:
    return [(t, k) for t, k, _ in aio_segments(ai)]


def list_title_candidates(ai: dict | None) -> list[tuple[str, str]]:
    """(raw candidate, block_type) from list titles and highlighted words."""
    out = []

    def visit(blocks, btype, depth=0):
        if depth > 8:
            return
        for b in _list(blocks):
            if not isinstance(b, dict):
                continue
            bt = _str(b.get("type")) or btype
            for w in _list(b.get("snippet_highlighted_words")):
                if isinstance(w, str):
                    out.append((w, bt))
            for item in _list(b.get("list")):
                if isinstance(item, dict):
                    if _str(item.get("title")):
                        out.append((item["title"], "list"))
                    shop = _str(_dict(item.get("shopping_result")).get("title"))
                    if shop:
                        out.append((shop, "shopping_result"))
                    visit([item], "list", depth + 1)
            visit(b.get("text_blocks"), bt, depth + 1)

    if ai:
        visit(ai.get("text_blocks"), "paragraph")
    cleaned = []
    for raw, bt in out:
        t = re.sub(r"\s*[:\-–—]\s*$", "", raw.strip())
        n = entity_norm(t)
        if not (2 <= len(t) <= 60) or not n or n in LIST_TITLE_STOPLIST or CATEGORY_HINT_RE.match(n):
            continue
        cleaned.append((t, bt))
    return cleaned


def _whole_word_in(needle: str, hay: str) -> bool:
    return bool(needle) and re.search(rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])", hay) is not None


def match_local_pack(cand_norm: str, places: list[dict]) -> dict | None:
    for p in places:
        pn = p.get("entity_norm") or ""
        if not pn:
            continue
        if cand_norm == pn or _whole_word_in(pn, cand_norm):
            return p
        if len(cand_norm.split()) >= 2 and _whole_word_in(cand_norm, pn):
            return p
    return None


# --- Row builders ----------------------------------------------------------------

def organic_rows(run: str, query_id: str, search_body: dict) -> list[dict]:
    items = [r for r in _list(_dict(search_body).get("organic_results")) if isinstance(r, dict) and _str(r.get("link"))]
    items.sort(key=lambda r: r.get("position") if isinstance(r.get("position"), (int, float)) else 10 ** 6)
    rows = []
    for rank, r in enumerate(items[:10], start=1):
        h = host(r["link"])
        rows.append({
            "run": run, "query_id": query_id, "rank": rank, "url": r["link"], "url_norm": normalize_url(r["link"]),
            "host": h, "root_domain": root_domain(h), "title": _str(r.get("title")), "source": _str(r.get("source")),
        })
    return rows


def aio_reference_rows(run: str, query_id: str, ai: dict | None) -> list[dict]:
    if not ai:
        return []
    cites: dict[int, int] = {}

    def count(blocks, depth=0):
        if depth > 8:
            return
        for b in _list(blocks):
            if not isinstance(b, dict):
                continue
            for i in _list(b.get("reference_indexes")):
                if isinstance(i, int):
                    cites[i] = cites.get(i, 0) + 1
            count(b.get("list"), depth + 1)
            count(b.get("text_blocks"), depth + 1)

    count(ai.get("text_blocks"))
    rows = []
    for pos, ref in enumerate(_list(ai.get("references"))):
        if not isinstance(ref, dict) or not _str(ref.get("link")):
            continue
        idx = ref.get("index") if isinstance(ref.get("index"), int) else pos
        h = host(ref["link"])
        rows.append({
            "run": run, "query_id": query_id, "ref_index": idx, "url": ref["link"],
            "url_norm": normalize_url(ref["link"]), "host": h, "root_domain": root_domain(h),
            "source": _str(ref.get("source")), "title": _str(ref.get("title")), "cited_in_blocks": cites.get(idx, 0),
        })
    return rows


def local_pack_rows(run: str, query_id: str, search_body: dict) -> list[dict]:
    lr = _dict(search_body).get("local_results")
    places = _list(lr.get("places")) if isinstance(lr, dict) else _list(lr)
    rows = []
    for pos, p in enumerate(places, start=1):
        if not isinstance(p, dict) or not _str(p.get("title")):
            continue
        website = _str(_dict(p.get("links")).get("website")) or _str(p.get("website"))
        rows.append({
            "run": run, "query_id": query_id, "position": p.get("position") if isinstance(p.get("position"), int) else pos,
            "title": p["title"], "entity_norm": entity_norm(p["title"]), "website": website,
            "root_domain": root_domain(host(website)) if website else "",
            "rating": p.get("rating"), "reviews": p.get("reviews"), "type": _str(p.get("type")),
        })
    return rows


def aio_mention_rows(run: str, query_id: str, ai: dict | None, alias_index: AliasIndex,
                     local_pack: list[dict] | None = None) -> list[dict]:
    if not ai:
        return []
    found: dict[str, dict] = {}

    def add(entity: str, method: str, block_type: str, lp_domain: str = ""):
        n = entity_norm(entity)
        if not n:
            return
        row = found.setdefault(n, {"run": run, "query_id": query_id, "entity": entity, "entity_norm": n,
                                   "methods": set(), "block_types": set(), "local_pack_match_domain": ""})
        row["methods"].add(method)
        row["block_types"].add(block_type)
        if lp_domain and not row["local_pack_match_domain"]:
            row["local_pack_match_domain"] = lp_domain

    for text, kind, btype in aio_segments(ai):
        for entity, methods in find_aliases([(text, kind)], alias_index).items():
            for m in methods:
                add(entity, m, btype)
    for raw, btype in list_title_candidates(ai):
        if find_aliases([(raw, "list_item")], alias_index):
            continue  # e.g. "CeraVe Oil Control Gel-Cream": the brand is already counted by the dictionary
        add(raw, "list_title", btype)
        place = match_local_pack(entity_norm(raw), local_pack or [])
        if place is not None:
            add(raw, "local_pack_name", btype, place.get("root_domain") or "")
    rows = []
    for row in found.values():
        row["methods"] = "|".join(sorted(row["methods"]))
        row["block_types"] = "|".join(sorted(row["block_types"]))
        rows.append(row)
    return sorted(rows, key=lambda r: r["entity_norm"])


def query_status_row(query, run: str, search_wrap: dict | None, aio_wrap: dict | None,
                     ledger_error: str | None = None) -> dict:
    status, detail = aio_status(search_wrap, aio_wrap, ledger_error)
    body = _dict((search_wrap or {}).get("response"))
    org = organic_rows(run, query.id, body) if search_wrap else []
    lp = local_pack_rows(run, query.id, body) if search_wrap else []
    ai = aio_content(search_wrap, aio_wrap)
    meta = _dict((search_wrap or {}).get("_meta"))
    return {
        "run": run, "query_id": query.id, "sector": query.sector, "query_type": query.query_type,
        "city": query.city, "location": query.location, "collected_at": meta.get("collected_at_utc", ""),
        "superseded": bool(meta.get("supersedes")), "supersedes": meta.get("supersedes") or "",
        "search_outcome": meta.get("outcome", "") if search_wrap else "",
        "aio_status": status, "aio_detail": detail or "", "n_organic": len(org),
        "n_organic_domains": len({r["root_domain"] for r in org if r["root_domain"]}),
        "has_local_pack": bool(lp), "n_local_pack": len(lp),
        "n_aio_refs": len(aio_reference_rows(run, query.id, ai)),
    }
