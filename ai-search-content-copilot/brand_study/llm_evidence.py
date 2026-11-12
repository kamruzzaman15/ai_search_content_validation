"""Item set and evidence records for the LLM judges (no model calls here).

The item set is what the rules and the human overrides leave `unknown` (the `labels_to_review` table), recomputed
from the raw cache WITHOUT the LLM layer so that re-running the LLM step is idempotent.

Each item gets an evidence record built from data the study already collected (URLs, titles, snippets, queries,
local-pack places, AIO segments), plus an optional homepage fetch. The record sent to the judges is hashed
(`evidence_sha`); counts, timestamps and fetch metadata are stored beside it, never inside it.
"""
from __future__ import annotations
import hashlib
import ipaddress
import json
import os
import random
import re
import socket
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable
from urllib.parse import urljoin, urlsplit

from brand_study import cache, config, extract
from brand_study.domains import host as url_host
from brand_study.domains import normalize_url, root_domain
from brand_study.entities import entity_norm
from brand_study.labeling import Labeler, _sector_map, candidate_rule_categories

MAX_URLS, MAX_QUERIES, MAX_SNIPPETS, MAX_PLACES, MAX_SEGMENTS = 3, 3, 3, 3, 2
TITLE_CHARS, SNIPPET_CHARS, SEGMENT_CHARS, RECORD_CHARS = 200, 300, 300, 2500
SURFACE_ORDER = {"organic": 0, "aio_reference": 1, "local_pack": 2}
REDIRECT_CODES = (301, 302, 303, 307, 308)


# --- Small helpers -----------------------------------------------------------------

def clip(text, n: int) -> str:
    t = " ".join(str(text or "").split())
    return t if len(t) <= n else t[: n - 1].rstrip() + "…"


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha12(obj) -> str:
    return hashlib.sha256(canonical(obj).encode("utf-8")).hexdigest()[:12]


def run_order(run: str) -> tuple[int, str]:
    """Run-first ordering: run1's entries keep their place when run2 is added."""
    runs = list(config.DEFAULT_RUNS)
    return (runs.index(run) if run in runs else len(runs), run)


def atomic_write_text(path: Path, text: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return path


def collected_runs(runs, root: Path | None = None) -> list[str]:
    return [r for r in runs if any(True for _ in cache.iter_run_files(r, root))]


# --- Item set ----------------------------------------------------------------------

def build_items(queries, runs, labeler: Labeler, root: Path | None = None, ledger_rows=None) -> tuple[list[dict], dict]:
    """Items still `unknown` under rules + human overrides only. `labeler` must not carry LLM labels."""
    from brand_study import analyze
    if labeler.llm_overrides:
        raise ValueError("build_items needs a Labeler without LLM labels")
    frames = analyze.build_frames(queries, runs, labeler, root, [] if ledger_rows is None else ledger_rows)
    review = frames["labels_to_review"]
    sectors = _sector_map(frames["query_status"])
    cands = candidate_rule_categories(frames["aio_mentions"], labeler, sectors)
    items = []
    for r in review.itertuples():
        items.append({
            "item_id": f"{r.key_type}:{r.key}", "key_type": r.key_type, "key": r.key,
            "example_raw": r.example_raw, "count": int(r.count), "n_queries": int(r.n_queries),
            "sectors": r.sectors, "surfaces": r.surfaces, "example_url": r.example_url,
            "example_title": r.example_title,
            "rule_categories": sorted(cands.get(r.key, set())) if r.key_type == "entity" else [],
        })
    items.sort(key=lambda i: (i["key_type"], i["key"]))
    return items, frames


def order_items(items: list[dict], seed: int = config.LLM_ITEM_SEED, limit: int | None = None) -> list[dict]:
    """Sorted, then shuffled with a fixed seed; `--limit N` takes the first N."""
    out = sorted(items, key=lambda i: (i["key_type"], i["key"]))
    random.Random(seed).shuffle(out)
    return out[:limit] if limit is not None else out


# --- Evidence index (one pass over the raw cache) --------------------------------------

class EvidenceIndex:
    def __init__(self):
        self.domain: dict[str, dict] = defaultdict(lambda: {
            "urls": [], "queries": [], "snippets": [], "places": [], "sectors": set(), "surfaces": set(),
            "hosts": Counter()})
        self.segments: dict[tuple[str, str], list[str]] = {}
        self.places: dict[tuple[str, str], list[dict]] = {}
        self.query_info: dict[str, dict] = {}
        self.query_sector: dict[str, str] = {}

    def query_entry(self, qid: str) -> dict:
        return self.query_info.get(qid, {"text": "", "query_type": "", "city": ""})


def _place_entry(p: dict, city: str) -> dict:
    return {"title": clip(p.get("title"), TITLE_CHARS), "type": clip(p.get("type"), 80),
            "address": clip(p.get("address"), 150), "rating": p.get("rating"), "reviews": p.get("reviews"),
            "city": city}


def build_index(queries, runs, root: Path | None = None) -> EvidenceIndex:
    idx = EvidenceIndex()
    for q in queries:
        idx.query_info[q.id] = {"text": q.query, "query_type": q.query_type, "city": q.city}
        idx.query_sector[q.id] = q.sector
    for run in runs:
        ro = run_order(run)
        data = extract.load_run(run, root)
        for q in queries:
            d = data.get(q.id)
            if not d or d.get("search") is None:
                continue
            sw, aw = d["search"], d["ai_overview"]
            body = sw.get("response") or {}
            raw_org = [r for r in (body.get("organic_results") or []) if isinstance(r, dict) and
                       isinstance(r.get("link"), str) and r.get("link")]
            raw_org.sort(key=lambda r: r.get("position") if isinstance(r.get("position"), (int, float)) else 10 ** 6)
            for rank, r in enumerate(raw_org[:10], start=1):
                _add_domain_hit(idx, ro, q, "organic", rank, r["link"], r.get("title"), r.get("snippet"),
                                r.get("source"))
            ai = extract.aio_content(sw, aw)
            if ai:
                for pos, ref in enumerate(ai.get("references") or []):
                    if isinstance(ref, dict) and isinstance(ref.get("link"), str) and ref.get("link"):
                        i = ref.get("index") if isinstance(ref.get("index"), int) else pos
                        _add_domain_hit(idx, ro, q, "aio_reference", i, ref["link"], ref.get("title"),
                                        ref.get("snippet"), ref.get("source"))
                idx.segments[(run, q.id)] = [t for t, _, _ in extract.aio_segments(ai)]
            lr = body.get("local_results")
            places = lr.get("places") if isinstance(lr, dict) else lr
            plist = []
            for pos, p in enumerate(places if isinstance(places, list) else [], start=1):
                if not isinstance(p, dict) or not isinstance(p.get("title"), str) or not p.get("title"):
                    continue
                website = (p.get("links") or {}).get("website") if isinstance(p.get("links"), dict) else None
                website = website if isinstance(website, str) else (p.get("website") if isinstance(
                    p.get("website"), str) else "")
                entry = _place_entry(p, q.city)
                plist.append({**entry, "entity_norm": entity_norm(p["title"])})
                rd = root_domain(url_host(website)) if website else ""
                if rd:
                    dom = idx.domain[rd]
                    dom["places"].append(((ro, q.city, entry["title"], pos), entry))
                    dom["sectors"].add(q.sector)
                    dom["surfaces"].add("local_pack")
                    dom["queries"].append(((ro, q.id), q.id))
                    un = normalize_url(website)
                    if un:
                        dom["urls"].append(((ro, q.id, 1000 + pos, un), un))
                        raw_host = urlsplit(un).hostname or ""
                        if raw_host == rd or raw_host.endswith("." + rd):
                            dom["hosts"][raw_host] += 1
            idx.places[(run, q.id)] = plist
    return idx


def _add_domain_hit(idx, ro, q, surface, pos, link, title, snippet, source):
    h = url_host(link)
    rd = root_domain(h)
    if not rd:
        return
    dom = idx.domain[rd]
    un = normalize_url(link)
    dom["urls"].append(((ro, q.id, pos, un), un))
    dom["queries"].append(((ro, q.id), q.id))
    dom["sectors"].add(q.sector)
    dom["surfaces"].add(surface)
    raw_host = (urlsplit(un).hostname or "") if un else ""
    if raw_host and (raw_host == rd or raw_host.endswith("." + rd)):
        dom["hosts"][raw_host] += 1
    entry = {"title": clip(title, TITLE_CHARS), "snippet": clip(snippet, SNIPPET_CHARS),
             "source": clip(source, 80)}
    if entry["title"] or entry["snippet"]:
        dom["snippets"].append(((ro, SURFACE_ORDER[surface], q.id, pos), entry))


def _first_unique(entries, key=lambda v: canonical(v), n: int = 3) -> list:
    out, seen = [], set()
    for _, v in sorted(entries, key=lambda e: e[0]):
        k = key(v)
        if k in seen:
            continue
        seen.add(k)
        out.append(v)
        if len(out) >= n:
            break
    return out


def target_host(idx: EvidenceIndex, domain: str) -> str:
    """Most frequent host among the item's URLs (ties: alphabetical), else the root domain."""
    hosts = idx.domain[domain]["hosts"] if domain in idx.domain else Counter()
    if not hosts:
        return domain
    return sorted(hosts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


# --- Evidence records ----------------------------------------------------------------

def _trim(ev: dict, lists: tuple[str, ...]) -> dict:
    """Drop trailing list entries (longest list first) until the record fits RECORD_CHARS."""
    while len(canonical(ev)) > RECORD_CHARS:
        cands = [k for k in lists if len(ev.get(k) or []) > 1]
        if not cands:
            break
        k = max(cands, key=lambda k: (len(canonical(ev[k])), k))
        ev[k] = ev[k][:-1]
    return ev


def homepage_evidence(record: dict | None) -> dict | None:
    if record is None:
        return None
    return record.get("content") or {"available": False}


def domain_evidence(item: dict, idx: EvidenceIndex, fetched: dict | None) -> dict:
    d = idx.domain.get(item["key"]) or {"urls": [], "queries": [], "snippets": [], "places": [],
                                         "sectors": set(), "surfaces": set()}
    queries = _first_unique([(k, qid) for k, qid in d["queries"]], key=lambda v: v, n=MAX_QUERIES)
    ev = {
        "key": item["key"], "key_type": "domain",
        "sectors": sorted(s for s in d["sectors"] if s), "surfaces": sorted(d["surfaces"]),
        "example_urls": _first_unique(d["urls"], key=lambda v: v, n=MAX_URLS),
        "queries": [idx.query_entry(qid) for qid in queries],
        "snippets": _first_unique(d["snippets"], key=lambda v: (v["title"], v["snippet"]), n=MAX_SNIPPETS),
        "local_pack": _first_unique(d["places"], key=lambda v: (v["title"], v["address"]), n=MAX_PLACES),
    }
    ev = _trim(ev, ("snippets", "local_pack", "queries", "example_urls"))
    hp = homepage_evidence(fetched)
    if hp is not None:
        ev["homepage"] = hp
    return ev


def _segment_excerpt(seg: str, name: str) -> str:
    text = " ".join(seg.split())
    i = text.lower().find((name or "").lower())
    if i > SEGMENT_CHARS // 2:
        text = "…" + text[i - 100:]
    return clip(text, SEGMENT_CHARS)


def entity_evidence(item: dict, idx: EvidenceIndex, mentions) -> dict:
    norm = item["key"]
    rows = mentions[mentions["entity_norm"] == norm] if mentions is not None and not mentions.empty else []
    qkeys, methods, blocks, sectors, lp_domains = [], set(), set(), set(), []
    for r in (rows.itertuples() if len(rows) else []):
        ro = run_order(r.run)
        qkeys.append(((ro, r.query_id), (r.run, r.query_id)))
        methods |= set(str(r.methods).split("|")) - {""}
        blocks |= set(str(r.block_types).split("|")) - {""}
        sectors.add(idx.query_sector.get(r.query_id, ""))
        if r.local_pack_match_domain:
            lp_domains.append(((ro, r.query_id), r.local_pack_match_domain))
    ordered = [v for _, v in sorted(set(qkeys))]
    segs, seen = [], set()
    for run, qid in ordered:
        for si, seg in enumerate(idx.segments.get((run, qid), [])):
            if re.search(rf"(?<![a-z0-9]){re.escape(norm)}(?![a-z0-9])", entity_norm(seg)):
                ex = _segment_excerpt(seg, item.get("example_raw") or norm)
                if ex not in seen:
                    seen.add(ex)
                    segs.append(ex)
            if len(segs) >= MAX_SEGMENTS:
                break
        if len(segs) >= MAX_SEGMENTS:
            break
    place = None
    for run, qid in ordered:
        plist = idx.places.get((run, qid), [])
        p = extract.match_local_pack(norm, plist)
        if p is not None:
            place = {k: v for k, v in p.items() if k != "entity_norm"}
            break
    qids = []
    for _, qid in ordered:
        if qid not in qids:
            qids.append(qid)
    ev = {
        "key": norm, "key_type": "entity", "name": clip(item.get("example_raw") or norm, TITLE_CHARS),
        "sectors": sorted(s for s in sectors if s), "surfaces": ["aio_mention"],
        "queries": [idx.query_entry(qid) for qid in qids[:MAX_QUERIES]],
        "aio_block_types": sorted(blocks), "detection_methods": sorted(methods),
        "aio_segments": segs, "local_pack": [place] if place else [],
    }
    return _trim(ev, ("aio_segments", "queries"))


def build_evidence(items: list[dict], idx: EvidenceIndex, mentions, runs, fetch_dir: Path | None = None,
                   fetch_enabled: bool = True) -> list[dict]:
    """items.jsonl records. A domain without a cached fetch has `hash_final=False` while fetching is enabled."""
    out = []
    for it in items:
        meta = {"count": it["count"], "n_queries": it["n_queries"]}
        hash_final = True
        if it["key_type"] == "domain":
            rec = read_fetch_cache(it["key"], fetch_dir)
            if rec is None and fetch_enabled:
                hash_final = False
            if rec is not None:
                meta["fetch"] = rec.get("meta", {})
            ev = domain_evidence(it, idx, rec)
        else:
            ev = entity_evidence(it, idx, mentions)
        out.append({"item_id": it["item_id"], "key_type": it["key_type"], "key": it["key"], "evidence": ev,
                    "evidence_sha": sha12(ev), "meta": meta, "rule_categories": list(it["rule_categories"]),
                    "runs": list(runs), "hash_final": hash_final,
                    "example_url": normalize_url(it.get("example_url")) or it.get("example_url") or "",
                    "example_title": it.get("example_title") or "", "sectors": it.get("sectors") or "",
                    "surfaces": it.get("surfaces") or ""})
    return out


def items_path() -> Path:
    return config.llm_dir() / "items.jsonl"


def write_items(records: list[dict], path: Path | None = None) -> Path:
    return atomic_write_text(path or items_path(), "".join(canonical(r) + "\n" for r in records))


def read_items(path: Path | None = None) -> list[dict]:
    p = path or items_path()
    if not p.exists():
        return []
    out = []
    with p.open(encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                try:
                    out.append(json.loads(ln))
                except ValueError:
                    continue
    return out


# --- Homepage fetch --------------------------------------------------------------------

def fetch_dir() -> Path:
    return config.llm_dir() / "fetch"


def fetch_cache_path(domain: str, base: Path | None = None) -> Path:
    safe = re.sub(r"[^a-z0-9.-]", "_", domain.lower())[:200] or "_"
    return (base or fetch_dir()) / f"{safe}.json"


def read_fetch_cache(domain: str, base: Path | None = None) -> dict | None:
    p = fetch_cache_path(domain, base)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except ValueError:
        return None


class FetchRefused(Exception):
    pass


def _check_public(host: str, resolver: Callable) -> None:
    try:
        infos = resolver(host, 443, 0, socket.SOCK_STREAM)
    except (OSError, UnicodeError) as exc:
        raise FetchRefused(f"DNS lookup failed for {host}: {exc}")
    addrs = {info[4][0] for info in infos}
    if not addrs:
        raise FetchRefused(f"no address for {host}")
    for a in addrs:
        ip = ipaddress.ip_address(str(a).split("%")[0])
        mapped = getattr(ip, "ipv4_mapped", None)
        ip = mapped or ip
        # is_global also rejects ranges that are not is_private on Python 3.9 (e.g. 100.64.0.0/10 CGNAT).
        if not ip.is_global or ip.is_multicast:
            raise FetchRefused(f"{host} resolves to a non-public address")


def parse_html(body: bytes, final_host: str) -> dict:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(body, "lxml")
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    desc = soup.find("meta", attrs={"name": re.compile(r"^description$", re.I)})
    og = soup.find("meta", attrs={"property": re.compile(r"^og:site_name$", re.I)})
    desc_text = desc.get("content") if desc else ""
    og_text = og.get("content") if og else ""
    for t in soup(["script", "style", "noscript", "svg", "template", "head"]):
        t.decompose()
    text = " ".join(soup.get_text(" ").split())
    return {"title": clip(title, TITLE_CHARS), "meta_description": clip(desc_text, 300),
            "og_site_name": clip(og_text, 100),
            "text_excerpt": clip(text, config.FETCH_EXCERPT_CHARS), "final_host": final_host}


def fetch_homepage(domain: str, host: str, session, resolver: Callable = socket.getaddrinfo,
                   base: Path | None = None, now: Callable = cache.utc_now) -> dict:
    """One GET of https://<host>/ plus at most FETCH_MAX_REDIRECTS manual redirect hops; no retries.
    Every hop's host must resolve to public addresses only. The result (or the failure) is cached."""
    url = f"https://{host}/"
    chain, status, error, content = [], None, "", None
    try:
        for hop in range(config.FETCH_MAX_REDIRECTS + 1):
            parts = urlsplit(url)
            if parts.scheme not in ("http", "https") or not parts.hostname:
                raise FetchRefused(f"unsupported redirect target {url[:100]}")
            _check_public(parts.hostname, resolver)
            chain.append(url)
            resp = session.get(url, headers={"User-Agent": config.FETCH_USER_AGENT,
                                             "Accept": "text/html,application/xhtml+xml"},
                               timeout=config.FETCH_TIMEOUT, allow_redirects=False, stream=True)
            try:
                status = resp.status_code
                if status in REDIRECT_CODES and (resp.headers or {}).get("Location"):
                    if hop >= config.FETCH_MAX_REDIRECTS:
                        raise FetchRefused(f"more than {config.FETCH_MAX_REDIRECTS} redirects")
                    url = urljoin(url, resp.headers["Location"])
                    continue
                if status >= 400:
                    raise FetchRefused(f"HTTP {status}")
                ctype = str((resp.headers or {}).get("Content-Type", "")).lower()
                if "text/html" not in ctype and "application/xhtml" not in ctype:
                    raise FetchRefused(f"not HTML ({ctype[:60] or 'no content type'})")
                body, truncated = b"", False
                for chunk in resp.iter_content(65536):
                    body += chunk or b""
                    if len(body) > config.FETCH_MAX_BYTES:
                        body, truncated = body[: config.FETCH_MAX_BYTES], True
                        break
                content = parse_html(body, parts.hostname)
                if truncated:
                    error = "body truncated at 1 MB"
                break
            finally:
                close = getattr(resp, "close", None)
                if close:
                    close()
    except FetchRefused as exc:
        error = str(exc)
    except Exception as exc:  # timeouts, TLS and connection errors: recorded, never fatal
        error = f"{type(exc).__name__}: {str(exc)[:200]}"
    rec = {"domain": domain, "content": content if content is not None else {"available": False},
           "meta": {"fetched_at": now(), "status": status, "error": error, "redirects": chain}}
    atomic_write_text(fetch_cache_path(domain, base), json.dumps(rec, ensure_ascii=False, indent=1))
    return rec


def fetch_items(items: list[dict], idx: EvidenceIndex, session, resolver: Callable = socket.getaddrinfo,
                refetch: bool = False, sleep: Callable = None, log: Callable = print,
                base: Path | None = None) -> dict:
    """Fetch the homepages of the given (already --limit-ed) domain items that have no cached fetch."""
    import time
    sleep = sleep or time.sleep
    done = {"fetched": 0, "cached": 0, "failed": 0}
    first = True
    for it in items:
        if it["key_type"] != "domain":
            continue
        if not refetch and read_fetch_cache(it["key"], base) is not None:
            done["cached"] += 1
            continue
        if not first:
            sleep(config.FETCH_SLEEP_S)
        first = False
        rec = fetch_homepage(it["key"], target_host(idx, it["key"]), session, resolver, base)
        done["fetched"] += 1
        if rec["meta"]["error"] and rec["content"].get("available") is False:
            done["failed"] += 1
        if done["fetched"] % 25 == 0:
            log(f"  fetched {done['fetched']} homepages ...")
    return done
