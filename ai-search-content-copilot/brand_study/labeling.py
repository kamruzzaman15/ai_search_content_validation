from __future__ import annotations
import csv
import re
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from brand_study import config
from brand_study.config import ConfigError
from brand_study.domains import host as url_host
from brand_study.domains import is_platform_hosted, root_domain
from brand_study.entities import AMBIGUITY_MODES, AliasIndex, entity_norm

RULE_COLUMNS = ("match_type", "pattern", "entity", "category", "sector", "subtype", "basis", "ambiguous", "notes")
NOT_A_BRAND_HINT = re.compile(
    r"\b(acid|serum|serums|cream|creams|lotion|cleanser|moisturizer|sunscreen|spf|vitamin|retinol|niacinamide|"
    r"ceramide|peptide|toner|mask|roast|brew|espresso|latte|beans|blend|decaf|skin|wifi|seating|price|parking)\b")


@dataclass(frozen=True)
class Rule:
    match_type: str
    pattern: str
    entity: str
    category: str
    sector: str
    subtype: str
    basis: str
    ambiguous: str


def _read_rows(path: Path) -> list[tuple[int, dict]]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        lines = [ln for ln in f]
    out = []
    reader = csv.DictReader([ln for ln in lines if not ln.lstrip().startswith("#")])
    # line numbers: map back to the original file, skipping comment lines
    numbers = [i for i, ln in enumerate(lines, start=1) if not ln.lstrip().startswith("#")][1:]
    for n, row in zip(numbers, reader):
        out.append((n, row))
    return out


class Labeler:
    def __init__(self, rules: list[Rule], overrides: dict[tuple[str, str], str] | None = None):
        self.rules = rules
        self.overrides = overrides or {}
        self.domain_rules: dict[str, list[Rule]] = defaultdict(list)
        self.name_rules: dict[str, list[Rule]] = defaultdict(list)
        alias_rows = []
        for r in rules:
            if r.match_type == "domain":
                self.domain_rules[url_host(r.pattern) or r.pattern.lower()].append(r)
            else:
                keys = {entity_norm(r.pattern)} | ({entity_norm(r.entity)} if r.entity else set())
                for k in keys:
                    self.name_rules[k].append(r)
                if r.category != "not_a_brand":
                    alias_rows.append((r.sector or "any", (r.pattern, r.entity or r.pattern, r.ambiguous)))
        self.alias_index = AliasIndex.from_names([row for _, row in alias_rows])
        self._alias_rows = alias_rows
        self._sector_indexes: dict[str, AliasIndex] = {}

    def alias_index_for(self, sector: str | None) -> AliasIndex:
        """Aliases whose rule applies in this sector (sector-specific + 'any'), so a dictionary
        match always has a rule that label_entity() will also pick."""
        if not sector:
            return self.alias_index
        if sector not in self._sector_indexes:
            self._sector_indexes[sector] = AliasIndex.from_names(
                [row for s, row in self._alias_rows if s in ("any", sector)])
        return self._sector_indexes[sector]

    @staticmethod
    def _pick(rules: list[Rule], sector: str | None) -> Rule | None:
        for r in rules:
            if r.sector in ("any", "") or sector is None or r.sector == sector:
                return r
        return None

    def label_domain(self, host: str | None, sector: str | None = None) -> tuple[str, str, str]:
        h = (host or "").lower()
        if not h:
            return "unknown", "", "none"
        rd = root_domain(h)
        cat = self.overrides.get(("domain", rd)) or self.overrides.get(("domain", h))
        if cat:
            return cat, "", "override"
        r = self._pick(self.domain_rules.get(h, []), sector)
        if r:
            return r.category, r.entity, "domain_rule"
        labels = h.split(".")
        for i in range(1, len(labels) - 1):
            r = self._pick(self.domain_rules.get(".".join(labels[i:]), []), sector)
            if r:
                return r.category, r.entity, "parent_domain_rule"
        if rd != h:
            r = self._pick(self.domain_rules.get(rd, []), sector)
            if r:
                return r.category, r.entity, "parent_domain_rule"
        if is_platform_hosted(h):
            return "small_local", "", "platform_hosted"
        if h.endswith((".gov", ".edu")) or ".gov." in h or ".edu." in h:
            return "directory_media_other", "", "tld"
        return "unknown", "", "none"

    def label_entity(self, name: str | None, sector: str | None = None) -> tuple[str, str, str]:
        n = entity_norm(name or "")
        if not n:
            return "unknown", "", "none"
        cat = self.overrides.get(("entity", n))
        if cat:
            return cat, name or "", "override"
        r = self._pick(self.name_rules.get(n, []), sector)
        if r:
            return r.category, r.entity or name or "", "name_rule"
        return "unknown", name or "", "none"

    def label(self, root_domain: str | None = None, entity_norm: str | None = None, sector: str | None = None,
              host: str | None = None) -> tuple[str, str, str]:
        if entity_norm:
            return self.label_entity(entity_norm, sector)
        return self.label_domain(host or root_domain, sector)


def load_rules(path: Path | str = config.DEFAULT_RULES) -> list[Rule]:
    rules, errors = [], []
    for n, row in _read_rows(Path(path)):
        r = Rule(*[(row.get(c) or "").strip() for c in RULE_COLUMNS[:-1]])
        r = Rule(r.match_type, r.pattern, r.entity, r.category, r.sector or "any", r.subtype, r.basis,
                 r.ambiguous or "no")
        if r.match_type not in ("domain", "name"):
            errors.append(f"line {n}: invalid match_type {r.match_type!r}")
        if not r.pattern:
            errors.append(f"line {n}: empty pattern")
        if r.category not in config.CATEGORIES:
            errors.append(f"line {n}: invalid category {r.category!r}")
        if r.sector not in ("any",) + config.SECTORS:
            errors.append(f"line {n}: invalid sector {r.sector!r}")
        if r.ambiguous not in AMBIGUITY_MODES:
            errors.append(f"line {n}: invalid ambiguous {r.ambiguous!r}")
        rules.append(r)
    if errors:
        raise ConfigError(f"{Path(path).name} invalid:\n  " + "\n  ".join(errors))
    return rules


def load_overrides(path: Path | str | None = config.DEFAULT_OVERRIDES) -> dict[tuple[str, str], str]:
    if path is None or not Path(path).exists():
        return {}
    out: dict[tuple[str, str], str] = {}
    errors = []
    for n, row in _read_rows(Path(path)):
        kt = (row.get("key_type") or "").strip()
        key = (row.get("key") or "").strip()
        cat = (row.get("category") or "").strip()
        if kt not in ("domain", "entity"):
            errors.append(f"line {n}: invalid key_type {kt!r}")
            continue
        if not key:
            errors.append(f"line {n}: empty key")
            continue
        if cat not in config.CATEGORIES:
            errors.append(f"line {n}: invalid category {cat!r}")
            continue
        norm = root_domain(url_host(key)) if kt == "domain" else entity_norm(key)
        if not norm:
            errors.append(f"line {n}: key {key!r} normalises to empty")
            continue
        if (kt, norm) in out:
            warnings.warn(f"{Path(path).name} line {n}: duplicate override for {kt} {norm!r}; last row wins")
        out[(kt, norm)] = cat
    if errors:
        raise ConfigError(f"{Path(path).name} invalid:\n  " + "\n  ".join(errors))
    return out


# --- Applying labels to tidy frames ----------------------------------------------------

def _sector_map(status: pd.DataFrame) -> dict[tuple[str, str], str]:
    if status is None or status.empty:
        return {}
    return {(r.run, r.query_id): r.sector for r in status.itertuples()}


def label_domain_frame(df: pd.DataFrame, labeler: Labeler, sectors: dict) -> pd.DataFrame:
    df = df.copy()
    if df.empty:
        for c in ("entity", "category", "label_source"):
            df[c] = pd.Series(dtype=object)
        return df
    out = [labeler.label_domain(getattr(r, "host", "") or r.root_domain, sectors.get((r.run, r.query_id)))
           for r in df.itertuples()]
    df["category"] = [o[0] for o in out]
    df["entity"] = [o[1] for o in out]
    df["label_source"] = [o[2] for o in out]
    return df


def label_mentions(mentions: pd.DataFrame, labeler: Labeler, sectors: dict) -> pd.DataFrame:
    m = mentions.copy()
    if m.empty:
        for c in ("confirmed", "category", "label_source"):
            m[c] = pd.Series(dtype=object)
        return m
    confirmed, cats, sources = [], [], []
    for r in m.itertuples():
        methods = set(str(r.methods).split("|"))
        sector = sectors.get((r.run, r.query_id))
        cat, _, src = labeler.label_entity(r.entity, sector)
        # A candidate-mode alias is itself a name rule, so only an override may confirm a candidate-only hit.
        rule_ok = src == "override" or (src == "name_rule" and bool(methods - {"candidate"}))
        is_conf = bool(methods & {"dictionary", "local_pack_name"}) or rule_ok
        if is_conf and cat == "unknown" and "local_pack_name" in methods and r.local_pack_match_domain:
            dcat, _, dsrc = labeler.label_domain(r.local_pack_match_domain, sector)
            if dcat != "unknown":
                cat, src = dcat, f"local_pack_website:{dsrc}"
        if not is_conf:
            cat, src = "unknown", "unconfirmed"
        confirmed.append(is_conf)
        cats.append(cat)
        sources.append(src)
    m["confirmed"] = confirmed
    m["category"] = cats
    m["label_source"] = sources
    return m


def build_review_table(organic: pd.DataFrame, refs: pd.DataFrame, local: pd.DataFrame, mentions: pd.DataFrame,
                       status: pd.DataFrame) -> pd.DataFrame:
    sectors = _sector_map(status)
    lp_domains = set(local.loc[local["root_domain"].astype(str) != "", "root_domain"]) if not local.empty else set()
    acc: dict[tuple[str, str], dict] = {}

    def add(kt, key, raw, run, qid, surface, url="", title="", suggestion=("", "")):
        if not key:
            return
        e = acc.setdefault((kt, key), {"key_type": kt, "key": key, "example_raw": raw, "count": 0, "queries": set(),
                                       "sectors": set(), "surfaces": set(), "example_url": url,
                                       "example_title": title, "suggested_category": "", "suggest_reason": ""})
        e["count"] += 1
        e["queries"].add((run, qid))
        e["sectors"].add(sectors.get((run, qid), ""))
        e["surfaces"].add(surface)
        if suggestion[0] and not e["suggested_category"]:
            e["suggested_category"], e["suggest_reason"] = suggestion

    for surface, df in (("organic", organic), ("aio_reference", refs), ("local_pack", local)):
        if df.empty or "category" not in df:
            continue
        for r in df[df["category"] == "unknown"].itertuples():
            sug = ("small_local", "local_pack_website") if r.root_domain in lp_domains else ("", "")
            add("domain", r.root_domain, getattr(r, "url", "") or getattr(r, "website", ""), r.run, r.query_id,
                surface, getattr(r, "url", "") or getattr(r, "website", ""), getattr(r, "title", ""), sug)
    if not mentions.empty and "category" in mentions:
        for r in mentions[mentions["category"] == "unknown"].itertuples():
            if r.confirmed and "local_pack_name" in str(r.methods):
                sug = ("small_local", "local_pack_name")
            elif "shopping_result" not in str(r.block_types) and NOT_A_BRAND_HINT.search(r.entity_norm):
                # product names ("... Gel Moisturizer") contain these words but are brands, so no hint for them
                sug = ("not_a_brand", "not_a_brand_pattern")
            else:
                sug = ("", "")
            add("entity", r.entity_norm, r.entity, r.run, r.query_id, "aio_mention", "", "", sug)

    rows = []
    for e in acc.values():
        e["n_queries"] = len(e.pop("queries"))
        e["sectors"] = "|".join(sorted(s for s in e["sectors"] if s))
        e["surfaces"] = "|".join(sorted(e["surfaces"]))
        e["category"] = ""
        rows.append(e)
    cols = ["key_type", "key", "example_raw", "count", "n_queries", "sectors", "surfaces", "example_url",
            "example_title", "suggested_category", "suggest_reason", "category"]
    out = pd.DataFrame(rows, columns=cols)
    if out.empty:
        return out
    out["_has_sug"] = out["suggested_category"] != ""
    out = out.sort_values(["count", "_has_sug", "key"], ascending=[False, False, True]).drop(columns="_has_sug")
    return out.reset_index(drop=True)
