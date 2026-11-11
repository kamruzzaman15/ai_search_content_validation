from __future__ import annotations
import re
import unicodedata
from dataclasses import dataclass

AMBIGUITY_MODES = ("no", "case", "case_mid", "candidate")
CORPORATE_SUFFIXES = ("inc", "llc", "co", "corp", "ltd")
SEGMENT_KINDS = ("sentence", "list_item", "heading", "cell")

_QUOTES = str.maketrans({"‘": "'", "’": "'", "‛": "'", "ʼ": "'", "`": "'", "´": "'",
                         "“": '"', "”": '"'})
_AMP = re.compile(r"(?<=\w)\s*[&+]\s*(?=\w)")
_NON_ALNUM = re.compile(r"[^0-9A-Za-z]+")
# Don't split after common abbreviations ("e.g. Target", "St. Petersburg"): the next word would look like
# a sentence start and case_mid aliases would be missed there.
_ABBREVIATIONS = ("e.g", "i.e", "E.g", "I.e", "St", "Ft", "Mt", "Dr", "Mr", "Mrs", "Ms", "vs", "approx", "Ave", "Blvd")
_SENTENCE_SPLIT = re.compile("".join(rf"(?<!\b{re.escape(a)})" for a in _ABBREVIATIONS) + r"[.!?]\s+")


def _base(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.translate(_QUOTES).replace("'", "")
    return _AMP.sub(" and ", text)


def _collapse(text: str) -> str:
    return _NON_ALNUM.sub(" ", text).strip()


def entity_norm_cased(text: str) -> str:
    return _collapse(_base(text))


def entity_norm(text: str) -> str:
    out = _collapse(_base(text).casefold())
    tokens = out.split()
    while len(tokens) > 1 and tokens[-1] in CORPORATE_SUFFIXES:
        tokens.pop()
    return " ".join(tokens)


@dataclass(frozen=True)
class AliasEntry:
    alias: str          # normalised: entity_norm (no/candidate) or entity_norm_cased (case/case_mid)
    entity: str         # display name from the rule file
    mode: str = "no"


class AliasIndex:
    def __init__(self, entries: list[AliasEntry]):
        for e in entries:
            if e.mode not in AMBIGUITY_MODES:
                raise ValueError(f"invalid ambiguity mode {e.mode!r} for {e.entity!r}")
        self.entries = sorted([e for e in entries if e.alias], key=lambda e: (-len(e.alias), e.alias))
        self._compiled = [(e, self._compile(e)) for e in self.entries]

    @staticmethod
    def _compile(e: AliasEntry) -> re.Pattern:
        body = re.escape(e.alias).replace(r"\ ", r"\s+")
        flags = 0 if e.mode in ("case", "case_mid") else re.IGNORECASE
        return re.compile(rf"(?<![0-9A-Za-z]){body}(?![0-9A-Za-z])", flags)

    @classmethod
    def from_names(cls, rows: list[tuple[str, str, str]]) -> "AliasIndex":
        """rows: (raw alias, entity, mode)."""
        entries = []
        for raw, entity, mode in rows:
            mode = mode or "no"
            alias = entity_norm_cased(raw) if mode in ("case", "case_mid") else entity_norm(raw)
            entries.append(AliasEntry(alias, entity, mode))
        return cls(entries)

    def iter_compiled(self):
        return iter(self._compiled)


def split_sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_SPLIT.split(text or "") if s.strip()]


def find_aliases(segments: list[tuple[str, str]], index: AliasIndex) -> dict[str, set[str]]:
    """segments: (text, kind). Returns entity -> {"dictionary"|"candidate"}."""
    found: dict[str, set[str]] = {}
    for seg_text, _kind in segments:
        for piece in split_sentences(seg_text):
            text = entity_norm_cased(piece)
            if not text:
                continue
            taken: list[tuple[int, int]] = []
            for entry, pattern in index.iter_compiled():
                for m in pattern.finditer(text):
                    s, e = m.span()
                    if any(s < te and ts < e for ts, te in taken):
                        continue
                    if entry.mode == "case_mid" and s == 0:
                        continue
                    taken.append((s, e))
                    method = "candidate" if entry.mode == "candidate" else "dictionary"
                    found.setdefault(entry.entity, set()).add(method)
    return found
