from __future__ import annotations
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_DIR / "data"
DEFAULT_QUERIES = DATA_DIR / "queries_v1.csv"
DEFAULT_RULES = DATA_DIR / "brand_rules_v1.csv"
DEFAULT_OVERRIDES = DATA_DIR / "label_overrides.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "brand_study"


def output_dir() -> Path:
    return Path(os.environ.get("BRAND_STUDY_OUTPUT_DIR") or DEFAULT_OUTPUT_DIR)


def raw_dir() -> Path:
    return output_dir() / "raw"


def results_dir() -> Path:
    return output_dir() / "results"


def ledger_path() -> Path:
    return output_dir() / "ledger.jsonl"


def lock_path() -> Path:
    return output_dir() / ".collect.lock"


# --- SerpApi -----------------------------------------------------------------
SERPAPI_ENDPOINT = "https://serpapi.com/search.json"
ACCOUNT_ENDPOINT = "https://serpapi.com/account.json"
GL = "us"
HL = "en"
GOOGLE_DOMAIN = "google.com"
DEVICE = "desktop"
STATE_LOCATION = "Florida,United States"
# Canonical names from SerpApi's Locations API (see SERPAPI_NOTES.md).
CITY_LOCATIONS = {
    "Miami": "Miami,Florida,United States",
    "Tampa": "Tampa,Florida,United States",
    "Orlando": "Orlando,Florida,United States",
    "Jacksonville": "Jacksonville,Florida,United States",
    "St. Petersburg": "St. Petersburg,Florida,United States",
    "Fort Lauderdale": "Fort Lauderdale,Florida,United States",
}

# --- Budget ------------------------------------------------------------------
DEFAULT_STUDY_CAP = int(os.environ.get("BRAND_STUDY_CAP", "300"))
DEFAULT_FOLLOWUP_RATE = 1.0
DEFAULT_RETRIES = 1
DEFAULT_RETRY_HEADROOM = 10
DEFAULT_SLEEP_S = 1.0
DEFAULT_MIN_HOURS_BETWEEN_RUNS = 12.0

# --- Runs --------------------------------------------------------------------
DEFAULT_RUNS = ("run1", "run2")
RUN_LABEL_RE = r"^[a-z0-9_-]{1,32}$"

# --- Categories --------------------------------------------------------------
CATEGORIES = (
    "big_brand",
    "small_local",
    "marketplace_retailer",
    "directory_media_other",
    "not_a_brand",
    "unknown",
)
HEADLINE_EXCLUDED = ("not_a_brand",)
SECTORS = ("coffee", "skincare")
QUERY_TYPES = ("broad", "local")

CATEGORY_DEFINITIONS = {
    "big_brand": (
        "National or global chains and brands, including national service chains "
        "(subtype national_chain). Operational test: big if ANY of (a) ownership - majority-owned "
        "by a publicly traded company or multinational group; (b) distribution - sold nationally "
        "through major US retailers (Walmart, Target, Costco, Ulta, Sephora, CVS, Walgreens, national "
        "grocery chains); (c) footprint - a service/store chain with locations in several US states "
        "or national franchising."
    ),
    "small_local": (
        "Small or independent businesses: none of (a)-(c) holds. For local queries: independent local "
        "businesses, including Florida-only/regional chains (subtype regional_chain). For broad product "
        "queries: small or independent brands. Results are always split by query_type."
    ),
    "marketplace_retailer": "Marketplaces and retailers: Amazon, Walmart, Sephora, Ulta, delivery apps and similar.",
    "directory_media_other": (
        "Directories, review sites, social media, publishers, and government/education sites "
        "(Yelp, TripAdvisor, Reddit, Instagram, healthline, .gov, .edu, ...)."
    ),
    "not_a_brand": "Ingredients, features and attributes. Excluded from headline numerators and denominators.",
    "unknown": "Not yet labeled. Never treated as small; always shown next to big and small, and in the bounds.",
}

# --- Analysis ----------------------------------------------------------------
DEFAULT_PROVISIONAL_THRESHOLD = 0.10

# --- Ollama (Part 2) ---------------------------------------------------------
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODELS_DIR = os.environ.get("BRAND_STUDY_OLLAMA_MODELS_DIR")
# Unverified tag: confirm with `ollama search`/the Ollama library before Part 2.
REWRITER_MODEL = os.environ.get("STUDY_REWRITER_MODEL", "gemma4:31b")
# PLACEHOLDER: set the real Qwen 3.x 27B Ollama tag (env STUDY_JUDGE_MODEL).
JUDGE_MODEL = os.environ.get("STUDY_JUDGE_MODEL", "REPLACE_ME_qwen3_27b_tag")


def is_placeholder_tag(tag: str) -> bool:
    return (tag or "").startswith("REPLACE_ME")


class ConfigError(Exception):
    """Configuration/validation problem; the CLI maps it to exit code 3."""


def get_api_key(required: bool = True) -> str | None:
    key = os.environ.get("SERPAPI_API_KEY", "").strip()
    if not key:
        if required:
            raise ConfigError("SERPAPI_API_KEY not set (env or .env)")
        return None
    return key
