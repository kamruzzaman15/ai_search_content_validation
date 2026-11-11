from __future__ import annotations
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

MULTI_PART_SUFFIXES = {
    "co.uk", "org.uk", "ac.uk", "gov.uk", "com.au", "net.au", "org.au", "co.nz", "co.jp",
    "com.br", "com.mx", "co.in", "co.za", "com.sg", "com.tr", "com.cn",
}
# Hosting platforms where each subdomain is a separate business: keep the full host.
PLATFORM_HOSTS = (
    "square.site", "myshopify.com", "wixsite.com", "godaddysites.com", "business.site",
    "wordpress.com", "blogspot.com", "squarespace.com", "weebly.com", "toasttab.com",
    "sites.google.com", "linktr.ee",
)
# Google surfaces that are counted as their own unit instead of collapsing into google.com.
KEEP_WHOLE_HOSTS = {"shopping.google.com", "maps.google.com"}
STRIP_PREFIXES = ("www.", "m.", "amp.")
TRACKING_PARAMS = {"gclid", "srsltid", "fbclid", "msclkid"}


def unwrap_google_redirect(url: str) -> str:
    try:
        parts = urlsplit(url)
    except ValueError:
        return url
    hostname = (parts.hostname or "").lower()
    if (hostname.endswith("google.com") or hostname.startswith("google.")) and parts.path == "/url":
        params = dict(parse_qsl(parts.query))
        target = params.get("q") or params.get("url")
        if target and target.startswith(("http://", "https://")):
            return target
    return url


def normalize_url(url: str | None) -> str:
    if not url or not isinstance(url, str):
        return ""
    url = unwrap_google_redirect(url.strip())
    try:
        parts = urlsplit(url)
    except ValueError:
        return ""
    if not parts.netloc:
        return ""
    query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
             if not k.lower().startswith("utm_") and k.lower() not in TRACKING_PARAMS]
    path = parts.path.rstrip("/")
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, urlencode(query), ""))


def host(url: str | None) -> str:
    if not url or not isinstance(url, str):
        return ""
    url = unwrap_google_redirect(url.strip())
    if "://" not in url:
        url = "http://" + url
    try:
        parts = urlsplit(url)
        h = parts.hostname or ""
    except ValueError:
        return ""
    h = h.strip(".").lower()
    if h in ("google.com", "www.google.com") and parts.path.startswith("/search") and \
            ("ibp=oshop" in parts.query or "tbm=shop" in parts.query):
        return "shopping.google.com"  # Google Shopping product panels cited by AI Overviews
    if not h or "." not in h or " " in h:
        return ""
    for prefix in STRIP_PREFIXES:
        if h.startswith(prefix) and h.count(".") >= 2:
            h = h[len(prefix):]
            break
    try:
        h = h.encode("idna").decode("ascii")
    except UnicodeError:
        return ""
    return h


def _platform_for(h: str) -> str | None:
    for p in PLATFORM_HOSTS:
        if h == p or h.endswith("." + p):
            return p
    return None


def is_platform_hosted(h: str) -> bool:
    p = _platform_for(h)
    return p is not None and h != p


def root_domain(h: str | None) -> str:
    if not h:
        return ""
    h = h.lower().strip(".")
    if _platform_for(h) or h in KEEP_WHOLE_HOSTS:
        return h
    labels = h.split(".")
    if len(labels) <= 2:
        return h
    if ".".join(labels[-2:]) in MULTI_PART_SUFFIXES:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


def url_root_domain(url: str | None) -> str:
    return root_domain(host(url))
