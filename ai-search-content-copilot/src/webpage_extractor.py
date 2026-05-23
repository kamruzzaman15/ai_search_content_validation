import json
import re
import requests
import trafilatura
from bs4 import BeautifulSoup


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}


def _fetch_with_playwright(url: str, timeout: int = 30) -> str:
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=timeout * 1000)
        page.wait_for_timeout(2000)  # let JS render
        html = page.content()
        browser.close()
    return html


def fetch_html(url: str, timeout: int = 30) -> str:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
        if resp.status_code == 403:
            raise requests.exceptions.HTTPError("403")
        resp.raise_for_status()
        return resp.text
    except (requests.exceptions.HTTPError, requests.exceptions.Timeout):
        return _fetch_with_playwright(url, timeout=timeout)


def _extract_json_ld(soup: BeautifulSoup) -> dict:
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string or "")
            if isinstance(data, dict):
                return data
            if isinstance(data, list) and data:
                return data[0]
        except (json.JSONDecodeError, TypeError):
            continue
    return {}


def _extract_headings(soup: BeautifulSoup) -> list[str]:
    headings = []
    for tag in soup.find_all(["h1", "h2", "h3"]):
        text = tag.get_text(strip=True)
        if text:
            headings.append(text)
    return headings


def _extract_bullets(soup: BeautifulSoup) -> list[str]:
    bullets = []
    for li in soup.find_all("li"):
        text = li.get_text(strip=True)
        if text and len(text) < 300:
            bullets.append(text)
    return list(dict.fromkeys(bullets))  # deduplicate while preserving order


def _extract_paragraphs(soup: BeautifulSoup) -> list[str]:
    paragraphs = []
    for p in soup.find_all("p"):
        text = p.get_text(strip=True)
        if len(text) > 40:
            paragraphs.append(text)
    return paragraphs


def _extract_tables(soup: BeautifulSoup) -> list[list[str]]:
    tables = []
    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
            if any(cells):
                rows.append(cells)
        if rows:
            tables.append(rows)
    return tables


def build_evidence_bundle(url: str) -> dict:
    html = fetch_html(url)
    soup = BeautifulSoup(html, "lxml")

    title_tag = soup.find("title")
    page_title = title_tag.get_text(strip=True) if title_tag else ""

    meta_desc_tag = soup.find("meta", attrs={"name": "description"})
    meta_description = ""
    if meta_desc_tag and meta_desc_tag.get("content"):
        meta_description = meta_desc_tag["content"].strip()

    raw_text = trafilatura.extract(html, include_tables=True, include_links=False) or ""

    bundle = {
        "source_url": url,
        "page_title": page_title,
        "meta_description": meta_description,
        "headings": _extract_headings(soup),
        "paragraphs": _extract_paragraphs(soup),
        "bullets": _extract_bullets(soup),
        "tables": _extract_tables(soup),
        "json_ld": _extract_json_ld(soup),
        "raw_extracted_text": raw_text,
    }
    return bundle
