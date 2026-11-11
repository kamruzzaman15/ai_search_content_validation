from __future__ import annotations

from brand_study.domains import host, normalize_url, root_domain, unwrap_google_redirect, url_root_domain


def test_strip_prefixes():
    assert host("https://www.yelp.com/biz/x") == "yelp.com"
    assert host("https://m.facebook.com/page") == "facebook.com"
    assert host("http://amp.example.com:8080/a") == "example.com"


def test_root_domain():
    assert url_root_domain("https://shop.starbucks.com/x") == "starbucks.com"
    assert root_domain("news.bbc.co.uk") == "bbc.co.uk"


def test_platform_hosts_kept_whole():
    assert url_root_domain("https://blackandbrew.square.site/") == "blackandbrew.square.site"
    assert url_root_domain("https://smallskin.myshopify.com/x") == "smallskin.myshopify.com"
    assert url_root_domain("https://sites.google.com/view/cafe") == "sites.google.com"


def test_redirect_unwrap():
    assert unwrap_google_redirect("https://www.google.com/url?q=https://indiecafe.com/&sa=U") == "https://indiecafe.com/"
    assert host("https://www.google.com/url?url=https://www.indiecafe.com/menu") == "indiecafe.com"


def test_normalize_url_drops_fragment_and_tracking():
    u = normalize_url("HTTPS://WWW.Example.com/a/?utm_source=x&b=1&gclid=2#:~:text=hello")
    assert u == "https://www.example.com/a?b=1"


def test_garbage():
    for bad in (None, "", "not a url", "http://", 42, "https://[bad"):
        assert host(bad) == ""
    assert normalize_url("nonsense") == ""


def test_idn():
    assert host("https://café.example.com/").startswith("xn--")


def test_google_shopping_and_maps_are_own_units():
    assert host("https://www.google.com/search?ibp=oshop&prds=1") == "shopping.google.com"
    assert root_domain("shopping.google.com") == "shopping.google.com"
    assert root_domain(host("https://maps.google.com/?cid=1")) == "maps.google.com"
    assert url_root_domain("https://www.google.com/search?q=coffee") == "google.com"
