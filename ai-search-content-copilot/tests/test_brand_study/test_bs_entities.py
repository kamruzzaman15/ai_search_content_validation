from __future__ import annotations

import pytest

from brand_study.entities import AliasIndex, entity_norm, entity_norm_cased, find_aliases


@pytest.fixture
def index():
    return AliasIndex.from_names([
        ("Target", "Target", "case_mid"), ("CVS", "CVS", "case"), ("The Ordinary", "The Ordinary", "case"),
        ("illy", "illy", "candidate"), ("illy coffee", "illy", "no"), ("La Roche-Posay", "La Roche-Posay", "no"),
        ("Roche", "Roche", "no"), ("Dunkin'", "Dunkin'", "no"), ("Clean & Clear", "Clean & Clear", "case"),
    ])


def found(index, text, kind="sentence"):
    return find_aliases([(text, kind)], index)


def test_normalisation_rules():
    assert entity_norm("Café Bustelo") == "cafe bustelo"
    assert entity_norm("Dunkin’") == entity_norm("DUNKIN") == "dunkin"
    assert entity_norm("Kiehl's") == "kiehls"
    assert entity_norm("Clean & Clear") == "clean and clear"
    assert entity_norm("A+B Labs") == "a and b labs"
    assert entity_norm("Death Wish Coffee Co.") == "death wish coffee"
    assert entity_norm("Acme Inc LLC") == "acme"
    assert entity_norm("The Ordinary") == "the ordinary"
    assert entity_norm_cased("Clean & Clear!") == "Clean and Clear"


def test_target_ambiguity(index):
    assert found(index, "target dark spots") == {}
    assert found(index, "Target dark spots with vitamin C.") == {}
    assert found(index, "available at Target and CVS") == {"Target": {"dictionary"}, "CVS": {"dictionary"}}
    assert found(index, "Target", kind="list_item") == {}


def test_ordinary_ambiguity(index):
    assert found(index, "an ordinary moisturizer") == {}
    assert found(index, "The Ordinary Niacinamide") == {"The Ordinary": {"dictionary"}}


def test_illy_candidate(index):
    assert found(index, "silly") == {}
    assert found(index, "illy") == {"illy": {"candidate"}}
    assert found(index, "Try illy coffee today") == {"illy": {"dictionary"}}


def test_longest_match_first(index):
    assert found(index, "La Roche-Posay toleriane") == {"La Roche-Posay": {"dictionary"}}
    assert found(index, "Roche alone") == {"Roche": {"dictionary"}}


def test_apostrophes_and_ampersand(index):
    assert found(index, "Grab a DUNKIN' latte. Or Clean & Clear.") == {"Dunkin'": {"dictionary"},
                                                                        "Clean & Clear": {"dictionary"}}


def test_invalid_mode():
    with pytest.raises(ValueError):
        AliasIndex.from_names([("x", "x", "maybe")])


def test_abbreviation_does_not_start_a_sentence(index):
    assert "Target" in found(index, "Big-box stores, e.g. Target and CVS, stock it.", "paragraph")
    assert "Target" in found(index, "Walmart vs. Target: which is cheaper?", "paragraph")
    assert "Target" not in found(index, "Use vitamin C daily. Target dark spots at night.", "paragraph")
