# -*- coding: utf-8 -*-
"""
Tests for SerbianSpellChecker — corrections persistence, lookup, and clearing.
These tests do not require phunspell to be installed.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import tempfile
from spell_checker import SerbianSpellChecker


@pytest.fixture
def tmp_corrections_file(tmp_path):
    return str(tmp_path / "korekcija.txt")


@pytest.fixture
def checker(tmp_corrections_file):
    return SerbianSpellChecker(spell_corrections_file=tmp_corrections_file)


def test_add_and_retrieve_correction(checker):
    checker.add_user_correction("doslo", "došlo", rating=3)
    assert checker.get_user_correction("doslo") == "došlo"


def test_lookup_is_case_insensitive(checker):
    checker.add_user_correction("Doslo", "došlo")
    assert checker.get_user_correction("DOSLO") == "došlo"
    assert checker.get_user_correction("doslo") == "došlo"


def test_unknown_word_returns_none(checker):
    assert checker.get_user_correction("nepostoji") is None


def test_correction_persisted_to_file(tmp_corrections_file):
    c1 = SerbianSpellChecker(spell_corrections_file=tmp_corrections_file)
    c1.add_user_correction("strasn", "strašan", rating=2)

    c2 = SerbianSpellChecker(spell_corrections_file=tmp_corrections_file)
    assert c2.get_user_correction("strasn") == "strašan"


def test_add_correction_increments_count(checker):
    checker.add_user_correction("uzasan", "užasan")
    checker.add_user_correction("uzasan", "užasan")
    entry = checker.user_corrections["uzasan"]
    assert entry[2] == 2   # count is the third element of the tuple


def test_add_correction_keeps_higher_rating(checker):
    checker.add_user_correction("rec", "reč", rating=1)
    checker.add_user_correction("rec", "reč", rating=4)
    assert checker.user_corrections["rec"][1] == 4   # rating


def test_get_all_corrections_shape(checker):
    checker.add_user_correction("doslo", "došlo", rating=2)
    results = checker.get_all_corrections()
    assert len(results) == 1
    keys = set(results[0].keys())
    assert keys == {'original', 'corrected', 'rating', 'count', 'last_used'}


def test_clear_corrections(checker, tmp_corrections_file):
    checker.add_user_correction("doslo", "došlo")
    assert checker.clear_corrections() == 1
    assert checker.user_corrections == {}
    assert not os.path.exists(tmp_corrections_file)


def test_get_user_correction_does_not_mutate_count(checker):
    checker.add_user_correction("rec", "reč")
    count_before = checker.user_corrections["rec"][2]
    checker.get_user_correction("rec")
    checker.get_user_correction("rec")
    count_after = checker.user_corrections["rec"][2]
    assert count_before == count_after