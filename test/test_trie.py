# -*- coding: utf-8 -*-
"""
Tests for PrefixTrie — the O(word_length) dictionary lookup structure.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from text_analyzer import PrefixTrie


@pytest.fixture
def trie():
    t = PrefixTrie()
    for term in ("test", "teror", "terorizam", "bomba", "nasilje"):
        t.insert(term)
    return t


def test_exact_match(trie):
    assert trie.find_prefix_match("bomba") == "bomba"


def test_prefix_match_returns_longest(trie):
    # "terorizam" is longer than "teror" — longest match wins
    assert trie.find_prefix_match("terorizam") == "terorizam"


def test_prefix_match_shorter_term(trie):
    # "terorist" starts with "teror" but "terorizam" doesn't match fully
    assert trie.find_prefix_match("terorist") == "teror"


def test_no_match_returns_none(trie):
    assert trie.find_prefix_match("miran") is None


def test_empty_string(trie):
    assert trie.find_prefix_match("") is None


def test_single_char_no_match(trie):
    assert trie.find_prefix_match("t") is None


def test_insert_and_find_unicode():
    t = PrefixTrie()
    t.insert("mržnja")
    assert t.find_prefix_match("mržnja") == "mržnja"
    assert t.find_prefix_match("mržnjom") == "mržnja"


def test_overlapping_terms():
    t = PrefixTrie()
    t.insert("na")
    t.insert("napad")
    # "napad" is longer — should win
    assert t.find_prefix_match("napadač") == "napad"
    # "na" wins when "napad" doesn't match further
    assert t.find_prefix_match("nasilje") == "na"


def test_empty_trie():
    t = PrefixTrie()
    assert t.find_prefix_match("anything") is None