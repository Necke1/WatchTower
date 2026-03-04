# -*- coding: utf-8 -*-
"""
Tests for TextAnalyzer — weight validation, emoji regex, risk levels,
script detection, and the full analyze_text pipeline.
These tests do not require CLASSLA or phunspell.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import tempfile
from text_analyzer import TextAnalyzer


@pytest.fixture
def tmp_dict(tmp_path):
    """Write a small test dictionary and return its path."""
    d = tmp_path / "recnik.txt"
    d.write_text(
        "# test dictionary\n"
        "terorizam 10\n"
        "bomba 8\n"
        "nasilje 5\n"
        "💣 7\n",
        encoding="utf-8"
    )
    return str(d)


@pytest.fixture
def analyzer(tmp_dict):
    return TextAnalyzer(
        dictionary_file=tmp_dict,
        use_spellcheck=False,
    )


# ------------------------------------------------------------------
# Weight validation (fix 1)
# ------------------------------------------------------------------

def test_negative_weight_skipped(tmp_path):
    d = tmp_path / "recnik.txt"
    d.write_text("bomba -3\nnasilje 5\n", encoding="utf-8")
    a = TextAnalyzer(dictionary_file=str(d), use_spellcheck=False)
    assert "bomba" not in a.words_set
    assert "nasilje" in a.words_set


def test_zero_weight_skipped(tmp_path):
    d = tmp_path / "recnik.txt"
    d.write_text("bomba 0\nnasilje 5\n", encoding="utf-8")
    a = TextAnalyzer(dictionary_file=str(d), use_spellcheck=False)
    assert "bomba" not in a.words_set


def test_valid_weight_loaded(tmp_dict, analyzer):
    assert analyzer.term_to_weight.get("terorizam") == 10
    assert analyzer.term_to_weight.get("bomba") == 8


# ------------------------------------------------------------------
# Emoji regex (fix 2)
# ------------------------------------------------------------------

def test_emoji_pattern_compiled(analyzer):
    assert analyzer._emoji_pattern is not None


def test_emoji_detected_in_text(analyzer):
    result = analyzer.analyze_text("Pažnja 💣 opasno!")
    freqs = result["analysis"]["term_frequencies"]
    assert "💣" in freqs


def test_emoji_count_correct(analyzer):
    result = analyzer.analyze_text("💣 💣 bomba 💣")
    freqs = result["analysis"]["term_frequencies"]
    assert freqs.get("💣", 0) == 3


# ------------------------------------------------------------------
# GPU flag (fix 3)
# ------------------------------------------------------------------

def test_use_gpu_stored(tmp_dict):
    a = TextAnalyzer(dictionary_file=tmp_dict, use_spellcheck=False, use_gpu=True)
    assert a.use_gpu is True


def test_use_gpu_default_false(tmp_dict):
    a = TextAnalyzer(dictionary_file=tmp_dict, use_spellcheck=False)
    assert a.use_gpu is False


# ------------------------------------------------------------------
# Script detection
# ------------------------------------------------------------------

def test_detect_latin(analyzer):
    assert analyzer.detect_script("Ovo je latinski tekst.") == "latin"


def test_detect_cyrillic(analyzer):
    assert analyzer.detect_script("Ово је ћирилички текст.") == "cyrillic"


def test_detect_mixed(analyzer):
    assert analyzer.detect_script("Ово је mixed tekst.") == "mixed"


def test_detect_no_alpha(analyzer):
    assert analyzer.detect_script("12345 !@#$%") == "latin"


# ------------------------------------------------------------------
# Risk level calculation
# ------------------------------------------------------------------

def test_no_risk(analyzer):
    result = analyzer.analyze_text("Danas je lep dan.")
    assert result["analysis"]["risk_level"] == "BEZ RIZIKA"
    assert result["analysis"]["total_score"] == 0


def test_low_risk(analyzer):
    result = analyzer.analyze_text("Nasilje je problem.")
    assert result["analysis"]["risk_level"] in ("NIZAK RIZIK", "MINIMALAN RIZIK")


def test_high_risk(analyzer):
    text = " ".join(["terorizam bomba nasilje"] * 10)
    result = analyzer.analyze_text(text)
    assert result["analysis"]["risk_level"] == "VISOK RIZIK"


# ------------------------------------------------------------------
# Result dict shape
# ------------------------------------------------------------------

def test_result_keys(analyzer):
    result = analyzer.analyze_text("Test tekst.")
    assert "original_text"  in result
    assert "processed_text" in result
    assert "spell_checking" in result
    assert "analysis"       in result
    assert "statistics"     in result
    assert "metadata"       in result


def test_metadata_contains_detected_script(analyzer):
    result = analyzer.analyze_text("Test.")
    assert "detected_script" in result["metadata"]


# ------------------------------------------------------------------
# Cyrillic input
# ------------------------------------------------------------------

def test_cyrillic_text_analyzed(analyzer):
    result = analyzer.analyze_text("Тероризам је проблем.")
    # After normalisation the term should be found
    assert result["analysis"]["total_score"] > 0