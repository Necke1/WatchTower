# -*- coding: utf-8 -*-
"""
Tests for chat_analyzer — export detection, text extraction,
message normalisation, and combine_messages_text.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from chat_analyzer import (
    is_chat_export,
    combine_messages_text,
    _extract_msg_text,
    _normalise_msg,
)


# ------------------------------------------------------------------
# is_chat_export
# ------------------------------------------------------------------

VALID_EXPORT = {
    "name": "Test Chat",
    "type": "personal_chat",
    "messages": [
        {"id": 1, "type": "message", "date": "2024-01-01", "from": "Ana", "text": "hello"}
    ]
}

def test_valid_export_detected():
    assert is_chat_export(VALID_EXPORT) is True


def test_empty_messages_rejected():
    assert is_chat_export({"messages": []}) is False


def test_non_dict_rejected():
    assert is_chat_export([1, 2, 3]) is False # type: ignore


def test_missing_messages_key():
    assert is_chat_export({"name": "chat"}) is False


def test_messages_with_unknown_keys():
    export = {"messages": [{"foo": "bar"}]}
    assert is_chat_export(export) is False


def test_messages_with_known_key():
    export = {"messages": [{"text": "hello"}]}
    assert is_chat_export(export) is True


# ------------------------------------------------------------------
# _extract_msg_text
# ------------------------------------------------------------------

def test_plain_string_text():
    assert _extract_msg_text({"text": "hello world"}) == "hello world"


def test_rich_text_segments():
    msg = {"text": [{"type": "plain", "text": "hi "}, {"type": "link", "text": "example.com"}]}
    assert _extract_msg_text(msg) == "hi example.com"


def test_content_field_fallback():
    assert _extract_msg_text({"content": "fallback"}) == "fallback"


def test_emoji_field_appended():
    result = _extract_msg_text({"text": "hello", "sticker_emoji": "💣"})
    assert "💣" in result


def test_empty_message():
    assert _extract_msg_text({}) == ""


def test_non_string_text_returns_empty():
    assert _extract_msg_text({"text": 12345}) == ""


# ------------------------------------------------------------------
# _normalise_msg
# ------------------------------------------------------------------

def test_telegram_fields_normalised():
    msg = {"id": 5, "from": "Marko", "from_id": "user123", "date": "2024-06-01", "text": "test"}
    n = _normalise_msg(msg)
    assert n["user_name"] == "Marko"
    assert n["user_id"]   == "user123"
    assert n["date"]      == "2024-06-01"
    assert n["message_id"] == 5


def test_fallback_fields():
    msg = {"sender": "Jovana", "sender_id": "j1", "timestamp": "2024-01-01T10:00", "text": "hi"}
    n = _normalise_msg(msg)
    assert n["user_name"] == "Jovana"
    assert n["user_id"]   == "j1"


def test_unknown_user_fallback():
    n = _normalise_msg({"text": "anonymous"})
    assert n["user_name"] == "Unknown"


# ------------------------------------------------------------------
# combine_messages_text
# ------------------------------------------------------------------

def test_combines_all_messages():
    messages = [
        {"text": "first"},
        {"text": "second"},
        {"text": "third"},
    ]
    result = combine_messages_text(messages)
    assert "first"  in result
    assert "second" in result
    assert "third"  in result


def test_skips_non_dicts():
    messages = [{"text": "valid"}, "not a dict", 42]
    result = combine_messages_text(messages)
    assert result == "valid"


def test_empty_list():
    assert combine_messages_text([]) == ""