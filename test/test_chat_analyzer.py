# -*- coding: utf-8 -*-
"""
Tests for chat_analyzer — export detection, text extraction,
message normalisation, combine_messages_text, and TXT chat parsing.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from chat_analyzer import (
    is_chat_export,
    is_txt_chat_export,
    parse_txt_chat_export,
    combine_messages_text,
    _extract_msg_text,
    _normalise_msg,
)


# ------------------------------------------------------------------
# is_chat_export  (JSON)
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
    assert n["user_name"]  == "Marko"
    assert n["user_id"]    == "user123"
    assert n["date"]       == "2024-06-01"
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
    messages = [{"text": "first"}, {"text": "second"}, {"text": "third"}]
    result = combine_messages_text(messages)
    assert "first" in result and "second" in result and "third" in result

def test_skips_non_dicts():
    messages = [{"text": "valid"}, "not a dict", 42]
    assert combine_messages_text(messages) == "valid"

def test_empty_list():
    assert combine_messages_text([]) == ""


# ------------------------------------------------------------------
# is_txt_chat_export
# ------------------------------------------------------------------

SAMPLE_TXT_CHAT = """\
[2024-03-04 12:00:00] Ana Jovanović: Hello, how are you?
[2024-03-04 12:05:30] Marko Nikolić: I'm doing great, thanks!
[2024-03-04 12:06:15] Ana Jovanović: <Photo: photo_1@04-03-2024_12-06-15.jpg>
"""

def test_txt_chat_detected():
    assert is_txt_chat_export(SAMPLE_TXT_CHAT) is True

def test_plain_text_not_detected():
    assert is_txt_chat_export("Ovo je obican tekst bez poruka.\nDrugi red.") is False

def test_single_matching_line_not_enough():
    # Only one matching line — requires at least 2
    assert is_txt_chat_export("[2024-03-04 12:00:00] Ana: Hello") is False

def test_txt_chat_with_T_separator():
    text = "[2024-03-04T12:00:00] Ana: Hi\n[2024-03-04T12:01:00] Marko: Hey"
    assert is_txt_chat_export(text) is True


# ------------------------------------------------------------------
# parse_txt_chat_export
# ------------------------------------------------------------------

def test_parse_message_count():
    parsed = parse_txt_chat_export(SAMPLE_TXT_CHAT)
    # 3 lines = 3 messages (media one has empty text but is still parsed)
    assert len(parsed["messages"]) == 3

def test_parse_sender_extracted():
    parsed = parse_txt_chat_export(SAMPLE_TXT_CHAT)
    assert parsed["messages"][0]["from"] == "Ana Jovanović"
    assert parsed["messages"][1]["from"] == "Marko Nikolić"

def test_parse_text_extracted():
    parsed = parse_txt_chat_export(SAMPLE_TXT_CHAT)
    assert parsed["messages"][0]["text"] == "Hello, how are you?"
    assert parsed["messages"][1]["text"] == "I'm doing great, thanks!"

def test_parse_media_message_empty_text():
    parsed = parse_txt_chat_export(SAMPLE_TXT_CHAT)
    assert parsed["messages"][2]["text"] == ""

def test_parse_date_extracted():
    parsed = parse_txt_chat_export(SAMPLE_TXT_CHAT)
    assert parsed["messages"][0]["date"] == "2024-03-04 12:00:00"

def test_parse_multiline_message():
    text = (
        "[2024-03-04 12:00:00] Ana: First line\n"
        "continuation of the message\n"
        "[2024-03-04 12:01:00] Marko: Next message\n"
    )
    parsed = parse_txt_chat_export(text)
    assert len(parsed["messages"]) == 2
    assert "continuation" in parsed["messages"][0]["text"]

def test_parse_type_and_name():
    parsed = parse_txt_chat_export(SAMPLE_TXT_CHAT)
    assert parsed["type"] == "txt_chat"
    assert "TXT" in parsed["name"]

def test_parse_message_ids_sequential():
    parsed = parse_txt_chat_export(SAMPLE_TXT_CHAT)
    ids = [m["id"] for m in parsed["messages"]]
    assert ids == list(range(1, len(ids) + 1))