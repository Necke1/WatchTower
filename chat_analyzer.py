# -*- coding: utf-8 -*-
"""
WatchTower: Chat Export Analyzer
=================================
Handles detection and per-message analysis of chat export JSON files
(Telegram-style and similar formats).

Public API
----------
is_chat_export(parsed)          → bool
process_chat_export(parsed, filename, raw_bytes, analyzer) → dict

The returned dict is placed under the "chat_analysis" key in the
/api/analyze/file response when a chat export is detected.
"""

from collections import Counter
from text_analyzer import TextAnalyzer


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def is_chat_export(parsed: dict) -> bool:
    """
    Return True if the parsed JSON looks like a chat message export.

    Heuristic: top-level dict with a "messages" array whose first entry
    contains at least one recognised message field.
    """
    if not isinstance(parsed, dict):
        return False
    messages = parsed.get("messages")
    if not isinstance(messages, list) or len(messages) == 0:
        return False
    first = messages[0]
    if not isinstance(first, dict):
        return False
    msg_keys = {
        "id", "type", "date", "from", "from_id",
        "text", "message_id", "sender", "sender_id",
        "content", "timestamp",
    }
    return bool(msg_keys & set(first.keys()))


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def _extract_msg_text(msg: dict) -> str:
    """
    Pull the full readable text out of a single message dict.

    Handles two common shapes:
      • "text": "hello"                          — plain string
      • "text": [{"type":"plain","text":"hi"}]   — Telegram rich-text segments

    Also appends any emoji stored in adjacent fields:
      sticker_emoji, emoji, reaction
    """
    raw = msg.get("text") or msg.get("content") or msg.get("message") or ""

    if isinstance(raw, str):
        text = raw
    elif isinstance(raw, list):
        parts = []
        for seg in raw:
            if isinstance(seg, str):
                parts.append(seg)
            elif isinstance(seg, dict):
                parts.append(seg.get("text", ""))
        text = "".join(parts)
    else:
        text = ""

    # Append any standalone emoji fields
    for field in ("sticker_emoji", "emoji", "reaction"):
        val = msg.get(field)
        if isinstance(val, str) and val.strip():
            text = text.strip() + " " + val.strip()

    return text.strip()


# ---------------------------------------------------------------------------
# Message normalisation
# ---------------------------------------------------------------------------

def _normalise_msg(msg: dict) -> dict:
    """
    Map whatever field names the export uses onto our standard keys:
      user_name, user_id, date, text, message_id

    Supports Telegram, WhatsApp-ish, and generic export shapes.
    """
    user_name = (
        msg.get("from")
        or msg.get("sender")
        or msg.get("author")
        or msg.get("username")
        or "Unknown"
    )
    user_id = str(
        msg.get("from_id")
        or msg.get("sender_id")
        or msg.get("author_id")
        or msg.get("user_id")
        or ""
    )
    date = (
        msg.get("date")
        or msg.get("timestamp")
        or msg.get("time")
        or ""
    )
    return {
        "message_id": msg.get("id") or msg.get("message_id"),
        "user_name":  user_name,
        "user_id":    user_id,
        "date":       str(date),
        "text":       _extract_msg_text(msg),
    }


# ---------------------------------------------------------------------------
# Verdict sentence
# ---------------------------------------------------------------------------

def _build_verdict(msg: dict, score: int, risk: str, term_freq: dict) -> str:
    """
    Build the human-readable verdict sentence for one flagged message.

    Format (as requested):
      "Korisnik <name> sa ID-jem <id> napisao/la poruku '<text+emoji>'
       koja prema analizi ovog alata izgleda sumnjivo, jer ima rizični
       skor od <score> što odgovara nivou rizika: <risk>.
       Pronađeni termini: <term ×count w weight>, …"
    """
    terms_str = ", ".join(
        f'"{t}" (×{c})'
        for t, c in sorted(term_freq.items(), key=lambda x: x[1], reverse=True)
    )
    return (
        f'Korisnik {msg["user_name"]} sa ID-jem {msg["user_id"]} '
        f'napisao/la poruku "{msg["text"]}" '
        f'koja prema analizi ovog alata izgleda sumnjivo, '
        f'jer ima rizični skor od {score} '
        f'što odgovara nivou rizika: {risk}. '
        f'Pronađeni termini: {terms_str}.'
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def combine_messages_text(messages: list) -> str:
    """
    Public helper: join all extractable text from a list of raw message dicts.
    Used by watchtower.analyze_file() to build the whole-document combined text.
    """
    return " ".join(
        _extract_msg_text(m)
        for m in messages
        if isinstance(m, dict)
    )


def process_chat_export(
    parsed:    dict,
    filename:  str,
    raw_bytes: bytes,
    analyzer:  TextAnalyzer,
) -> dict:
    """
    Analyse every message in a chat export and return structured results.

    Parameters
    ----------
    parsed    : The already-parsed JSON dict (top-level chat export object).
    filename  : Original upload filename (for file_info metadata).
    raw_bytes : Raw bytes of the uploaded file (for size metadata).
    analyzer  : A configured TextAnalyzer instance (provided by the caller).

    Returns
    -------
    A dict placed under "chat_analysis" in the API response:

        chat_meta         — name / type / id of the conversation
        flagged_messages  — list of messages with score > 0, each with:
                              user_name, user_id, date, text,
                              total_score, risk_level, risk_description,
                              term_frequencies, term_weights, verdict
        stats             — counts, averages, risk distribution, top users
    """
    raw_messages = parsed.get("messages", [])

    chat_meta = {
        "chat_name": parsed.get("name") or parsed.get("title") or "Unknown Chat",
        "chat_type": parsed.get("type", "unknown"),
        "chat_id":   parsed.get("id"),
    }

    # Normalise messages — skip service entries (join/leave/pin notices etc.)
    messages = []
    for raw_msg in raw_messages:
        if not isinstance(raw_msg, dict):
            continue
        entry_type = raw_msg.get("type", "message")
        if entry_type and entry_type != "message":
            continue
        m = _normalise_msg(raw_msg)
        if m["text"]:
            messages.append(m)

    if not messages:
        raise ValueError(
            "No analysable text messages found in the export. "
            "All entries appear to be service messages or have empty text."
        )

    # Analyse each message
    flagged:     list[dict] = []
    all_results: list[dict] = []
    user_scores: dict       = {}

    for msg in messages:
        raw    = analyzer.analyze_text(msg["text"])
        an     = raw["analysis"]
        score  = an["total_score"]
        risk   = an["risk_level"]
        tfreq  = an["term_frequencies"]
        tweights = {t: analyzer.term_to_weight.get(t, 1) for t in tfreq}

        entry = {
            # ── Identity (fields requested) ───────────────────────────────
            "user_name":        msg["user_name"],
            "user_id":          msg["user_id"],
            "date":             msg["date"],
            "text":             msg["text"],
            # ── Risk ─────────────────────────────────────────────────────
            "total_score":      score,
            "risk_level":       risk,
            "risk_description": an["risk_description"],
            "term_frequencies": tfreq,
            "term_weights":     tweights,
            # ── Human-readable verdict ────────────────────────────────────
            "verdict": _build_verdict(msg, score, risk, tfreq) if score > 0 else None,
        }

        all_results.append(entry)

        # Accumulate per-user totals
        uid = msg["user_id"] or msg["user_name"]
        if uid not in user_scores:
            user_scores[uid] = {
                "user_name":   msg["user_name"],
                "user_id":     msg["user_id"],
                "total_score": 0,
                "flagged":     0,
                "messages":    0,
            }
        user_scores[uid]["total_score"] += score
        user_scores[uid]["messages"]    += 1

        if score > 0:
            flagged.append(entry)
            user_scores[uid]["flagged"] += 1

    scores = [r["total_score"] for r in all_results]
    stats = {
        "total_messages_in_export": len(raw_messages),
        "analysable_messages":      len(messages),
        "flagged_messages_count":   len(flagged),
        "clean_messages_count":     len(all_results) - len(flagged),
        "total_score_sum":          sum(scores),
        "average_score":            round(sum(scores) / max(len(scores), 1), 2),
        "max_score":                max(scores, default=0),
        "risk_distribution":        dict(Counter(r["risk_level"] for r in all_results)),
        "top_users_by_score":       sorted(
            user_scores.values(), key=lambda u: u["total_score"], reverse=True
        )[:10],
    }

    return {
        "chat_meta":        chat_meta,
        "flagged_messages": flagged,
        "stats":            stats,
        "file_info": {
            "filename":   filename,
            "size_bytes": len(raw_bytes),
        },
    }