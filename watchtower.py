#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WatchTower: Serbian Text Analysis Tool for Content Risk Assessment
Version: 1.1 - Optimized & Modular
Author: Nemanja Mosurović (@Necke1)
Date: 2025

Entry point — handles CLI arguments, demo mode, and utility commands.
"""

import json
import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

from constants import (
    DEFAULT_DICTIONARY_FILE,
    OUTPUT_FILE,
    SAMPLE_TEXTS,
)
from text_analyzer import TextAnalyzer
from report_generator import ReportGenerator
from chat_analyzer import (
    is_chat_export,
    is_txt_chat_export,
    parse_txt_chat_export,
    process_chat_export,
    combine_messages_text,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='WatchTower: Serbian Text Analysis Tool for Content Risk Assessment'
    )
    parser.add_argument('--text',   type=str, help='Text to analyze (enclose in quotes)')
    parser.add_argument('--file',   type=str, help='File containing text to analyze')
    parser.add_argument('--dictionary', type=str, default=DEFAULT_DICTIONARY_FILE,
                        help=f'Dictionary file with weights (default: {DEFAULT_DICTIONARY_FILE})')
    parser.add_argument('--output', type=str, default=OUTPUT_FILE,
                        help=f'Output file (default: {OUTPUT_FILE})')
    parser.add_argument('--no-spellcheck',       action='store_true', help='Disable spell checking')
    parser.add_argument('--auto-correct',        action='store_true', help='Enable automatic spelling correction')
    parser.add_argument('--interactive-correct', action='store_true',
                        help='Enable interactive correction with user confirmation and rating')
    parser.add_argument('--interactive-context', action='store_true',
                        help='After analysis, review each flagged term and provide feedback '
                             '(false positive / lower / higher / confirmed). '
                             'Patterns are saved to korekcija_paterna.txt and applied automatically next time.')
    parser.add_argument('--json',  action='store_true', help='Output results in JSON format')
    parser.add_argument('--demo',  action='store_true', help='Run demo with sample texts')
    parser.add_argument('--show-corrections',  action='store_true',
                        help='Display learned corrections from korekcija.txt')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Utility commands
# ---------------------------------------------------------------------------

def show_learned_corrections(analyzer: TextAnalyzer):
    """Print all learned corrections held by the spell checker."""
    if analyzer.spell_checker is None:
        print("Spell checker nije dostupan — korekcije nisu učitane.")
        return

    corrections = analyzer.spell_checker.get_all_corrections()

    if not corrections:
        print("Nema sačuvanih korekcija.")
        return

    print("\n" + "=" * 70)
    print(f"NAUČENE KOREKCIJE ({len(corrections)} ukupno)")
    print("=" * 70)
    print(f"Fajl: {analyzer.spell_checker.spell_corrections_file}\n")
    for i, c in enumerate(corrections, 1):
        print(f"{i:3d}. {c['original']} → {c['corrected']} "
              f"| ocena:{c['rating']} | broj:{c['count']} | datum:{c['last_used']}")
    print("\n" + "=" * 70)
    print(f"Ukupno: {len(corrections)} naučenih korekcija")
    print("=" * 70)



def run_demo(analyzer: TextAnalyzer):
    print("\n" + "=" * 70)
    print("WATCHTOWER DEMONSTRACIJA (Modular Version)")
    print("=" * 70)

    for i, sample in enumerate(SAMPLE_TEXTS, 1):
        print(f"\n{'=' * 70}")
        print(f"[TEST {i}/{len(SAMPLE_TEXTS)}]: {sample['name']}")
        print(f"Opis: {sample['description']}")
        print(f"{'=' * 70}")
        print(f"Tekst: {sample['text']}")

        results  = analyzer.analyze_text(sample['text'])
        analysis = results['analysis']
        spell    = results['spell_checking']

        # Cyrillic conversion notice
        if results['original_text'] != results['processed_text']:
            if any('\u0400' <= c <= '\u04FF' for c in results['original_text']):
                print(f"🔤 Konvertovano: {results['processed_text']}")

        print()

        # Spell results
        if spell['errors_found'] > 0:
            print(f"📝 Pravopisne greške: {spell['errors_found']}")
            if analyzer.auto_correct and spell['corrections_made']:
                print(f"   Ispravljeno: {spell['corrections_made']} reči")
                if spell['corrected_text'] != spell['original_text']:
                    print(f"   Ispravljen tekst: {spell['corrected_text']}")
            else:
                for err in spell['error_details'][:3]:
                    sugg = ', '.join(err['suggestions'][:3]) if err['suggestions'] else 'nema'
                    print(f"   - {err['word']} (predlozi: {sugg})")
        else:
            print("✓ Pravopis: Nema grešaka")

        # Risk summary
        print(f"\n🎯 Nivo rizika: {analysis['risk_level']}")
        print(f"📊 Skor: {analysis['total_score']} | Termina: {analysis['unique_terms_count']}")
        if analysis['term_frequencies']:
            print("🔍 Pronađeni termini:")
            for term, count in sorted(analysis['term_frequencies'].items(),
                                      key=lambda x: x[1], reverse=True):
                w = analyzer.term_to_weight.get(term, 1)
                print(f"   - {term}: {count}x (težina: {w})")
        else:
            print("✓ Nisu pronađeni rizični termini")

        print(f"⏱️  Vreme: {results['statistics']['processing_time_seconds']:.3f}s")

    print("\n" + "=" * 70)
    print("DEMONSTRACIJA ZAVRŠENA")
    print("=" * 70)
    print("\nOpcije:")
    print("  --auto-correct           Automatska korekcija")
    print("  --interactive-correct    Interaktivna korekcija sa ocenom")
    print("  --show-corrections       Prikaži naučene korekcije")
    print("\nPrimeri:")
    print("  python watchtower.py --demo --auto-correct")
    print("  python watchtower.py --text 'Vaš tekst' --interactive-correct")
    print("  python watchtower.py --show-corrections")


# ---------------------------------------------------------------------------
# JSON extraction helpers  (module-level so they're not redefined per call)
# ---------------------------------------------------------------------------

_PRIORITY_KEYS = [
    "text", "tekst", "content", "sadrzaj", "message", "poruka",
    "body", "telo", "description", "opis", "data", "podatak",
]


def _find_by_key(obj, keys):
    """Recursively find the first string value under any of the given keys."""
    if isinstance(obj, dict):
        for k in keys:
            if k in obj and isinstance(obj[k], str) and obj[k].strip():
                return obj[k]
        for v in obj.values():
            r = _find_by_key(v, keys)
            if r:
                return r
    elif isinstance(obj, list):
        for item in obj:
            r = _find_by_key(item, keys)
            if r:
                return r
    return None


def _collect_strings(obj, depth=0):
    """Recursively collect all non-empty strings from a JSON object."""
    if depth > 10:
        return []
    if isinstance(obj, str):
        return [obj] if obj.strip() else []
    if isinstance(obj, list):
        out = []
        for item in obj:
            out.extend(_collect_strings(item, depth + 1))
        return out
    if isinstance(obj, dict):
        out = []
        for v in obj.values():
            out.extend(_collect_strings(v, depth + 1))
        return out
    return []


# ---------------------------------------------------------------------------
# File analysis  (shared by CLI --file and the API upload route)
# ---------------------------------------------------------------------------

def analyze_file(raw_bytes: bytes, filename: str, analyzer: TextAnalyzer,
                 output_json: bool = False) -> dict:
    """
    Analyse the contents of an uploaded / opened file.

    Handles three cases:
      .txt (plain)  → full content analysed as a single text.
      .txt (chat)   → auto-detected by timestamped line pattern:
                        [YYYY-MM-DD HH:MM:SS] Sender: text
                        Each message analysed individually, same output
                        shape as a JSON chat export.
      .json         → auto-detected:
                        • Chat export (top-level "messages" array):
                            each message is analysed individually.
                        • Generic JSON:
                            text extracted by key priority, then analysed
                            as a whole.

    Returns the same dict shape used by /api/analyze/file, so api.py can
    return it directly and the CLI can print a summary from it.

    Parameters
    ----------
    raw_bytes   : Raw file bytes (UTF-8 or latin-1).
    filename    : Original filename — used only for metadata and extension detection.
    analyzer    : Configured TextAnalyzer instance.
    output_json : If True, formatted_report is JSON; otherwise plain text.
    """
    _start = datetime.now()
    extension = Path(filename).suffix.lower()

    if extension not in (".txt", ".json"):
        raise ValueError(f"Unsupported file type '{extension}'. Only .txt and .json are supported.")

    try:
        raw_text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raw_text = raw_bytes.decode("latin-1")

    # ── .txt ────────────────────────────────────────────────────────────────
    if extension == ".txt":
        if is_txt_chat_export(raw_text):
            parsed        = parse_txt_chat_export(raw_text)
            chat_analysis = process_chat_export(parsed, filename, raw_bytes, analyzer)
            msgs_text     = parsed.get("messages", [])
            text_to_analyze  = combine_messages_text(msgs_text) or " "
            extraction_notes = [
                f"TXT chat export detected — {len(msgs_text)} messages parsed. "
                "Individual message analysis in chat_analysis field."
            ]
        else:
            text_to_analyze  = raw_text
            extraction_notes = ["Plain text — entire file content used."]
            chat_analysis    = None

    else:
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")

        if is_chat_export(parsed):
            chat_analysis = process_chat_export(parsed, filename, raw_bytes, analyzer)
            msgs_text        = parsed.get("messages", [])
            text_to_analyze  = combine_messages_text(msgs_text) or " "
            extraction_notes = [
                f"Chat export detected — {len(msgs_text)} messages found. "
                "Individual message analysis in chat_analysis field."
            ]
        else:
            chat_analysis = None
            found = _find_by_key(parsed, _PRIORITY_KEYS)
            if found:
                text_to_analyze  = found
                extraction_notes = ["Extracted from JSON key matching priority list."]
            else:
                all_strings = _collect_strings(parsed)
                if not all_strings:
                    raise ValueError("No text content found in the JSON file.")
                text_to_analyze  = "\n".join(all_strings)
                extraction_notes = [
                    f"No known text key found — joined {len(all_strings)} "
                    "string value(s) from the document."
                ]

    if not text_to_analyze.strip():
        raise ValueError("The file contains no readable text.")

    result = analyzer.analyze_text(text_to_analyze)
    total_time = (datetime.now() - _start).total_seconds()

    # ── Build formatted report ───────────────────────────────────────────────
    if chat_analysis and not output_json:
        # For chat exports the overall risk uses average score per message
        # and the percentage of flagged messages — not the raw sum — so a
        # chat with many low-scoring messages isn't inflated to VISOK RIZIK.
        st = chat_analysis["stats"]
        overall_level, overall_desc = analyzer.calculate_chat_risk_level(
            weighted_score   = st.get("weighted_score_sum", st["total_score_sum"]),
            total_messages   = st["analysable_messages"],
            flagged_messages = st["flagged_messages_count"],
            weighted_avg     = st.get("weighted_avg", 0.0),
        )
        from constants import RISK_LEVELS
        overall_recs = RISK_LEVELS[overall_level]["recommendations"]

        # Inject the chat-level risk into chat_analysis so the API can use it
        chat_analysis["overall_risk_level"]       = overall_level
        chat_analysis["overall_risk_description"] = overall_desc
        chat_analysis["overall_recommendations"]  = overall_recs

        report = ReportGenerator.generate_chat_report(
            chat_analysis            = chat_analysis,
            overall_risk_level       = overall_level,
            overall_risk_description = overall_desc,
            overall_recommendations  = overall_recs,
            total_processing_time    = total_time,
        )
    elif output_json:
        report = ReportGenerator.generate_json_report(result)
    else:
        report = ReportGenerator.generate_text_report(result, analyzer)

    return {
        **result,
        "formatted_report": report,
        "total_processing_time": total_time,
        "term_weights": {
            t: analyzer.term_to_weight.get(t, 1)
            for t in result["analysis"]["term_frequencies"]
        },
        "file_info": {
            "filename":         filename,
            "extension":        extension,
            "size_bytes":       len(raw_bytes),
            "chars_extracted":  len(text_to_analyze),
            "extraction_notes": extraction_notes,
        },
        "chat_analysis": chat_analysis,
    }


# ---------------------------------------------------------------------------
# Interactive context review  (CLI equivalent of browser context IC mode)
# ---------------------------------------------------------------------------

_FEEDBACK_OPTIONS = {
    '1': ('false_positive',  'Lažni alarm       — term nije rizičan u ovom kontekstu  (× 0.0)'),
    '2': ('lower_severity',  'Niži rizik        — rizičan ali manje nego procenjeno   (× 0.35)'),
    '3': ('higher_severity', 'Viši rizik        — opasnije nego što skor pokazuje     (× 3.0)'),
    '4': ('confirmed',       'Potvrđena pretnja — direktna operativna pretnja         (× 2.0)'),
    '5': ('skip',            'Preskoči ovu pojavu'),
}


def _split_into_units(raw_text: str, result: dict) -> list:
    """
    Return a list of clean text units for the interactive context reviewer.

    For chat exports  → individual message dicts (user_name, date, text).
    For plain text    → individual sentences split on .  !  ?  or newline.

    A "unit" is the smallest piece of text in which a term appears with a
    specific meaning — the right granularity for per-occurrence feedback.
    """
    ca = result.get('chat_analysis')
    if ca:
        # Use flagged messages (already have clean text extracted by chat_analyzer)
        flagged = ca.get('flagged_messages', [])
        units = []
        for m in flagged:
            txt = (m.get('text') or '').strip()
            if txt:
                units.append({
                    'text':      txt,
                    'label':     f"{m.get('user_name','?')}  [{m.get('date','')}]",
                    'risk':      m.get('risk_level', ''),
                    'score':     m.get('total_score', 0),
                })
        return units

    # Plain text — split on sentence-ending punctuation or blank lines
    import re as _re
    parts = _re.split(r'(?<=[.!?])\s+|\n{2,}', raw_text.strip())
    return [{'text': p.strip(), 'label': '', 'risk': '', 'score': 0}
            for p in parts if p.strip()]


def _highlight(text: str, term: str, width: int = 80) -> str:
    """
    Return a terminal-safe snippet of `text` with `term` wrapped in [brackets].
    Finds the first occurrence, extracts a window of `width` chars around it,
    and replaces matched text (case-insensitive, word-boundary aware).
    """
    import re as _re
    lo = text.lower()
    idx = lo.find(term.lower())
    if idx < 0:
        # Fallback: first `width` chars of text
        return text[:width].replace('\n', ' ')

    half   = width // 2
    start  = max(0, idx - half)
    end    = min(len(text), idx + len(term) + half)
    window = text[start:end].replace('\n', ' ')

    # Re-find term inside the window (offset may have shifted)
    rel = window.lower().find(term.lower())
    if rel >= 0:
        window = (window[:rel]
                  + '[' + window[rel:rel + len(term)] + ']'
                  + window[rel + len(term):])

    prefix = '…' if start > 0 else ''
    suffix = '…' if end < len(text) else ''
    return prefix + window + suffix


def _ask_feedback(analyzer, term: str, unit_text: str, term_weight) -> bool:
    """
    Show the menu for one (term, unit_text) pair and process the analyst's answer.
    Returns True to continue, False if the analyst interrupted the session.
    """
    print()
    for key, (_, label) in _FEEDBACK_OPTIONS.items():
        print(f"    [{key}] {label}")
    print("        — nebo unesite decimalni broj (npr. 1.5) za prilagođeni multiplikator")
    print()

    while True:
        try:
            raw = input("    Vaš izbor: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nInteraktivni pregled prekinut.")
            return False

        if raw in _FEEDBACK_OPTIONS:
            feedback, _ = _FEEDBACK_OPTIONS[raw]
            if feedback == 'skip':
                print("    → Preskočeno.\n")
                return True
            fp = analyzer.submit_feedback(unit_text, term, feedback)
            if fp:
                print(f"    ✓ Sačuvano: '{term}' + {sorted(fp)} → {feedback}\n")
            else:
                print(f"    ⚠  Patern nije sačuvan (prozor sadrži samo stop-reči).\n")
            return True

        # Custom multiplier
        try:
            mult = float(raw)
            if not (0.0 <= mult <= 10.0):
                raise ValueError("out of range")
            lemmas = analyzer.extract_lemmas(analyzer.normalize_text(unit_text))
            fp = analyzer.learned_patterns.learn(term, lemmas, mult, label='[cli:custom]')
            if fp:
                print(f"    ✓ Sačuvano: '{term}' + {sorted(fp)} → multiplikator {mult}\n")
            else:
                print(f"    ⚠  Patern nije sačuvan (stop-reči).\n")
            return True
        except ValueError:
            print("    Unesite broj od 1–5 ili decimalni multiplikator (0.0–10.0).")


def build_context_items(results: dict, raw_text: str, analyzer) -> list:
    """
    Build the list of context items that the browser IC dialog consumes.

    Reuses the same _split_into_units / _highlight helpers that power the
    terminal interactive mode, so browser and CLI always show identical snippets.

    Each item:
        term         — the matched dictionary term
        weight       — term weight from the dictionary
        snippet      — windowed text with term wrapped in [brackets]
        unit_text    — full clean sentence / message text (sent to /api/feedback)
        label        — "UserName  [date]" for chat messages, "" for plain text
        risk         — risk level of the containing message (chat only)
        score        — score of the containing message (chat only)
        occurrence_n — which occurrence of this term this item represents (1-based)
        occ_total    — total occurrences of this term across all units
    """
    term_freq    = results['analysis']['term_frequencies']
    term_weights = {t: analyzer.term_to_weight.get(t, 1) for t in term_freq}

    if not term_freq:
        return []

    units = _split_into_units(raw_text, results)
    if not units:
        return []

    # Sort terms by weight descending — highest risk reviewed first
    terms = sorted(term_freq.items(), key=lambda x: term_weights.get(x[0], 1), reverse=True)

    items = []
    for term, _total_count in terms:
        containing = [u for u in units if term.lower() in u['text'].lower()]
        for j, unit in enumerate(containing, 1):
            items.append({
                'term':        term,
                'weight':      term_weights.get(term, 1),
                'snippet':     _highlight(unit['text'], term),
                'unit_text':   unit['text'],
                'label':       unit.get('label', ''),
                'risk':        unit.get('risk', ''),
                'score':       unit.get('score', 0),
                'occurrence_n':  j,
                'occ_total':     len(containing),
            })

    return items


def run_interactive_context(analyzer, results: dict, raw_text: str):
    """
    Walk through every occurrence of every flagged term — one occurrence at a
    time — and ask the analyst to classify each one.

    For chat exports each 'occurrence' is the individual flagged message that
    contained the term.  For plain text each 'occurrence' is the sentence that
    contained the term.

    This means the same term seen in five different messages produces five
    separate prompts, each showing the correct context — so "napad" in a news
    headline and "napad" in an operational planning message can be classified
    differently.

    Confirmed answers are saved to korekcija_paterna.txt via LearnedPatternStore
    and applied automatically on future analyses.
    """
    term_freq    = results['analysis']['term_frequencies']
    term_weights = {t: analyzer.term_to_weight.get(t, 1) for t in term_freq}

    if not term_freq:
        print("\n✓ Nisu pronađeni rizični termini — nema šta da se pregleda.")
        return

    # Split source into clean text units
    units = _split_into_units(raw_text, results)
    if not units:
        print("\n⚠  Nije moguće izvući pojedinačne jedinice teksta za pregled.")
        return

    # Sort terms by weight desc (highest risk first)
    terms = sorted(term_freq.items(), key=lambda x: term_weights.get(x[0], 1), reverse=True)

    # For each term, collect only the units that actually contain it
    # Each (term, unit) pair becomes one prompt
    work_items = []   # list of (term, unit, occurrence_n, total_occurrences_for_term)
    for term, _total_count in terms:
        containing = [u for u in units if term.lower() in u['text'].lower()]
        for j, unit in enumerate(containing, 1):
            work_items.append((term, unit, j, len(containing)))

    if not work_items:
        print("\n✓ Termini pronađeni ali nije moguće locirati ih u pojedinačnim porukama.")
        return

    print("\n" + "=" * 70)
    print("INTERAKTIVNA ANALIZA KONTEKSTA")
    print("=" * 70)
    print(f"Pronađeno {len(terms)} term(a) u {len(work_items)} pojav(a).")
    print("Svaka pojava u drugoj poruci/rečenici se prikazuje zasebno.")
    print("Naučeni paterna → biblioteke/korekcija_paterna.txt\n")

    for idx, (term, unit, occ_n, occ_total) in enumerate(work_items, 1):
        weight  = term_weights.get(term, '—')
        snippet = _highlight(unit['text'], term)

        print(f"─── Pojava {idx}/{len(work_items)}  "
              f"(term '{term}', pojava {occ_n}/{occ_total}) "
              + "─" * max(0, 70 - 30 - len(term)))
        print(f"  Term:    {term}   |   Težina: {weight}")
        if unit['label']:
            risk_tag = f"  [{unit['risk']} · skor {unit['score']}]" if unit['risk'] else ''
            print(f"  Poruka:  {unit['label']}{risk_tag}")
        print(f"  Tekst:   {snippet}")

        ok = _ask_feedback(analyzer, term, unit['text'], weight)
        if not ok:
            return

    print("=" * 70)
    print("Interaktivni pregled konteksta završen.")
    print(f"Fajl paterna: biblioteke/korekcija_paterna.txt")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_arguments()

    # Initialize core components first — show-corrections needs the spell checker
    analyzer = TextAnalyzer(
        dictionary_file     = args.dictionary,
        use_spellcheck      = not args.no_spellcheck,
        auto_correct        = args.auto_correct,
        interactive_correct = args.interactive_correct
    )

    # Utility commands
    if args.show_corrections:
        show_learned_corrections(analyzer)
        return

    if args.demo:
        run_demo(analyzer)
        return

    # Get input text / file
    if args.text:
        text_to_analyze = args.text
        _run_text_analysis(args, analyzer, text_to_analyze)

    elif args.file:
        try:
            raw_bytes = Path(args.file).read_bytes()
        except Exception as e:
            print(f"Error reading file: {e}")
            return
        try:
            result = analyze_file(raw_bytes, args.file, analyzer, output_json=args.json)
        except ValueError as e:
            print(f"Greška: {e}")
            return

        total_time = result.get("total_processing_time",
                                 result["statistics"]["processing_time_seconds"])
        print(f"\nAnaliziran fajl: {args.file}")

        # Chat export — print brief terminal summary; full report goes to file
        if result.get("chat_analysis"):
            ca = result["chat_analysis"]
            st = ca["stats"]
            overall = ca.get("overall_risk_level", result["analysis"]["risk_level"])
            print(f"\nChat export: {ca['chat_meta']['chat_name']}")
            print(f"Nivo rizika: {overall}")
            print(f"Poruka ukupno: {st['analysable_messages']} | "
                  f"Označenih: {st['flagged_messages_count']} | "
                  f"Prosečan skor: {st['average_score']:.2f} | "
                  f"Maks. skor: {st['max_score']}")
            print(f"Vreme obrade: {total_time:.3f}s")
            print(f"\nDetaljan izveštaj sačuvan u: {args.output}")
        else:
            an = result["analysis"]
            print(f"\nNivo rizika: {an['risk_level']}")
            print(f"Skor: {an['total_score']} | Termina: {an['unique_terms_count']}")
            print(f"Vreme obrade: {total_time:.3f}s")

        ReportGenerator.save_report(result["formatted_report"], args.output)

        if args.interactive_context:
            run_interactive_context(analyzer, result, raw_bytes.decode('utf-8', errors='replace'))

    else:
        print("Unesite tekst za analizu (Ctrl+D ili Ctrl+Z kada završite):")
        print("-" * 50)
        try:
            text_to_analyze = sys.stdin.read()
        except (KeyboardInterrupt, EOFError):
            print("\nOtkazano.")
            return
        _run_text_analysis(args, analyzer, text_to_analyze)


def _run_text_analysis(args, analyzer: TextAnalyzer, text: str):
    """Shared helper for --text and stdin paths."""
    if not text.strip():
        print("Nema teksta za analizu.")
        return

    print(f"\nAnaliziranje teksta ({len(text)} karaktera)...")
    results = analyzer.analyze_text(text)

    if args.json:
        report        = ReportGenerator.generate_json_report(results)
        json_filename = f"{os.path.splitext(args.output)[0]}.json"
        print("\n" + report)
        ReportGenerator.save_report(report, json_filename)
    else:
        report = ReportGenerator.generate_text_report(results, analyzer)
        print("\n" + report)
        ReportGenerator.save_report(report, args.output)

    print("\n" + "-" * 50)
    print("REZIME ANALIZE:")
    print(f"Nivo rizika:      {results['analysis']['risk_level']}")
    print(f"Ukupan skor:      {results['analysis']['total_score']}")
    print(f"Pronađeno termina:{results['analysis']['unique_terms_count']}")
    print(f"Vreme obrade:     {results['statistics']['processing_time_seconds']:.3f}s")
    print("-" * 50)

    if args.interactive_context:
        run_interactive_context(analyzer, results, text)


if __name__ == "__main__":
    main()