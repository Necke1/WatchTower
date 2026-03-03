#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WatchTower: Serbian Text Analysis Tool for Content Risk Assessment
Author: Nemanja Mosurović (@Necke1)
Date: 2025

Entry point — handles CLI arguments, demo mode, and utility commands.
"""

import json
import os
import sys
import argparse
from pathlib import Path

from constants import (
    DEFAULT_DICTIONARY_FILE,
    OUTPUT_FILE,
    SAMPLE_TEXTS,
)
from text_analyzer import TextAnalyzer
from report_generator import ReportGenerator
from chat_analyzer import is_chat_export, process_chat_export, combine_messages_text


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
    print(f"Fajl: {analyzer.spell_checker.user_corrections_file}\n")
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
# File analysis  (shared by CLI --file and the API upload route)
# ---------------------------------------------------------------------------

def analyze_file(raw_bytes: bytes, filename: str, analyzer: TextAnalyzer,
                 output_json: bool = False) -> dict:
    """
    Analyse the contents of an uploaded / opened file.

    Handles two cases:
      .txt  → full content analysed as a single text.
      .json → auto-detected:
                • Chat export (top-level "messages" array):
                    each message is analysed individually.
                    Returns chat_analysis dict alongside standard fields.
                • Generic JSON:
                    text extracted by key priority, then analysed as a whole.

    Returns the same dict shape used by /api/analyze/file, so api.py can
    return it directly and the CLI can print a summary from it.

    Parameters
    ----------
    raw_bytes   : Raw file bytes (UTF-8 or latin-1).
    filename    : Original filename — used only for metadata and extension detection.
    analyzer    : Configured TextAnalyzer instance.
    output_json : If True, formatted_report is JSON; otherwise plain text.
    """
    extension = Path(filename).suffix.lower()

    if extension not in (".txt", ".json"):
        raise ValueError(f"Unsupported file type '{extension}'. Only .txt and .json are supported.")

    try:
        raw_text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raw_text = raw_bytes.decode("latin-1")

    # ── .txt ────────────────────────────────────────────────────────────────
    if extension == ".txt":
        text_to_analyze  = raw_text
        extraction_notes = ["Plain text — entire file content used."]
        chat_analysis    = None

    # ── .json ────────────────────────────────────────────────────────────────
    else:
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")

        # ── Branch A: chat export ──────────────────────────────────────────
        if is_chat_export(parsed):
            chat_analysis = process_chat_export(parsed, filename, raw_bytes, analyzer)
            msgs_text        = parsed.get("messages", [])
            text_to_analyze  = combine_messages_text(msgs_text) or " "
            extraction_notes = [
                f"Chat export detected — {len(msgs_text)} messages found. "
                "Individual message analysis in chat_analysis field."
            ]

        # ── Branch B: generic JSON ─────────────────────────────────────────
        else:
            chat_analysis = None

            PRIORITY_KEYS = [
                "text", "tekst", "content", "sadrzaj", "message", "poruka",
                "body", "telo", "description", "opis", "data", "podatak",
            ]

            def _find_by_key(obj, keys):
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

            found = _find_by_key(parsed, PRIORITY_KEYS)
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
    report = (ReportGenerator.generate_json_report(result)
              if output_json
              else ReportGenerator.generate_text_report(result, analyzer))

    return {
        **result,
        "formatted_report": report,
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

        an = result["analysis"]
        print(f"\nAnaliziran fajl: {args.file}")

        # Chat export — print per-user summary
        if result.get("chat_analysis"):
            ca = result["chat_analysis"]
            st = ca["stats"]
            print(f"\nChat export: {ca['chat_meta']['chat_name']}")
            print(f"Poruka ukupno: {st['analysable_messages']} | "
                  f"Označenih: {st['flagged_messages_count']} | "
                  f"Maks. skor: {st['max_score']}")
            if ca["flagged_messages"]:
                print("\nOznačene poruke:")
                for m in ca["flagged_messages"]:
                    print(f"  [{m['risk_level']}] {m['user_name']} ({m['date']}): "
                          f"skor={m['total_score']}  {m['text'][:60]}…")
        else:
            print(f"\nNivo rizika: {an['risk_level']}")
            print(f"Skor: {an['total_score']} | Termina: {an['unique_terms_count']}")

        ReportGenerator.save_report(result["formatted_report"], args.output)

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


if __name__ == "__main__":
    main()