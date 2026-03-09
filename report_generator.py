# -*- coding: utf-8 -*-
"""
WatchTower: Report Generator
Formats analysis results as human-readable text or JSON.
"""

import json
from typing import Dict, Any, Optional, TYPE_CHECKING

from constants import OUTPUT_FILE

if TYPE_CHECKING:
    from text_analyzer import TextAnalyzer


class ReportGenerator:
    """Generate text or JSON reports from analysis results."""

    # ------------------------------------------------------------------
    # Text report
    # ------------------------------------------------------------------

    @staticmethod
    def generate_text_report(results: Dict[str, Any],
                             analyzer: Optional['TextAnalyzer'] = None) -> str:
        """
        Build a formatted text report.
        Pass *analyzer* to enable O(1) per-term weight lookups.
        """
        lines = []

        # ── Header ──────────────────────────────────────────────────────
        lines += [
            "=" * 70,
            "ANALIZA TEKSTA / TEXT ANALYSIS",
            f"Datum: {results['metadata']['timestamp']}",
            "=" * 70, ""
        ]

        # ── Spell checking ───────────────────────────────────────────────
        spell = results['spell_checking']
        lines.append("--- PROVERA PRAVOPISA ---")

        orig = results['original_text']
        lines.append(f"Originalni tekst:\n{orig if len(orig) <= 200 else orig[:200] + '...'}")

        # Show Cyrillic → Latin conversion only when relevant (>20 % Cyrillic)
        if orig != results['processed_text']:
            cyrillic_count = sum(1 for c in orig if '\u0400' <= c <= '\u04FF')
            total_alpha    = sum(1 for c in orig if c.isalpha())
            if total_alpha > 0 and (cyrillic_count / total_alpha) > 0.2:
                proc = results['processed_text']
                lines.append(
                    f"\nKonvertovano u latinicu:\n"
                    f"{proc if len(proc) <= 200 else proc[:200] + '...'}"
                )

        lines += [
            "",
            f"Pronađeno {spell['errors_found']} pogrešno napisanih reči",
            f"Automatski ispravljeno: {spell['corrections_made']} reči"
        ]

        if spell['error_details']:
            lines.append("\nDetalji o greškama:")
            for error in spell['error_details'][:10]:
                sugg = ", ".join(error['suggestions']) if error['suggestions'] else "nema predloga"
                lines.append(f"  ✗ {error['word']} (Predlozi: {sugg})")

        if spell.get('correction_feedback'):
            learned = [f for f in spell['correction_feedback'] if f.get('choice') == 'learned_correction']
            new_c   = [f for f in spell['correction_feedback'] if f.get('choice') != 'learned_correction']

            if learned:
                lines.append("\nNaučene korekcije (automatski primenjene):")
                for fb in learned:
                    lines.append(f"  ✓ {fb['original']} → {fb['corrected']} [Naučeno]")
            if new_c:
                lines.append("\nNove korisničke korekcije:")
                for fb in new_c:
                    rating_str = f" [Ocena: {fb['rating']}/4]" if fb.get('rating') else ""
                    saved_str  = " [Sačuvano]"               if fb.get('rating') else ""
                    lines.append(
                        f"  ✓ {fb['original']} → {fb['corrected']} ({fb['choice']})"
                        f"{rating_str}{saved_str}"
                    )

        lines.append("")

        # ── Risk analysis ────────────────────────────────────────────────
        analysis = results['analysis']
        lines.append("--- ANALIZA RIZIKA ---")

        if analysis['term_frequencies']:
            lines += [f"✓ Pronađeno {analysis['unique_terms_count']} jedinstvenih termina", "",
                      "Učestalost reči:"]
            for term, count in sorted(analysis['term_frequencies'].items(),
                                      key=lambda x: x[1], reverse=True):
                weight = (analyzer.term_to_weight.get(term, 1)
                          if analyzer and hasattr(analyzer, 'term_to_weight') else 1)
                lines.append(f"  - {term}: {count}x (težina: {weight})")
        else:
            lines.append("✓ Nije pronađeno reči od značaja")

        lines.append("")

        # ── Statistics ───────────────────────────────────────────────────
        lines += [
            "--- STATISTIKA ---",
            f"Ukupno reči u tekstu: {analysis['total_words']}",
            f"Ukupno pojavljivanja rizičnih termina: {analysis['total_occurrences']}",
            f"Jedinstvenih termina: {analysis['unique_terms_count']}",
            f"Gustina relevantnih reči: {analysis['term_density']:.2f}%",
            f"Ukupan skor (težinski): {analysis['total_score']}",
            f"Vreme obrade: {results['statistics']['processing_time_seconds']:.3f}s",
            ""
        ]

        # ── Risk assessment ──────────────────────────────────────────────
        lines += [
            "--- PROCENA RIZIKA ---",
            f"Nivo rizika: {analysis['risk_level']}",
            f"Opis: {analysis['risk_description']}",
            "",
            "Preporuke:"
        ]
        for i, rec in enumerate(analysis['recommendations'], 1):
            lines.append(f"{i}. {rec}")

        lines += ["", "=" * 70]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # JSON report
    # ------------------------------------------------------------------

    @staticmethod
    def generate_json_report(results: Dict[str, Any]) -> str:
        """Build a compact JSON report."""
        payload = {
            'metadata': results['metadata'],
            'summary': {
                'risk_level':         results['analysis']['risk_level'],
                'risk_description':   results['analysis']['risk_description'],
                'total_score':        results['analysis']['total_score'],
                'unique_terms_found': results['analysis']['unique_terms_count'],
                'total_occurrences':  results['analysis']['total_occurrences'],
                'term_density':       results['analysis']['term_density'],
                'processing_time':    results['statistics']['processing_time_seconds']
            },
            'terms_found': results['analysis']['term_frequencies'],
            'spell_checking': {
                'errors_found':        results['spell_checking']['errors_found'],
                'corrections_made':    results['spell_checking']['corrections_made'],
                'correction_feedback': results['spell_checking'].get('correction_feedback', [])
            }
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # Chat export report
    # ------------------------------------------------------------------

    @staticmethod
    def generate_chat_report(chat_analysis: dict,
                             overall_risk_level: str,
                             overall_risk_description: str,
                             overall_recommendations: list,
                             total_processing_time: float) -> str:
        """
        Build a full text report for a chat export analysis.

        The overall risk level is derived from the average score across ALL
        messages (flagged and unflagged), not from the combined-text analysis.
        """
        meta  = chat_analysis["chat_meta"]
        st    = chat_analysis["stats"]
        lines = []

        # ── Header ──────────────────────────────────────────────────────
        lines += [
            "=" * 70,
            "ANALIZA CHAT EXPORTA / CHAT EXPORT ANALYSIS",
            f"Chat: {meta['chat_name']}  |  Tip: {meta['chat_type']}",
            "=" * 70, "",
        ]

        # ── Overall risk ─────────────────────────────────────────────────
        avg  = st["average_score"]
        lines += [
            "--- PROCENA RIZIKA ČETA ---",
            f"Nivo rizika:  {overall_risk_level}",
            f"Opis:         {overall_risk_description}",
            f"Prosečan skor po poruci: {avg:.2f}  "
            f"(suma: {st['total_score_sum']} / {st['analysable_messages']} poruka)",
            "",
            "Preporuke:",
        ]
        for i, rec in enumerate(overall_recommendations, 1):
            lines.append(f"  {i}. {rec}")
        lines.append("")

        # ── Statistics ───────────────────────────────────────────────────
        lines += [
            "--- STATISTIKA ---",
            f"Ukupno poruka u exportu:  {st['total_messages_in_export']}",
            f"Analiziranih poruka:      {st['analysable_messages']}",
            f"Označenih (skor > 0):     {st['flagged_messages_count']}",
            f"Čistih poruka:            {st['clean_messages_count']}",
            f"Maksimalni skor:          {st['max_score']}",
            f"Vreme obrade:             {total_processing_time:.3f}s",
            "",
            "Raspodela rizika:",
        ]
        for level, count in sorted(st["risk_distribution"].items(),
                                   key=lambda x: x[1], reverse=True):
            lines.append(f"  {level}: {count}")
        lines.append("")

        # ── Top users ────────────────────────────────────────────────────
        top = st.get("top_users_by_score", [])
        if top:
            lines.append("--- TOP KORISNICI PO UKUPNOM SKORU ---")
            for u in top:
                lines.append(
                    f"  {u['user_name']} (ID: {u['user_id'] or '—'}): "
                    f"skor={u['total_score']}  označenih={u['flagged']}/{u['messages']}"
                )
            lines.append("")

        # ── Flagged messages ─────────────────────────────────────────────
        flagged = chat_analysis["flagged_messages"]
        if flagged:
            lines.append(f"--- OZNAČENE PORUKE ({len(flagged)}) ---")
            for m in flagged:
                lines += [
                    "",
                    f"  Korisnik:  {m['user_name']}  (ID: {m['user_id'] or '—'})",
                    f"  Datum:     {m['date']}",
                    f"  Nivo:      {m['risk_level']}  |  Skor: {m['total_score']}",
                    f"  Poruka:    {m['text']}",
                ]
                if m["term_frequencies"]:
                    terms = ", ".join(
                        f"{t}×{c}" for t, c in
                        sorted(m["term_frequencies"].items(),
                               key=lambda x: x[1], reverse=True)
                    )
                    lines.append(f"  Termini:   {terms}")
                if m.get("verdict"):
                    lines.append(f"  Nalaz:     {m['verdict']}")

        lines += ["", "=" * 70]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------

    @staticmethod
    def save_report(report_text: str, filename: str = OUTPUT_FILE) -> None:
        """Write *report_text* to *filename*."""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"Report saved to: {filename}")
        except Exception as e:
            print(f"Error saving report: {e}")