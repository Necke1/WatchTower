# -*- coding: utf-8 -*-
"""
WatchTower: Learned Context Patterns
=====================================
Automatically learns which word-contexts make a matched dictionary term
more or less risky — based on analyst feedback on real messages.

Analogy with the spell checker
-------------------------------
  Spell checker:     analyst confirms  "oruzjem → oružjem"
                     → saved to korekcija.txt
                     → applied automatically next time "oruzjem" appears

  Pattern learner:   analyst marks     "parking u centru" as false positive on "park"
                     → saved to korekcija_paterna.txt
                     → next time "park" appears near "parking", score → 0

File format (korekcija_paterna.txt)
-------------------------------------
  # term | context_words | multiplier | count:N | last_used:DATE
  park   | centru,parking | 0.00 | count:4 | last_used:2026-03-05
  napad  | organizujemo   | 2.00 | count:7 | last_used:2026-03-05

Multiplier meanings
--------------------
  0.00  — complete false positive (ignore this term in this context)
  0.15  — strong dampener (negation context)
  0.35  — mild dampener  (journalistic / academic context)
  1.00  — neutral (no change)
  2.00  — amplifier (confirmed threat context)
  3.00  — strong amplifier (direct operational planning)

Confidence & blending
----------------------
A single analyst vote should not immediately zero-out or double a term.
Confidence grows with repeated confirmation:

  count 1 → confidence 0.50  (tentative — multiplier blended 50% toward 1.0)
  count 3 → confidence 0.75
  count 5 → confidence 1.00  (fully trusted — raw multiplier applied)

Blending formula:
  effective = 1.0 + (raw_multiplier - 1.0) * confidence

Examples at count=1:
  raw=0.0  → effective=0.5   (not yet zeroed out, just halved)
  raw=2.0  → effective=1.5   (not yet doubled, but raised)

Examples at count=5:
  raw=0.0  → effective=0.0   (fully suppressed)
  raw=2.0  → effective=2.0   (fully amplified)

Priority over hand-written rules
----------------------------------
Learned patterns take PRIORITY over the CONTEXT_RULES in constants.py
because they come from real analyst-reviewed data and are more specific.
Hand-written rules act as a fallback when no learned pattern matches.
"""

import os
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import filelock as _filelock_module
    _FILELOCK_AVAILABLE = True
except ImportError:
    _FILELOCK_AVAILABLE = False

# Pattern reaches full confidence after this many confirmed uses
_FULL_CONFIDENCE_COUNT = 5

# Common Serbian words that carry no contextual signal — never saved
# as part of a pattern fingerprint.
_STOPWORDS = frozenset({
    'i', 'u', 'je', 'su', 'se', 'na', 'da', 'ne', 'a', 'ali',
    'ili', 'kao', 'za', 'od', 'do', 'iz', 'sa', 'po', 'pri',
    'to', 'taj', 'ta', 'te', 'ti', 'mu', 'ga', 'ju', 'bi', 'li',
    'već', 'još', 'baš', 'tek', 'sve', 'svi', 'ovaj', 'ova', 'ovo',
    'the', 'a', 'an', 'of', 'in', 'is', 'it', 'and', 'or',
})


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _confidence(count: int) -> float:
    """Return 0.50–1.0 confidence based on how many times a pattern was confirmed."""
    if count >= _FULL_CONFIDENCE_COUNT:
        return 1.0
    # Linear ramp: 0.50 at count=1, 1.0 at count=_FULL_CONFIDENCE_COUNT
    return 0.5 + (count - 1) * (0.5 / (_FULL_CONFIDENCE_COUNT - 1))


def _effective_multiplier(raw: float, count: int) -> float:
    """
    Blend the raw multiplier toward 1.0 proportional to confidence.
    Prevents a single analyst mistake from having full immediate effect.
    """
    conf = _confidence(count)
    return round(1.0 + (raw - 1.0) * conf, 4)


def _fingerprint(window_lemmas: List[str]) -> frozenset:
    """
    Extract the most signal-rich words from a context window.
    Filters stop-words and very short tokens (< 3 chars).
    Returns a frozenset used as the pattern key and for fuzzy matching.
    """
    return frozenset(
        w for w in window_lemmas
        if w not in _STOPWORDS and len(w) >= 3
    )


# ---------------------------------------------------------------------------
# LearnedPatternStore
# ---------------------------------------------------------------------------

class LearnedPatternStore:
    """
    Persisted store of learned (term, context) → multiplier patterns.

    Matching is fuzzy: a pattern fires if ANY of its stored context words
    appears in the live window around the term.  When multiple patterns
    match, the one with the most overlap (highest specificity) wins.
    """

    def __init__(self, patterns_file: str):
        self.patterns_file = patterns_file
        # key: (term_str, frozenset_of_context_words)
        # val: [raw_multiplier: float, count: int, last_used: str]
        self._patterns: Dict[Tuple[str, frozenset], list] = {}
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self):
        if not os.path.exists(self.patterns_file):
            logger.info("No learned patterns file yet: %s", self.patterns_file)
            return
        try:
            loaded = 0
            with open(self.patterns_file, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    try:
                        parts      = [p.strip() for p in line.split('|')]
                        term       = parts[0].lower()
                        ctx        = frozenset(w.strip() for w in parts[1].split(',') if w.strip())
                        multiplier = float(parts[2])
                        count      = int(parts[3].split(':')[1])
                        last_used  = parts[4].split(':', 1)[1]
                        self._patterns[(term, ctx)] = [multiplier, count, last_used]
                        loaded += 1
                    except Exception:
                        logger.warning("Skipping malformed pattern line: %s", line)
            if loaded:
                logger.info("Loaded %d learned context patterns.", loaded)
        except Exception as e:
            logger.error("Error loading learned patterns: %s", e)

    def _save(self):
        def _write():
            os.makedirs(os.path.dirname(os.path.abspath(self.patterns_file)), exist_ok=True)
            with open(self.patterns_file, 'w', encoding='utf-8') as f:
                f.write("# WatchTower Learned Context Patterns\n")
                f.write("# Format: term | context_words | multiplier | count:N | last_used:DATE\n")
                f.write("# multiplier 0.0=ignore  0.35=dampener 1.0=neutral 2.0=amplifier  3.0=strong\n#\n")
                for (term, ctx), (mult, count, last_used) in sorted(
                    self._patterns.items(), key=lambda x: x[1][1], reverse=True
                ):
                    f.write(
                        f"{term} | {','.join(sorted(ctx))} | {mult:.2f}"
                        f" | count:{count} | last_used:{last_used}\n"
                    )
        try:
            if _FILELOCK_AVAILABLE:
                with _filelock_module.FileLock(self.patterns_file + ".lock", timeout=5): # type: ignore
                    _write()
            else:
                _write()
        except Exception as e:
            logger.error("Error saving learned patterns: %s", e)

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def learn(self,
              term:          str,
              window_lemmas: List[str],
              multiplier:    float,
              label:         str = "") -> Optional[frozenset]:
        """
        Record analyst feedback as a reusable context pattern.

        Parameters
        ----------
        term          : The matched dictionary term (e.g. "napad").
        window_lemmas : Lemmas in the surrounding window (both sides).
        multiplier    : Score modifier for this context.
        label         : Human-readable label for the log.

        Returns the saved fingerprint frozenset, or None if the window
        contained only stop-words (nothing meaningful to learn from).
        """
        fp = _fingerprint(window_lemmas)
        if not fp:
            logger.info("Pattern for '%s' not saved — window had only stop-words.", term)
            return None

        term  = term.lower()
        today = datetime.now().strftime('%Y-%m-%d')
        key   = (term, fp)

        if key in self._patterns:
            old_mult, old_count, _ = self._patterns[key]
            # Weighted average: existing votes + 1 new vote
            new_mult = round(
                (old_mult * old_count + multiplier) / (old_count + 1), 3
            )
            self._patterns[key] = [new_mult, old_count + 1, today]
            logger.info(
                "Updated pattern '%s'+%s: %.2f→%.2f (count=%d) %s",
                term, set(fp), old_mult, new_mult, old_count + 1, label
            )
        else:
            self._patterns[key] = [multiplier, 1, today]
            logger.info(
                "New pattern '%s'+%s: multiplier=%.2f %s",
                term, set(fp), multiplier, label
            )

        self._save()
        return fp

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def get_multiplier(self,
                       term:          str,
                       window_lemmas: List[str]) -> Optional[float]:
        """
        Return the confidence-blended multiplier for this (term, context) pair,
        or None if no learned pattern matches.

        Fuzzy match: a pattern fires if ANY of its fingerprint words appears
        in window_lemmas.  When multiple patterns match, the one with the
        greatest word overlap (most specific) wins.
        """
        term        = term.lower()
        live_window = set(window_lemmas)
        best: Optional[Tuple[int, float]] = None   # (overlap_count, eff_multiplier)

        for (stored_term, fp), (raw_mult, count, _) in self._patterns.items():
            if stored_term != term:
                continue
            overlap = len(fp & live_window)
            if overlap == 0:
                continue
            eff = _effective_multiplier(raw_mult, count)
            if best is None or overlap > best[0]:
                best = (overlap, eff)

        return best[1] if best else None

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_all_patterns(self) -> List[dict]:
        return [
            {
                'term':        term,
                'context':     sorted(ctx),
                'multiplier':  mult,
                'effective_at_count_1': round(_effective_multiplier(mult, 1), 3),
                'effective_at_full':    round(_effective_multiplier(mult, _FULL_CONFIDENCE_COUNT), 3),
                'count':       count,
                'confidence':  round(_confidence(count), 2),
                'last_used':   last_used,
                'label': (
                    'Ignoriši (lažni alarm)'        if mult <= 0.1 else
                    'Pojačivač (potvrđena pretnja)'  if mult >= 1.5 else
                    'Prigušivač (kontekst)'          if mult < 1.0 else
                    'Neutralno'
                ),
            }
            for (term, ctx), (mult, count, last_used) in sorted(
                self._patterns.items(), key=lambda x: x[1][1], reverse=True
            )
        ]

    def clear(self) -> int:
        count = len(self._patterns)
        self._patterns = {}
        for f in [self.patterns_file, self.patterns_file + ".lock"]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except Exception as e:
                    logger.error("Error deleting %s: %s", f, e)
        return count