# -*- coding: utf-8 -*-
"""
WatchTower: Text Analyzer  (Optimized)
Core analysis engine for Serbian content risk assessment.

Performance improvements over v1.1:
  1. Cyrillic → Latin via str.translate()      — ~20x faster than char loop
  2. PrefixTrie for term matching               — O(word_len) vs O(words × dict)
  3. Spell-check deduplication cache            — check each unique word once
  4. Spell-check word-count threshold           — auto-skip above 50k words
  5. Pre-compiled regex patterns                — avoid recompile on every call
  6. CLASSLA chunked batching                   — avoids single giant document
  7. Emoji set pre-separated at dict load       — no per-call regex on terms
"""

import os
import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter
from datetime import datetime

logger = logging.getLogger(__name__)

from constants import (
    DEFAULT_DICTIONARY_FILE,
    RISK_THRESHOLDS,
    RISK_LEVELS,
)
from spell_checker import SerbianSpellChecker

try:
    import classla
    CLASSLA_AVAILABLE = True
except ImportError:
    logger.warning("CLASSLA-Stanza not found. Install with: pip install classla")
    CLASSLA_AVAILABLE = False


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Above this word count spell checking is automatically skipped for speed.
# A 50 000-word document still gets full spell checking; a 1M-word bulk feed
# almost certainly doesn't need it and phunspell would dominate runtime.
SPELL_CHECK_WORD_LIMIT = 50_000

# CLASSLA is fed in chunks so it never has to parse a giant single document.
# Each chunk is ~500 sentences; tune lower if you hit memory limits.
CLASSLA_CHUNK_SENTENCES = 500

# Pre-compiled patterns (compiled once at import time, not per call).
_RE_WORDS      = re.compile(r'\b\w+\b', re.UNICODE)
_RE_WHITESPACE = re.compile(r'\s+')
_RE_WORD_ONLY  = re.compile(r'^\w+$', re.UNICODE)

# ---------------------------------------------------------------------------
# Cyrillic → Latin translation table (built once at import time)
# ---------------------------------------------------------------------------
# str.maketrans() with a dict accepts multi-character replacement values,
# so the digraph letters (Lj, Nj, Dž) are handled correctly and the entire
# conversion is done in a single C-level pass through the string (~20x faster
# than the manual Python while-loop used in v1.1).
#
# Note: Љ (U+0409), Њ (U+040A), Џ (U+040F) etc. are each a single Unicode
# codepoint — NOT two characters — so str.translate handles them perfectly.

_CYRILLIC_TABLE = str.maketrans({
    'А': 'A',  'Б': 'B',  'В': 'V',  'Г': 'G',  'Д': 'D',
    'Ђ': 'Đ',  'Е': 'E',  'Ж': 'Ž',  'З': 'Z',  'И': 'I',
    'Ј': 'J',  'К': 'K',  'Л': 'L',  'Љ': 'Lj', 'М': 'M',
    'Н': 'N',  'Њ': 'Nj', 'О': 'O',  'П': 'P',  'Р': 'R',
    'С': 'S',  'Т': 'T',  'Ћ': 'Ć',  'У': 'U',  'Ф': 'F',
    'Х': 'H',  'Ц': 'C',  'Ч': 'Č',  'Џ': 'Dž', 'Ш': 'Š',
    'а': 'a',  'б': 'b',  'в': 'v',  'г': 'g',  'д': 'd',
    'ђ': 'đ',  'е': 'e',  'ж': 'ž',  'з': 'z',  'и': 'i',
    'ј': 'j',  'к': 'k',  'л': 'l',  'љ': 'lj', 'м': 'm',
    'н': 'n',  'њ': 'nj', 'о': 'o',  'п': 'p',  'р': 'r',
    'с': 's',  'т': 't',  'ћ': 'ć',  'у': 'u',  'ф': 'f',
    'х': 'h',  'ц': 'c',  'ч': 'č',  'џ': 'dž', 'ш': 'š',
})

# ---------------------------------------------------------------------------
# Latin → Cyrillic conversion (used when mixed-script text is detected)
# ---------------------------------------------------------------------------
# str.maketrans() does not support multi-character keys, so Serbian digraphs
# (lj→љ, nj→њ, dž→џ and their uppercase variants) must be substituted with
# str.replace() BEFORE the single-character str.translate() pass.
# Order matters: longer/more-specific patterns must come before shorter ones
# that share a prefix (e.g. 'lj' before 'l').

_DIGRAPH_TO_CYRILLIC = [
    ('LJ', 'Љ'), ('NJ', 'Њ'), ('DŽ', 'Џ'),   # all-caps
    ('Lj', 'Љ'), ('Nj', 'Њ'), ('Dž', 'Џ'),   # title-case
    ('lj', 'љ'), ('nj', 'њ'), ('dž', 'џ'),   # lowercase
]

_LATIN_SINGLE_TABLE = str.maketrans({
    'A': 'А', 'B': 'Б', 'C': 'Ц', 'Č': 'Ч', 'Ć': 'Ћ',
    'D': 'Д', 'Đ': 'Ђ', 'E': 'Е', 'F': 'Ф', 'G': 'Г',
    'H': 'Х', 'I': 'И', 'J': 'Ј', 'K': 'К', 'L': 'Л',
    'M': 'М', 'N': 'Н', 'O': 'О', 'P': 'П', 'R': 'Р',
    'S': 'С', 'Š': 'Ш', 'T': 'Т', 'U': 'У', 'V': 'В',
    'Z': 'З', 'Ž': 'Ж',
    'a': 'а', 'b': 'б', 'c': 'ц', 'č': 'ч', 'ć': 'ћ',
    'd': 'д', 'đ': 'ђ', 'e': 'е', 'f': 'ф', 'g': 'г',
    'h': 'х', 'i': 'и', 'j': 'ј', 'k': 'к', 'l': 'л',
    'm': 'м', 'n': 'н', 'o': 'о', 'p': 'п', 'r': 'р',
    's': 'с', 'š': 'ш', 't': 'т', 'u': 'у', 'v': 'в',
    'z': 'з', 'ž': 'ж',
})


# ---------------------------------------------------------------------------
# Trie (prefix tree) for O(word_length) term matching
# ---------------------------------------------------------------------------

class PrefixTrie:
    """
    Dictionary terms are inserted as paths in the trie.
    `find_prefix_match(word)` walks the trie character-by-character along
    the word and returns the longest dictionary term that is a prefix of
    that word (or an exact match).

    Complexity:
        insert  — O(term_length)
        lookup  — O(word_length)   ← vs O(dict_size × word_length) before

    For a 500-term dictionary and 1M-word document the trie reduces the
    matching work from ~500M comparisons to ~7M character steps.
    """

    _END = '$'   # terminal marker stored inside a trie node dict

    def __init__(self):
        self.root: dict = {}

    def insert(self, term: str) -> None:
        node = self.root
        for ch in term:
            node = node.setdefault(ch, {})
        node[self._END] = term   # store full term at terminal for easy retrieval

    def find_prefix_match(self, word: str) -> Optional[str]:
        """
        Return the (longest) dictionary term that is a prefix of *word*,
        or None if no such term exists.
        An exact match (word == term) is also returned here.
        """
        node    = self.root
        matched = None
        for ch in word:
            if ch not in node:
                break
            node = node[ch]
            if self._END in node:
                matched = node[self._END]   # keep updating → longest match wins
        return matched


# ---------------------------------------------------------------------------
# TextAnalyzer
# ---------------------------------------------------------------------------

class TextAnalyzer:
    """Main text analysis engine for Serbian content (optimized)."""

    def __init__(self,
                 dictionary_file:    str  = DEFAULT_DICTIONARY_FILE,
                 use_spellcheck:     bool = True,
                 auto_correct:       bool = False,
                 interactive_correct: bool = False,
                 use_gpu:            bool = False):

        self.dictionary_file     = dictionary_file
        self.words_set           = set()
        self.term_to_weight      = {}
        self.emoji_terms:  set   = set()
        self._emoji_pattern      = None   # compiled regex; built by load_dictionary()
        self._trie               = PrefixTrie()

        self.use_spellcheck      = use_spellcheck
        self.auto_correct        = auto_correct
        self.interactive_correct = interactive_correct
        self.use_gpu             = use_gpu

        if self.auto_correct and self.interactive_correct:
            logger.warning("Both auto-correct and interactive-correct enabled. "
                           "Interactive mode takes precedence.")
            self.auto_correct = False

        # NLP pipeline
        self.nlp_pipeline = None
        if CLASSLA_AVAILABLE:
            try:
                self.nlp_pipeline = classla.Pipeline(   # type: ignore
                    'sr',
                    processors='tokenize,pos,lemma',
                    use_gpu=use_gpu,
                )
                logger.info("NLP pipeline initialized successfully.")
            except Exception as e:
                logger.warning("Could not initialize NLP pipeline: %s", e)
        else:
            logger.info("NLP pipeline disabled: CLASSLA not available.")

        self.spell_checker = SerbianSpellChecker() if use_spellcheck else None

        self.load_dictionary()

    # ------------------------------------------------------------------
    # Dictionary
    # ------------------------------------------------------------------

    def load_dictionary(self) -> None:
        """
        Load risky words + weights from file and build the trie.
        Format: one term per line, optional integer weight after a space.
        Lines starting with # are comments.
        Weights must be positive integers; zero/negative values are skipped.
        """
        self.words_set      = set()
        self.term_to_weight = {}
        self.emoji_terms    = set()
        self._emoji_pattern = None
        self._trie          = PrefixTrie()

        if not os.path.exists(self.dictionary_file):
            logger.warning("Dictionary file '%s' not found.", self.dictionary_file)
            self._create_default_dictionary()
            return

        try:
            with open(self.dictionary_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue

                    parts  = line.split()
                    term   = parts[0].lower()
                    weight = 1
                    if len(parts) >= 2:
                        try:
                            weight = int(parts[1])
                            if weight <= 0:
                                logger.warning(
                                    "Zero/negative weight ignored on line %d: %s",
                                    line_num, line
                                )
                                continue
                        except ValueError:
                            logger.warning("Invalid weight on line %d: %s", line_num, line)
                            continue

                    self.words_set.add(term)
                    self.term_to_weight[term] = weight

                    if _RE_WORD_ONLY.match(term):
                        self._trie.insert(term)
                    else:
                        self.emoji_terms.add(term)

            # Compile a single combined regex for all emoji/special terms so
            # find_all_risky_terms can use one finditer() pass instead of N
            # individual str.count() calls.
            if self.emoji_terms:
                self._emoji_pattern = re.compile(
                    '|'.join(re.escape(e) for e in self.emoji_terms)
                )

            logger.info(
                "Loaded %d terms from dictionary (%d emoji/special).",
                len(self.words_set), len(self.emoji_terms)
            )
        except Exception as e:
            logger.error("Error loading dictionary: %s", e)
            self._create_default_dictionary()

    def _create_default_dictionary(self) -> None:
        default_terms = {
            'terorizam': 10, 'terorista': 9,  'bomba':      8,
            'nasilje':    7, 'ekstremizam': 8, 'mržnja':     6,
            'ubistvo':    9, 'napad':       7, '💣':         8, '🔫': 7,
        }
        self.words_set      = set(default_terms)
        self.term_to_weight = dict(default_terms)
        self.emoji_terms    = {'💣', '🔫'}
        self._trie          = PrefixTrie()
        for t in default_terms:
            if _RE_WORD_ONLY.match(t):
                self._trie.insert(t)
        self._emoji_pattern = re.compile(
            '|'.join(re.escape(e) for e in self.emoji_terms)
        )
        logger.info("Using default dictionary with %d terms.", len(self.words_set))

    # ------------------------------------------------------------------
    # Text processing
    # ------------------------------------------------------------------

    @staticmethod
    def detect_script(text: str) -> str:
        """
        Return 'cyrillic', 'latin', or 'mixed' based on alphabetic character
        proportions.  Non-alphabetic characters (digits, punctuation, emoji)
        are ignored.  A text with no alphabetic content is treated as 'latin'
        so the rest of the pipeline does nothing harmful to it.

        Thresholds: >95% Cyrillic → 'cyrillic', <5% Cyrillic → 'latin',
        anything in between → 'mixed'.
        """
        cyrillic = sum(1 for c in text if '\u0400' <= c <= '\u04FF')
        total    = sum(1 for c in text if c.isalpha())
        if total == 0:
            return 'latin'
        ratio = cyrillic / total
        if ratio > 0.95:
            return 'cyrillic'
        if ratio < 0.05:
            return 'latin'
        return 'mixed'

    @staticmethod
    def latin_to_cyrillic(text: str) -> str:
        """
        Convert Serbian Latin text to Cyrillic.

        Digraphs (lj→љ, nj→њ, dž→џ and their uppercase variants) are
        substituted first via str.replace() so that the subsequent single-
        character str.translate() pass does not incorrectly split them
        (e.g. 'lj' must become 'љ' not 'лј').

        Non-Serbian Latin characters (q, w, x, y, digits, punctuation) are
        left unchanged by both passes.
        """
        for latin, cyrillic in _DIGRAPH_TO_CYRILLIC:
            text = text.replace(latin, cyrillic)
        return text.translate(_LATIN_SINGLE_TABLE)

    @staticmethod
    def cyrillic_to_latin(text: str) -> str:
        """
        Convert Serbian Cyrillic text to Latin using str.translate().
        Single C-level pass — ~20x faster than the v1.1 Python while-loop.
        """
        return text.translate(_CYRILLIC_TABLE)

    def normalize_text(self, text: str) -> str:
        """
        Cyrillic → Latin + collapse whitespace.
        Called AFTER spell checking so it only normalises a clean text.
        Safe to call on already-Latin text (translate is a no-op on non-Cyrillic chars).
        """
        text = self.cyrillic_to_latin(text)
        text = _RE_WHITESPACE.sub(' ', text)
        return text.strip()

    def extract_lemmas(self, text: str) -> List[str]:
        """
        Return lemmas using the NLP pipeline when available.
        Falls back to simple regex tokenisation.
        For large texts CLASSLA is fed in sentence-sized chunks to avoid
        loading one massive document into memory at once.
        """
        if not self.nlp_pipeline:
            return [w.lower() for w in _RE_WORDS.findall(text)]

        try:
            # Split into rough sentence chunks to keep memory bounded.
            sentences  = text.split('.')
            lemmas: List[str] = []

            for start in range(0, len(sentences), CLASSLA_CHUNK_SENTENCES):
                chunk     = '. '.join(sentences[start:start + CLASSLA_CHUNK_SENTENCES])
                if not chunk.strip():
                    continue
                doc       = self.nlp_pipeline(chunk)
                for sentence in doc.sentences:
                    for word in sentence.words:
                        lemmas.append(
                            word.lemma.lower() if word.lemma else word.text.lower()
                        )
            return lemmas

        except Exception as e:
            logger.error("Error in NLP processing: %s", e)
            return [w.lower() for w in _RE_WORDS.findall(text)]

    # ------------------------------------------------------------------
    # Spell checking
    # ------------------------------------------------------------------

    def spell_check_text(self, text: str) -> Dict[str, Any]:
        """
        Run spell checking with a per-unique-word cache so each word is
        sent to phunspell exactly once regardless of how many times it
        appears in the text.

        Auto-skipped when text exceeds SPELL_CHECK_WORD_LIMIT words.
        """
        empty = {
            'original_text': text, 'corrected_text': text,
            'errors_found': 0, 'corrections_made': 0,
            'error_details': [], 'correction_feedback': [],
        }
        if not self.spell_checker or not self.use_spellcheck:
            return empty

        all_words = _RE_WORDS.findall(text)

        # Auto-threshold: skip spell check for very large texts
        if len(all_words) > SPELL_CHECK_WORD_LIMIT:
            logger.info(
                "Spell checking skipped (text has %s words > limit of %s).",
                f"{len(all_words):,}", f"{SPELL_CHECK_WORD_LIMIT:,}"
            )
            return empty

        # --- Build per-unique-word result cache ----------------------------
        # This is the core optimization: instead of calling phunspell N times
        # for a word that appears N times, we call it exactly once.
        unique_words = {w for w in all_words if len(w) >= 2}
        spell_cache: Dict[str, Tuple[bool, Optional[List[str]]]] = {}
        for word in unique_words:
            spell_cache[word] = self.spell_checker.check_word(word)

        # --- Collect errors (deduplicated) ---------------------------------
        error_details       = []
        corrections         = []
        correction_feedback = []
        seen_errors: set    = set()

        if self.interactive_correct:
            n_errors = sum(1 for w in unique_words
                           if not spell_cache[w][0] and len(w) >= 2)
            if n_errors:
                print(f"\n{'='*60}\nINTERAKTIVNI MOD KOREKCIJE")
                print(f"Pronađeno {n_errors} potencijalnih grešaka (jedinstvenih).")
                print("  Broj (1-5) = Primeni predlog | 0 = Zadrži | m = Ručno")
                print(f"{'='*60}\n")

        for word in all_words:
            if len(word) < 2:
                continue
            is_correct, suggestions = spell_cache.get(word, (True, None))
            if is_correct:
                continue
            if word in seen_errors:
                continue   # report each misspelling once
            seen_errors.add(word)
            error_details.append({'word': word, 'suggestions': suggestions})

            corrected_word = word
            user_choice    = None
            rating         = None

            if self.interactive_correct and suggestions:
                print(f"\n❌ Greška: '{word}'")
                for i, s in enumerate(suggestions[:5], 1):
                    print(f"   {i}. {s}")
                print("   0. Zadrži original  |  m. Unesi ručno")
                while True:
                    choice = input("   Opcija (1-5/0/m): ").strip().lower()
                    if choice == '0':
                        user_choice = 'kept_original'
                        break
                    elif choice == 'm':
                        manual = input("   Ispravka: ").strip()
                        if manual:
                            corrected_word = manual
                            user_choice    = 'manual_correction'
                        break
                    elif choice.isdigit() and 1 <= int(choice) <= len(suggestions[:5]):
                        corrected_word = suggestions[int(choice) - 1]
                        user_choice    = f'suggestion_{choice}'
                        r_in = input("   Ocena (1-4): ").strip()
                        if r_in.isdigit() and 1 <= int(r_in) <= 4:
                            rating = int(r_in)
                        break
                    else:
                        print("   ❌ Nevalidna opcija.")

            elif self.auto_correct and suggestions:
                corrected_word = suggestions[0]
                user_choice    = 'auto_corrected'

            if corrected_word != word:
                corrections.append((word, corrected_word))
                correction_feedback.append({
                    'original': word, 'corrected': corrected_word,
                    'choice': user_choice, 'rating': rating,
                    'suggestions': suggestions,
                })

        # Apply all corrections in one pass
        corrected_text = text
        for original, corrected in corrections:
            pattern        = r'\b' + re.escape(original) + r'\b'
            corrected_text = re.sub(pattern, corrected, corrected_text,
                                    flags=re.IGNORECASE | re.UNICODE)

        if self.interactive_correct and correction_feedback:
            print(f"\n{'='*60}\nREZIME KOREKCIJA")
            print(f"Ispravljeno: {len(corrections)}/{len(error_details)}")
            rated = [f for f in correction_feedback if f.get('rating')]
            if rated:
                avg = sum(f['rating'] for f in rated) / len(rated)
                print(f"Prosečna ocena: {avg:.1f}/4")
            print(f"{'='*60}\n")

        return {
            'original_text':       text,
            'corrected_text':      corrected_text,
            'errors_found':        len(error_details),
            'corrections_made':    len(corrections),
            'error_details':       error_details,
            'correction_feedback': correction_feedback,
        }

    # ------------------------------------------------------------------
    # Risk analysis
    # ------------------------------------------------------------------

    def find_all_risky_terms(self,
                             text:   str,
                             lemmas: List[str]) -> Tuple[Dict[str, int], int]:
        """
        Unified search for risky terms.

        Word terms  → trie lookup: O(word_length) per lemma
        Emoji terms → single compiled regex finditer() over raw text

        Returns (term_frequencies, total_weighted_score).
        """
        found_terms = Counter()
        total_score = 0

        # ── Word / lemma matching via trie ─────────────────────────────
        for lemma in lemmas:
            matched = self._trie.find_prefix_match(lemma)
            if matched:
                found_terms[lemma] += 1
                total_score += self.term_to_weight.get(matched, 1)

        # ── Emoji / special-char matching via single regex pass ────────
        if self._emoji_pattern:
            for match in self._emoji_pattern.finditer(text):
                term = match.group(0)
                found_terms[term] += 1
                total_score += self.term_to_weight.get(term, 1)

        return dict(found_terms), total_score

    def calculate_risk_level(self,
                             total_score:  int,
                             unique_terms: int,
                             total_words:  int) -> Tuple[str, str]:
        density = (unique_terms / max(total_words, 1)) * 100

        if (total_score  >= RISK_THRESHOLDS['high']['score'] or
                unique_terms >= RISK_THRESHOLDS['high']['unique_terms']):
            level = 'VISOK RIZIK'
        elif (total_score  >= RISK_THRESHOLDS['medium']['score'] or
              unique_terms >= RISK_THRESHOLDS['medium']['unique_terms']):
            level = 'SREDNJI RIZIK'
        elif (total_score  >= RISK_THRESHOLDS['low']['score'] or
              unique_terms >= RISK_THRESHOLDS['low']['unique_terms']):
            level = 'NIZAK RIZIK'
        elif total_score > 0:
            level = 'MINIMALAN RIZIK'
        else:
            level = 'BEZ RIZIKA'

        description = RISK_LEVELS[level]['description']
        if level in ('VISOK RIZIK', 'SREDNJI RIZIK') and density > 5:
            description += f" (gustina: {density:.1f}%)"

        return level, description

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Full analysis pipeline.

        Pipeline
        --------
        1. Detect script ('latin', 'cyrillic', 'mixed').
        2. Mixed → convert fully to Cyrillic so both halves use the same dict.
        3. Spell check in the native/unified script.
        4. Normalise corrected text to Latin (cyrillic_to_latin + whitespace).
        5. Lemmatise + term match on the Latin-normalised text.
        """
        start_time = datetime.now()

        original_text = text

        # Step 1+2: detect script; unify mixed text to Cyrillic before spell check
        script      = self.detect_script(text)
        spell_input = self.latin_to_cyrillic(text) if script == 'mixed' else text

        # Step 3: spell check in the original/unified script
        spell_results = self.spell_check_text(spell_input)
        corrected     = spell_results['corrected_text']

        # Step 4: normalise corrected text to Latin
        normalized  = self.normalize_text(corrected)

        # Step 5: lemmatise + term match on Latin-normalised text
        lemmas      = self.extract_lemmas(normalized)
        total_words = len(lemmas)

        term_frequencies, total_score = self.find_all_risky_terms(normalized, lemmas)

        unique_terms_count = len(term_frequencies)
        total_occurrences  = sum(term_frequencies.values())
        term_density       = (total_occurrences / max(total_words, 1)) * 100

        risk_level, risk_description = self.calculate_risk_level(
            total_score, unique_terms_count, total_words
        )

        processing_time = (datetime.now() - start_time).total_seconds()

        return {
            'original_text':  original_text,
            'processed_text': normalized,
            'spell_checking': spell_results,
            'analysis': {
                'total_words':        total_words,
                'term_frequencies':   term_frequencies,
                'unique_terms_count': unique_terms_count,
                'total_occurrences':  total_occurrences,
                'total_score':        total_score,
                'term_density':       term_density,
                'risk_level':         risk_level,
                'risk_description':   risk_description,
                'recommendations':    RISK_LEVELS[risk_level]['recommendations'],
            },
            'statistics': {
                'processing_time_seconds': processing_time,
                'errors_found':            spell_results['errors_found'],
                'corrections_made':        spell_results['corrections_made'],
            },
            'metadata': {
                'timestamp':                   start_time.isoformat(),
                'dictionary_size':             len(self.words_set),
                'spell_check_enabled':         self.use_spellcheck,
                'auto_correct_enabled':        self.auto_correct,
                'interactive_correct_enabled': self.interactive_correct,
                'detected_script':             script,
            },
        }