# -*- coding: utf-8 -*-
"""
WatchTower: Serbian Spell Checker
Supports Latin and Cyrillic scripts with a learned-corrections system.
"""

import os
import re
from typing import Optional, Tuple, List
from datetime import datetime

from constants import USER_CORRECTIONS_FILE, MAX_SUGGESTIONS

try:
    import phunspell
    PHUNSPELL_AVAILABLE = True
except ImportError:
    PHUNSPELL_AVAILABLE = False


class SerbianSpellChecker:
    """Serbian spell checker supporting both Latin and Cyrillic scripts with learning."""

    def __init__(self, user_corrections_file: str = USER_CORRECTIONS_FILE):
        self.latin_dict    = None
        self.cyrillic_dict = None
        self.initialized   = False
        self.user_corrections_file = user_corrections_file
        self.user_corrections = {}  # {original: (corrected, rating, count, last_used)}

        if PHUNSPELL_AVAILABLE:
            try:
                self.latin_dict    = phunspell.Phunspell('sr-Latn')  # type: ignore
                self.cyrillic_dict = phunspell.Phunspell('sr')        # type: ignore
                self.initialized   = True
                print("Spell checker initialized successfully.")
            except Exception as e:
                print(f"Warning: Could not initialize spell checker dictionaries: {e}")
                print("Spell checking will be disabled.")
        else:
            print("Spell checker disabled: Phunspell not available.")

        self._load_user_corrections()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_user_corrections(self):
        """Load user corrections from file."""
        if not os.path.exists(self.user_corrections_file):
            print(f"User corrections file not found. Will create: {self.user_corrections_file}")
            return

        try:
            with open(self.user_corrections_file, 'r', encoding='utf-8') as f:
                loaded_count = 0
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    try:
                        parts = line.split('|')
                        correction_part = parts[0].strip()
                        if '→' in correction_part:
                            original, corrected = correction_part.split('→')
                            original  = original.strip().lower()
                            corrected = corrected.strip()

                            rating    = 0
                            count     = 1
                            last_used = datetime.now().strftime('%Y-%m-%d')

                            for part in parts[1:]:
                                part = part.strip()
                                if part.startswith('rating:'):
                                    rating = int(part.split(':')[1])
                                elif part.startswith('count:'):
                                    count = int(part.split(':')[1])
                                elif part.startswith('last_used:'):
                                    last_used = part.split(':')[1]

                            self.user_corrections[original] = (corrected, rating, count, last_used)
                            loaded_count += 1
                    except Exception:
                        print(f"Warning: Could not parse correction line: {line}")

            if loaded_count > 0:
                print(f"Loaded {loaded_count} user corrections.")
        except Exception as e:
            print(f"Error loading user corrections: {e}")

    def _save_user_corrections(self):
        """Save user corrections to file."""
        try:
            with open(self.user_corrections_file, 'w', encoding='utf-8') as f:
                f.write("# WatchTower User Corrections Dictionary\n")
                f.write("# Format: original → corrected | rating:X | count:Y | last_used:DATE\n")
                f.write("# This file stores learned corrections from interactive mode\n#\n")

                sorted_corrections = sorted(
                    self.user_corrections.items(),
                    key=lambda x: x[1][2],  # Sort by count
                    reverse=True
                )

                for original, (corrected, rating, count, last_used) in sorted_corrections:
                    f.write(f"{original} → {corrected} | rating:{rating} | count:{count} | last_used:{last_used}\n")

            print(f"✓ Saved {len(self.user_corrections)} corrections to {self.user_corrections_file}")
        except Exception as e:
            print(f"Error saving user corrections: {e}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clear_corrections(self) -> int:
        """
        Clear all learned corrections from memory and delete the backing file.
        Returns the number of corrections that were removed.
        """
        count = len(self.user_corrections)
        self.user_corrections = {}
        if os.path.exists(self.user_corrections_file):
            try:
                os.remove(self.user_corrections_file)
            except Exception as e:
                print(f"Error deleting corrections file: {e}")
        return count

    def get_all_corrections(self) -> list:
        """
        Return all learned corrections as a list of dicts with keys:
        original, corrected, rating, count, last_used.
        """
        return [
            {
                'original':  original,
                'corrected': corrected,
                'rating':    rating,
                'count':     count,
                'last_used': last_used,
            }
            for original, (corrected, rating, count, last_used)
            in self.user_corrections.items()
        ]

    def add_user_correction(self, original: str, corrected: str, rating: int = 0):
        """Add or update a user correction and persist it immediately."""
        original_lower = original.lower()
        today = datetime.now().strftime('%Y-%m-%d')

        if original_lower in self.user_corrections:
            old_corrected, old_rating, old_count, _ = self.user_corrections[original_lower]
            new_rating = max(old_rating, rating) if rating > 0 else old_rating
            self.user_corrections[original_lower] = (corrected, new_rating, old_count + 1, today)
        else:
            self.user_corrections[original_lower] = (corrected, rating, 1, today)

        self._save_user_corrections()

    def get_user_correction(self, word: str) -> Optional[str]:
        """Return a previously learned correction for *word*, or None."""
        entry = self.user_corrections.get(word.lower())
        if entry:
            return entry[0]   # corrected form only — usage tracking is add_user_correction's job
        return None

    def check_word(self, word: str) -> Tuple[bool, Optional[List[str]]]:
        """
        Check spelling of a single word.
        Checks user corrections FIRST, then phunspell.

        Returns:
            (is_correct, suggestions)  — suggestions is None when the word is correct.
        """
        if not self.initialized or not self.latin_dict or not self.cyrillic_dict:
            return True, None

        clean_word = re.sub(r'[^\w\'-]', '', word, flags=re.UNICODE)
        if not clean_word or len(clean_word) < 2:
            return True, None

        # Check learned corrections first
        user_correction = self.get_user_correction(clean_word)
        if user_correction:
            return False, [user_correction]

        if self._is_cyrillic(clean_word):
            is_correct  = self.cyrillic_dict.lookup(clean_word)
            suggestions = list(self.cyrillic_dict.suggest(clean_word))[:MAX_SUGGESTIONS] if not is_correct else None
        elif self._is_latin(clean_word):
            is_correct  = self.latin_dict.lookup(clean_word)
            suggestions = list(self.latin_dict.suggest(clean_word))[:MAX_SUGGESTIONS] if not is_correct else None
        else:
            return True, None

        return is_correct, suggestions

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_cyrillic(word: str) -> bool:
        return any('\u0400' <= ch <= '\u04FF' for ch in word)

    @staticmethod
    def _is_latin(word: str) -> bool:
        return any(('\u0041' <= ch <= '\u007A' or '\u00C0' <= ch <= '\u02AF') for ch in word)