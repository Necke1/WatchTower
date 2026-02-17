#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WatchTower: Serbian Text Analysis Tool for Content Risk Assessment
Version: 1.1 - Optimized
Author: Nemanja Mosurović (@Necke1)
Date: 2025
Project Phase: Part 1 - Core Analysis Engine (Optimized)

This tool analyzes Serbian text for potentially extremist content.
Optimizations: Unified term search, weight caching, reduced redundancy.
"""

import os
import sys
import re
import json
import argparse
from typing import Dict, List, Tuple, Set, Optional, Any
from collections import Counter
from datetime import datetime

# External libraries
try:
    import classla
    CLASSLA_AVAILABLE = True
except ImportError:
    print("Warning: CLASSLA-Stanza not found. Install with: pip install classla")
    CLASSLA_AVAILABLE = False

try:
    import phunspell
    PHUNSPELL_AVAILABLE = True
except ImportError:
    print("Warning: Phunspell not found. Install with: pip install phunspell")
    PHUNSPELL_AVAILABLE = False

# Constants
DEFAULT_DICTIONARY_FILE = "recnik.txt"
USER_CORRECTIONS_FILE = "korekcija.txt"
OUTPUT_FILE = "rezultat.txt"
MAX_SUGGESTIONS = 5
RISK_THRESHOLDS = {
    'high': {'score': 50, 'unique_terms': 10},
    'medium': {'score': 20, 'unique_terms': 5},
    'low': {'score': 10, 'unique_terms': 3}
}

# Risk level descriptions in Serbian
RISK_LEVELS = {
    'VISOK RIZIK': {
        'description': 'Tekst sadrži značajan broj reči koje ukazuju na ekstremizam',
        'recommendations': [
            'Hitna provera od strane moderatora',
            'Potencijalno prijaviti nadležnim organima',
            'Preventivno blokiranje sadržaja dok se ne proveri'
        ]
    },
    'SREDNJI RIZIK': {
        'description': 'Tekst sadrži umerenu količinu sumnjivog sadržaja',
        'recommendations': [
            'Detaljna provera konteksta',
            'Praćenje dalje aktivnosti autora',
            'Podizanje prioriteta za moderaciju'
        ]
    },
    'NIZAK RIZIK': {
        'description': 'Tekst sadrži mali broj relevantnih reči',
        'recommendations': [
            'Ostaviti u sistemu za monitoring',
            'Proveriti ukoliko se pojave slični sadržaji',
            'Bez hitne akcije'
        ]
    },
    'MINIMALAN RIZIK': {
        'description': 'Tekst sadrži pojedinačne relevantne reči',
        'recommendations': [
            'Verovatno bezopasno',
            'Može biti deo normalnog diskursa',
            'Nema potrebe za akcijom'
        ]
    },
    'BEZ RIZIKA': {
        'description': 'Tekst ne sadrži reči od značaja',
        'recommendations': [
            'Nema potrebe za daljom proverom',
            'Standardni monitoring',
            'Nema akcije potrebne'
        ]
    }
}


class SerbianSpellChecker:
    """Serbian spell checker supporting both Latin and Cyrillic scripts with learning."""
    
    def __init__(self, latin_dict_path: Optional[str] = None, 
                 cyrillic_dict_path: Optional[str] = None,
                 user_corrections_file: str = USER_CORRECTIONS_FILE):
        """
        Initialize Serbian spell checker with dictionaries.
        
        Args:
            latin_dict_path: Path to Latin script dictionary
            cyrillic_dict_path: Path to Cyrillic script dictionary
            user_corrections_file: Path to user corrections file
        """
        self.latin_dict = None
        self.cyrillic_dict = None
        self.initialized = False
        self.user_corrections_file = user_corrections_file
        self.user_corrections = {}  # {original: (corrected, rating, count, last_used)}
        
        if PHUNSPELL_AVAILABLE:
            try:
                self.latin_dict = phunspell.Phunspell('sr-Latn') # type: ignore
                self.cyrillic_dict = phunspell.Phunspell('sr') # type: ignore
                self.initialized = True
                print("Spell checker initialized successfully.")
            except Exception as e:
                print(f"Warning: Could not initialize spell checker dictionaries: {e}")
                print("Spell checking will be disabled.")
        else:
            print("Spell checker disabled: Phunspell not available.")
        
        # Load user corrections
        self._load_user_corrections()
    
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
                    # Skip empty lines and comments
                    if not line or line.startswith('#'):
                        continue
                    
                    # Parse: original → corrected | rating:X | count:Y | last_used:DATE
                    try:
                        parts = line.split('|')
                        if len(parts) >= 1:
                            correction_part = parts[0].strip()
                            if '→' in correction_part:
                                original, corrected = correction_part.split('→')
                                original = original.strip().lower()
                                corrected = corrected.strip()
                                
                                # Parse metadata
                                rating = 0
                                count = 1
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
                    except Exception as e:
                        print(f"Warning: Could not parse correction line: {line}")
                        continue
            
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
                f.write("# This file stores learned corrections from interactive mode\n")
                f.write("#\n")
                
                # Sort by count (most used first)
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
    
    def add_user_correction(self, original: str, corrected: str, rating: int = 0):
        """
        Add or update a user correction.
        
        Args:
            original: Original word
            corrected: Corrected word
            rating: User rating (1-4)
        """
        original_lower = original.lower()
        today = datetime.now().strftime('%Y-%m-%d')
        
        if original_lower in self.user_corrections:
            # Update existing correction
            old_corrected, old_rating, old_count, _ = self.user_corrections[original_lower]
            new_count = old_count + 1
            # Keep highest rating
            new_rating = max(old_rating, rating) if rating > 0 else old_rating
            self.user_corrections[original_lower] = (corrected, new_rating, new_count, today)
        else:
            # Add new correction
            self.user_corrections[original_lower] = (corrected, rating, 1, today)
        
        # Save to file immediately
        self._save_user_corrections()
    
    def get_user_correction(self, word: str) -> Optional[str]:
        """
        Check if user has a learned correction for this word.
        
        Args:
            word: Word to check
            
        Returns:
            Corrected word if found, None otherwise
        """
        word_lower = word.lower()
        if word_lower in self.user_corrections:
            corrected, rating, count, last_used = self.user_corrections[word_lower]
            # Update usage count and date
            today = datetime.now().strftime('%Y-%m-%d')
            self.user_corrections[word_lower] = (corrected, rating, count + 1, today)
            return corrected
        return None
    
    def _is_cyrillic(self, word: str) -> bool:
        """Check if a word is written in Cyrillic script."""
        return any('\u0400' <= char <= '\u04FF' for char in word)
    
    def _is_latin(self, word: str) -> bool:
        """Check if a word is written in Latin script."""
        return any(('\u0041' <= char <= '\u007A' or '\u00C0' <= char <= '\u02AF') for char in word)
    
    def check_word(self, word: str) -> Tuple[bool, Optional[List[str]]]:
        """
        Check spelling of a single word.
        Checks user corrections FIRST, then phunspell.
        
        Args:
            word: Word to check
            
        Returns:
            Tuple of (is_correct, suggestions)
        """
        if not self.initialized or not self.latin_dict or not self.cyrillic_dict:
            return True, None
        
        clean_word = re.sub(r'[^\w\'-]', '', word, flags=re.UNICODE)
        
        if not clean_word or len(clean_word) < 2:
            return True, None
        
        # LEARNING MODE: Check user corrections first
        user_correction = self.get_user_correction(clean_word)
        if user_correction:
            # User has corrected this before - it's "correct" now
            # But return the learned correction as first suggestion
            return False, [user_correction]
        
        # Determine script and check spelling with phunspell
        if self._is_cyrillic(clean_word):
            is_correct = self.cyrillic_dict.lookup(clean_word)
            suggestions = list(self.cyrillic_dict.suggest(clean_word))[:MAX_SUGGESTIONS] if not is_correct else None
        elif self._is_latin(clean_word):
            is_correct = self.latin_dict.lookup(clean_word)
            suggestions = list(self.latin_dict.suggest(clean_word))[:MAX_SUGGESTIONS] if not is_correct else None
        else:
            return True, None
        
        return is_correct, suggestions


class TextAnalyzer:
    """Main text analysis engine for Serbian content."""
    
    def __init__(self, dictionary_file: str = DEFAULT_DICTIONARY_FILE,
                 use_spellcheck: bool = True,
                 auto_correct: bool = False,
                 interactive_correct: bool = False):
        """
        Initialize text analyzer.
        
        Args:
            dictionary_file: Path to dictionary file with weights
            use_spellcheck: Enable/disable spell checking
            auto_correct: Enable automatic correction of spelling errors
            interactive_correct: Enable interactive correction with user confirmation
        """
        self.dictionary_file = dictionary_file
        self.words_set = set()
        self.weights_dict = {}
        self.term_to_weight = {}  # OPTIMIZATION: O(1) weight lookup cache
        self.use_spellcheck = use_spellcheck
        self.auto_correct = auto_correct
        self.interactive_correct = interactive_correct
        
        # Validate: auto_correct and interactive_correct are mutually exclusive
        if self.auto_correct and self.interactive_correct:
            print("Warning: Both auto-correct and interactive-correct enabled.")
            print("         Interactive mode will take precedence.")
            self.auto_correct = False
        
        # Initialize NLP pipeline for Serbian
        self.nlp_pipeline = None
        if CLASSLA_AVAILABLE:
            try:
                self.nlp_pipeline = classla.Pipeline( # type: ignore
                    'sr', 
                    processors='tokenize,pos,lemma',
                    use_gpu=False
                )
                print("NLP pipeline initialized successfully.")
            except Exception as e:
                print(f"Warning: Could not initialize NLP pipeline: {e}")
        else:
            print("NLP pipeline disabled: CLASSLA not available.")
        
        # Initialize spell checker
        self.spell_checker = None
        if use_spellcheck:
            self.spell_checker = SerbianSpellChecker()
        
        # Load dictionary
        self.load_dictionary()
    
    def load_dictionary(self) -> None:
        """
        Load dictionary of risky words with weights from file.
        
        File format:
        - One term per line
        - Optional weight separated by space (default: 1)
        - Lines starting with # are comments
        """
        self.words_set = set()
        self.weights_dict = {}
        self.term_to_weight = {}  # Clear cache
        
        if not os.path.exists(self.dictionary_file):
            print(f"Warning: Dictionary file '{self.dictionary_file}' not found.")
            self._create_default_dictionary()
            return
        
        try:
            with open(self.dictionary_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    
                    if not line or line.startswith('#'):
                        continue
                    
                    parts = line.split()
                    term = parts[0].lower()
                    
                    weight = 1
                    if len(parts) >= 2:
                        try:
                            weight = int(parts[1])
                        except ValueError:
                            print(f"Warning: Invalid weight on line {line_num}: {line}")
                    
                    self.words_set.add(term)
                    self.weights_dict[term] = weight
                    self.term_to_weight[term] = weight  # OPTIMIZATION: Populate cache
            
            print(f"Loaded {len(self.words_set)} terms from dictionary.")
            
        except Exception as e:
            print(f"Error loading dictionary: {e}")
            self._create_default_dictionary()
    
    def _create_default_dictionary(self) -> None:
        """Create a minimal default dictionary for testing."""
        default_terms = {
            'terorizam': 10,
            'terorista': 9,
            'bomba': 8,
            'nasilje': 7,
            'ekstremizam': 8,
            'mržnja': 6,
            'ubistvo': 9,
            'napad': 7,
            '💣': 8,
            '🔫': 7
        }
        
        self.words_set = set(default_terms.keys())
        self.weights_dict = default_terms
        self.term_to_weight = default_terms.copy()  # OPTIMIZATION: Copy to cache
        print("Using default dictionary with", len(self.words_set), "terms.")
    
    def cyrillic_to_latin(self, text: str) -> str:
        """
        Convert Serbian Cyrillic text to Latin script.
        
        Args:
            text: Text in Cyrillic
            
        Returns:
            Text converted to Latin script
        """
        conversion_table = {
            'А': 'A', 'Б': 'B', 'В': 'V', 'Г': 'G', 'Д': 'D',
            'Ђ': 'Đ', 'Е': 'E', 'Ж': 'Ž', 'З': 'Z', 'И': 'I',
            'Ј': 'J', 'К': 'K', 'Л': 'L', 'Љ': 'Lj', 'М': 'M',
            'Н': 'N', 'Њ': 'Nj', 'О': 'O', 'П': 'P', 'Р': 'R',
            'С': 'S', 'Т': 'T', 'Ћ': 'Ć', 'У': 'U', 'Ф': 'F',
            'Х': 'H', 'Ц': 'C', 'Ч': 'Č', 'Џ': 'Dž', 'Ш': 'Š',
            'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd',
            'ђ': 'đ', 'е': 'e', 'ж': 'ž', 'з': 'z', 'и': 'i',
            'ј': 'j', 'к': 'k', 'л': 'l', 'љ': 'lj', 'м': 'm',
            'н': 'n', 'њ': 'nj', 'о': 'o', 'п': 'p', 'р': 'r',
            'с': 's', 'т': 't', 'ћ': 'ć', 'у': 'u', 'ф': 'f',
            'х': 'h', 'ц': 'c', 'ч': 'č', 'џ': 'dž', 'ш': 'š'
        }
        
        result = []
        i = 0
        while i < len(text):
            char = text[i]
            
            if i + 1 < len(text):
                two_chars = text[i:i+2]
                if two_chars in ['Љ', 'Њ', 'Џ', 'љ', 'њ', 'џ']:
                    result.append(conversion_table.get(two_chars, two_chars))
                    i += 2
                    continue
            
            result.append(conversion_table.get(char, char))
            i += 1
        
        return ''.join(result)
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize text for analysis while preserving emojis and special chars.
        Converts Cyrillic to Latin for consistent analysis.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text in Latin script
        """
        text = self.cyrillic_to_latin(text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def spell_check_text(self, text: str) -> Dict[str, Any]:
        """
        Perform spell checking on text with optional interactive correction.
        
        Args:
            text: Text to check
            
        Returns:
            Dictionary with spell checking results
        """
        if not self.spell_checker or not self.use_spellcheck:
            return {
                'original_text': text,
                'corrected_text': text,
                'errors_found': 0,
                'corrections_made': 0,
                'error_details': [],
                'correction_feedback': []
            }
        
        words = re.findall(r'\b\w+\b', text, flags=re.UNICODE)
        
        error_details = []
        corrections = []
        corrections_made = 0
        correction_feedback = []
        
        # Show interactive mode message once
        if self.interactive_correct:
            errors_to_check = [w for w in words if len(w) >= 2 and not self.spell_checker.check_word(w)[0]]
            if errors_to_check:
                print(f"\n{'='*60}")
                print(f"INTERAKTIVNI MOD KOREKCIJE")
                print(f"Pronađeno {len(errors_to_check)} potencijalnih grešaka.")
                print(f"Za svaku grešku, izaberite akciju:")
                print(f"  Broj (1-5) = Primeni predlog")
                print(f"  0 = Zadrži original")
                print(f"  m = Unesi ručno")
                print(f"{'='*60}\n")
        
        for word in words:
            if len(word) < 2:
                continue
                
            is_correct, suggestions = self.spell_checker.check_word(word)
            
            if not is_correct:
                error_details.append({
                    'word': word,
                    'suggestions': suggestions
                })
                
                corrected_word = word
                user_choice = None
                rating = None
                
                # Interactive mode: Ask user for each correction
                if self.interactive_correct and suggestions:
                    print(f"\n❌ Greška pronađena: '{word}'")
                    print(f"   Predlozi:")
                    for i, sugg in enumerate(suggestions[:5], 1):
                        print(f"   {i}. {sugg}")
                    print(f"   0. Zadrži original")
                    print(f"   m. Unesi ručno")
                    
                    while True:
                        choice = input("\n   Izaberite opciju (1-5/0/m): ").strip().lower()
                        
                        if choice == '0':
                            corrected_word = word
                            user_choice = 'kept_original'
                            break
                        elif choice == 'm':
                            manual = input("   Unesite ispravku: ").strip()
                            if manual:
                                corrected_word = manual
                                user_choice = 'manual_correction'
                                break
                        elif choice.isdigit() and 1 <= int(choice) <= len(suggestions[:5]):
                            corrected_word = suggestions[int(choice) - 1]
                            user_choice = f'suggestion_{choice}'
                            
                            # Ask for rating
                            print(f"\n   Ocenite korekciju '{word}' → '{corrected_word}':")
                            print(f"   1 = Loše  2 = Osrednje  3 = Dobro  4 = Odlično")
                            rating_input = input("   Ocena (1-4): ").strip()
                            if rating_input.isdigit() and 1 <= int(rating_input) <= 4:
                                rating = int(rating_input)
                            break
                        else:
                            print("   ❌ Nevalidna opcija. Pokušajte ponovo.")
                
                # Auto-correct mode: Use first suggestion
                elif self.auto_correct and suggestions:
                    corrected_word = suggestions[0]
                    user_choice = 'auto_corrected'
                
                # Apply correction if word changed
                if corrected_word != word:
                    corrections.append((word, corrected_word))
                    corrections_made += 1
                    
                    correction_feedback.append({
                        'original': word,
                        'corrected': corrected_word,
                        'choice': user_choice,
                        'rating': rating,
                        'suggestions': suggestions
                    })
        
        # Apply corrections to text
        corrected_text = text
        for original, corrected in corrections:
            pattern = r'\b' + re.escape(original) + r'\b'
            corrected_text = re.sub(pattern, corrected, corrected_text, flags=re.IGNORECASE | re.UNICODE)
        
        # Show summary for interactive mode
        if self.interactive_correct and correction_feedback:
            print(f"\n{'='*60}")
            print(f"REZIME KOREKCIJA")
            print(f"Ukupno ispravljeno: {corrections_made}/{len(error_details)}")
            
            # Count learned corrections
            learned_count = sum(1 for f in correction_feedback if f.get('choice') == 'learned_correction')
            saved_count = sum(1 for f in correction_feedback if f.get('choice') in ['manual_correction', 'suggestion_1', 'suggestion_2', 'suggestion_3', 'suggestion_4', 'suggestion_5'] and f.get('rating'))
            
            if learned_count > 0:
                print(f"Naučene korekcije automatski primenjene: {learned_count}")
            if saved_count > 0:
                print(f"Nove korekcije sačuvane: {saved_count}")
            
            avg_rating = sum(f['rating'] for f in correction_feedback if f['rating']) / max(1, sum(1 for f in correction_feedback if f['rating']))
            if avg_rating > 0:
                print(f"Prosečna ocena: {avg_rating:.1f}/4 {'⭐' * int(avg_rating)}")
            
            print(f"Korekcije se čuvaju u: {self.spell_checker.user_corrections_file}")
            print(f"={'='*60}\n")
        
        return {
            'original_text': text,
            'corrected_text': corrected_text,
            'errors_found': len(error_details),
            'corrections_made': corrections_made,
            'error_details': error_details,
            'correction_feedback': correction_feedback
        }
    
    def extract_lemmas(self, text: str) -> List[str]:
        """
        Extract lemmas (base forms) from text using NLP pipeline.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of lemmas
        """
        if not self.nlp_pipeline:
            words = re.findall(r'\b\w+\b', text, flags=re.UNICODE)
            return [word.lower() for word in words]
        
        try:
            doc = self.nlp_pipeline(text)
            lemmas = []
            
            for sentence in doc.sentences:
                for word in sentence.words:
                    lemma = word.lemma.lower() if word.lemma else word.text.lower()
                    lemmas.append(lemma)
            
            return lemmas
            
        except Exception as e:
            print(f"Error in NLP processing: {e}")
            words = re.findall(r'\b\w+\b', text, flags=re.UNICODE)
            return [word.lower() for word in words]
    
    def find_all_risky_terms(self, text: str, lemmas: List[str]) -> Tuple[Dict[str, int], int]:
        """
        OPTIMIZED: Unified search for ALL risky terms (lemmas + emojis).
        Returns Counter dict and total weighted score in one pass.
        
        Args:
            text: Raw text for emoji search
            lemmas: List of lemmas for word search
            
        Returns:
            Tuple of (term_frequencies_dict, total_score)
        """
        found_terms = Counter()
        total_score = 0
        
        # Search in lemmas
        for lemma in lemmas:
            matched_term = None
            
            # Exact match first
            if lemma in self.words_set:
                matched_term = lemma
            else:
                # Prefix match
                for term in self.words_set:
                    if lemma.startswith(term):
                        matched_term = term
                        break
            
            if matched_term:
                found_terms[lemma] += 1
                total_score += self.term_to_weight.get(matched_term, 1)  # O(1) lookup!
        
        # Search emojis/special chars in raw text
        for term in self.words_set:
            if not re.match(r'^\w+$', term, re.UNICODE):
                count = text.count(term)
                if count > 0:
                    found_terms[term] += count
                    total_score += self.term_to_weight.get(term, 1) * count  # O(1) lookup!
        
        return dict(found_terms), total_score
    
    def calculate_risk_level(self, total_score: int, unique_terms: int, 
                           total_words: int) -> Tuple[str, str]:
        """
        Calculate risk level based on thresholds.
        
        Args:
            total_score: Weighted risk score
            unique_terms: Number of unique risky terms
            total_words: Total words in text
            
        Returns:
            Tuple of (risk_level, description)
        """
        density = (unique_terms / max(total_words, 1)) * 100 if total_words > 0 else 0
        
        if (total_score >= RISK_THRESHOLDS['high']['score'] or 
            unique_terms >= RISK_THRESHOLDS['high']['unique_terms']):
            level = 'VISOK RIZIK'
        elif (total_score >= RISK_THRESHOLDS['medium']['score'] or 
              unique_terms >= RISK_THRESHOLDS['medium']['unique_terms']):
            level = 'SREDNJI RIZIK'
        elif (total_score >= RISK_THRESHOLDS['low']['score'] or 
              unique_terms >= RISK_THRESHOLDS['low']['unique_terms']):
            level = 'NIZAK RIZIK'
        elif total_score > 0:
            level = 'MINIMALAN RIZIK'
        else:
            level = 'BEZ RIZIKA'
        
        description = RISK_LEVELS[level]['description']
        
        if level in ['VISOK RIZIK', 'SREDNJI RIZIK'] and density > 5:
            description += f" (gustina: {density:.1f}%)"
        
        return level, description
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        OPTIMIZED: Comprehensive text analysis with reduced redundancy.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with analysis results
        """
        start_time = datetime.now()
        
        original_text = text
        normalized_text = self.normalize_text(text)
        spell_results = self.spell_check_text(normalized_text)
        
        # Extract lemmas
        lemmas = self.extract_lemmas(normalized_text)
        total_words = len(lemmas)
        
        # OPTIMIZED: Single function call gets all terms + total score
        term_frequencies, total_score = self.find_all_risky_terms(normalized_text, lemmas)
        
        # Calculate from dict (no redundant structures)
        unique_terms_count = len(term_frequencies)
        total_occurrences = sum(term_frequencies.values())
        term_density = (total_occurrences / max(total_words, 1)) * 100
        
        # Risk calculation
        risk_level, risk_description = self.calculate_risk_level(
            total_score, unique_terms_count, total_words
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # OPTIMIZED: Simplified results structure
        results = {
            'original_text': original_text,
            'processed_text': normalized_text,
            'spell_checking': spell_results,
            'analysis': {
                'total_words': total_words,
                'term_frequencies': term_frequencies,  # Single source of truth!
                'unique_terms_count': unique_terms_count,
                'total_occurrences': total_occurrences,
                'total_score': total_score,
                'term_density': term_density,
                'risk_level': risk_level,
                'risk_description': risk_description,
                'recommendations': RISK_LEVELS[risk_level]['recommendations']
            },
            'statistics': {
                'processing_time_seconds': processing_time,
                'errors_found': spell_results['errors_found'],
                'corrections_made': spell_results['corrections_made']
            },
            'metadata': {
                'timestamp': start_time.isoformat(),
                'dictionary_size': len(self.words_set),
                'spell_check_enabled': self.use_spellcheck,
                'auto_correct_enabled': self.auto_correct,
                'interactive_correct_enabled': self.interactive_correct
            }
        }
        
        return results


class ReportGenerator:
    """Generate reports from analysis results."""
    
    @staticmethod
    def generate_text_report(results: Dict[str, Any], analyzer: Optional['TextAnalyzer'] = None) -> str:
        """
        OPTIMIZED: Generate text report with O(1) weight lookups.
        
        Args:
            results: Analysis results dictionary
            analyzer: Optional reference to TextAnalyzer for weight cache access
            
        Returns:
            Formatted text report
        """
        report_lines = []
        
        # Header
        report_lines.append("=" * 70)
        report_lines.append("ANALIZA TEKSTA / TEXT ANALYSIS")
        report_lines.append(f"Datum: {results['metadata']['timestamp']}")
        report_lines.append("=" * 70)
        report_lines.append("")
        
        # Spell checking results
        spell_info = results['spell_checking']
        report_lines.append("--- PROVERA PRAVOPISA ---")
        
        orig_text = results['original_text']
        truncated = orig_text if len(orig_text) <= 200 else orig_text[:200] + "..."
        report_lines.append(f"Originalni tekst:\n{truncated}")
        
        # Show Cyrillic conversion ONLY if significant portion was Cyrillic
        if orig_text != results['processed_text']:
            # Count Cyrillic characters
            cyrillic_count = sum(1 for char in orig_text if '\u0400' <= char <= '\u04FF')
            total_alpha = sum(1 for char in orig_text if char.isalpha())
            
            # Show conversion only if >20% of letters are Cyrillic
            if total_alpha > 0 and (cyrillic_count / total_alpha) > 0.2:
                proc_preview = results['processed_text']
                proc_truncated = proc_preview if len(proc_preview) <= 200 else proc_preview[:200] + "..."
                report_lines.append(f"\nKonvertovano u latinicu:\n{proc_truncated}")
        
        report_lines.append("")
        report_lines.append(f"Pronađeno {spell_info['errors_found']} pogrešno napisanih reči")
        report_lines.append(f"Automatski ispravljeno: {spell_info['corrections_made']} reči")
        
        if spell_info['error_details']:
            report_lines.append("")
            report_lines.append("Detalji o greškama:")
            for error in spell_info['error_details'][:10]:
                suggestions = ", ".join(error['suggestions']) if error['suggestions'] else "nema predloga"
                report_lines.append(f"  ✗ {error['word']} (Predlozi: {suggestions})")
        
        # Show correction feedback if interactive mode was used
        if 'correction_feedback' in spell_info and spell_info['correction_feedback']:
            report_lines.append("")
            
            # Separate learned vs new corrections
            learned = [f for f in spell_info['correction_feedback'] if f.get('choice') == 'learned_correction']
            new_corrections = [f for f in spell_info['correction_feedback'] if f.get('choice') != 'learned_correction']
            
            if learned:
                report_lines.append("Naučene korekcije (automatski primenjene):")
                for feedback in learned:
                    report_lines.append(f"  ✓ {feedback['original']} → {feedback['corrected']} [Naučeno]")
            
            if new_corrections:
                report_lines.append("" if learned else "")
                report_lines.append("Nove korisničke korekcije:")
                for feedback in new_corrections:
                    rating_str = f" [Ocena: {feedback['rating']}/4]" if feedback.get('rating') else ""
                    saved_str = " [Sačuvano]" if feedback.get('rating') else ""
                    report_lines.append(f"  ✓ {feedback['original']} → {feedback['corrected']} ({feedback['choice']}){rating_str}{saved_str}")
        
        report_lines.append("")
        
        # Analysis results
        analysis = results['analysis']
        report_lines.append("--- ANALIZA RIZIKA ---")
        
        if analysis['term_frequencies']:
            report_lines.append(f"✓ Pronađeno {analysis['unique_terms_count']} jedinstvenih termina")
            report_lines.append("")
            report_lines.append("Učestalost reči:")
            
            for term, count in sorted(analysis['term_frequencies'].items(), 
                                     key=lambda x: x[1], reverse=True):
                # OPTIMIZED: O(1) weight lookup via cache
                if analyzer and hasattr(analyzer, 'term_to_weight'):
                    weight = analyzer.term_to_weight.get(term, 1)
                else:
                    weight = 1
                
                report_lines.append(f"  - {term}: {count}x (težina: {weight})")
        else:
            report_lines.append("✓ Nije pronađeno reči od značaja")
        
        report_lines.append("")
        
        # Statistics - OPTIMIZED: No redundant counts
        report_lines.append("--- STATISTIKA ---")
        report_lines.append(f"Ukupno reči u tekstu: {analysis['total_words']}")
        report_lines.append(f"Ukupno pojavljivanja rizičnih termina: {analysis['total_occurrences']}")
        report_lines.append(f"Jedinstvenih termina: {analysis['unique_terms_count']}")
        report_lines.append(f"Gustina relevantnih reči: {analysis['term_density']:.2f}%")
        report_lines.append(f"Ukupan skor (težinski): {analysis['total_score']}")
        report_lines.append(f"Vreme obrade: {results['statistics']['processing_time_seconds']:.3f}s")
        report_lines.append("")
        
        # Risk assessment
        report_lines.append("--- PROCENA RIZIKA ---")
        report_lines.append(f"Nivo rizika: {analysis['risk_level']}")
        report_lines.append(f"Opis: {analysis['risk_description']}")
        report_lines.append("")
        report_lines.append("Preporuke:")
        for i, recommendation in enumerate(analysis['recommendations'], 1):
            report_lines.append(f"{i}. {recommendation}")
        
        report_lines.append("")
        report_lines.append("=" * 70)
        
        return "\n".join(report_lines)
    
    @staticmethod
    def generate_json_report(results: Dict[str, Any]) -> str:
        """
        Generate a JSON report from analysis results.
        
        Args:
            results: Analysis results dictionary
            
        Returns:
            JSON string
        """
        json_output = {
            'metadata': results['metadata'],
            'summary': {
                'risk_level': results['analysis']['risk_level'],
                'risk_description': results['analysis']['risk_description'],
                'total_score': results['analysis']['total_score'],
                'unique_terms_found': results['analysis']['unique_terms_count'],
                'total_occurrences': results['analysis']['total_occurrences'],
                'term_density': results['analysis']['term_density'],
                'processing_time': results['statistics']['processing_time_seconds']
            },
            'terms_found': results['analysis']['term_frequencies'],
            'spell_checking': {
                'errors_found': results['spell_checking']['errors_found'],
                'corrections_made': results['spell_checking']['corrections_made'],
                'correction_feedback': results['spell_checking'].get('correction_feedback', [])
            }
        }
        
        return json.dumps(json_output, ensure_ascii=False, indent=2)
    
    @staticmethod
    def save_report(report_text: str, filename: str = OUTPUT_FILE) -> None:
        """Save report to file."""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"Report saved to: {filename}")
        except Exception as e:
            print(f"Error saving report: {e}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='WatchTower: Serbian Text Analysis Tool for Content Risk Assessment'
    )
    
    parser.add_argument('--text', type=str, help='Text to analyze (enclose in quotes)')
    parser.add_argument('--file', type=str, help='File containing text to analyze')
    parser.add_argument('--dictionary', type=str, default=DEFAULT_DICTIONARY_FILE,
                       help=f'Dictionary file with weights (default: {DEFAULT_DICTIONARY_FILE})')
    parser.add_argument('--output', type=str, default=OUTPUT_FILE,
                       help=f'Output file (default: {OUTPUT_FILE})')
    parser.add_argument('--no-spellcheck', action='store_true', help='Disable spell checking')
    parser.add_argument('--auto-correct', action='store_true', help='Enable automatic spelling correction')
    parser.add_argument('--interactive-correct', action='store_true', 
                       help='Enable interactive correction with user confirmation and rating')
    parser.add_argument('--json', action='store_true', help='Output results in JSON format')
    parser.add_argument('--demo', action='store_true', help='Run demo with sample texts')
    parser.add_argument('--show-corrections', action='store_true', 
                       help='Display learned corrections from user_corrections.txt')
    parser.add_argument('--clear-corrections', action='store_true', 
                       help='Clear all learned corrections')
    
    return parser.parse_args()


def show_learned_corrections():
    """Display all learned corrections."""
    if not os.path.exists(USER_CORRECTIONS_FILE):
        print(f"Nije pronađen fajl sa korekcijama: {USER_CORRECTIONS_FILE}")
        print("Koristite --interactive-correct da napravite korekcije.")
        return
    
    corrections = []
    try:
        with open(USER_CORRECTIONS_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    corrections.append(line)
    except Exception as e:
        print(f"Greška pri učitavanju korekcija: {e}")
        return
    
    if not corrections:
        print("Nema sačuvanih korekcija.")
        return
    
    print("\n" + "=" * 70)
    print(f"NAUČENE KOREKCIJE ({len(corrections)} ukupno)")
    print("=" * 70)
    print(f"Fajl: {USER_CORRECTIONS_FILE}\n")
    
    for i, correction in enumerate(corrections, 1):
        print(f"{i:3d}. {correction}")
    
    print("\n" + "=" * 70)
    print(f"Ukupno: {len(corrections)} naučenih korekcija")
    print("=" * 70)


def clear_learned_corrections():
    """Clear all learned corrections with confirmation."""
    if not os.path.exists(USER_CORRECTIONS_FILE):
        print(f"Nije pronađen fajl sa korekcijama: {USER_CORRECTIONS_FILE}")
        return
    
    # Count corrections
    count = 0
    try:
        with open(USER_CORRECTIONS_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip() and not line.strip().startswith('#'):
                    count += 1
    except Exception:
        pass
    
    if count == 0:
        print("Nema sačuvanih korekcija za brisanje.")
        return
    
    # Ask for confirmation
    print(f"\n⚠️  UPOZORENJE: Brisanje {count} naučenih korekcija!")
    print(f"Fajl: {USER_CORRECTIONS_FILE}")
    confirmation = input("\nDa li ste sigurni? Unesite 'DA' za potvrdu: ").strip()
    
    if confirmation == 'DA':
        try:
            os.remove(USER_CORRECTIONS_FILE)
            print(f"✓ Obrisano {count} korekcija.")
            print(f"✓ Fajl {USER_CORRECTIONS_FILE} je uklonjen.")
        except Exception as e:
            print(f"Greška pri brisanju: {e}")
    else:
        print("Otkazano. Korekcije nisu obrisane.")


def run_demo(analyzer: TextAnalyzer, generator: ReportGenerator):
    """Run demonstration with sample texts."""
    print("\n" + "=" * 70)
    print("WATCHTOWER DEMONSTRACIJA (Optimized Version)")
    print("=" * 70)
    
    sample_texts = [
        {
            'name': 'Bezopasan tekst',
            'text': 'Danas je lep dan. Sunce sija i ptice pevaju. Deca se igraju u parku.',
            'description': 'Normalan, bezopasan tekst bez rizičnih termina'
        },
        {
            'name': 'Tekst sa pojedinim rizičnim terminima',
            'text': 'Terorizam je ozbiljan problem u savremenom svetu. Države saraduju u borbi protiv ovog fenomena.',
            'description': 'Akademski/novinski stil sa pojedinačnim terminima'
        },
        {
            'name': 'Tekst sa više rizičnih termina',
            'text': 'Ekstremisti koriste nasilje i bombe da postignu svoje ciljeve. Mržnja vodi ka terorizmu.',
            'description': 'Više rizičnih termina, srednji nivo rizika'
        },
        {
            'name': 'Tekst sa emodžijima',
            'text': 'Pažnja na potencijalne pretnje! 💣 Može biti opasno. 🔫 Treba biti oprezan.',
            'description': 'Test detekcije emoji karaktera u tekstu'
        },
        {
            'name': 'Tekst sa pravopisnim greškama',
            'text': 'Terorista je pustio bombu i doslo je do velikog praska. Napad je bio uzasan i strasn.',
            'description': 'Test spell checkera - "doslo" → "došlo", "uzasan" → "užasan", "strasn" → "strašan"'
        },
        {
            'name': 'Tekst sa greškama i visokim rizikom',
            'text': 'Ekstremisti planiraju napad bombama i oruzjem. Nasilje i mrznја vode ka terorizmu i ubistima.',
            'description': 'Kombinacija: pravopisne greške + visok rizik sadržaj'
        },
        {
            'name': 'Ćirilički tekst',
            'text': 'Тероризам је озбиљан проблем. Екстремисти користе насиље.',
            'description': 'Test ćiriličnog pisma - bi trebao detektovati termine'
        }
    ]
    
    for i, sample in enumerate(sample_texts, 1):
        print(f"\n{'=' * 70}")
        print(f"[TEST {i}/{len(sample_texts)}]: {sample['name']}")
        print(f"Opis: {sample['description']}")
        print(f"{'=' * 70}")
        print(f"Tekst: {sample['text']}")
        
        results = analyzer.analyze_text(sample['text'])
        
        # Show Cyrillic conversion
        if results['original_text'] != results['processed_text']:
            has_cyrillic = any('\u0400' <= char <= '\u04FF' for char in results['original_text'])
            if has_cyrillic:
                print(f"🔤 Konvertovano u latinicu: {results['processed_text']}")
        
        print()
        
        # Spell checking results
        spell_info = results['spell_checking']
        if spell_info['errors_found'] > 0:
            print(f"📝 Pravopisne greške: {spell_info['errors_found']}")
            if analyzer.auto_correct:
                print(f"   Ispravljeno: {spell_info['corrections_made']} reči")
                if spell_info['corrected_text'] != spell_info['original_text']:
                    print(f"   Ispravljen tekst: {spell_info['corrected_text']}")
            else:
                for error in spell_info['error_details'][:3]:
                    sugg = ', '.join(error['suggestions'][:3]) if error['suggestions'] else 'nema'
                    print(f"   - {error['word']} (predlozi: {sugg})")
        else:
            print(f"✓ Pravopis: Nema grešaka")
        
        # Analysis results
        analysis = results['analysis']
        print(f"\n🎯 Nivo rizika: {analysis['risk_level']}")
        print(f"📊 Skor: {analysis['total_score']} | Termina: {analysis['unique_terms_count']}")
        
        if analysis['term_frequencies']:
            print(f"🔍 Pronađeni termini:")
            for term, count in sorted(analysis['term_frequencies'].items(), 
                                     key=lambda x: x[1], reverse=True):
                weight = analyzer.term_to_weight.get(term, 1)  # O(1) lookup!
                print(f"   - {term}: {count}x (težina: {weight})")
        else:
            print(f"✓ Nisu pronađeni rizični termini")
        
        print(f"⏱️  Vreme: {results['statistics']['processing_time_seconds']:.3f}s")
    
    print("\n" + "=" * 70)
    print("DEMONSTRACIJA ZAVRŠENA")
    print("=" * 70)
    print("\nOpcije:")
    print("  --auto-correct           Automatska korekcija bez potvrde")
    print("  --interactive-correct    Interaktivna korekcija sa potvrdom i ocenom")
    print("  --show-corrections       Prikaži naučene korekcije")
    print("  --clear-corrections      Obriši sve naučene korekcije")
    print("\nPrimer:")
    print("  python watchtower.py --demo --auto-correct")
    print("  python watchtower.py --text 'Vaš tekst' --interactive-correct")
    print("  python watchtower.py --show-corrections")


def main():
    """Main function."""
    args = parse_arguments()
    
    # Handle utility commands first
    if args.show_corrections:
        show_learned_corrections()
        return
    
    if args.clear_corrections:
        clear_learned_corrections()
        return
    
    # Initialize analyzer
    analyzer = TextAnalyzer(
        dictionary_file=args.dictionary,
        use_spellcheck=not args.no_spellcheck,
        auto_correct=args.auto_correct,
        interactive_correct=args.interactive_correct
    )
    
    # Initialize report generator
    generator = ReportGenerator()
    
    # Run demo if requested
    if args.demo:
        run_demo(analyzer, generator)
        return
    
    # Get text to analyze
    text_to_analyze = ""
    
    if args.text:
        text_to_analyze = args.text
    elif args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                text_to_analyze = f.read()
        except Exception as e:
            print(f"Error reading file: {e}")
            return
    else:
        print("Unesite tekst za analizu (Ctrl+D ili Ctrl+Z kada završite):")
        print("-" * 50)
        try:
            text_to_analyze = sys.stdin.read()
        except (KeyboardInterrupt, EOFError):
            print("\nOtkazano.")
            return
    
    if not text_to_analyze.strip():
        print("Nema teksta za analizu.")
        return
    
    print(f"\nAnaliziranje teksta ({len(text_to_analyze)} karaktera)...")
    
    # Analyze text
    results = analyzer.analyze_text(text_to_analyze)
    
    # Generate report (pass analyzer for weight cache access)
    if args.json:
        report = generator.generate_json_report(results)
        print("\n" + report)
        json_filename = f"{os.path.splitext(args.output)[0]}.json"
        generator.save_report(report, json_filename)
    else:
        report = generator.generate_text_report(results, analyzer)  # Pass analyzer!
        print("\n" + report)
        generator.save_report(report, args.output)
    
    # Print quick summary
    print("\n" + "-" * 50)
    print("REZIME ANALIZE:")
    print(f"Nivo rizika: {results['analysis']['risk_level']}")
    print(f"Ukupan skor: {results['analysis']['total_score']}")
    print(f"Pronađeno termina: {results['analysis']['unique_terms_count']}")
    print(f"Vreme obrade: {results['statistics']['processing_time_seconds']:.3f}s")
    print("-" * 50)


if __name__ == "__main__":
    main()