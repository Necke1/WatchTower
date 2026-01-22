"""
Serbian Text Analysis Tool 
"""

import classla
from collections import Counter
from typing import Set, Dict, List, Tuple
from phunspell import Phunspell

# Initialize the Serbian NLP pipeline
nlpc = classla.Pipeline('sr')

######################################################################## 
# UTILITY FUNCTIONS
######################################################################## 

def load_words_with_weights(word_file: str) -> Tuple[Set[str], Dict[str, int]]:
    """Load words and weights from dictionary file.
    
    Format options:
    - word           (weight defaults to 1)
    - word weight    (e.g., 'teror 10')
    
    Args:
        word_file: Path to the dictionary file
        
    Returns:
        Tuple of (set of words/prefixes, dictionary of weights)
    """
    ter_words = set()
    weights = {}
    
    try:
        with open(word_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                
                word = parts[0].lower()
                ter_words.add(word)
                
                # Check if weight is provided
                if len(parts) >= 2:
                    try:
                        weight = int(parts[1])
                        weights[word] = weight
                    except ValueError:
                        # If second part is not a number, default weight to 1
                        weights[word] = 1
                else:
                    # No weight provided, default to 1
                    weights[word] = 1
                    
        return ter_words, weights
        
    except FileNotFoundError:
        print(f"Error: File '{word_file}' not found.")
        return set(), {}
    except Exception as e:
        print(f"Error loading dictionary: {e}")
        return set(), {}


def ispisivanje(text: str, filename: str = 'result.txt') -> None:
    """Write results to a text file.
    
    Args:
        text: Text to write
        filename: Output file path
    """
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"Results written to {filename}")
    except Exception as e:
        print(f"Error writing to file: {e}")


def cyrillic_to_latin(text: str) -> str:
    """Convert Serbian Cyrillic text to Latin script.
    
    Args:
        text: Text in Cyrillic script
        
    Returns:
        Transliterated text in Latin script
    """
    cyrillic_to_latin_map = {
        'А': 'A', 'Б': 'B', 'В': 'V', 'Г': 'G', 'Д': 'D', 'Ђ': 'Đ', 
        'Е': 'E', 'Ж': 'Ž', 'З': 'Z', 'И': 'I', 'Ј': 'J', 'К': 'K', 
        'Л': 'L', 'Љ': 'Lj', 'М': 'M', 'Н': 'N', 'Њ': 'Nj', 'О': 'O', 
        'П': 'P', 'Р': 'R', 'С': 'S', 'Т': 'T', 'Ћ': 'Ć', 'У': 'U', 
        'Ф': 'F', 'Х': 'H', 'Ц': 'C', 'Ч': 'Č', 'Џ': 'Dž', 'Ш': 'Š',
        'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'ђ': 'đ', 
        'е': 'e', 'ж': 'ž', 'з': 'z', 'и': 'i', 'ј': 'j', 'к': 'k', 
        'л': 'l', 'љ': 'lj', 'м': 'm', 'н': 'n', 'њ': 'nj', 'о': 'o', 
        'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'ћ': 'ć', 'у': 'u', 
        'ф': 'f', 'х': 'h', 'ц': 'c', 'ч': 'č', 'џ': 'dž', 'ш': 'š'
    }
    
    return ''.join(cyrillic_to_latin_map.get(char, char) for char in text)


######################################################################## 
# SPELL CHECKING FUNCTIONS
######################################################################## 

class SerbianSpellChecker:
    """Serbian spell checker supporting both Latin and Cyrillic scripts."""
    
    def __init__(self):
        """Initialize both Latin and Cyrillic dictionaries."""
        try:
            self.latin_dict = Phunspell('sr-Latn')
            self.cyrillic_dict = Phunspell('sr')
            self.initialized = True
        except Exception as e:
            print(f"Warning: Could not initialize spell checker: {e}")
            print("Spell checking will be disabled.")
            self.initialized = False
    
    def _is_cyrillic(self, word: str) -> bool:
        """Check if word contains Cyrillic characters.
        
        Args:
            word: Word to check
            
        Returns:
            True if word contains Cyrillic characters
        """
        # Cyrillic Unicode range: 0400-04FF
        for char in word:
            if '\u0400' <= char <= '\u04FF':
                return True
        return False
    
    def _is_latin(self, word: str) -> bool:
        """Check if word contains Latin characters.
        
        Args:
            word: Word to check
            
        Returns:
            True if word contains Latin characters
        """
        # Basic Latin + Latin Extended-A (includes Serbian Latin diacritics)
        for char in word:
            if ('\u0041' <= char <= '\u007A' or  # Basic Latin
                '\u00C0' <= char <= '\u00FF' or  # Latin-1 Supplement
                '\u0100' <= char <= '\u017F'):   # Latin Extended-A
                return True
        return False
    
    def lookup(self, word: str) -> bool:
        """Check spelling based on detected script.
        
        Args:
            word: Word to check
            
        Returns:
            True if word is spelled correctly
        """
        if not self.initialized:
            return True  # Skip if not initialized
        
        if self._is_cyrillic(word):
            return self.cyrillic_dict.lookup(word)
        elif self._is_latin(word):
            return self.latin_dict.lookup(word)
        else:
            # Try both if script can't be determined
            return (self.latin_dict.lookup(word) or 
                    self.cyrillic_dict.lookup(word))
    
    def lookup_list(self, words: List[str]) -> List[str]:
        """Check list of words and return misspelled ones.
        
        Args:
            words: List of words to check
            
        Returns:
            List of misspelled words
        """
        if not self.initialized:
            return []
        
        return [word for word in words if not self.lookup(word)]
    
    def suggest(self, word: str) -> List[str]:
        """Get suggestions based on detected script.
        
        Args:
            word: Word to get suggestions for
            
        Returns:
            List of suggested corrections
        """
        if not self.initialized:
            return []
        
        if self._is_cyrillic(word):
            return list(self.cyrillic_dict.suggest(word))
        else:
            # Default to Latin for mixed or unknown scripts
            return list(self.latin_dict.suggest(word))


def initialize_spellchecker():
    """Initialize Serbian spell checker.
    
    Returns:
        SerbianSpellChecker object or None if initialization fails
    """
    if Phunspell is None:
        print("Warning: phunspell not available. Spell checking disabled.")
        return None
    
    try:
        checker = SerbianSpellChecker()
        if checker.initialized:
            return checker
        return None
    except Exception as e:
        print(f"Error initializing spell checker: {e}")
        return None


def spell_check_and_correct_text(text: str, spellchecker, auto_correct: bool = True) -> Dict:
    """Check spelling and optionally auto-correct the text.
    
    Args:
        text: Text to check
        spellchecker: SerbianSpellChecker object
        auto_correct: If True, automatically apply first suggestion
        
    Returns:
        Dictionary with original text, corrected text, and correction details
    """
    if spellchecker is None or not spellchecker.initialized:
        return {
            'original_text': text,
            'corrected_text': text,
            'corrections': {},
            'misspelled_count': 0,
            'corrected_count': 0
        }
    
    doc = nlpc(text)
    corrections = {}
    corrected_text = text
    
    # Collect all misspelled words and their positions
    for sent in doc.sentences:
        for word in sent.words:
            word_text = word.text
            
            # Skip punctuation, short words, and numbers
            if len(word_text) < 2 or not word_text.isalpha():
                continue
            
            if not spellchecker.lookup(word_text):
                suggestions = spellchecker.suggest(word_text)
                
                if suggestions:
                    corrections[word_text] = {
                        'suggestions': suggestions[:5],
                        'chosen': suggestions[0] if auto_correct else None
                    }
                    
                    # Auto-correct by replacing with first suggestion
                    if auto_correct:
                        corrected_text = corrected_text.replace(word_text, suggestions[0])
                else:
                    corrections[word_text] = {
                        'suggestions': [],
                        'chosen': None
                    }
    
    return {
        'original_text': text,
        'corrected_text': corrected_text,
        'corrections': corrections,
        'misspelled_count': len(corrections),
        'corrected_count': sum(1 for c in corrections.values() if c['chosen'])
    }


######################################################################## 
# MAIN ANALYSIS FUNCTIONS
######################################################################## 

def check_terror_words(text: str, recnik: Set[str], weights: Dict[str, int] = None) -> Tuple[List[str], List[int]]:
    """Analyze text and find all matching words with their weights.
    
    Args:
        text: Text to analyze
        recnik: Set of keywords/prefixes to search for
        weights: Dictionary of word weights
        
    Returns:
        Tuple of (list of found lemmas, list of corresponding weights)
    """
    doc = nlpc(text)
    found_words = []
    found_weights = []
    
    for sent in doc.sentences:
        for word in sent.words:
            lemma = word.lemma.lower()
            # Check if lemma matches any prefix in dictionary
            for prefix in recnik:
                if lemma.startswith(prefix):
                    found_words.append(lemma)
                    # Get weight for this word (default to 1 if not specified)
                    weight = weights.get(prefix, 1) if weights else 1
                    found_weights.append(weight)
                    break  # Only count once per word
    
    return found_words, found_weights


def calculate_risk_level(total_score: int, unique_count: int, total_words: int) -> Tuple[str, str]:
    """Determine risk level based on threshold analysis.
    
    Args:
        total_score: Weighted sum of all matched words
        unique_count: Number of unique matched words
        total_words: Total words in text
        
    Returns:
        Tuple of (risk_level, description)
    """
    # Calculate density (percentage of flagged words)
    density = (total_words / max(total_words, 1)) * 100 if total_words > 0 else 0
    
    # Risk thresholds
    if total_score >= 50 or unique_count >= 10:
        return "VISOK RIZIK", "Tekst sadrži značajan broj reči koje ukazuju na ekstremizam"
    elif total_score >= 20 or unique_count >= 5:
        return "SREDNJI RIZIK", "Tekst sadrži umerenu količinu sumnjivog sadržaja"
    elif total_score >= 10 or unique_count >= 3:
        return "NIZAK RIZIK", "Tekst sadrži mali broj relevantnih reči"
    elif total_score > 0:
        return "MINIMALAN RIZIK", "Tekst sadrži pojedinačne relevantne reči"
    else:
        return "BEZ RIZIKA", "Tekst ne sadrži reči od značaja"


def analyze_text(text: str, recnik: Set[str], weights: Dict[str, int] = None, 
                spellchecker=None, auto_correct: bool = True) -> Dict:
    """Comprehensive text analysis: spell check first, then risk assessment.
    
    Args:
        text: Text to analyze
        recnik: Dictionary of keywords
        weights: Word weights for scoring
        spellchecker: SerbianSpellChecker object
        auto_correct: If True, use corrected text for analysis
        
    Returns:
        Dictionary with analysis results
    """
    # STEP 1: Spell check and correct the text
    spell_results = spell_check_and_correct_text(text, spellchecker, auto_correct)
    
    # STEP 2: Use corrected text for analysis (or original if no corrections)
    analysis_text = spell_results['corrected_text'] if auto_correct else text
    
    # Get word count from NLP
    doc = nlpc(analysis_text)
    total_word_count = sum(len(sent.words) for sent in doc.sentences)
    
    # Find matches and weights
    matches, match_weights = check_terror_words(analysis_text, recnik, weights)
    word_count = Counter(matches)
    
    # Calculate weighted score
    total_score = sum(match_weights)
    
    # Determine risk level
    risk_level, risk_description = calculate_risk_level(
        total_score, len(word_count), total_word_count
    )
    
    return {
        'spell_check': spell_results,
        'analysis_text': analysis_text,
        'unique_words': set(matches),
        'word_count': dict(word_count),
        'total_matches': len(matches),
        'unique_matches': len(word_count),
        'total_score': total_score,
        'total_word_count': total_word_count,
        'match_density': (len(matches) / max(total_word_count, 1)) * 100,
        'risk_level': risk_level,
        'risk_description': risk_description
    }


def format_results(results: Dict, original_text: str) -> str:
    """Format analysis results for output.
    
    Args:
        results: Analysis results dictionary
        original_text: Original text before correction
        
    Returns:
        Formatted string with results
    """
    output = []
    output.append("=" * 70)
    output.append("ANALIZA TEKSTA / TEXT ANALYSIS")
    output.append("=" * 70)
    
    # SPELL CHECK SECTION
    spell = results['spell_check']
    output.append(f"\n--- PROVERA PRAVOPISA ---")
    output.append(f"Originalni tekst:\n{spell['original_text']}\n")
    
    if spell['misspelled_count'] > 0:
        output.append(f"Pronađeno {spell['misspelled_count']} pogrešno napisanih reči")
        output.append(f"Automatski ispravljeno: {spell['corrected_count']} reči\n")
        
        output.append("Ispravke:")
        for word, correction in spell['corrections'].items():
            if correction['chosen']:
                output.append(f"  ✓ {word} → {correction['chosen']}")
            else:
                sugg_str = ', '.join(correction['suggestions']) if correction['suggestions'] else 'Nema predloga'
                output.append(f"  ✗ {word} (Predlozi: {sugg_str})")
        
        output.append(f"\nIspravljeni tekst:\n{spell['corrected_text']}\n")
    else:
        output.append("✓ Nema pravopisnih grešaka\n")
    
    output.append("-" * 70)
    
    # RISK ASSESSMENT SECTION
    output.append(f"\n--- ANALIZA RIZIKA ---")
    output.append(f"Analizirani tekst:\n{results['analysis_text']}\n")
    
    if results['total_matches'] > 0:
        output.append(f"✓ Pronađeno reči od značaja: {results['unique_words']}")
        output.append(f"\nUčestalost reči:")
        for word, count in sorted(results['word_count'].items(), 
                                  key=lambda x: x[1], reverse=True):
            output.append(f"  - {word}: {count}x")
        
        output.append(f"\n--- STATISTIKA ---")
        output.append(f"Ukupno pojavljivanja: {results['total_matches']}")
        output.append(f"Jedinstvenih reči: {results['unique_matches']}")
        output.append(f"Ukupno reči u tekstu: {results['total_word_count']}")
        output.append(f"Gustina relevantnih reči: {results['match_density']:.2f}%")
        output.append(f"Ukupan skor (težinski): {results['total_score']}")
        
        output.append(f"\n--- PROCENA RIZIKA ---")
        output.append(f"Nivo rizika: {results['risk_level']}")
        output.append(f"Opis: {results['risk_description']}")
    else:
        output.append("✗ Nisu nađene reči od značaja.")
        output.append(f"\n--- PROCENA RIZIKA ---")
        output.append(f"Nivo rizika: {results['risk_level']}")
    
    output.append("\n" + "=" * 70)
    return "\n".join(output)


######################################################################## 
# MAIN PROGRAM
######################################################################## 

def main():
    """Main program execution."""
    
    # Load dictionary with weights from single file
    recnik, weights = load_words_with_weights('recnik.txt')
    
    if not recnik:
        print("Warning: Dictionary is empty or failed to load.")
        return
    
    # Initialize spell checker
    spellchecker = initialize_spellchecker()
    
    # Test text with intentional spelling errors
    text = ("terorista je pustio 🚀 i doslo je do velikog praska terorista. "
            "Novac koji je dobio za napad preko ofšor banke je potrošio na bombe. "
            "Terorizam je ozbiljan problem.")
    
    print("=" * 70)
    print("Faza 1: Provera pravopisa i ispravka teksta...")
    print("=" * 70)
    
    # Analyze text (spell check first, then risk assessment)
    results = analyze_text(text, recnik, weights, spellchecker, auto_correct=True)
    
    print("\nFaza 2: Analiza rizika ispravljenog teksta...")
    print("=" * 70)
    
    # Format and display results
    formatted_output = format_results(results, text)
    print(f"\n{formatted_output}")
    
    # Save results to file
    ispisivanje(formatted_output)
    
    # Example: Convert Cyrillic to Latin if needed
    # cyrillic_text = "Тероризам је озбиљан проблем"
    # latin_text = cyrillic_to_latin(cyrillic_text)
    # print(f"\nCyrillic: {cyrillic_text}")
    # print(f"Latin: {latin_text}")


if __name__ == "__main__":
    main()