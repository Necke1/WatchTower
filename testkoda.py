#vs code settings python.analysis.typeCheckingMode

import phunspell

class SerbianSpellChecker:
    def __init__(self):
        self.latin_dict = phunspell.Phunspell('sr-Latn')
        self.cyrillic_dict = phunspell.Phunspell('sr')
    
    def _is_cyrillic(self, word):
        """Check if word contains Cyrillic characters"""
        # Cyrillic Unicode range: 0400-04FF
        for char in word:
            if '\u0400' <= char <= '\u04FF':
                return True
        return False
    
    def _is_latin(self, word):
        """Check if word contains Latin characters"""
        # Basic Latin + Latin Extended-A (includes Serbian Latin diacritics)
        for char in word:
            if ('\u0041' <= char <= '\u007A' or  # Basic Latin
                '\u00C0' <= char <= '\u00FF' or  # Latin-1 Supplement
                '\u0100' <= char <= '\u017F'):   # Latin Extended-A
                return True
        return False
    
    def lookup(self, word):
        """Check spelling based on detected script"""
        if self._is_cyrillic(word):
            return self.cyrillic_dict.lookup(word)
        elif self._is_latin(word):
            return self.latin_dict.lookup(word)
        else:
            # Try both if script can't be determined
            return (self.latin_dict.lookup(word) or 
                    self.cyrillic_dict.lookup(word))
    
    def lookup_list(self, words):
        """Check list of words"""
        return [word for word in words if not self.lookup(word)]
    
    def suggest(self, word):
        """Get suggestions based on detected script"""
        if self._is_cyrillic(word):
            return list(self.cyrillic_dict.suggest(word))
        else:
            # Default to Latin for mixed or unknown scripts
            return list(self.latin_dict.suggest(word))

# Usage
checker = SerbianSpellChecker()

# Test with mixed scripts
test_words = ["reč", "реч", "škola", "школа", "testirati", "тестирати", "xyz"]

print("Spell checking with auto-detection:")
for word in test_words:
    is_correct = checker.lookup(word)
    print(f"'{word}' -> {'✓' if is_correct else '✗'}")

print(f"\nMisspelled words: {checker.lookup_list(test_words)}")

print("\nSuggestions for 'teest' (Latin):")
for suggestion in checker.suggest('teest'):
    print(f"  - {suggestion}")

print("\nSuggestions for 'тест' (Cyrillic):")
for suggestion in checker.suggest('тест'):
    print(f"  - {suggestion}")