

########################################################################
#preparation part of the program to be usible 
import classla
nlpc = classla.Pipeline('sr') 



#function to load the words from the our crafted txt dictionary for words
def load_words(word_file):
    with open(word_file, 'r', encoding='utf-8') as f:
        ter_words = set(word.strip().lower() for line in f for word in line.split())
    return ter_words
#function to print results of our program to txt file
def ispisivanje(text):
    with open('result.txt', 'w', encoding='utf-8') as f:
        f.write(text)



########################################################################
######################---- Main program ---#############################


#function to compare the text with our dictionary
def check_terror_words(text):
    doc = nlpc(text)
    found_words = set()
    for sent in doc.sentences:
        for word in sent.words:
            lemma = word.lemma.lower()
            if  any(lemma.startswith(prefix) for prefix in recnik): #using the prefix dictionary as it will simplify the search of the words 
                found_words.add(lemma)
    return found_words 
#need to find an option on how  cound duplicate words as for now word_count is not working propeprly because of this func

#loading the dictionary for checking the text
recnik= load_words('recnik.txt')
text="terorista je pustio 🚀 i doslo je do velikog praska terorista. Novac koji je dobio za napad preko ofšor banke je potrošio na bombe" #test text for testing the functions 

# will be using hunspell-sr for spellcheking of the words in serbian


matches = check_terror_words(text)
word_count = {}
for word in matches:
    word_count[word]= word_count.get(word,0)+1


if matches:
    print(f"Sledeće reči od značaja su pronađene: {matches}")
    print(f"Učetalost reči se može videti {word_count}")
else:
    print(f"Nisu nađene reči od značaja")



#calling for the  funtion that prints results in the txt file
#ispisivanje(text)


#Mapping of values 
recnik_value = {}


########################################################################
##################--- Additional function options ---###################


#function for conversion fo text in Cyrillinc to latin
def cyrillic_to_latin(text):
    # Mapping of Serbian Cyrillic characters to their Latin equivalents
    cyrillic_to_latin_map = {
        'А': 'A', 'Б': 'B', 'В': 'V', 'Г': 'G', 'Д': 'D', 'Ђ': 'Đ', 'Е': 'E', 'Ж': 'Ž', 
        'З': 'Z', 'И': 'I', 'Ј': 'J', 'К': 'K', 'Л': 'L', 'Љ': 'Lj', 'М': 'M', 'Н': 'N', 
        'Њ': 'Nj', 'О': 'O', 'П': 'P', 'Р': 'R', 'С': 'S', 'Т': 'T', 'Ћ': 'Ć', 'У': 'U', 
        'Ф': 'F', 'Х': 'H', 'Ц': 'C', 'Ч': 'Č', 'Џ': 'Dž', 'Ш': 'Š',
        'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'ђ': 'đ', 'е': 'e', 'ж': 'ž', 
        'з': 'z', 'и': 'i', 'ј': 'j', 'к': 'k', 'л': 'l', 'љ': 'lj', 'м': 'm', 'н': 'n', 
        'њ': 'nj', 'о': 'o', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'ћ': 'ć', 'у': 'u', 
        'ф': 'f', 'х': 'h', 'ц': 'c', 'ч': 'č', 'џ': 'dž', 'ш': 'š'
    }

    # Transliterate character by character
    transliterated_text = ''.join(cyrillic_to_latin_map.get(char, char) for char in text)
    return transliterated_text

