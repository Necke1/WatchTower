import classla
nlpc = classla.Pipeline('sr') 


def load_words(word_file):
    with open(word_file, 'r', encoding='utf-8') as f:
        ter_words = set(word.strip().lower() for line in f for word in line.split())
    return ter_words

def check_terror_words(text):
    doc = nlpc(text)
    found_words = set()
    for sent in doc.sentences:
        for word in sent.words:
            lemma = word.lemma.lower()
            if lemma in recnik:
                found_words.add(lemma)
    return found_words



recnik=load_words('recnik.txt')
text="terorista je pustio 🚀 i doslo je do velikog praska."



matches = check_terror_words(text)

if matches:
    print(f"teroristicne reci pronadjene: {matches}")
else:
    print(f"Nisu nadjene teroristicne reci")







"""
print(docc1)
print(docc1.ents)
"""
