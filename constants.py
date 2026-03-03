# -*- coding: utf-8 -*-
"""
WatchTower: Constants and Configuration
"""

# File paths
DEFAULT_DICTIONARY_FILE = "biblioteke/recnik.txt"
USER_CORRECTIONS_FILE   = "biblioteke/korekcija.txt"
OUTPUT_FILE             = "biblioteke/rezultat.txt"

# Spell checker
MAX_SUGGESTIONS = 5

# Risk score thresholds
RISK_THRESHOLDS = {
    'high':   {'score': 50, 'unique_terms': 10},
    'medium': {'score': 25, 'unique_terms': 5},
    'low':    {'score': 10, 'unique_terms': 3}
}

# ---------------------------------------------------------------------------
# Demo sample texts  
# ---------------------------------------------------------------------------
SAMPLE_TEXTS = [
    {
        'name': 'Bezopasan tekst',
        'text': 'Danas je lep dan. Sunce sija i ptice pevaju. Deca se igraju u parku.',
        'description': 'Normalan, bezopasan tekst bez rizičnih termina'
    },
    {
        'name': 'Tekst sa pojedinim rizičnim terminima',
        'text': 'Terorizam je ozbiljan problem u savremenom svetu. '
                'Države saraduju u borbi protiv ovog fenomena.',
        'description': 'Akademski/novinski stil sa pojedinačnim terminima'
    },
    {
        'name': 'Tekst sa više rizičnih termina',
        'text': 'Ekstremisti koriste nasilje i bombe da postignu svoje ciljeve. '
                'Mržnja vodi ka terorizmu.',
        'description': 'Više rizičnih termina, srednji nivo rizika'
    },
    {
        'name': 'Tekst sa emodžijima',
        'text': 'Pažnja na potencijalne pretnje! 💣 Može biti opasno. 🔫 Treba biti oprezan.',
        'description': 'Test detekcije emoji karaktera'
    },
    {
        'name': 'Tekst sa pravopisnim greškama',
        'text': 'Terorista je pustio bombu i doslo je do velikog praska. '
                'Napad je bio uzasan i strasn.',
        'description': 'Test spell checkera'
    },
    {
        'name': 'Tekst sa greškama i visokim rizikom',
        'text': 'Ekstremisti planiraju napad bombama i oruzjem. '
                'Nasilje i mrznja vode ka terorizmu i ubistima.',
        'description': 'Pravopisne greške + visok rizik'
    },
    {
        'name': 'Ćirilički tekst',
        'text': 'Тероризам је озбиљан проблем. Екстремисти користе насиље.',
        'description': 'Test ćiriličnog pisma'
    },
]

# Risk level descriptions and recommendations in Serbian
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