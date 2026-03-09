# -*- coding: utf-8 -*-
"""
WatchTower: Constants and Configuration
"""

import os

# File paths  (env-var overridable for deployment flexibility)
DEFAULT_DICTIONARY_FILE = os.getenv("WT_DICT_FILE",        "biblioteke/recnik.txt")
SPELL_CORRECTIONS_FILE   = os.getenv("WT_CORRECTIONS_FILE", "biblioteke/korekcija_pravopisa.txt")
LEARNED_PATTERNS_FILE   = os.getenv("WT_PATTERNS_FILE",    "biblioteke/korekcija_paterna.txt")
OUTPUT_FILE             = os.getenv("WT_OUTPUT_FILE",       "biblioteke/rezultat.txt")

# Spell checker
MAX_SUGGESTIONS = 5

# Single-document risk thresholds
# Uses two proportional signals so the result scales with document length.
#
#   score_per_word  — total weighted score divided by total word count
#   term_density    — unique risky terms as a % of total words
#
# A level fires only when BOTH of the following are true:
#   1. The absolute score meets the min_score floor  (prevents short texts
#      with a single risky word from being inflated by density alone)
#   2. At least one proportional signal exceeds its threshold
RISK_THRESHOLDS = {
    'high':   {'score_per_word': 2.0, 'term_density': 5.0, 'min_score': 30},
    'medium': {'score_per_word': 0.5, 'term_density': 2.0, 'min_score': 15},
    'low':    {'score_per_word': 0.1, 'term_density': 0.5, 'min_score':  5},
}

# Chat-level risk thresholds
# Chat-level risk thresholds — uses WEIGHTED average, not raw average.
# Each message's score is first multiplied by CHAT_MESSAGE_RISK_WEIGHTS[risk_level]
# before being averaged, so a single VISOK RIZIK message can never be
# diluted away by many clean messages.
#
#   avg_weighted   — exponentially-weighted average risk per message
#   flagged_pct    — percentage of messages with score > 0  (0.0–100.0)
#   min_score_sum  — weighted score floor
#
# A level fires only when BOTH conditions are true:
#   1. weighted_score_sum >= min_score_sum
#   2. At least one proportional signal exceeds its threshold
CHAT_RISK_THRESHOLDS = {
    'high':   {'avg_weighted': 20.0, 'flagged_pct': 30.0, 'min_score_sum': 200},
    'medium': {'avg_weighted':  4.0, 'flagged_pct': 15.0, 'min_score_sum':  30},
    'low':    {'avg_weighted':  0.5, 'flagged_pct':  5.0, 'min_score_sum':   5},
}

# Exponential severity multipliers for chat-level scoring.
# A message's raw score is multiplied by this factor before being summed
# into the weighted total.  Doubling between each level ensures that a
# single VISOK RIZIK message (×12) outweighs many MINIMALAN ones (×1).
#
# Example: 100 messages — 10×MINIMALAN (score 3) + 1×VISOK (score 40):
#   weighted = 10×3×1 + 1×40×12 = 510
#   weighted_avg = 5.1  →  SREDNJI RIZIK   (not diluted by 89 clean messages)
CHAT_MESSAGE_RISK_WEIGHTS = {
    'VISOK RIZIK':    12,
    'SREDNJI RIZIK':   5,
    'NIZAK RIZIK':     2,
    'MINIMALAN RIZIK': 1,
    'BEZ RIZIKA':      0,
}

# Per-message risk thresholds
# Raised slightly to account for amplifier bonuses on genuinely dangerous messages.
# Thresholds:
#   score >= 40  → VISOK RIZIK    (e.g. "planiramo napad bombama" → amplified cluster)
#   score >= 25  → SREDNJI RIZIK  (e.g. "prikupljamo oružje" amplified)
#   score >=  8  → NIZAK RIZIK    (e.g. single high-weight term, no context)
#   score  >  0  → MINIMALAN RIZIK
#
# Note: with context scoring, innocent messages will drop toward 0-3 while
# genuinely dangerous ones will jump to 30-80+, creating much clearer separation.
MESSAGE_RISK_THRESHOLDS = {
    'high':   40,
    'medium': 25,
    'low':     8,
}

# ---------------------------------------------------------------------------
# Context-Aware Scoring Rules
# ---------------------------------------------------------------------------
#
# These rules modify the raw dictionary weight of a matched term based on
# the words surrounding it in the same sentence / nearby window.
#
# IMPORTANT: multipliers are applied in this priority order:
#   1. Negation       (strongest dampener — "ne planiramo napad")
#   2. Academic/media dampener ("film o terorizmu", "istorija rata")
#   3. Intent amplifier ("organizujemo napad", "skupljamo oružje")
#   4. Cluster bonus  (added after per-term scoring is complete)
#
# Each rule has a 'window' — the number of tokens looked at before (and/or
# after) the matched term.  Smaller window = more precise, less recall.
#
# All multipliers must be > 0.  Values < 1.0 reduce the score (dampener).
# Values > 1.0 increase the score (amplifier).  1.0 = no change.

CONTEXT_RULES = {

    # ── Negation ─────────────────────────────────────────────────────────────
    # Words that negate or hypothetically frame a risky term.
    # Window: only look BEFORE the matched term (negations precede).
    # Multiplier: 0.15 — almost entirely removes the score. "Ne planiramo
    # napad" should not score like an actual threat.
    'negation_words': {
        # Serbian Latin
        'ne', 'nije', 'nema', 'nemam', 'nemamo', 'nemaju', 'nisam', 'nisi',
        'nismo', 'niste', 'nisu', 'nikad', 'nikada', 'nikako', 'nipošto',
        'bez', 'niti', 'ni', 'neću', 'nećemo', 'odbijam', 'odbijamo',
        'sprečiti', 'sprečili', 'sprečavamo', 'sprečen', 'suprotstavljamo',
        'protiv', 'borba-protiv', 'odbrana', 'zaštita',
        # Common journalistic/conditional framing
        'navodno', 'tobože', 'hipotetički', 'zamislimo', 'primer',
        'ako', 'ukoliko', 'pretpostavimo',
    },
    'negation_window': 5,        # tokens before the matched term
    'negation_multiplier': 0.15,

    # ── Academic / Media / Historical dampeners ───────────────────────────────
    # Words that indicate the risky term is being discussed in a journalistic,
    # academic, historical, or fictional context rather than as a real threat.
    # Window: look both before AND after (context can follow the term too).
    # Multiplier: 0.35 — significantly reduces but doesn't eliminate the score,
    # because even academic texts should not score zero (they still contain the
    # vocabulary and may warrant a lower-level flag for human review).
    'dampener_words': {
        # Media / journalism
        'film', 'filmski', 'serija', 'dokumentarni', 'dokumentarac',
        'vesti', 'novine', 'novinski', 'članak', 'reportaža', 'izveštaj',
        'izveštaji', 'mediji', 'medijski', 'televizija', 'emisija',
        # Academic / research
        'istorija', 'istorijski', 'istorijat', 'analiza', 'istraživanje',
        'istraživač', 'akademski', 'naučni', 'studija', 'knjiga', 'roman',
        'literatura', 'predavanje', 'nastava', 'škola', 'udžbenik',
        # Law enforcement / prevention framing
        'hapšenje', 'uhapšen', 'osuđen', 'osuđeni', 'procesuiran',
        'policija', 'tužilaštvo', 'sud', 'presuda', 'istraga',
        'antiteroristički', 'kontraterorizam', 'sprečen', 'osujetiti',
        # Historical / past tense markers
        'nekada', 'nekad', 'ranije', 'prošlost', 'tokom-rata',
        # Fiction / game markers
        'igra', 'video-igra', 'scenario', 'priča', 'fikcija',
    },
    'dampener_window': 7,        # tokens before AND after the matched term
    'dampener_multiplier': 0.35,

    # ── Intent amplifiers ─────────────────────────────────────────────────────
    # Words that indicate active intent, planning, or recruitment — the most
    # reliable signal that a message is a real threat rather than discussion.
    # Window: look BEFORE the matched term (intent typically precedes the noun).
    # Multiplier: 2.0 — doubles the raw weight when these precede a risky term.
    #
    # Examples:
    #   "organizujemo napad"    → napad(9) × 2.0 = 18
    #   "skupljamo oružje"      → oružje(9) × 2.0 = 18
    #   "pošaljite donacije za" → donacija(2) × 2.0 = 4  (still low alone)
    #   "regrutujemo borce"     → combined with cluster → high
    'amplifier_words': {
        # Planning / organizing
        'organizujemo', 'organizovati', 'planiramo', 'planirati',
        'pripremamo', 'pripremiti', 'sprovodimo', 'sprovesti',
        'koordinišemo', 'koordinirati', 'izvršiti', 'realizovati',
        # Recruitment
        'regrutujemo', 'regrutovati', 'tražimo', 'tražiti',
        'pozivamo', 'pridružite', 'pridruži', 'prijavite', 'prijaviti',
        # Acquisition / financing
        'skupljamo', 'prikupljamo', 'prikupiti', 'nabavljamo', 'nabaviti',
        'obezbeđujemo', 'obezbediti', 'finansiramo', 'doniramo',
        'šaljite', 'pošaljite', 'slati', 'poslati', 'kupujemo', 'kupiti',
        # Targeting / concealment
        'napadamo', 'ciljamo', 'tajno', 'klandestino', 'sakrivamo',
        'koristimo', 'upotrebljavamo',
        # Commands / calls to action
        'mora', 'moramo', 'moraju', 'treba', 'trebamo', 'trebaju',
        'hitno', 'odmah', 'svi', 'budite', 'pripremite', 'čekajte',
    },
    'amplifier_window': 6,       # tokens before the matched term
    'amplifier_multiplier': 2.0,

    # ── Cluster / co-occurrence bonus ─────────────────────────────────────────
    # When two or more high-weight terms appear within a short window of each
    # other, it strongly suggests a coherent threat message rather than an
    # isolated mention.
    #
    # The bonus is added to the TOTAL score AFTER per-term scoring is done.
    # For each pair of qualifying terms within the window:
    #   bonus = (weight_a + weight_b) × (cluster_bonus_multiplier - 1)
    #
    # Example:
    #   "skupljamo sredstva za oružje i bombu" →
    #     oružje(9) + bomba(9) = 18 base score
    #     cluster bonus = (9+9) × 0.5 = 9
    #     total = 27 from these two terms alone  →  SREDNJI RIZIK
    #
    # cluster_min_weight: only terms with weight >= this value trigger the bonus.
    # Set to 7 so common low-weight terms (park=1, telefon=1) never contribute.
    'cluster_window': 15,
    'cluster_min_weight': 7,
    'cluster_bonus_multiplier': 1.5,  # adds 50% of each qualifying term's weight

    # ── Score cap ─────────────────────────────────────────────────────────────
    # Maximum adjusted score for a single term occurrence.
    # Prevents a single amplified term from dominating the entire score and
    # allows the risk bands to remain meaningful.
    # Default: 25  (= max raw weight 10 × amplifier 2.0 × some rounding room)
    'max_single_term_score': 25,
}


# ---------------------------------------------------------------------------
# Demo sample texts  (single source of truth — used by both CLI and API)
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
    # ── NEW context-test samples ─────────────────────────────────────────────
    {
        'name': 'Negacija — trebalo bi biti BEZ RIZIKA',
        'text': 'Mi nismo teroristi i nema govora o napadu ni o oružju. '
                'Borba protiv ekstremizma je naš jedini cilj.',
        'description': 'Negacija ispred rizičnih termina — skor treba biti nizak'
    },
    {
        'name': 'Akademski kontekst — niži skor',
        'text': 'Istorijska analiza pokazuje da je terorizam kao fenomen '
                'prisutan od 19. veka. Film dokumentarac o ratu prikazuje nasilje.',
        'description': 'Novinski/akademski kontekst — dampener treba da smanji skor'
    },
    {
        'name': 'Visok rizik sa pojačivačem — VISOK RIZIK',
        'text': 'Organizujemo napad na institucije. Skupljamo oružje i bombe. '
                'Regrutujemo borce koji su voljni da se pridruže.',
        'description': 'Pojačivači namere + klaster — treba VISOK RIZIK'
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