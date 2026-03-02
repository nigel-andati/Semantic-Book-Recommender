"""
Expanded label sets for high-precision book classification.
Organized by emotional nuance, historical/political themes, psychological depth, and atmosphere.
Designed for phrase-level understanding and precise thematic alignment.
"""

# ---------------------------------------------------------------------------
# EMOTION / TONE (Nuanced, precision-focused)
# ---------------------------------------------------------------------------

# Sadness & heaviness
EMOTION_SADNESS = [
    "sad",
    "sorrowful",
    "grief-filled",
    "melancholic",
    "somber",
    "mournful",
    "bleak",
    "heavy-hearted",
    "emotionally painful",
    "stories about loss",
    "coping with loss",
    "dealing with death",
    "mourning",
    "emotionally weighty",
]

# Hope & warmth
EMOTION_HOPE = [
    "hopeful",
    "uplifting",
    "heartwarming",
    "life-affirming",
    "redemptive",
    "inspiring",
    "optimistic",
    "resilience",
    "recovery",
    "second chances",
]

# Dark & disturbing
EMOTION_DARK = [
    "dark",
    "unsettling",
    "haunting",
    "grim",
    "psychologically dark",
    "morally complex",
    "tragic",
    "chilling",
    "human suffering",
    "ethical dilemmas",
    "moral ambiguity",
]

# Comfort & calm
EMOTION_COMFORT = [
    "cozy",
    "gentle",
    "comforting",
    "peaceful",
    "wholesome",
    "slice-of-life",
    "slow-paced",
    "reflective",
]

# Nostalgia & reflection
EMOTION_NOSTALGIA = [
    "nostalgic",
    "bittersweet",
    "reflective on the past",
    "coming-of-age reflection",
    "memory-driven",
]

# Combined for zero-shot (deduplicated, rich vocabulary)
EMOTION_LABELS = list(dict.fromkeys(
    EMOTION_SADNESS + EMOTION_HOPE + EMOTION_DARK + EMOTION_COMFORT + EMOTION_NOSTALGIA
))

# ---------------------------------------------------------------------------
# THEME: Historical / Political / Social
# ---------------------------------------------------------------------------

# Colonialism & power
THEME_COLONIALISM = [
    "colonialism",
    "imperialism",
    "decolonization",
    "postcolonial identity",
    "cultural displacement",
    "cultural erasure",
    "indigenous experiences",
    "resistance to colonial rule",
    "aftermath of empire",
    "power imbalance between cultures",
]

# Justice & inequality
THEME_JUSTICE = [
    "social justice",
    "racial inequality",
    "oppression",
    "systemic oppression",
    "resistance",
    "class struggle",
    "economic inequality",
    "civil rights",
    "activism",
    "political resistance",
]

# Identity & belonging
THEME_IDENTITY = [
    "identity",
    "belonging",
    "diaspora",
    "migration",
    "exile",
    "identity crisis",
    "bicultural identity",
    "searching for belonging",
    "generational conflict",
    "cultural conflict",
]

# ---------------------------------------------------------------------------
# THEME: Psychological & Philosophical
# ---------------------------------------------------------------------------

# Psychological depth
THEME_PSYCHOLOGICAL = [
    "trauma",
    "healing from trauma",
    "emotional growth",
    "loneliness",
    "isolation",
    "alienation",
    "self-discovery",
    "inner conflict",
    "personal transformation",
]

# Philosophical
THEME_PHILOSOPHICAL = [
    "meaning of life",
    "existential themes",
    "moral dilemmas",
    "ethical conflicts",
    "human nature",
    "free will",
    "fate vs choice",
]

# Combined theme labels
THEME_LABELS = list(dict.fromkeys(
    THEME_COLONIALISM + THEME_JUSTICE + THEME_IDENTITY
    + THEME_PSYCHOLOGICAL + THEME_PHILOSOPHICAL
))

# ---------------------------------------------------------------------------
# GENRE
# ---------------------------------------------------------------------------
GENRE_LABELS = [
    "fiction",
    "non-fiction",
    "romance",
    "mystery",
    "thriller",
    "science fiction",
    "fantasy",
    "historical fiction",
    "biography",
    "memoir",
    "literary fiction",
    "horror",
    "young adult",
    "self-help",
    "philosophy",
    "political",
    "crime",
    "adventure",
    "drama",
    "comedy",
]

# ---------------------------------------------------------------------------
# ATMOSPHERE & TONE
# ---------------------------------------------------------------------------
ATMOSPHERE_LABELS = [
    "mysterious",
    "suspenseful",
    "eerie",
    "tense",
    "foreboding",
    "adventurous",
    "epic journey",
    "quest-driven",
    "introspective",
    "character-driven",
    "emotionally immersive",
    "fast-paced thriller",
    "slow reflective narrative",
    "lyrical",
    "dreamy",
    "gritty",
    "surreal",
    "claustrophobic",
    "intimate",
    "meditative",
]

# ---------------------------------------------------------------------------
# SEMANTIC VOCABULARY (Phrase-level for embedding matching)
# Maps canonical labels to synonyms AND full phrases.
# Enables "a sad story about colonialism" -> sad + colonialism
# ---------------------------------------------------------------------------
SEMANTIC_VOCAB: dict[str, list[str]] = {
    # Sadness & heaviness
    "sad": [
        "sadness", "sorrow", "grief", "melancholy", "depressing", "heartbreaking",
        "feeling down", "blue", "downcast", "stories about loss", "coping with loss",
        "dealing with death", "mourning a loved one", "emotionally weighty",
        "reflective on suffering", "heavy-hearted",
    ],
    "sorrowful": ["sorrow", "grief-filled", "mournful", "weeping", "lament"],
    "grief-filled": ["grief", "bereavement", "loss", "mourning", "sorrow"],
    "melancholic": ["melancholy", "sad", "pensive", "gloomy", "wistful"],
    "somber": ["serious", "grave", "solemn", "dark", "gloomy"],
    "mournful": ["mourning", "grief", "sad", "lamenting", "sorrowful"],
    "bleak": ["dark", "grim", "hopeless", "dismal", "dreary"],
    "emotionally weighty": ["heavy", "weighty", "profound", "intense emotion"],
    "stories about loss": ["loss", "grief", "death", "bereavement", "mourning"],
    "coping with loss": ["grief", "bereavement", "healing", "loss"],
    "dealing with death": ["death", "dying", "mourning", "grief", "loss"],

    # Hope & warmth
    "hopeful": [
        "hope", "optimistic", "uplifting", "inspiring", "encouraging",
        "positive outlook", "second chances", "redemption",
    ],
    "uplifting": ["inspiring", "hopeful", "heartwarming", "feel-good", "positive"],
    "heartwarming": ["warm", "touching", "sweet", "comforting", "uplifting"],
    "life-affirming": ["affirming", "positive", "hopeful", "celebrating life"],
    "redemptive": ["redemption", "second chance", "forgiveness", "healing"],
    "inspiring": ["inspirational", "motivating", "uplifting", "hopeful"],
    "resilience": ["resilient", "overcoming", "perseverance", "strength"],
    "recovery": ["healing", "recovering", "rebuilding", "second chances"],
    "second chances": ["redemption", "new beginning", "starting over"],

    # Dark & disturbing
    "dark": [
        "darkness", "bleak", "grim", "dismal", "gloomy", "noir",
        "psychologically dark", "disturbing",
    ],
    "unsettling": ["disturbing", "uncomfortable", "eerie", "creepy", "ominous"],
    "haunting": ["unforgettable", "lingering", "ghostly", "eerie", "melancholic"],
    "grim": ["dark", "bleak", "harsh", "hopeless", "somber"],
    "psychologically dark": ["dark psychology", "disturbing", "unsettling", "twisted"],
    "morally complex": ["moral ambiguity", "gray area", "ethical complexity", "no clear good or evil"],
    "tragic": ["tragedy", "devastating", "heart-wrenching", "sorrowful"],
    "chilling": ["frightening", "eerie", "disturbing", "unnerving"],
    "human suffering": ["suffering", "pain", "trauma", "anguish"],
    "ethical dilemmas": ["moral dilemma", "ethical conflict", "difficult choices"],
    "moral ambiguity": ["morally gray", "complex morality", "no clear right or wrong"],

    # Comfort & calm
    "cozy": [
        "comfortable", "warm", "snug", "inviting", "homely",
        "comforting read", "feel-good",
    ],
    "gentle": ["soft", "mild", "tender", "kind", "soothing"],
    "comforting": ["soothing", "reassuring", "warm", "safe", "gentle"],
    "peaceful": ["calm", "serene", "tranquil", "quiet", "peaceful"],
    "wholesome": ["heartwarming", "pure", "sweet", "feel-good", "family-friendly"],
    "slice-of-life": ["everyday life", "mundane", "realistic", "domestic"],
    "slow-paced": ["leisurely", "contemplative", "unhurried", "meditative"],
    "reflective": ["introspective", "contemplative", "meditative", "thoughtful"],

    # Nostalgia & reflection
    "nostalgic": [
        "nostalgia", "reminiscent", "looking back", "memories", "sentimental",
        "reflective on the past",
    ],
    "bittersweet": ["mixed feelings", "poignant", "touching", "melancholic joy", "sweet and sad"],
    "reflective on the past": ["memory", "reminiscence", "looking back", "nostalgia"],
    "coming-of-age reflection": ["coming of age", "growing up", "youth", "adolescence"],
    "memory-driven": ["memories", "recollection", "past", "nostalgia"],

    # Historical / Political / Social
    "colonialism": [
        "colonial", "empire", "imperialism", "colonization", "postcolonial",
        "cultural displacement", "resistance to colonial rule", "aftermath of empire",
    ],
    "postcolonial identity": ["postcolonial", "decolonization", "colonial legacy", "identity after empire"],
    "cultural displacement": ["displacement", "exile", "migration", "uprooted"],
    "oppression": ["oppressed", "tyranny", "subjugation", "repression", "injustice"],
    "resistance": ["rebellion", "revolt", "defiance", "fighting back", "activism"],
    "social justice": ["justice", "equality", "civil rights", "activism"],
    "racial inequality": ["race", "racism", "discrimination", "inequality"],
    "class struggle": ["class", "economic inequality", "working class", "poverty"],
    "identity": ["self", "who am I", "cultural identity", "selfhood", "identity crisis"],
    "migration": ["immigration", "displacement", "refugee", "exile", "diaspora"],
    "diaspora": ["displacement", "migration", "exile", "scattered community"],
    "belonging": ["community", "fitting in", "acceptance", "connection", "searching for belonging"],

    # Psychological & Philosophical
    "trauma": [
        "traumatic", "PTSD", "wound", "healing", "survivor",
        "healing from trauma", "coping with trauma",
    ],
    "healing from trauma": ["trauma recovery", "healing", "survivor", "overcoming"],
    "emotional growth": ["personal growth", "transformation", "development", "coming into one's own"],
    "loneliness": ["alone", "isolated", "solitude", "alienation", "isolation"],
    "self-discovery": ["finding oneself", "journey of self", "self-exploration", "identity"],
    "meaning of life": ["existential", "philosophy", "purpose", "existence"],
    "existential themes": ["existentialism", "meaning", "existence", "philosophical"],
    "moral dilemmas": ["ethical dilemma", "moral conflict", "difficult choice", "ethical complexity"],

    # Atmosphere
    "mysterious": ["mystery", "enigmatic", "puzzling", "curious", "intriguing", "whodunit"],
    "suspenseful": ["suspense", "tense", "nail-biting", "edge-of-your-seat", "thrilling"],
    "introspective": ["inner thoughts", "character-driven", "psychological depth", "reflective"],
    "character-driven": ["character study", "focus on characters", "psychological", "inner life"],
    "fast-paced thriller": ["thriller", "action", "fast", "page-turner", "suspenseful"],
    "slow reflective narrative": ["slow-paced", "contemplative", "meditative", "reflective", "unhurried"],
}
