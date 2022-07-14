# NB: File contains derogatory language and slurs, done as part of trying to combat attempts
# to use slang/ spelling changes to skirt detection

BASE_AFRICAN_SLUR = "nigger"
BASE_ARAB_SLUR = "arab"
BASE_ASIAN_SLUR = "chink"
BASE_CAUCASIAN_SLUR = "honkie"
BASE_HISPANIC_SLUR = "spic"
BASE_HOMOSEXUAL_SLUR = "faggot"
BASE_JEWISH_SLUR = "kike"
BASE_MUSLIM_SLUR = "raghead"
BASE_REFUGEE_SLUR = "refugee"
BASE_WOMAN_SLUR = "whore"


# Extra slang slurs sourced from:
# https://en.wikipedia.org/wiki/List_of_ethnic_slurs
# https://en.wikipedia.org/wiki/List_of_religious_slurs
# https://en.wikipedia.org/wiki/LGBT_slang#Slurs
# https://en.wikipedia.org/wiki/Category:Pejorative_terms_for_women

#
# NB: Code currently only handles monograms, while bigrams are sometimes used as slurs

# With a sufficiently large training set or a broad enough input dict, this would be unneccesary,
# but such labelled datasets are sparsely available

# Reducing these slurs all to a single slur for each class reduces some information, but due to the broad range of
# slurs and limited training data, this was a necessity for this paper

_AFRICAN_SLURS = [
    "coon", "blackie", "blacky", "jigaboo", "negress", "negroid", "nigg", "niglet", "nigglet",
    "niggerette", "nig", "nignog",  "sheboon", "reggin", "spook", "tyrone", "golliwogg", "wog", "wogg",
    "quadroon", "mandingo", "jig", "jigg", "jigger", "zigabo", "zigaboo", "darky", "darkey", "darkie",
    "niggur", "nlgger", "mulatto", "mulattoe",
]
_ARAB_SLURS = ["naffer", "jawa", "jacker", "hucka", "durka", "derka", "snigger", "sandnigger", "hajj", "hajji",
    "haji", "hadji", "palesimian", "paki",
]
_ASIAN_SLURS = [
    "azn", "ayshan", "chigger", "chink", "ching", "chong", "chang", "gink", "mongoloid", "goloid", "rangoon",
    "zipperhead", "yigger", "japanigger", "chinazi", "chingchong", "chine", "nip", "dink", "chinaman", "gook",
]
_CAUCASIAN_SLURS = [
    "wypepo", "wypipo", "wigger", "huuwhyte", "huwhyte", "wigga", "wigglet", "whigger", "cracker", "honky",
    "honkie", "honkey"
]
_HISPANIC_SLURS = [
    "beaner", "spic", "spick", "spec", "spik", "spick", "wab", "wetback",
]
_HOMOSEXUAL_SLURS = [
    "batty", "fag", "fagg", "fagot", "faggit", "fagit", "lezzie", "poofter", "pooftah",
    "poufter", "pufter", "puffter", "poofta", "bulldyke", "kuweers", "kweers",
]
_JEWISH_SLURS = [
    "jewnose", "dreidl", "gew", "joo", "hebe", "kike", "heeb", "jewbacca", "wej", "zhid", "yom", "zog", "koshie",
    "jewess", "jewlet", "hooknose", "gargamel", "yid", "kyke", "itzig", "jewboy", "joow",
]
_MUSLIM_SLURS = [
    "moeslimes", "alkaida", "towelhead", "mozzie", "osama", "moolie", "mozlem", "moslem", "moslim",
    "mooslim", "muzzie", "raghead", "muzrat", "musrat", "muzzy",
]
_REFUGEE_SLURS = []
_WOMEN_SLURS = [
    "munter", "minger", "skank", "thot",  "karen", "feminazi", "skintern",
]


AFRICAN_SLUR_MAP = dict.fromkeys(_AFRICAN_SLURS, BASE_AFRICAN_SLUR)
ARAB_SLUR_MAP = dict.fromkeys(_ARAB_SLURS, BASE_ARAB_SLUR)
ASIAN_SLUR_MAP = dict.fromkeys(_ASIAN_SLURS, BASE_ASIAN_SLUR)
CAUCASIAN_SLUR_MAP = dict.fromkeys(_CAUCASIAN_SLURS, BASE_CAUCASIAN_SLUR)
HISPANIC_SLUR_MAP = dict.fromkeys(_HISPANIC_SLURS, BASE_HISPANIC_SLUR)
HOMOSEXUAL_SLUR_MAP = dict.fromkeys(_HOMOSEXUAL_SLURS, BASE_HOMOSEXUAL_SLUR)
JEWISH_SLUR_MAP = dict.fromkeys(_JEWISH_SLURS, BASE_JEWISH_SLUR)
MUSLIM_SLUR_MAP = dict.fromkeys(_MUSLIM_SLURS, BASE_MUSLIM_SLUR)
REFUGEE_SLUR_MAP = dict.fromkeys(_REFUGEE_SLURS, BASE_REFUGEE_SLUR)
WOMEN_SLUR_MAP = dict.fromkeys(_WOMEN_SLURS, BASE_WOMAN_SLUR)

SPECIFIC_TERM_MAPS = {
    "mudshark": f"{BASE_AFRICAN_SLUR} lover",
    "mudsharks": f"{BASE_AFRICAN_SLUR} lovers",
    "pleb": "plebian",
    "plebs": "plebians",
    "poc": "black person",
    "pocs": "black people",
    "wamen": "women",
    "wimen": "women",
    "wahmen": "women",
}

# quite a few instances of contractions being present without the mark, so is missed by ekphrasis
EXTRA_SLANG_MAP = {
    "dont": "do not",
    "arent": "are not",
    "cant": "can not",
    "wasnt": "was not",
    "werent": "were not",
    "im": "i am",
    "id": "i would",
    "ive": "i have",
    "isnt": "is not",
    "theres": "there is",
    "couldnt": "could not",
    "didnt": "did not",
    "theyd": "they would",
    "theyre": "they are",
    "theyll": "they will",
    "theyve": "they have",
    "hes": "he is",
    "hed": "he would",
    "shes": "she is",
    "thats": "that is",
    "weve": "we have",
    "youd": "you would",
    "youll": "you will",
    "hasnt": "has not",
    "havent": "have not",
    "youve": "you have",
}

SLUR_MAP_SINGULARS = {
    **AFRICAN_SLUR_MAP,
    **ARAB_SLUR_MAP,
    **ASIAN_SLUR_MAP,
    **CAUCASIAN_SLUR_MAP,
    **HISPANIC_SLUR_MAP,
    **HOMOSEXUAL_SLUR_MAP,
    **JEWISH_SLUR_MAP,
    **MUSLIM_SLUR_MAP,
    **REFUGEE_SLUR_MAP,
    **WOMEN_SLUR_MAP,
}

SLUR_MAP_PLURALS = {key+"s": val+"s" for key, val in SLUR_MAP_SINGULARS.items()}
SLUR_MAP_PLURALS.update({key+"z": val+"s" for key, val in SLUR_MAP_SINGULARS.items()})
SLUR_MAP = {**SLUR_MAP_SINGULARS, **SLUR_MAP_PLURALS, **SPECIFIC_TERM_MAPS}

MORE_SWEARS = {
    "yuh": "yeah",
    "u": "you",
    "r": "are",
    "gaf": "give a fuck",
}

SLANG_MAP = {**SLUR_MAP, **MORE_SWEARS, **EXTRA_SLANG_MAP}