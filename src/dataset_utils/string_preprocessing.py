import pickle as pkl
import re
import string
from functools import lru_cache
from pathlib import Path
from typing import List

import cld3
import emoji
import pkg_resources
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.dicts.noslang.slangdict import slangdict
from sklearn.feature_extraction.text import strip_accents_ascii
from symspellpy import SymSpell, Verbosity

from src.dataset_utils.LW_constants import SLANG_MAP


@lru_cache()
def get_spellchecker() -> SymSpell:
    symspell = SymSpell()
    dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
    symspell.load_dictionary(dictionary_path, 0, 1)
    return symspell


@lru_cache()
def get_ekphrasis_processor() -> TextPreProcessor:
    """Cached to stop Ekphrasis from repeatedly reading from file and reinitialising"""
    return TextPreProcessor(
        annotate="elongated",
        unpack_contractions=True,
        spell_correction=False,
        spell_correct_elong=False,
    )


@lru_cache()
def get_slangdict() -> dict:
    slangdict_path = Path(__file__).parent / "slangdict.pickle"

    if not slangdict_path.is_file():
        base_slangdict = slangdict
        with open(slangdict_path, "wb") as file:
            pkl.dump(slangdict, file)
    else:
        with open(slangdict_path, "rb") as file:
            base_slangdict = pkl.load(file)

    # Remove unneeded slang dict entries for this case.
    # The base ekiphrasis dictionary is useful for some things, but does make some slips for our use case.
    unrequired_keys = [
        "kkk",
        "yo",
        "cunt",
        "don",
        "ah",
        "gaf",
        "homo",
    ]

    for key in unrequired_keys:
        base_slangdict.pop(key, "")

    base_slangdict.update(SLANG_MAP)

    return base_slangdict


def check_is_english(clean_str: str) -> tuple:
    input_str = clean_str.replace(" i ", " I ")
    detected = cld3.get_language(input_str)
    return detected.language, detected.probability


def dict_replace(input_str: str, replace_dict: dict) -> str:
    wordlist = input_str.split(" ")
    out = " ".join(
        replace_dict[w] if w in replace_dict else w for w in wordlist
    )
    return out


def remove_emoji(text: str) -> str:
    regex_pattern = emoji.get_emoji_regexp()
    return regex_pattern.sub("", text)


def clean_text(single_spaced_str: str) -> List[str]:
    slang_dict = get_slangdict()
    text_preprocessor = get_ekphrasis_processor()

    single_spaced_str = (" ".join(single_spaced_str.split()))     # replace multiple whitespaces with just one whitespace
    input_str = single_spaced_str.replace(" ' ", "'").replace(" ’ ", "'")    # replace any spit contractions into one word

    first_pass_replace = dict_replace(input_str, slang_dict)           # First pass, replace any visible slurs
    clean_str = text_preprocessor.pre_process_doc(first_pass_replace)  # First pass, handle repetitions
    final_pass_replaced = dict_replace(clean_str, slang_dict)          # Replace found after cleaning
    final_pass_cleaned = text_preprocessor.pre_process_doc(final_pass_replaced)  # Handle new contractions introduced

    # Check spellings using SymSpell
    spell_corrector = get_spellchecker()

    tokens = []
    for word in final_pass_cleaned.split(" "):
        if re.match(r"<[a-z]*>", word):  # add in tokens
            if word != ("<elongated>"):  # remove elongated token added in as part of preprocessing
                tokens.append(word)
        else:
            no_punct_word = word.translate(str.maketrans('', '',  string.punctuation))  # removes punctuation
            unaccented_word = strip_accents_ascii(no_punct_word)
            if len(unaccented_word) > 2:
                spell_correct_res = spell_corrector.lookup(unaccented_word.lower(), Verbosity.TOP)
                if len(spell_correct_res) == 0:  # no similar word, just return as given
                    tokens.append(unaccented_word)
                else:
                    tokens.append(spell_correct_res[0].term)  # give corrected word
            elif len(unaccented_word) >= 1:
                tokens.append(unaccented_word.lower())

    return tokens


def preprocess_text(raw_str: str) -> List[str]:
    no_emoji = remove_emoji(raw_str)
    cleaned_text = clean_text(no_emoji.lower())
    return cleaned_text
