import unittest

import pytest

from datasets import get_hateplain_df, get_raw_df
from src.dataset_utils.constants import BASE_AFRICAN_SLUR, BASE_ASIAN_SLUR, BASE_ARAB_SLUR, BASE_CAUCASIAN_SLUR, \
    BASE_HISPANIC_SLUR, BASE_HOMOSEXUAL_SLUR, BASE_JEWISH_SLUR, BASE_MUSLIM_SLUR, BASE_WOMAN_SLUR, BASE_REFUGEE_SLUR
from src.dataset_utils.preprocess_hatexplain import preprocess_hatexplain_df

from src.dataset_utils.string_preprocessing import clean_text, preprocess_text, check_is_english


# NB: Slurs and offensive language may be present in test cases, as they are important in the dataset


class TestCleanText:

    @pytest.mark.parametrize(["test_input", "expected"], [
        ("korrect", "correct"),
        ("aplle", "apple"),
        ("orangg", "orange"),
        ("lfmon", "lemon"),
        ("somnthing", "something"),
    ])
    def test_simple_phrases_works_as_expected(self, test_input, expected):
        assert " ".join(clean_text(test_input)) == expected

    @pytest.mark.parametrize(["test_input", "expected"], [
        ("i looove a song", "i love a song"),
        ("i can handle eeeeemphasis", "i can handle emphasis"),
        ("something itttteraaateed", "something iterated"),
    ])
    def test_elongation_works_as_expected(self, test_input, expected):
        assert " ".join(clean_text(test_input)) == expected


    @pytest.mark.parametrize(["test_input", "expected"], [
        ("wtf is that", "what the fuck is that"),
        ("kys loser", "kill yourself loser"),
        ("we have to get those ppl", "we have to get those people"),
    ])
    def test_slang_works_as_expected(self, test_input, expected):
        assert " ".join(clean_text(test_input)) == expected

    @pytest.mark.parametrize(["test_input", "expected"], [
        ("idk what is thib", "i do not know what is this"),
        ("wtf is this ttufb", "what the fuck is this stuff"),
    ])
    def test_with_mispelling_and_slang_works_as_normal(self, test_input, expected):
        assert " ".join(clean_text(test_input)) == expected

    @pytest.mark.parametrize("test_input, expected", [
        (" a racist might say niglet and niglets", f"a racist might say {BASE_AFRICAN_SLUR} and {BASE_AFRICAN_SLUR}s"),
        ("fagits are losers", f"{BASE_HOMOSEXUAL_SLUR}s are losers"),
        ("hooknose is a slur against jewish people", f"{BASE_JEWISH_SLUR} is a slur against jewish people"),
        (f"a misogynist would use a word like feminazi or thot",
         f"a misogynist would use a word like {BASE_WOMAN_SLUR} or {BASE_WOMAN_SLUR}"),
        (f"slurs againsy hispanic peiple include wetback and beaner",
         f"slurs against hispanic people include {BASE_HISPANIC_SLUR} and {BASE_HISPANIC_SLUR}")
    ])
    def test_text_containing_slurs_processed_as_expected(self, test_input, expected):
        assert " ".join(clean_text(test_input)) == expected

    @pytest.mark.parametrize(["test_input", "expected"], [
        ("not a bünch of fãgś", "not a bunch of fags"),
        ("ńïggęr ìš å ślûr", "nigger is a slur"),
         ])
    def test_with_accents_works_as_expected(self, test_input, expected):
        assert " ".join(clean_text(test_input)) == expected

    @pytest.mark.parametrize(["test_input", "expected"], [
        ("lmaoooo look at thosr faggs", f"laughing my ass off look at those {BASE_HOMOSEXUAL_SLUR}s"),
        ("wtf is goijg on with these feminazis", f"what the fuck is going on with these {BASE_WOMAN_SLUR}s"),
        ("dw i can   parse  rare slurs likr jigaboo", f"do not worry i can parse rare slurs like {BASE_AFRICAN_SLUR}")
    ])
    def test_with_broad_test_cases_works_as_expected(self, test_input, expected):
        assert " ".join(clean_text(test_input)) == expected

    @pytest.mark.parametrize(["test_input", "expected"], [
        ("i don't want to deal with this", "i do not want to deal with this"),
        ("i'm a short word", "i am a short word"),
    ])
    def test_with_contractions_works_as_expected(self, test_input, expected):
        assert " ".join(clean_text(test_input)) == expected

    @pytest.mark.parametrize(["test_input", "expected"], [
        (" i don \' t líkë this    much", "i do not like this much"),
        ("\" i am not gáy \" said the gay man", "i am not gay said the gay man")
    ])
    def test_with_complex_cases_works_as_expected(self, test_input, expected):
        assert " ".join(clean_text(test_input)) == expected


class TestPreprocessText:

    @pytest.mark.parametrize(["test_input", "expected"], [
        ("I am an input", f"i am an input"),
        ("i    believe this should  work", "i believe this should work")
        ])
    def test_for_simple_case_works_as_expected(self, test_input, expected):
        assert preprocess_text(test_input) == expected.split(" ")

    @pytest.mark.parametrize(["test_input", "expected"], [
        ("look at! this thing :) <token>", f"look at this thing <token>"),
        ("<user> is a poop head!", "<user> is a poop head"),
    ])
    def test_for_grammar_works_as_expected(self, test_input, expected):
        assert preprocess_text(test_input) == expected.split(" ")



    @pytest.mark.parametrize(["test_input", "expected"], [
        ("hey <user> what does 🙃 looj like", f"hey <user> what does look like"),
    ])
    def test_for_emoji_works_as_expected(self, test_input, expected):
        assert preprocess_text(test_input) == expected.split(" ")


class TestCheckIsEnglish:

    @pytest.mark.parametrize("test_input", [
            "hello this is an english string",
            "hello i am in english"
        ]
    )
    def test_for_normal_strings_returns_as_expected(self, test_input):
        prediction, probability = check_is_english(test_input)

        assert prediction == "en"
        assert probability > 0.5


    @pytest.mark.parametrize("test_input", [
            "bonjour je suis francais",
            "salut ce nest pas anglais"
        ]
    )
    def test_for_non_english_works_as_expected(self, test_input):
        prediction, probability = check_is_english(test_input)

        assert prediction != "en"


class TestIntegration:

    def test_for_contractions_works_as_expected(self):
        post_id = "1182358602553679872_twitter"  # has input "<user> yeah you ` re a retard"

        input_df = get_raw_df().query(f"""post_id == "{post_id}" """)

        cleaned_df = preprocess_hatexplain_df(input_df)
        output_str = cleaned_df.iloc[[0]]["clean_text"].values[0]
        assert output_str == "<user> yeah you are a retard"

