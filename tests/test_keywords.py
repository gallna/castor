import pytest

from castor.keywords import Signal, match_keyword

EN = {"turn_keywords": ("over",), "end_keywords": ("over and out",)}
PL = {"turn_keywords": ("odbiór",), "end_keywords": ("bez odbioru",)}


def test_phrase_without_keyword_is_plain_speech():
    match = match_keyword("what is the weather like", **EN)

    assert match.signal is Signal.NONE
    assert match.text == "what is the weather like"


def test_trailing_over_ends_turn_and_is_stripped():
    match = match_keyword("tell me a joke over", **EN)

    assert match.signal is Signal.END_TURN
    assert match.text == "tell me a joke"


def test_over_and_out_ends_conversation_not_just_turn():
    match = match_keyword("thanks for the chat over and out", **EN)

    assert match.signal is Signal.END_CONVERSATION
    assert match.text == "thanks for the chat"


def test_over_in_the_middle_of_phrase_is_ignored():
    match = match_keyword("move it over there", **EN)

    assert match.signal is Signal.NONE
    assert match.text == "move it over there"


def test_word_containing_over_is_not_a_keyword():
    match = match_keyword("that was a hangover", **EN)

    assert match.signal is Signal.NONE


def test_keyword_alone_gives_empty_text():
    match = match_keyword("over", **EN)

    assert match.signal is Signal.END_TURN
    assert match.text == ""


@pytest.mark.parametrize(
    ("phrase", "expected"),
    [("Tell me more, OVER.", "Tell me more,"), ("tell me more over!", "tell me more")],
)
def test_matching_ignores_case_and_punctuation(phrase, expected):
    match = match_keyword(phrase, **EN)

    assert match.signal is Signal.END_TURN
    assert match.text == expected


def test_polish_turn_keyword():
    match = match_keyword("opowiedz mi coś odbiór", **PL)

    assert match.signal is Signal.END_TURN
    assert match.text == "opowiedz mi coś"


def test_polish_turn_keyword_without_diacritics():
    match = match_keyword("opowiedz mi coś odbior", **PL)

    assert match.signal is Signal.END_TURN


def test_polish_end_keyword():
    match = match_keyword("dziękuję bez odbioru", **PL)

    assert match.signal is Signal.END_CONVERSATION
    assert match.text == "dziękuję"


def test_empty_phrase():
    match = match_keyword("", **EN)

    assert match.signal is Signal.NONE
    assert match.text == ""


EN_WITH_LAST = {**EN, "last_keywords": ("send last",)}
PL_WITH_LAST = {**PL, "last_keywords": ("wyślij ostatnie",)}


def test_send_last_keyword_is_recognised_and_stripped():
    match = match_keyword("what is the capital of france send last", **EN_WITH_LAST)

    assert match.signal is Signal.END_TURN_LAST
    assert match.text == "what is the capital of france"


def test_polish_send_last_keyword_without_diacritics():
    match = match_keyword("jaka jest stolica francji wyslij ostatnie", **PL_WITH_LAST)

    assert match.signal is Signal.END_TURN_LAST
    assert match.text == "jaka jest stolica francji"


def test_over_still_ends_turn_when_last_keywords_are_configured():
    match = match_keyword("tell me a joke over", **EN_WITH_LAST)

    assert match.signal is Signal.END_TURN
