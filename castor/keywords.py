"""Detects the spoken keywords that end a turn or the whole conversation."""

import re
import unicodedata
from dataclasses import dataclass
from enum import Enum


class Signal(Enum):
    NONE = "none"
    END_TURN = "end_turn"
    END_TURN_LAST = "end_turn_last"  # end the turn, sending only the last sentence
    END_CONVERSATION = "end_conversation"


@dataclass(frozen=True)
class KeywordMatch:
    signal: Signal
    text: str  # the phrase with the keyword removed


def _normalize(word: str) -> str:
    """Lowercase, drop diacritics and punctuation, so "Odbiór." matches "odbior"."""
    decomposed = unicodedata.normalize("NFKD", word.lower())
    without_accents = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"\W", "", without_accents)


def _ends_with(words: list[str], keyword: str) -> bool:
    keyword_words = [_normalize(w) for w in keyword.split()]
    if len(words) < len(keyword_words):
        return False
    tail = [_normalize(w) for w in words[-len(keyword_words):]]
    return tail == keyword_words


def match_keyword(
    phrase: str,
    turn_keywords: tuple[str, ...],
    end_keywords: tuple[str, ...],
    last_keywords: tuple[str, ...] = (),
) -> KeywordMatch:
    """Check whether a recognised phrase ends with a keyword.

    Keywords only count at the very end of a phrase (i.e. right before a pause),
    so "move it over there" is ordinary speech. End keywords are checked first
    because "over and out" contains "over".
    """
    words = phrase.split()
    for signal, keywords in (
        (Signal.END_CONVERSATION, end_keywords),
        (Signal.END_TURN_LAST, last_keywords),
        (Signal.END_TURN, turn_keywords),
    ):
        for keyword in keywords:
            if _ends_with(words, keyword):
                remaining = words[: len(words) - len(keyword.split())]
                return KeywordMatch(signal, " ".join(remaining))
    return KeywordMatch(Signal.NONE, " ".join(words))
