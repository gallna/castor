"""Turns Gemini's (possibly markdown) reply into plain text suitable for TTS."""

import re

_CODE_BLOCK = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE = re.compile(r"`([^`]*)`")
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_LINE_MARKER = re.compile(r"^\s*(?:#+|[-*+]|\d+[.)])\s+", re.MULTILINE)
# Single "_" is left alone so identifiers like snake_case survive.
_EMPHASIS = re.compile(r"(\*\*|__|\*)(\S.*?\S|\S)\1")
_WHITESPACE = re.compile(r"\s+")


def to_speakable(text: str) -> str:
    text = _CODE_BLOCK.sub(" ", text)
    text = _INLINE_CODE.sub(r"\1", text)
    text = _LINK.sub(r"\1", text)
    text = _LINE_MARKER.sub("", text)
    text = _EMPHASIS.sub(r"\2", text)
    return _WHITESPACE.sub(" ", text).strip()
