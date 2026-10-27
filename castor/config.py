"""Per-language settings: speech models, keywords and Gemini instructions."""

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
PROMPTS_DIR = ROOT / "prompts"
WHISPER_DIR = MODELS_DIR / "whisper"  # downloaded here on first use

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_WHISPER_MODEL = "small"  # multilingual; "base" is faster, "medium" recognises Polish better
DEFAULT_PLAYER = "aplay -q"


@dataclass(frozen=True)
class LanguageProfile:
    code: str
    piper_voice: Path
    turn_keywords: tuple[str, ...]
    last_keywords: tuple[str, ...]  # like turn_keywords, but send only the last sentence
    end_keywords: tuple[str, ...]
    system_instruction: str
    initial_prompt_file: Path


LANGUAGES = {
    "en": LanguageProfile(
        code="en",
        piper_voice=MODELS_DIR / "piper" / "en_US-lessac-medium.onnx",
        turn_keywords=("over",),
        last_keywords=("send last",),
        end_keywords=("over and out",),
        system_instruction=(
            "You are talking with the user by voice. Your replies are read aloud by a "
            "text-to-speech engine, so reply in English using short, plain spoken "
            "sentences: no markdown, lists, emoji or code. Keep replies under about "
            "80 words unless asked for more. The user's words come from speech "
            "recognition and may contain mistakes; infer what they meant."
        ),
        initial_prompt_file=PROMPTS_DIR / "initial_prompt_en.txt",
    ),
    "pl": LanguageProfile(
        code="pl",
        piper_voice=MODELS_DIR / "piper" / "pl_PL-gosia-medium.onnx",
        turn_keywords=("odbiór",),
        last_keywords=("wyślij ostatnie",),
        end_keywords=("bez odbioru",),
        system_instruction=(
            "Rozmawiasz z użytkownikiem głosowo. Twoje odpowiedzi są czytane na głos "
            "przez syntezator mowy, więc odpowiadaj po polsku krótkimi, prostymi "
            "zdaniami: bez formatowania markdown, list, emoji i kodu. Mieść się w "
            "około 80 słowach, chyba że użytkownik poprosi o więcej. Wypowiedzi "
            "użytkownika pochodzą z rozpoznawania mowy i mogą zawierać błędy; "
            "domyślaj się, co miał na myśli."
        ),
        initial_prompt_file=PROMPTS_DIR / "initial_prompt_pl.txt",
    ),
}
