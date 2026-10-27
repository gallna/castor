"""A single Gemini chat session; the SDK's chat object keeps the full history."""

import httpx
from google import genai
from google.genai import errors, types

RETRY_ATTEMPTS = 3
REQUEST_TIMEOUT_MS = 60_000


class ChatError(RuntimeError):
    pass


class GeminiChat:
    def __init__(self, api_key: str, model: str, system_instruction: str) -> None:
        # Keep a reference: genai.Client closes its HTTP connection when garbage-collected,
        # which would break every later send_message() on the chat.
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=REQUEST_TIMEOUT_MS,
                retry_options=types.HttpRetryOptions(attempts=RETRY_ATTEMPTS),
            ),
        )
        self._chat = self._client.chats.create(
            model=model,
            config=types.GenerateContentConfig(system_instruction=system_instruction),
        )

    def send(self, text: str) -> str:
        try:
            response = self._chat.send_message(text)
        except (errors.APIError, httpx.HTTPError) as exc:
            raise ChatError(f"Gemini request failed: {exc}") from exc
        reply = (response.text or "").strip()
        if not reply:
            raise ChatError("Gemini returned an empty reply (possibly blocked by safety filters).")
        return reply
