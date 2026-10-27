import gc
import weakref

from castor import gemini_chat
from castor.gemini_chat import GeminiChat


class FakeClient:
    """Stands in for genai.Client, which closes its HTTP connection when garbage-collected."""

    last = None

    def __init__(self, **kwargs):
        self.chats = self
        FakeClient.last = weakref.ref(self)

    def create(self, **kwargs):
        return object()


def test_client_is_not_garbage_collected_while_chat_is_alive(monkeypatch):
    monkeypatch.setattr(gemini_chat.genai, "Client", FakeClient)

    chat = GeminiChat(api_key="key", model="model", system_instruction="be brief")
    gc.collect()

    assert FakeClient.last() is not None
    assert chat is not None
