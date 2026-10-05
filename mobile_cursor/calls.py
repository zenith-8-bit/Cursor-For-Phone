from .ollama import OllamaClient

class CallConversationController:
    """
    Logic layer for a future phone-call conversation.

    The Android InCallService owns call state/control.
    A local STT service supplies transcribed caller speech.
    Qwen generates the response.
    A local TTS service turns the response into audio.

    Actual injection of generated audio into a cellular call is device/
    Telecom-role dependent and must be implemented on the Android side
    for the target device.
    """

    def __init__(self, ollama=None):
        self.ollama = ollama or OllamaClient()

    def respond(self, transcript: str, context: str = "") -> str:
        system = (
            "You are a phone-call conversation controller. "
            "Reply naturally and briefly. Never invent private information. "
            "If you need information, ask for it."
        )
        user = f"Context:\n{context}\n\nCaller said:\n{transcript}"
        data = self.ollama.chat_json(system, user)
        return data.get("response", data.get("message", ""))
