"""TTS client factory."""

from vyakta.media.config import MediaSettings, TTSProvider
from vyakta.media.voice.base import TTSClient, TTSError
from vyakta.media.voice.kyutai import KyutaiTTSClient
from vyakta.media.voice.openai import OpenAITTSClient


def get_tts_client(settings: MediaSettings | None = None) -> TTSClient:
    s = settings or MediaSettings()
    if s.tts_provider == TTSProvider.OPENAI:
        return OpenAITTSClient(s)
    if s.tts_provider == TTSProvider.KYUTAI:
        return KyutaiTTSClient(s)
    raise TTSError(f"Unsupported TTS provider: {s.tts_provider}")
