"""OpenAI TTS client implementation."""

import asyncio
from pathlib import Path

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from vyakta.media.config import MediaSettings
from vyakta.media.voice.base import TTSClient, TTSError


class OpenAITTSClient(TTSClient):
    """OpenAI text-to-speech client."""

    def __init__(self, settings: MediaSettings):
        super().__init__(settings)
        try:
            import openai
        except ImportError as exc:
            raise TTSError("openai package is not installed") from exc
        self._client = openai.AsyncOpenAI(
            api_key=settings.get_tts_api_key(),
            timeout=60,
        )
        self._synthesize_with_retry = _make_retry()(self._synthesize_raw)

    async def synthesize(self, text: str, output_path: Path) -> float:
        await self._synthesize_with_retry(text, output_path)
        return await _probe_duration(output_path, self._settings.ffprobe_path)

    async def _synthesize_raw(self, text: str, output_path: Path) -> None:
        try:
            response = await self._client.audio.speech.create(
                model=self._settings.tts_model,
                voice=self._settings.tts_voice,
                input=text,
                speed=self._settings.tts_speed,
                response_format=self._settings.tts_format,
            )
            import aiofiles.os

            await aiofiles.os.makedirs(output_path.parent, exist_ok=True)
            async with _aiofiles_open(output_path, "wb") as f:
                await f.write(response.content)
        except Exception as exc:
            raise TTSError(f"OpenAI TTS error: {exc}") from exc


def _make_retry():
    return retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((TTSError, ConnectionError, TimeoutError)),
        reraise=True,
    )


async def _probe_duration(path: Path, ffprobe: str) -> float:
    """Use ffprobe to get audio duration."""
    proc = await asyncio.create_subprocess_exec(
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise TTSError(f"ffprobe failed: {stderr.decode().strip()}")
    try:
        return float(stdout.decode().strip())
    except ValueError as exc:
        raise TTSError(f"ffprobe returned invalid duration: {stdout.decode().strip()}") from exc


# Lazy import to avoid hard dependency on aiofiles at import time
def _aiofiles_open(path, mode):
    import aiofiles

    return aiofiles.open(path, mode)
