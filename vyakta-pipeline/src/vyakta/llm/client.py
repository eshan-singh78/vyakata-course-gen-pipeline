"""Async LLM client with cost tracking and batch support."""

import asyncio
import time
from abc import ABC, abstractmethod

import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from vyakta.config import LLMProvider, Settings, get_settings
from vyakta.models import UsageStats

# Per-provider pricing: $ per 1M tokens (input, output)
_PRICING: dict[str, tuple[float, float]] = {
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-opus-4-7": (15.0, 75.0),
    "claude-haiku-4-5": (0.25, 1.25),
    "gpt-4o": (2.5, 10.0),
    "gpt-4o-mini": (0.15, 0.6),
    "gpt-4-turbo": (10.0, 30.0),
}


class LLMError(Exception):
    """Generic LLM client error."""


_tiktoken_encoder = None


def _get_tiktoken_encoder():
    global _tiktoken_encoder
    if _tiktoken_encoder is None:
        import tiktoken

        _tiktoken_encoder = tiktoken.get_encoding("cl100k_base")
    return _tiktoken_encoder


def count_tokens(text: str, model: str) -> int:
    """Estimate token count for a given text and model."""
    try:
        enc = _get_tiktoken_encoder()
        return len(enc.encode(text))
    except Exception:
        # Fallback: ~4 characters per token
        return len(text) // 4


class Usage:
    """Mutable usage accumulator."""

    def __init__(self) -> None:
        self.tokens_in = 0
        self.tokens_out = 0
        self.cost_usd = 0.0
        self.latency_ms = 0

    def add(self, other: "Usage") -> None:
        self.tokens_in += other.tokens_in
        self.tokens_out += other.tokens_out
        self.cost_usd += other.cost_usd
        self.latency_ms += other.latency_ms

    def to_stats(self) -> UsageStats:
        return UsageStats(
            tokens_in=self.tokens_in,
            tokens_out=self.tokens_out,
            cost_usd=self.cost_usd,
            latency_ms=self.latency_ms,
        )


def _estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    price_in, price_out = _PRICING.get(model, (3.0, 15.0))
    return (tokens_in * price_in + tokens_out * price_out) / 1_000_000


# Approximate context windows (input + output)
_CONTEXT_LIMITS: dict[str, int] = {
    "claude-sonnet-4-6": 200_000,
    "claude-opus-4-7": 200_000,
    "claude-haiku-4-5": 200_000,
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4-turbo": 128_000,
}


_log = structlog.get_logger("vyakta.llm")


def _check_token_limit(prompt: str, model: str, max_output_tokens: int = 4096) -> None:
    """Warn if prompt is likely to exceed the model's context window."""
    limit = _CONTEXT_LIMITS.get(model, 128_000)
    estimated = count_tokens(prompt, model)
    if estimated + max_output_tokens > limit:
        _log.warning(
            "prompt_may_exceed_context_window",
            estimated_tokens=estimated,
            max_output_tokens=max_output_tokens,
            limit=limit,
            model=model,
        )


class LLMClient(ABC):
    """Abstract async LLM client."""

    def __init__(self, settings_obj: Settings):
        self._settings = settings_obj

    @abstractmethod
    async def complete(self, prompt: str, system: str | None = None) -> tuple[str, Usage]:
        """Return (text, usage)."""

    async def complete_batch(
        self, prompts: list[str], system: str | None = None
    ) -> list[tuple[str, Usage]]:
        sem = asyncio.Semaphore(self._settings.max_concurrent_llm_calls)

        async def _bounded(prompt: str) -> tuple[str, Usage]:
            async with sem:
                return await self.complete(prompt, system)

        return await asyncio.gather(*[_bounded(p) for p in prompts])


def _make_retry(max_retries: int):
    return retry(
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((LLMError, ConnectionError, TimeoutError)),
        reraise=True,
    )


class AnthropicClient(LLMClient):
    def __init__(self, settings_obj: Settings):
        super().__init__(settings_obj)
        try:
            import anthropic
        except ImportError as exc:
            raise LLMError("anthropic package is not installed") from exc
        self._client = anthropic.AsyncAnthropic(
            api_key=settings_obj.get_api_key(),
            timeout=settings_obj.timeout_seconds,
        )
        self._complete_with_retry = _make_retry(settings_obj.max_retries)(
            self._complete_raw
        )

    async def complete(self, prompt: str, system: str | None = None) -> tuple[str, Usage]:
        return await self._complete_with_retry(prompt, system)

    async def _complete_raw(self, prompt: str, system: str | None = None) -> tuple[str, Usage]:
        import anthropic

        try:
            _check_token_limit(prompt, self._settings.model_name, max_output_tokens=4096)
            start = time.perf_counter()
            response = await self._client.messages.create(
                model=self._settings.model_name,
                max_tokens=4096,
                temperature=self._settings.temperature,
                system=system or "You are a helpful assistant. Output only valid JSON.",
                messages=[{"role": "user", "content": prompt}],
            )
            latency = int((time.perf_counter() - start) * 1000)
            # Extract text from TextBlock only
            text_parts: list[str] = []
            tokens_out = 0
            for block in response.content:
                if isinstance(block, anthropic.types.TextBlock):
                    text_parts.append(block.text)
                elif hasattr(block, "text"):
                    text_parts.append(block.text)
            text = " ".join(text_parts)
            tokens_in = response.usage.input_tokens if response.usage else 0
            tokens_out = response.usage.output_tokens if response.usage else 0
            cost = (
                _estimate_cost(self._settings.model_name, tokens_in, tokens_out)
                if self._settings.enable_cost_tracking
                else 0.0
            )
            usage = Usage()
            usage.tokens_in = tokens_in
            usage.tokens_out = tokens_out
            usage.cost_usd = cost
            usage.latency_ms = latency
            return text, usage
        except Exception as exc:
            raise LLMError(f"Anthropic API error: {exc}") from exc


class OpenAIClient(LLMClient):
    def __init__(self, settings_obj: Settings):
        super().__init__(settings_obj)
        try:
            import openai
        except ImportError as exc:
            raise LLMError("openai package is not installed") from exc
        self._client = openai.AsyncOpenAI(
            api_key=settings_obj.get_api_key(),
            timeout=settings_obj.timeout_seconds,
        )
        self._complete_with_retry = _make_retry(settings_obj.max_retries)(
            self._complete_raw
        )

    async def complete(self, prompt: str, system: str | None = None) -> tuple[str, Usage]:
        return await self._complete_with_retry(prompt, system)

    async def _complete_raw(self, prompt: str, system: str | None = None) -> tuple[str, Usage]:
        try:
            _check_token_limit(prompt, self._settings.model_name, max_output_tokens=4096)
            start = time.perf_counter()
            response = await self._client.chat.completions.create(
                model=self._settings.model_name,
                temperature=self._settings.temperature,
                messages=[
                    {
                        "role": "system",
                        "content": system or "You are a helpful assistant. Output only valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
            )
            latency = int((time.perf_counter() - start) * 1000)
            content = response.choices[0].message.content
            if content is None:
                raise LLMError("OpenAI returned empty content")
            usage_data = response.usage
            tokens_in = usage_data.prompt_tokens if usage_data else 0
            tokens_out = usage_data.completion_tokens if usage_data else 0
            cost = (
                _estimate_cost(self._settings.model_name, tokens_in, tokens_out)
                if self._settings.enable_cost_tracking
                else 0.0
            )
            usage = Usage()
            usage.tokens_in = tokens_in
            usage.tokens_out = tokens_out
            usage.cost_usd = cost
            usage.latency_ms = latency
            return content, usage
        except Exception as exc:
            raise LLMError(f"OpenAI API error: {exc}") from exc


def get_client(settings_obj: Settings | None = None) -> LLMClient:
    s = settings_obj or get_settings()
    if s.llm_provider == LLMProvider.ANTHROPIC:
        return AnthropicClient(s)
    if s.llm_provider == LLMProvider.OPENAI:
        return OpenAIClient(s)
    raise LLMError(f"Unsupported provider: {s.llm_provider}")
