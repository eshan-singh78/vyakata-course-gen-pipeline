"""Abstract base class for pipeline stages."""

import json
from abc import ABC, abstractmethod
from typing import TypeVar

import structlog
from pydantic import BaseModel, ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from vyakta.config import Settings
from vyakta.llm.client import LLMClient, Usage, get_client
from vyakta.models import StageResult

logger = structlog.get_logger()

InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)


class StageError(Exception):
    """Raised when a stage fails irrecoverably."""


class Stage(ABC):
    """One async pipeline stage."""

    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None):
        self.client = client or get_client(settings)
        self.settings = settings
        self._log = logger.bind(stage=self.__class__.__name__)

    @property
    @abstractmethod
    def prompt_template(self) -> str:
        """System/instruction prompt for the LLM."""

    @property
    @abstractmethod
    def output_model(self) -> type[OutputT]:
        """Pydantic model used to validate stage output."""

    @abstractmethod
    def build_prompt(self, data: InputT) -> str:
        """Combine the template with input data into a single user prompt."""

    async def run(self, data: InputT) -> StageResult[OutputT]:
        """Execute the stage end-to-end."""
        raw_prompt = self.build_prompt(data)
        validated_dict, usage = await self._call_llm(raw_prompt)
        return StageResult(
            output=self.output_model.model_validate(validated_dict),
            usage=usage.to_stats(),
        )

    async def _complete_raw(self, prompt: str) -> tuple[str, Usage]:
        return await self.client.complete(prompt, system=self.prompt_template)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((StageError, ValidationError)),
        reraise=True,
    )
    async def _call_llm(self, prompt: str) -> tuple[dict, Usage]:
        raw_response, usage = await self._complete_raw(prompt)
        parsed = self._parse_json(raw_response)
        validated = self._validate(parsed)
        return validated.model_dump(), usage

    def _parse_json(self, raw: str) -> dict:
        raw = raw.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            import re

            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        # json_repair fallback
        try:
            import json_repair

            repaired = json_repair.loads(raw)
            if isinstance(repaired, dict):
                return repaired
            if isinstance(repaired, list):
                return {"__batch__": repaired}
        except Exception:
            pass
        raise StageError("Failed to parse LLM output as JSON")

    def _validate(self, data: dict) -> OutputT:
        try:
            return self.output_model.model_validate(data)
        except ValidationError as exc:
            raise StageError(f"Output validation failed: {exc}") from exc

    @staticmethod
    def wrap_content(tag: str, content: str) -> str:
        """Wrap user content in unbreakable delimiters to mitigate prompt injection."""
        start = f"<<<START_{tag}>>>"
        end = f"<<<END_{tag}>>>"
        # Escape any delimiter-like sequences inside content
        safe = content.replace(start, f"[ESCAPED_{tag}_START]").replace(end, f"[ESCAPED_{tag}_END]")
        return f"{start}\n{safe}\n{end}"
