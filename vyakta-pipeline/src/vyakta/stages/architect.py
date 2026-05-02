"""Stage 2: Curriculum Architect."""

import json

from vyakta.models import CourseStructure, NormalizedContent, StageResult
from vyakta.prompts import ARCHITECT_PROMPT
from vyakta.stages.base import Stage


class ArchitectStage(Stage):
    """Turns normalized content into a course structure."""

    prompt_template = ARCHITECT_PROMPT
    output_model = CourseStructure

    def build_prompt(self, data: NormalizedContent) -> str:
        payload = json.dumps(data.model_dump(), ensure_ascii=False)
        wrapped = self.wrap_content("NORMALIZED_DATA", payload)
        return f"{self.prompt_template}\n\n{wrapped}"

    async def run(self, data: NormalizedContent) -> StageResult[CourseStructure]:
        raw_prompt = self.build_prompt(data)
        parsed, usage = await self._call_llm(raw_prompt)
        validated = self._validate(parsed)
        return StageResult(output=validated, usage=usage.to_stats())
