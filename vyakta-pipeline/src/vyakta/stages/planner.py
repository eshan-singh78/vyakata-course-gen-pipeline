"""Stage 3: Chapter & Video Planner."""

import json

from vyakta.models import CourseStructure, StageResult, VideoPlan
from vyakta.prompts import PLANNER_PROMPT
from vyakta.stages.base import Stage


class PlannerStage(Stage):
    """Breaks chapters into 2–5 minute video units."""

    prompt_template = PLANNER_PROMPT
    output_model = VideoPlan

    def build_prompt(self, data: CourseStructure) -> str:
        payload = json.dumps(data.model_dump(), ensure_ascii=False)
        wrapped = self.wrap_content("COURSE_STRUCTURE", payload)
        return f"{self.prompt_template}\n\n{wrapped}"

    async def run(self, data: CourseStructure) -> StageResult[VideoPlan]:
        raw_prompt = self.build_prompt(data)
        parsed, usage = await self._call_llm(raw_prompt)
        validated = self._validate(parsed)
        return StageResult(output=validated, usage=usage.to_stats())
