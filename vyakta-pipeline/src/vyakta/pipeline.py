"""Async pipeline orchestrator with fixed resume logic and cost tracking."""

import json
from datetime import UTC, datetime
from pathlib import Path

import aiofiles
from pydantic import BaseModel

from vyakta.config import Settings, get_settings
from vyakta.llm.client import LLMClient, Usage, get_client
from vyakta.models import (
    CourseStructure,
    FinalScripts,
    NormalizedContent,
    PipelineRun,
    ScriptOutput,
    StageResult,
    VideoPlan,
)
from vyakta.stages.architect import ArchitectStage
from vyakta.stages.base import StageError
from vyakta.stages.normalizer import NormalizerStage
from vyakta.stages.planner import PlannerStage
from vyakta.stages.scriptor import ScriptorStage

_CHECKPOINTS = {
    "stage1": ("stage1_normalizer.json", NormalizedContent),
    "stage2": ("stage2_architect.json", CourseStructure),
    "stage3": ("stage3_planner.json", VideoPlan),
    "stage4": ("stage4_scripts.json", FinalScripts),
}

_RESUME_ORDER = ["stage1", "stage2", "stage3", "stage4"]


class Pipeline:
    """Runs the 4-stage async pipeline with proper resume semantics."""

    def __init__(
        self,
        client: LLMClient | None = None,
        settings_obj: Settings | None = None,
    ):
        self.client = client or get_client(settings_obj)
        self.settings = settings_obj or get_settings()
        self.output_dir = self.settings.output_dir

        self.normalizer = NormalizerStage(self.client, self.settings)
        self.architect = ArchitectStage(self.client, self.settings)
        self.planner = PlannerStage(self.client, self.settings)
        self.scriptor = ScriptorStage(self.client, self.settings)

        self.run_meta = PipelineRun()

    async def _ensure_output_dir(self) -> None:
        import aiofiles.os

        await aiofiles.os.makedirs(self.output_dir, exist_ok=True)

    async def run(
        self,
        raw_input: str,
        resume_from: str | None = None,
    ) -> tuple[FinalScripts, PipelineRun]:
        """Execute full pipeline or resume from a checkpoint.

        Resume semantics:
        - If resume_from is None: run all stages from scratch.
        - If resume_from is "stageN": stageN checkpoint is already done.
          Continue from stageN+1.
        - If requested checkpoint is missing: raise clear error, do NOT skip.
        """
        if resume_from is not None and resume_from not in _RESUME_ORDER:
            raise StageError(
                f"Invalid resume point: '{resume_from}'. "
                f"Valid options: {', '.join(_RESUME_ORDER)}"
            )

        await self._ensure_output_dir()

        stage1_data: NormalizedContent | None = None
        stage2_data: CourseStructure | None = None
        stage3_data: VideoPlan | None = None
        stage4_data: list[ScriptOutput] | None = None
        total_usage = Usage()

        # Determine resume point index (-1 = run everything)
        resume_idx = _RESUME_ORDER.index(resume_from) if resume_from else -1

        # Populate stages_completed for previously finished stages when resuming
        if resume_from:
            self.run_meta.stages_completed = list(_RESUME_ORDER[: resume_idx + 1])

        # Load checkpoint at resume point (if any)
        if resume_from:
            loaded = await self._load_checkpoint(resume_from)
            if loaded is None:
                raise StageError(
                    f"Resume requested from '{resume_from}' but checkpoint "
                    f"'{resume_from}' is missing. Cannot resume."
                )
            if resume_from == "stage1":
                stage1_data = loaded
            elif resume_from == "stage2":
                stage2_data = loaded
            elif resume_from == "stage3":
                stage3_data = loaded
            elif resume_from == "stage4":
                stage4_data = loaded.scripts if hasattr(loaded, "scripts") else loaded

        # Stage 1 (skip if resuming from stage1 or later)
        if resume_idx < 0 and stage1_data is None:
            result: StageResult = await self.normalizer.run(raw_input)
            stage1_data = result.output
            total_usage.add(self._usage_from_stats(result.usage))
            await self._save_checkpoint("stage1", stage1_data)
            self.run_meta.stages_completed.append("stage1")

        # Stage 2 (skip if resuming from stage2 or later)
        if resume_idx < 1 and stage2_data is None:
            result = await self.architect.run(stage1_data)
            stage2_data = result.output
            total_usage.add(self._usage_from_stats(result.usage))
            await self._save_checkpoint("stage2", stage2_data)
            self.run_meta.stages_completed.append("stage2")

        # Stage 3 (skip if resuming from stage3 or later)
        if resume_idx < 2 and stage3_data is None:
            result = await self.planner.run(stage2_data)
            stage3_data = result.output
            total_usage.add(self._usage_from_stats(result.usage))
            await self._save_checkpoint("stage3", stage3_data)
            self.run_meta.stages_completed.append("stage3")

        # Stage 4 (skip if resuming from stage4)
        if resume_idx < 3 and stage4_data is None:
            result = await self.scriptor.run(stage3_data)
            stage4_data = result.output
            total_usage.add(self._usage_from_stats(result.usage))
            final = FinalScripts(scripts=stage4_data)
            await self._save_checkpoint("stage4", final)
            self.run_meta.stages_completed.append("stage4")
            self.run_meta.total_usage = total_usage.to_stats()
            self.run_meta.completed_at = datetime.now(UTC)
            return final, self.run_meta

        # Resumed to completion
        if isinstance(stage4_data, FinalScripts):
            final = stage4_data
        else:
            final = FinalScripts(scripts=stage4_data)
        self.run_meta.total_usage = total_usage.to_stats()
        self.run_meta.completed_at = datetime.now(UTC)
        return final, self.run_meta

    def _checkpoint_path(self, name: str) -> Path:
        filename, _ = _CHECKPOINTS[name]
        return self.output_dir / filename

    async def _save_checkpoint(self, name: str, data: BaseModel) -> None:
        path = self._checkpoint_path(name)
        payload = json.dumps(data.model_dump(), indent=2, default=str)
        temp = path.with_suffix(".tmp")
        async with aiofiles.open(temp, "w", encoding="utf-8") as f:
            await f.write(payload)
        temp.rename(path)
        self.run_meta.checkpoints.append(str(path))

    async def _load_checkpoint(self, name: str):
        path = self._checkpoint_path(name)
        if not path.exists():
            return None
        try:
            async with aiofiles.open(path, "r", encoding="utf-8") as f:
                content = await f.read()
            raw = json.loads(content)
        except json.JSONDecodeError as exc:
            raise StageError(
                f"Checkpoint file {path} is corrupted (invalid JSON). "
                f"Delete it and restart."
            ) from exc
        _, model_cls = _CHECKPOINTS[name]
        try:
            return model_cls.model_validate(raw)
        except Exception as exc:
            raise StageError(
                f"Checkpoint file {path} contains invalid data. "
                f"Delete it and restart."
            ) from exc

    @staticmethod
    def _usage_from_stats(stats) -> Usage:
        u = Usage()
        u.tokens_in = stats.tokens_in
        u.tokens_out = stats.tokens_out
        u.cost_usd = stats.cost_usd
        u.latency_ms = stats.latency_ms
        return u
