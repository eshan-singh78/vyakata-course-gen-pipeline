"""Stage 4: Script Generator — batched, parallel, with partial checkpointing."""

import asyncio
import json
from pathlib import Path
from typing import Any

import aiofiles
from pydantic import ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from vyakta.llm.client import Usage
from vyakta.models import ScriptOutput, StageResult, VideoPlan
from vyakta.prompts import SCRIPTOR_BATCH_PROMPT
from vyakta.stages.base import Stage, StageError


class ScriptorStage(Stage):
    """Generates TTS-optimized narration scripts in parallel batches."""

    prompt_template = SCRIPTOR_BATCH_PROMPT
    output_model = ScriptOutput

    def build_prompt(self, data: VideoPlan) -> str:
        payload = json.dumps(data.model_dump(), ensure_ascii=False)
        wrapped = self.wrap_content("VIDEO_PLAN", payload)
        return f"{self.prompt_template}\n\n{wrapped}"

    def _build_batch_prompt(self, videos: list[dict[str, Any]], chapter_title: str) -> str:
        batch_plan = VideoPlan(
            chapters=[{"chapter_title": chapter_title, "videos": videos}]
        )
        return self.build_prompt(batch_plan)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((StageError, ValidationError)),
        reraise=True,
    )
    async def _process_one_batch(
        self, prompt: str, batch_idx: int
    ) -> tuple[list[ScriptOutput], Usage]:
        """Process a single batch with its own retry logic."""
        raw_response, usage = await self._complete_raw(prompt)
        parsed = self._parse_json(raw_response)

        if isinstance(parsed, dict) and "__batch__" in parsed:
            items = parsed["__batch__"]
        elif isinstance(parsed, list):
            items = parsed
        elif isinstance(parsed, dict):
            items = [parsed]
        else:
            raise StageError(f"Unexpected batch JSON type: {type(parsed)}")

        scripts: list[ScriptOutput] = []
        for item in items:
            script = ScriptOutput.model_validate(item)
            scripts.append(script)
        return scripts, usage

    async def run(self, data: VideoPlan) -> StageResult[list[ScriptOutput]]:
        all_scripts: list[ScriptOutput] = []
        total_usage = Usage()

        if not data.chapters:
            return StageResult(output=[], usage=total_usage.to_stats())

        batch_size = self.settings.batch_size if self.settings else 5

        # Build per-chapter batches so prompts never cross chapter boundaries
        prompts: list[tuple[int, str]] = []  # (global_batch_idx, prompt)
        batch_idx = 0
        for chapter in data.chapters:
            videos = [v.model_dump() for v in chapter.videos]
            if not videos:
                continue
            for i in range(0, len(videos), batch_size):
                chunk = videos[i : i + batch_size]
                prompt = self._build_batch_prompt(chunk, chapter.chapter_title)
                prompts.append((batch_idx, prompt))
                batch_idx += 1

        self._log.info(
            "scriptor_start",
            total_chapters=len(data.chapters),
            batch_size=batch_size,
            num_batches=len(prompts),
        )

        # Execute all batches in parallel with concurrency limit
        sem = asyncio.Semaphore(
            self.settings.max_concurrent_llm_calls if self.settings else 5
        )

        _BatchOk = tuple[int, list[ScriptOutput], Usage]
        _BatchErr = tuple[int, Exception]

        async def _bounded(idx: int, prompt: str) -> _BatchOk | _BatchErr:
            async with sem:
                try:
                    scripts, usage = await self._process_one_batch(prompt, idx)
                    return idx, scripts, usage
                except Exception as exc:
                    return idx, exc

        tasks = [
            asyncio.create_task(_bounded(idx, p)) for idx, p in prompts
        ]
        results = await asyncio.gather(*tasks)

        # Partial checkpoint path
        partial_path: Path | None = None
        if self.settings:
            partial_path = self.settings.output_dir / "stage4_partial.json"

        for res in results:
            if len(res) == 2:  # Exception case
                idx, exc = res
                self._log.error("batch_failed", batch=idx, error=str(exc))
                continue

            idx, scripts, usage = res
            all_scripts.extend(scripts)
            total_usage.add(usage)
            self._log.info(
                "batch_complete",
                batch=idx,
                scripts_in_batch=len(scripts),
                total_so_far=len(all_scripts),
            )

            # Save partial checkpoint after each successful batch
            if partial_path:
                partial_path.parent.mkdir(parents=True, exist_ok=True)
                temp_path = partial_path.with_suffix(".tmp")
                async with aiofiles.open(temp_path, "w", encoding="utf-8") as f:
                    await f.write(
                        json.dumps(
                            {"scripts": [s.model_dump() for s in all_scripts]},
                            indent=2,
                        )
                    )
                temp_path.rename(partial_path)

        self._log.info(
            "scriptor_complete",
            scripts_generated=len(all_scripts),
            total_tokens_in=total_usage.tokens_in,
            total_tokens_out=total_usage.tokens_out,
            total_cost=round(total_usage.cost_usd, 4),
            failed_batches=sum(1 for r in results if len(r) == 2),
        )
        return StageResult(output=all_scripts, usage=total_usage.to_stats())
