"""Unit tests for pipeline orchestrator and resume logic."""

import json
from pathlib import Path

import pytest

from vyakta.llm.client import LLMClient, Usage
from vyakta.models import (
    CourseStructure,
    FinalScripts,
    NormalizedContent,
    VideoPlan,
)
from vyakta.pipeline import Pipeline
from vyakta.stages.architect import ArchitectStage
from vyakta.stages.base import StageError
from vyakta.stages.normalizer import NormalizerStage
from vyakta.stages.planner import PlannerStage
from vyakta.stages.scriptor import ScriptorStage


class FakeLLMClient(LLMClient):
    """Test double for pipeline-level tests."""

    def __init__(self, responses=None):
        from vyakta.config import Settings

        super().__init__(Settings())
        self.responses = responses or []
        self.call_count = 0

    async def complete(self, prompt: str, system: str | None = None) -> tuple[str, Usage]:
        if self.call_count < len(self.responses):
            resp = self.responses[self.call_count]
        else:
            resp = self.responses[-1] if self.responses else "not json"
        self.call_count += 1
        usage = Usage()
        usage.tokens_in = 10
        usage.tokens_out = 20
        usage.cost_usd = 0.0001
        return resp, usage


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    return tmp_path / "output"


@pytest.fixture
def full_responses():
    return [
        # stage1
        json.dumps({"title": "T", "sections": [{"heading": "H", "content": "C"}]}),
        # stage2
        json.dumps(
            {
                "course_title": "Web",
                "modules": [
                    {
                        "module_title": "M",
                        "chapters": [
                            {"chapter_title": "C", "source_sections": ["s1"]}
                        ],
                    }
                ],
            }
        ),
        # stage3
        json.dumps(
            {
                "chapters": [
                    {
                        "chapter_title": "C",
                        "videos": [
                            {"video_title": "V1", "concept_focus": "F"}
                        ],
                    }
                ]
            }
        ),
        # stage4 batch
        json.dumps(
            [{"video_title": "V1", "script_chunks": ["Hello"]}]
        ),
    ]


@pytest.mark.asyncio
async def test_full_pipeline(tmp_output_dir: Path, full_responses):
    from vyakta.config import Settings

    cfg = Settings(output_dir=tmp_output_dir)
    client = FakeLLMClient(responses=full_responses)
    pipeline = Pipeline(client=client, settings_obj=cfg)
    final, meta = await pipeline.run("<html><body><p>Hello world</p></body></html>")
    assert len(final.scripts) == 1
    assert meta.stages_completed == ["stage1", "stage2", "stage3", "stage4"]
    assert meta.total_usage.tokens_in == 40


@pytest.mark.asyncio
async def test_resume_from_stage2(tmp_output_dir: Path, full_responses):
    from vyakta.config import Settings

    cfg = Settings(output_dir=tmp_output_dir)
    client = FakeLLMClient(responses=full_responses)
    pipeline = Pipeline(client=client, settings_obj=cfg)

    # Run first to create checkpoints
    await pipeline.run("<html><body><p>Hello world</p></body></html>")
    assert client.call_count == 4

    # New pipeline, resume from stage2
    client2 = FakeLLMClient(responses=full_responses[2:])
    pipeline2 = Pipeline(client=client2, settings_obj=cfg)
    final, meta = await pipeline2.run(
        "<html><body><p>Hello world</p></body></html>", resume_from="stage2"
    )
    assert len(final.scripts) == 1
    assert meta.stages_completed == ["stage1", "stage2", "stage3", "stage4"]
    assert client2.call_count == 2


@pytest.mark.asyncio
async def test_resume_missing_checkpoint(tmp_output_dir: Path):
    from vyakta.config import Settings

    cfg = Settings(output_dir=tmp_output_dir)
    client = FakeLLMClient(responses=[])
    pipeline = Pipeline(client=client, settings_obj=cfg)
    with pytest.raises(StageError, match="checkpoint 'stage2' is missing"):
        await pipeline.run("<html><body><p>X</p></body></html>", resume_from="stage2")


@pytest.mark.asyncio
async def test_cli_command_chaining(tmp_output_dir: Path, full_responses):
    """Verify that normalize output can be fed into architect, etc."""
    from vyakta.config import Settings

    cfg = Settings(output_dir=tmp_output_dir)
    client = FakeLLMClient(responses=full_responses)

    # Stage 1
    stage1 = NormalizerStage(client, cfg)
    result1 = await stage1.run("<html><body><p>Test</p></body></html>")
    # Verify it serializes to raw NormalizedContent (not StageResult envelope)
    raw1 = result1.output.model_dump_json()
    parsed1 = NormalizedContent.model_validate_json(raw1)
    assert parsed1.title == "T"

    # Stage 2
    stage2 = ArchitectStage(client, cfg)
    result2 = await stage2.run(parsed1)
    raw2 = result2.output.model_dump_json()
    parsed2 = CourseStructure.model_validate_json(raw2)
    assert parsed2.course_title == "Web"

    # Stage 3
    stage3 = PlannerStage(client, cfg)
    result3 = await stage3.run(parsed2)
    raw3 = result3.output.model_dump_json()
    parsed3 = VideoPlan.model_validate_json(raw3)
    assert len(parsed3.chapters[0].videos) == 1

    # Stage 4
    stage4 = ScriptorStage(client, cfg)
    result4 = await stage4.run(parsed3)
    raw4 = json.dumps({"scripts": [s.model_dump() for s in result4.output]})
    parsed4 = FinalScripts.model_validate_json(raw4)
    assert len(parsed4.scripts) == 1
