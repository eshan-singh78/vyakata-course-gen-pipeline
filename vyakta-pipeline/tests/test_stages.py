"""Unit tests for pipeline stages with mocked async LLM client."""

import json

import pytest

from vyakta.llm.client import LLMClient, Usage
from vyakta.models import (
    CourseStructure,
    NormalizedContent,
    VideoPlan,
)
from vyakta.stages.architect import ArchitectStage
from vyakta.stages.base import StageError
from vyakta.stages.normalizer import NormalizerStage
from vyakta.stages.planner import PlannerStage
from vyakta.stages.scriptor import ScriptorStage


class FakeLLMClient(LLMClient):
    """Test double that returns canned responses."""

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


class TestNormalizerStage:
    @pytest.mark.asyncio
    async def test_success(self):
        fake = FakeLLMClient(
            responses=[
                json.dumps(
                    {
                        "title": "Clean Title",
                        "sections": [{"heading": "Sec1", "content": "Text"}],
                    }
                )
            ]
        )
        stage = NormalizerStage(client=fake)
        result = await stage.run("<html><body><p>Text</p></body></html>")
        assert isinstance(result.output, NormalizedContent)
        assert result.output.title == "Clean Title"
        assert result.usage.tokens_in == 10

    @pytest.mark.asyncio
    async def test_html_parsing(self):
        fake = FakeLLMClient(
            responses=[
                json.dumps(
                    {
                        "title": "Parsed",
                        "sections": [{"heading": "H", "content": "C"}],
                    }
                )
            ]
        )
        stage = NormalizerStage(client=fake)
        html = "<html><script>alert(1)</script><p>Keep</p></html>"
        result = await stage.run(html)
        assert result.output.title == "Parsed"

    @pytest.mark.asyncio
    async def test_json_decode_error(self):
        fake = FakeLLMClient(responses=["not json"])
        stage = NormalizerStage(client=fake)
        with pytest.raises(StageError):
            await stage.run("<html><body><p>Hello</p></body></html>")


class TestArchitectStage:
    @pytest.mark.asyncio
    async def test_success(self):
        fake = FakeLLMClient(
            responses=[
                json.dumps(
                    {
                        "course_title": "Web Sec",
                        "modules": [
                            {
                                "module_title": "M1",
                                "chapters": [
                                    {
                                        "chapter_title": "C1",
                                        "source_sections": ["s1"],
                                    }
                                ],
                            }
                        ],
                    }
                )
            ]
        )
        stage = ArchitectStage(client=fake)
        inp = NormalizedContent(
            title="T", sections=[{"heading": "H", "content": "C"}]
        )
        result = await stage.run(inp)
        assert isinstance(result.output, CourseStructure)
        assert result.output.course_title == "Web Sec"


class TestPlannerStage:
    @pytest.mark.asyncio
    async def test_success(self):
        fake = FakeLLMClient(
            responses=[
                json.dumps(
                    {
                        "chapters": [
                            {
                                "chapter_title": "C1",
                                "videos": [
                                    {
                                        "video_title": "V1",
                                        "concept_focus": "F",
                                    }
                                ],
                            }
                        ]
                    }
                )
            ]
        )
        stage = PlannerStage(client=fake)
        inp = CourseStructure(
            course_title="T",
            modules=[
                {
                    "module_title": "M",
                    "chapters": [
                        {"chapter_title": "C", "source_sections": ["s1"]}
                    ],
                }
            ],
        )
        result = await stage.run(inp)
        assert isinstance(result.output, VideoPlan)
        assert len(result.output.chapters[0].videos) == 1


class TestScriptorStage:
    @pytest.mark.asyncio
    async def test_single_video(self):
        fake = FakeLLMClient(
            responses=[
                json.dumps(
                    [
                        {
                            "video_title": "V1",
                            "script_chunks": ["Hello", "World"],
                        }
                    ]
                )
            ]
        )
        stage = ScriptorStage(client=fake)
        inp = VideoPlan(
            chapters=[
                {
                    "chapter_title": "C1",
                    "videos": [
                        {
                            "video_title": "V1",
                            "concept_focus": "F",
                        }
                    ],
                }
            ]
        )
        result = await stage.run(inp)
        assert isinstance(result.output, list)
        assert len(result.output) == 1
        assert result.output[0].video_title == "V1"

    @pytest.mark.asyncio
    async def test_batch_of_three(self):
        fake = FakeLLMClient(
            responses=[
                json.dumps(
                    [
                        {
                            "video_title": "V1",
                            "script_chunks": ["A"],
                        },
                        {
                            "video_title": "V2",
                            "script_chunks": ["B"],
                        },
                        {
                            "video_title": "V3",
                            "script_chunks": ["C"],
                        },
                    ]
                )
            ]
        )
        stage = ScriptorStage(client=fake)
        inp = VideoPlan(
            chapters=[
                {
                    "chapter_title": "C1",
                    "videos": [
                        {"video_title": "V1", "concept_focus": "F1"},
                        {"video_title": "V2", "concept_focus": "F2"},
                        {"video_title": "V3", "concept_focus": "F3"},
                    ],
                }
            ]
        )
        result = await stage.run(inp)
        assert len(result.output) == 3
        assert fake.call_count == 1  # batched into 1 API call
