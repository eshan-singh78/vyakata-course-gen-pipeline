"""Unit tests for Pydantic models."""

import pytest
from pydantic import ValidationError

from vyakta.models import (
    CourseStructure,
    FinalScripts,
    NormalizedContent,
    PipelineRun,
    ScriptOutput,
    StageResult,
    UsageStats,
    VideoPlan,
)


class TestNormalizedContent:
    def test_valid(self):
        data = {
            "title": "Test",
            "sections": [{"heading": "H1", "content": "Body"}],
        }
        obj = NormalizedContent.model_validate(data)
        assert obj.title == "Test"
        assert len(obj.sections) == 1

    def test_missing_title(self):
        with pytest.raises(ValidationError):
            NormalizedContent.model_validate({"sections": []})


class TestCourseStructure:
    def test_valid(self):
        data = {
            "course_title": "Web Sec",
            "modules": [
                {
                    "module_title": "Basics",
                    "chapters": [
                        {"chapter_title": "OWASP", "source_sections": ["s1"]}
                    ],
                }
            ],
        }
        obj = CourseStructure.model_validate(data)
        assert obj.course_title == "Web Sec"
        assert len(obj.modules[0].chapters) == 1


class TestVideoPlan:
    def test_valid(self):
        data = {
            "chapters": [
                {
                    "chapter_title": "SQLi",
                    "videos": [
                        {"video_title": "What is SQLi", "concept_focus": "Injection"}
                    ],
                }
            ]
        }
        obj = VideoPlan.model_validate(data)
        assert obj.chapters[0].videos[0].concept_focus == "Injection"


class TestScriptOutput:
    def test_valid(self):
        data = {
            "video_title": "Intro",
            "script_chunks": ["Welcome.", "Today we learn."],
        }
        obj = ScriptOutput.model_validate(data)
        assert len(obj.script_chunks) == 2

    def test_empty_chunks(self):
        with pytest.raises(ValidationError):
            ScriptOutput.model_validate({"video_title": "X", "script_chunks": []})


class TestFinalScripts:
    def test_valid(self):
        data = {"scripts": [{"video_title": "V1", "script_chunks": ["chunk"]}]}
        obj = FinalScripts.model_validate(data)
        assert len(obj.scripts) == 1


class TestUsageStats:
    def test_defaults(self):
        u = UsageStats()
        assert u.tokens_in == 0
        assert u.cost_usd == 0.0


class TestStageResult:
    def test_envelope(self):
        content = NormalizedContent(
            title="T", sections=[{"heading": "H", "content": "C"}]
        )
        usage = UsageStats(tokens_in=100, tokens_out=50, cost_usd=0.001)
        result = StageResult(output=content, usage=usage)
        assert result.output.title == "T"
        assert result.usage.cost_usd == 0.001


class TestPipelineRun:
    def test_defaults(self):
        run = PipelineRun()
        assert run.version == "3.0.0"
        assert run.completed_at is None
        assert run.stages_completed == []
