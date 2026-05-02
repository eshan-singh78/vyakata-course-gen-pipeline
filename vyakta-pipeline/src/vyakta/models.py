"""Pydantic models for all pipeline JSON I/O schemas."""

from datetime import UTC, datetime
from typing import Generic, TypeVar

from pydantic import BaseModel, Field


class Section(BaseModel):
    heading: str = Field(description="Section heading")
    content: str = Field(description="Clean paragraph text")


class NormalizedContent(BaseModel):
    title: str = Field(description="Document title")
    sections: list[Section] = Field(description="Normalized sections")


class ChapterRef(BaseModel):
    chapter_title: str = Field(description="Chapter title")
    source_sections: list[str] = Field(description="Section references used")


class Module(BaseModel):
    module_title: str = Field(description="Module title")
    chapters: list[ChapterRef] = Field(description="Chapters in this module")


class CourseStructure(BaseModel):
    course_title: str = Field(description="Course title")
    modules: list[Module] = Field(description="Modules in the course")


class VideoPlanItem(BaseModel):
    video_title: str = Field(description="Video title")
    concept_focus: str = Field(description="The single concept this video covers")


class ChapterPlan(BaseModel):
    chapter_title: str = Field(description="Chapter title")
    videos: list[VideoPlanItem] = Field(description="Videos for this chapter")


class VideoPlan(BaseModel):
    chapters: list[ChapterPlan] = Field(description="Chapters with video plans")


class ScriptOutput(BaseModel):
    video_title: str = Field(description="Video title")
    script_chunks: list[str] = Field(
        description="TTS-optimized narration chunks", min_length=1
    )


class FinalScripts(BaseModel):
    scripts: list[ScriptOutput] = Field(description="All generated scripts")


# v3 production models

T = TypeVar("T")


class UsageStats(BaseModel):
    tokens_in: int = Field(default=0, description="Input tokens consumed")
    tokens_out: int = Field(default=0, description="Output tokens consumed")
    cost_usd: float = Field(default=0.0, description="Estimated cost in USD")
    latency_ms: int = Field(default=0, description="API call latency in milliseconds")


class StageResult(BaseModel, Generic[T]):
    """Envelope for stage output with telemetry."""

    output: T = Field(description="Stage output data")
    usage: UsageStats = Field(default_factory=UsageStats)


class PipelineRun(BaseModel):
    """Metadata for a pipeline execution."""

    version: str = Field(default="3.0.0")
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = Field(default=None)
    stages_completed: list[str] = Field(default_factory=list)
    total_usage: UsageStats = Field(default_factory=UsageStats)
    checkpoints: list[str] = Field(default_factory=list)
