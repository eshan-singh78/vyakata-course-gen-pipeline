"""Media pipeline models for audio, visual, and assembly I/O schemas."""

from pathlib import Path

from pydantic import BaseModel, Field


class AudioSegment(BaseModel):
    """One TTS audio segment for a script chunk."""

    chunk_index: int = Field(description="Index within the video")
    audio_path: Path = Field(description="Path to generated audio file")
    duration_seconds: float = Field(
        default=0.0, description="Duration measured via ffprobe"
    )


class VideoAudio(BaseModel):
    """All audio segments for one video."""

    video_title: str = Field(description="Video title")
    segments: list[AudioSegment] = Field(description="Per-chunk audio segments")
    merged_audio_path: Path | None = Field(
        default=None, description="Concatenated audio file"
    )


class SlideFrame(BaseModel):
    """One rendered slide frame."""

    chunk_index: int = Field(description="Index within the video")
    html_path: Path = Field(description="Path to generated HTML")
    image_path: Path = Field(description="Path to rendered PNG")


class VideoFrames(BaseModel):
    """All frames for one video."""

    video_title: str = Field(description="Video title")
    frames: list[SlideFrame] = Field(description="Per-chunk frames")


class VideoOutput(BaseModel):
    """Final assembled video artifact."""

    video_title: str = Field(description="Video title")
    audio_path: Path = Field(description="Merged audio file")
    frames_dir: Path = Field(description="Directory containing frame images")
    final_mp4_path: Path = Field(description="Final assembled MP4")
    duration_seconds: float = Field(default=0.0, description="Total duration")


class MediaPipelineRun(BaseModel):
    """Metadata for a media pipeline execution."""

    videos_requested: int = Field(default=0)
    videos_completed: int = Field(default=0)
    videos_failed: int = Field(default=0)
    total_audio_segments: int = Field(default=0)
    total_frames: int = Field(default=0)
    errors: list[str] = Field(default_factory=list)
