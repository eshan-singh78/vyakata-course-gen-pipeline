"""Media pipeline orchestrator: TTS → Visuals → FFmpeg assembly."""

import asyncio
from pathlib import Path

import aiofiles
import structlog

from vyakta.media.assembly.ffmpeg import FFmpegError, FFmpegWrapper
from vyakta.media.config import MediaSettings
from vyakta.media.models import (
    AudioSegment,
    MediaPipelineRun,
    SlideFrame,
    VideoAudio,
    VideoFrames,
    VideoOutput,
)
from vyakta.media.visuals.puppeteer import PuppeteerError, PuppeteerWrapper
from vyakta.media.visuals.renderer import SlideRenderer
from vyakta.media.voice import get_tts_client
from vyakta.models import FinalScripts, ScriptOutput

_log = structlog.get_logger("vyakta.media.pipeline")


class MediaPipeline:
    """Runs the 3-layer media pipeline: Voice → Visuals → Assembly."""

    def __init__(self, settings: MediaSettings | None = None):
        self.settings = settings or MediaSettings()
        self._tts = None
        self._renderer = SlideRenderer(
            template_dir=self.settings.slide_template_dir,
            width=self.settings.slide_width,
            height=self.settings.slide_height,
        )
        self._puppeteer = PuppeteerWrapper(self.settings)
        self._ffmpeg = FFmpegWrapper(self.settings)
        self.run_meta = MediaPipelineRun()

    @property
    def tts(self):
        if self._tts is None:
            self._tts = get_tts_client(self.settings)
        return self._tts

    async def run(
        self,
        scripts: FinalScripts,
        output_dir: Path | None = None,
    ) -> tuple[list[VideoOutput], MediaPipelineRun]:
        """Generate full videos from scripts."""
        base_dir = output_dir or self.settings.media_output_dir
        await self._ensure_dir(base_dir)

        self.run_meta.videos_requested = len(scripts.scripts)

        # Pre-flight checks for external tools
        try:
            self._puppeteer.check_available()
            self._ffmpeg.check_available()
        except (PuppeteerError, FFmpegError) as exc:
            self.run_meta.errors.append(str(exc))
            self.run_meta.videos_failed = self.run_meta.videos_requested
            _log.error("preflight_failed", error=str(exc))
            return [], self.run_meta

        # Process videos with concurrency limit
        sem = asyncio.Semaphore(self.settings.max_concurrent_videos)

        async def _bounded(script: ScriptOutput) -> VideoOutput | Exception:
            async with sem:
                try:
                    return await self._process_one_video(script, base_dir)
                except Exception as exc:
                    _log.error("video_failed", video_title=script.video_title, error=str(exc))
                    return exc

        tasks = [asyncio.create_task(_bounded(s)) for s in scripts.scripts]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        outputs: list[VideoOutput] = []
        for res in results:
            if isinstance(res, Exception):
                self.run_meta.videos_failed += 1
                self.run_meta.errors.append(str(res))
            else:
                outputs.append(res)
                self.run_meta.videos_completed += 1

        _log.info(
            "media_pipeline_complete",
            requested=self.run_meta.videos_requested,
            completed=self.run_meta.videos_completed,
            failed=self.run_meta.videos_failed,
        )
        return outputs, self.run_meta

    async def _process_one_video(
        self, script: ScriptOutput, base_dir: Path
    ) -> VideoOutput:
        video_dir = base_dir / _sanitize_filename(script.video_title)
        await self._ensure_dir(video_dir)

        final_mp4 = video_dir / "final.mp4"
        if self.settings.skip_existing and final_mp4.exists():
            _log.info("skip_existing", video=script.video_title, path=str(final_mp4))
            dur = await _probe_duration_safe(final_mp4, self.settings.ffprobe_path)
            return VideoOutput(
                video_title=script.video_title,
                audio_path=video_dir / "audio_merged.mp3",
                frames_dir=video_dir / "frames",
                final_mp4_path=final_mp4,
                duration_seconds=dur,
            )

        audio_dir = video_dir / "audio"
        frames_dir = video_dir / "frames"
        slides_dir = video_dir / "slides"

        await self._ensure_dir(audio_dir)
        await self._ensure_dir(frames_dir)
        await self._ensure_dir(slides_dir)

        # ---- Layer 1: Voice ----
        audio = await self._generate_audio(script, audio_dir)
        self.run_meta.total_audio_segments += len(audio.segments)

        # ---- Layer 2: Visuals ----
        html_paths = await self._renderer.render_video_slides(
            script.video_title, script.script_chunks, slides_dir
        )
        image_paths = await self._puppeteer.render_slides(html_paths, frames_dir)

        frames = VideoFrames(
            video_title=script.video_title,
            frames=[
                SlideFrame(
                    chunk_index=i,
                    html_path=html_paths[i],
                    image_path=image_paths[i],
                )
                for i in range(len(image_paths))
            ],
        )
        self.run_meta.total_frames += len(frames.frames)

        # ---- Layer 3: Assembly ----
        durations = [s.duration_seconds for s in audio.segments]
        dur = await self._ffmpeg.assemble_video(
            video_title=script.video_title,
            audio_path=audio.merged_audio_path,
            frame_paths=[f.image_path for f in frames.frames],
            durations=durations,
            output_path=final_mp4,
        )

        return VideoOutput(
            video_title=script.video_title,
            audio_path=audio.merged_audio_path,
            frames_dir=frames_dir,
            final_mp4_path=final_mp4,
            duration_seconds=dur,
        )

    async def _generate_audio(self, script: ScriptOutput, audio_dir: Path) -> VideoAudio:
        """Generate TTS audio for each chunk and merge."""
        sem = asyncio.Semaphore(5)
        segments: list[AudioSegment] = []

        async def _synth(idx: int, text: str) -> AudioSegment:
            async with sem:
                path = audio_dir / f"chunk_{idx:03d}.{self.settings.tts_format}"
                dur = await self.tts.synthesize(text, path)
                return AudioSegment(
                    chunk_index=idx,
                    audio_path=path,
                    duration_seconds=dur,
                )

        tasks = [
            asyncio.create_task(_synth(i, text))
            for i, text in enumerate(script.script_chunks)
        ]
        segments = await asyncio.gather(*tasks)

        # Merge audio segments with FFmpeg concat demuxer
        merged_path = audio_dir.parent / f"audio_merged.{self.settings.tts_format}"
        await _merge_audio(segments, merged_path, self.settings.ffmpeg_path)

        return VideoAudio(
            video_title=script.video_title,
            segments=list(segments),
            merged_audio_path=merged_path,
        )

    @staticmethod
    async def _ensure_dir(path: Path) -> None:
        import aiofiles.os

        await aiofiles.os.makedirs(path, exist_ok=True)


def _sanitize_filename(name: str) -> str:
    """Replace filesystem-unfriendly characters."""
    import re

    sanitized = re.sub(r"[^\w\-_.]", "_", name).strip("_-.")[:100]
    if not sanitized:
        sanitized = "untitled"
    return sanitized


async def _merge_audio(
    segments: list[AudioSegment],
    output_path: Path,
    ffmpeg: str,
) -> None:
    """Concatenate audio segments with FFmpeg."""
    lines = [f"file '{s.audio_path.resolve()}'" for s in segments]
    concat_path = output_path.with_suffix(".concat.txt")
    async with aiofiles.open(concat_path, "w", encoding="utf-8") as f:
        await f.write("\n".join(lines) + "\n")

    try:
        proc = await asyncio.create_subprocess_exec(
            ffmpeg,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_path),
            "-c",
            "copy",
            str(output_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise FFmpegError(
                f"Audio merge failed (exit {proc.returncode}): "
                f"{stderr.decode().strip()}"
            )
    finally:
        concat_path.unlink(missing_ok=True)


async def _probe_duration_safe(path: Path, ffprobe: str) -> float:
    try:
        proc = await asyncio.create_subprocess_exec(
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        return float(stdout.decode().strip())
    except Exception:
        return 0.0
