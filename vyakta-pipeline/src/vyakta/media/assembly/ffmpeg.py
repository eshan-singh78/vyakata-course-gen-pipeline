"""FFmpeg-based video assembly layer."""

import asyncio
from pathlib import Path

import aiofiles  # noqa: F401
import structlog

from vyakta.media.config import MediaSettings

_log = structlog.get_logger("vyakta.media.assembly")


class FFmpegError(Exception):
    """Raised when FFmpeg assembly fails."""


class FFmpegWrapper:
    """Assembles audio + frame images into MP4 using FFmpeg."""

    def __init__(self, settings: MediaSettings):
        self._settings = settings

    def check_available(self) -> None:
        """Raise FFmpegError if ffmpeg/ffprobe are not available."""
        import shutil

        if not shutil.which(self._settings.ffmpeg_path):
            raise FFmpegError(
                f"ffmpeg not found: {self._settings.ffmpeg_path}. "
                "Install FFmpeg and ensure it is on PATH."
            )
        if not shutil.which(self._settings.ffprobe_path):
            raise FFmpegError(
                f"ffprobe not found: {self._settings.ffprobe_path}. "
                "Install FFmpeg and ensure it is on PATH."
            )

    async def assemble_video(
        self,
        video_title: str,
        audio_path: Path,
        frame_paths: list[Path],
        durations: list[float],
        output_path: Path,
    ) -> float:
        """Combine frame images + audio into final MP4. Return duration."""
        if len(frame_paths) != len(durations):
            raise FFmpegError("frame_paths and durations must have same length")

        import aiofiles.os

        await aiofiles.os.makedirs(output_path.parent, exist_ok=True)

        # Build concat demuxer file
        concat_lines: list[str] = []
        for frame, dur in zip(frame_paths, durations):
            concat_lines.append(f"file '{frame.resolve()}'")
            concat_lines.append(f"duration {dur:.3f}")
        # FFmpeg concat demuxer requires a final file line without duration
        if frame_paths:
            concat_lines.append(f"file '{frame_paths[-1].resolve()}'")

        concat_path = output_path.with_suffix(".concat.txt")
        async with aiofiles.open(concat_path, "w", encoding="utf-8") as f:
            await f.write("\n".join(concat_lines) + "\n")

        try:
            cmd = [
                self._settings.ffmpeg_path,
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_path),
                "-i",
                str(audio_path),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-r",
                str(self._settings.output_fps),
                "-shortest",
                str(output_path),
            ]

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                raise FFmpegError(
                    f"FFmpeg failed (exit {proc.returncode}): "
                    f"{stderr.decode().strip()}"
                )

            # Probe final duration
            total_dur = await _probe_duration(output_path, self._settings.ffprobe_path)
            _log.info(
                "video_assembled",
                video_title=video_title,
                output=str(output_path),
                duration=round(total_dur, 2),
            )
            return total_dur
        finally:
            Path(concat_path).unlink(missing_ok=True)


async def _probe_duration(path: Path, ffprobe: str) -> float:
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
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise FFmpegError(f"ffprobe failed: {stderr.decode().strip()}")
    try:
        return float(stdout.decode().strip())
    except ValueError as exc:
        raise FFmpegError(f"ffprobe returned invalid duration: {stdout.decode().strip()}") from exc
