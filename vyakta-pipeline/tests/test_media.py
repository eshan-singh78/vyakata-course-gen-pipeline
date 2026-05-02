"""Unit tests for media pipeline components."""

from pathlib import Path

import pytest

from vyakta.media.assembly.ffmpeg import FFmpegError, FFmpegWrapper
from vyakta.media.config import MediaSettings, TTSProvider
from vyakta.media.pipeline import MediaPipeline, _sanitize_filename
from vyakta.media.visuals.puppeteer import PuppeteerError, PuppeteerWrapper
from vyakta.media.visuals.renderer import SlideRenderer
from vyakta.media.voice.base import TTSClient
from vyakta.models import FinalScripts, ScriptOutput


class FakeTTSClient(TTSClient):
    """Test double for TTS."""

    def __init__(self):
        super().__init__(MediaSettings())
        self.call_count = 0

    async def synthesize(self, text: str, output_path: Path) -> float:
        self.call_count += 1
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"FAKE_AUDIO")
        return 1.5


class TestSanitizeFilename:
    def test_simple(self):
        assert _sanitize_filename("Hello World") == "Hello_World"

    def test_special_chars(self):
        assert _sanitize_filename("A/B<C>D") == "A_B_C_D"

    def test_truncate(self):
        long_name = "a" * 200
        assert len(_sanitize_filename(long_name)) == 100


class TestMediaSettings:
    def test_defaults(self):
        s = MediaSettings()
        assert s.tts_provider == TTSProvider.OPENAI
        assert s.slide_width == 1920

    def test_speed_validation(self):
        with pytest.raises(ValueError, match="between 0.25 and 4.0"):
            MediaSettings(tts_speed=5.0)

    def test_positive_validation(self):
        with pytest.raises(ValueError, match="positive"):
            MediaSettings(slide_width=0)

    def test_openai_api_key_required(self):
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            MediaSettings(tts_provider="openai").get_tts_api_key()


class TestSlideRenderer:
    @pytest.mark.asyncio
    async def test_render_slides(self, tmp_path: Path):
        renderer = SlideRenderer(width=1920, height=1080)
        chunks = ["Chunk one", "Chunk two"]
        paths = await renderer.render_video_slides("My Video", chunks, tmp_path)
        assert len(paths) == 2
        for p in paths:
            assert p.exists()
            content = p.read_text()
            assert "My Video" in content
            assert "VYAKTA" in content


class TestMediaPipeline:
    @pytest.mark.asyncio
    async def test_empty_scripts(self, tmp_path: Path):
        pipeline = MediaPipeline(settings=MediaSettings(media_output_dir=tmp_path))
        scripts = FinalScripts(scripts=[])
        outputs, meta = await pipeline.run(scripts, output_dir=tmp_path)
        assert outputs == []
        assert meta.videos_requested == 0

    @pytest.mark.asyncio
    async def test_preflight_failure_no_node(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(
            "vyakta.media.visuals.puppeteer.PuppeteerWrapper.check_available",
            lambda self: (_ for _ in ()).throw(PuppeteerError("no node")),
        )
        pipeline = MediaPipeline(settings=MediaSettings(media_output_dir=tmp_path))
        scripts = FinalScripts(
            scripts=[ScriptOutput(video_title="V1", script_chunks=["Hello"])]
        )
        outputs, meta = await pipeline.run(scripts, output_dir=tmp_path)
        assert outputs == []
        assert meta.videos_failed == 1
        assert "no node" in meta.errors[0]

    @pytest.mark.asyncio
    async def test_skip_existing(self, tmp_path: Path):
        video_dir = tmp_path / "My_Video"
        video_dir.mkdir(parents=True, exist_ok=True)
        (video_dir / "final.mp4").write_bytes(b"FAKE")

        settings = MediaSettings(media_output_dir=tmp_path, skip_existing=True)
        pipeline = MediaPipeline(settings=settings)
        # Bypass preflight by monkeypatching
        pipeline._puppeteer.check_available = lambda: None
        pipeline._ffmpeg.check_available = lambda: None

        scripts = FinalScripts(
            scripts=[ScriptOutput(video_title="My Video", script_chunks=["Hello"])]
        )
        outputs, meta = await pipeline.run(scripts, output_dir=tmp_path)
        assert len(outputs) == 1
        assert outputs[0].final_mp4_path.exists()
        assert meta.videos_completed == 1

    @pytest.mark.asyncio
    async def test_full_mocked_pipeline(self, tmp_path: Path, monkeypatch):
        """End-to-end media pipeline with all external tools mocked."""
        settings = MediaSettings(
            media_output_dir=tmp_path,
            skip_existing=False,
            tts_format="mp3",
        )
        pipeline = MediaPipeline(settings=settings)
        pipeline._tts = FakeTTSClient()

        # Mock Puppeteer to just copy HTML to "PNG"
        async def fake_render(html_paths, output_dir):
            output_dir.mkdir(parents=True, exist_ok=True)
            out = []
            for i, hp in enumerate(html_paths):
                op = output_dir / f"frame_{i:03d}.png"
                op.write_bytes(hp.read_bytes())
                out.append(op)
            return out

        pipeline._puppeteer.render_slides = fake_render
        pipeline._puppeteer.check_available = lambda: None

        # Mock FFmpeg to just create empty files
        async def fake_assemble(**kwargs):
            kwargs["output_path"].parent.mkdir(parents=True, exist_ok=True)
            kwargs["output_path"].write_bytes(b"FAKE_MP4")
            return sum(kwargs["durations"])

        pipeline._ffmpeg.assemble_video = fake_assemble
        pipeline._ffmpeg.check_available = lambda: None

        # Mock audio merge to avoid invalid MP3 concat
        import vyakta.media.pipeline as _mp

        orig_merge = _mp._merge_audio

        async def fake_merge(segments, output_path, ffmpeg):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"FAKE_MERGED_AUDIO")

        _mp._merge_audio = fake_merge

        try:
            scripts = FinalScripts(
                scripts=[
                    ScriptOutput(
                        video_title="Intro",
                        script_chunks=["Welcome to the course.", "Today we learn security."],
                    )
                ]
            )
            outputs, meta = await pipeline.run(scripts, output_dir=tmp_path)
            assert len(outputs) == 1
            assert outputs[0].video_title == "Intro"
            assert outputs[0].final_mp4_path.exists()
            assert meta.videos_completed == 1
            assert meta.videos_failed == 0
            assert pipeline._tts.call_count == 2
        finally:
            _mp._merge_audio = orig_merge


class TestFFmpegWrapper:
    def test_check_available_missing(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda x: None)
        fw = FFmpegWrapper(MediaSettings())
        with pytest.raises(FFmpegError, match="ffmpeg not found"):
            fw.check_available()


class TestPuppeteerWrapper:
    def test_check_available_missing_node(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda x: None)
        pw = PuppeteerWrapper(MediaSettings())
        with pytest.raises(PuppeteerError, match="Node.js"):
            pw.check_available()
