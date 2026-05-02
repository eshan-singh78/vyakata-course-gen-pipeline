"""HTML slide renderer using Jinja2."""

from pathlib import Path

import structlog

_log = structlog.get_logger("vyakta.media.visuals")


class SlideRenderer:
    """Renders script chunks into HTML slide files."""

    def __init__(
        self,
        template_dir: Path | None = None,
        width: int = 1920,
        height: int = 1080,
    ):
        from jinja2 import Environment, FileSystemLoader

        tpl_dir = template_dir or Path(__file__).parent / "templates"
        self._env = Environment(loader=FileSystemLoader(tpl_dir))
        self._template = self._env.get_template("slide.html")
        self._width = width
        self._height = height

    async def render_video_slides(
        self,
        video_title: str,
        chunks: list[str],
        output_dir: Path,
    ) -> list[Path]:
        """Generate one HTML file per chunk. Return ordered list of paths."""
        import aiofiles.os

        await aiofiles.os.makedirs(output_dir, exist_ok=True)
        paths: list[Path] = []
        total = len(chunks)
        for idx, chunk in enumerate(chunks, start=1):
            html = self._template.render(
                video_title=video_title,
                content=chunk,
                slide_number=idx,
                total_slides=total,
                width=self._width,
                height=self._height,
                progress_percent=round((idx / total) * 100, 1),
            )
            path = output_dir / f"slide_{idx:03d}.html"
            async with aiofiles.open(path, "w", encoding="utf-8") as f:
                await f.write(html)
            paths.append(path)
        _log.info(
            "slides_rendered",
            video_title=video_title,
            slides=total,
            output_dir=str(output_dir),
        )
        return paths
