"""Puppeteer wrapper for rendering HTML slides to PNG frames."""

import asyncio
import json
from pathlib import Path

import aiofiles  # noqa: F401
import structlog

from vyakta.media.config import MediaSettings

_log = structlog.get_logger("vyakta.media.visuals")


class PuppeteerError(Exception):
    """Raised when Puppeteer rendering fails."""


class PuppeteerWrapper:
    """Renders HTML files to PNG images via Puppeteer (Node.js)."""

    def __init__(self, settings: MediaSettings):
        self._settings = settings
        self._node_script = Path(__file__).parent / "puppeteer_render.js"

    async def render_slides(
        self,
        html_paths: list[Path],
        output_dir: Path,
    ) -> list[Path]:
        """Render a list of HTML files to PNG images."""
        if not html_paths:
            return []

        import aiofiles.os

        await aiofiles.os.makedirs(output_dir, exist_ok=True)

        # Build config JSON for the Node.js script
        config = {
            "width": self._settings.slide_width,
            "height": self._settings.slide_height,
            "slides": [
                {
                    "html_path": str(p.resolve()),
                    "output_path": str((output_dir / f"frame_{i:03d}.png").resolve()),
                }
                for i, p in enumerate(html_paths)
            ],
        }

        config_path = output_dir / "puppeteer_config.json"
        async with aiofiles.open(config_path, "w", encoding="utf-8") as f:
            await f.write(json.dumps(config))

        try:
            cmd = ["node", str(self._node_script), str(config_path)]
            if self._settings.puppeteer_executable:
                cmd[0] = self._settings.puppeteer_executable

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                raise PuppeteerError(
                    f"Puppeteer failed (exit {proc.returncode}): "
                    f"{stderr.decode().strip()}"
                )
            _log.info(
                "puppeteer_complete",
                frames=len(html_paths),
                output_dir=str(output_dir),
            )
        finally:
            Path(config_path).unlink(missing_ok=True)

        return [Path(s["output_path"]) for s in config["slides"]]

    def check_available(self) -> None:
        """Raise PuppeteerError if Node.js / Puppeteer is not available."""
        import shutil

        node_cmd = self._settings.puppeteer_executable or "node"
        if not shutil.which(node_cmd):
            raise PuppeteerError(
                f"Node.js ('{node_cmd}') not found. "
                "Install Node.js and Puppeteer: npm install puppeteer"
            )
        if not self._node_script.exists():
            raise PuppeteerError(
                f"Puppeteer render script missing: {self._node_script}"
            )
