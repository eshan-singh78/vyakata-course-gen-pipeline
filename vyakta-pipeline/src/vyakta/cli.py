"""Typer CLI entrypoint with async support."""

import asyncio
from pathlib import Path

import structlog
import typer

from vyakta.config import Settings
from vyakta.llm.client import get_client
from vyakta.media.config import MediaSettings
from vyakta.media.pipeline import MediaPipeline
from vyakta.models import (
    CourseStructure,
    FinalScripts,
    NormalizedContent,
    VideoPlan,
)
from vyakta.pipeline import Pipeline
from vyakta.stages.architect import ArchitectStage
from vyakta.stages.normalizer import NormalizerStage
from vyakta.stages.planner import PlannerStage
from vyakta.stages.scriptor import ScriptorStage

app = typer.Typer(help="Vyakta Course Generation Pipeline v3")


def _configure_logging(verbose: bool = False, quiet: bool = False) -> None:
    level = "DEBUG" if verbose else ("WARNING" if quiet else "INFO")
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer(colors=True),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    import logging

    logging.getLogger().setLevel(getattr(logging, level))


def _read_input(path: Path) -> str:
    if not path.exists():
        raise typer.BadParameter(f"File not found: {path}")
    return path.read_text(encoding="utf-8")


def _base_settings(
    output_dir: Path | None = None,
    max_concurrent: int | None = None,
    batch_size: int | None = None,
) -> Settings:
    """Load settings from env, then apply CLI overrides."""
    overrides: dict = {}
    if output_dir is not None:
        overrides["output_dir"] = output_dir
    if max_concurrent is not None:
        overrides["max_concurrent_llm_calls"] = max_concurrent
    if batch_size is not None:
        overrides["batch_size"] = batch_size
    base = Settings()
    if overrides:
        return base.model_copy(update=overrides)
    return base


@app.command()
def normalize(
    input_file: Path = typer.Option(..., "--input", "-i", help="Raw HTML/text file"),
    output: Path = typer.Option(..., "--output", "-o", help="Output JSON path"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    quiet: bool = typer.Option(False, "--quiet", "-q"),
) -> None:
    """Run Stage 1: Content Normalizer."""
    _configure_logging(verbose, quiet)

    async def _run():
        cfg = _base_settings()
        client = get_client(cfg)
        stage = NormalizerStage(client, cfg)
        result = await stage.run(_read_input(input_file))
        output.write_text(
            result.output.model_dump_json(indent=2), encoding="utf-8"
        )
        typer.echo(f"Normalized content written to {output}")
        typer.echo(
            f"Tokens: {result.usage.tokens_in} in / {result.usage.tokens_out} out | "
            f"Cost: ${result.usage.cost_usd:.4f}"
        )

    asyncio.run(_run())


@app.command()
def architect(
    input_file: Path = typer.Option(..., "--input", "-i", help="Stage 1 JSON"),
    output: Path = typer.Option(..., "--output", "-o", help="Output JSON path"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    quiet: bool = typer.Option(False, "--quiet", "-q"),
) -> None:
    """Run Stage 2: Curriculum Architect."""
    _configure_logging(verbose, quiet)

    async def _run():
        cfg = _base_settings()
        client = get_client(cfg)
        stage = ArchitectStage(client, cfg)
        data = NormalizedContent.model_validate_json(_read_input(input_file))
        result = await stage.run(data)
        output.write_text(
            result.output.model_dump_json(indent=2), encoding="utf-8"
        )
        typer.echo(f"Course structure written to {output}")
        typer.echo(
            f"Tokens: {result.usage.tokens_in} in / {result.usage.tokens_out} out | "
            f"Cost: ${result.usage.cost_usd:.4f}"
        )

    asyncio.run(_run())


@app.command()
def planner(
    input_file: Path = typer.Option(..., "--input", "-i", help="Stage 2 JSON"),
    output: Path = typer.Option(..., "--output", "-o", help="Output JSON path"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    quiet: bool = typer.Option(False, "--quiet", "-q"),
) -> None:
    """Run Stage 3: Chapter & Video Planner."""
    _configure_logging(verbose, quiet)

    async def _run():
        cfg = _base_settings()
        client = get_client(cfg)
        stage = PlannerStage(client, cfg)
        data = CourseStructure.model_validate_json(_read_input(input_file))
        result = await stage.run(data)
        output.write_text(
            result.output.model_dump_json(indent=2), encoding="utf-8"
        )
        typer.echo(f"Video plan written to {output}")
        typer.echo(
            f"Tokens: {result.usage.tokens_in} in / {result.usage.tokens_out} out | "
            f"Cost: ${result.usage.cost_usd:.4f}"
        )

    asyncio.run(_run())


@app.command()
def script(
    input_file: Path = typer.Option(..., "--input", "-i", help="Stage 3 JSON"),
    output: Path = typer.Option(..., "--output", "-o", help="Output JSON path"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    quiet: bool = typer.Option(False, "--quiet", "-q"),
) -> None:
    """Run Stage 4: Script Generator."""
    _configure_logging(verbose, quiet)

    async def _run():
        cfg = _base_settings()
        client = get_client(cfg)
        stage = ScriptorStage(client, cfg)
        data = VideoPlan.model_validate_json(_read_input(input_file))
        result = await stage.run(data)
        final = FinalScripts(scripts=result.output)
        output.write_text(final.model_dump_json(indent=2), encoding="utf-8")
        typer.echo(f"Scripts written to {output}")
        typer.echo(
            f"Scripts: {len(result.output)} | "
            f"Tokens: {result.usage.tokens_in} in / {result.usage.tokens_out} out | "
            f"Cost: ${result.usage.cost_usd:.4f}"
        )

    asyncio.run(_run())


@app.command()
def run(
    input_file: Path = typer.Option(..., "--input", "-i", help="Raw HTML/text file"),
    output_dir: Path = typer.Option("./output", "--output-dir", "-o", help="Checkpoint directory"),
    resume_from: str | None = typer.Option(
        None,
        "--resume-from",
        help="Resume from a stage: stage1, stage2, or stage3",
    ),
    max_concurrent: int = typer.Option(
        None, "--max-concurrent", "-c", help="Max concurrent LLM calls"
    ),
    batch_size: int = typer.Option(
        None, "--batch-size", "-b", help="Videos per batch in Stage 4"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    quiet: bool = typer.Option(False, "--quiet", "-q"),
) -> None:
    """Run the full pipeline."""
    _configure_logging(verbose, quiet)

    async def _run():
        cfg = _base_settings(
            output_dir=output_dir,
            max_concurrent=max_concurrent,
            batch_size=batch_size,
        )
        pipeline = Pipeline(settings_obj=cfg)
        raw = _read_input(input_file)
        try:
            final, meta = await pipeline.run(raw, resume_from=resume_from)
        except Exception as exc:
            logger = structlog.get_logger()
            logger.error("pipeline_failed", error=str(exc))
            raise typer.Exit(code=1) from exc
        typer.echo(f"Pipeline complete. Checkpoints saved to {output_dir}")
        typer.echo(f"Total scripts generated: {len(final.scripts)}")
        typer.echo(
            f"Total tokens: {meta.total_usage.tokens_in} in / "
            f"{meta.total_usage.tokens_out} out | "
            f"Total cost: ${meta.total_usage.cost_usd:.4f}"
        )

    asyncio.run(_run())


@app.command()
def media(
    input_file: Path = typer.Option(
        ..., "--input", "-i", help="Stage 4 JSON (FinalScripts)"
    ),
    output_dir: Path = typer.Option(
        "./media_output", "--output-dir", "-o", help="Media output dir"
    ),
    tts_provider: str = typer.Option(
        "openai", "--tts-provider", help="TTS provider: openai, kyutai"
    ),
    skip_existing: bool = typer.Option(
        True, "--skip-existing/--no-skip-existing"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    quiet: bool = typer.Option(False, "--quiet", "-q"),
) -> None:
    """Run Media Pipeline: TTS → Visuals → FFmpeg assembly."""
    _configure_logging(verbose, quiet)

    async def _run():
        scripts = FinalScripts.model_validate_json(_read_input(input_file))
        cfg = MediaSettings(
            media_output_dir=output_dir,
            tts_provider=tts_provider,
            skip_existing=skip_existing,
        )
        pipeline = MediaPipeline(settings=cfg)
        try:
            outputs, meta = await pipeline.run(scripts, output_dir=output_dir)
        except Exception as exc:
            logger = structlog.get_logger()
            logger.error("media_pipeline_failed", error=str(exc))
            raise typer.Exit(code=1) from exc

        typer.echo(
            f"Media pipeline complete. "
            f"{meta.videos_completed}/{meta.videos_requested} videos generated."
        )
        for out in outputs:
            typer.echo(
                f"  [{out.video_title}] "
                f"{out.duration_seconds:.1f}s → {out.final_mp4_path}"
            )
        if meta.videos_failed:
            typer.echo(f"Failed: {meta.videos_failed}")
            for err in meta.errors:
                typer.echo(f"  Error: {err}")

    asyncio.run(_run())


def main() -> None:
    app()


if __name__ == "__main__":
    main()
