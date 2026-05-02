# Vyakta Course Generation Pipeline (v2)

A production-grade, 4-stage pipeline that transforms raw cybersecurity learning content into structured, TTS-optimized video scripts using LLMs.

## Architecture

```
Raw Content
    |
[1] Content Normalizer     — clean & standardize
    |
[2] Curriculum Architect  — course → modules → chapters
    |
[3] Chapter & Video Planner — chapters → 2–5 min videos
    |
[4] Script Generator     — TTS-optimized narration chunks
```

## Quick Start

```bash
# 1. Install
make install

# 2. Configure
cp .env.example .env
# Edit .env with your API key

# 3. Run the full pipeline
vyakta run --input raw_content.html --output-dir ./out

# Or run stages individually
vyakta normalize  --input raw.html    --output stage1.json
vyakta architect  --input stage1.json --output stage2.json
vyakta planner    --input stage2.json --output stage3.json
vyakta script     --input stage3.json --output stage4.json

# Resume from an intermediate stage
vyakta run --input raw.html --output-dir ./out --resume-from stage2
```

## Project Structure

- `src/vyakta/stages/` — the 4 pipeline stages
- `src/vyakta/llm/` — unified Anthropic + OpenAI client
- `src/vyakta/prompts/` — prompt templates
- `src/vyakta/models.py` — Pydantic schemas for all JSON I/O
- `tests/` — pytest suite with mocked LLM responses

## Configuration

All settings are loaded from environment variables (see `.env.example`) via Pydantic Settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `anthropic` | `anthropic` or `openai` |
| `MODEL_NAME` | `claude-sonnet-4-6` | Model to use |
| `MAX_RETRIES` | `3` | Retry attempts per LLM call |
| `TEMPERATURE` | `0.2` | Sampling temperature |
| `OUTPUT_DIR` | `./output` | Checkpoint directory |
