"""Prompt templates for each pipeline stage."""

from vyakta.prompts.architect import ARCHITECT_PROMPT
from vyakta.prompts.normalizer import NORMALIZER_PROMPT
from vyakta.prompts.planner import PLANNER_PROMPT
from vyakta.prompts.scriptor import SCRIPTOR_BATCH_PROMPT, SCRIPTOR_PROMPT

__all__ = [
    "NORMALIZER_PROMPT",
    "ARCHITECT_PROMPT",
    "PLANNER_PROMPT",
    "SCRIPTOR_PROMPT",
    "SCRIPTOR_BATCH_PROMPT",
]
