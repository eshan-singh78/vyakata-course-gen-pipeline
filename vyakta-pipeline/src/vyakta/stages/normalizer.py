"""Stage 1: Content Normalizer with real HTML parsing."""

from vyakta.models import NormalizedContent, StageResult
from vyakta.prompts import NORMALIZER_PROMPT
from vyakta.stages.base import Stage, StageError


class NormalizerStage(Stage):
    """Cleans and standardizes raw HTML into normalized sections."""

    prompt_template = NORMALIZER_PROMPT
    output_model = NormalizedContent

    def _preprocess_html(self, raw_html: str) -> str:
        """Use BeautifulSoup to strip noise and extract clean text."""
        try:
            from bs4 import BeautifulSoup, Comment
        except ImportError as exc:
            raise RuntimeError("beautifulsoup4 is required for HTML parsing") from exc

        soup = BeautifulSoup(raw_html, "html.parser")

        # Remove scripts, styles, nav, footer, aside, comments
        for tag_name in ["script", "style", "nav", "footer", "aside", "header"]:
            for tag in soup.find_all(tag_name):
                tag.decompose()
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()

        # Extract headings and paragraphs with structure
        lines: list[str] = []
        for tag in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li"]):
            text = tag.get_text(strip=True)
            if text:
                lines.append(text)

        return "\n\n".join(lines)

    def build_prompt(self, data: str) -> str:
        cleaned = self._preprocess_html(data)
        wrapped = self.wrap_content("USER_CONTENT", cleaned)
        return f"{self.prompt_template}\n\n{wrapped}"

    async def run(self, data: str) -> StageResult[NormalizedContent]:
        cleaned = self._preprocess_html(data)
        if not cleaned.strip():
            raise StageError("Input HTML contains no extractable text content.")
        raw_prompt = self.build_prompt(data)
        parsed, usage = await self._call_llm(raw_prompt)
        validated = self._validate(parsed)
        return StageResult(output=validated, usage=usage.to_stats())
