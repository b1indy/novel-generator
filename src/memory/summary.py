"""Volume summary generation via LLM.

Provides SummaryGenerator, which uses an LLM to produce structured volume
summaries and compact multi-volume context summaries.
"""

import json
import logging
from typing import Any

from ..llm.client import LLMClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SummaryGenerator
# ---------------------------------------------------------------------------


class SummaryGenerator:
    """Generates volume summaries and compact context strings using an LLM.

    Does not own a :class:`~storage.table_store.TableStore` directly; it
    only requires an :class:`~llm.client.LLMClient` for generation.  The
    caller is responsible for persisting generated summaries.
    """

    def __init__(self, llm_client: LLMClient) -> None:
        """*llm_client* is used for all generation calls."""
        self._llm = llm_client

    # ------------------------------------------------------------------
    # Volume summary
    # ------------------------------------------------------------------

    def generate_volume_summary(
        self,
        volume_num: int,
        volume_title: str,
        chapters: list[dict[str, str]],
        llm_client: LLMClient,
    ) -> dict[str, Any]:
        """Generate a structured volume summary from the full chapter list.

        Args:
            volume_num: The volume number (1-indexed).
            volume_title: Human-readable volume title.
            chapters: List of dicts with ``title`` and ``content`` keys.
            llm_client: The LLM client to use for generation.

        Returns:
            A dict with keys: ``volume``, ``title``, ``chapter_count``,
            ``summary``, ``key_events``, ``character_changes``,
            ``unresolved_foreshadowing``, ``ending_state``.

        Raises:
            ValueError: If the LLM response cannot be parsed as JSON.
        """
        chapter_text = self._format_chapters(chapters)
        prompt = _VOLUME_SUMMARY_PROMPT.format(
            volume_num=volume_num,
            volume_title=volume_title,
            chapter_count=len(chapters),
            chapters=chapter_text,
        )

        messages: list[dict[str, str]] = [
            {"role": "system", "content": _VOLUME_SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        # Retry on empty/malformed LLM responses (up to 3 retries).
        summary: dict[str, Any] = {}
        for attempt in range(4):
            response = llm_client.chat(messages, temperature=0.4)
            try:
                summary = self._parse_summary_response(response)
                break
            except ValueError:
                if attempt < 3:
                    logger.warning("Summary parse failed, retry %d/3", attempt + 1)
                    messages.append({"role": "assistant", "content": response})
                    messages.append({"role": "user", "content": "请重新输出JSON格式的卷小结，确保是有效JSON。"})
                else:
                    raise

        # Ensure required top-level fields are present.
        summary.setdefault("volume", volume_num)
        summary.setdefault("title", volume_title)
        summary.setdefault("chapter_count", len(chapters))

        logger.info("Generated summary for volume %d: %s", volume_num, volume_title)
        return summary

    # ------------------------------------------------------------------
    # Compact summary
    # ------------------------------------------------------------------

    def generate_compact_summary(self, volume_summary: dict[str, Any]) -> str:
        """Compress a volume summary dict into a ~200-character string
        suitable for multi-volume context injection.

        Uses a lightweight approach: combines the summary text and key
        events into a single paragraph, then truncates.  For non-trivial
        compression the caller can optionally use the LLM-based
        :meth:`generate_compact_summary_llm`.
        """
        parts: list[str] = []

        text = volume_summary.get("summary", "")
        if text:
            parts.append(text)

        key_events = volume_summary.get("key_events", [])
        if key_events:
            events_str = "Key events: " + "; ".join(key_events)
            parts.append(events_str)

        character_changes = volume_summary.get("character_changes", {})
        if character_changes:
            if isinstance(character_changes, dict):
                changes_str = "Character changes: " + "; ".join(
                    f"{name}: {change}"
                    for name, change in character_changes.items()
                )
            else:
                changes_str = "Character changes: " + str(character_changes)
            parts.append(changes_str)

        combined = " | ".join(parts)

        # Truncate at roughly 200 characters, trying to break at a sentence
        # boundary or space.
        max_chars = 200
        if len(combined) <= max_chars:
            return combined

        truncated = combined[:max_chars]
        # Try to break at the last period, question mark, or exclamation mark.
        for punct in ("。", ".", "！", "!", "？", "?"):
            last = truncated.rfind(punct)
            if last > max_chars // 2:
                return truncated[: last + 1]

        # Fall back to the last space.
        last_space = truncated.rfind(" ")
        if last_space > max_chars // 2:
            return truncated[:last_space] + "..."

        return truncated + "..."

    def generate_compact_summary_llm(
        self,
        volume_summary: dict[str, Any],
    ) -> str:
        """Use an LLM to produce a ~200-character compressed version of
        *volume_summary*.

        This is more expensive but produces a higher-quality compaction
        than the deterministic :meth:`generate_compact_summary`.
        """
        summary_json = json.dumps(volume_summary, ensure_ascii=False, indent=2)
        prompt = _COMPACT_SUMMARY_PROMPT.format(summary_json=summary_json)

        messages: list[dict[str, str]] = [
            {"role": "system", "content": _COMPACT_SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        response = self._llm.chat(messages, temperature=0.3, max_tokens=300)
        return response.strip()

    # ------------------------------------------------------------------
    # Multi-volume context
    # ------------------------------------------------------------------

    def generate_multi_volume_context(
        self,
        summaries: list[dict[str, Any]],
    ) -> str:
        """Produce a combined context string from a list of volume summaries
        for injection into a later volume's writing prompt.

        Each summary is compacted first, then concatenated with volume
        number labels.
        """
        if not summaries:
            return "(no prior volumes)"

        lines: list[str] = []
        for summary in summaries:
            vol = summary.get("volume", "?")
            title = summary.get("title", "")
            compact = self.generate_compact_summary(summary)
            lines.append(f"[Vol {vol}] {title}: {compact}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_chapters(chapters: list[dict[str, str]]) -> str:
        """Format a list of chapter dicts into a single text block for the
        summary prompt."""
        parts: list[str] = []
        for i, ch in enumerate(chapters, 1):
            title = ch.get("title", f"Chapter {i}")
            content = ch.get("content", "")
            parts.append(f"--- Chapter {i}: {title} ---\n{content}")
        return "\n\n".join(parts)

    @staticmethod
    def _parse_summary_response(response: str) -> dict[str, Any]:
        """Parse the LLM's JSON response into a summary dict.

        Raises :class:`ValueError` on parse failure.
        """
        text = response.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines)

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Failed to parse volume summary response as JSON: {exc}"
            ) from exc

        if not isinstance(data, dict):
            raise ValueError(
                f"Expected JSON object for volume summary, got {type(data).__name__}"
            )

        return data


# ---------------------------------------------------------------------------
# LLM prompts
# ---------------------------------------------------------------------------

_VOLUME_SUMMARY_SYSTEM_PROMPT = """\
You are a careful literary analyst specializing in Chinese web novels.
Your task is to read the full text of a completed volume and produce a
structured summary in JSON format.

Output ONLY valid JSON (no markdown fences, no commentary). The JSON must
be an object with exactly these keys:

{
  "summary": "<~500 character plot summary>",
  "key_events": ["<event 1>", "<event 2>", ...],
  "character_changes": {"<character name>": "<description of change>", ...},
  "unresolved_foreshadowing": ["<description of unresolved clue>", ...],
  "ending_state": "<brief description of where things stand at volume end>"
}

Keep the summary field around 500 characters. Include only the most
important plot developments, character arc moments, and unresolved threads."""

_VOLUME_SUMMARY_PROMPT = """\
Please generate a structured summary for Volume {volume_num}: "{volume_title}"
which contains {chapter_count} chapters.

=== FULL VOLUME TEXT ===
{chapters}

=== INSTRUCTIONS ===
Output a JSON object with the following fields:
- summary: ~500 character plot summary
- key_events: list of major events in order
- character_changes: object mapping character names to what changed
- unresolved_foreshadowing: list of clues/threads still open
- ending_state: where things stand at the end of this volume"""

_COMPACT_SUMMARY_SYSTEM_PROMPT = """\
You are a text compression specialist. Given a JSON volume summary, produce
a single compact paragraph (about 200 characters) that captures the essential
plot, key events, and ending state. Output ONLY the compressed text — no
JSON, no commentary."""

_COMPACT_SUMMARY_PROMPT = """\
Compress the following volume summary into a single paragraph of about 200
characters. Focus on the most important plot points and how the volume ends.

=== VOLUME SUMMARY ===
{summary_json}

=== INSTRUCTIONS ===
Output ONLY the compressed text (no JSON, no commentary)."""
