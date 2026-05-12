"""Foreshadowing tracking with status management and consistency checks.

Provides ForeshadowingManager, which wraps TableStore for foreshadowing CRUD,
resolution tracking, priority filtering, LLM-driven extraction, prompt-context
generation, and basic consistency validation.
"""

import json
import logging
from typing import Any

from ..storage.table_store import TableStore
from ..llm.client import LLMClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_FORESHADOWING_FIELDS: dict[str, Any] = {
    "description": "",
    "planted_in": "",
    "hint_detail": "",
    "status": "pending",  # pending | resolved
    "resolved_in": None,
    "resolution_note": "",
    "related_characters": [],
    "planned_resolution": "",
    "priority": "medium",  # high | medium | low
}


# ---------------------------------------------------------------------------
# ForeshadowingManager
# ---------------------------------------------------------------------------


class ForeshadowingManager:
    """Manages the foreshadowing / planted-flag table for a single novel.

    Wraps :class:`~storage.table_store.TableStore` to provide
    foreshadowing-specific operations: plant, resolve, filter by priority,
    LLM-driven extraction, prompt-context generation, and basic
    consistency checks against characters and items tables.
    """

    _ID_PREFIX: str = "fh"

    def __init__(self, novel_path: str) -> None:
        """*novel_path* is the novel root directory (e.g. ``data/novels/my-novel``)."""
        self._store = TableStore(novel_path)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add_foreshadowing(
        self,
        description: str,
        planted_in: str,
        **kwargs: Any,
    ) -> str:
        """Plant a new foreshadowing entry and return its auto-generated ID.

        *description* — what the foreshadowing is.
        *planted_in* — where it was planted (e.g. ``"vol_001/ch_003"``).

        Additional keyword arguments (``hint_detail``, ``related_characters``,
        ``planned_resolution``, ``priority``) are merged into the record.
        """
        foreshadowing = self._store.load_foreshadowing()
        fh_id = self._store.get_next_id(self._ID_PREFIX, foreshadowing)

        record: dict[str, Any] = dict(_DEFAULT_FORESHADOWING_FIELDS)
        record["description"] = description
        record["planted_in"] = planted_in
        record.update(kwargs)
        foreshadowing[fh_id] = record
        self._store.save_foreshadowing(foreshadowing)

        logger.debug("Planted foreshadowing %s (id=%s)", description[:40], fh_id)
        return fh_id

    def resolve_foreshadowing(
        self,
        fh_id: str,
        resolved_in: str,
        resolution_note: str = "",
    ) -> None:
        """Mark a foreshadowing entry as resolved.

        *resolved_in* — where it was resolved (e.g. ``"vol_003/ch_007"``).
        *resolution_note* — brief description of how it was resolved.
        """
        foreshadowing = self._store.load_foreshadowing()
        if fh_id not in foreshadowing:
            raise KeyError(f"Foreshadowing not found: {fh_id}")

        foreshadowing[fh_id]["status"] = "resolved"
        foreshadowing[fh_id]["resolved_in"] = resolved_in
        foreshadowing[fh_id]["resolution_note"] = resolution_note
        self._store.save_foreshadowing(foreshadowing)

        logger.debug("Resolved foreshadowing %s in %s", fh_id, resolved_in)

    def get_entry(self, fh_id: str) -> dict[str, Any]:
        """Return a single foreshadowing entry by ID.

        Raises :class:`KeyError` if not found.
        """
        foreshadowing = self._store.load_foreshadowing()
        if fh_id not in foreshadowing:
            raise KeyError(f"Foreshadowing not found: {fh_id}")
        return foreshadowing[fh_id]

    def get_pending(self) -> dict[str, dict[str, Any]]:
        """Return all unresolved (``status == "pending"``) foreshadowing entries."""
        foreshadowing = self._store.load_foreshadowing()
        return {
            fid: rec
            for fid, rec in foreshadowing.items()
            if rec.get("status") == "pending"
        }

    def get_by_priority(self, priority: str) -> dict[str, dict[str, Any]]:
        """Return entries filtered by priority (``"high"``, ``"medium"``, ``"low"``).

        Raises :class:`ValueError` for an invalid priority value.
        """
        valid = frozenset({"high", "medium", "low"})
        if priority not in valid:
            raise ValueError(
                f"Invalid priority {priority!r}; must be one of {sorted(valid)}"
            )
        foreshadowing = self._store.load_foreshadowing()
        return {
            fid: rec
            for fid, rec in foreshadowing.items()
            if rec.get("priority") == priority
        }

    def get_all(self) -> dict[str, dict[str, Any]]:
        """Return the full foreshadowing table."""
        return self._store.load_foreshadowing()

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def extract_from_chapter(
        self,
        chapter_content: str,
        llm_client: LLMClient,
    ) -> list[dict[str, Any]]:
        """Use an LLM to identify new foreshadowing planted and any existing
        foreshadowing resolved in *chapter_content*.

        Returns a list of change dicts with ``action`` set to ``"plant"``
        or ``"resolve"``.
        """
        existing = self.get_all()
        existing_context = self._existing_foreshadowing_context(existing)

        prompt = _FORESHADOWING_EXTRACTION_PROMPT.format(
            existing_foreshadowing=existing_context or "(none yet)",
            chapter_content=chapter_content,
        )

        messages: list[dict[str, str]] = [
            {"role": "system", "content": _FORESHADOWING_EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        try:
            response = llm_client.chat(messages, temperature=0.3)
            return self._parse_extraction_response(response)
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("Failed to parse foreshadowing extraction response: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Prompt context
    # ------------------------------------------------------------------

    def prompt_context(self) -> str:
        """Generate a compact string of pending foreshadowing for injection
        into writing prompts.

        Each entry includes ``hint_detail`` to remind the writer of the
        subtle clue that was planted.
        """
        pending = self.get_pending()
        if not pending:
            return "(no pending foreshadowing)"

        lines: list[str] = []
        for fh_id, rec in pending.items():
            priority = rec.get("priority", "medium")
            desc = rec.get("description", fh_id)
            hint = rec.get("hint_detail", "")
            planted = rec.get("planted_in", "?")
            planned = rec.get("planned_resolution", "")
            related = rec.get("related_characters", [])

            parts = [f"[{fh_id}] [{priority.upper()}] {desc}"]
            parts.append(f"  planted: {planted}")
            if hint:
                parts.append(f"  hint: {hint}")
            if related:
                parts.append(f"  characters: {', '.join(related)}")
            if planned:
                parts.append(f"  planned resolution: {planned}")

            lines.extend(parts)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Consistency check
    # ------------------------------------------------------------------

    def check_consistency(
        self,
        characters: dict[str, dict[str, Any]],
        items: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Run basic consistency checks against the character and item tables.

        Returns a list of issue dicts, each with ``type``, ``severity``,
        ``foreshadowing_id``, and ``description`` keys.

        Checks performed:

        * Foreshadowing referencing dead or departed characters
        * Resolved foreshadowing that still has ``status == "pending"``
          (this cannot happen through our API, but manual JSON edits could
          create the condition)
        * High-priority foreshadowing without a ``planned_resolution``
        """
        issues: list[dict[str, Any]] = []
        all_fh = self.get_all()

        # Collect dead/departed character names for reference checks.
        dead_or_gone: set[str] = set()
        for rec in characters.values():
            if rec.get("status") in ("dead", "departed"):
                dead_or_gone.add(rec.get("name", ""))
            # Also collect aliases.
            for alias in rec.get("aliases", []):
                dead_or_gone.add(alias)
        dead_or_gone.discard("")

        for fh_id, rec in all_fh.items():
            # Check: high priority without planned resolution.
            if rec.get("priority") == "high" and rec.get("status") == "pending":
                if not rec.get("planned_resolution"):
                    issues.append({
                        "type": "missing_plan",
                        "severity": "minor",
                        "foreshadowing_id": fh_id,
                        "description": (
                            f"High-priority foreshadowing {fh_id} has no "
                            f"planned_resolution set."
                        ),
                    })

            # Check: references to dead/departed characters in pending entries.
            if rec.get("status") == "pending":
                related = rec.get("related_characters", [])
                for char_name in related:
                    if char_name in dead_or_gone:
                        issues.append({
                            "type": "dead_character_reference",
                            "severity": "major",
                            "foreshadowing_id": fh_id,
                            "description": (
                                f"Pending foreshadowing {fh_id} references "
                                f"dead/departed character {char_name!r}."
                            ),
                        })

        return issues

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _existing_foreshadowing_context(
        foreshadowing: dict[str, dict[str, Any]],
    ) -> str:
        """Build a compact text representation of existing foreshadowing for
        the extraction prompt."""
        if not foreshadowing:
            return ""

        lines: list[str] = []
        for fh_id, rec in foreshadowing.items():
            status = rec.get("status", "pending")
            priority = rec.get("priority", "medium")
            desc = rec.get("description", fh_id)
            planted = rec.get("planted_in", "?")
            resolved = rec.get("resolved_in", "")

            parts = [f"- [{fh_id}] ({status}, {priority}) {desc}"]
            parts.append(f"  planted: {planted}")
            if resolved:
                parts.append(f"  resolved: {resolved}")

            lines.extend(parts)

        return "\n".join(lines)

    @staticmethod
    def _parse_extraction_response(response: str) -> list[dict[str, Any]]:
        """Parse the LLM's JSON response into a list of change dicts."""
        text = response.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines)

        data = json.loads(text)
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            return []
        return data


# ---------------------------------------------------------------------------
# LLM prompts
# ---------------------------------------------------------------------------

_FORESHADOWING_EXTRACTION_SYSTEM_PROMPT = """\
You are a careful reader and analyst of web-novel chapters written in Chinese.
Your task is to identify foreshadowing (伏笔) — clues, hints, or setups that
will pay off later — in the chapter text.

Output ONLY valid JSON (no markdown fences, no commentary). The JSON must be a
list of objects, each with an "action" field:

- action: "plant" — new foreshadowing is planted in this chapter
- action: "resolve" — a previously planted foreshadowing is resolved here

For "plant" entries, include:
  "action": "plant",
  "description": <string — what is being set up>,
  "hint_detail": <string — the subtle clue the reader sees>,
  "related_characters": [<string>, ...],
  "planned_resolution": <string or null — where you think this might pay off>,
  "priority": <"high" | "medium" | "low">

For "resolve" entries, include:
  "action": "resolve",
  "fh_id": <existing foreshadowing id, e.g. "fh_001">,
  "resolution_note": <string — how it was resolved>

Only report what the chapter text clearly shows. Do not invent foreshadowing
that is not actually present in the text."""

_FORESHADOWING_EXTRACTION_PROMPT = """\
Below is a list of currently tracked foreshadowing and a chapter of the novel.
Identify any new foreshadowing planted in this chapter AND any previously
planted foreshadowing that gets resolved here.

=== EXISTING FORESHADOWING ===
{existing_foreshadowing}

=== CHAPTER CONTENT ===
{chapter_content}

=== INSTRUCTIONS ===
Output a JSON list of changes. Use action "plant" for new foreshadowing and
action "resolve" for existing foreshadowing that gets resolved. Reference
existing entries by their fh_id. Only report what the chapter text clearly
shows."""
