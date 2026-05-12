"""Character tracking table with auto-extraction from chapter content.

Provides CharacterManager, which wraps TableStore for character CRUD,
relationship tracking, status management, and LLM-driven extraction of
character information from chapter text.
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

_DEFAULT_CHARACTER_FIELDS: dict[str, Any] = {
    "name": "",
    "aliases": [],
    "traits": [],
    "cultivation": "",
    "relationships": {},  # {target_name: relation_description}
    "status": "alive",  # alive | dead | departed | missing
    "current_location": "",
    "first_appearance": "",
    "last_appearance": "",
    "notes": "",
}


# ---------------------------------------------------------------------------
# CharacterManager
# ---------------------------------------------------------------------------


class CharacterManager:
    """Manages the character state table for a single novel.

    Wraps :class:`~storage.table_store.TableStore` to provide
    character-specific operations: add, update, status changes,
    relationship tracking, LLM-driven extraction, and compact
    prompt-context generation.
    """

    _ID_PREFIX: str = "char"

    def __init__(self, novel_path: str) -> None:
        """*novel_path* is the novel root directory (e.g. ``data/novels/my-novel``)."""
        self._store = TableStore(novel_path)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add_character(self, name: str, **kwargs: Any) -> str:
        """Add a new character and return its auto-generated ID.

        Any keyword arguments are merged into the character record
        (e.g. ``traits``, ``cultivation``, ``status``, ``location``).
        """
        characters = self._store.load_characters()
        char_id = self._store.get_next_id(self._ID_PREFIX, characters)

        record: dict[str, Any] = dict(_DEFAULT_CHARACTER_FIELDS)
        record["name"] = name
        record.update(kwargs)
        characters[char_id] = record
        self._store.save_characters(characters)

        logger.debug("Added character %r (id=%s)", name, char_id)
        return char_id

    def update_character(self, char_id: str, **kwargs: Any) -> None:
        """Update fields on an existing character.

        Automatically sets ``last_appearance`` to the value provided (if any),
        or leaves it unchanged.  Unknown keys are silently added.
        """
        characters = self._store.load_characters()
        if char_id not in characters:
            raise KeyError(f"Character not found: {char_id}")

        characters[char_id].update(kwargs)
        self._store.save_characters(characters)

    def get_character(self, char_id: str) -> dict[str, Any]:
        """Return a single character record by ID.

        Raises :class:`KeyError` if the character does not exist.
        """
        characters = self._store.load_characters()
        if char_id not in characters:
            raise KeyError(f"Character not found: {char_id}")
        return characters[char_id]

    def get_active_characters(self) -> dict[str, dict[str, Any]]:
        """Return all characters whose status is neither ``"dead"`` nor ``"departed"``."""
        characters = self._store.load_characters()
        return {
            cid: rec
            for cid, rec in characters.items()
            if rec.get("status") not in ("dead", "departed")
        }

    def get_all_characters(self) -> dict[str, dict[str, Any]]:
        """Return the full characters table."""
        return self._store.load_characters()

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def set_status(self, char_id: str, status: str) -> None:
        """Set a character's status.

        Valid values: ``alive``, ``dead``, ``departed``, ``missing``.
        """
        valid = frozenset({"alive", "dead", "departed", "missing"})
        if status not in valid:
            raise ValueError(
                f"Invalid status {status!r}; must be one of {sorted(valid)}"
            )
        self.update_character(char_id, status=status)

    # ------------------------------------------------------------------
    # Relationships
    # ------------------------------------------------------------------

    def add_relationship(self, char_id: str, target_name: str, relation: str) -> None:
        """Record a relationship between *char_id* and *target_name*.

        The relationship is stored inside the character's ``relationships``
        dict as ``{target_name: relation}``.
        """
        character = self.get_character(char_id)
        relationships: dict[str, str] = character.get("relationships", {})
        relationships[target_name] = relation
        self.update_character(char_id, relationships=relationships)

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def extract_from_chapter(
        self,
        chapter_content: str,
        llm_client: LLMClient,
    ) -> list[dict[str, Any]]:
        """Use an LLM to extract new and updated character information
        from *chapter_content*.

        Returns a list of change dicts, each with an ``action`` key
        (``"new"`` or ``"update"``) and the relevant fields.

        The LLM is asked to identify:

        * New characters (name, traits, cultivation if applicable)
        * Status changes (e.g. alive → dead)
        * Location changes
        * New relationships
        * Other notable updates
        """
        existing = self.get_all_characters()
        existing_context = self._existing_characters_context(existing)

        prompt = _CHARACTER_EXTRACTION_PROMPT.format(
            existing_characters=existing_context or "(none yet)",
            chapter_content=chapter_content,
        )

        messages: list[dict[str, str]] = [
            {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        try:
            response = llm_client.chat(messages, temperature=0.3)
            return self._parse_extraction_response(response)
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("Failed to parse character extraction response: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Prompt context
    # ------------------------------------------------------------------

    def prompt_context(self) -> str:
        """Generate a compact string describing active characters for
        injection into writing prompts.

        Format::

             [name] (status, location) traits: ... | relationships: ...
        """
        active = self.get_active_characters()
        if not active:
            return "(no active characters)"

        lines: list[str] = []
        for char_id, rec in active.items():
            parts: list[str] = []
            parts.append(f"[{rec.get('name', char_id)}]")

            extra: list[str] = []
            status = rec.get("status", "alive")
            if status != "alive":
                extra.append(status)
            location = rec.get("current_location", "")
            if location:
                extra.append(location)
            cultivation = rec.get("cultivation", "")
            if cultivation:
                extra.append(cultivation)
            if extra:
                parts.append(f"({', '.join(extra)})")

            traits = rec.get("traits", [])
            if traits:
                parts.append(f"traits: {', '.join(traits)}")

            relationships = rec.get("relationships", {})
            if relationships:
                rel_parts = [f"{t}: {r}" for t, r in relationships.items()]
                parts.append(f"relationships: {'; '.join(rel_parts)}")

            lines.append(" ".join(parts))

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _existing_characters_context(characters: dict[str, dict[str, Any]]) -> str:
        """Build a compact text representation of existing characters for
        the extraction prompt."""
        if not characters:
            return ""

        lines: list[str] = []
        for char_id, rec in characters.items():
            name = rec.get("name", char_id)
            status = rec.get("status", "alive")
            location = rec.get("current_location", "")
            traits = rec.get("traits", [])
            cultivation = rec.get("cultivation", "")
            rels = rec.get("relationships", {})

            parts = [f"- {name} (id={char_id}, status={status})"]
            if location:
                parts.append(f"  location: {location}")
            if cultivation:
                parts.append(f"  cultivation: {cultivation}")
            if traits:
                parts.append(f"  traits: {', '.join(traits)}")
            if rels:
                rel_str = "; ".join(f"{t} -> {r}" for t, r in rels.items())
                parts.append(f"  relationships: {rel_str}")

            lines.extend(parts)

        return "\n".join(lines)

    @staticmethod
    def _parse_extraction_response(response: str) -> list[dict[str, Any]]:
        """Parse the LLM's JSON response into a list of change dicts."""
        # Strip markdown code fences if present.
        text = response.strip()
        if text.startswith("```"):
            # Remove opening fence line.
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            # Remove closing fence if present.
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines)

        data = json.loads(text)
        if isinstance(data, dict):
            # Wrap single-object responses.
            data = [data]
        if not isinstance(data, list):
            return []
        return data


# ---------------------------------------------------------------------------
# LLM prompts
# ---------------------------------------------------------------------------

_EXTRACTION_SYSTEM_PROMPT = """\
You are a careful reader and analyst of web-novel chapters written in Chinese.
Your task is to extract character information from the chapter text and output
a structured JSON list of changes.

Output ONLY valid JSON (no markdown fences, no commentary). The JSON must be a
list of objects, each with an "action" field:

- action: "new" — a previously unseen character appears
- action: "update" — an existing character's state changes

For "new" characters, include:
  "action": "new",
  "name": <string>,
  "traits": [<string>, ...],
  "cultivation": <string or null>,
  "status": "alive",
  "current_location": <string or null>,
  "notes": <string or null>,
  "relationships": {{"<target_name>": "<relation>"}}

For "update" characters, include:
  "action": "update",
  "char_id": <existing character id, e.g. "char_001">,
  "changes": {{"field_name": <new_value>, ...}}
  (possible fields: status, current_location, cultivation, traits, notes)

Only report changes that are clearly evidenced by the chapter text.
Do not invent or assume information not present in the text."""

_CHARACTER_EXTRACTION_PROMPT = """\
Below is a list of currently known characters and a chapter of the novel.
Identify any new characters introduced and any changes to existing characters.

=== EXISTING CHARACTERS ===
{existing_characters}

=== CHAPTER CONTENT ===
{chapter_content}

=== INSTRUCTIONS ===
Output a JSON list of changes. For each new character use action "new".
For each character whose state changed use action "update" with the
existing char_id. Only report what the chapter text clearly shows."""
