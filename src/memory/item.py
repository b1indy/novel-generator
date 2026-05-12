"""Item tracking table with auto-extraction from chapter content.

Provides ItemManager, which wraps TableStore for item CRUD, ownership
transfer, LLM-driven extraction, and compact prompt-context generation.
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

_DEFAULT_ITEM_FIELDS: dict[str, Any] = {
    "name": "",
    "type": "",  # weapon | armor | consumable | treasure | material | other
    "grade": "",  # 凡品 | 灵器 | 宝器 | 仙器 | 神器 | ...
    "owner": "",
    "status": "in_use",  # in_use | stored | destroyed | lost
    "first_appearance": "",
    "significance": "",  # main_plot | side_plot | character_signature | background
    "notes": "",
}


# ---------------------------------------------------------------------------
# ItemManager
# ---------------------------------------------------------------------------


class ItemManager:
    """Manages the item state table for a single novel.

    Wraps :class:`~storage.table_store.TableStore` to provide item-specific
    operations: add, update, transfer, LLM-driven extraction, and compact
    prompt-context generation.
    """

    _ID_PREFIX: str = "item"

    def __init__(self, novel_path: str) -> None:
        """*novel_path* is the novel root directory (e.g. ``data/novels/my-novel``)."""
        self._store = TableStore(novel_path)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add_item(self, name: str, **kwargs: Any) -> str:
        """Add a new item and return its auto-generated ID.

        Keyword arguments are merged into the item record
        (e.g. ``type``, ``grade``, ``owner``, ``significance``).
        """
        items = self._store.load_items()
        item_id = self._store.get_next_id(self._ID_PREFIX, items)

        record: dict[str, Any] = dict(_DEFAULT_ITEM_FIELDS)
        record["name"] = name
        record.update(kwargs)
        items[item_id] = record
        self._store.save_items(items)

        logger.debug("Added item %r (id=%s)", name, item_id)
        return item_id

    def update_item(self, item_id: str, **kwargs: Any) -> None:
        """Update fields on an existing item.

        Raises :class:`KeyError` if the item does not exist.
        """
        items = self._store.load_items()
        if item_id not in items:
            raise KeyError(f"Item not found: {item_id}")

        items[item_id].update(kwargs)
        self._store.save_items(items)

    def get_item(self, item_id: str) -> dict[str, Any]:
        """Return a single item record by ID.

        Raises :class:`KeyError` if the item does not exist.
        """
        items = self._store.load_items()
        if item_id not in items:
            raise KeyError(f"Item not found: {item_id}")
        return items[item_id]

    def get_active_items(self) -> dict[str, dict[str, Any]]:
        """Return items whose status is neither ``"destroyed"`` nor ``"lost"``."""
        items = self._store.load_items()
        return {
            iid: rec
            for iid, rec in items.items()
            if rec.get("status") not in ("destroyed", "lost")
        }

    def get_all_items(self) -> dict[str, dict[str, Any]]:
        """Return the full items table."""
        return self._store.load_items()

    # ------------------------------------------------------------------
    # Ownership
    # ------------------------------------------------------------------

    def transfer_item(self, item_id: str, new_owner: str) -> None:
        """Change the owner of an item to *new_owner*."""
        item = self.get_item(item_id)
        old_owner = item.get("owner", "(none)")
        self.update_item(item_id, owner=new_owner)
        logger.debug(
            "Transferred item %r from %s to %s", item.get("name", item_id), old_owner, new_owner
        )

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def extract_from_chapter(
        self,
        chapter_content: str,
        llm_client: LLMClient,
    ) -> list[dict[str, Any]]:
        """Use an LLM to extract new and changed items from *chapter_content*.

        Returns a list of change dicts, each with an ``action`` key
        (``"new"``, ``"update"``, or ``"transfer"``) and the relevant fields.

        The LLM is asked to identify:

        * New items introduced (name, type, grade, owner, significance)
        * Item status changes (e.g. destroyed, lost)
        * Ownership transfers
        * Other notable updates
        """
        existing = self.get_all_items()
        existing_context = self._existing_items_context(existing)

        prompt = _ITEM_EXTRACTION_PROMPT.format(
            existing_items=existing_context or "(none yet)",
            chapter_content=chapter_content,
        )

        messages: list[dict[str, str]] = [
            {"role": "system", "content": _ITEM_EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        try:
            response = llm_client.chat(messages, temperature=0.3)
            return self._parse_extraction_response(response)
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("Failed to parse item extraction response: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Prompt context
    # ------------------------------------------------------------------

    def prompt_context(self) -> str:
        """Generate a compact string describing active items for injection
        into writing prompts.

        Format::

             [name] (type, grade) owner: ... | significance: ...
        """
        active = self.get_active_items()
        if not active:
            return "(no active items)"

        lines: list[str] = []
        for item_id, rec in active.items():
            parts: list[str] = []
            parts.append(f"[{rec.get('name', item_id)}]")

            extra: list[str] = []
            item_type = rec.get("type", "")
            if item_type:
                extra.append(item_type)
            grade = rec.get("grade", "")
            if grade:
                extra.append(grade)
            status = rec.get("status", "in_use")
            if status != "in_use":
                extra.append(status)
            if extra:
                parts.append(f"({', '.join(extra)})")

            owner = rec.get("owner", "")
            if owner:
                parts.append(f"owner: {owner}")

            significance = rec.get("significance", "")
            if significance:
                parts.append(f"significance: {significance}")

            lines.append(" ".join(parts))

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _existing_items_context(items: dict[str, dict[str, Any]]) -> str:
        """Build a compact text representation of existing items for the
        extraction prompt."""
        if not items:
            return ""

        lines: list[str] = []
        for item_id, rec in items.items():
            name = rec.get("name", item_id)
            item_type = rec.get("type", "")
            grade = rec.get("grade", "")
            owner = rec.get("owner", "")
            status = rec.get("status", "in_use")
            significance = rec.get("significance", "")

            parts = [f"- {name} (id={item_id}, type={item_type}, status={status})"]
            if grade:
                parts.append(f"  grade: {grade}")
            if owner:
                parts.append(f"  owner: {owner}")
            if significance:
                parts.append(f"  significance: {significance}")

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

_ITEM_EXTRACTION_SYSTEM_PROMPT = """\
You are a careful reader and analyst of web-novel chapters written in Chinese.
Your task is to extract item/equipment/treasure information from the chapter
text and output a structured JSON list of changes.

Output ONLY valid JSON (no markdown fences, no commentary). The JSON must be a
list of objects, each with an "action" field:

- action: "new" — a previously unseen item is introduced
- action: "update" — an existing item's properties change
- action: "transfer" — an item changes owner

For "new" items, include:
  "action": "new",
  "name": <string>,
  "type": <one of: weapon, armor, consumable, treasure, material, other>,
  "grade": <string or null>,
  "owner": <string or null>,
  "status": <one of: in_use, stored>,
  "significance": <one of: main_plot, side_plot, character_signature, background>,
  "notes": <string or null>

For "update" items, include:
  "action": "update",
  "item_id": <existing item id, e.g. "item_001">,
  "changes": {{"field_name": <new_value>, ...}}

For "transfer" items, include:
  "action": "transfer",
  "item_id": <existing item id>,
  "new_owner": <string>

Only report changes that are clearly evidenced by the chapter text."""

_ITEM_EXTRACTION_PROMPT = """\
Below is a list of currently known items and a chapter of the novel.
Identify any new items introduced, item status changes, and ownership
transfers.

=== EXISTING ITEMS ===
{existing_items}

=== CHAPTER CONTENT ===
{chapter_content}

=== INSTRUCTIONS ===
Output a JSON list of changes. Use action "new" for new items, "update"
for status/attribute changes, and "transfer" for ownership changes.
Only report what the chapter text clearly shows."""
