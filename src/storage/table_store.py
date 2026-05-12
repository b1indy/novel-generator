"""JSON file I/O for structured tracking tables (characters, items, foreshadowing, etc.).

Each table is a single JSON file stored under the novel directory.
Load methods return an empty dict/list when the file does not exist so
callers can safely access data on first run.
"""

import json
from pathlib import Path
from typing import Any


class TableStore:
    """Manages structured tracking data as JSON files under a novel directory.

    The *novel_path* argument is the absolute or relative path to a novel
    root (e.g. ``data/novels/my-novel``).

    Files managed:
        characters.json     – role state table
        items.json          – item state table
        foreshadowing.json  – foreshadowing / planted-flag table
        volume_summaries.json – per-volume condensed summaries (list)
        meta.json           – novel metadata
    """

    def __init__(self, novel_path: str) -> None:
        self._root = Path(novel_path)

    # ------------------------------------------------------------------
    # Characters
    # ------------------------------------------------------------------

    def save_characters(self, characters: dict[str, dict[str, Any]]) -> None:
        """Persist the full characters table."""
        self._write_json(self._root / "characters.json", characters)

    def load_characters(self) -> dict[str, dict[str, Any]]:
        """Load characters table. Returns empty dict on first run."""
        data = self._read_json(self._root / "characters.json")
        if isinstance(data, dict):
            return data  # type: ignore[return-value]
        return {}

    # ------------------------------------------------------------------
    # Items
    # ------------------------------------------------------------------

    def save_items(self, items: dict[str, dict[str, Any]]) -> None:
        """Persist the full items table."""
        self._write_json(self._root / "items.json", items)

    def load_items(self) -> dict[str, dict[str, Any]]:
        """Load items table. Returns empty dict on first run."""
        data = self._read_json(self._root / "items.json")
        if isinstance(data, dict):
            return data  # type: ignore[return-value]
        return {}

    # ------------------------------------------------------------------
    # Foreshadowing
    # ------------------------------------------------------------------

    def save_foreshadowing(self, foreshadowing: dict[str, dict[str, Any]]) -> None:
        """Persist the full foreshadowing table."""
        self._write_json(self._root / "foreshadowing.json", foreshadowing)

    def load_foreshadowing(self) -> dict[str, dict[str, Any]]:
        """Load foreshadowing table. Returns empty dict on first run."""
        data = self._read_json(self._root / "foreshadowing.json")
        if isinstance(data, dict):
            return data  # type: ignore[return-value]
        return {}

    # ------------------------------------------------------------------
    # Volume summaries
    # ------------------------------------------------------------------

    def save_volume_summaries(self, summaries: list[dict[str, Any]]) -> None:
        """Persist the volume summaries list."""
        self._write_json(self._root / "volume_summaries.json", summaries)

    def load_volume_summaries(self) -> list[dict[str, Any]]:
        """Load volume summaries. Returns empty list on first run."""
        data = self._read_json(self._root / "volume_summaries.json")
        if isinstance(data, list):
            return data  # type: ignore[return-value]
        return []

    # ------------------------------------------------------------------
    # Meta
    # ------------------------------------------------------------------

    def save_meta(self, meta: dict[str, Any]) -> None:
        """Persist novel metadata."""
        self._write_json(self._root / "meta.json", meta)

    def load_meta(self) -> dict[str, Any]:
        """Load novel metadata. Returns empty dict on first run."""
        data = self._read_json(self._root / "meta.json")
        if isinstance(data, dict):
            return data  # type: ignore[return-value]
        return {}

    # ------------------------------------------------------------------
    # ID generation
    # ------------------------------------------------------------------

    def get_next_id(self, prefix: str, data: dict[str, Any]) -> str:
        """Generate the next sequential ID for the given prefix.

        Scans existing keys that start with ``prefix + '_'`` and returns
        ``f"{prefix}_{next:03d}"``.  Examples::

            >>> store.get_next_id("char", {"char_001": {...}})
            "char_002"
            >>> store.get_next_id("item", {})
            "item_001"
        """
        if not data:
            return f"{prefix}_001"

        max_num = 0
        key_prefix = f"{prefix}_"
        for key in data:
            if key.startswith(key_prefix):
                try:
                    num = int(key[len(key_prefix):])
                    if num > max_num:
                        max_num = num
                except ValueError:
                    pass
        return f"{prefix}_{max_num + 1:03d}"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_dir(self) -> None:
        self._root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _write_json(filepath: Path, data: Any) -> None:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _read_json(filepath: Path) -> Any:
        if not filepath.is_file():
            return None
        return json.loads(filepath.read_text(encoding="utf-8"))
