"""Markdown file I/O for novel content, organized by novel/volume/chapter."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class NovelStore:
    """Manages novel content stored as Markdown files under data/novels/.

    Directory layout:
        data/novels/{novel}/
            meta.json
            total_outline.json
            volumes/
                vol_001/
                    outline.json
                    synopsis.txt
                    chapters/
                        ch_001.md
                        ch_002.md
                        ...
    """

    def __init__(self, base_path: str = "data/novels") -> None:
        self._base = Path(base_path)

    # ------------------------------------------------------------------
    # Novel lifecycle
    # ------------------------------------------------------------------

    def create_novel(self, name: str, meta: dict[str, Any]) -> None:
        """Create a new novel directory and write meta.json.

        *name* is the directory-safe novel identifier.
        *meta* should contain at least: title, author_style, genre, created_at.
        """
        novel_dir = self._base / name
        novel_dir.mkdir(parents=True, exist_ok=True)

        default_meta: dict[str, Any] = {
            "title": name,
            "author_style": "",
            "genre": "",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        default_meta.update(meta)
        self._write_json(novel_dir / "meta.json", default_meta)

    def novel_exists(self, name: str) -> bool:
        """Return True if the novel directory exists."""
        return (self._base / name).is_dir()

    def list_novels(self) -> list[str]:
        """Return a sorted list of all novel names."""
        if not self._base.is_dir():
            return []
        return sorted(
            entry.name
            for entry in self._base.iterdir()
            if entry.is_dir()
        )

    def get_volume_count(self, novel_name: str) -> int:
        """Return the number of volumes for a novel."""
        volumes_dir = self._base / novel_name / "volumes"
        if not volumes_dir.is_dir():
            return 0
        return sum(
            1
            for entry in volumes_dir.iterdir()
            if entry.is_dir() and entry.name.startswith("vol_")
        )

    # ------------------------------------------------------------------
    # Chapter I/O
    # ------------------------------------------------------------------

    def save_chapter(
        self,
        novel_name: str,
        volume_num: int,
        chapter_num: int,
        content: str,
        title: str = "",
    ) -> None:
        """Save a chapter as a Markdown file.

        Written to:
            data/novels/{novel}/volumes/vol_{N:03d}/chapters/ch_{M:03d}.md
        """
        chapter_dir = self._chapter_dir(novel_name, volume_num)
        chapter_dir.mkdir(parents=True, exist_ok=True)

        filepath = chapter_dir / f"ch_{chapter_num:03d}.md"
        body = content
        if title:
            body = f"# {title}\n\n{content}"
        self._write_text(filepath, body)

    def load_chapter(
        self, novel_name: str, volume_num: int, chapter_num: int
    ) -> dict[str, str]:
        """Load a chapter and return {'title': ..., 'content': ...}."""
        filepath = self._chapter_dir(novel_name, volume_num) / f"ch_{chapter_num:03d}.md"
        raw = self._read_text(filepath)

        title = ""
        content = raw
        if raw.startswith("# "):
            lines = raw.split("\n", 1)
            title = lines[0][2:].strip()
            content = lines[1].lstrip("\n") if len(lines) > 1 else ""
        return {"title": title, "content": content}

    def load_volume(
        self, novel_name: str, volume_num: int
    ) -> list[dict[str, str]]:
        """Load all chapters in a volume, sorted by chapter number.

        Returns a list of dicts with 'title' and 'content' keys.
        """
        chapter_dir = self._chapter_dir(novel_name, volume_num)
        if not chapter_dir.is_dir():
            return []

        chapters: list[dict[str, str]] = []
        for entry in sorted(chapter_dir.iterdir()):
            if entry.suffix == ".md" and entry.stem.startswith("ch_"):
                chapters.append(self.load_chapter(novel_name, volume_num,
                                                  _stem_to_num(entry.stem)))
        return chapters

    def load_last_n_chapters(
        self, novel_name: str, n: int
    ) -> list[dict[str, object]]:
        """Load the last *n* chapters across all volumes.

        Each item: {'title': str, 'content': str, 'volume': int, 'chapter': int}.
        """
        volume_count = self.get_volume_count(novel_name)
        collected: list[dict[str, object]] = []

        for vol_num in range(volume_count, 0, -1):
            chapter_dir = self._chapter_dir(novel_name, vol_num)
            if not chapter_dir.is_dir():
                continue
            chapter_files = sorted(
                entry for entry in chapter_dir.iterdir()
                if entry.suffix == ".md" and entry.stem.startswith("ch_")
            )
            for entry in reversed(chapter_files):
                ch_num = _stem_to_num(entry.stem)
                ch = self.load_chapter(novel_name, vol_num, ch_num)
                collected.append({
                    "title": ch["title"],
                    "content": ch["content"],
                    "volume": vol_num,
                    "chapter": ch_num,
                })
                if len(collected) >= n:
                    return collected
        return collected

    # ------------------------------------------------------------------
    # Volume outline I/O
    # ------------------------------------------------------------------

    def save_volume_outline(
        self,
        novel_name: str,
        volume_num: int,
        outline: dict[str, Any],
        synopsis: str,
    ) -> None:
        """Save a volume outline JSON file and its synopsis text file."""
        vol_dir = self._volume_dir(novel_name, volume_num)
        vol_dir.mkdir(parents=True, exist_ok=True)
        self._write_json(vol_dir / "outline.json", outline)
        self._write_text(vol_dir / "synopsis.txt", synopsis)

    def load_volume_outline(
        self, novel_name: str, volume_num: int
    ) -> dict[str, Any]:
        """Load a volume outline. Returns empty dict when missing."""
        filepath = self._volume_dir(novel_name, volume_num) / "outline.json"
        return self._read_json(filepath)

    # ------------------------------------------------------------------
    # Total outline I/O
    # ------------------------------------------------------------------

    def save_total_outline(self, novel_name: str, outline: dict[str, Any]) -> None:
        """Save the total outline JSON for a novel."""
        novel_dir = self._base / novel_name
        novel_dir.mkdir(parents=True, exist_ok=True)
        self._write_json(novel_dir / "total_outline.json", outline)

    def load_total_outline(self, novel_name: str) -> dict[str, Any]:
        """Load the total outline. Returns empty dict when missing."""
        filepath = self._base / novel_name / "total_outline.json"
        return self._read_json(filepath)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _volume_dir(self, novel_name: str, volume_num: int) -> Path:
        return self._base / novel_name / "volumes" / f"vol_{volume_num:03d}"

    def _chapter_dir(self, novel_name: str, volume_num: int) -> Path:
        return self._volume_dir(novel_name, volume_num) / "chapters"

    @staticmethod
    def _write_text(filepath: Path, text: str) -> None:
        filepath.write_text(text, encoding="utf-8")

    @staticmethod
    def _read_text(filepath: Path) -> str:
        if not filepath.is_file():
            raise FileNotFoundError(f"File not found: {filepath}")
        return filepath.read_text(encoding="utf-8")

    @staticmethod
    def _write_json(filepath: Path, data: Any) -> None:
        filepath.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _read_json(filepath: Path) -> dict[str, Any]:
        if not filepath.is_file():
            return {}
        return json.loads(filepath.read_text(encoding="utf-8"))


# ------------------------------------------------------------------
# Utility
# ------------------------------------------------------------------

def _stem_to_num(stem: str) -> int:
    """Extract the numeric suffix from a stem like 'ch_005'."""
    parts = stem.split("_", 1)
    return int(parts[1]) if len(parts) > 1 else 0
