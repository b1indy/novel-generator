"""Outline management CRUD with automatic context gathering and versioning.

Orchestrates NovelStore (file I/O) and OutlineGenerator (LLM calls) to
provide a high-level API for outline lifecycle management.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..config.style import StyleManager
from ..llm.client import LLMClient
from ..storage.novel_store import NovelStore
from ..storage.table_store import TableStore
from .generator import OutlineGenerator

logger = logging.getLogger(__name__)


class OutlineManager:
    """High-level API for outline CRUD and context gathering.

    Wraps :class:`NovelStore` for persistence and :class:`OutlineGenerator`
    for LLM-driven outline creation. Automatically gathers cross-volume
    context (summaries, last chapters, memory tables) so callers do not
    need to manually assemble it.

    Usage::

        llm = LLMClient("config.yaml")
        store = NovelStore("data/novels")
        style = StyleManager()
        gen = OutlineGenerator(llm, style)
        mgr = OutlineManager(store, gen, style, "data/novels")
        mgr.create_total_outline("my-novel", "星辰变", "玄幻", "辰东式热血", 2000000)
        ctx = mgr.get_context_for_volume("my-novel", 2)
    """

    def __init__(
        self,
        novel_store: NovelStore,
        outline_generator: OutlineGenerator,
        style_manager: StyleManager,
        base_path: str = "data/novels",
    ) -> None:
        """Initialize the outline manager.

        Args:
            novel_store: NovelStore instance for file I/O.
            outline_generator: OutlineGenerator instance for LLM calls.
            style_manager: StyleManager for style metadata lookups.
            base_path: Root path where novel directories are stored.
        """
        self._store = novel_store
        self._generator = outline_generator
        self._style = style_manager
        self._base = Path(base_path)

    # ------------------------------------------------------------------
    # Total outline
    # ------------------------------------------------------------------

    def create_total_outline(
        self,
        novel_name: str,
        title: str,
        genre: str,
        style_name: str,
        target_words: int,
    ) -> dict[str, Any]:
        """Generate and persist the total outline for a novel.

        Args:
            novel_name: Directory-safe novel identifier.
            title: Display title of the novel.
            genre: Genre name (e.g. "玄幻").
            style_name: Author style name (e.g. "辰东式热血").
            target_words: Target total word count.

        Returns:
            The generated total outline dict.

        Raises:
            ValueError: If the LLM response cannot be parsed.
        """
        outline = self._generator.generate_total_outline(
            novel_title=title,
            genre=genre,
            style_name=style_name,
            target_words=target_words,
        )

        # Attach metadata for later lookups.
        outline["_meta"] = {
            "title": title,
            "genre": genre,
            "style_name": style_name,
            "target_words": target_words,
            "novel_name": novel_name,
        }

        self._store.save_total_outline(novel_name, outline)
        logger.info("Total outline saved for novel '%s'", novel_name)

        return outline

    def update_total_outline(
        self,
        novel_name: str,
        feedback: str,
        change_request: str = "",
    ) -> dict[str, Any]:
        """Load, optimize, and re-save the total outline.

        Args:
            novel_name: Novel identifier.
            feedback: Human-readable reason for the change.
            change_request: Specific description of what to change.

        Returns:
            The updated total outline dict.

        Raises:
            FileNotFoundError: If the total outline does not exist.
            ValueError: If the LLM response cannot be parsed.
        """
        current = self._store.load_total_outline(novel_name)
        if not current:
            raise FileNotFoundError(
                f"Total outline not found for novel '{novel_name}'"
            )

        updated = self._generator.optimize_outline(
            current_outline=current,
            feedback=feedback,
            change_request=change_request,
        )

        self._store.save_total_outline(novel_name, updated)
        logger.info("Total outline updated for novel '%s'", novel_name)

        return updated

    def get_total_outline(self, novel_name: str) -> dict[str, Any]:
        """Load the total outline for a novel.

        Args:
            novel_name: Novel identifier.

        Returns:
            The total outline dict, or an empty dict if not found.
        """
        return self._store.load_total_outline(novel_name)

    # ------------------------------------------------------------------
    # Volume outline
    # ------------------------------------------------------------------

    def create_volume_outline(
        self,
        novel_name: str,
        volume_num: int,
    ) -> dict[str, Any]:
        """Generate and persist the outline for a specific volume.

        Automatically gathers all necessary context:
        - Total outline
        - Previous volume summaries (volumes 1..N-1)
        - Last 2 chapters of volume N-1
        - Current memory tables

        Args:
            novel_name: Novel identifier.
            volume_num: Volume number to generate (1-indexed).

        Returns:
            The generated volume outline dict.

        Raises:
            FileNotFoundError: If the total outline does not exist.
            ValueError: If the LLM response cannot be parsed.
        """
        context = self.get_context_for_volume(novel_name, volume_num)

        total_outline = context["total_outline"]
        prev_summaries = context["volume_summaries"]
        last_chapters = context["last_chapters"]
        memory_context = {
            "characters": context["characters"],
            "items": context["items"],
            "foreshadowing": context["foreshadowing"],
        }

        outline = self._generator.generate_volume_outline(
            total_outline=total_outline,
            volume_num=volume_num,
            prev_summaries=prev_summaries,
            last_chapters=last_chapters,
            memory_context=memory_context,
        )

        # Attach metadata.
        outline["_meta"] = {
            "novel_name": novel_name,
            "volume_num": volume_num,
            "title": outline.get("volume_title", f"第{volume_num}卷"),
        }

        synopsis = outline.get("synopsis", "")
        self._store.save_volume_outline(novel_name, volume_num, outline, synopsis)
        logger.info(
            "Volume %d outline saved for novel '%s'", volume_num, novel_name
        )

        return outline

    def get_volume_outline(
        self, novel_name: str, volume_num: int
    ) -> dict[str, Any]:
        """Load a volume outline.

        Args:
            novel_name: Novel identifier.
            volume_num: Volume number (1-indexed).

        Returns:
            The volume outline dict, or an empty dict if not found.
        """
        return self._store.load_volume_outline(novel_name, volume_num)

    # ------------------------------------------------------------------
    # Context assembly
    # ------------------------------------------------------------------

    def get_context_for_volume(
        self,
        novel_name: str,
        volume_num: int,
    ) -> dict[str, Any]:
        """Assemble all context needed for writing a volume.

        This is the central method that gathers everything the volume writer
        and chapter writer need to produce coherent content:

        - Total outline (always)
        - Volume outlines for volumes 1..N (for summary overview)
        - Volume summaries for volumes 1..N-1 (condensed prior events)
        - Last 2 chapters of volume N-1 (for narrative continuity)
        - Current memory tables (characters, items, foreshadowing)

        Args:
            novel_name: Novel identifier.
            volume_num: Target volume number (1-indexed).

        Returns:
            A dict with keys: ``total_outline``, ``volume_outlines``,
            ``volume_summaries``, ``last_chapters``, ``characters``,
            ``items``, ``foreshadowing``, ``meta``.

        Raises:
            FileNotFoundError: If the total outline does not exist.
        """
        total_outline = self._store.load_total_outline(novel_name)
        if not total_outline:
            raise FileNotFoundError(
                f"Total outline not found for novel '{novel_name}'. "
                f"Run create_total_outline first."
            )

        table_store = self._make_table_store(novel_name)

        # Volume summaries for volumes 1..N-1.
        all_summaries = table_store.load_volume_summaries()
        prev_summaries = [s for s in all_summaries if s.get("volume", 0) < volume_num]

        # Last 2 chapters of the previous volume.
        last_chapters: list[dict[str, Any]] = []
        if volume_num > 1:
            last_chapters = self._store.load_last_n_chapters(novel_name, 2)
            # Filter to only chapters from volume N-1.
            last_chapters = [
                ch for ch in last_chapters if ch.get("volume") == volume_num - 1
            ][:2]

        # Load volume outlines for volumes 1..volume_num-1.
        volume_outlines: list[dict[str, Any]] = []
        for vn in range(1, volume_num):
            vo = self._store.load_volume_outline(novel_name, vn)
            if vo:
                volume_outlines.append(vo)

        # Memory tables.
        characters = table_store.load_characters()
        items = table_store.load_items()
        foreshadowing = table_store.load_foreshadowing()
        meta = table_store.load_meta()

        context: dict[str, Any] = {
            "total_outline": total_outline,
            "volume_outlines": volume_outlines,
            "volume_summaries": prev_summaries,
            "last_chapters": last_chapters,
            "characters": characters,
            "items": items,
            "foreshadowing": foreshadowing,
            "meta": meta,
        }

        logger.info(
            "Context assembled for novel '%s' volume %d: "
            "%d prev summaries, %d last chapters, "
            "%d characters, %d items, %d foreshadowing",
            novel_name,
            volume_num,
            len(prev_summaries),
            len(last_chapters),
            len(characters),
            len(items),
            len(foreshadowing),
        )

        return context

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def get_outline_summary(self, novel_name: str) -> dict[str, Any]:
        """Return a human-readable overview of all outlines for a novel.

        Includes total outline synopsis, volume listing with titles and
        chapter counts, and overall progress info.

        Args:
            novel_name: Novel identifier.

        Returns:
            A dict with keys: ``title``, ``genre``, ``style``,
            ``target_words``, ``synopsis``, ``volumes_planned``,
            ``volumes_written``, ``volume_details``.
        """
        total = self._store.load_total_outline(novel_name)

        meta = total.get("_meta", {})
        volume_plan = total.get("volume_plan", [])
        volumes_planned = len(volume_plan)

        volume_count = self._store.get_volume_count(novel_name)

        volume_details: list[dict[str, Any]] = []
        for vp in volume_plan:
            vnum = vp.get("volume", 0)
            written = vnum <= volume_count

            # Try to load the actual volume outline for more detail.
            vol_outline: dict[str, Any] = {}
            if written:
                vol_outline = self._store.load_volume_outline(novel_name, vnum)

            detail: dict[str, Any] = {
                "volume": vnum,
                "planned_title": vp.get("title", ""),
                "planned_summary": vp.get("summary", ""),
                "estimated_chapters": vp.get("estimated_chapters", 0),
                "written": written,
                "actual_title": vol_outline.get("volume_title", ""),
                "actual_chapters": len(vol_outline.get("chapter_plan", [])),
                "synopsis": vol_outline.get("synopsis", ""),
            }
            volume_details.append(detail)

        summary: dict[str, Any] = {
            "title": meta.get("title", novel_name),
            "genre": meta.get("genre", ""),
            "style": meta.get("style_name", ""),
            "target_words": meta.get("target_words", 0),
            "synopsis": total.get("synopsis", ""),
            "volumes_planned": volumes_planned,
            "volumes_written": volume_count,
            "volume_details": volume_details,
        }

        return summary

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_table_store(self, novel_name: str) -> TableStore:
        """Create a TableStore pointed at the given novel directory.

        Args:
            novel_name: Novel identifier.

        Returns:
            Configured :class:`TableStore` instance.
        """
        novel_path = self._base / novel_name
        return TableStore(str(novel_path))
