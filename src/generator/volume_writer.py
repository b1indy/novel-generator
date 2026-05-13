"""Volume-level orchestration for novel generation.

Provides VolumeWriter, which iterates through the chapter plan of a volume
outline, generates each chapter via ChapterWriter, saves results, updates
memory tables, generates volume summaries, and reports progress.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import yaml

from ..config.style import StyleManager
from ..llm.client import LLMClient
from ..memory.character import CharacterManager
from ..memory.foreshadowing import ForeshadowingManager
from ..memory.item import ItemManager
from ..memory.summary import SummaryGenerator
from ..outline.manager import OutlineManager
from ..storage.novel_store import NovelStore
from ..storage.table_store import TableStore
from .chapter_writer import ChapterWriter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional rich import
# ---------------------------------------------------------------------------

try:
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )

    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Warn when prompt context approaches this fraction of the 1M token window.
_TOKEN_LIMIT_WARN_RATIO: float = 0.85
_TOKEN_LIMIT_MAX: int = 1_000_000
_TOKEN_CHECK_THRESHOLD: int = int(_TOKEN_LIMIT_MAX * _TOKEN_LIMIT_WARN_RATIO)


# ---------------------------------------------------------------------------
# VolumeWriter
# ---------------------------------------------------------------------------


class VolumeWriter:
    """Orchestrates the generation of an entire volume of a novel.

    Coordinates ChapterWriter (chapter generation), memory managers
    (state tracking), NovelStore (persistence), OutlineManager (context
    assembly), and SummaryGenerator (volume summary).

    Usage::

        llm = LLMClient("config.yaml")
        style = StyleManager()
        store = NovelStore("data/novels")
        tbl = TableStore("data/novels/my-novel")
        char_mgr = CharacterManager("data/novels/my-novel")
        ... etc

        ch_writer = ChapterWriter(llm, style, char_mgr, item_mgr, fh_mgr)
        vol_writer = VolumeWriter(
            llm, ch_writer, char_mgr, item_mgr, fh_mgr,
            outline_mgr, store, tbl,
        )
        result = vol_writer.write_volume("my-novel", 1)
        print(f"Wrote {result['chapter_count']} chapters, "
              f"{result['total_words']} total words")
    """

    def __init__(
        self,
        llm_client: LLMClient,
        chapter_writer: ChapterWriter,
        character_manager: CharacterManager,
        item_manager: ItemManager,
        foreshadowing_manager: ForeshadowingManager,
        outline_manager: OutlineManager,
        novel_store: NovelStore,
        table_store: TableStore,
        summary_generator: SummaryGenerator | None = None,
        round_size: int = 15,
    ) -> None:
        """Initialise the volume writer with all required dependencies.

        Args:
            llm_client: LLM client for token counting and supplementary calls.
            chapter_writer: Configured ChapterWriter for chapter generation.
            character_manager: Character state tracker.
            item_manager: Item state tracker.
            foreshadowing_manager: Foreshadowing state tracker.
            outline_manager: OutlineManager for context assembly.
            novel_store: NovelStore for persisting chapters and outlines.
            table_store: TableStore for persisting volume summaries.
            summary_generator: Optional pre-configured SummaryGenerator.
            round_size: Chapters per round in continuous mode. Default 15.
        """
        self._llm = llm_client
        self._chapter_writer = chapter_writer
        self._characters = character_manager
        self._items = item_manager
        self._foreshadowing = foreshadowing_manager
        self._outline_manager = outline_manager
        self._novel_store = novel_store
        self._table_store = table_store
        self._summary_generator = summary_generator or SummaryGenerator(llm_client)
        self._round_size = round_size

        self._console = Console() if _HAS_RICH else None

    # ------------------------------------------------------------------
    # Volume write
    # ------------------------------------------------------------------

    def write_volume(
        self,
        novel_name: str,
        volume_num: int,
        words_per_chapter: int = 5000,
        batch_size: int = 3,
        generation_mode: str = "continuous",
    ) -> dict[str, Any]:
        """Orchestrate writing an entire volume.

        Args:
            novel_name: Directory-safe novel identifier.
            volume_num: Volume number to write (1-indexed).
            words_per_chapter: Target Chinese characters per chapter.
            batch_size: Chapters per LLM call (used in "batch" mode).
            generation_mode: Generation strategy:
                - "single": one chapter per LLM call (original)
                - "batch": batch_size chapters per LLM call
                - "continuous": multi-turn "继续" for max coherence (default)

        Returns:
            Dict with keys: ``volume``, ``title``, ``chapter_count``,
            ``total_words``, ``chapters``.
        """
        # -- Step 1: assemble context --
        context = self._outline_manager.get_context_for_volume(novel_name, volume_num)
        total_outline = context["total_outline"]

        # -- Step 2: load volume outline --
        volume_outline = self._novel_store.load_volume_outline(novel_name, volume_num)
        if not volume_outline:
            raise FileNotFoundError(
                f"Volume outline not found for novel '{novel_name}' volume {volume_num}. "
                f"Run OutlineManager.create_volume_outline first."
            )

        chapter_plan: list[dict[str, Any]] = volume_outline.get("chapter_plan", [])
        if not chapter_plan:
            raise ValueError(f"Volume {volume_num} outline has an empty chapter_plan")

        volume_title = volume_outline.get("volume_title", f"第{volume_num}卷")
        novel_title_str = total_outline.get("_meta", {}).get("title", novel_name)

        prev_chapters = context.get("last_chapters", [])
        self._info(
            f"开始生成 第{volume_num}卷「{volume_title}」"
            f"（共{len(chapter_plan)}章，每章目标{words_per_chapter}字，模式={generation_mode}）"
        )

        # -- Step 3: generate chapters based on mode --
        written_chapters: list[dict[str, Any]] = []
        recent_texts: list[str] = []
        for ch in prev_chapters:
            content = ch.get("content", "")
            if content:
                recent_texts.append(content)
        recent_texts = recent_texts[-2:]

        total_words_estimate = 0

        if generation_mode == "continuous":
            written_chapters, total_words_estimate = self._write_continuous(
                novel_name=novel_name,
                volume_num=volume_num,
                chapter_plan=chapter_plan,
                volume_outline=volume_outline,
                total_outline=total_outline,
                recent_texts=recent_texts,
                words_per_chapter=words_per_chapter,
                round_size=self._round_size,
            )
        else:
            bs = 1 if generation_mode == "single" else batch_size
            written_chapters, total_words_estimate = self._write_batched(
                novel_name=novel_name,
                volume_num=volume_num,
                chapter_plan=chapter_plan,
                volume_outline=volume_outline,
                total_outline=total_outline,
                recent_texts=recent_texts,
                words_per_chapter=words_per_chapter,
                batch_size=bs,
            )

        # -- Step 4: generate volume summary --
        self._info(f"  正在生成第{volume_num}卷总结 ...")
        summary = self._generate_and_save_summary(
            novel_name=novel_name,
            volume_num=volume_num,
            volume_title=volume_title,
            chapters=written_chapters,
        )

        # -- Step 5: return result --
        volume_result: dict[str, Any] = {
            "volume": volume_num,
            "title": volume_title,
            "chapter_count": len(written_chapters),
            "total_words": total_words_estimate,
            "chapters": written_chapters,
            "summary": summary,
        }

        self._info(
            f"第{volume_num}卷「{volume_title}」全部完成！"
            f"共{len(written_chapters)}章，约{total_words_estimate:,}字"
        )

        return volume_result

    # ------------------------------------------------------------------
    # Generation strategies
    # ------------------------------------------------------------------

    def _write_continuous(
        self,
        novel_name: str,
        volume_num: int,
        chapter_plan: list[dict[str, Any]],
        volume_outline: dict[str, Any],
        total_outline: dict[str, Any],
        recent_texts: list[str],
        words_per_chapter: int,
        round_size: int = 15,
    ) -> tuple[list[dict[str, Any]], int]:
        """Generate chapters using multi-turn continuous mode.

        Splits chapter_plan into rounds. Each round uses the "继续" pattern
        for intra-round coherence. Memory tables are updated after each round.

        Returns:
            (written_chapters, total_words_estimate)
        """
        rounds: list[list[dict[str, Any]]] = []
        for i in range(0, len(chapter_plan), round_size):
            rounds.append(chapter_plan[i : i + round_size])

        novel_title_str = total_outline.get("_meta", {}).get("title", novel_name)
        written_chapters: list[dict[str, Any]] = []
        total_words_estimate = 0
        current_recent = list(recent_texts)

        for round_idx, round_plans in enumerate(rounds):
            round_first = round_plans[0].get("chapter_num", 1)
            round_last = round_plans[-1].get("chapter_num", len(round_plans))

            # Build previous context.
            prev_context = ""
            if current_recent:
                prev_context = "\n\n---\n\n".join(current_recent[-2:])

            self._info(
                f"  连续生成 第{round_idx + 1}/{len(rounds)}轮: "
                f"ch_{round_first:03d}-ch_{round_last:03d}（{len(round_plans)}章）..."
            )

            try:
                results = self._chapter_writer.write_chapters_continuous(
                    novel_title=novel_title_str,
                    volume_num=volume_num,
                    chapter_plans=round_plans,
                    volume_outline=volume_outline,
                    total_outline=total_outline,
                    prev_context=prev_context,
                    target_words=words_per_chapter,
                )
            except Exception:
                logger.exception(
                    "Failed continuous generation round %d for vol_%03d",
                    round_idx + 1, volume_num,
                )
                raise

            # Save chapters and update memory after each round.
            for i, result in enumerate(results):
                ch_plan = round_plans[i] if i < len(round_plans) else round_plans[-1]
                ch_num = ch_plan.get("chapter_num", round_first + i)
                title = result.get("title", f"第{ch_num}章")
                content = result.get("content", "")

                word_count = self._count_chinese_chars(content)

                self._novel_store.save_chapter(
                    novel_name, volume_num, ch_num, content, title
                )

                total_words_estimate += word_count
                written_chapters.append({
                    "chapter_num": ch_num,
                    "title": title,
                    "word_count": word_count,
                })

                current_recent.append(content)
                if len(current_recent) > 2:
                    current_recent = current_recent[-2:]

                self._info(
                    f"  vol_{volume_num:03d}/ch_{ch_num:03d} 「{title}」"
                    f" 完成（~{word_count}字）"
                )

            # Update memory tables after each round.
            self._info(f"  第{round_idx + 1}轮完成，更新记忆表...")
            for result in results:
                content = result.get("content", "")
                if content:
                    try:
                        self._chapter_writer.update_memory_after_chapter(
                            content, volume_num, 0  # chapter_num not critical for extraction
                        )
                    except Exception:
                        logger.exception("Memory update failed for round %d", round_idx + 1)

        return written_chapters, total_words_estimate

    def _write_batched(
        self,
        novel_name: str,
        volume_num: int,
        chapter_plan: list[dict[str, Any]],
        volume_outline: dict[str, Any],
        total_outline: dict[str, Any],
        recent_texts: list[str],
        words_per_chapter: int,
        batch_size: int,
    ) -> tuple[list[dict[str, Any]], int]:
        """Generate chapters using batch mode (batch_size chapters per LLM call).

        Returns:
            (written_chapters, total_words_estimate)
        """
        novel_title_str = total_outline.get("_meta", {}).get("title", novel_name)
        written_chapters: list[dict[str, Any]] = []
        total_words_estimate = 0
        current_recent = list(recent_texts)

        # Group into batches.
        batches: list[list[dict[str, Any]]] = []
        for i in range(0, len(chapter_plan), batch_size):
            batches.append(chapter_plan[i : i + batch_size])

        for batch_idx, batch in enumerate(batches):
            batch_first = batch[0].get("chapter_num", 1)
            batch_last = batch[-1].get("chapter_num", len(batch))

            prev_context = ""
            if current_recent:
                prev_context = "\n\n---\n\n".join(current_recent[-2:])

            if batch_size == 1:
                ch_num = batch_first
                self._info(f"  正在生成 vol_{volume_num:03d}/ch_{ch_num:03d} ...")
                try:
                    result = self._chapter_writer.write_chapter(
                        novel_title=novel_title_str,
                        volume_num=volume_num,
                        chapter_num=ch_num,
                        volume_outline=volume_outline,
                        chapter_plan_item=batch[0],
                        total_outline=total_outline,
                        prev_context=prev_context,
                        target_words=words_per_chapter,
                    )
                except Exception:
                    logger.exception("Failed to generate vol_%03d/ch_%03d", volume_num, ch_num)
                    raise
                results = [result]
            else:
                self._info(
                    f"  正在批量生成 vol_{volume_num:03d}/ch_{batch_first:03d}"
                    f"-ch_{batch_last:03d}（{len(batch)}章）..."
                )
                try:
                    results = self._chapter_writer.write_chapters_batch(
                        novel_title=novel_title_str,
                        volume_num=volume_num,
                        chapter_plans=batch,
                        volume_outline=volume_outline,
                        total_outline=total_outline,
                        prev_context=prev_context,
                        target_words=words_per_chapter,
                    )
                except Exception:
                    logger.exception(
                        "Failed to generate batch vol_%03d/ch_%03d-ch_%03d",
                        volume_num, batch_first, batch_last,
                    )
                    raise

            # Process results.
            for i, result in enumerate(results):
                ch_plan = batch[i] if i < len(batch) else batch[-1]
                ch_num = ch_plan.get("chapter_num", batch_first + i)
                title = result.get("title", f"第{ch_num}章")
                content = result.get("content", "")

                word_count = self._count_chinese_chars(content)

                self._novel_store.save_chapter(
                    novel_name, volume_num, ch_num, content, title
                )
                self._chapter_writer.update_memory_after_chapter(
                    content, volume_num, ch_num
                )

                total_words_estimate += word_count
                written_chapters.append({
                    "chapter_num": ch_num,
                    "title": title,
                    "word_count": word_count,
                })

                current_recent.append(content)
                if len(current_recent) > 2:
                    current_recent = current_recent[-2:]

                self._info(
                    f"  vol_{volume_num:03d}/ch_{ch_num:03d} 「{title}」"
                    f" 完成（~{word_count}字）"
                )

        return written_chapters, total_words_estimate

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _build_chapter_prompt(
        self,
        recent_texts: list[str],
        chapter_plan_item: dict[str, Any],
    ) -> str:
        """Assemble the previous-chapter continuity text for a single chapter.

        Combines the last 1-2 generated chapters into a single string
        that is passed as ``prev_context`` to
        :meth:`ChapterWriter.write_chapter`.

        Args:
            recent_texts: The full text of the last 1-2 chapters (most
                recent last).
            chapter_plan_item: The current chapter plan item (used to
                add targeted hints for continuity).

        Returns:
            A string of previous chapter text, or ``""`` if no previous
            chapters are available.
        """
        if not recent_texts:
            return ""

        # Combine recent chapters, with a separator.
        parts: list[str] = []
        for i, text in enumerate(recent_texts):
            # Strip the title line for cleaner context injection (the
            # LLM will receive the title via chapter plan anyway).
            cleaned = text
            if cleaned.startswith("## "):
                # Keep the title for context.
                pass
            parts.append(cleaned)

        return "\n\n---\n\n".join(parts)

    # ------------------------------------------------------------------
    # Context size check
    # ------------------------------------------------------------------

    def _check_context_size(self, context_text: str) -> None:
        """Warn if *context_text* token count approaches 1M token limit.

        Uses :meth:`LLMClient.count_tokens` for estimation.  Issues a
        ``WARNING`` log when the estimated token count exceeds 85% of
        the 1M window (i.e. ~850k tokens).

        Args:
            context_text: The context string to estimate tokens for.
        """
        if not context_text:
            return

        tokens = self._llm.count_tokens(context_text)

        if tokens > _TOKEN_CHECK_THRESHOLD:
            logger.warning(
                "Context size ~%d tokens exceeds %d%% of 1M token window "
                "(%d tokens). Consider reducing context or splitting the volume.",
                tokens,
                int(_TOKEN_LIMIT_WARN_RATIO * 100),
                _TOKEN_CHECK_THRESHOLD,
            )

    # ------------------------------------------------------------------
    # Summary generation
    # ------------------------------------------------------------------

    def _generate_and_save_summary(
        self,
        novel_name: str,
        volume_num: int,
        volume_title: str,
        chapters: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Generate a volume summary via :class:`SummaryGenerator` and
        persist it alongside existing summaries.

        Args:
            novel_name: Novel identifier.
            volume_num: Volume number.
            volume_title: Human-readable volume title.
            chapters: List of chapter dicts as written (each with
                ``chapter_num``, ``title``, ``word_count``).

        Returns:
            The generated volume summary dict.
        """
        # Load full chapter contents for the summary prompt.
        full_chapters: list[dict[str, str]] = []
        for ch in chapters:
            ch_num = ch["chapter_num"]
            try:
                loaded = self._novel_store.load_chapter(novel_name, volume_num, ch_num)
                full_chapters.append(loaded)
            except FileNotFoundError:
                logger.warning(
                    "Chapter vol_%03d/ch_%03d not found for summary generation",
                    volume_num, ch_num,
                )
                full_chapters.append({"title": ch.get("title", ""), "content": ""})

        summary = self._summary_generator.generate_volume_summary(
            volume_num=volume_num,
            volume_title=volume_title,
            chapters=full_chapters,
            llm_client=self._llm,
        )

        # Persist: read existing summaries, append, write back.
        existing = self._table_store.load_volume_summaries()
        existing.append(summary)
        self._table_store.save_volume_summaries(existing)

        logger.info(
            "Volume %d summary saved (%d total summaries)",
            volume_num, len(existing),
        )

        return summary

    # ------------------------------------------------------------------
    # Progress reporting
    # ------------------------------------------------------------------

    def _progress_wrapper(
        self,
        items: list[dict[str, Any]],
        description: str,
    ) -> Any:
        """Wrap *items* in a ``rich.progress.Progress`` bar if rich is
        available, otherwise return the plain list unchanged.

        Args:
            items: The chapter plan list to iterate over.
            description: Label for the progress bar.

        Returns:
            An iterable — either the original list or a rich ``track``
            wrapper.
        """
        if _HAS_RICH and self._console is not None:
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=self._console,
            )
            task = progress.add_task(description, total=len(items))
            progress.start()

            # We need to yield items while updating progress.
            # Return a generator wrapper.
            def _tracked() -> Any:
                for item in items:
                    yield item
                    progress.update(task, advance=1)
                progress.stop()

            return _tracked()

        return items

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_config(config_path: str = "config.yaml") -> dict[str, Any]:
        """Load generation defaults from a YAML config file.

        If the file does not exist or cannot be parsed, returns sensible
        defaults so the CLI does not always need to pass these parameters.

        Args:
            config_path: Path to the ``config.yaml`` file.

        Returns:
            A dict with keys ``words_per_chapter`` (int) and
            ``chapters_per_volume`` (int).
        """
        defaults: dict[str, Any] = {
            "words_per_chapter": 5000,
            "chapters_per_volume": 50,
        }
        config_file = Path(config_path)
        if not config_file.exists():
            logger.debug("config.yaml not found at %s, using defaults", config_path)
            return defaults
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict):
                generation = data.get("generation", {})
                if isinstance(generation, dict):
                    defaults["words_per_chapter"] = generation.get(
                        "words_per_chapter", defaults["words_per_chapter"]
                    )
                    defaults["chapters_per_volume"] = generation.get(
                        "chapters_per_volume", defaults["chapters_per_volume"]
                    )
        except Exception:
            logger.warning(
                "Failed to parse %s, using defaults", config_path, exc_info=True
            )
        return defaults

    @staticmethod
    def _count_chinese_chars(text: str) -> int:
        """Count Chinese characters in *text*.

        This is a rough word-count equivalent for Chinese fiction
        (one character = roughly one word).

        Args:
            text: The text to count.

        Returns:
            Number of CJK Unified Ideographs found.
        """
        import re

        return len(re.findall(r"[一-鿿㐀-䶿]", text))

    def _info(self, message: str) -> None:
        """Log at INFO level and also print to console if rich is not available.

        Args:
            message: The message to emit.
        """
        logger.info(message)
        if not _HAS_RICH:
            print(message)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def chapter_writer(self) -> ChapterWriter:
        """The :class:`ChapterWriter` used for single-chapter generation."""
        return self._chapter_writer

    @property
    def outline_manager(self) -> OutlineManager:
        """The :class:`OutlineManager` used for context assembly."""
        return self._outline_manager
