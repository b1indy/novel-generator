"""Single-chapter generation with prompt assembly and memory update.

Provides ChapterWriter, which constructs the full LLM prompt for a single
chapter (injecting style, genre, outline, memory, and continuity context),
calls the LLM, parses the result, and updates all three memory managers
after each chapter.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from ..config.style import StyleManager
from ..llm.client import LLMClient
from ..memory.character import CharacterManager
from ..memory.foreshadowing import ForeshadowingManager
from ..memory.item import ItemManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# System prompt for chapter generation
# ---------------------------------------------------------------------------

_CHAPTER_SYSTEM_PROMPT = """\
你是一位专业的中国网络小说作家。你的任务是根据给定的大纲、角色设定、\
记忆表和上下文，创作一章小说章节。

写作要求：
1. 严格按照提供的写作风格和流派设定进行创作
2. 确保内容与总大纲、卷大纲的剧情方向一致
3. 角色行为需与角色设定表一致，不产生性格或状态矛盾
4. 注意伏笔的铺设（如大纲要求）和已有伏笔的回收
5. 章节内容要有起承转合，结尾留有悬念或阅读动力
6. 对话自然，描写生动，避免模板化表达
7. 每章开头不要重复上一章的结尾内容，应直接推进剧情

输出格式：
- 第一行为章节标题，格式：## 第X章 标题
- 空一行后开始章节正文
- 正文为纯中文小说内容，不需要任何标记或前缀"""


# ---------------------------------------------------------------------------
# ChapterWriter
# ---------------------------------------------------------------------------


class ChapterWriter:
    """Generates a single novel chapter with full context injection.

    Assembles the LLM prompt from style, genre, outline, memory, and
    continuity context, calls the LLM, and parses the ``## 第X章 标题``
    + body format.

    Usage::

        llm = LLMClient("config.yaml")
        style = StyleManager()
        char_mgr = CharacterManager("data/novels/my-novel")
        item_mgr = ItemManager("data/novels/my-novel")
        fh_mgr = ForeshadowingManager("data/novels/my-novel")
        writer = ChapterWriter(llm, style, char_mgr, item_mgr, fh_mgr)
        result = writer.write_chapter(
            novel_title="星辰变",
            volume_num=1,
            chapter_num=1,
            volume_outline=vol_outline,
            chapter_plan_item=ch_plan,
            total_outline=total_outline,
        )
        writer.update_memory_after_chapter(result["content"], 1, 1)
    """

    # Maximum tokens for chapter generation output.
    _DEFAULT_MAX_TOKENS: int = 8192
    _DEFAULT_TEMPERATURE: float = 0.8

    def __init__(
        self,
        llm_client: LLMClient,
        style_manager: StyleManager,
        character_manager: CharacterManager,
        item_manager: ItemManager,
        foreshadowing_manager: ForeshadowingManager,
    ) -> None:
        """Initialise the chapter writer with all required dependencies.

        Args:
            llm_client: Configured LLM client for API calls.
            style_manager: Style manager for author-style and genre prompts.
            character_manager: Character state tracker for the novel.
            item_manager: Item state tracker for the novel.
            foreshadowing_manager: Foreshadowing state tracker for the novel.
        """
        self._llm = llm_client
        self._style = style_manager
        self._characters = character_manager
        self._items = item_manager
        self._foreshadowing = foreshadowing_manager

        self._temperature: float = self._DEFAULT_TEMPERATURE
        self._max_tokens: int = self._DEFAULT_MAX_TOKENS

    # ------------------------------------------------------------------
    # Chapter generation
    # ------------------------------------------------------------------

    def write_chapter(
        self,
        novel_title: str,
        volume_num: int,
        chapter_num: int,
        volume_outline: dict[str, Any],
        chapter_plan_item: dict[str, Any],
        total_outline: dict[str, Any],
        prev_context: str = "",
        target_words: int = 3000,
    ) -> dict[str, str]:
        """Generate a single chapter.

        Args:
            novel_title: Display title of the novel.
            volume_num: Volume number (1-indexed).
            chapter_num: Chapter number within the volume (1-indexed).
            volume_outline: The full volume outline dict (from
                :meth:`OutlineManager.get_volume_outline`).
            chapter_plan_item: Dict from ``volume_outline["chapter_plan"]``
                with keys ``chapter_num``, ``title_hint``, ``key_events``,
                ``character_focus``.
            total_outline: The complete total outline dict.
            prev_context: Full text of the previous 1-2 chapters for
                narrative continuity. May be empty for the very first
                chapter.
            target_words: Approximate target word count for this chapter
                in Chinese characters. Default 3000.

        Returns:
            Dict with ``title`` (e.g. ``"第1章 荒古禁地"``) and
            ``content`` (the full chapter body including the title line
            in ``## 第X章 标题`` format).
        """
        # Resolve style & genre from outline metadata.
        meta = total_outline.get("_meta", {})
        style_name = meta.get("style_name", "")
        genre = meta.get("genre", "")

        # Build system + user messages.
        system_msg = self._build_system_prompt(style_name, genre)
        user_msg = self._build_user_prompt(
            novel_title=novel_title,
            volume_num=volume_num,
            chapter_num=chapter_num,
            total_outline=total_outline,
            volume_outline=volume_outline,
            chapter_plan_item=chapter_plan_item,
            prev_context=prev_context,
            target_words=target_words,
        )

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        # Estimate total prompt tokens for diagnostics.
        prompt_text = system_msg + "\n" + user_msg
        est_tokens = self._llm.count_tokens(prompt_text)
        logger.info(
            "Generating vol_%03d/ch_%03d: prompt ~%d tokens, target ~%d chars",
            volume_num, chapter_num, est_tokens, target_words,
        )

        response = self._llm.chat(
            messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )

        return self._parse_chapter_response(response, volume_num, chapter_num)

    # ------------------------------------------------------------------
    # Memory update
    # ------------------------------------------------------------------

    def update_memory_after_chapter(
        self,
        chapter_content: str,
        volume_num: int,
        chapter_num: int,
    ) -> None:
        """Extract character, item, and foreshadowing changes from
        *chapter_content* and apply them to all three memory managers.

        Tracks ``first_appearance`` and ``last_appearance`` on characters
        and items, as well as ``planted_in`` on foreshadowing entries.

        Args:
            chapter_content: Full text of the chapter (body, excluding
                the title line).
            volume_num: Volume number (1-indexed).
            chapter_num: Chapter number within the volume (1-indexed).
        """
        chapter_ref = f"vol_{volume_num:03d}/ch_{chapter_num:03d}"

        # --- Characters ---
        try:
            char_changes = self._characters.extract_from_chapter(
                chapter_content, self._llm
            )
            self._apply_character_changes(char_changes, chapter_ref)
            logger.debug(
                "Applied %d character changes for %s", len(char_changes), chapter_ref
            )
        except Exception:
            logger.exception(
                "Failed to extract/apply character changes for %s", chapter_ref
            )

        # --- Items ---
        try:
            item_changes = self._items.extract_from_chapter(
                chapter_content, self._llm
            )
            self._apply_item_changes(item_changes, chapter_ref)
            logger.debug(
                "Applied %d item changes for %s", len(item_changes), chapter_ref
            )
        except Exception:
            logger.exception(
                "Failed to extract/apply item changes for %s", chapter_ref
            )

        # --- Foreshadowing ---
        try:
            fh_changes = self._foreshadowing.extract_from_chapter(
                chapter_content, self._llm
            )
            self._apply_foreshadowing_changes(fh_changes, chapter_ref)
            logger.debug(
                "Applied %d foreshadowing changes for %s", len(fh_changes), chapter_ref
            )
        except Exception:
            logger.exception(
                "Failed to extract/apply foreshadowing changes for %s", chapter_ref
            )

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_system_prompt(self, style_name: str, genre: str) -> str:
        """Assemble the system prompt with optional style and genre injection.

        Args:
            style_name: Author style name (e.g. ``"辰东式热血"``).
            genre: Genre name (e.g. ``"玄幻"``).

        Returns:
            The full system prompt string.
        """
        parts: list[str] = [_CHAPTER_SYSTEM_PROMPT]

        # Inject genre guidance.
        if genre:
            try:
                parts.append("\n" + self._style.get_genre_prompt(genre))
            except KeyError:
                logger.debug("Genre %r not found, skipping genre injection", genre)

        # Inject author style.
        if style_name:
            try:
                parts.append("\n" + self._style.get_style_prompt(style_name))
            except KeyError:
                logger.debug(
                    "Style %r not found, skipping style injection", style_name
                )

        return "\n".join(parts)

    def _build_user_prompt(
        self,
        novel_title: str,
        volume_num: int,
        chapter_num: int,
        total_outline: dict[str, Any],
        volume_outline: dict[str, Any],
        chapter_plan_item: dict[str, Any],
        prev_context: str,
        target_words: int,
    ) -> str:
        """Construct the user message containing all context for chapter generation.

        Args:
            novel_title: Display title of the novel.
            volume_num: Volume number (1-indexed).
            chapter_num: Chapter number within the volume (1-indexed).
            total_outline: The complete total outline dict.
            volume_outline: The full volume outline dict.
            chapter_plan_item: The current chapter's plan entry.
            prev_context: Text of the previous 1-2 chapters.
            target_words: Approximate target Chinese character count.

        Returns:
            The complete user prompt string.
        """
        blocks: list[str] = []

        # -- Novel & volume header --
        volume_title = volume_outline.get("volume_title", f"第{volume_num}卷")
        blocks.append(
            f"请为小说《{novel_title}》创作第{volume_num}卷「{volume_title}」"
            f"的第{chapter_num}章。\n"
            f"本章目标字数：约{target_words}字。\n"
        )

        # -- Total outline summary --
        total_summary = self._format_total_outline_summary(total_outline)
        blocks.append(f"【总大纲摘要】\n{total_summary}")

        # -- Volume outline summary --
        volume_summary = self._format_volume_outline_summary(volume_outline)
        blocks.append(f"【卷大纲信息】\n{volume_summary}")

        # -- Current chapter plan --
        chapter_plan = self._format_chapter_plan(chapter_plan_item)
        blocks.append(f"【本章写作计划】\n{chapter_plan}")

        # -- Memory context --
        memory_context = self._format_memory_context()
        blocks.append(f"【当前状态表】\n{memory_context}")

        # -- Previous chapter continuity --
        if prev_context.strip():
            # Use more context for cross-volume transitions (first chapter
            # of a new volume needs richer continuity from the previous volume).
            limit = 8000 if chapter_num == 1 else 4000
            safe_prev = prev_context
            if len(safe_prev) > limit:
                safe_prev = safe_prev[-limit:]
                logger.debug(
                    "Truncated prev_context from %d to %d chars for prompt size",
                    len(prev_context), limit,
                )

            # For the first chapter of a new volume, add explicit transition
            # instructions to prevent plot thread dropping.
            if chapter_num == 1 and volume_num > 1:
                transition_note = (
                    "注意：这是新一卷的开篇。上文是前一卷的结尾章节。"
                    "你必须：\n"
                    "1. 自然延续前卷结尾的剧情状态和氛围\n"
                    "2. 保留前卷末尾埋下的所有伏笔和悬念\n"
                    "3. 角色状态、关系、位置需与前卷结尾一致\n"
                    "4. 不要重新引入已在前卷建立的设定\n"
                    "5. 可以有时间推进，但需明确交代过渡"
                )
                blocks.append(
                    f"【前卷结尾衔接（关键：这是跨卷过渡）】\n{safe_prev}\n\n"
                    f"【跨卷衔接要求】\n{transition_note}"
                )
            else:
                blocks.append(f"【上文衔接（已有章节内容，请保持连贯）】\n{safe_prev}")
        else:
            blocks.append("【上文衔接】这是本小说/本卷的第一章，无上文内容。请正常开篇。")

        # -- Output instructions --
        blocks.append(
            "请按照系统提示中的格式输出。确保内容紧凑、有张力，"
            "符合计划中的关键事件和角色焦点。"
        )

        return "\n\n".join(blocks)

    # ------------------------------------------------------------------
    # Context formatting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_total_outline_summary(total_outline: dict[str, Any]) -> str:
        """Produce a compact summary of the total outline for prompt injection."""
        parts: list[str] = []

        world = total_outline.get("world_setting", "")
        if world:
            parts.append(f"世界观：{world[:300]}")

        plot = total_outline.get("main_plot_arc", "")
        if plot:
            parts.append(f"主线剧情：{plot[:400]}")

        synopsis = total_outline.get("synopsis", "")
        if synopsis:
            parts.append(f"全书导语：{synopsis}")

        # Volume plan overview.
        volume_plan = total_outline.get("volume_plan", [])
        if volume_plan:
            vp_lines = []
            for vp in volume_plan:
                v = vp.get("volume", "?")
                t = vp.get("title", "")
                s = vp.get("summary", "")
                vp_lines.append(f"  第{v}卷「{t}」: {s}")
            parts.append("分卷规划：\n" + "\n".join(vp_lines))

        return "\n".join(parts) if parts else "(无总大纲)"

    @staticmethod
    def _format_volume_outline_summary(volume_outline: dict[str, Any]) -> str:
        """Produce a compact summary of the volume outline."""
        parts: list[str] = []

        title = volume_outline.get("volume_title", "")
        arc = volume_outline.get("volume_arc", "")
        synopsis = volume_outline.get("synopsis", "")

        if title:
            parts.append(f"卷标题：{title}")
        if arc:
            parts.append(f"卷故事弧线：{arc}")
        if synopsis:
            parts.append(f"卷导语：{synopsis}")

        return "\n".join(parts) if parts else "(无卷大纲)"

    @staticmethod
    def _format_chapter_plan(chapter_plan_item: dict[str, Any]) -> str:
        """Format a single chapter plan item for the prompt."""
        parts: list[str] = []

        title_hint = chapter_plan_item.get("title_hint", "")
        if title_hint:
            parts.append(f"标题建议：{title_hint}")

        key_events = chapter_plan_item.get("key_events", [])
        if key_events:
            events_str = "\n".join(f"  - {e}" for e in key_events)
            parts.append(f"关键事件：\n{events_str}")

        character_focus = chapter_plan_item.get("character_focus", "")
        if character_focus:
            parts.append(f"角色焦点：{character_focus}")

        return "\n".join(parts) if parts else "(无具体计划)"

    def _format_memory_context(self) -> str:
        """Gather compact context strings from all three memory managers."""
        char_ctx = self._characters.prompt_context()
        item_ctx = self._items.prompt_context()
        fh_ctx = self._foreshadowing.prompt_context()

        blocks: list[str] = []
        blocks.append(f"角色：\n{char_ctx}")
        blocks.append(f"物品/法宝：\n{item_ctx}")
        blocks.append(f"伏笔/未回收线索：\n{fh_ctx}")

        return "\n\n".join(blocks)

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_chapter_response(
        response: str,
        volume_num: int,
        chapter_num: int,
    ) -> dict[str, str]:
        """Parse the LLM response into a ``{title, content}`` dict.

        Expects the first line to be ``## 第X章 标题``. If not found,
        generates a fallback title.

        Args:
            response: Raw LLM response text.
            volume_num: Volume number for fallback title.
            chapter_num: Chapter number for fallback title.

        Returns:
            Dict with ``title`` and ``content`` keys.
        """
        text = response.strip()

        # Try to extract title from first line: "## 第X章 标题"
        title_pattern = re.compile(r"^##\s*(第[一二三四五六七八九十百千\d]+章\s*\S.*)", re.MULTILINE)
        match = title_pattern.match(text)

        if match:
            title = match.group(1).strip()
            # Content is everything after the title line, skipping the blank line.
            body_start = match.end()
            body = text[body_start:].lstrip("\n")
            # Generate a full title with chapter number for saving.
            full_content = f"## {title}\n\n{body}"
            return {"title": title, "content": full_content}

        # Fallback: try to find "第X章" anywhere in the first few lines.
        fallback_pattern = re.compile(r"(第[一二三四五六七八九十百千\d]+章\s*\S+)")
        fallback_match = fallback_pattern.search(text[:200])
        if fallback_match:
            title = fallback_match.group(1).strip()
            body = text
            full_content = f"## {title}\n\n{body}"
            logger.warning(
                "Title not on first line, extracted from body: %r", title
            )
            return {"title": title, "content": full_content}

        # Last resort: generate a synthetic title.
        title = f"第{chapter_num}章"
        logger.warning(
            "No chapter title found in LLM response for vol_%03d/ch_%03d; "
            "using fallback title %r",
            volume_num, chapter_num, title,
        )
        return {"title": title, "content": f"## {title}\n\n{text}"}

    # ------------------------------------------------------------------
    # Continuous generation (multi-turn "继续" approach)
    # ------------------------------------------------------------------

    def write_chapters_continuous(
        self,
        novel_title: str,
        volume_num: int,
        chapter_plans: list[dict[str, Any]],
        volume_outline: dict[str, Any],
        total_outline: dict[str, Any],
        prev_context: str = "",
        target_words: int = 3000,
        max_continues: int = 10,
    ) -> list[dict[str, str]]:
        """Generate chapters in a multi-turn conversation for maximum coherence.

        Uses the "继续" pattern: starts a conversation with full context,
        asks the LLM to generate multiple chapters, and sends "继续" when
        the output is truncated, keeping the full conversation context.

        Args:
            novel_title: Display title of the novel.
            volume_num: Volume number (1-indexed).
            chapter_plans: List of chapter plan items to generate.
            volume_outline: The full volume outline dict.
            total_outline: The complete total outline dict.
            prev_context: Full text of the previous 1-2 chapters.
            target_words: Approximate target word count per chapter.
            max_continues: Maximum number of "继续" rounds. Default 10.

        Returns:
            List of dicts with ``title`` and ``content`` keys.
        """
        if not chapter_plans:
            return []

        meta = total_outline.get("_meta", {})
        style_name = meta.get("style_name", "")
        genre = meta.get("genre", "")

        # Build the initial system + user prompt.
        system_msg = self._build_continuous_system_prompt(
            style_name, genre, chapter_plans, target_words
        )
        user_msg = self._build_continuous_user_prompt(
            novel_title=novel_title,
            volume_num=volume_num,
            chapter_plans=chapter_plans,
            total_outline=total_outline,
            volume_outline=volume_outline,
            prev_context=prev_context,
            target_words=target_words,
        )

        # Multi-turn conversation loop.
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        all_response_parts: list[str] = []
        continue_count = 0

        logger.info(
            "Starting continuous generation for vol_%03d: %d chapters planned",
            volume_num, len(chapter_plans),
        )

        while continue_count <= max_continues:
            # Use higher max_tokens for continuous generation.
            max_tok = min(16384, self._max_tokens * 2)

            response = self._llm.chat(
                messages,
                temperature=self._temperature,
                max_tokens=max_tok,
            )

            all_response_parts.append(response)
            messages.append({"role": "assistant", "content": response})

            # Check if we have all expected chapters.
            combined = "\n\n".join(all_response_parts)
            found_chapters = len(re.findall(r"##\s*第[一二三四五六七八九十百千\d]+章", combined))

            logger.info(
                "Continuous round %d: got %d chapters so far (target %d)",
                continue_count + 1, found_chapters, len(chapter_plans),
            )

            if found_chapters >= len(chapter_plans):
                break  # All chapters generated.

            # Check if response was truncated (ended mid-sentence).
            last_char = response.strip()[-1] if response.strip() else ""
            seems_truncated = last_char not in ("。", "！", "？", "」", "』", "…", '"', "'")

            if not seems_truncated and found_chapters < len(chapter_plans):
                # Response ended naturally but not all chapters done.
                # Ask for remaining chapters explicitly.
                remaining = len(chapter_plans) - found_chapters
                next_ch = chapter_plans[found_chapters].get("chapter_num", found_chapters + 1)
                messages.append({
                    "role": "user",
                    "content": f"请继续生成第{next_ch}章及后续{remaining - 1}章。",
                })
            else:
                # Truncated mid-generation, send "继续".
                messages.append({"role": "user", "content": "继续"})

            continue_count += 1

        # Parse all generated chapters.
        combined = "\n\n".join(all_response_parts)
        return self._parse_continuous_response(combined, volume_num, chapter_plans)

    def _build_continuous_system_prompt(
        self,
        style_name: str,
        genre: str,
        chapter_plans: list[dict[str, Any]],
        target_words: int,
    ) -> str:
        """Build system prompt for continuous generation."""
        first_ch = chapter_plans[0].get("chapter_num", 1)
        last_ch = chapter_plans[-1].get("chapter_num", len(chapter_plans))

        parts: list[str] = [
            f"你是一位专业的中国网络小说作家。你的任务是连续创作第{first_ch}章到第{last_ch}章"
            f"（共{len(chapter_plans)}章），每章约{target_words}字。\n\n"
            f"写作要求：\n"
            f"1. 严格按照提供的写作风格和流派设定进行创作\n"
            f"2. 确保内容与总大纲、卷大纲的剧情方向一致\n"
            f"3. 角色行为需与角色设定表一致\n"
            f"4. 注意伏笔的铺设和已有伏笔的回收\n"
            f"5. 每章要有起承转合，结尾留有悬念\n"
            f"6. 对话自然，描写生动，避免模板化表达\n"
            f"7. 章节之间自然过渡，不要重复上一章结尾\n\n"
            f"输出格式：\n"
            f"- 每章以 ## 第X章 标题 开头\n"
            f"- 空一行后开始章节正文\n"
            f"- 一章结束后直接开始下一章（不要加分隔符）\n"
            f"- 如果输出被截断，下次从截断处继续\n"
            f"- 正文为纯中文小说内容"
        ]

        if genre:
            try:
                parts.append("\n" + self._style.get_genre_prompt(genre))
            except KeyError:
                pass

        if style_name:
            try:
                parts.append("\n" + self._style.get_style_prompt(style_name))
            except KeyError:
                pass

        return "\n".join(parts)

    def _build_continuous_user_prompt(
        self,
        novel_title: str,
        volume_num: int,
        chapter_plans: list[dict[str, Any]],
        total_outline: dict[str, Any],
        volume_outline: dict[str, Any],
        prev_context: str,
        target_words: int,
    ) -> str:
        """Build user prompt for continuous generation."""
        blocks: list[str] = []

        volume_title = volume_outline.get("volume_title", f"第{volume_num}卷")
        first_ch = chapter_plans[0].get("chapter_num", 1)
        last_ch = chapter_plans[-1].get("chapter_num", len(chapter_plans))

        blocks.append(
            f"请为小说《{novel_title}》连续创作第{volume_num}卷「{volume_title}」"
            f"的第{first_ch}章至第{last_ch}章。\n"
            f"每章目标字数：约{target_words}字。\n"
        )

        # Total outline summary.
        total_summary = self._format_total_outline_summary(total_outline)
        blocks.append(f"【总大纲摘要】\n{total_summary}")

        # Volume outline summary.
        volume_summary = self._format_volume_outline_summary(volume_outline)
        blocks.append(f"【卷大纲信息】\n{volume_summary}")

        # Chapter plans for this round.
        plan_lines: list[str] = []
        for cp in chapter_plans:
            ch_num = cp.get("chapter_num", "?")
            title_hint = cp.get("title_hint", "")
            key_events = cp.get("key_events", [])
            char_focus = cp.get("character_focus", "")

            plan_lines.append(f"第{ch_num}章：{title_hint}")
            if key_events:
                for evt in key_events:
                    plan_lines.append(f"  - {evt}")
            if char_focus:
                plan_lines.append(f"  角色焦点: {char_focus}")
        blocks.append(f"【本批次章节计划】\n" + "\n".join(plan_lines))

        # Memory context.
        memory_context = self._format_memory_context()
        blocks.append(f"【当前状态表】\n{memory_context}")

        # Previous chapter continuity.
        if prev_context.strip():
            limit = 8000 if first_ch == 1 and volume_num > 1 else 4000
            safe_prev = prev_context
            if len(safe_prev) > limit:
                safe_prev = safe_prev[-limit:]

            if first_ch == 1 and volume_num > 1:
                blocks.append(
                    f"【前卷结尾衔接】\n{safe_prev}\n\n"
                    f"【跨卷衔接要求】\n"
                    f"这是新一卷的开篇。必须自然延续前卷结尾的剧情，保留所有伏笔。"
                )
            else:
                blocks.append(f"【上文衔接】\n{safe_prev}")
        else:
            blocks.append("【上文衔接】这是本小说/本卷的开篇，无上文内容。")

        blocks.append(
            f"请从第{first_ch}章开始，连续输出{len(chapter_plans)}章完整内容。"
            f"每章以 ## 第X章 标题 开头，一章结束后直接开始下一章。"
        )

        return "\n\n".join(blocks)

    @staticmethod
    def _parse_continuous_response(
        response: str,
        volume_num: int,
        chapter_plans: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        """Parse a continuous generation response into individual chapters.

        Splits on ``## 第X章`` markers.
        """
        text = response.strip()

        # Split on chapter headers.
        pattern = re.compile(r"(?=##\s*第[一二三四五六七八九十百千\d]+章)")
        blocks = pattern.split(text)
        blocks = [b.strip() for b in blocks if b.strip()]

        results: list[dict[str, str]] = []
        title_pattern = re.compile(r"^##\s*(第[一二三四五六七八九十百千\d]+章\s*\S.*)")

        for i, block in enumerate(blocks):
            match = title_pattern.match(block)
            if match:
                title = match.group(1).strip()
                body = block[match.end():].lstrip("\n")
                full_content = f"## {title}\n\n{body}"
            else:
                ch_num = chapter_plans[i].get("chapter_num", i + 1) if i < len(chapter_plans) else i + 1
                title = f"第{ch_num}章"
                full_content = f"## {title}\n\n{block}"
                logger.warning("No title in continuous block %d, using fallback", i)

            results.append({"title": title, "content": full_content})

        if len(results) < len(chapter_plans):
            logger.warning(
                "Continuous response has %d chapters but expected %d",
                len(results), len(chapter_plans),
            )

        return results

    def write_chapters_batch(
        self,
        novel_title: str,
        volume_num: int,
        chapter_plans: list[dict[str, Any]],
        volume_outline: dict[str, Any],
        total_outline: dict[str, Any],
        prev_context: str = "",
        target_words: int = 3000,
    ) -> list[dict[str, str]]:
        """Generate multiple chapters in a single LLM call for better coherence.

        Args:
            novel_title: Display title of the novel.
            volume_num: Volume number (1-indexed).
            chapter_plans: List of chapter plan items to generate.
            volume_outline: The full volume outline dict.
            total_outline: The complete total outline dict.
            prev_context: Full text of the previous 1-2 chapters.
            target_words: Approximate target word count per chapter.

        Returns:
            List of dicts with ``title`` and ``content`` keys, one per chapter.
        """
        if not chapter_plans:
            return []

        if len(chapter_plans) == 1:
            # Single chapter - use the regular method.
            result = self.write_chapter(
                novel_title=novel_title,
                volume_num=volume_num,
                chapter_num=chapter_plans[0].get("chapter_num", 1),
                volume_outline=volume_outline,
                chapter_plan_item=chapter_plans[0],
                total_outline=total_outline,
                prev_context=prev_context,
                target_words=target_words,
            )
            return [result]

        # Build batch prompt.
        meta = total_outline.get("_meta", {})
        style_name = meta.get("style_name", "")
        genre = meta.get("genre", "")

        system_msg = self._build_batch_system_prompt(style_name, genre, len(chapter_plans))
        user_msg = self._build_batch_user_prompt(
            novel_title=novel_title,
            volume_num=volume_num,
            chapter_plans=chapter_plans,
            total_outline=total_outline,
            volume_outline=volume_outline,
            prev_context=prev_context,
            target_words=target_words,
        )

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        # Use higher max_tokens for batch generation.
        batch_max_tokens = self._max_tokens * len(chapter_plans)
        batch_max_tokens = min(batch_max_tokens, 65536)  # Cap at 64K tokens.

        prompt_text = system_msg + "\n" + user_msg
        est_tokens = self._llm.count_tokens(prompt_text)
        logger.info(
            "Generating batch of %d chapters for vol_%03d: prompt ~%d tokens",
            len(chapter_plans), volume_num, est_tokens,
        )

        response = self._llm.chat(
            messages,
            temperature=self._temperature,
            max_tokens=batch_max_tokens,
        )

        return self._parse_batch_response(response, volume_num, chapter_plans)

    def _build_batch_system_prompt(
        self, style_name: str, genre: str, batch_size: int
    ) -> str:
        """Build system prompt for batch chapter generation."""
        parts: list[str] = [
            f"你是一位专业的中国网络小说作家。你的任务是根据给定的大纲、角色设定、"
            f"记忆表和上下文，连续创作{batch_size}章小说章节。\n\n"
            f"写作要求：\n"
            f"1. 严格按照提供的写作风格和流派设定进行创作\n"
            f"2. 确保内容与总大纲、卷大纲的剧情方向一致\n"
            f"3. 角色行为需与角色设定表一致，不产生性格或状态矛盾\n"
            f"4. 注意伏笔的铺设和已有伏笔的回收\n"
            f"5. 每章内容要有起承转合，结尾留有悬念\n"
            f"6. 对话自然，描写生动，避免模板化表达\n"
            f"7. 每章开头不要重复上一章的结尾内容，应直接推进剧情\n"
            f"8. 章节之间要有自然的过渡和连贯性\n\n"
            f"输出格式：\n"
            f"- 每章以 ## 第X章 标题 开头\n"
            f"- 空一行后开始章节正文\n"
            f"- 章节之间用 --- 分隔\n"
            f"- 正文为纯中文小说内容，不需要任何其他标记"
        ]

        if genre:
            try:
                parts.append("\n" + self._style.get_genre_prompt(genre))
            except KeyError:
                pass

        if style_name:
            try:
                parts.append("\n" + self._style.get_style_prompt(style_name))
            except KeyError:
                pass

        return "\n".join(parts)

    def _build_batch_user_prompt(
        self,
        novel_title: str,
        volume_num: int,
        chapter_plans: list[dict[str, Any]],
        total_outline: dict[str, Any],
        volume_outline: dict[str, Any],
        prev_context: str,
        target_words: int,
    ) -> str:
        """Build user prompt for batch chapter generation."""
        blocks: list[str] = []

        volume_title = volume_outline.get("volume_title", f"第{volume_num}卷")
        first_ch = chapter_plans[0].get("chapter_num", 1)
        last_ch = chapter_plans[-1].get("chapter_num", len(chapter_plans))

        blocks.append(
            f"请为小说《{novel_title}》创作第{volume_num}卷「{volume_title}」"
            f"的第{first_ch}章至第{last_ch}章（共{len(chapter_plans)}章）。\n"
            f"每章目标字数：约{target_words}字。\n"
        )

        # Total outline summary.
        total_summary = self._format_total_outline_summary(total_outline)
        blocks.append(f"【总大纲摘要】\n{total_summary}")

        # Volume outline summary.
        volume_summary = self._format_volume_outline_summary(volume_outline)
        blocks.append(f"【卷大纲信息】\n{volume_summary}")

        # Chapter plans for the batch.
        plan_lines: list[str] = []
        for cp in chapter_plans:
            ch_num = cp.get("chapter_num", "?")
            title_hint = cp.get("title_hint", "")
            key_events = cp.get("key_events", [])
            char_focus = cp.get("character_focus", "")

            plan_lines.append(f"第{ch_num}章：{title_hint}")
            if key_events:
                for evt in key_events:
                    plan_lines.append(f"  - {evt}")
            if char_focus:
                plan_lines.append(f"  角色焦点: {char_focus}")
        blocks.append(f"【本批次章节计划】\n" + "\n".join(plan_lines))

        # Memory context.
        memory_context = self._format_memory_context()
        blocks.append(f"【当前状态表】\n{memory_context}")

        # Previous chapter continuity.
        if prev_context.strip():
            limit = 8000 if first_ch == 1 and volume_num > 1 else 4000
            safe_prev = prev_context
            if len(safe_prev) > limit:
                safe_prev = safe_prev[-limit:]

            if first_ch == 1 and volume_num > 1:
                transition_note = (
                    "注意：这是新一卷的开篇。上文是前一卷的结尾章节。"
                    "你必须自然延续前卷结尾的剧情状态和氛围，保留所有伏笔和悬念。"
                )
                blocks.append(
                    f"【前卷结尾衔接】\n{safe_prev}\n\n"
                    f"【跨卷衔接要求】\n{transition_note}"
                )
            else:
                blocks.append(f"【上文衔接】\n{safe_prev}")
        else:
            blocks.append("【上文衔接】这是本小说/本卷的开篇，无上文内容。请正常开篇。")

        blocks.append(
            f"请按照系统提示中的格式，连续输出{len(chapter_plans)}章完整内容。"
            f"每章之间用 --- 分隔。确保章节之间剧情连贯、节奏自然。"
        )

        return "\n\n".join(blocks)

    @staticmethod
    def _parse_batch_response(
        response: str,
        volume_num: int,
        chapter_plans: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        """Parse a batch LLM response into individual chapter dicts.

        Splits on ``---`` separators and extracts ``## 第X章 标题`` from each block.
        """
        text = response.strip()

        # Split on --- separator.
        # Handle various separator formats: ---, ---\n, \n---\n, etc.
        blocks = re.split(r"\n\s*---\s*\n", text)

        # Clean up blocks.
        blocks = [b.strip() for b in blocks if b.strip()]

        results: list[dict[str, str]] = []
        title_pattern = re.compile(r"^##\s*(第[一二三四五六七八九十百千\d]+章\s*\S.*)")

        for i, block in enumerate(blocks):
            # Try to extract title.
            match = title_pattern.match(block)
            if match:
                title = match.group(1).strip()
                body = block[match.end():].lstrip("\n")
                full_content = f"## {title}\n\n{body}"
            else:
                # Fallback.
                ch_num = chapter_plans[i].get("chapter_num", i + 1) if i < len(chapter_plans) else i + 1
                title = f"第{ch_num}章"
                full_content = f"## {title}\n\n{block}"
                logger.warning("No title found in batch block %d, using fallback", i)

            results.append({"title": title, "content": full_content})

        # If we got fewer blocks than expected, log warning.
        if len(results) < len(chapter_plans):
            logger.warning(
                "Batch response has %d blocks but expected %d chapters",
                len(results), len(chapter_plans),
            )

        return results

    # ------------------------------------------------------------------
    # Memory change application
    # ------------------------------------------------------------------

    def _apply_character_changes(
        self, changes: list[dict[str, Any]], chapter_ref: str
    ) -> None:
        """Apply character extraction results to CharacterManager.

        Args:
            changes: List of change dicts from
                :meth:`CharacterManager.extract_from_chapter`.
            chapter_ref: Chapter reference string for
                ``first_appearance`` / ``last_appearance``.
        """
        for change in changes:
            action = change.get("action", "")
            try:
                if action == "new":
                    char_id = self._characters.add_character(
                        name=change.get("name", ""),
                        traits=change.get("traits", []),
                        cultivation=change.get("cultivation", ""),
                        status=change.get("status", "alive"),
                        current_location=change.get("current_location", ""),
                        notes=change.get("notes", ""),
                        relationships=change.get("relationships", {}),
                        first_appearance=chapter_ref,
                        last_appearance=chapter_ref,
                    )
                    logger.debug(
                        "New character %r added (id=%s)", change.get("name"), char_id
                    )
                elif action == "update":
                    char_id = change.get("char_id", "")
                    if not char_id:
                        logger.warning("Character update missing char_id; skipping")
                        continue
                    changes_dict = change.get("changes", {})
                    if changes_dict:
                        changes_dict["last_appearance"] = chapter_ref
                        self._characters.update_character(char_id, **changes_dict)
                        logger.debug("Character %s updated: %s", char_id, list(changes_dict))
                else:
                    logger.debug("Unknown character change action: %r", action)
            except KeyError as exc:
                logger.warning("Failed to apply character change %r: %s", change, exc)

    def _apply_item_changes(
        self, changes: list[dict[str, Any]], chapter_ref: str
    ) -> None:
        """Apply item extraction results to ItemManager.

        Args:
            changes: List of change dicts from
                :meth:`ItemManager.extract_from_chapter`.
            chapter_ref: Chapter reference string for
                ``first_appearance``.
        """
        for change in changes:
            action = change.get("action", "")
            try:
                if action == "new":
                    item_id = self._items.add_item(
                        name=change.get("name", ""),
                        type=change.get("type", ""),
                        grade=change.get("grade", ""),
                        owner=change.get("owner", ""),
                        status=change.get("status", "in_use"),
                        significance=change.get("significance", ""),
                        notes=change.get("notes", ""),
                        first_appearance=chapter_ref,
                    )
                    logger.debug("New item %r added (id=%s)", change.get("name"), item_id)
                elif action == "update":
                    item_id = change.get("item_id", "")
                    if not item_id:
                        logger.warning("Item update missing item_id; skipping")
                        continue
                    changes_dict = change.get("changes", {})
                    if changes_dict:
                        self._items.update_item(item_id, **changes_dict)
                        logger.debug("Item %s updated: %s", item_id, list(changes_dict))
                elif action == "transfer":
                    item_id = change.get("item_id", "")
                    new_owner = change.get("new_owner", "")
                    if item_id and new_owner:
                        self._items.transfer_item(item_id, new_owner)
                        logger.debug(
                            "Item %s transferred to %s", item_id, new_owner
                        )
                else:
                    logger.debug("Unknown item change action: %r", action)
            except KeyError as exc:
                logger.warning("Failed to apply item change %r: %s", change, exc)

    def _apply_foreshadowing_changes(
        self, changes: list[dict[str, Any]], chapter_ref: str
    ) -> None:
        """Apply foreshadowing extraction results to ForeshadowingManager.

        Args:
            changes: List of change dicts from
                :meth:`ForeshadowingManager.extract_from_chapter`.
            chapter_ref: Chapter reference string for ``planted_in``
                and ``resolved_in``.
        """
        for change in changes:
            action = change.get("action", "")
            try:
                if action == "plant":
                    fh_id = self._foreshadowing.add_foreshadowing(
                        description=change.get("description", ""),
                        planted_in=chapter_ref,
                        hint_detail=change.get("hint_detail", ""),
                        related_characters=change.get("related_characters", []),
                        planned_resolution=change.get("planned_resolution", ""),
                        priority=change.get("priority", "medium"),
                    )
                    logger.debug(
                        "Planted foreshadowing %s: %s",
                        fh_id, change.get("description", "")[:40],
                    )
                elif action == "resolve":
                    fh_id = change.get("fh_id", "")
                    resolution_note = change.get("resolution_note", "")
                    if fh_id:
                        self._foreshadowing.resolve_foreshadowing(
                            fh_id, chapter_ref, resolution_note
                        )
                        logger.debug("Resolved foreshadowing %s", fh_id)
                else:
                    logger.debug("Unknown foreshadowing change action: %r", action)
            except KeyError as exc:
                logger.warning(
                    "Failed to apply foreshadowing change %r: %s", change, exc
                )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def temperature(self) -> float:
        """Sampling temperature for chapter generation."""
        return self._temperature

    @temperature.setter
    def temperature(self, value: float) -> None:
        self._temperature = value

    @property
    def max_tokens(self) -> int:
        """Max tokens for chapter generation output."""
        return self._max_tokens

    @max_tokens.setter
    def max_tokens(self, value: int) -> None:
        self._max_tokens = value
