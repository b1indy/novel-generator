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
            # Truncate if extremely long to control prompt size.
            safe_prev = prev_context
            if len(safe_prev) > 4000:
                safe_prev = safe_prev[-4000:]
                logger.debug(
                    "Truncated prev_context from %d to %d chars for prompt size",
                    len(prev_context), 4000,
                )
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
