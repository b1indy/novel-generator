"""LLM-driven outline generation for novels.

Provides OutlineGenerator which creates total outlines, per-volume outlines,
and supports iterative outline optimization using the LLM backend.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from ..config.style import StyleManager
from ..llm.client import LLMClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_TOTAL = """\
你是一位专业的网络小说大纲策划师。你的任务是根据给定的风格和流派，为一部网络小说设计完整的总大纲。

你必须严格按照JSON格式输出，不要输出任何JSON之外的内容。输出格式如下：
```json
{
  "world_setting": "世界观设定的详细描述（500-1000字）",
  "main_plot_arc": "主线剧情概述，包含起因、发展、高潮、结局的整体规划（800-1500字）",
  "main_characters": [
    {
      "name": "主角姓名",
      "role": "主角/反派/重要配角",
      "brief": "角色简介（100-200字），包含性格、背景、成长方向"
    }
  ],
  "volume_plan": [
    {
      "volume": 1,
      "title": "卷标题",
      "summary": "本卷剧情概要（1-2句话）",
      "estimated_chapters": 50,
      "key_milestones": ["关键剧情节点1", "关键剧情节点2"]
    }
  ],
  "synopsis": "全书导语简介（约100字符），用于吸引读者"
}
```

在规划大纲时请注意：
- 主线剧情需有清晰的起承转合
- 角色之间要有合理的关系网络和冲突
- 分卷规划需根据用户指定的每卷章数目标来设计（每卷默认50章左右）
- 大纲结构要适合超长篇小说（200万+字），为后续发展留下空间
- 世界观设定需丰富且有层次，能支撑长篇叙事
"""

_SYSTEM_PROMPT_VOLUME = """\
你是一位专业的网络小说章节策划师。你的任务是根据总大纲和已有上下文，为指定卷设计详细的卷大纲。

你必须严格按照JSON格式输出，不要输出任何JSON之外的内容。输出格式如下：
```json
{
  "volume_title": "本卷标题",
  "volume_arc": "本卷故事弧线描述（300-500字），包含本卷在总剧情中的定位、主要冲突、角色成长方向",
  "chapter_plan": [
    {
      "chapter_num": 1,
      "title_hint": "章节标题建议",
      "key_events": ["本章关键事件1", "本章关键事件2"],
      "character_focus": "本章重点刻画的角色及其发展"
    }
  ],
  "synopsis": "本卷导语简介（约100字符），概括本卷核心看点"
}
```

在规划卷大纲时请注意：
- chapter_plan 数组必须恰好包含指定数量的章节（由用户参数决定，默认50章），不多不少
- 卷大纲需与总大纲保持一致，推进主线剧情
- 每章需有明确的剧情推进和角色发展
- 注意前后卷之间的衔接，避免剧情断裂
- 合理分配本章的伏笔铺设与回收
- 章节之间要有节奏变化（紧张/舒缓交替）
"""

_SYSTEM_PROMPT_OPTIMIZE = """\
你是一位专业的网络小说大纲编辑。你的任务是根据反馈意见，修改现有的大纲，同时保持不需要改动的部分不变。

你必须严格按照JSON格式输出，输出格式与输入的大纲格式完全一致。不要输出任何JSON之外的内容。

修改原则：
- 仅修改与反馈和修改请求相关的部分
- 保持未涉及的部分原样不变
- 确保修改后的部分与其他部分逻辑一致
- 如果修改涉及角色或剧情线，需同步更新所有相关部分
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_json(text: str) -> dict[str, Any]:
    """Extract and parse a JSON object from LLM response text.

    Handles responses wrapped in markdown code fences, responses with
    leading/trailing text, and pure JSON responses.

    Args:
        text: Raw LLM response that should contain a JSON object.

    Returns:
        Parsed JSON dict.

    Raises:
        ValueError: If no valid JSON object can be extracted.
    """
    if not text.strip():
        raise ValueError("Empty response from LLM")

    # Try direct parse first (fast path).
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from markdown code fences: ```json ... ```
    fence_pattern = re.compile(r"```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL)
    matches = fence_pattern.findall(text)
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue

    # Try to find the outermost { ... } block.
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        try:
            return json.loads(text[brace_start : brace_end + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract valid JSON from LLM response: {text[:200]}...")


def _serialize_context(obj: Any) -> str:
    """Serialize a Python object to a JSON string for prompt injection.

    Args:
        obj: Any JSON-serializable object.

    Returns:
        Compact JSON string (without indentation to save tokens).
    """
    return json.dumps(obj, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# OutlineGenerator
# ---------------------------------------------------------------------------


class OutlineGenerator:
    """Generates novel outlines using LLM with style injection.

    Handles total outline creation, per-volume outline generation with full
    context injection, and iterative outline optimization.

    Usage::

        llm = LLMClient("config.yaml")
        style = StyleManager()
        gen = OutlineGenerator(llm, style)
        total = gen.generate_total_outline("仙逆", "仙侠", "辰东式热血", 2000000)
        vol1 = gen.generate_volume_outline(total, 1, [], [], {})
    """

    def __init__(self, llm_client: LLMClient, style_manager: StyleManager) -> None:
        """Initialize the outline generator.

        Args:
            llm_client: Configured LLM client for API calls.
            style_manager: Style manager for prompt injection.
        """
        self._llm = llm_client
        self._style = style_manager
        self._temperature: float = 0.7
        self._max_tokens: int = 8192

    # ------------------------------------------------------------------
    # Total outline
    # ------------------------------------------------------------------

    def generate_total_outline(
        self,
        novel_title: str,
        genre: str,
        style_name: str,
        target_words: int,
    ) -> dict[str, Any]:
        """Generate the complete total outline for a new novel.

        The returned outline includes world setting, main plot arc, character
        introductions, a volume plan, and a ~100 character synopsis.

        Args:
            novel_title: The title of the novel.
            genre: Genre name (must exist in StyleManager presets, e.g. "玄幻").
            style_name: Author style name (e.g. "辰东式热血").
            target_words: Target total word count for the novel.

        Returns:
            A structured dict with keys: ``world_setting``, ``main_plot_arc``,
            ``main_characters``, ``volume_plan``, ``synopsis``.

        Raises:
            ValueError: If the LLM response cannot be parsed as valid JSON.
        """
        style_prompt = self._style.get_style_prompt(style_name)
        genre_prompt = self._style.get_genre_prompt(genre)

        chapters_per_vol = 50  # default; actual is read from config by caller
        chapters_estimate = max(10, target_words // 3000)
        volumes_estimate = max(3, chapters_estimate // chapters_per_vol)

        user_prompt = (
            f"请为网络小说《{novel_title}》设计总大纲。\n\n"
            f"小说基本信息：\n"
            f"- 流派：{genre}\n"
            f"- 目标总字数：{target_words:,}字\n"
            f"- 预估总卷数：约{volumes_estimate}卷\n"
            f"- 预估总章数：约{chapters_estimate}章\n\n"
            f"{genre_prompt}\n\n"
            f"请按照要求的JSON格式输出总大纲。确保大纲内容丰富、结构清晰，"
            f"能够支撑一部{target_words:,}字的长篇网络小说。"
        )

        messages: list[dict[str, str]] = [
            {"role": "system", "content": f"{_SYSTEM_PROMPT_TOTAL}\n\n{style_prompt}"},
            {"role": "user", "content": user_prompt},
        ]

        logger.info(
            "Generating total outline for '%s' (genre=%s, style=%s, words=%d)",
            novel_title,
            genre,
            style_name,
            target_words,
        )

        response = self._llm.chat(
            messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )

        outline = _extract_json(response)

        # Validate required keys.
        required = {"world_setting", "main_plot_arc", "main_characters", "volume_plan", "synopsis"}
        missing = required - set(outline.keys())
        if missing:
            logger.warning("Total outline missing keys: %s", missing)
            for key in missing:
                outline[key] = "" if key != "main_characters" and key != "volume_plan" else []

        logger.info(
            "Total outline generated: %d characters, %d volumes planned",
            len(outline.get("main_characters", [])),
            len(outline.get("volume_plan", [])),
        )

        return outline

    # ------------------------------------------------------------------
    # Volume outline
    # ------------------------------------------------------------------

    def generate_volume_outline(
        self,
        total_outline: dict[str, Any],
        volume_num: int,
        prev_summaries: list[dict[str, Any]],
        last_chapters: list[dict[str, Any]],
        memory_context: dict[str, Any],
        chapters_per_volume: int = 50,
    ) -> dict[str, Any]:
        """Generate a detailed outline for a specific volume.

        Feeds in the total outline, previous volume summaries, the last
        chapters of the previous volume, and current memory tables to ensure
        continuity.

        Args:
            total_outline: The complete total outline dict.
            volume_num: The volume number to generate (1-indexed).
            prev_summaries: List of volume summary dicts for volumes 1..N-1.
            last_chapters: List of the last 2 chapters from volume N-1, each
                as a dict with ``title`` and ``content`` keys.
            memory_context: Dict containing current memory tables
                (``characters``, ``items``, ``foreshadowing``).
            chapters_per_volume: Number of chapters to plan for this volume.
                Defaults to 50.

        Returns:
            A structured dict with keys: ``volume_title``, ``volume_arc``,
            ``chapter_plan``, ``synopsis``.

        Raises:
            ValueError: If the LLM response cannot be parsed as valid JSON.
        """
        # Look up volume title hint from total outline volume plan.
        volume_plan = total_outline.get("volume_plan", [])
        volume_title_hint = ""
        for vp in volume_plan:
            if vp.get("volume") == volume_num:
                volume_title_hint = vp.get("title", "")
                break

        # Look up style from the total outline metadata, defaulting to built-in.
        style_name = total_outline.get("_meta", {}).get("style_name", "")
        style_prompt = ""
        if style_name:
            try:
                style_prompt = self._style.get_style_prompt(style_name)
            except KeyError:
                logger.debug("Style '%s' not found, proceeding without style injection", style_name)

        # Build context blocks.
        context_blocks: list[str] = []

        # Total outline summary.
        total_summary = {
            "world_setting": total_outline.get("world_setting", ""),
            "main_plot_arc": total_outline.get("main_plot_arc", ""),
            "main_characters": total_outline.get("main_characters", []),
            "synopsis": total_outline.get("synopsis", ""),
        }
        context_blocks.append(f"【总大纲摘要】\n{_serialize_context(total_summary)}")

        # Target volume info from volume plan.
        if volume_title_hint:
            context_blocks.append(
                f"【本卷在总大纲中的规划】\n卷{volume_num}「{volume_title_hint}」"
            )

        # Previous volume summaries.
        if prev_summaries:
            context_blocks.append(
                f"【前续卷小结】\n{_serialize_context(prev_summaries)}"
            )

        # Last chapters of previous volume (for continuity).
        if last_chapters:
            last_ch_text = ""
            for i, ch in enumerate(last_chapters):
                # Truncate chapter content to avoid token blowup; 2000 chars each.
                content = ch.get("content", "")[:2000]
                last_ch_text += (
                    f"\n--- 倒数第{len(last_chapters) - i}章: "
                    f"{ch.get('title', '无标题')} ---\n{content}\n"
                )
            context_blocks.append(f"【前卷最后章节衔接】{last_ch_text}")

        # Memory tables.
        if memory_context:
            context_blocks.append(
                f"【当前记忆表】\n{_serialize_context(memory_context)}"
            )

        context_text = "\n\n".join(context_blocks)

        user_prompt = (
            f"请为第{volume_num}卷设计详细的卷大纲。\n\n"
            f"生成本卷大纲，规划恰好{chapters_per_volume}章。\n\n"
            f"参考上下文：\n{context_text}\n\n"
            f"请按照要求的JSON格式输出卷大纲。注意：\n"
            f"- 章节规划需恰好包含{chapters_per_volume}章的具体内容\n"
            f"- 确保与前续剧情衔接自然\n"
            f"- 推进总大纲中的主线剧情\n"
            f"- 合理安排伏笔的铺设与回收\n"
            f"- 如果这是第一卷，请包含引入世界观和主要角色的章节"
        )

        system_content = _SYSTEM_PROMPT_VOLUME
        if style_prompt:
            system_content += f"\n\n{style_prompt}"

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_prompt},
        ]

        logger.info(
            "Generating volume %d outline (chapters_per_volume=%d)",
            volume_num,
            chapters_per_volume,
        )

        # Retry loop: if chapter_plan count deviates significantly from target,
        # re-ask the LLM with a stricter prompt (up to 3 retries).
        outline: dict[str, Any] = {}
        for attempt in range(4):
            response = self._llm.chat(
                messages,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )

            outline = _extract_json(response)

            chapter_plan = outline.get("chapter_plan", [])
            actual_count = len(chapter_plan)
            if abs(actual_count - chapters_per_volume) <= 5:
                break  # close enough

            logger.warning(
                "Volume %d outline has %d chapters (target %d), retry %d/3",
                volume_num, actual_count, chapters_per_volume, attempt + 1,
            )
            # Add a correction message for the next attempt.
            messages.append({"role": "assistant", "content": response})
            messages.append({
                "role": "user",
                "content": (
                    f"你生成了{actual_count}章，但要求是恰好{chapters_per_volume}章。"
                    f"请重新生成，chapter_plan 数组必须恰好包含{chapters_per_volume}个元素。"
                ),
            })

        # Validate required keys.
        required = {"volume_title", "volume_arc", "chapter_plan", "synopsis"}
        missing = required - set(outline.keys())
        if missing:
            logger.warning("Volume outline missing keys: %s", missing)
            for key in missing:
                outline[key] = "" if key != "chapter_plan" else []

        logger.info(
            "Volume %d outline generated: %d chapters planned",
            volume_num,
            len(outline.get("chapter_plan", [])),
        )

        return outline

    # ------------------------------------------------------------------
    # Outline optimization
    # ------------------------------------------------------------------

    def optimize_outline(
        self,
        current_outline: dict[str, Any],
        feedback: str,
        change_request: str,
    ) -> dict[str, Any]:
        """Modify an existing outline based on user feedback or plot needs.

        Preserves the structure and unchanged parts of the outline, only
        modifying portions relevant to the change request.

        Args:
            current_outline: The existing outline dict (total or volume).
            feedback: Human-readable reason or context for the change.
            change_request: Specific description of what to change.

        Returns:
            Updated outline dict with requested changes applied.

        Raises:
            ValueError: If the LLM response cannot be parsed as valid JSON.
        """
        # Detect outline type for more targeted instructions.
        is_total = "world_setting" in current_outline
        outline_type = "总大纲" if is_total else "卷大纲"

        style_name = current_outline.get("_meta", {}).get("style_name", "")
        style_prompt = ""
        if style_name:
            try:
                style_prompt = self._style.get_style_prompt(style_name)
            except KeyError:
                logger.debug("Style '%s' not found, proceeding without style injection", style_name)

        user_prompt = (
            f"请修改以下{outline_type}。\n\n"
            f"=== 修改原因 ===\n{feedback}\n\n"
            f"=== 具体修改要求 ===\n{change_request}\n\n"
            f"=== 当前大纲 ===\n{_serialize_context(current_outline)}\n\n"
            f"请按照原格式输出修改后的完整{outline_type}JSON。"
        )

        system_content = _SYSTEM_PROMPT_OPTIMIZE
        if style_prompt:
            system_content += f"\n\n{style_prompt}"

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_prompt},
        ]

        logger.info(
            "Optimizing %s outline (feedback=%s, change=%s)",
            outline_type,
            feedback[:80],
            change_request[:80],
        )

        response = self._llm.chat(
            messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )

        updated = _extract_json(response)

        # Carry over _meta if present and not in updated output.
        if "_meta" in current_outline and "_meta" not in updated:
            updated["_meta"] = current_outline["_meta"]

        logger.info("Outline optimization complete")

        return updated

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def temperature(self) -> float:
        """Sampling temperature used for LLM calls."""
        return self._temperature

    @temperature.setter
    def temperature(self, value: float) -> None:
        self._temperature = value

    @property
    def max_tokens(self) -> int:
        """Max tokens used for LLM outline generation calls."""
        return self._max_tokens

    @max_tokens.setter
    def max_tokens(self, value: int) -> None:
        self._max_tokens = value
