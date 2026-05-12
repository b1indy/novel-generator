"""Style configuration module for web novel generator.

Manages novel genre styles and author writing styles that get injected
into generation prompts. Presets are stored as module-level data
structures with a StyleManager class providing access methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar


@dataclass
class AuthorStyle:
    """Definition of an author's writing style preset.

    Attributes:
        name: Style identifier (e.g. "辰东式热血").
        description: Human-readable summary of the style.
        narrative_rhythm: Overall pacing ("快速"/"中等"/"慢速").
        dialogue_ratio: Proportion of dialogue, 0.0 to 1.0.
        description_detail: Level of environmental/detail description, 0.0 to 1.0.
        battle_style: How combat/fights are written.
        emotional_depth: Depth of emotional expression, 0.0 to 1.0.
        sentence_style: Sentence construction approach ("简洁"/"平衡"/"细腻").
        rhythm_detail: Extended pacing description for prompt injection.
        dialogue_detail: Extended dialogue description.
        detail_notes: Extended description detail notes.
        battle_notes: Extended battle style notes.
        emotion_notes: Extended emotional depth notes.
        sentence_notes: Extended sentence style notes.
    """

    name: str
    description: str
    narrative_rhythm: str
    dialogue_ratio: float
    description_detail: float
    battle_style: str
    emotional_depth: float
    sentence_style: str
    rhythm_detail: str = ""
    dialogue_detail: str = ""
    detail_notes: str = ""
    battle_notes: str = ""
    emotion_notes: str = ""
    sentence_notes: str = ""


# ---------------------------------------------------------------------------
# Built-in author-style presets (module-level, no file I/O needed)
# ---------------------------------------------------------------------------

AUTHOR_PRESETS: dict[str, AuthorStyle] = {
    "辰东式热血": AuthorStyle(
        name="辰东式热血",
        description="辰东式热血风格：史诗级战斗，宏大世界观，悲剧英雄主义，节奏快速",
        narrative_rhythm="快速",
        dialogue_ratio=0.30,
        description_detail=0.65,
        battle_style="气势磅礴，注重意境与力量感的结合",
        emotional_depth=0.60,
        sentence_style="简洁",
        rhythm_detail="快节奏，战斗与高潮密集，章节结尾常留悬念",
        dialogue_detail="对话简洁有力，角色语言个性鲜明，少冗长对白",
        detail_notes="中等偏高，注重场景氛围营造，擅长用环境烘托情绪",
        battle_notes="气势磅礴，注重意境与力量感的结合，战斗描写兼具诗意与震撼力",
        emotion_notes="深沉克制，以行动而非言语表达情感，悲剧色彩浓厚",
        sentence_notes="简洁有力，善用短句营造紧张感，关键时刻使用排比增强气势",
    ),
    "猫腻式文艺": AuthorStyle(
        name="猫腻式文艺",
        description="猫腻式文艺风格：文学性散文笔法，哲学思辨，复杂人物塑造，节奏适中",
        narrative_rhythm="中等",
        dialogue_ratio=0.40,
        description_detail=0.80,
        battle_style="写意留白，以智斗和意境为主",
        emotional_depth=0.85,
        sentence_style="细腻",
        rhythm_detail="中等节奏，张弛有度，文戏与武戏交替推进，注重铺垫与回响",
        dialogue_detail="对话富有哲理，角色间常有机锋之辩，潜台词丰富",
        detail_notes="细腻入微，善于捕捉人物微表情与环境细节，文字具有画面感",
        battle_notes="写意留白，以智斗和意境为主，不追求招式铺陈而重气势与心境",
        emotion_notes="深刻复杂，善写人物内心矛盾与成长，情感线细腻绵长",
        sentence_notes="文笔细腻，句式多变，兼具古典韵味与现代节奏感",
    ),
    "番茄式爽文": AuthorStyle(
        name="番茄式爽文",
        description="番茄式爽文风格：力量幻想，升级快速，正邪分明，节奏极快",
        narrative_rhythm="快速",
        dialogue_ratio=0.25,
        description_detail=0.45,
        battle_style="直截了当，突出力量碾压与等级压制",
        emotional_depth=0.40,
        sentence_style="简洁",
        rhythm_detail="极快节奏，升级突破与战斗胜利接连不断，爽点密集",
        dialogue_detail="对话直接明了，服务于剧情推进，少绕弯子",
        detail_notes="中等偏低，重点描写力量体系和战斗过程，环境描写简洁",
        battle_notes="直截了当，突出力量碾压与等级压制，战斗过程清晰利落",
        emotion_notes="简单直接，以热血与爽感为主，情感表达明快不拖沓",
        sentence_notes="简洁直白，以短句和动作描写驱动节奏，阅读流畅度高",
    ),
    "老鹰式细节": AuthorStyle(
        name="老鹰式细节",
        description="老鹰式细节风格：缜密世界观，政治权谋，慢热铺垫，节奏较慢",
        narrative_rhythm="慢速",
        dialogue_ratio=0.35,
        description_detail=0.90,
        battle_style="战术推演，注重逻辑与细节真实感",
        emotional_depth=0.70,
        sentence_style="平衡",
        rhythm_detail="慢节奏，注重铺垫与逻辑自洽，世界观逐步展开，厚积薄发",
        dialogue_detail="对话信息量大，常有暗藏机锋的政治对话与推理式交流",
        detail_notes="极高细腻度，对世界观设定、社会结构、修炼体系有详尽描写",
        battle_notes="战术推演式，注重逻辑与细节真实感，每场战斗有明确的策略逻辑",
        emotion_notes="深沉内敛，人物情感在漫长铺垫中自然流露，重理性轻煽情",
        sentence_notes="平稳扎实，句式均衡，以清晰准确为首要目标，不追求华丽",
    ),
    "唐家三少式": AuthorStyle(
        name="唐家三少式",
        description="唐家三少式风格：浪漫副线，清晰等级体系，团队羁绊，节奏适中",
        narrative_rhythm="中等",
        dialogue_ratio=0.38,
        description_detail=0.60,
        battle_style="团队配合，招式清晰，突出协作与友情力量",
        emotional_depth=0.65,
        sentence_style="平衡",
        rhythm_detail="中等节奏，感情线与战斗线交替推进，团队成长贯穿始终",
        dialogue_detail="对话自然流畅，角色间互动丰富，团队氛围感强",
        detail_notes="适中，力量体系与等级设定清晰，场景描写为剧情服务",
        battle_notes="团队配合为主，招式描述清晰有条理，突出协作精神与友情力量",
        emotion_notes="温暖真挚，以友情、爱情、师生情为情感主线，感人但不煽情",
        sentence_notes="流畅自然，平实中见功力，适合长时间连续阅读",
    ),
}


# ---------------------------------------------------------------------------
# Built-in genre presets
# ---------------------------------------------------------------------------

GENRE_PRESETS: dict[str, str] = {
    "玄幻": (
        "东方玄幻世界，可包含修炼体系（如炼气、筑基、金丹、元婴等境界）、"
        "灵药法宝、秘境探险、宗门势力争斗。世界通常由凡界与仙界构成多层结构，"
        "存在上古遗迹与失落传承。"
    ),
    "都市": (
        "现代都市背景，可融入异能、重生、系统流等元素。"
        "场景涵盖校园、职场、商业、娱乐圈等现代生活场景。"
        "注意现代社会的逻辑自洽，力量体系需与都市背景协调。"
    ),
    "科幻": (
        "科幻世界观，可涵盖星际文明、人工智能、基因改造、虚拟现实等主题。"
        "科技设定需有一定自洽性，可结合硬科幻或软科幻风格。"
        "常见元素：太空战舰、机甲、星联网、外星文明、末日废土。"
    ),
    "仙侠": (
        "中国传统仙侠世界，修炼体系以道家思想为基础。"
        "涵盖御剑飞行、炼丹炼器、天劫渡劫、飞升成仙等经典元素。"
        "世界体系通常包含人界、灵界、仙界等多重境界。"
        "注重因果轮回、天道规则等哲学概念。"
    ),
    "历史": (
        "以真实或架空历史为背景，注重时代氛围与历史逻辑。"
        "可融入权谋斗争、战争策略、科技改良、思想启蒙等元素。"
        "架空历史可改变历史走向但需保持时代合理性。"
        "常见题材：王朝争霸、科举仕途、商战经营、文化传播。"
    ),
    "悬疑": (
        "悬疑推理向，以谜题、案件、阴谋为核心驱动。"
        "注重线索铺设与反转设计，逻辑链需完整自洽。"
        "氛围营造偏冷峻压抑，善用信息差制造悬念。"
        "常见元素：密室推理、连环案件、身份谜团、记忆篡改、心理博弈。"
    ),
}


# ---------------------------------------------------------------------------
# StyleManager
# ---------------------------------------------------------------------------


class StyleManager:
    """Manages author styles and genre presets for novel generation.

    Provides methods to query built-in presets, create custom styles,
    and generate prompt-ready style strings for injection into LLM
    system prompts.

    Usage::

        manager = StyleManager()
        prompt = manager.get_style_prompt("辰东式热血")
        genre_guide = manager.get_genre_prompt("玄幻")
    """

    # Module-level presets used as defaults
    _builtin_authors: ClassVar[dict[str, AuthorStyle]] = AUTHOR_PRESETS
    _builtin_genres: ClassVar[dict[str, str]] = GENRE_PRESETS

    def __init__(self) -> None:
        """Initialize StyleManager with an empty custom-style registry."""
        self._custom_styles: dict[str, AuthorStyle] = {}

    # ------------------------------------------------------------------
    # Author style methods
    # ------------------------------------------------------------------

    def get_style(self, style_name: str) -> AuthorStyle:
        """Look up an author style by name.

        Searches custom styles first, then built-in presets.

        Args:
            style_name: Name of the style to look up.

        Returns:
            The matching AuthorStyle.

        Raises:
            KeyError: If the style name is not found.
        """
        if style_name in self._custom_styles:
            return self._custom_styles[style_name]
        if style_name in self._builtin_authors:
            return self._builtin_authors[style_name]
        raise KeyError(
            f"未找到风格'{style_name}'。可用风格: {', '.join(self._all_style_names())}"
        )

    def get_style_prompt(self, style_name: str) -> str:
        """Generate a prompt-ready Chinese description of a writing style.

        The returned string is designed to be injected into the system
        prompt of an LLM chapter-generation call.

        Args:
            style_name: Name of the author style preset.

        Returns:
            A formatted multi-line string describing the writing style.

        Example::

            >>> manager = StyleManager()
            >>> print(manager.get_style_prompt("辰东式热血"))
            写作风格要求：
            - 叙事节奏：快节奏，战斗与高潮密集
            - 对话占比：中等（约30%），对话简洁有力
            ...
        """
        style = self.get_style(style_name)

        dialogue_pct = int(style.dialogue_ratio * 100)

        def _dialogue_label(ratio: float) -> str:
            if ratio <= 0.25:
                return "偏低"
            if ratio <= 0.35:
                return "中等"
            return "偏高"

        return (
            f"写作风格要求：\n"
            f"- 叙事节奏：{style.rhythm_detail}\n"
            f"- 对话占比：{_dialogue_label(style.dialogue_ratio)}"
            f"（约{dialogue_pct}%），{style.dialogue_detail}\n"
            f"- 描写细腻度：{style.detail_notes}\n"
            f"- 战斗描写：{style.battle_notes}\n"
            f"- 情感描写：{style.emotion_notes}\n"
            f"- 句式风格：{style.sentence_notes}"
        )

    def create_custom_style(self, name: str, **params: str | float) -> AuthorStyle:
        """Define a custom author style.

        Args:
            name: Unique name for the custom style.
            **params: Style parameters. Supported keys match the
                      :class:`AuthorStyle` fields:
                      description, narrative_rhythm, dialogue_ratio,
                      description_detail, battle_style, emotional_depth,
                      sentence_style, rhythm_detail, dialogue_detail,
                      detail_notes, battle_notes, emotion_notes,
                      sentence_notes.

        Returns:
            The newly created :class:`AuthorStyle` instance.

        Raises:
            ValueError: If the style name conflicts with a built-in or
                        an existing custom style.
        """
        if name in self._builtin_authors:
            raise ValueError(
                f"'{name}' 是内置风格，不可覆盖。请使用其他名称。"
            )
        if name in self._custom_styles:
            raise ValueError(
                f"自定义风格'{name}'已存在。请使用其他名称或先删除旧风格。"
            )

        # Provide defaults for missing fields so callers can be minimal.
        defaults: dict[str, str | float] = {
            "description": name,
            "narrative_rhythm": "中等",
            "dialogue_ratio": 0.35,
            "description_detail": 0.60,
            "battle_style": "标准战斗描写",
            "emotional_depth": 0.60,
            "sentence_style": "平衡",
            "rhythm_detail": "中等节奏",
            "dialogue_detail": "对话自然流畅",
            "detail_notes": "描写细腻度适中",
            "battle_notes": "标准战斗描写",
            "emotion_notes": "情感表达自然",
            "sentence_notes": "句式平衡",
        }
        merged = {**defaults, **params}
        merged["name"] = name

        style = AuthorStyle(**merged)  # type: ignore[arg-type]
        self._custom_styles[name] = style
        return style

    def delete_custom_style(self, name: str) -> None:
        """Remove a custom style.

        Args:
            name: Name of the custom style to delete.

        Raises:
            KeyError: If the name is not a custom style.
        """
        if name not in self._custom_styles:
            raise KeyError(f"自定义风格'{name}'不存在。")
        del self._custom_styles[name]

    def list_styles(self) -> list[dict[str, str]]:
        """Return all available author styles with descriptions.

        Returns:
            A list of dicts, each with ``name`` and ``description`` keys.
        """
        results: list[dict[str, str]] = []
        for name, style in self._builtin_authors.items():
            results.append({"name": name, "description": style.description})
        for name, style in self._custom_styles.items():
            results.append({"name": name, "description": style.description})
        return results

    # ------------------------------------------------------------------
    # Genre methods
    # ------------------------------------------------------------------

    def get_genre_prompt(self, genre_name: str) -> str:
        """Generate world-building guidance for a specific novel genre.

        Args:
            genre_name: Name of the genre preset.

        Returns:
            A string with world-building hints for the genre.

        Raises:
            KeyError: If the genre name is not found.
        """
        if genre_name in self._builtin_genres:
            return f"世界观构建指导（{genre_name}）：\n{self._builtin_genres[genre_name]}"
        raise KeyError(
            f"未找到流派'{genre_name}'。可用流派: {', '.join(self._builtin_genres)}"
        )

    def list_genres(self) -> list[str]:
        """Return the names of all available genre presets."""
        return list(self._builtin_genres.keys())

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _all_style_names(self) -> list[str]:
        """Return all known style names (built-in + custom)."""
        names: list[str] = list(self._builtin_authors.keys())
        names.extend(self._custom_styles.keys())
        return names
