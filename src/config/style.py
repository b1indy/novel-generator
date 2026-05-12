"""Style configuration module for web novel generator.

Manages novel genre styles and author writing styles that get injected
into generation prompts. Presets are stored as module-level data
structures with a StyleManager class providing access methods.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

logger = logging.getLogger(__name__)

# Directory for persisting individual custom style JSON files
CUSTOM_STYLES_DIR: str = "data/custom_styles/"


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
    genre_hints: str = ""


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
        """Initialize StyleManager and load any persisted custom styles."""
        self._custom_styles: dict[str, AuthorStyle] = {}
        self._custom_bases: dict[str, str | None] = {}
        self._load_custom_styles_from_dir()

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

        Works for both built-in presets (which carry rich detail text) and
        custom styles (where the prompt is built from core parameters).

        Args:
            style_name: Name of the author style preset.

        Returns:
            A formatted multi-line string describing the writing style.
        """
        style = self.get_style(style_name)

        # Custom styles: build prompt from core parameters so it always
        # reflects the user's current values, even after edits.
        if style_name in self._custom_styles:
            return self._build_prompt_from_params(style)

        # Built-in styles: use rich pre-written detail fields.
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

    def create_custom_style(
        self, name: str, base_style: str | None = None, **params: str | float
    ) -> dict[str, Any]:
        """Define a custom author style, optionally based on an existing one.

        Args:
            name: Unique name for the custom style.
            base_style: If provided, start from this style's parameters and
                        override with any supplied **params.
            **params: Style parameter overrides. Supported keys:
                      narrative_rhythm, dialogue_ratio, description_detail,
                      battle_style, emotional_depth, sentence_style,
                      genre_hints, description.

        Returns:
            A dict of the core style parameters (narrative_rhythm,
            dialogue_ratio, description_detail, battle_style,
            emotional_depth, sentence_style, genre_hints).

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

        # Determine base parameter values.
        if base_style is not None:
            base = self.get_style(base_style)
            base_params: dict[str, Any] = {
                "narrative_rhythm": base.narrative_rhythm,
                "dialogue_ratio": base.dialogue_ratio,
                "description_detail": base.description_detail,
                "battle_style": base.battle_style,
                "emotional_depth": base.emotional_depth,
                "sentence_style": base.sentence_style,
                "genre_hints": "",
            }
            description = params.pop("description", f"{name}（基于{base_style}）")
        else:
            base_params = {
                "narrative_rhythm": "中等",
                "dialogue_ratio": 0.35,
                "description_detail": 0.60,
                "battle_style": "标准战斗描写",
                "emotional_depth": 0.60,
                "sentence_style": "平衡",
                "genre_hints": "",
            }
            description = params.pop("description", name)

        # Merge overrides into base.
        core_fields = {
            "narrative_rhythm", "dialogue_ratio", "description_detail",
            "battle_style", "emotional_depth", "sentence_style", "genre_hints",
        }
        for k, v in params.items():
            if k in core_fields:
                base_params[k] = v

        # Build the AuthorStyle with auto-generated detail text.
        style = self._params_to_author_style(name, base_params, description)
        self._custom_styles[name] = style
        self._custom_bases[name] = base_style

        # Return the core params dict (caller can save it later).
        return dict(base_params)

    def delete_custom_style(self, name: str) -> None:
        """Remove a custom style from memory and persistent storage.

        Args:
            name: Name of the custom style to delete.

        Raises:
            KeyError: If the name is not a custom style.
        """
        if name not in self._custom_styles:
            raise KeyError(f"自定义风格'{name}'不存在。")
        del self._custom_styles[name]
        self._custom_bases.pop(name, None)
        # Also remove the persisted JSON file if present.
        file_path = Path(CUSTOM_STYLES_DIR) / f"{name}.json"
        try:
            file_path.unlink(missing_ok=True)
        except OSError:
            pass

    def list_styles(self) -> list[dict[str, str]]:
        """Return all available author styles with descriptions.

        Built-in styles are listed as-is; custom styles are prefixed with
        ``[自定义]`` so UIs can distinguish them.

        Returns:
            A list of dicts, each with ``name`` and ``description`` keys.
        """
        results: list[dict[str, str]] = []
        for name, style in self._builtin_authors.items():
            results.append({"name": name, "description": style.description})
        for name, style in self._custom_styles.items():
            results.append({
                "name": f"[自定义] {name}",
                "description": style.description,
            })
        return results

    def get_style_params(self, style_name: str) -> dict[str, Any]:
        """Return the core customization parameters of a style as a dict.

        Args:
            style_name: Name of the style to inspect.

        Returns:
            Dict with keys: narrative_rhythm, dialogue_ratio,
            description_detail, battle_style, emotional_depth,
            sentence_style, and the extended detail notes.
        """
        style = self.get_style(style_name)
        return {
            "narrative_rhythm": style.narrative_rhythm,
            "dialogue_ratio": style.dialogue_ratio,
            "description_detail": style.description_detail,
            "battle_style": style.battle_style,
            "emotional_depth": style.emotional_depth,
            "sentence_style": style.sentence_style,
            "rhythm_detail": style.rhythm_detail,
            "dialogue_detail": style.dialogue_detail,
            "detail_notes": style.detail_notes,
            "battle_notes": style.battle_notes,
            "emotion_notes": style.emotion_notes,
            "sentence_notes": style.sentence_notes,
        }

    def save_custom_style(
        self, name: str, style_params: dict[str, Any]
    ) -> None:
        """Persist a single custom style to its own JSON file.

        The file is written under ``CUSTOM_STYLES_DIR/<name>.json``.
        Call this separately after :meth:`create_custom_style` to make
        the style survive restarts.

        Args:
            name: Name of the custom style to save.
            style_params: Dict with all style fields (narrative_rhythm,
                          dialogue_ratio, description_detail, battle_style,
                          emotional_depth, sentence_style, genre_hints).
        """
        styles_dir = Path(CUSTOM_STYLES_DIR)
        styles_dir.mkdir(parents=True, exist_ok=True)

        data: dict[str, Any] = {
            "name": name,
            "base_style": self._custom_bases.get(name, ""),
            "params": style_params,
        }
        file_path = styles_dir / f"{name}.json"
        file_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("已保存自定义风格 '%s' 到 %s", name, file_path)

    def load_custom_styles(self) -> dict[str, dict[str, Any]]:
        """Load all saved custom styles from the custom-styles directory.

        Returns:
            Dict mapping style name to core params dict.
        """
        return self._load_custom_styles_from_dir()

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

    def _load_custom_styles_from_dir(self) -> dict[str, dict[str, Any]]:
        """Scan ``CUSTOM_STYLES_DIR`` for JSON files and register each."""
        styles_dir = Path(CUSTOM_STYLES_DIR)
        if not styles_dir.is_dir():
            return {}

        loaded: dict[str, dict[str, Any]] = {}
        for file_path in sorted(styles_dir.glob("*.json")):
            try:
                data = json.loads(file_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning(
                    "无法读取自定义风格文件 %s: %s", file_path, exc
                )
                continue

            if not isinstance(data, dict):
                continue

            name = data.get("name") or file_path.stem
            params = data.get("params")
            if not isinstance(params, dict):
                continue

            if name in self._builtin_authors:
                continue
            if name in self._custom_styles:
                loaded[name] = params
                continue

            base_style = data.get("base_style", "") or None
            description = str(name)
            if base_style:
                description = f"{name}（基于{base_style}）"

            style = self._params_to_author_style(name, params, description)
            self._custom_styles[name] = style
            self._custom_bases[name] = base_style
            loaded[name] = params

        if loaded:
            logger.info(
                "从 %s 加载了 %d 个自定义风格", CUSTOM_STYLES_DIR, len(loaded)
            )
        return loaded

    @staticmethod
    def _params_to_author_style(
        name: str, params: dict[str, Any], description: str = ""
    ) -> AuthorStyle:
        """Build an :class:`AuthorStyle` from a core-params dict.

        Missing detail fields are filled with auto-generated text so
        :meth:`_build_prompt_from_params` produces quality output.
        """
        narrative_rhythm = str(params.get("narrative_rhythm", "中等"))
        dialogue_ratio = float(params.get("dialogue_ratio", 0.35))
        description_detail = float(params.get("description_detail", 0.60))
        battle_style = str(params.get("battle_style", "标准战斗描写"))
        emotional_depth = float(params.get("emotional_depth", 0.60))
        sentence_style = str(params.get("sentence_style", "平衡"))
        genre_hints = str(params.get("genre_hints", ""))

        rhythm_map: dict[str, str] = {
            "快速": "快节奏，战斗与高潮密集，章节结尾常留悬念",
            "fast": "快节奏，战斗与高潮密集，章节结尾常留悬念",
            "中等": "中等节奏，张弛有度，文戏与武戏交替推进",
            "moderate": "中等节奏，张弛有度，文戏与武戏交替推进",
            "慢速": "慢节奏，注重铺垫与逻辑自洽，世界观逐步展开",
            "slow": "慢节奏，注重铺垫与逻辑自洽，世界观逐步展开",
        }

        sentence_map: dict[str, str] = {
            "简洁": "简洁有力，善用短句营造紧张感",
            "concise": "简洁有力，善用短句营造紧张感",
            "平衡": "流畅自然，句式均衡",
            "balanced": "流畅自然，句式均衡",
            "细腻": "文笔细腻，句式多变，注重细节刻画",
            "elaborate": "文笔细腻，句式多变，注重细节刻画",
        }

        detail_map: dict[float, str] = {
            0.9: "极高细腻度，对世界观设定、环境氛围有详尽描写",
            0.8: "细腻入微，善于捕捉微表情与环境细节",
            0.7: "中等偏高，注重场景氛围营造，擅长用环境烘托情绪",
            0.6: "描写细腻度适中，场景为剧情服务",
            0.5: "中等偏低，重点描写核心元素，环境描写简洁",
            0.4: "简明扼要，以动作和对话驱动叙事",
        }

        emotion_map: dict[float, str] = {
            0.9: "深刻复杂，善写人物内心矛盾与成长，情感线细腻绵长",
            0.8: "深沉内敛，人物情感在漫长铺垫中自然流露",
            0.7: "温暖真挚，情感表达自然而克制",
            0.6: "情感表达自然，重行动轻言语",
            0.5: "简单直接，以热血与爽感为主",
            0.4: "轻情感重剧情，情感为辅线快速推进",
        }

        def _closest_key(d: dict[float, str], target: float) -> str:
            return min(d.keys(), key=lambda k: abs(k - target))

        rhythm_detail = rhythm_map.get(
            narrative_rhythm, "节奏适中，根据剧情需要调整"
        )
        sentence_notes = sentence_map.get(
            sentence_style, "句式平衡，阅读流畅"
        )

        if dialogue_ratio <= 0.25:
            dialogue_detail = "以叙述和描写推动剧情，对话简洁直接"
        elif dialogue_ratio <= 0.35:
            dialogue_detail = "对话简洁有力，语言个性鲜明"
        else:
            dialogue_detail = "以对话推动剧情，角色互动丰富，潜台词丰富"

        detail_notes = detail_map.get(description_detail) or detail_map[
            _closest_key(detail_map, description_detail)
        ]
        battle_notes_val = battle_style
        emotion_notes = emotion_map.get(emotional_depth) or emotion_map[
            _closest_key(emotion_map, emotional_depth)
        ]

        return AuthorStyle(
            name=name,
            description=description or name,
            narrative_rhythm=narrative_rhythm,
            dialogue_ratio=dialogue_ratio,
            description_detail=description_detail,
            battle_style=battle_style,
            emotional_depth=emotional_depth,
            sentence_style=sentence_style,
            rhythm_detail=rhythm_detail,
            dialogue_detail=dialogue_detail,
            detail_notes=detail_notes,
            battle_notes=battle_notes_val,
            emotion_notes=emotion_notes,
            sentence_notes=sentence_notes,
            genre_hints=genre_hints,
        )

    def _build_prompt_from_params(self, style: AuthorStyle) -> str:
        """Build a prompt string from an AuthorStyle's parameters.

        Used for custom styles where the prompt is generated from the
        core parameters instead of hand-written detail fields.
        """
        dialogue_pct = int(style.dialogue_ratio * 100)

        def _dialogue_label(ratio: float) -> str:
            if ratio <= 0.25:
                return "偏低"
            if ratio <= 0.35:
                return "中等"
            return "偏高"

        lines = [
            "写作风格要求：",
            f"- 叙事节奏：{style.rhythm_detail}",
            f"- 对话占比：{_dialogue_label(style.dialogue_ratio)}"
            f"（约{dialogue_pct}%），{style.dialogue_detail}",
            f"- 描写细腻度：{style.detail_notes}",
            f"- 战斗描写：{style.battle_notes}",
            f"- 情感描写：{style.emotion_notes}",
            f"- 句式风格：{style.sentence_notes}",
        ]

        if getattr(style, "genre_hints", ""):
            lines.append(f"- 流派倾向：{style.genre_hints}")

        return "\n".join(lines)
