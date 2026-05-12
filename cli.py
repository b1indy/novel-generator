#!/usr/bin/env python3
"""CLI entry point for the web novel generator.

Ties together all modules: LLM client, style manager, storage, memory,
outline, generator, and auditor.  Provides an interactive command-line
interface powered by ``argparse`` with ``rich``-formatted output.

Usage::

    python cli.py new
    python cli.py list
    python cli.py status --novel my-novel
    python cli.py write --novel my-novel --volume 1
"""

from __future__ import annotations

import argparse
import locale
import logging
import os
import sys

# ---------------------------------------------------------------------------
# Multi-byte character input fix
#   Chinese characters in UTF-8 are 3 bytes.  Without proper locale /
#   readline configuration the terminal may treat each byte as a separate
#   character, requiring multiple backspace presses to delete one glyph.
# ---------------------------------------------------------------------------


def _setup_encoding() -> None:
    """Configure locale and readline so multi-byte UTF-8 input works properly.

    Without this, Chinese characters (3 bytes each in UTF-8) are treated
    as multiple characters by readline, requiring multiple backspace
    presses to delete a single glyph.
    """
    # 1. Ensure environment-level encoding — prefer a CJK-aware locale so
    #    readline correctly computes character display widths.
    for var in ("LANG", "LC_CTYPE", "LC_ALL"):
        val = os.environ.get(var, "")
        if val and "utf" not in val.lower():
            os.environ[var] = "zh_CN.UTF-8"

    if not any("UTF-8" in os.environ.get(v, "") for v in ("LANG", "LC_ALL", "LC_CTYPE")):
        os.environ["LANG"] = "zh_CN.UTF-8"

    # 2. Set Python locale from environment.
    #    Try CJK-aware locale first (better character-width tables for
    #    Chinese), fall back to generic UTF-8.
    ctype_set = False
    for loc in ("zh_CN.UTF-8", "en_US.UTF-8", "C.UTF-8"):
        try:
            locale.setlocale(locale.LC_CTYPE, loc)
            ctype_set = True
            break
        except locale.Error:
            continue

    if not ctype_set:
        try:
            locale.setlocale(locale.LC_CTYPE, "")
        except locale.Error:
            pass

    # 3. Ensure stdin/stdout use UTF-8 (belt-and-suspenders with
    #    PYTHONIOENCODING for subprocess / pipe scenarios).
    if sys.stdin.encoding and "utf" not in sys.stdin.encoding.lower():
        import io
        sys.stdin = io.TextIOWrapper(
            sys.stdin.buffer, encoding="utf-8", errors="replace"
        )
    if sys.stdout.encoding and "utf" not in sys.stdout.encoding.lower():
        import io
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
        )


_setup_encoding()
del _setup_encoding
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Rich imports (required dependency)
# ---------------------------------------------------------------------------
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.prompt import Prompt, Confirm, IntPrompt
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from src.llm.client import LLMClient, LLMError
from src.config.style import StyleManager
from src.storage.novel_store import NovelStore
from src.storage.table_store import TableStore
from src.memory.character import CharacterManager
from src.memory.item import ItemManager
from src.memory.foreshadowing import ForeshadowingManager
from src.memory.summary import SummaryGenerator
from src.outline.generator import OutlineGenerator
from src.outline.manager import OutlineManager
from src.generator.chapter_writer import ChapterWriter
from src.generator.volume_writer import VolumeWriter
from src.auditor.auditor import Auditor

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("cli")

console = Console()
"""Global rich console used throughout the CLI."""


# ===================================================================
# Config loading
# ===================================================================


def load_config_path() -> Path:
    """Locate a usable config file.

    Returns the path to ``config.yaml`` if it exists, otherwise
    ``config.yaml.example`` as a fallback.  The caller is expected to
    show a warning when the example file is used.
    """
    config_yaml = Path("config.yaml")
    if config_yaml.is_file():
        return config_yaml
    return Path("config.yaml.example")


def load_generation_config(config_path: Path) -> dict[str, int]:
    """Extract generation parameters from the YAML config.

    Returns a ``dict`` with keys ``chapters_per_volume`` and
    ``words_per_chapter``, defaulting to 50 and 5000 respectively when
    the section or keys are missing.
    """
    defaults = {"chapters_per_volume": 50, "words_per_chapter": 5000}
    try:
        raw = yaml.safe_load(config_path.read_text())
    except (yaml.YAMLError, OSError):
        return defaults
    if not isinstance(raw, dict):
        return defaults
    generation = raw.get("generation")
    if not isinstance(generation, dict):
        return defaults
    return {
        "chapters_per_volume": int(
            generation.get("chapters_per_volume", defaults["chapters_per_volume"])
        ),
        "words_per_chapter": int(
            generation.get("words_per_chapter", defaults["words_per_chapter"])
        ),
    }


# ===================================================================
# Initialization helpers
# ===================================================================


class AppContext:
    """Holds all initialized managers and stores for the current session.

    Created once at startup so commands do not need to re-instantiate
    the full dependency chain.
    """

    llm_client: LLMClient
    style_manager: StyleManager
    novel_store: NovelStore
    outline_generator: OutlineGenerator
    outline_manager: OutlineManager
    chapters_per_volume: int
    words_per_chapter: int

    def __init__(self, config_path: str) -> None:
        self.llm_client = LLMClient(config_path)
        self.style_manager = StyleManager()
        self.novel_store = NovelStore("data/novels")

        # Load generation params from config.
        gen_cfg = load_generation_config(Path(config_path))
        self.chapters_per_volume = gen_cfg["chapters_per_volume"]
        self.words_per_chapter = gen_cfg["words_per_chapter"]

        # Load custom styles.
        self.style_manager.load_custom_styles()

        self.outline_generator = OutlineGenerator(self.llm_client, self.style_manager)
        self.outline_manager = OutlineManager(
            novel_store=self.novel_store,
            outline_generator=self.outline_generator,
            style_manager=self.style_manager,
            base_path="data/novels",
        )

    def make_table_store(self, novel_name: str) -> TableStore:
        return TableStore(str(Path("data/novels") / novel_name))

    def make_character_manager(self, novel_name: str) -> CharacterManager:
        return CharacterManager(str(Path("data/novels") / novel_name))

    def make_item_manager(self, novel_name: str) -> ItemManager:
        return ItemManager(str(Path("data/novels") / novel_name))

    def make_foreshadowing_manager(self, novel_name: str) -> ForeshadowingManager:
        return ForeshadowingManager(str(Path("data/novels") / novel_name))

    def make_chapter_writer(self, novel_name: str) -> ChapterWriter:
        return ChapterWriter(
            llm_client=self.llm_client,
            style_manager=self.style_manager,
            character_manager=self.make_character_manager(novel_name),
            item_manager=self.make_item_manager(novel_name),
            foreshadowing_manager=self.make_foreshadowing_manager(novel_name),
        )

    def make_volume_writer(self, novel_name: str) -> VolumeWriter:
        return VolumeWriter(
            llm_client=self.llm_client,
            chapter_writer=self.make_chapter_writer(novel_name),
            character_manager=self.make_character_manager(novel_name),
            item_manager=self.make_item_manager(novel_name),
            foreshadowing_manager=self.make_foreshadowing_manager(novel_name),
            outline_manager=self.outline_manager,
            novel_store=self.novel_store,
            table_store=self.make_table_store(novel_name),
            summary_generator=SummaryGenerator(self.llm_client),
        )

    def make_auditor(self) -> Auditor:
        return Auditor(self.llm_client)


# ===================================================================
# Command: new
# ===================================================================


def cmd_new(ctx: AppContext) -> None:
    """Interactive wizard to create a new novel."""
    console.print()
    console.print(Panel.fit(
        "[bold cyan]创建新小说[/bold cyan]",
        border_style="cyan",
    ))

    # 1. Novel name.
    novel_name = Prompt.ask("请输入小说名称（用于目录标识，英文/拼音）")
    if ctx.novel_store.novel_exists(novel_name):
        console.print(
            f"[red]小说'{novel_name}'已存在。请使用其他名称。[/red]"
        )
        return

    title = Prompt.ask("请输入小说标题（中文显示名称）", default=novel_name)

    # 2. Select genre.
    genres = ctx.style_manager.list_genres()
    console.print()
    console.print("[bold]可用流派：[/bold]")
    for i, g in enumerate(genres, 1):
        console.print(f"  {i}. {g}")
    genre_idx = IntPrompt.ask(
        "请选择流派（输入编号）",
        default=1,
        show_choices=False,
    )
    if 1 <= genre_idx <= len(genres):
        genre = genres[genre_idx - 1]
    else:
        console.print("[red]无效选择，使用第一个流派。[/red]")
        genre = genres[0]
    console.print(f"  [green]已选择流派: {genre}[/green]")

    # 3. Select author style.
    styles = ctx.style_manager.list_styles()
    console.print()
    console.print("[bold]可用作家风格：[/bold]")
    style_table = Table(show_header=True, header_style="bold")
    style_table.add_column("#", style="dim")
    style_table.add_column("风格名称")
    style_table.add_column("简介")
    for i, s in enumerate(styles, 1):
        style_table.add_row(str(i), s["name"], s["description"])
    console.print(style_table)
    style_idx = IntPrompt.ask(
        "请选择作家风格（输入编号）",
        default=1,
        show_choices=False,
    )
    if 1 <= style_idx <= len(styles):
        style_name = styles[style_idx - 1]["name"]
    else:
        console.print("[red]无效选择，使用第一种风格。[/red]")
        style_name = styles[0]["name"]
    # Strip "[自定义] " prefix introduced by list_styles() so lookups work.
    _display_name = style_name
    if style_name.startswith("[自定义] "):
        style_name = style_name[len("[自定义] "):]
    console.print(f"  [green]已选择风格: {_display_name}[/green]")

    # 3.5 Style customization.
    style_name = _customize_style(ctx, style_name)

    # 4. Target words.
    target_words = IntPrompt.ask(
        "请输入目标总字数",
        default=2_000_000,
    )

    # 5. Show summary and confirm.
    console.print()
    summary_panel = Panel.fit(
        f"[bold]小说名称:[/bold] {novel_name}\n"
        f"[bold]标题:[/bold] {title}\n"
        f"[bold]流派:[/bold] {genre}\n"
        f"[bold]作家风格:[/bold] {style_name}\n"
        f"[bold]目标字数:[/bold] {target_words:,}",
        title="创建确认",
        border_style="yellow",
    )
    console.print(summary_panel)

    if not Confirm.ask("确认创建？", default=True):
        console.print("[yellow]已取消。[/yellow]")
        return

    # 6. Create novel and generate total outline.
    with console.status("[cyan]正在创建小说并生成总大纲...[/cyan]", spinner="dots"):
        try:
            ctx.novel_store.create_novel(
                novel_name,
                {
                    "title": title,
                    "author_style": style_name,
                    "genre": genre,
                },
            )
        except Exception as exc:
            console.print(f"[red]创建小说失败: {exc}[/red]")
            return

    # 7-9. Generate, display, and iterate on total outline.
    while True:
        with console.status("[cyan]正在生成总大纲（LLM调用中）...[/cyan]", spinner="dots"):
            try:
                total_outline = ctx.outline_manager.create_total_outline(
                    novel_name=novel_name,
                    title=title,
                    genre=genre,
                    style_name=style_name,
                    target_words=target_words,
                )
            except Exception as exc:
                console.print(f"[red]生成总大纲失败: {exc}[/red]")
                return

        # Display the outline.
        _display_total_outline(total_outline)

        choice = Prompt.ask(
            "大纲是否满意？",
            choices=["y", "n", "q"],
            default="y",
        )
        if choice == "y":
            break
        if choice == "q":
            console.print("[yellow]已取消。[/yellow]")
            return

        # Changes requested.
        feedback = Prompt.ask("请描述大纲的问题或需要修改的方向")
        change_request = Prompt.ask("具体修改要求", default="请根据上述反馈修改大纲")
        with console.status("[cyan]正在优化大纲...[/cyan]", spinner="dots"):
            try:
                total_outline = ctx.outline_manager.update_total_outline(
                    novel_name=novel_name,
                    feedback=feedback,
                    change_request=change_request,
                )
            except Exception as exc:
                console.print(f"[red]优化大纲失败: {exc}[/red]")
                return
        _display_total_outline(total_outline)

    # 10. Generate first volume outline.
    with console.status("[cyan]正在生成第一卷大纲...[/cyan]", spinner="dots"):
        try:
            volume_outline = ctx.outline_manager.create_volume_outline(
                novel_name=novel_name,
                volume_num=1,
                chapters_per_volume=ctx.chapters_per_volume,
            )
        except Exception as exc:
            console.print(f"[red]生成第一卷大纲失败: {exc}[/red]")
            return

    _display_volume_outline(volume_outline, volume_num=1)

    # 11. Ask to start writing.
    if Confirm.ask("是否开始生成第一卷？", default=True):
        _write_volume(ctx, novel_name, 1, ctx.words_per_chapter)
    else:
        console.print(
            "[yellow]可以稍后使用 'python cli.py write --novel "
            f"{novel_name} --volume 1' 开始生成。[/yellow]"
        )


# ===================================================================
# Command: outline
# ===================================================================


def cmd_outline(ctx: AppContext, args: argparse.Namespace) -> None:
    """Manage novel outlines."""
    novel_name = _require_novel(ctx, args.novel)
    if novel_name is None:
        return

    if args.update:
        # Update total outline.
        console.print("[bold]更新总大纲[/bold]")
        feedback = Prompt.ask("请描述修改原因或上下文")
        change_request = Prompt.ask("具体修改要求", default="请根据上述反馈修改大纲")
        with console.status("[cyan]正在优化总大纲...[/cyan]", spinner="dots"):
            try:
                updated = ctx.outline_manager.update_total_outline(
                    novel_name=novel_name,
                    feedback=feedback,
                    change_request=change_request,
                )
            except Exception as exc:
                console.print(f"[red]更新大纲失败: {exc}[/red]")
                return
        _display_total_outline(updated)
        console.print("[green]总大纲已更新并保存。[/green]")

    elif args.volume is not None:
        # Generate a specific volume outline.
        volume_num = args.volume
        with console.status(
            f"[cyan]正在生成第{volume_num}卷大纲...[/cyan]", spinner="dots"
        ):
            try:
                vol_outline = ctx.outline_manager.create_volume_outline(
                    novel_name=novel_name,
                    volume_num=volume_num,
                    chapters_per_volume=ctx.chapters_per_volume,
                )
            except Exception as exc:
                console.print(f"[red]生成第{volume_num}卷大纲失败: {exc}[/red]")
                return
        _display_volume_outline(vol_outline, volume_num=volume_num)

    else:
        # View outlines.
        summary = ctx.outline_manager.get_outline_summary(novel_name)
        _display_outline_summary(summary)


# ===================================================================
# Command: write
# ===================================================================


def cmd_write(ctx: AppContext, args: argparse.Namespace) -> None:
    """Generate a volume."""
    novel_name = _require_novel(ctx, args.novel)
    if novel_name is None:
        return
    volume_num: int = args.volume
    words: int = args.words if args.words is not None else ctx.words_per_chapter
    _write_volume(ctx, novel_name, volume_num, words)


# ===================================================================
# Command: audit
# ===================================================================


def cmd_audit(ctx: AppContext, args: argparse.Namespace) -> None:
    """Audit a volume."""
    novel_name = _require_novel(ctx, args.novel)
    if novel_name is None:
        return
    volume_num: int = args.volume

    # Load volume outline and chapters.
    try:
        volume_outline = ctx.novel_store.load_volume_outline(novel_name, volume_num)
    except Exception:
        console.print(f"[red]未找到第{volume_num}卷大纲。[/red]")
        return
    if not volume_outline:
        console.print(f"[red]第{volume_num}卷大纲为空。[/red]")
        return

    try:
        chapters = ctx.novel_store.load_volume(novel_name, volume_num)
    except Exception:
        console.print(f"[red]无法加载第{volume_num}卷章节。[/red]")
        return
    if not chapters:
        console.print(f"[red]第{volume_num}卷没有章节。[/red]")
        return

    # Gather memory tables.
    tbl = ctx.make_table_store(novel_name)
    memory_tables: dict[str, Any] = {
        "characters": tbl.load_characters(),
        "items": tbl.load_items(),
        "foreshadowing": tbl.load_foreshadowing(),
    }

    # Build a text representation of the volume outline for the auditor.
    import json
    outline_text = json.dumps(volume_outline, ensure_ascii=False, indent=2)

    # Run audit.
    auditor = ctx.make_auditor()
    with console.status("[cyan]正在进行审计（LLM调用中）...[/cyan]", spinner="dots"):
        try:
            report = auditor.audit_volume(
                novel_name=novel_name,
                volume_num=volume_num,
                volume_outline=outline_text,
                chapters=chapters,
                memory_tables=memory_tables,
            )
        except Exception as exc:
            console.print(f"[red]审计失败: {exc}[/red]")
            return

    _display_audit_report(report)

    # Fix if requested.
    if args.fix:
        if not report.all_issues:
            console.print("[green]没有问题需要修复。[/green]")
            return

        if not Confirm.ask(
            f"发现 {len(report.all_issues)} 个问题，是否自动修复？", default=True
        ):
            return

        with console.status("[cyan]正在修复问题...[/cyan]", spinner="dots"):
            try:
                fixed_chapters = auditor.fix_issues(
                    novel_name=novel_name,
                    volume_num=volume_num,
                    audit_report=report,
                    chapters=chapters,
                )
            except Exception as exc:
                console.print(f"[red]修复失败: {exc}[/red]")
                return

        # Save fixed chapters.
        for idx, ch in enumerate(fixed_chapters):
            ch_num = idx + 1
            title = ch.get("title", f"第{ch_num}章")
            content = ch.get("content", "")
            ctx.novel_store.save_chapter(novel_name, volume_num, ch_num, content, title)

        console.print("[green]修复完成，已保存所有章节。[/green]")


# ===================================================================
# Command: status
# ===================================================================


def cmd_status(ctx: AppContext, args: argparse.Namespace) -> None:
    """Show current progress for a novel."""
    novel_name = _require_novel(ctx, args.novel)
    if novel_name is None:
        return

    summary = ctx.outline_manager.get_outline_summary(novel_name)
    tbl = ctx.make_table_store(novel_name)

    meta = tbl.load_meta()
    characters = tbl.load_characters()
    foreshadowing = tbl.load_foreshadowing()
    volume_summaries = tbl.load_volume_summaries()

    writen_volumes = summary["volumes_written"]
    planned_volumes = summary["volumes_planned"]
    next_volume = writen_volumes + 1 if writen_volumes < planned_volumes else None

    # Count characters.
    active_chars = sum(
        1 for c in characters.values()
        if c.get("status") not in ("dead", "departed")
    )
    total_chars = len(characters)
    pending_fh = sum(
        1 for f in foreshadowing.values()
        if f.get("status") == "pending"
    )

    # Build status table.
    table = Table(title=f"小说进度: {summary['title']}", show_header=False)
    table.add_column("属性", style="bold cyan")
    table.add_column("值", style="white")

    table.add_row("标题", summary["title"])
    table.add_row("流派", summary["genre"])
    table.add_row("作家风格", summary["style"])
    table.add_row(
        "目标总字数",
        f"{summary['target_words']:,}" if summary["target_words"] else "-",
    )
    table.add_row("每卷章数（配置）", str(ctx.chapters_per_volume))
    table.add_row("每章字数（配置）", f"{ctx.words_per_chapter:,}")
    table.add_row(
        "已写卷数",
        f"{writen_volumes}/{planned_volumes}" if planned_volumes else str(writen_volumes),
    )

    total_word_count = 0
    for vs in volume_summaries:
        total_word_count += vs.get("word_count", 0)
    if total_word_count > 0:
        table.add_row("已写总字数", f"{total_word_count:,}")

    table.add_row("角色总数", str(total_chars))
    table.add_row("活跃角色", str(active_chars))
    table.add_row("待回收伏笔", str(pending_fh))

    if next_volume:
        table.add_row(
            "下一卷",
            f"第{next_volume}卷",
        )

    console.print()
    console.print(table)

    # Volume detail table.
    if summary["volume_details"]:
        console.print()
        vol_table = Table(title="卷详情", show_header=True, header_style="bold")
        vol_table.add_column("卷", style="dim")
        vol_table.add_column("状态")
        vol_table.add_column("计划标题")
        vol_table.add_column("实际标题")
        vol_table.add_column("计划章数")
        vol_table.add_column("导语")

        for vd in summary["volume_details"]:
            status_text = "[green]已写[/green]" if vd["written"] else "[dim]未写[/dim]"
            vol_table.add_row(
                str(vd["volume"]),
                status_text,
                vd["planned_title"],
                vd["actual_title"] or "-",
                str(vd["estimated_chapters"]),
                Text(vd["synopsis"] or "-", style="dim"),
            )

        console.print(vol_table)


# ===================================================================
# Command: continue
# ===================================================================


def cmd_continue(ctx: AppContext, args: argparse.Namespace) -> None:
    """Continue writing the next unwritten volume."""
    novel_name = _require_novel(ctx, args.novel)
    if novel_name is None:
        return

    summary = ctx.outline_manager.get_outline_summary(novel_name)
    written = summary["volumes_written"]
    planned = summary["volumes_planned"]

    if planned and written >= planned:
        console.print("[green]所有卷已写完！[/green]")
        return

    next_volume = written + 1
    console.print(
        f"\n[bold]自动检测到下一卷: 第{next_volume}卷[/bold]"
    )

    # 1. Generate volume outline if not exists.
    vol_outline = ctx.novel_store.load_volume_outline(novel_name, next_volume)
    if not vol_outline:
        console.print(f"[cyan]正在生成第{next_volume}卷大纲...[/cyan]")
        try:
            vol_outline = ctx.outline_manager.create_volume_outline(
                novel_name=novel_name,
                volume_num=next_volume,
                chapters_per_volume=ctx.chapters_per_volume,
            )
        except Exception as exc:
            console.print(f"[red]生成大纲失败: {exc}[/red]")
            return
        _display_volume_outline(vol_outline, volume_num=next_volume)
        if not Confirm.ask("大纲是否满意，继续生成？", default=True):
            console.print("[yellow]已取消。[/yellow]")
            return

    # 2. Write the volume.
    _write_volume(ctx, novel_name, next_volume, ctx.words_per_chapter)

    # 3. Audit.
    console.print()
    if Confirm.ask("是否对刚生成的卷进行审计？", default=True):
        # Fake a namespace for cmd_audit arguments.
        audit_args = argparse.Namespace(
            novel=novel_name, volume=next_volume, fix=None
        )
        cmd_audit(ctx, audit_args)


# ===================================================================
# Command: list
# ===================================================================


def cmd_list(ctx: AppContext) -> None:
    """List all novels."""
    novels = ctx.novel_store.list_novels()
    if not novels:
        console.print("[dim]还没有任何小说。使用 'python cli.py new' 创建。[/dim]")
        return

    table = Table(title="小说列表", show_header=True, header_style="bold")
    table.add_column("名称")
    table.add_column("标题")
    table.add_column("流派")
    table.add_column("风格")
    table.add_column("卷数")
    table.add_column("总字数")

    for name in novels:
        tbl = ctx.make_table_store(name)
        meta = tbl.load_meta()
        volume_count = ctx.novel_store.get_volume_count(name)
        volume_summaries = tbl.load_volume_summaries()
        total_words = sum(vs.get("word_count", 0) for vs in volume_summaries)

        table.add_row(
            name,
            meta.get("title", name),
            meta.get("genre", "-"),
            meta.get("author_style", "-"),
            str(volume_count),
            f"{total_words:,}" if total_words else "-",
        )

    console.print()
    console.print(table)


# ===================================================================
# Style customization helper
# ===================================================================


def _customize_style(ctx: AppContext, style_name: str) -> str:
    """Interactive style customization flow for the ``new`` command.

    Shows the selected style's parameters, optionally lets the user tweak
    them, and optionally saves the result as a new named custom style.

    Args:
        ctx: Application context.
        style_name: Currently selected style name.

    Returns:
        The final style name to use (the original, or a new custom name).
    """
    # Show current parameters.
    params = ctx.style_manager.get_style_params(style_name)
    console.print()
    console.print(Panel.fit(
        f"[bold]叙事节奏:[/bold] {params['narrative_rhythm']}\n"
        f"[bold]对话占比:[/bold] {params['dialogue_ratio']}\n"
        f"[bold]描写细腻度:[/bold] {params['description_detail']}\n"
        f"[bold]战斗描写风格:[/bold] {params['battle_style']}\n"
        f"[bold]情感深度:[/bold] {params['emotional_depth']}\n"
        f"[bold]句式风格:[/bold] {params['sentence_style']}",
        title=f"[bold]风格参数: {style_name}[/bold]",
        border_style="blue",
    ))

    if not Confirm.ask("是否要自定义此风格？(y/n)", default=False):
        return style_name

    console.print()
    console.print("[bold cyan]逐项自定义参数（回车保留当前值）[/bold cyan]")

    # 叙事节奏
    rhythm_map = {"fast": "快速", "moderate": "中等", "slow": "慢速"}
    current_rhythm = params["narrative_rhythm"]
    console.print("  可选: fast(快速) / moderate(中等) / slow(慢速)")
    rhythm_input = Prompt.ask(
        f"  叙事节奏 [当前: {current_rhythm}]",
        default="",
    )
    if rhythm_input.strip():
        new_rhythm = rhythm_map.get(rhythm_input.strip(), current_rhythm)
    else:
        new_rhythm = current_rhythm

    # 对话占比
    dialogue_str = Prompt.ask(
        f"  对话占比（0.0-1.0）[当前: {params['dialogue_ratio']}]",
        default="",
    )
    new_dialogue = float(dialogue_str) if dialogue_str.strip() else params["dialogue_ratio"]

    # 描写细腻度
    detail_str = Prompt.ask(
        f"  描写细腻度（0.0-1.0）[当前: {params['description_detail']}]",
        default="",
    )
    new_detail = float(detail_str) if detail_str.strip() else params["description_detail"]

    # 战斗描写风格
    battle_input = Prompt.ask(
        f"  战斗描写风格 [当前: {params['battle_style']}]",
        default="",
    )
    new_battle = battle_input.strip() if battle_input.strip() else params["battle_style"]

    # 情感深度
    emotion_str = Prompt.ask(
        f"  情感深度（0.0-1.0）[当前: {params['emotional_depth']}]",
        default="",
    )
    new_emotion = float(emotion_str) if emotion_str.strip() else params["emotional_depth"]

    # 句式风格
    sentence_choices = {"concise": "简洁", "balanced": "平衡", "elaborate": "细腻"}
    current_sentence = params["sentence_style"]
    console.print("  可选: concise(简洁) / balanced(平衡) / elaborate(细腻)")
    sentence_input = Prompt.ask(
        f"  句式风格 [当前: {current_sentence}]",
        default="",
    )
    if sentence_input.strip():
        new_sentence = sentence_choices.get(sentence_input.strip(), current_sentence)
    else:
        new_sentence = current_sentence

    # Show customized summary.
    console.print()
    console.print(Panel.fit(
        f"[bold]叙事节奏:[/bold] {new_rhythm}\n"
        f"[bold]对话占比:[/bold] {new_dialogue}\n"
        f"[bold]描写细腻度:[/bold] {new_detail}\n"
        f"[bold]战斗描写风格:[/bold] {new_battle}\n"
        f"[bold]情感深度:[/bold] {new_emotion}\n"
        f"[bold]句式风格:[/bold] {new_sentence}",
        title="[bold]自定义后风格[/bold]",
        border_style="green",
    ))

    # Ask to save.
    if Confirm.ask("是否保存此自定义风格供以后使用？(y/n)", default=True):
        custom_name = Prompt.ask("请输入自定义风格名称")
        try:
            style_params = ctx.style_manager.create_custom_style(
                name=custom_name,
                narrative_rhythm=new_rhythm,
                dialogue_ratio=new_dialogue,
                description_detail=new_detail,
                battle_style=new_battle,
                emotional_depth=new_emotion,
                sentence_style=new_sentence,
                description=custom_name,
            )
            ctx.style_manager.save_custom_style(custom_name, style_params)
            console.print(f"[green]自定义风格 '{custom_name}' 已保存。[/green]")
            return custom_name
        except ValueError as exc:
            console.print(f"[red]保存失败: {exc}[/red]")
            console.print("[yellow]仍将使用自定义参数，但仅本次生效。[/yellow]")

    # Even if not saved, create an in-memory temporary style for this novel.
    temp_name = f"_temp_{style_name}_custom"
    try:
        ctx.style_manager.create_custom_style(
            name=temp_name,
            narrative_rhythm=new_rhythm,
            dialogue_ratio=new_dialogue,
            description_detail=new_detail,
            battle_style=new_battle,
            emotional_depth=new_emotion,
            sentence_style=new_sentence,
            description=f"临时自定义({style_name})",
        )
        return temp_name
    except ValueError:
        # Already exists from a prior temp creation; update in place.
        ctx.style_manager.delete_custom_style(temp_name)
        ctx.style_manager.create_custom_style(
            name=temp_name,
            narrative_rhythm=new_rhythm,
            dialogue_ratio=new_dialogue,
            description_detail=new_detail,
            battle_style=new_battle,
            emotional_depth=new_emotion,
            sentence_style=new_sentence,
            description=f"临时自定义({style_name})",
        )
        return temp_name


# ===================================================================
# Display helpers
# ===================================================================


def _display_total_outline(outline: dict[str, Any]) -> None:
    """Render the total outline in a rich Panel."""
    import json

    meta = outline.get("_meta", {})
    title = meta.get("title", "未知")

    content_lines: list[str] = []

    synopsis = outline.get("synopsis", "")
    if synopsis:
        content_lines.append(f"[bold]导语:[/bold] {synopsis}")
        content_lines.append("")

    world = outline.get("world_setting", "")
    if world:
        content_lines.append(f"[bold]世界观设定:[/bold]\n{world}")
        content_lines.append("")

    plot = outline.get("main_plot_arc", "")
    if plot:
        content_lines.append(f"[bold]主线剧情:[/bold]\n{plot}")
        content_lines.append("")

    characters = outline.get("main_characters", [])
    if characters:
        content_lines.append("[bold]主要角色:[/bold]")
        for ch in characters:
            name = ch.get("name", "?")
            role = ch.get("role", "")
            brief = ch.get("brief", "")
            content_lines.append(f"  - [cyan]{name}[/cyan] ({role}): {brief}")
        content_lines.append("")

    volume_plan = outline.get("volume_plan", [])
    if volume_plan:
        content_lines.append("[bold]分卷规划:[/bold]")
        for vp in volume_plan:
            vnum = vp.get("volume", "?")
            vtitle = vp.get("title", "")
            vsummary = vp.get("summary", "")
            vch = vp.get("estimated_chapters", "?")
            content_lines.append(
                f"  [bold]第{vnum}卷「{vtitle}」[/bold] (~{vch}章): {vsummary}"
            )

    panel = Panel.fit(
        "\n".join(content_lines),
        title=f"[bold cyan]总大纲: {title}[/bold cyan]",
        border_style="cyan",
    )
    console.print()
    console.print(panel)


def _display_volume_outline(outline: dict[str, Any], volume_num: int) -> None:
    """Render a volume outline in a rich Panel."""
    title = outline.get("volume_title", f"第{volume_num}卷")
    arc = outline.get("volume_arc", "")
    synopsis = outline.get("synopsis", "")
    chapter_plan = outline.get("chapter_plan", [])

    content_lines: list[str] = []

    if synopsis:
        content_lines.append(f"[bold]导语:[/bold] {synopsis}")
        content_lines.append("")

    if arc:
        content_lines.append(f"[bold]卷故事弧线:[/bold]\n{arc}")
        content_lines.append("")

    if chapter_plan:
        content_lines.append(f"[bold]章节计划 ({len(chapter_plan)}章):[/bold]")
        for ch in chapter_plan:
            ch_num = ch.get("chapter_num", "?")
            title_hint = ch.get("title_hint", "")
            key_events = ch.get("key_events", [])
            char_focus = ch.get("character_focus", "")

            content_lines.append(f"  [cyan]第{ch_num}章[/cyan]: {title_hint}")
            for evt in key_events:
                content_lines.append(f"    - {evt}")
            if char_focus:
                content_lines.append(f"    [dim]角色焦点: {char_focus}[/dim]")

    panel = Panel.fit(
        "\n".join(content_lines),
        title=f"[bold cyan]第{volume_num}卷大纲: {title}[/bold cyan]",
        border_style="cyan",
    )
    console.print()
    console.print(panel)


def _display_outline_summary(summary: dict[str, Any]) -> None:
    """Render the outline summary from OutlineManager.get_outline_summary."""
    console.print()
    info_panel = Panel.fit(
        f"标题: {summary['title']}\n"
        f"流派: {summary['genre']}\n"
        f"风格: {summary['style']}\n"
        f"目标字数: {summary['target_words']:,}\n"
        f"已写卷数: {summary['volumes_written']}/{summary['volumes_planned']}\n"
        f"全书导语: {summary.get('synopsis', '-')}",
        title="小说信息",
        border_style="blue",
    )
    console.print(info_panel)

    details = summary.get("volume_details", [])
    if details:
        vol_table = Table(title="卷大纲概览", show_header=True, header_style="bold")
        vol_table.add_column("卷", style="dim")
        vol_table.add_column("状态")
        vol_table.add_column("计划标题")
        vol_table.add_column("计划简介")
        vol_table.add_column("计划/实际章数")

        for vd in details:
            status_text = "[green]已写[/green]" if vd["written"] else "[dim]未写[/dim]"
            planned_ch = str(vd["estimated_chapters"])
            actual_ch = str(vd["actual_chapters"]) if vd["actual_chapters"] else "-"
            vol_table.add_row(
                str(vd["volume"]),
                status_text,
                vd["planned_title"],
                Text(vd["planned_summary"] or "-", style="dim"),
                f"{planned_ch} / {actual_ch}",
            )

        console.print()
        console.print(vol_table)


def _display_audit_report(report: Any) -> None:
    """Render an audit report with color-coded severity."""
    console.print()

    # Overall score with color.
    score = report.overall_score
    if score >= 8:
        score_color = "green"
    elif score >= 5:
        score_color = "yellow"
    else:
        score_color = "red"

    score_text = Text()
    score_text.append("总体评分: ", style="bold")
    score_text.append(f"{score:.1f}/10", style=f"bold {score_color}")

    header_lines: list[str] = [
        str(score_text),
        f"逻辑问题: {len(report.logic_issues)} | AI味问题: {len(report.ai_flavor_issues)}",
    ]

    if report.summary:
        header_lines.append(f"\n[dim]{report.summary}[/dim]")

    console.print(Panel.fit(
        "\n".join(header_lines),
        title=f"[bold]审计报告 - 第{report.volume_num}卷[/bold]",
        border_style="blue",
    ))

    # Logic issues.
    if report.logic_issues:
        console.print()
        console.print("[bold]逻辑一致性问题:[/bold]")
        for issue in report.logic_issues:
            _print_issue(issue)

    # AI-flavor issues.
    if report.ai_flavor_issues:
        console.print()
        console.print("[bold]AI写作痕迹:[/bold]")
        for issue in report.ai_flavor_issues:
            _print_issue(issue)

    if not report.all_issues:
        console.print("[green]未发现问题，恭喜！[/green]")


def _print_issue(issue: Any) -> None:
    """Print a single audit issue with color-coded severity."""
    sev_color = {
        "critical": "red",
        "major": "yellow",
        "minor": "dim",
    }.get(issue.severity, "white")

    issue_text = (
        f"[{sev_color}][{issue.severity.upper()}][/{sev_color}] "
        f"[bold]{issue.location}[/bold]: {issue.description}"
    )
    console.print(f"  {issue_text}")
    if issue.suggestion:
        console.print(f"    [dim]建议: {issue.suggestion}[/dim]")


# ===================================================================
# Shared helpers
# ===================================================================


def _require_novel(ctx: AppContext, name: str | None) -> str | None:
    """Validate that *name* is a known novel, prompting if missing.

    Returns the novel name on success, ``None`` on failure.
    """
    if name and ctx.novel_store.novel_exists(name):
        return name
    if name:
        console.print(f"[red]小说'{name}'不存在。[/red]")
        return None

    novels = ctx.novel_store.list_novels()
    if not novels:
        console.print("[red]还没有任何小说。使用 'python cli.py new' 创建。[/red]")
        return None

    console.print("[bold]可用小说:[/bold]")
    for i, n in enumerate(novels, 1):
        console.print(f"  {i}. {n}")
    return None


def _write_volume(
    ctx: AppContext,
    novel_name: str,
    volume_num: int,
    words_per_chapter: int = 3000,
) -> None:
    """Common volume writing logic shared by ``new``, ``write``, and ``continue``.

    Args:
        ctx: Application context.
        novel_name: Novel identifier.
        volume_num: Volume number (1-indexed).
        words_per_chapter: Target characters per chapter.
    """
    # Ensure volume outline exists.
    vol_outline = ctx.novel_store.load_volume_outline(novel_name, volume_num)
    if not vol_outline:
        console.print(
            f"[yellow]第{volume_num}卷大纲不存在，正在生成...[/yellow]"
        )
        try:
            vol_outline = ctx.outline_manager.create_volume_outline(
                novel_name=novel_name,
                volume_num=volume_num,
            )
        except Exception as exc:
            console.print(f"[red]生成第{volume_num}卷大纲失败: {exc}[/red]")
            return
        _display_volume_outline(vol_outline, volume_num=volume_num)

    volume_writer = ctx.make_volume_writer(novel_name)

    console.print()
    try:
        result = volume_writer.write_volume(
            novel_name=novel_name,
            volume_num=volume_num,
            target_words_per_chapter=words_per_chapter,
        )
    except Exception as exc:
        console.print(f"[red]生成第{volume_num}卷失败: {exc}[/red]")
        return

    # Display result.
    console.print()
    result_panel = Panel.fit(
        f"卷标题: {result['title']}\n"
        f"章数: {result['chapter_count']}\n"
        f"总字数: ~{result['total_words']:,}",
        title=f"[bold green]第{volume_num}卷完成[/bold green]",
        border_style="green",
    )
    console.print(result_panel)

    # Chapter listing.
    ch_table = Table(title="章节列表", show_header=True, header_style="bold")
    ch_table.add_column("章节", style="dim")
    ch_table.add_column("标题")
    ch_table.add_column("字数", justify="right")
    for ch in result.get("chapters", []):
        ch_table.add_row(
            str(ch["chapter_num"]),
            ch["title"],
            f"{ch['word_count']:,}",
        )
    console.print(ch_table)


# ===================================================================
# Argument parser
# ===================================================================


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser with all subcommands."""

    # Shared examples and notes.
    _common_network_note = (
        "💡 网络故障恢复提示\n"
        "  如果因网络问题导致 LLM 调用失败，直接重新运行相同命令即可。\n"
        "  已生成的内容（大纲、章节）会保存在 data/ 目录下，不会丢失。\n"
        "  可用 'python cli.py status --novel <名称>' 查看当前进度。"
    )

    parser = argparse.ArgumentParser(
        prog="python cli.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "网络小说生成器 —— 按卷生成超长篇小说，支持记忆追踪与质量审计。\n\n"
            "典型工作流程:\n"
            "  1. python cli.py new                 创建新小说，生成总大纲及第一卷大纲\n"
            "  2. python cli.py write --novel X --volume 1   生成第一卷正文\n"
            "  3. python cli.py audit --novel X --volume 1   审计第一卷质量\n"
            "  4. python cli.py continue --novel X           继续生成下一卷\n\n"
            "如果中途因网络问题中断，直接用相同命令重试即可，已生成内容不会丢失。\n"
            "随时用 'python cli.py status --novel <名称>' 查看进度。"
        ),
        epilog=(
            "更多信息: https://github.com/b1indy/novel-generator\n"
            "配置文件: config.yaml（首次使用请复制 config.yaml.example 并填入 API key)"
        ),
    )

    sub = parser.add_subparsers(
        dest="command",
        title="可用命令",
        description="使用 'python cli.py <命令> --help' 查看各命令的详细用法。",
    )

    # --- new ---
    new_parser = sub.add_parser(
        "new",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help="创建新小说（交互式引导）",
        description=(
            "交互式创建一部新小说，引导你完成:\n"
            "  1. 设定小说名称与标题\n"
            "  2. 选择流派（玄幻/都市/科幻/仙侠/历史/悬疑）\n"
            "  3. 选择作家风格（辰东式/猫腻式/番茄式/老鹰式/唐家三少式）\n"
            "  4. 可选自定义风格参数并保存\n"
            "  5. 设定目标总字数\n"
            "  6. 生成总大纲（含世界观、主线、分卷规划、100字导语）\n"
            "  7. 生成第一卷大纲（含章节计划、导语）\n"
            "  8. 可选直接开始生成第一卷正文"
        ),
        epilog=(
            "示例:\n"
            "  python cli.py new\n\n"
            "网络故障恢复:\n"
            "  如果大纲生成步骤失败（如网络超时），小说目录已创建，此时可:\n"
            "  - python cli.py outline --novel <名称>              查看大纲状态\n"
            "  - python cli.py outline --novel <名称> --volume 1   重新生成第一卷大纲\n"
            "  - python cli.py status --novel <名称>               查看完整进度"
        ),
    )

    # --- outline ---
    outline_parser = sub.add_parser(
        "outline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help="查看或生成小说大纲",
        description=(
            "管理小说大纲（总大纲和卷大纲）。根据参数不同有三种模式:\n\n"
            "【查看模式】仅指定小说名称，展示总大纲和所有卷大纲概览。\n\n"
            "【生成卷大纲】指定 --volume，为某一卷生成大纲。生成时会自动注入\n"
            "前续卷的剧情小结和末两章，保证跨卷逻辑连贯。\n\n"
            "【修改总大纲】指定 --update，交互式修改总大纲。写作过程中如果\n"
            "剧情走向与原大纲偏离，可以用此命令灵活调整。"
        ),
        epilog=(
            "示例:\n"
            "  python cli.py outline --novel my-novel              查看所有大纲\n"
            "  python cli.py outline --novel my-novel --volume 3   生成第3卷大纲\n"
            "  python cli.py outline --novel my-novel --update     修改总大纲\n\n"
            f"{_common_network_note}"
        ),
    )
    outline_parser.add_argument(
        "--novel", type=str,
        help="小说名称（即 new 时输入的英文/拼音标识）",
    )
    outline_parser.add_argument(
        "--volume", type=int, metavar="N",
        help="为第 N 卷生成大纲（如 --volume 3）",
    )
    outline_parser.add_argument(
        "--update", action="store_true",
        help="交互式修改总大纲",
    )

    # --- write ---
    write_parser = sub.add_parser(
        "write",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help="生成指定卷的小说正文",
        description=(
            "按卷大纲逐章生成小说正文。每章包含标题，生成后自动更新\n"
            "角色表、物品表、伏笔表。整卷完成后自动生成本卷小结。\n\n"
            "需要先有卷大纲。如果目标卷的大纲不存在，会自动先生成大纲。"
        ),
        epilog=(
            "示例:\n"
            "  python cli.py write --novel my-novel --volume 1               默认每章5000字\n"
            "  python cli.py write --novel my-novel --volume 2 --words 6000  每章6000字\n\n"
            f"{_common_network_note}"
        ),
    )
    write_parser.add_argument(
        "--novel", type=str, required=True,
        help="小说名称（必填）",
    )
    write_parser.add_argument(
        "--volume", type=int, required=True, metavar="N",
        help="要生成的卷号，从 1 开始（必填）",
    )
    write_parser.add_argument(
        "--words", type=int, default=None, metavar="N",
        help="每章目标字数，覆盖 config.yaml 中的 words_per_chapter 设置",
    )

    # --- audit ---
    audit_parser = sub.add_parser(
        "audit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help="审计小说质量（逻辑+AI味）",
        description=(
            "对指定卷进行双维度质量审计，单次 LLM 调用同时检查:\n"
            "  1. 逻辑一致性 — 角色行为、时间线、物品归属、伏笔衔接、世界观\n"
            "  2. AI写作痕迹 — 重复句式、模板化描写、生硬对话、情感平淡\n\n"
            "审计结果按严重程度（critical/major/minor）分类展示。\n"
            "使用 --fix 可自动修复发现的问题。"
        ),
        epilog=(
            "示例:\n"
            "  python cli.py audit --novel my-novel --volume 1         仅审计，输出报告\n"
            "  python cli.py audit --novel my-novel --volume 1 --fix   审计并自动修复\n\n"
            "💡 建议每写完一卷就审计一次，及时修复问题比事后补救更省力。"
        ),
    )
    audit_parser.add_argument(
        "--novel", type=str, required=True,
        help="小说名称（必填）",
    )
    audit_parser.add_argument(
        "--volume", type=int, required=True, metavar="N",
        help="要审计的卷号（必填）",
    )
    audit_parser.add_argument(
        "--fix", action="store_true",
        help="审计后自动修复发现的问题",
    )

    # --- status ---
    status_parser = sub.add_parser(
        "status",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help="查看小说进度与状态",
        description=(
            "显示小说的完整状态，包括:\n"
            "  - 基本信息（标题、流派、风格、目标字数）\n"
            "  - 创作进度（已写卷数/计划卷数、总字数）\n"
            "  - 角色统计（总数、活跃数）\n"
            "  - 伏笔统计（待回收数量）\n"
            "  - 各卷详情（状态、标题、章数、导语）\n"
            "  - 下一卷号\n\n"
            "如果不指定 --novel，会列出可用小说名称。"
        ),
        epilog=(
            "示例:\n"
            "  python cli.py status --novel my-novel   查看指定小说进度\n"
            "  python cli.py status                    列出所有小说后选择"
        ),
    )
    status_parser.add_argument(
        "--novel", type=str,
        help="小说名称（不指定则列出所有小说）",
    )

    # --- continue ---
    continue_parser = sub.add_parser(
        "continue",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help="自动继续生成下一卷",
        description=(
            "一键式继续创作：自动检测下一未写卷 → 生成卷大纲（如未生成）\n"
            "→ 逐章写作 → 可选审计。适合日常连载式操作，无需手动指定卷号。\n\n"
            "如果所有计划卷已写完，会提示完成。\n"
            "如果想在计划外追加卷，请使用 outline --volume N 生成新卷大纲。"
        ),
        epilog=(
            "示例:\n"
            "  python cli.py continue --novel my-novel\n\n"
            f"{_common_network_note}"
        ),
    )
    continue_parser.add_argument(
        "--novel", type=str, required=True,
        help="小说名称（必填）",
    )

    # --- list ---
    list_parser = sub.add_parser(
        "list",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help="列出所有小说",
        description=(
            "列出 data/ 目录下所有已创建的小说，显示:\n"
            "  名称、标题、流派、风格、已写卷数、总字数"
        ),
        epilog="示例:\n  python cli.py list",
    )

    return parser


# ===================================================================
# Main entry point
# ===================================================================


def main() -> None:
    """Parse arguments, initialise the app context, and dispatch commands."""
    parser = build_parser()

    if len(sys.argv) == 1:
        parser.print_help()
        return

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    # -- Load config -----------------------------------------------------------
    config_path = load_config_path()
    is_fallback = config_path.name == "config.yaml.example"

    if is_fallback:
        console.print(
            Panel.fit(
                "[yellow]未找到 config.yaml，使用 config.yaml.example 作为配置模板。\n"
                "请复制 config.yaml.example 为 config.yaml 并填入实际的 API 密钥。[/yellow]",
                title="[yellow]警告[/yellow]",
                border_style="yellow",
            )
        )

    # -- Initialize app context ------------------------------------------------
    try:
        ctx = AppContext(str(config_path))
    except LLMError as exc:
        console.print(f"[red]初始化 LLM 客户端失败: {exc}[/red]")
        console.print(
            "[dim]请确保:\n"
            "  1. config.yaml 存在且包含有效的 llm.api_key\n"
            "  2. 或设置环境变量 DEEPSEEK_API_KEY[/dim]"
        )
        sys.exit(1)
    except Exception as exc:
        console.print(f"[red]初始化失败: {exc}[/red]")
        sys.exit(1)

    # -- Dispatch --------------------------------------------------------------
    dispatch: dict[str, Any] = {
        "new": lambda: cmd_new(ctx),
        "list": lambda: cmd_list(ctx),
        "outline": lambda: cmd_outline(ctx, args),
        "write": lambda: cmd_write(ctx, args),
        "audit": lambda: cmd_audit(ctx, args),
        "status": lambda: cmd_status(ctx, args),
        "continue": lambda: cmd_continue(ctx, args),
    }

    handler = dispatch.get(args.command)
    if handler is None:
        console.print(f"[red]未知命令: {args.command}[/red]")
        parser.print_help()
        sys.exit(1)

    try:
        handler()
    except KeyboardInterrupt:
        console.print()
        console.print("[yellow]操作已被用户中断。[/yellow]")
        sys.exit(0)
    except LLMError as exc:
        console.print(f"[red]LLM 错误: {exc}[/red]")
        sys.exit(1)
    except Exception as exc:
        console.print(f"[red]发生错误: {exc}[/red]")
        logger.exception("Unhandled exception in CLI")
        sys.exit(1)


if __name__ == "__main__":
    main()
