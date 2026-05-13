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
#
#   Chinese characters occupy 3 bytes in UTF-8 and are displayed 2 columns
#   wide.  Without proper locale and readline configuration the line-editor
#   may treat each byte or each column as a separate deletion unit,
#   requiring multiple backspace presses to erase a single glyph.
#
#   This MUST run before any module that might import readline (including
#   rich).  We use gnureadline (a more up-to-date wrapper around GNU
#   readline 8.x) which has better CJK multi-byte handling than the
#   stdlib readline module bundled with CPython.
# ---------------------------------------------------------------------------


def _setup_encoding() -> None:
    """Configure locale and readline for correct CJK multi-byte input."""

    # 1. Force CJK-aware UTF-8 locale in environment variables.
    #    GNU readline checks these at init time for character-width tables;
    #    en_US.UTF-8 has correct *encoding* but lacks CJK width data.
    for var in ("LANG", "LC_ALL", "LC_CTYPE"):
        os.environ[var] = "zh_CN.UTF-8"

    # 2. Activate locale in Python (must happen before readline import).
    for loc in ("zh_CN.UTF-8", "en_US.UTF-8", "C.UTF-8"):
        try:
            locale.setlocale(locale.LC_ALL, loc)
            break
        except locale.Error:
            continue

    # 3. Replace stdlib readline with gnureadline (better CJK support).
    #    Must complete before rich or any input-handling import.
    try:
        import gnureadline  # noqa: F401
        sys.modules["readline"] = gnureadline
    except ImportError:
        pass  # gnureadline not installed; stdlib readline may still work


_setup_encoding()
del _setup_encoding

# ---------------------------------------------------------------------------
# Readline-based input helpers (replaces rich.prompt for CJK safety)
#
#   Rich's Prompt.ask() bypasses readline and manages its own cursor via
#   ANSI escape codes.  Its column arithmetic breaks on CJK characters
#   (2 columns wide, 3 UTF-8 bytes), causing backspace to erase part of
#   previously printed output.  These wrappers use readline (or
#   gnureadline) for actual input, while rich handles display-only output.
# ---------------------------------------------------------------------------


def _ask(prompt: str, default: str | None = None, choices: list[str] | None = None) -> str:
    """Read a line of input via readline, with optional default and choices."""
    import readline  # noqa: F811 — may be gnureadline after _setup_encoding

    suffix = ""
    if default is not None:
        suffix = f" [{default}]"
    if choices:
        suffix = f" ({'/'.join(choices)}){suffix}"

    while True:
        try:
            value = input(f"{prompt}{suffix}: ").strip()
        except EOFError:
            return default or ""
        if not value and default is not None:
            return default
        if choices and value not in choices:
            console.print(f"[red]无效输入，请选择: {', '.join(choices)}[/red]")
            continue
        if value:
            return value


def _confirm(prompt: str, default: bool = True) -> bool:
    """Ask a yes/no question via readline."""
    hint = "Y/n" if default else "y/N"
    try:
        value = input(f"{prompt} [{hint}]: ").strip().lower()
    except EOFError:
        return default
    if not value:
        return default
    return value in ("y", "yes", "是")


def _int_ask(prompt: str, default: int | None = None) -> int:
    """Ask for an integer via readline."""
    suffix = f" [{default}]" if default is not None else ""
    while True:
        try:
            value = input(f"{prompt}{suffix}: ").strip()
        except EOFError:
            if default is not None:
                return default
            continue
        if not value and default is not None:
            return default
        try:
            return int(value)
        except ValueError:
            console.print("[red]请输入有效的数字。[/red]")


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
# Prompt/Confirm/IntPrompt removed — using readline-based _ask/_confirm/_int_ask
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from src.llm.client import LLMClient, LLMError
from src.llm.token_tracker import TokenTracker
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
    defaults = {
        "chapters_per_volume": 50,
        "words_per_chapter": 5000,
        "batch_size": 3,
        "generation_mode": "continuous",
        "round_size": 15,
    }
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
        "batch_size": int(
            generation.get("batch_size", defaults["batch_size"])
        ),
        "generation_mode": str(
            generation.get("generation_mode", defaults["generation_mode"])
        ),
        "round_size": int(
            generation.get("round_size", defaults["round_size"])
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
    batch_size: int
    generation_mode: str
    round_size: int

    def __init__(self, config_path: str) -> None:
        self.token_tracker = TokenTracker()
        self.llm_client = LLMClient(config_path, token_tracker=self.token_tracker)
        self.style_manager = StyleManager()
        self.novel_store = NovelStore("data/novels")

        # Load generation params from config.
        gen_cfg = load_generation_config(Path(config_path))
        self.chapters_per_volume = gen_cfg["chapters_per_volume"]
        self.words_per_chapter = gen_cfg["words_per_chapter"]
        self.batch_size = gen_cfg["batch_size"]
        self.generation_mode = gen_cfg["generation_mode"]
        self.round_size = gen_cfg["round_size"]

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
            round_size=self.round_size,
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
    novel_name = _ask("请输入小说名称（用于目录标识，英文/拼音）")
    if ctx.novel_store.novel_exists(novel_name):
        console.print(
            f"[red]小说'{novel_name}'已存在。请使用其他名称。[/red]"
        )
        return

    title = _ask("请输入小说标题（中文显示名称）", default=novel_name)

    # 2. Select genre.
    genres = ctx.style_manager.list_genres()
    console.print()
    console.print("[bold]可用流派：[/bold]")
    for i, g in enumerate(genres, 1):
        console.print(f"  {i}. {g}")
    genre_idx = _int_ask("请选择流派（输入编号）", default=1)
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
    style_idx = _int_ask("请选择作家风格（输入编号）", default=1)
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
    target_words = _int_ask("请输入目标总字数", default=2_000_000)

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

    if not _confirm("确认创建？", default=True):
        console.print("[yellow]已取消。[/yellow]")
        return

    # 6. Create novel and generate total outline.
    ctx.token_tracker.set_category("outline")
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

        choice = _ask("大纲是否满意？", choices=["y", "n", "q"], default="y")
        if choice == "y":
            break
        if choice == "q":
            console.print("[yellow]已取消。[/yellow]")
            return

        # Changes requested.
        feedback = _ask("请描述大纲的问题或需要修改的方向")
        change_request = _ask("具体修改要求", default="请根据上述反馈修改大纲")
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
    if _confirm("是否开始生成第一卷？", default=True):
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
        feedback = _ask("请描述修改原因或上下文")
        change_request = _ask("具体修改要求", default="请根据上述反馈修改大纲")
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
    auto_audit: bool = getattr(args, "audit", False)
    _write_volume(ctx, novel_name, volume_num, words, auto_audit=auto_audit)


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
    ctx.token_tracker.set_category("auditing")
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

        if not _confirm(
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
        if not _confirm("大纲是否满意，继续生成？", default=True):
            console.print("[yellow]已取消。[/yellow]")
            return

    # 2. Write the volume.
    _write_volume(ctx, novel_name, next_volume, ctx.words_per_chapter)

    # 3. Audit.
    console.print()
    if _confirm("是否对刚生成的卷进行审计？", default=True):
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
# Command: token
# ===================================================================


def cmd_token(ctx: AppContext) -> None:
    """Display token usage statistics for the current session."""
    console.print()
    console.print(ctx.token_tracker.report())


# ===================================================================
# Style customization helper
# ===================================================================


def _customize_style(ctx: AppContext, style_name: str) -> str:
    """LLM-driven style customization via conversation.

    Instead of manually tweaking parameters, the user describes their novel
    concept and the LLM generates a comprehensive style document. The LLM
    asks clarifying questions if needed.

    Args:
        ctx: Application context.
        style_name: Currently selected base style name.

    Returns:
        The final style name to use (the original, or a new custom name).
    """
    # Show current style info.
    console.print()
    try:
        params = ctx.style_manager.get_style_params(style_name)
        console.print(Panel.fit(
            f"[bold]叙事节奏:[/bold] {params['narrative_rhythm']}\n"
            f"[bold]对话占比:[/bold] {params['dialogue_ratio']}\n"
            f"[bold]描写细腻度:[/bold] {params['description_detail']}\n"
            f"[bold]战斗描写风格:[/bold] {params['battle_style']}\n"
            f"[bold]情感深度:[/bold] {params['emotional_depth']}\n"
            f"[bold]句式风格:[/bold] {params['sentence_style']}",
            title=f"[bold]当前基础风格: {style_name}[/bold]",
            border_style="blue",
        ))
    except KeyError:
        console.print(f"[dim]当前风格: {style_name}[/dim]")

    if not _confirm("是否要通过对话自定义风格？(y/n)", default=False):
        return style_name

    # Collect user's novel concept.
    console.print()
    console.print(Panel.fit(
        "[bold cyan]请描述你的小说构想[/bold cyan]\n\n"
        "可以包含以下内容（想到什么说什么，不限格式）：\n"
        "- 故事背景、世界观\n"
        "- 主角设定、性格特点\n"
        "- 核心冲突、剧情走向\n"
        "- 希望的叙事风格、氛围\n"
        "- 参考作品（如果有的话）\n"
        "- 其他任何你觉得重要的设定\n\n"
        "[dim]输入完毕后单独输入一行 'END' 结束[/dim]",
        border_style="cyan",
    ))

    # Collect multi-line input.
    lines: list[str] = []
    while True:
        try:
            line = _ask("", default="")
        except (EOFError, KeyboardInterrupt):
            break
        if line.strip().upper() == "END":
            break
        lines.append(line)

    user_concept = "\n".join(lines).strip()
    if not user_concept:
        console.print("[yellow]未输入内容，使用原风格。[/yellow]")
        return style_name

    # LLM conversation loop.
    console.print()
    console.print("[cyan]正在分析你的构想...[/cyan]")

    conversation: list[dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "你是一位资深的网络小说写作顾问。用户会给你一段小说构想，你需要：\n\n"
                "1. 分析构想中的关键要素（世界观、角色、冲突、氛围等）\n"
                "2. 如果信息不够明确，提出2-3个关键问题帮助澄清\n"
                "3. 如果信息足够，直接生成风格文档\n\n"
                "风格文档必须包含以下维度（用中文输出JSON格式）：\n"
                "{\n"
                '  "style_name": "自定义风格名称（根据构想起一个贴切的名字）",\n'
                '  "description": "风格简介（100字以内）",\n'
                '  "narrative_rhythm": "fast/moderate/slow",\n'
                '  "dialogue_ratio": 0.0-1.0,\n'
                '  "description_detail": 0.0-1.0,\n'
                '  "battle_style": "战斗描写风格描述",\n'
                '  "emotional_depth": 0.0-1.0,\n'
                '  "sentence_style": "concise/balanced/elaborate",\n'
                '  "tone": "整体基调（如：热血、阴郁、轻松、沉重等）",\n'
                '  "pacing": "节奏特点描述",\n'
                '  "world_building_depth": "世界观构建深度（如：宏大细腻、点到为止等）",\n'
                '  "character_depth": "角色刻画深度（如：心理描写丰富、以行动展示性格等）",\n'
                '  "genre_hints": "针对该类型小说的特别写作建议"\n'
                "}\n\n"
                "如果需要提问，用JSON格式返回：\n"
                '{"action": "ask", "questions": ["问题1", "问题2"]}\n\n'
                "如果信息足够生成风格文档，用JSON格式返回：\n"
                '{"action": "generate", "style": {上述风格文档}}'
            ),
        },
        {
            "role": "user",
            "content": f"我的小说构想：\n\n{user_concept}\n\n基础风格参考：{style_name}",
        },
    ]

    ctx.token_tracker.set_category("other")

    while True:
        try:
            response = ctx.llm_client.chat(
                conversation,
                temperature=0.7,
                max_tokens=4096,
            )
        except LLMError as exc:
            console.print(f"[red]LLM 调用失败: {exc}[/red]")
            console.print("[yellow]使用原风格继续。[/yellow]")
            return style_name

        # Parse response.
        try:
            result = _parse_json_response(response)
        except (ValueError, KeyError):
            # If JSON parsing fails, treat as a conversational response.
            console.print()
            console.print(Panel.fit(response, border_style="cyan"))
            user_reply = _ask("请回复（或输入 'ok' 接受建议）", default="ok")
            if user_reply.strip().lower() == "ok":
                # Try to extract style from the response text.
                console.print("[yellow]无法解析风格文档，使用原风格。[/yellow]")
                return style_name
            conversation.append({"role": "assistant", "content": response})
            conversation.append({"role": "user", "content": user_reply})
            continue

        action = result.get("action", "")

        if action == "ask":
            # LLM wants to ask clarifying questions.
            questions = result.get("questions", [])
            console.print()
            console.print(Panel.fit(
                "[bold]我需要了解更多：[/bold]",
                border_style="yellow",
            ))
            for i, q in enumerate(questions, 1):
                console.print(f"  {i}. {q}")

            console.print()
            answers: list[str] = []
            for i, q in enumerate(questions, 1):
                answer = _ask(f"  问题{i}", default="")
                answers.append(f"问：{q}\n答：{answer}")

            conversation.append({"role": "assistant", "content": response})
            conversation.append({
                "role": "user",
                "content": "\n\n".join(answers),
            })
            console.print("[cyan]继续分析...[/cyan]")
            continue

        if action == "generate":
            # LLM generated the style document.
            style_doc = result.get("style", {})
            custom_name = style_doc.get("style_name", f"{style_name}_custom")
            description = style_doc.get("description", "")

            # Display the generated style.
            console.print()
            console.print(Panel.fit(
                f"[bold]风格名称:[/bold] {custom_name}\n"
                f"[bold]简介:[/bold] {description}\n\n"
                f"[bold]叙事节奏:[/bold] {style_doc.get('narrative_rhythm', 'moderate')}\n"
                f"[bold]对话占比:[/bold] {style_doc.get('dialogue_ratio', 0.3)}\n"
                f"[bold]描写细腻度:[/bold] {style_doc.get('description_detail', 0.7)}\n"
                f"[bold]战斗描写风格:[/bold] {style_doc.get('battle_style', '')}\n"
                f"[bold]情感深度:[/bold] {style_doc.get('emotional_depth', 0.6)}\n"
                f"[bold]句式风格:[/bold] {style_doc.get('sentence_style', 'balanced')}\n"
                f"[bold]整体基调:[/bold] {style_doc.get('tone', '')}\n"
                f"[bold]节奏特点:[/bold] {style_doc.get('pacing', '')}\n"
                f"[bold]世界观深度:[/bold] {style_doc.get('world_building_depth', '')}\n"
                f"[bold]角色刻画:[/bold] {style_doc.get('character_depth', '')}\n"
                f"[bold]类型建议:[/bold] {style_doc.get('genre_hints', '')}",
                title="[bold green]生成的风格文档[/bold green]",
                border_style="green",
            ))

            # Ask for confirmation.
            console.print()
            choice = _ask(
                "是否接受此风格？",
                choices=["y", "n", "q"],
                default="y",
            )

            if choice == "q":
                console.print("[yellow]使用原风格。[/yellow]")
                return style_name

            if choice == "n":
                feedback = _ask("请描述需要调整的地方", default="")
                if feedback.strip():
                    conversation.append({"role": "assistant", "content": response})
                    conversation.append({
                        "role": "user",
                        "content": f"请调整风格：{feedback}",
                    })
                    console.print("[cyan]正在调整...[/cyan]")
                    continue
                return style_name

            # Accept and save.
            try:
                style_params = ctx.style_manager.create_custom_style(
                    name=custom_name,
                    narrative_rhythm=style_doc.get("narrative_rhythm", "moderate"),
                    dialogue_ratio=float(style_doc.get("dialogue_ratio", 0.3)),
                    description_detail=float(style_doc.get("description_detail", 0.7)),
                    battle_style=style_doc.get("battle_style", ""),
                    emotional_depth=float(style_doc.get("emotional_depth", 0.6)),
                    sentence_style=style_doc.get("sentence_style", "balanced"),
                    description=description,
                )
                # Store extra fields in the style params for later use.
                style_params["tone"] = style_doc.get("tone", "")
                style_params["pacing"] = style_doc.get("pacing", "")
                style_params["world_building_depth"] = style_doc.get("world_building_depth", "")
                style_params["character_depth"] = style_doc.get("character_depth", "")
                style_params["genre_hints"] = style_doc.get("genre_hints", "")

                ctx.style_manager.save_custom_style(custom_name, style_params)
                console.print(f"[green]风格 '{custom_name}' 已保存。[/green]")
                return custom_name
            except ValueError as exc:
                console.print(f"[red]保存失败: {exc}[/red]")
                console.print("[yellow]使用原风格继续。[/yellow]")
                return style_name

        # Unknown action.
        console.print(f"[yellow]未知响应，使用原风格。[/yellow]")
        return style_name


def _parse_json_response(text: str) -> dict[str, Any]:
    """Extract JSON from LLM response text."""
    import json

    text = text.strip()

    # Try direct parse.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from code fences.
    import re
    fence_pattern = re.compile(r"```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL)
    matches = fence_pattern.findall(text)
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue

    # Try finding outermost { ... }.
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        try:
            return json.loads(text[brace_start : brace_end + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError("Could not extract JSON from response")


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
    auto_audit: bool = False,
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
        ctx.token_tracker.set_category("outline")
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

    ctx.token_tracker.set_category("generation")
    console.print()
    try:
        result = volume_writer.write_volume(
            novel_name=novel_name,
            volume_num=volume_num,
            words_per_chapter=words_per_chapter,
            batch_size=ctx.batch_size,
            generation_mode=ctx.generation_mode,
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

    # Auto-audit if requested.
    if auto_audit:
        console.print()
        console.print("[cyan]自动审计中...[/cyan]")
        audit_args = argparse.Namespace(
            novel=novel_name, volume=volume_num, fix=None
        )
        cmd_audit(ctx, audit_args)


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
    write_parser.add_argument(
        "--audit", action="store_true", default=False,
        help="生成完成后自动进行审计",
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

    # --- token ---
    token_parser = sub.add_parser(
        "token",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help="查看本次会话的Token使用统计",
        description=(
            "显示本次CLI会话中所有LLM调用的Token使用量，\n"
            "按类别分组（生成、大纲、审计、记忆、摘要）。"
        ),
        epilog="示例:\n  python cli.py token",
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
        "token": lambda: cmd_token(ctx),
    }

    handler = dispatch.get(args.command)
    if handler is None:
        console.print(f"[red]未知命令: {args.command}[/red]")
        parser.print_help()
        sys.exit(1)

    try:
        handler()
        # Show token usage after operations that use LLM.
        if args.command in ("new", "write", "audit", "continue"):
            console.print()
            console.print(ctx.token_tracker.report())
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
