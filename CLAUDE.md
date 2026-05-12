# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## How to run

```bash
# Install dependencies
pip install -r requirements.txt

# Set up config (uses DeepSeek API by default; any OpenAI-compatible API works)
cp config.yaml.example config.yaml
# Edit config.yaml → fill in api_key

# Syntax check all source files (there are no formal tests yet)
python3 -c "import ast; [ast.parse(open(f'<path>').read()) for f in ...]"

# Run CLI
python cli.py new                    # Interactive novel creation
python cli.py outline --novel X      # View outlines
python cli.py write --novel X --volume 1  # Generate volume 1
python cli.py audit --novel X --volume 1  # Audit volume 1
python cli.py status --novel X       # Check progress
python cli.py continue --novel X     # Auto-detect and write next volume
```

## Architecture

**Dependency graph** (top → bottom):

```
cli.py (argparse + rich)
 ├── outline/manager.py  ─── outline/generator.py
 ├── generator/volume_writer.py  ─── generator/chapter_writer.py
 │                                    └── memory/{character,item,foreshadowing}.py
 │                                    └── memory/summary.py
 ├── auditor/auditor.py
 └── config/style.py
       │
       └── llm/client.py  ─── OpenAI-compatible API (DeepSeek default)
       └── storage/{novel_store,table_store}.py  ─── Markdown + JSON on disk
```

**Key relationships**:

- `AppContext` (in `cli.py`) is the DI container — it initializes all modules once and passes them around. Factory methods (`make_*`) create per-novel instances of memory managers and writers.
- `OutlineManager.get_context_for_volume()` is the central context-assembly function. It gathers total outline + previous volume summaries + last 2 chapters of preceding volume + current memory tables. Every other module calls this instead of assembling context manually.
- Memory managers (`CharacterManager`, `ItemManager`, `ForeshadowingManager`) all wrap `TableStore` for persistence and use `LLMClient` for auto-extraction from chapter text via `extract_from_chapter()`. They also provide `prompt_context()` for injection into writing prompts.
- The **auditor is a single-LLM-call design**: one prompt checks both logic consistency and AI-flavor patterns. Regex pre-scans catch obvious AI-pattern repetitions before the LLM call to give the model concrete examples.

## Data layout on disk

```
data/novels/<novel-name>/
  meta.json              # title, author_style, genre
  total_outline.json     # world_setting, main_plot_arc, volume_plan, synopsis
  characters.json        # {char_001: {name, status, traits, relationships, ...}}
  items.json             # {item_001: {name, type, owner, status, ...}}
  foreshadowing.json     # {fh_001: {description, status, planted_in, resolved_in, ...}}
  volume_summaries.json  # [{volume, summary, key_events, ...}]
  volumes/
    vol_001/
      outline.json       # volume_title, chapter_plan, volume_arc, synopsis
      synopsis.txt
      chapters/
        ch_001.md        # "## 第1章 标题\n\n正文..."
        ch_002.md
```

## Encoding / locale

The top of `cli.py` contains a `_setup_encoding()` block that MUST run before any imports that touch `readline`. It:
1. Forces `LANG`/`LC_ALL`/`LC_CTYPE` env vars to `zh_CN.UTF-8`
2. Calls `locale.setlocale(LC_ALL, "zh_CN.UTF-8")`
3. Injects `gnureadline` as `sys.modules["readline"]`

This is required for correct Chinese-character backspace handling in interactive input. Do not remove or weaken this block without testing with PTY-simulated CJK input.

## Config

`config.yaml` (not committed, gitignored) — copy from `config.yaml.example`. Supports env var overrides: `DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL`, `DEEPSEEK_MODEL`. The `generation` section controls `chapters_per_volume` (default 50) and `words_per_chapter` (default 5000).

## Style system

`StyleManager` in `src/config/style.py` has:
- 5 built-in author presets (辰东式热血, 猫腻式文艺, 番茄式爽文, 老鹰式细节, 唐家三少式)
- 6 genre presets (玄幻, 都市, 科幻, 仙侠, 历史, 悬疑)
- Custom styles: `create_custom_style(name, base_style=None, **params)` → `save_custom_style(name, params)`. Saved as `data/custom_styles/<name>.json`. Custom styles appear with `[自定义]` prefix in listings.
- `get_style_prompt(name)` returns a Chinese-format string injected into LLM system prompts.
