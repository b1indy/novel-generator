"""Dual-dimension auditor — single-LLM-call logic-consistency + AI-flavor check.

Provides the Auditor class, which audits an entire volume for both logical
errors (character continuity, timeline, items, foreshadowing, world-building,
power progression) and AI-generated-writing patterns (repetitive sentence
structures, template descriptions, stiff dialogue, etc.) in a single LLM call
to save tokens.  Also provides automated fix-application based on audit reports.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from ..llm.client import LLMClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex patterns for pre-scanning AI-flavor repetitions
# ---------------------------------------------------------------------------

_AI_PATTERN_STRINGS: list[str] = [
    r"就在这时",
    r"突然之间",
    r"眼中闪过一丝",
    r"嘴角微微上扬",
    r"心中不由得",
    r"不由得倒吸一口",
    r"说时迟那时快",
    r"一股强大的",
    r"缓缓地",
    r"深深地吸了一口气",
]

# Pre-compiled regexes (case-insensitive matching is not needed for Chinese).
_AI_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p) for p in _AI_PATTERN_STRINGS
]

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class AuditIssue:
    """A single issue found during audit."""

    severity: str  # "critical", "major", "minor"
    category: str  # "logic" or "ai_flavor"
    location: str  # e.g. "ch_003" or "ch_005-007"
    description: str  # what is wrong
    suggestion: str  # how to fix it
    affected_text: Optional[str] = None  # the problematic text snippet


@dataclass
class AuditReport:
    """Complete audit report for a volume."""

    volume_num: int
    logic_issues: list[AuditIssue] = field(default_factory=list)
    ai_flavor_issues: list[AuditIssue] = field(default_factory=list)
    overall_score: float = 0.0  # 0-10
    summary: str = ""

    @property
    def all_issues(self) -> list[AuditIssue]:
        """Return a combined list of logic and AI-flavor issues."""
        return self.logic_issues + self.ai_flavor_issues

    @property
    def critical_issues(self) -> list[AuditIssue]:
        """Return only critical-severity issues."""
        return [i for i in self.all_issues if i.severity == "critical"]

    @property
    def passed(self) -> bool:
        """True when there are no critical issues."""
        return len(self.critical_issues) == 0


# ---------------------------------------------------------------------------
# Auditor
# ---------------------------------------------------------------------------


class Auditor:
    """Dual-dimension auditor for web-novel volumes.

    Accepts an :class:`LLMClient` for making API calls and provides
    methods to audit an entire volume (:meth:`audit_volume`) and to
    apply fixes based on the resulting report (:meth:`fix_issues`).

    The audit is performed in a **single** LLM call that covers both
    logic consistency and AI-flavor detection, preceded by a cheap
    regex pre-scan to surface obvious repetitive patterns to the LLM
    as additional context.
    """

    def __init__(self, llm_client: LLMClient) -> None:
        """*llm_client* is used for both audit and fix calls."""
        self._llm = llm_client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def audit_volume(
        self,
        novel_name: str,
        volume_num: int,
        volume_outline: str,
        chapters: list[dict[str, str]],
        memory_tables: dict[str, Any],
    ) -> AuditReport:
        """Audit an entire volume and return an :class:`AuditReport`.

        Args:
            novel_name: Human-readable novel title (used in prompts).
            volume_num: The volume number (1-indexed).
            volume_outline: The volume-level outline text.
            chapters: List of ``{"title": ..., "content": ...}`` dicts,
                one per chapter in reading order.
            memory_tables: Dict with keys ``"characters"``, ``"items"``,
                ``"foreshadowing"``, each mapping to the corresponding
                state table (dict-of-dicts as produced by the memory
                managers).

        Returns:
            An :class:`AuditReport` populated with logic issues,
            AI-flavor issues, overall score, and a summary.
        """
        prompt = self._build_audit_prompt(volume_outline, chapters, memory_tables)

        messages: list[dict[str, str]] = [
            {"role": "system", "content": _AUDIT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        logger.info(
            "Auditing volume %d of %r (%d chapters) …",
            volume_num,
            novel_name,
            len(chapters),
        )

        response = self._llm.chat(messages, temperature=0.15)

        report = self._parse_audit_response(response, volume_num)

        logger.info(
            "Audit complete: score=%.1f logic=%d ai_flavor=%d critical=%d passed=%s",
            report.overall_score,
            len(report.logic_issues),
            len(report.ai_flavor_issues),
            len(report.critical_issues),
            report.passed,
        )

        return report

    def audit_volume_batched(
        self,
        novel_name: str,
        volume_num: int,
        volume_outline: str,
        chapters: list[dict[str, str]],
        memory_tables: dict[str, Any],
        batch_size: int = 10,
        overlap: int = 3,
    ) -> AuditReport:
        """Audit a volume in overlapping batches for better cross-batch continuity.

        Each batch covers ``batch_size`` chapters, overlapping ``overlap``
        chapters with the previous batch. This ensures cross-batch logic
        consistency is checked.

        Args:
            novel_name: Human-readable novel title.
            volume_num: The volume number (1-indexed).
            volume_outline: The volume-level outline text.
            chapters: List of chapter dicts.
            memory_tables: Character/item/foreshadowing tables.
            batch_size: Chapters per batch (default 10).
            overlap: Chapters overlapping between batches (default 3).

        Returns:
            Merged :class:`AuditReport` with deduplicated issues.
        """
        total = len(chapters)
        if total <= batch_size:
            # Small enough to audit in one call.
            return self.audit_volume(
                novel_name, volume_num, volume_outline,
                chapters, memory_tables,
            )

        # Generate overlapping batch ranges.
        batches: list[tuple[int, int]] = []
        start = 0
        while start < total:
            end = min(start + batch_size, total)
            batches.append((start, end))
            if end >= total:
                break
            start = end - overlap

        logger.info(
            "Batch audit: %d chapters, %d batches (size=%d, overlap=%d)",
            total, len(batches), batch_size, overlap,
        )

        all_logic: list[AuditIssue] = []
        all_ai_flavor: list[AuditIssue] = []
        scores: list[float] = []
        summaries: list[str] = []

        # Track seen issues to deduplicate in overlap zones.
        seen_descriptions: set[str] = set()

        for batch_idx, (batch_start, batch_end) in enumerate(batches):
            batch_chapters = chapters[batch_start:batch_end]
            batch_label = f"ch_{batch_start + 1:03d}-ch_{batch_end:03d}"

            logger.info(
                "  Batch %d/%d: %s (%d chapters)",
                batch_idx + 1, len(batches), batch_label, len(batch_chapters),
            )

            try:
                batch_report = self.audit_volume(
                    novel_name=novel_name,
                    volume_num=volume_num,
                    volume_outline=volume_outline,
                    chapters=batch_chapters,
                    memory_tables=memory_tables,
                )
            except Exception:
                logger.exception("Batch %d audit failed, skipping", batch_idx + 1)
                continue

            scores.append(batch_report.overall_score)
            if batch_report.summary:
                summaries.append(f"[{batch_label}] {batch_report.summary}")

            # Deduplicate issues by description prefix.
            for issue in batch_report.logic_issues:
                key = f"{issue.category}:{issue.description[:50]}"
                if key not in seen_descriptions:
                    seen_descriptions.add(key)
                    all_logic.append(issue)

            for issue in batch_report.ai_flavor_issues:
                key = f"{issue.category}:{issue.description[:50]}"
                if key not in seen_descriptions:
                    seen_descriptions.add(key)
                    all_ai_flavor.append(issue)

        # Compute weighted average score (larger batches weigh more).
        avg_score = sum(scores) / len(scores) if scores else 0.0

        merged = AuditReport(
            volume_num=volume_num,
            logic_issues=all_logic,
            ai_flavor_issues=all_ai_flavor,
            overall_score=round(avg_score, 1),
            summary="\n".join(summaries),
        )

        logger.info(
            "Batch audit complete: score=%.1f logic=%d ai_flavor=%d passed=%s",
            merged.overall_score,
            len(merged.logic_issues),
            len(merged.ai_flavor_issues),
            merged.passed,
        )

        return merged

    def fix_issues(
        self,
        novel_name: str,
        volume_num: int,
        audit_report: AuditReport,
        chapters: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """Apply fixes for issues identified in *audit_report*.

        Issues are grouped by location (chapter), and a single targeted
        LLM call is made per affected chapter to fix all issues at once.
        Chapters without issues are returned unchanged.

        Args:
            novel_name: Human-readable novel title.
            volume_num: The volume number.
            audit_report: The report produced by :meth:`audit_volume`.
            chapters: The original list of chapter dicts.

        Returns:
            A new list of chapter dicts with fixes applied.
        """
        if not audit_report.all_issues:
            logger.info("No issues to fix for volume %d.", volume_num)
            return chapters

        # Build a map of chapter index -> list of issues.
        chapter_issues: dict[int, list[AuditIssue]] = {}
        for issue in audit_report.all_issues:
            indices = self._parse_location(issue.location, len(chapters))
            for idx in indices:
                chapter_issues.setdefault(idx, []).append(issue)

        # Work on a copy so we don't mutate the input.
        fixed = list(chapters)

        for ch_idx, issues in sorted(chapter_issues.items()):
            ch_title = fixed[ch_idx].get("title", f"ch_{ch_idx + 1:03d}")
            ch_content = fixed[ch_idx].get("content", "")

            logger.info(
                "Fixing %d issue(s) in chapter %d (%s) …",
                len(issues),
                ch_idx + 1,
                ch_title,
            )

            fix_prompt = self._build_fix_prompt(
                ch_title, ch_content, issues
            )
            fix_messages: list[dict[str, str]] = [
                {"role": "system", "content": _FIX_SYSTEM_PROMPT},
                {"role": "user", "content": fix_prompt},
            ]

            try:
                new_content = self._llm.chat(fix_messages, temperature=0.4)
                fixed_content = self._extract_fixed_content(new_content, ch_content)
                fixed[ch_idx] = {
                    "title": ch_title,
                    "content": fixed_content,
                }
            except Exception as exc:
                logger.error(
                    "Failed to fix chapter %d (%s): %s",
                    ch_idx + 1,
                    ch_title,
                    exc,
                )
                # Keep the original chapter on failure.

        return fixed

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_audit_prompt(
        self,
        volume_outline: str,
        chapters: list[dict[str, str]],
        memory_tables: dict[str, Any],
    ) -> str:
        """Build the combined audit prompt covering both dimensions.

        The prompt includes:

        * Volume outline
        * Full chapter texts (with chapter markers)
        * Memory tables (characters, items, foreshadowing)
        * Regex pre-scan results for obvious AI-flavor patterns
        * Explicit JSON output format instructions
        """
        # --- Pre-scan for AI-flavor patterns ---------------------------------
        pattern_hits: dict[str, list[str]] = {}
        for i, ch in enumerate(chapters):
            ch_label = f"ch_{i + 1:03d}"
            content = ch.get("content", "")
            for pi, regex in enumerate(_AI_PATTERNS):
                matches = regex.findall(content)
                if matches:
                    pattern_key = _AI_PATTERN_STRINGS[pi]
                    pattern_hits.setdefault(pattern_key, []).append(
                        f"{ch_label} ({len(matches)} occurrences)"
                    )

        regex_report = ""
        if pattern_hits:
            lines = ["[Regex pre-scan — obvious AI-flavor repetitions detected]"]
            for pat, locations in sorted(pattern_hits.items()):
                lines.append(f"  Pattern {pat!r}: {', '.join(locations)}")
            regex_report = "\n".join(lines)
        else:
            regex_report = (
                "[Regex pre-scan] No obvious repetitive patterns detected "
                "by the hardcoded scanner.  Please still check for semantic "
                "AI-flavor issues."
            )

        # --- Format chapters -------------------------------------------------
        chapters_text_parts: list[str] = []
        for i, ch in enumerate(chapters):
            label = f"ch_{i + 1:03d}"
            title = ch.get("title", "(no title)")
            content = ch.get("content", "")
            chapters_text_parts.append(
                f"=== {label}: {title} ===\n{content}"
            )
        chapters_text = "\n\n".join(chapters_text_parts)

        # --- Format memory tables --------------------------------------------
        memory_text_parts: list[str] = []

        # Characters.
        characters: dict[str, Any] = memory_tables.get("characters", {})
        if characters:
            char_lines = ["[Characters]"]
            for cid, rec in characters.items():
                name = rec.get("name", cid)
                status = rec.get("status", "alive")
                location = rec.get("current_location", "")
                traits = rec.get("traits", [])
                cultivation = rec.get("cultivation", "")
                relationships = rec.get("relationships", {})

                parts = [f"  - {name} (id={cid}, status={status})"]
                if location:
                    parts.append(f"    location: {location}")
                if cultivation:
                    parts.append(f"    cultivation: {cultivation}")
                if traits:
                    parts.append(f"    traits: {', '.join(traits)}")
                if relationships:
                    rel_str = "; ".join(
                        f"{t} -> {r}" for t, r in relationships.items()
                    )
                    parts.append(f"    relationships: {rel_str}")
                char_lines.extend(parts)
            memory_text_parts.append("\n".join(char_lines))
        else:
            memory_text_parts.append("[Characters] (none recorded)")

        # Items.
        items: dict[str, Any] = memory_tables.get("items", {})
        if items:
            item_lines = ["[Items]"]
            for iid, rec in items.items():
                name = rec.get("name", iid)
                item_type = rec.get("type", "")
                grade = rec.get("grade", "")
                owner = rec.get("owner", "")
                status = rec.get("status", "in_use")
                significance = rec.get("significance", "")

                parts = [
                    f"  - {name} (id={iid}, type={item_type}, "
                    f"grade={grade}, status={status})"
                ]
                if owner:
                    parts.append(f"    owner: {owner}")
                if significance:
                    parts.append(f"    significance: {significance}")
                item_lines.extend(parts)
            memory_text_parts.append("\n".join(item_lines))
        else:
            memory_text_parts.append("[Items] (none recorded)")

        # Foreshadowing.
        foreshadowing: dict[str, Any] = memory_tables.get("foreshadowing", {})
        if foreshadowing:
            fh_lines = ["[Foreshadowing]"]
            for fid, rec in foreshadowing.items():
                status = rec.get("status", "pending")
                priority = rec.get("priority", "medium")
                desc = rec.get("description", fid)
                planted = rec.get("planted_in", "?")
                resolved = rec.get("resolved_in", "")
                planned = rec.get("planned_resolution", "")
                related = rec.get("related_characters", [])

                parts = [
                    f"  - [{fid}] ({status}, priority={priority}) {desc}"
                ]
                parts.append(f"    planted: {planted}")
                if resolved:
                    parts.append(f"    resolved: {resolved}")
                if related:
                    parts.append(f"    characters: {', '.join(related)}")
                if planned:
                    parts.append(f"    planned: {planned}")
                fh_lines.extend(parts)
            memory_text_parts.append("\n".join(fh_lines))
        else:
            memory_text_parts.append("[Foreshadowing] (none recorded)")

        memory_text = "\n\n".join(memory_text_parts)

        # --- Assemble final prompt -------------------------------------------
        return _AUDIT_USER_PROMPT.format(
            volume_outline=volume_outline,
            memory_tables=memory_text,
            chapters_text=chapters_text,
            regex_report=regex_report,
        )

    def _build_fix_prompt(
        self,
        chapter_title: str,
        chapter_content: str,
        issues: list[AuditIssue],
    ) -> str:
        """Build a targeted fix prompt for a single chapter with its issues."""
        issues_text_parts: list[str] = []
        for i, issue in enumerate(issues, 1):
            issues_text_parts.append(
                f"Issue #{i} [{issue.severity}] [{issue.category}]:\n"
                f"  Description: {issue.description}\n"
                f"  Suggested fix: {issue.suggestion}"
            )
            if issue.affected_text:
                issues_text_parts.append(
                    f"  Affected text: {issue.affected_text}"
                )
        issues_text = "\n\n".join(issues_text_parts)

        return _FIX_USER_PROMPT.format(
            chapter_title=chapter_title,
            chapter_content=chapter_content,
            issues_text=issues_text,
        )

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_audit_response(
        self, response_text: str, volume_num: int
    ) -> AuditReport:
        """Parse the LLM's JSON response into an :class:`AuditReport`.

        Handles:

        * Markdown code fences (`` ```json ... ``` ``)
        * Missing / extra fields with sensible defaults
        * Type coercion where safe (score as float, passed computed)
        * Partial recovery: if JSON is valid but missing sub-lists,
          those lists default to empty.

        Raises :class:`ValueError` if the response cannot be parsed at all.
        """
        text = _strip_code_fences(response_text)

        try:
            data: dict[str, Any] = json.loads(text)
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse audit response as JSON: %s", exc)
            raise ValueError(
                f"Audit response is not valid JSON: {exc}"
            ) from exc

        if not isinstance(data, dict):
            raise ValueError(
                f"Audit response must be a JSON object, got {type(data).__name__}"
            )

        # Parse logic issues.
        logic_issues: list[AuditIssue] = []
        for raw in data.get("logic_issues", []):
            try:
                logic_issues.append(self._parse_issue(raw, category="logic"))
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning("Skipping malformed logic issue: %s", exc)

        # Parse AI-flavor issues.
        ai_flavor_issues: list[AuditIssue] = []
        for raw in data.get("ai_flavor_issues", []):
            try:
                ai_flavor_issues.append(
                    self._parse_issue(raw, category="ai_flavor")
                )
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning("Skipping malformed AI-flavor issue: %s", exc)

        # Overall score — coerce to float, clamp to 0-10.
        raw_score = data.get("overall_score", 0)
        try:
            overall_score = float(raw_score)
        except (TypeError, ValueError):
            overall_score = 0.0
        overall_score = max(0.0, min(10.0, overall_score))

        summary: str = str(data.get("summary", ""))

        report = AuditReport(
            volume_num=volume_num,
            logic_issues=logic_issues,
            ai_flavor_issues=ai_flavor_issues,
            overall_score=overall_score,
            summary=summary,
        )

        # If the LLM provided a "passed" field, warn on mismatch.
        if "passed" in data:
            llm_passed = bool(data["passed"])
            if llm_passed != report.passed:
                logger.debug(
                    "LLM reported passed=%s but computed passed=%s "
                    "(%d critical issues); using computed value.",
                    llm_passed,
                    report.passed,
                    len(report.critical_issues),
                )

        return report

    @staticmethod
    def _parse_issue(raw: dict[str, Any], category: str) -> AuditIssue:
        """Parse a single issue dict into an :class:`AuditIssue`.

        Args:
            raw: The raw dict from the LLM response.
            category: ``"logic"`` or ``"ai_flavor"`` — used as fallback
                when the dict does not contain a ``category`` field.

        Raises:
            KeyError: A required field is missing.
            ValueError: A field has an invalid value.
        """
        severity = raw.get("severity", "minor")
        if severity not in ("critical", "major", "minor"):
            logger.debug(
                "Invalid severity %r, defaulting to 'minor'", severity
            )
            severity = "minor"

        cat = raw.get("category", category)
        if cat not in ("logic", "ai_flavor"):
            cat = category

        return AuditIssue(
            severity=severity,
            category=cat,
            location=str(raw.get("location", "unknown")),
            description=str(raw.get("description", "")),
            suggestion=str(raw.get("suggestion", raw.get("suggested_fix", ""))),
            affected_text=raw.get("affected_text"),
        )

    # ------------------------------------------------------------------
    # Fix helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_location(location: str, num_chapters: int) -> list[int]:
        """Parse a location string into a list of 0-indexed chapter indices.

        Supports formats like:

        * ``"ch_003"`` -> ``[2]``
        * ``"ch_005-007"`` -> ``[4, 5, 6]``
        * ``"ch_001,ch_003"`` -> ``[0, 2]`` (comma-separated)
        * ``"ch_001-ch_003"`` -> ``[0, 1, 2]``

        Out-of-range indices are silently clamped.
        """
        indices: set[int] = set()
        # Split on commas first.
        parts = re.split(r",\s*", location)
        for part in parts:
            part = part.strip()
            # Try range: "ch_005-ch_007" or "ch_005-007"
            range_match = re.match(
                r"ch_(\d+)\s*-\s*(?:ch_)?(\d+)", part
            )
            if range_match:
                start = int(range_match.group(1))
                end = int(range_match.group(2))
                for n in range(start, end + 1):
                    idx = n - 1
                    if 0 <= idx < num_chapters:
                        indices.add(idx)
                continue

            # Try single: "ch_005"
            single_match = re.match(r"ch_(\d+)", part)
            if single_match:
                n = int(single_match.group(1))
                idx = n - 1
                if 0 <= idx < num_chapters:
                    indices.add(idx)

        return sorted(indices)

    @staticmethod
    def _extract_fixed_content(
        llm_response: str, original_content: str
    ) -> str:
        """Extract the fixed chapter content from the LLM response.

        If the LLM wraps the output in markdown fences or adds commentary,
        strip those away.  Falls back to the original content when the
        response is empty or clearly not a chapter.
        """
        text = _strip_code_fences(llm_response).strip()

        _MIN_FIX_LENGTH = 50

        if not text:
            logger.warning(
                "Empty fix response; keeping original chapter content."
            )
            return original_content

        # Heuristic: if the response is very short (less than a minimum
        # absolute threshold or less than 10% of original length), it is
        # probably an error message, not a fix.
        if len(text) < _MIN_FIX_LENGTH or len(text) < len(original_content) * 0.1:
            logger.warning(
                "Fix response is suspiciously short (%d chars vs %d "
                "original); keeping original chapter content.",
                len(text),
                len(original_content),
            )
            return original_content

        return text


# ---------------------------------------------------------------------------
# Helper: strip markdown code fences
# ---------------------------------------------------------------------------


def _strip_code_fences(text: str) -> str:
    """Remove surrounding markdown code fences from *text* if present.

    Handles both `` ```json ``` `` and plain `` ``` ``` `` fences.
    Leading/trailing whitespace on fence lines is tolerated.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.split("\n")
        # Remove opening fence (may be "```json", "```", etc.).
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        # Remove closing fence.
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped


# ---------------------------------------------------------------------------
# LLM prompts
# ---------------------------------------------------------------------------

_AUDIT_SYSTEM_PROMPT = """\
You are a meticulous web-novel auditor. Your job is to review an entire \
volume of a Chinese web novel and produce a structured audit report that \
covers TWO dimensions simultaneously:

1. **Logic consistency** — factual, causal, and world-building errors.
2. **AI-flavor detection** — stylistic patterns that make writing feel \
generated rather than human-authored.

You must output ONLY valid JSON (no markdown fences, no commentary). \
The JSON must conform EXACTLY to the schema described below."""

_AUDIT_USER_PROMPT = """\
Please audit the following volume of a web novel.  You will receive:

1. The volume outline (the author's plan for this volume).
2. Memory tables (characters, items, foreshadowing) that track the \
current established state of the story world.
3. The full text of every chapter in this volume.
4. A regex pre-scan report flagging obvious repetitive phrase patterns.

=== DIMENSION 1: LOGIC CONSISTENCY ===

Check for:

- **Character continuity**: Dead characters must NOT reappear. Character \
traits, abilities, and relationships must stay consistent (or change for \
explained, narratively justified reasons).  Check that every character \
appearing in the chapters has a matching entry in the memory table (new \
characters are fine; report them as "minor" informational notes).

- **Timeline consistency**: Events must not violate causality. No "he \
arrived before he departed" style contradictions. Time skips must be \
explicit and reasonable.

- **Item tracking**: Items must not change hands without explanation. An \
item recorded as "destroyed" or "lost" must not be used later without a \
recovery explanation.

- **Foreshadowing**: Planted hints (especially high-priority ones) must \
be followed up — check whether any foreshadowing planted earlier is \
addressed in this volume.  Flag dropped threads and unresolved \
high-priority foreshadowing that should have been addressed.

- **World-building consistency**: The rules of this world (cultivation \
system, magic rules, social hierarchies, geography, etc.) must remain \
internally consistent.  If a rule established earlier is broken, flag it.

- **Power / ability progression**: Character power levels and abilities \
must progress consistently with the established cultivation / power \
system.  Sudden unexplained power-ups are suspicious.

=== DIMENSION 2: AI-FLAVOR DETECTION ===

Look for these common signs of AI-generated or template-driven writing:

- **Repetitive sentence structures**: The same sentence opening appearing \
multiple times in close proximity (e.g. "就在这时..." used 3+ times in \
one chapter, or the same transition phrase at every paragraph start).

- **Template-like descriptions**: Overuse of formulaic expressions such \
as "眼中闪过一丝...", "嘴角微微上扬...", "心中不由得...", \
"不由得倒吸一口..." etc.  These are common in Chinese web novels but \
excessive repetition makes writing feel mechanical.

- **Overly formal or stiff dialogue**: Characters speaking in unnaturally \
complete sentences, lacking contractions, colloquialisms, or individual \
voice.  Dialogue should match each character's personality and background.

- **Lack of emotional depth / variation**: Every emotional beat is \
described the same way.  Characters show only surface-level emotions. \
The narrative doesn't vary its emotional register.

- **Predictable plot beats**: Every chapter ending with a cliffhanger, \
every fight following the same structure (opponent underestimated -> \
protagonist reveals hidden power -> victory).  Flag overly formulaic \
chapter structures.

- **Overuse of transition phrases**: The same few transition phrases \
("与此同时", "另一方面", "而在另一边", etc.) used as crutches \
between scenes.

- **Generic environmental descriptions**: Settings described with the \
same stock phrases ("天色已晚", "月明星稀", "云雾缭绕") without \
specific, vivid detail that grounds the scene.

=== MEMORY TABLES (established story state) ===
{memory_tables}

=== VOLUME OUTLINE ===
{volume_outline}

=== CHAPTERS ===
{chapters_text}

=== REGEX PRE-SCAN ===
{regex_report}

=== OUTPUT FORMAT ===

Output a single JSON object with this exact structure:

{{
  "logic_issues": [
    {{
      "severity": "critical|major|minor",
      "category": "logic",
      "location": "ch_003 or ch_005-007",
      "description": "What the problem is, with specific references.",
      "suggestion": "Concrete suggestion for how to fix it.",
      "affected_text": "Optional: the problematic text snippet."
    }}
  ],
  "ai_flavor_issues": [
    {{
      "severity": "major|minor",
      "category": "ai_flavor",
      "location": "ch_007",
      "description": "What pattern was detected and why it's a problem.",
      "suggestion": "How to rewrite to sound more natural and human.",
      "affected_text": "Optional: the problematic text snippet."
    }}
  ],
  "overall_score": 7.5,
  "summary": "Brief overall assessment in 2-4 sentences."
}}

IMPORTANT RULES:
- Severity "critical" means the story is BROKEN (e.g. dead character \
reappears, fundamental world-building rule violated).  These MUST be fixed.
- Severity "major" means a significant problem that hurts reading experience.
- Severity "minor" means a small issue or improvement opportunity.
- Be specific in descriptions: cite chapter numbers, character names, \
and quote relevant text.
- Make suggestions actionable: tell the writer exactly what to change.
- The overall_score is 0-10 (10 = perfect).  Be honest but fair.
- If you find NO issues in a category, return an empty list [].
- Only report real problems.  Do not invent issues."""

_FIX_SYSTEM_PROMPT = """\
You are an expert web-novel editor.  Your task is to rewrite a chapter \
to fix specific issues identified by an auditor, while preserving the \
original story content, characters, plot progression, and authorial voice \
as much as possible.

Output ONLY the corrected chapter text (no markdown fences, no \
commentary, no preamble).  The output must be the complete revised \
chapter, not just the changed portions."""

_FIX_USER_PROMPT = """\
Please fix the following issues in the chapter below.  Make the minimal \
changes necessary — do NOT rewrite the entire chapter from scratch unless \
the issues are pervasive.  Preserve the author's style and the chapter's \
original structure.

=== CHAPTER: {chapter_title} ===

{chapter_content}

=== ISSUES TO FIX ===

{issues_text}

=== INSTRUCTIONS ===
Output the complete fixed chapter.  Only change what is necessary to \
address the listed issues.  Keep everything else exactly as it was."""
