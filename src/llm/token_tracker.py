"""Token usage tracking for LLM API calls.

Provides a singleton-style tracker that accumulates input/output token
counts across all LLM calls in a session, broken down by operation type
(generation, auditing, outline, memory, summary).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TokenUsage:
    """Accumulated token usage for a single category."""

    input_tokens: int = 0
    output_tokens: int = 0
    call_count: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def add(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.call_count += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "call_count": self.call_count,
        }


class TokenTracker:
    """Tracks token usage across all LLM calls in a session.

    Usage::

        tracker = TokenTracker()
        tracker.record("generation", 1500, 3000)
        tracker.record("auditing", 5000, 800)
        print(tracker.report())

    Categories:
        - generation: chapter/batch/continuous writing
        - outline: total and volume outline generation
        - auditing: logic + AI-flavor audit
        - memory: character/item/foreshadowing extraction
        - summary: volume summary generation
        - other: any other LLM calls
    """

    CATEGORIES = ("generation", "outline", "auditing", "memory", "summary", "other")

    def __init__(self) -> None:
        self._usage: dict[str, TokenUsage] = {cat: TokenUsage() for cat in self.CATEGORIES}
        self._current_category: str = "other"

    def set_category(self, category: str) -> None:
        """Set the current operation category for subsequent calls."""
        if category not in self.CATEGORIES:
            category = "other"
        self._current_category = category

    def record(self, category: str, input_tokens: int, output_tokens: int) -> None:
        """Record token usage for a specific category."""
        if category not in self.CATEGORIES:
            category = "other"
        self._usage[category].add(input_tokens, output_tokens)

    def record_auto(self, input_tokens: int, output_tokens: int) -> None:
        """Record token usage under the current category."""
        self.record(self._current_category, input_tokens, output_tokens)

    def get_usage(self, category: str) -> TokenUsage:
        """Get usage for a specific category."""
        return self._usage.get(category, TokenUsage())

    def get_total(self) -> TokenUsage:
        """Get total usage across all categories."""
        total = TokenUsage()
        for usage in self._usage.values():
            total.input_tokens += usage.input_tokens
            total.output_tokens += usage.output_tokens
            total.call_count += usage.call_count
        return total

    def report(self) -> str:
        """Generate a formatted usage report."""
        lines: list[str] = []
        lines.append("=" * 50)
        lines.append("Token 使用统计")
        lines.append("=" * 50)

        total = self.get_total()

        for cat in self.CATEGORIES:
            usage = self._usage[cat]
            if usage.call_count > 0:
                lines.append(
                    f"  {cat:12s}: {usage.input_tokens:>10,} 输入 + "
                    f"{usage.output_tokens:>10,} 输出 = "
                    f"{usage.total_tokens:>10,} 总计 "
                    f"({usage.call_count} 次调用)"
                )

        lines.append("-" * 50)
        lines.append(
            f"  {'总计':12s}: {total.input_tokens:>10,} 输入 + "
            f"{total.output_tokens:>10,} 输出 = "
            f"{total.total_tokens:>10,} 总计 "
            f"({total.call_count} 次调用)"
        )
        lines.append("=" * 50)

        return "\n".join(lines)

    def report_dict(self) -> dict[str, Any]:
        """Return usage as a dict for programmatic access."""
        result: dict[str, Any] = {}
        for cat in self.CATEGORIES:
            usage = self._usage[cat]
            if usage.call_count > 0:
                result[cat] = usage.to_dict()
        result["total"] = self.get_total().to_dict()
        return result

    def reset(self) -> None:
        """Reset all counters."""
        for usage in self._usage.values():
            usage.input_tokens = 0
            usage.output_tokens = 0
            usage.call_count = 0
