"""LLM client for DeepSeek API (OpenAI-compatible protocol).

Provides synchronous chat, streaming, and token counting with retry logic
and proper error handling.
"""

import logging
import os
import time
from pathlib import Path
from typing import Any, Generator, Optional

import yaml
from openai import APIStatusError, APITimeoutError, OpenAI, RateLimitError
from .token_tracker import TokenTracker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class LLMError(Exception):
    """Base exception for all LLM client errors."""


class LLMTimeoutError(LLMError):
    """Raised when an API call exhausts retries due to timeouts."""


class LLMRateLimitError(LLMError):
    """Raised when the API returns a rate-limit (429) response."""


# ---------------------------------------------------------------------------
# Token counting helpers
# ---------------------------------------------------------------------------

_ENCODING_CACHE: dict[str, Any] = {}


def _get_encoding(model: str) -> Any:
    """Return a tiktoken encoding for *model*, with caching."""
    if model in _ENCODING_CACHE:
        return _ENCODING_CACHE[model]

    try:
        import tiktoken

        # Try model-specific encoding first, fall back to cl100k_base.
        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
    except ImportError:
        enc = None

    _ENCODING_CACHE[model] = enc
    return enc


def _count_tokens_tiktoken(text: str, model: str) -> int:
    """Count tokens using tiktoken."""
    enc = _get_encoding(model)
    if enc is None:
        raise ImportError("tiktoken not available")
    return len(enc.encode(text))


def _count_tokens_fallback(text: str) -> int:
    """Approximate token count without tiktoken.

    Heuristic: Chinese characters ~0.5 tokens each, ASCII words ~1.3 tokens
    each.  This is intentionally coarse and only meant as a fallback.
    """
    import re

    chinese_chars = len(re.findall(r"[一-鿿]", text))
    non_chinese = text
    for ch in re.findall(r"[一-鿿]", text):
        non_chinese = non_chinese.replace(ch, "")
    words = len(non_chinese.split())

    return int(chinese_chars * 0.5 + words * 1.3) or 1


# ---------------------------------------------------------------------------
# LLMClient
# ---------------------------------------------------------------------------


class LLMClient:
    """Client for DeepSeek's OpenAI-compatible API.

    Loads configuration from a YAML file and overrides individual settings
    with environment variables (``DEEPSEEK_API_KEY``, ``DEEPSEEK_BASE_URL``,
    ``DEEPSEEK_MODEL``).

    Includes automatic retry (3 attempts, exponential backoff 1s/2s/4s) on
    transient API errors and synchronous / streaming chat methods.
    """

    # Defaults (used when neither config nor env var provides a value)
    _DEFAULT_BASE_URL: str = "https://api.deepseek.com"
    _DEFAULT_MODEL: str = "deepseek-chat"
    _DEFAULT_MAX_TOKENS: int = 8192
    _DEFAULT_TEMPERATURE: float = 0.8

    _MAX_RETRIES: int = 3
    """Maximum number of retries after the initial attempt (4 total calls)."""

    _RETRY_DELAYS: tuple[float, ...] = (1.0, 2.0, 4.0)
    """Exponential backoff delays in seconds for each retry."""

    _RETRYABLE_STATUSES: frozenset[int] = frozenset({429, 500, 502, 503, 504})

    # -- construction -------------------------------------------------------

    def __init__(
        self,
        config_path: str = "config.yaml",
        token_tracker: TokenTracker | None = None,
    ) -> None:
        """Create a new LLM client.

        Args:
            config_path: Path to the YAML configuration file.  Must contain
                an ``llm`` key with ``api_key``, ``base_url``, ``model``,
                ``max_tokens``, and ``temperature`` sub-keys.
            token_tracker: Optional TokenTracker for recording API usage.

        Raises:
            LLMError: If the config file is missing / unreadable or if no
                API key can be found.
        """
        self._config: dict[str, Any] = self._load_config(config_path)
        self._tracker = token_tracker

        # Resolve credentials & endpoint — env vars take precedence.
        self._api_key: str = os.getenv(
            "DEEPSEEK_API_KEY", self._config.get("api_key", "")
        )
        self._base_url: str = os.getenv(
            "DEEPSEEK_BASE_URL", self._config.get("base_url", self._DEFAULT_BASE_URL)
        )
        self._model: str = os.getenv(
            "DEEPSEEK_MODEL", self._config.get("model", self._DEFAULT_MODEL)
        )

        # Operational defaults from config (no env-var override).
        self._default_max_tokens: int = int(
            self._config.get("max_tokens", self._DEFAULT_MAX_TOKENS)
        )
        self._default_temperature: float = float(
            self._config.get("temperature", self._DEFAULT_TEMPERATURE)
        )

        if not self._api_key:
            raise LLMError(
                "No API key configured. Set DEEPSEEK_API_KEY in the environment "
                "or provide 'api_key' under the 'llm' key in config.yaml."
            )

        # OpenAI-compatible client (retries handled manually).
        self._client: OpenAI = OpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            max_retries=0,
        )

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _load_config(config_path: str) -> dict[str, Any]:
        """Load and return the ``llm`` section of *config_path*."""
        path = Path(config_path)
        if not path.exists():
            raise LLMError(f"Configuration file not found: {config_path}")
        try:
            with open(path, encoding="utf-8") as fh:
                raw: Any = yaml.safe_load(fh)
        except yaml.YAMLError as exc:
            raise LLMError(f"Failed to parse {config_path}: {exc}") from exc

        if not isinstance(raw, dict):
            raise LLMError(f"Unexpected YAML structure in {config_path}")

        return raw.get("llm", {})

    def _resolve_params(
        self,
        temperature: Optional[float],
        max_tokens: Optional[int],
    ) -> tuple[float, int]:
        """Return effective (temperature, max_tokens) using defaults when
        the caller passes ``None``.
        """
        return (
            self._default_temperature if temperature is None else temperature,
            self._default_max_tokens if max_tokens is None else max_tokens,
        )

    # -- retry decorator ----------------------------------------------------

    def _retry_call(self, callable_: Any, *args: Any, **kwargs: Any) -> Any:
        """Execute *callable_* with up to ``_MAX_RETRIES`` retries on
        transient errors, using exponential backoff.
        """
        last_exc: Optional[Exception] = None

        for attempt in range(self._MAX_RETRIES + 1):
            try:
                return callable_(*args, **kwargs)
            except APITimeoutError as exc:
                last_exc = exc
                if attempt < self._MAX_RETRIES:
                    delay = self._RETRY_DELAYS[attempt]
                    logger.warning(
                        "LLM call timed out (attempt %d/%d), retrying in %.1fs …",
                        attempt + 1,
                        self._MAX_RETRIES + 1,
                        delay,
                    )
                    time.sleep(delay)
            except RateLimitError as exc:
                last_exc = exc
                if attempt < self._MAX_RETRIES:
                    delay = self._RETRY_DELAYS[attempt]
                    logger.warning(
                        "LLM rate-limited (attempt %d/%d), retrying in %.1fs …",
                        attempt + 1,
                        self._MAX_RETRIES + 1,
                        delay,
                    )
                    time.sleep(delay)
            except APIStatusError as exc:
                if exc.status_code in self._RETRYABLE_STATUSES:
                    last_exc = exc
                    if attempt < self._MAX_RETRIES:
                        delay = self._RETRY_DELAYS[attempt]
                        logger.warning(
                            "LLM API error %d (attempt %d/%d), retrying in %.1fs …",
                            exc.status_code,
                            attempt + 1,
                            self._MAX_RETRIES + 1,
                            delay,
                        )
                        time.sleep(delay)
                else:
                    raise LLMError(f"API error {exc.status_code}: {exc}") from exc
            except Exception as exc:
                # Non-OpenAI exceptions are not retried.
                raise LLMError(f"Unexpected error during LLM call: {exc}") from exc

        # All retries exhausted — map to specific error type.
        if isinstance(last_exc, APITimeoutError):
            raise LLMTimeoutError(
                f"LLM call timed out after {self._MAX_RETRIES + 1} attempts"
            ) from last_exc
        if isinstance(last_exc, RateLimitError):
            raise LLMRateLimitError(
                f"LLM rate-limited after {self._MAX_RETRIES + 1} attempts"
            ) from last_exc
        raise LLMError(
            f"LLM call failed after {self._MAX_RETRIES + 1} attempts"
        ) from last_exc

    # -- public API ---------------------------------------------------------

    @property
    def model(self) -> str:
        """The active model name (e.g. ``deepseek-chat``)."""
        return self._model

    @property
    def base_url(self) -> str:
        """The API endpoint base URL."""
        return self._base_url

    @property
    def token_tracker(self) -> TokenTracker | None:
        """The token tracker, if configured."""
        return self._tracker

    @token_tracker.setter
    def token_tracker(self, tracker: TokenTracker) -> None:
        self._tracker = tracker

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> str:
        """Send a synchronous chat completion request and return the response
        text.

        Args:
            messages: List of message dicts with ``role`` and ``content`` keys.
            temperature: Sampling temperature (uses config default if ``None``).
            max_tokens: Max tokens to generate (uses config default if ``None``).
            stream: If ``True``, consume the stream internally and return the
                concatenated text.  For incremental consumption, use
                :meth:`chat_stream` instead.

        Returns:
            The response text as a string.

        Raises:
            LLMError: On non-retryable API errors.
            LLMTimeoutError: After all retries are exhausted due to timeouts.
            LLMRateLimitError: After all retries are exhausted due to 429.
        """
        effective_temp, effective_max = self._resolve_params(temperature, max_tokens)

        if stream:
            # Consume the generator internally so we still return a string.
            chunks: list[str] = []
            for chunk in self.chat_stream(
                messages, temperature=effective_temp, max_tokens=effective_max
            ):
                chunks.append(chunk)
            return "".join(chunks)

        def _call() -> str:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,  # type: ignore[arg-type]
                temperature=effective_temp,
                max_tokens=effective_max,
                stream=False,
            )
            content: Optional[str] = response.choices[0].message.content

            # Track token usage from API response.
            if self._tracker and hasattr(response, "usage") and response.usage:
                self._tracker.record_auto(
                    input_tokens=response.usage.prompt_tokens or 0,
                    output_tokens=response.usage.completion_tokens or 0,
                )

            return content or ""

        return self._retry_call(_call)

    def chat_stream(
        self,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Generator[str, None, None]:
        """Stream chat completion chunks as a generator.

        Args:
            messages: List of message dicts with ``role`` and ``content`` keys.
            temperature: Sampling temperature (uses config default if ``None``).
            max_tokens: Max tokens to generate (uses config default if ``None``).

        Yields:
            Text chunks as they arrive from the API.

        Raises:
            LLMError: On non-retryable API errors.
            LLMTimeoutError: After all retries are exhausted due to timeouts.
            LLMRateLimitError: After all retries are exhausted due to 429.
        """
        effective_temp, effective_max = self._resolve_params(temperature, max_tokens)

        def _call() -> Any:
            return self._client.chat.completions.create(
                model=self._model,
                messages=messages,  # type: ignore[arg-type]
                temperature=effective_temp,
                max_tokens=effective_max,
                stream=True,
            )

        stream_response = self._retry_call(_call)

        try:
            for chunk in stream_response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as exc:
            raise LLMError(f"Error while consuming stream: {exc}") from exc

    def count_tokens(self, text: str) -> int:
        """Return an approximate token count for *text*.

        Uses ``tiktoken`` when available (``cl100k_base`` encoding), otherwise
        falls back to a character-/word-based heuristic.

        Args:
            text: The text to count tokens for.

        Returns:
            Estimated number of tokens (always at least 1 for non-empty input).
        """
        if not text:
            return 0

        try:
            return _count_tokens_tiktoken(text, self._model)
        except (ImportError, Exception):
            logger.debug("tiktoken unavailable or failed; using fallback token count")
            return _count_tokens_fallback(text)
