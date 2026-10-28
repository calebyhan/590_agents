"""Redaction and size limits for displayed and persisted session data."""

import os
import re
from typing import Any, cast

_SECRET_PATTERNS = (
    re.compile(r"(?i)(authorization\s*[:=]\s*(?:bearer\s+)?)[^\s,;]+"),
    re.compile(r"(?i)\b((?:api[_-]?key|token|password|secret)\s*[:=]\s*)[^\s,;]+"),
    re.compile(r"\b(?:sk|ghp|github_pat)-?[A-Za-z0-9_\-]{16,}\b"),
)
_SECRET_ENV_NAMES = re.compile(r"(?i)(key|token|secret|password|credential|authorization)")


def redact_text(text: str) -> str:
    """Remove common credentials and configured secret environment values."""
    result = text
    for pattern in _SECRET_PATTERNS:
        result = pattern.sub(
            lambda match: (match.group(1) if match.lastindex else "") + "[REDACTED]",
            result,
        )
    for name, value in os.environ.items():
        if _SECRET_ENV_NAMES.search(name) and len(value) >= 6:
            result = result.replace(value, "[REDACTED]")
    return result


def bounded(text: str, limit: int) -> str:
    """Bound text while making omitted content explicit."""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n… ({len(text) - limit:,} characters omitted)"


def redact_data(value: Any) -> Any:  # noqa: ANN401
    """Recursively redact strings in JSON-compatible session data."""
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, list):
        return [redact_data(item) for item in cast(list[Any], value)]
    if isinstance(value, dict):
        mapping = cast(dict[Any, Any], value)
        return {str(key): redact_data(item) for key, item in mapping.items()}
    return value
