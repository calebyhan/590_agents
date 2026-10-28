"""Versioned, redacted, atomic persistence for CalebJR sessions."""

import json
import os
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from agents590.calebjr.redaction import redact_data

SCHEMA_VERSION = 1
MAX_SAVED_SESSIONS = 50
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def sessions_dir() -> Path:
    root = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return root / "agents590" / "calebjr" / "sessions"


def session_path(identifier: str | Path) -> Path:
    candidate = Path(identifier).expanduser()
    if candidate.is_absolute() or candidate.parent != Path("."):
        return candidate.resolve()
    name = candidate.name
    if not _SAFE_NAME.fullmatch(name):
        raise ValueError("Session names may contain only letters, digits, '.', '_' and '-'.")
    if not name.endswith(".json"):
        name += ".json"
    return sessions_dir() / name


def json_value(value: Any) -> Any:  # noqa: ANN401
    """Convert SDK response values to JSON-compatible canonical input."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        mapping = cast(dict[Any, Any], value)
        return {str(key): json_value(item) for key, item in mapping.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in cast(list[Any] | tuple[Any, ...], value)]
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return json_value(dump(exclude_none=True))
    if hasattr(value, "__dict__"):
        attributes = cast(dict[str, Any], vars(value))
        return {
            key: json_value(item) for key, item in attributes.items() if not key.startswith("_")
        }
    return str(value)


def save_envelope(path: Path, envelope: dict[str, Any]) -> Path:
    """Write a private session file atomically."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    payload = json.dumps(redact_data(envelope), ensure_ascii=False, indent=2)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        try:
            path.chmod(0o600)
        except OSError:
            pass
        if path.parent.resolve() == sessions_dir().resolve():
            saved = sorted(
                path.parent.glob("*.json"),
                key=lambda candidate: candidate.stat().st_mtime,
                reverse=True,
            )
            for expired in saved[MAX_SAVED_SESSIONS:]:
                expired.unlink(missing_ok=True)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return path


def load_envelope(identifier: str | Path) -> tuple[Path, dict[str, Any]]:
    """Load data only; this function never interprets or executes pending calls."""
    path = session_path(identifier)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to load session {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Unsupported session format in {path}.")
    envelope = cast(dict[str, Any], value)
    if envelope.get("version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported session format in {path}.")
    if not isinstance(envelope.get("history"), list) or not isinstance(
        envelope.get("transcript"), list
    ):
        raise ValueError(f"Malformed session data in {path}.")
    return path, envelope


def timestamp() -> str:
    return datetime.now(UTC).isoformat()
