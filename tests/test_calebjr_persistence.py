"""Context compaction and persistence remain local, bounded, and inert on load."""

import asyncio
import json
from pathlib import Path
from typing import cast

import pytest
from openai import AsyncOpenAI

from agents590.calebjr.events import CollectingEventSink, ContextCompacted
from agents590.calebjr.session import CalebJRSession
from agents590.calebjr.transcript import SCHEMA_VERSION, save_envelope
from agents590.contracts import RunContext
from agents590.provider import UNCOpenAISettings


class FakeClient:
    async def close(self) -> None:
        return None


class FakeSettings:
    model = "test-model"


def make_session(tmp_path: Path, events: CollectingEventSink | None = None) -> CalebJRSession:
    return CalebJRSession(
        settings=cast(UNCOpenAISettings, FakeSettings()),
        context=RunContext(cwd=tmp_path),
        event_sink=events or CollectingEventSink(),
        client=cast(AsyncOpenAI, FakeClient()),
    )


def test_compaction_retains_recent_turns_and_emits_event(tmp_path: Path) -> None:
    events = CollectingEventSink()
    session = make_session(tmp_path, events)
    session.history.extend(
        [
            {"role": "user", "content": "old goal"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "recent one"},
            {"role": "assistant", "content": "answer one"},
            {"role": "user", "content": "recent two"},
            {"role": "assistant", "content": "answer two"},
        ]
    )
    session.visible_transcript.extend(
        [
            ("user", "old goal"),
            ("assistant", "old answer"),
            ("user", "recent one"),
            ("assistant", "answer one"),
            ("user", "recent two"),
            ("assistant", "answer two"),
        ]
    )

    assert asyncio.run(session.compact())

    assert "[Context compaction summary]" in str(session.history[1])
    assert {"role": "user", "content": "recent one"} in session.history
    assert {"role": "user", "content": "recent two"} in session.history
    compacted = [event for event in events.events if isinstance(event, ContextCompacted)]
    assert len(compacted) == 1
    assert compacted[0].summarized_items == 2
    assert session.compaction_count == 1


def test_save_is_redacted_private_and_resumes_without_actions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("OPENAI_API_KEY", "super-secret-value")
    session = make_session(tmp_path)
    session.history.extend(
        [
            {"role": "user", "content": "token=abc123456789012345"},
            {"role": "assistant", "content": "super-secret-value"},
        ]
    )
    session.visible_transcript.extend(
        [("user", "token=abc123456789012345"), ("assistant", "super-secret-value")]
    )
    session.user_turn_count = session.assistant_turn_count = 1

    path = session.save("redacted-test")
    payload = path.read_text(encoding="utf-8")

    assert path.name == "redacted-test.json"
    assert "abc123456789012345" not in payload
    assert "super-secret-value" not in payload
    assert "[REDACTED]" in payload
    assert path.stat().st_mode & 0o077 == 0

    resumed = CalebJRSession.resume(
        "redacted-test",
        settings=cast(UNCOpenAISettings, FakeSettings()),
        client=cast(AsyncOpenAI, FakeClient()),
    )
    assert resumed.user_turn_count == 1
    assert resumed.model == "test-model"
    assert resumed.cwd == tmp_path
    assert resumed.history[-1]["content"] == "[REDACTED]"


def test_resume_rejects_pending_call_before_creating_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    path = tmp_path / "pending.json"
    save_envelope(
        path,
        {
            "version": SCHEMA_VERSION,
            "session_id": "pending",
            "cwd": str(tmp_path),
            "model": "test-model",
            "history": [
                {"role": "system", "content": "system"},
                {
                    "type": "function_call",
                    "call_id": "never-run",
                    "name": "shell",
                    "arguments": json.dumps({"command": "echo unsafe"}),
                },
            ],
            "transcript": [],
        },
    )

    class CountingSettings(FakeSettings):
        calls = 0

        def create_async_client(self, *, timeout: float = 30.0) -> AsyncOpenAI:
            self.calls += 1
            return cast(AsyncOpenAI, FakeClient())

    settings = CountingSettings()
    with pytest.raises(ValueError, match="pending tool call"):
        CalebJRSession.resume(path, settings=cast(UNCOpenAISettings, settings))
    assert settings.calls == 0
