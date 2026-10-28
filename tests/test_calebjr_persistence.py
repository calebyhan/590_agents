"""Context compaction and persistence remain local, bounded, and inert on load."""

import asyncio
import json
from pathlib import Path
from typing import cast

import pytest
from openai import AsyncOpenAI

from agents590.calebjr.events import CollectingEventSink, ContextCompacted
from agents590.calebjr.pricing import UsageRecord
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
    assert "User goals and constraints" in str(session.history[1])
    assert "Important results and errors" in str(session.history[1])
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
        CalebJRSession.resume(path, settings=cast(UNCOpenAISettings, settings), allow_external=True)
    assert settings.calls == 0


def test_usage_survives_save_and_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    session = make_session(tmp_path)
    session.usage.records.append(UsageRecord("gpt-6.1-terra", input_tokens=100, output_tokens=20))
    path = session.save("usage")

    resumed = CalebJRSession.resume(
        path,
        settings=cast(UNCOpenAISettings, FakeSettings()),
        client=cast(AsyncOpenAI, FakeClient()),
    )
    assert resumed.usage.input_tokens == 100
    assert resumed.usage.output_tokens == 20


def test_large_outputs_are_saved_as_private_redacted_hashed_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("OPENAI_API_KEY", "artifact-secret")
    session = make_session(tmp_path)
    output = "x" * 21_000 + " artifact-secret"
    session.history.append({"type": "function_call_output", "call_id": "large", "output": output})
    session.visible_transcript.append(("tool", output))

    path = session.save("artifacts")
    artifacts = list(path.with_suffix(".artifacts").glob("*.txt"))
    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert len(artifact.stem) == 64
    assert "artifact-secret" not in artifact.read_text(encoding="utf-8")
    assert "[REDACTED]" in artifact.read_text(encoding="utf-8")
    assert artifact.stat().st_mode & 0o077 == 0
    payload = path.read_text(encoding="utf-8")
    assert artifact.name in payload
    assert "sha256=" in payload


def test_external_resume_requires_explicit_trust(tmp_path: Path) -> None:
    path = tmp_path / "external.json"
    save_envelope(
        path,
        {
            "version": SCHEMA_VERSION,
            "cwd": str(tmp_path),
            "model": "test-model",
            "history": [{"role": "system", "content": "system"}],
            "transcript": [],
        },
    )
    with pytest.raises(ValueError, match="explicit trust"):
        CalebJRSession.resume(
            path,
            settings=cast(UNCOpenAISettings, FakeSettings()),
            client=cast(AsyncOpenAI, FakeClient()),
        )


def test_resume_rejects_missing_cwd_and_incompatible_version(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    save_envelope(
        missing,
        {
            "version": SCHEMA_VERSION,
            "cwd": str(tmp_path / "gone"),
            "model": "test-model",
            "history": [{"role": "system", "content": "system"}],
            "transcript": [],
        },
    )
    with pytest.raises(ValueError, match="no longer exists"):
        CalebJRSession.resume(
            missing,
            settings=cast(UNCOpenAISettings, FakeSettings()),
            client=cast(AsyncOpenAI, FakeClient()),
            allow_external=True,
        )

    incompatible = tmp_path / "future.json"
    incompatible.write_text(
        json.dumps({"version": SCHEMA_VERSION + 1, "history": [], "transcript": []})
    )
    with pytest.raises(ValueError, match="Unsupported session format"):
        CalebJRSession.resume(
            incompatible,
            settings=cast(UNCOpenAISettings, FakeSettings()),
            client=cast(AsyncOpenAI, FakeClient()),
            allow_external=True,
        )
