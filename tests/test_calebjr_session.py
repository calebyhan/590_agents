"""Exercise CalebJR's stateful core without contacting a provider."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import cast

from openai import AsyncOpenAI

from agents590.calebjr import CalebJR
from agents590.calebjr.events import CollectingEventSink, TurnFinished
from agents590.calebjr.repl import run_repl
from agents590.calebjr.session import CalebJRSession
from agents590.contracts import RunContext
from agents590.provider import UNCOpenAISettings


class FakeResponses:
    def __init__(self) -> None:
        self.inputs: list[list[object]] = []
        self.responses = [
            SimpleNamespace(
                output=[
                    SimpleNamespace(
                        type="function_call",
                        call_id="call-1",
                        name="cat",
                        arguments='{"path": "note.txt"}',
                    )
                ],
                output_text="",
            ),
            SimpleNamespace(
                output=[SimpleNamespace(type="message", text="first")],
                output_text="first answer",
            ),
            SimpleNamespace(
                output=[SimpleNamespace(type="message", text="second")],
                output_text="second answer",
            ),
        ]

    async def create(self, **kwargs: object) -> object:
        model_input = cast(list[object], kwargs["input"])
        self.inputs.append(list(model_input))
        return self.responses.pop(0)


class FakeClient:
    def __init__(self) -> None:
        self.responses = FakeResponses()
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class FakeSettings:
    model = "test-model"


class ClientCreatingSettings:
    model = "test-model"

    def __init__(self) -> None:
        self.client = FakeClient()
        self.timeouts: list[float] = []

    def create_async_client(self, *, timeout: float = 30.0) -> AsyncOpenAI:
        self.timeouts.append(timeout)
        return cast(AsyncOpenAI, self.client)


class FakeAgent(CalebJR):
    def __init__(self, settings: ClientCreatingSettings) -> None:
        self.settings = settings

    def provider_settings(self, context: RunContext) -> UNCOpenAISettings:
        return cast(UNCOpenAISettings, self.settings)


def test_two_turns_retain_model_and_tool_history(tmp_path: Path) -> None:
    (tmp_path / "note.txt").write_text("remember me")
    client = FakeClient()
    events = CollectingEventSink()
    session = CalebJRSession(
        settings=cast(UNCOpenAISettings, FakeSettings()),
        context=RunContext(cwd=tmp_path),
        event_sink=events,
        client=cast(AsyncOpenAI, client),
    )

    async def run() -> None:
        assert (await session.run_turn("read the note")).text == "first answer"
        assert (await session.run_turn("what did it say?")).text == "second answer"

    asyncio.run(run())

    final_input = client.responses.inputs[-1]
    assert {"role": "user", "content": "read the note"} in final_input
    assert {
        "type": "function_call_output",
        "call_id": "call-1",
        "output": "remember me",
    } in final_input
    assert {"role": "user", "content": "what did it say?"} in final_input
    assert session.user_turn_count == 2
    assert session.tool_call_count == 1
    assert [event.text for event in events.events if isinstance(event, TurnFinished)] == [
        "first answer",
        "second answer",
    ]


def test_one_shot_keeps_30_second_timeout(tmp_path: Path) -> None:
    (tmp_path / "note.txt").write_text("remember me")
    settings = ClientCreatingSettings()
    agent = FakeAgent(settings)

    result = asyncio.run(agent.run("read the note", RunContext(cwd=tmp_path)))

    assert result == "first answer"
    assert settings.timeouts == [30.0]
    assert settings.client.closed


def test_scripted_repl_reuses_session_and_selects_120_second_timeout(tmp_path: Path) -> None:
    (tmp_path / "note.txt").write_text("remember me")
    settings = ClientCreatingSettings()
    agent = FakeAgent(settings)
    lines = iter(["read the note", "what did it say?", "/context", "/exit"])
    output: list[str] = []

    result = asyncio.run(
        run_repl(
            agent,
            RunContext(cwd=tmp_path),
            input_fn=lambda prompt: next(lines),
            output_fn=output.append,
        )
    )

    assert result == 0
    assert settings.timeouts == [120.0]
    assert settings.client.closed
    assert "● Processing request." in output
    assert "first answer" in output
    assert "second answer" in output
    assert any("Session: 2 user turns, 1 tool events" in line for line in output)
    assert {"role": "user", "content": "read the note"} in settings.client.responses.inputs[-1]


def test_invalid_response_counter_resets_between_turns(tmp_path: Path) -> None:
    def invalid() -> SimpleNamespace:
        return SimpleNamespace(output=[], output_text="[cat](note.txt)")

    responses = FakeResponses()
    responses.responses = [
        *(invalid() for _ in range(CalebJRSession.MAX_INVALID)),
        SimpleNamespace(output=[], output_text="first answer"),
        invalid(),
        SimpleNamespace(output=[], output_text="second answer"),
    ]
    client = FakeClient()
    client.responses = responses
    session = CalebJRSession(
        settings=cast(UNCOpenAISettings, FakeSettings()),
        context=RunContext(cwd=tmp_path),
        client=cast(AsyncOpenAI, client),
    )

    async def run() -> None:
        assert (await session.run_turn("first")).text == "first answer"
        assert (await session.run_turn("second")).text == "second answer"

    asyncio.run(run())


def test_scripted_repl_eof_exits_and_closes_session(tmp_path: Path) -> None:
    settings = ClientCreatingSettings()
    agent = FakeAgent(settings)

    def eof(prompt: str) -> str:
        raise EOFError

    assert (
        asyncio.run(
            run_repl(
                agent,
                RunContext(cwd=tmp_path),
                input_fn=eof,
                output_fn=lambda text: None,
            )
        )
        == 0
    )
    assert settings.client.closed


def test_scripted_repl_cancellation_rolls_back_turn_and_continues(tmp_path: Path) -> None:
    class CancelThenAnswerResponses:
        def __init__(self) -> None:
            self.inputs: list[list[object]] = []
            self.cancel = True

        async def create(self, **kwargs: object) -> object:
            model_input = cast(list[object], kwargs["input"])
            self.inputs.append(list(model_input))
            if self.cancel:
                self.cancel = False
                raise asyncio.CancelledError
            return SimpleNamespace(output=[], output_text="recovered")

    settings = ClientCreatingSettings()
    responses = CancelThenAnswerResponses()
    settings.client.responses = cast(FakeResponses, responses)
    agent = FakeAgent(settings)
    lines = iter(["cancel this", "continue", "/exit"])
    output: list[str] = []

    assert (
        asyncio.run(
            run_repl(
                agent,
                RunContext(cwd=tmp_path),
                input_fn=lambda prompt: next(lines),
                output_fn=output.append,
            )
        )
        == 0
    )
    assert "Turn cancelled." in output
    assert "recovered" in output
    assert {"role": "user", "content": "cancel this"} not in responses.inputs[-1]
    assert {"role": "user", "content": "continue"} in responses.inputs[-1]
