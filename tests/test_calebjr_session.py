"""Exercise CalebJR's stateful core without contacting a provider."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import cast

from openai import AsyncOpenAI

from agents590.calebjr import CalebJR
from agents590.calebjr.events import (
    CollectingEventSink,
    ModelTextDelta,
    Progress,
    ReasoningSummary,
    ToolFinished,
    ToolStarted,
    TurnFinished,
)
from agents590.calebjr.repl import (
    INTERACTIVE_MAX_INVALID_RESPONSES,
    INTERACTIVE_MAX_REPEATED_CALLS,
    INTERACTIVE_MAX_STEPS,
    INTERACTIVE_MAX_TOOL_FAILURES,
    PlainRenderer,
    run_repl,
)
from agents590.calebjr.session import CalebJRSession
from agents590.calebjr.transcript import SCHEMA_VERSION, save_envelope
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
        self.sessions: list[CalebJRSession] = []

    def provider_settings(self, context: RunContext) -> UNCOpenAISettings:
        return cast(UNCOpenAISettings, self.settings)

    def create_session(
        self,
        context: RunContext,
        *,
        request_timeout: float = 30.0,
        max_steps: int = CalebJRSession.MAX_STEPS,
        max_invalid_responses: int = CalebJRSession.MAX_INVALID,
        max_tool_failures: int = CalebJRSession.MAX_TOOL_FAILURES,
        max_repeated_calls: int = CalebJRSession.MAX_REPEATED_CALLS,
    ) -> CalebJRSession:
        session = super().create_session(
            context,
            request_timeout=request_timeout,
            max_steps=max_steps,
            max_invalid_responses=max_invalid_responses,
            max_tool_failures=max_tool_failures,
            max_repeated_calls=max_repeated_calls,
        )
        self.sessions.append(session)
        return session


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


def test_function_call_history_uses_request_fields_only(tmp_path: Path) -> None:
    response = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="function_call",
                call_id="call-1",
                name="cat",
                arguments='{"path": "note.txt"}',
                parsed_arguments={"path": "note.txt"},
            )
        ],
        output_text="",
    )
    client = FakeClient()
    client.responses.responses = [response, SimpleNamespace(output=[], output_text="done")]
    session = CalebJRSession(
        settings=cast(UNCOpenAISettings, FakeSettings()),
        context=RunContext(cwd=tmp_path),
        client=cast(AsyncOpenAI, client),
    )

    assert asyncio.run(session.run_turn("read the note")).text == "done"
    function_call = next(
        item
        for item in client.responses.inputs[-1]
        if isinstance(item, dict) and item.get("type") == "function_call"
    )
    assert function_call == {
        "type": "function_call",
        "call_id": "call-1",
        "name": "cat",
        "arguments": '{"path": "note.txt"}',
    }


def test_one_shot_keeps_30_second_timeout(tmp_path: Path) -> None:
    (tmp_path / "note.txt").write_text("remember me")
    settings = ClientCreatingSettings()
    agent = FakeAgent(settings)

    result = asyncio.run(agent.run("read the note", RunContext(cwd=tmp_path)))

    assert result == "first answer"
    assert settings.timeouts == [30.0]
    assert settings.client.closed
    assert agent.sessions[0].max_steps == CalebJRSession.MAX_STEPS
    assert agent.sessions[0].max_invalid_responses == CalebJRSession.MAX_INVALID


def test_scripted_repl_reuses_session_and_selects_120_second_timeout(tmp_path: Path) -> None:
    (tmp_path / "note.txt").write_text("remember me")
    settings = ClientCreatingSettings()
    agent = FakeAgent(settings)
    lines = iter(
        [
            "read the note",
            "what did it say?",
            "/context",
            "/activity",
            "/activity show 1",
            "/activity output A1",
            "/transcript markdown",
            "/activity detailed",
            "/status",
            "/exit",
        ]
    )
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
    assert agent.sessions[0].max_steps == INTERACTIVE_MAX_STEPS
    assert agent.sessions[0].max_invalid_responses == INTERACTIVE_MAX_INVALID_RESPONSES
    assert agent.sessions[0].max_tool_failures == INTERACTIVE_MAX_TOOL_FAILURES
    assert agent.sessions[0].max_repeated_calls == INTERACTIVE_MAX_REPEATED_CALLS
    assert "● Processing request." in output
    assert "first answer" in output
    assert "second answer" in output
    assert any("Session: 2 user turns, 1 tool events" in line for line in output)
    assert any("Destructive confirmation: disabled" in line for line in output)
    assert "Activity mode: compact" in output
    assert any("Read note.txt" in line for line in output)
    assert any("A1 · Read note.txt" in line and "remember me" in line for line in output)
    assert any("## User" in line and "## Tool output" in line for line in output)
    assert "Activity mode changed to detailed." in output
    assert any("Activity mode: detailed" in line for line in output)
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


def test_repl_requires_confirmation_for_external_resume(tmp_path: Path) -> None:
    external = tmp_path / "external.json"
    save_envelope(
        external,
        {
            "version": SCHEMA_VERSION,
            "session_id": "external",
            "cwd": str(tmp_path),
            "model": "test-model",
            "history": [{"role": "system", "content": "system"}],
            "transcript": [],
        },
    )
    settings = ClientCreatingSettings()
    agent = FakeAgent(settings)
    lines = iter(["yes", "/exit"])
    output: list[str] = []
    assert (
        asyncio.run(
            run_repl(
                agent,
                RunContext(cwd=tmp_path),
                input_fn=lambda prompt: next(lines),
                output_fn=output.append,
                resume=external,
            )
        )
        == 0
    )
    assert any("outside the CalebJR data directory" in line for line in output)
    assert settings.client.closed


def test_scripted_repl_second_ctrl_c_exits(tmp_path: Path) -> None:
    settings = ClientCreatingSettings()
    agent = FakeAgent(settings)
    calls = 0
    output: list[str] = []

    def interrupt(prompt: str) -> str:
        nonlocal calls
        calls += 1
        raise KeyboardInterrupt

    assert (
        asyncio.run(
            run_repl(
                agent,
                RunContext(cwd=tmp_path),
                input_fn=interrupt,
                output_fn=output.append,
            )
        )
        == 0
    )
    assert calls == 2
    assert any("again to quit" in line for line in output)
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


def test_streaming_emits_text_and_provider_reasoning_summary(tmp_path: Path) -> None:
    final_response = SimpleNamespace(
        output=[SimpleNamespace(type="message", text="hello")],
        output_text="hello",
    )

    class Stream:
        async def __aenter__(self) -> "Stream":
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        def __aiter__(self) -> "Stream":
            self.events = iter(
                [
                    SimpleNamespace(type="response.reasoning_summary_text.delta", delta="Checked."),
                    SimpleNamespace(type="response.output_text.delta", delta="hel"),
                    SimpleNamespace(type="response.output_text.delta", delta="lo"),
                ]
            )
            return self

        async def __anext__(self) -> object:
            try:
                return next(self.events)
            except StopIteration:
                raise StopAsyncIteration from None

        async def get_final_response(self) -> object:
            return final_response

    class Responses:
        def stream(self, **kwargs: object) -> Stream:
            return Stream()

        async def create(self, **kwargs: object) -> object:
            raise AssertionError("non-streaming fallback should not be used")

    client = FakeClient()
    client.responses = cast(FakeResponses, Responses())
    events = CollectingEventSink()
    session = CalebJRSession(
        settings=cast(UNCOpenAISettings, FakeSettings()),
        context=RunContext(cwd=tmp_path),
        event_sink=events,
        client=cast(AsyncOpenAI, client),
    )

    assert asyncio.run(session.run_turn("say hello")).text == "hello"
    assert [event.text for event in events.events if isinstance(event, ModelTextDelta)] == [
        "hel",
        "lo",
    ]
    assert [event.text for event in events.events if isinstance(event, ReasoningSummary)] == [
        "Checked."
    ]


def test_streaming_capability_failure_falls_back(tmp_path: Path) -> None:
    class UnsupportedStream:
        async def __aenter__(self) -> "UnsupportedStream":
            raise NotImplementedError("streaming disabled")

        async def __aexit__(self, *args: object) -> None:
            return None

    class Responses:
        def stream(self, **kwargs: object) -> UnsupportedStream:
            return UnsupportedStream()

        async def create(self, **kwargs: object) -> object:
            return SimpleNamespace(output=[], output_text="fallback answer")

    client = FakeClient()
    client.responses = cast(FakeResponses, Responses())
    events = CollectingEventSink()
    session = CalebJRSession(
        settings=cast(UNCOpenAISettings, FakeSettings()),
        context=RunContext(cwd=tmp_path),
        event_sink=events,
        client=cast(AsyncOpenAI, client),
    )

    assert asyncio.run(session.run_turn("answer")).text == "fallback answer"
    assert any(
        isinstance(event, Progress) and "Streaming unavailable" in event.text
        for event in events.events
    )


def test_nonstreaming_surfaces_only_explicit_reasoning_summaries(tmp_path: Path) -> None:
    response = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="reasoning",
                summary=[SimpleNamespace(text="Provider summary.")],
                content="private reasoning must not be emitted",
            ),
            SimpleNamespace(type="message", text="answer"),
        ],
        output_text="answer",
    )
    client = FakeClient()
    client.responses.responses = [response]
    events = CollectingEventSink()
    session = CalebJRSession(
        settings=cast(UNCOpenAISettings, FakeSettings()),
        context=RunContext(cwd=tmp_path),
        event_sink=events,
        client=cast(AsyncOpenAI, client),
    )

    assert asyncio.run(session.run_turn("answer")).text == "answer"
    assert [event.text for event in events.events if isinstance(event, ReasoningSummary)] == [
        "Provider summary."
    ]
    assert all("private reasoning" not in str(event) for event in events.events)


def test_renderer_streams_clean_text_and_honors_modes() -> None:
    lines: list[str] = []
    writes: list[str] = []
    renderer = PlainRenderer(lines.append, write=writes.append, color=False)

    async def render() -> None:
        await renderer(ModelTextDelta("hel"))
        await renderer(ModelTextDelta("lo"))
        await renderer(TurnFinished("hello"))
        await renderer(ToolFinished("1", "cat", "x" * 2_001, False, 0.1))

    asyncio.run(render())
    assert writes[:3] == ["hel", "lo", "\n"]
    assert lines == [f"  {'x' * 2_000}\n… (1 characters omitted)"]
    assert not any("\033[" in text for text in [*lines, *writes])

    quiet_lines: list[str] = []
    quiet = PlainRenderer(quiet_lines.append, mode="quiet", color=False)

    async def render_quiet() -> None:
        await quiet(Progress("working"))
        await quiet(ToolFinished("1", "cat", "result", False, 0.1))
        await quiet(TurnFinished("answer"))

    asyncio.run(render_quiet())
    assert quiet_lines == ["answer"]

    compact_lines: list[str] = []
    compact = PlainRenderer(compact_lines.append, mode="compact", color=False)

    async def render_compact() -> None:
        await compact(ToolStarted("1", "cat", {"path": "secret.txt"}))
        await compact(ToolFinished("1", "cat", "x" * 501, False, 0.1))
        await compact(TurnFinished("answer"))

    asyncio.run(render_compact())
    assert compact_lines == ["✓ 1 action · 0.10s · inspected 1", "answer"]

    verbose_lines: list[str] = []
    verbose = PlainRenderer(verbose_lines.append, mode="verbose", color=False)
    asyncio.run(verbose(ToolFinished("1", "cat", "x" * 2_001, False, 0.1)))
    assert verbose_lines == [f"  {'x' * 2_001}"]

    colored_lines: list[str] = []
    colored = PlainRenderer(colored_lines.append, mode="default", color=True)
    asyncio.run(colored(ReasoningSummary("Explicit provider summary.")))
    assert colored_lines == ["\033[2mReasoning summary: Explicit provider summary.\033[0m"]
