"""Phase 5 security, cancellation, and usage accounting tests."""

import asyncio
from collections.abc import Awaitable, Callable
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from openai import AsyncOpenAI

from agents590.calebjr.events import CollectingEventSink, EventCallback, discard_event
from agents590.calebjr.pricing import MODEL_PRICING, UsageLedger, UsageRecord
from agents590.calebjr.session import CalebJRSession
from agents590.calebjr.tools import Tools
from agents590.contracts import RunContext
from agents590.provider import UNCOpenAISettings


class FakeSettings:
    model = "gpt-6.1-terra"


class Responses:
    def __init__(self, responses: list[object]) -> None:
        self.items = responses

    async def create(self, **kwargs: object) -> object:
        return self.items.pop(0)


class Client:
    def __init__(self, responses: list[object]) -> None:
        self.responses = Responses(responses)

    async def close(self) -> None:
        return None


def response(*, output: list[object] | None = None, text: str = "", usage: object = None) -> object:
    return SimpleNamespace(output=output or [], output_text=text, usage=usage)


def call(name: str, arguments: str, call_id: str = "call-1") -> object:
    return SimpleNamespace(type="function_call", call_id=call_id, name=name, arguments=arguments)


def session(
    tmp_path: Path,
    responses: list[object],
    *,
    event_sink: EventCallback = discard_event,
    confirm_destructive: Callable[[str], Awaitable[bool]] | None = None,
) -> CalebJRSession:
    return CalebJRSession(
        settings=cast(UNCOpenAISettings, FakeSettings()),
        context=RunContext(cwd=tmp_path),
        client=cast(AsyncOpenAI, Client(responses)),
        event_sink=event_sink,
        confirm_destructive=confirm_destructive,
    )


@pytest.mark.parametrize(
    ("model", "rates"),
    [
        ("gpt-6-astra", ("10", "50", "1", "12.5", "25")),
        ("gpt-6.1-sol", ("2", "10", "0.1", "2.5", "5")),
        ("gpt-6.1-terra", ("1", "5", "0.05", "1.25", "2.5")),
        ("gpt-6-luna", ("0.1", "0.5", "0.01", "0.125", "0.25")),
        ("gpt-6-sol", ("2", "10", "0.2", "2.5", "5")),
        ("gpt-5.6-sol", ("4", "20", "0.4", "5", "10")),
        ("gpt-5.6-terra", ("2", "12", "0.2", "2.5", "5")),
        ("gpt-5.6-luna", ("0.2", "1.2", "0.02", "0.25", "0.5")),
        ("gpt-5.5", ("5", "30", "0.5", None, None)),
        ("gpt-5.5-pro", ("30", "180", None, None, None)),
        ("gpt-5.4", ("2.5", "15", "0.25", None, None)),
        ("gpt-5.4-pro", ("30", "180", None, None, None)),
    ],
)
def test_pricing_table_matches_supplied_rates(
    model: str, rates: tuple[str | None, str | None, str | None, str | None, str | None]
) -> None:
    pricing = MODEL_PRICING[model]
    actual = (
        pricing.input,
        pricing.output,
        pricing.cached_input,
        pricing.cache_write_30m,
        pricing.cache_write_12h,
    )
    assert actual == tuple(Decimal(rate) if rate is not None else None for rate in rates)


def test_exact_cache_cost_and_unknown_model() -> None:
    assert len(MODEL_PRICING) == 12
    record = UsageRecord(
        model="gpt-6.1-terra",
        input_tokens=1_000_000,
        output_tokens=100_000,
        cached_input_tokens=100_000,
        cache_write_30m_tokens=100_000,
        cache_write_12h_tokens=100_000,
        cache_write_details_available=True,
    )
    # 700k ordinary + 100k cached + both write classes + 100k output.
    assert record.cost() == Decimal("1.580")
    assert UsageRecord(model="unknown", input_tokens=1).cost() is None


def test_mixed_model_and_safety_costs_use_each_requests_rates() -> None:
    ledger = UsageLedger(
        [
            UsageRecord("gpt-6.1-terra", input_tokens=1_000_000),
            UsageRecord("gpt-5.6-luna", output_tokens=1_000_000, safety=True),
        ]
    )
    assert ledger.cost(safety=False) == Decimal("1")
    assert ledger.cost(safety=True) == Decimal("1.2")
    assert ledger.cost() == Decimal("2.2")


def test_usage_ledger_extracts_provider_details_and_status(tmp_path: Path) -> None:
    usage = SimpleNamespace(
        input_tokens=1_000,
        output_tokens=200,
        input_tokens_details=SimpleNamespace(
            cached_tokens=400,
            cache_write_30m_tokens=0,
            cache_write_12h_tokens=0,
        ),
    )
    active = session(tmp_path, [response(text="done", usage=usage)])

    assert asyncio.run(active.run_turn("answer")).text == "done"
    assert active.usage.input_tokens == 1_000
    assert active.usage.cached_input_tokens == 400
    assert active.usage.output_tokens == 200
    assert active.usage.cost() == Decimal("0.00162")
    assert "1,200 total" in active.status_summary()
    assert "~$0.002" in active.toolbar_summary()


def test_safety_usage_is_separate_and_confirmation_can_decline(tmp_path: Path) -> None:
    decisions: list[str] = []

    async def decline(command: str) -> bool:
        decisions.append(command)
        return False

    safety_usage = SimpleNamespace(input_tokens=100, output_tokens=1, input_tokens_details=None)
    active = session(
        tmp_path,
        [
            response(output=[call("shell", '{"command": "rm target.txt"}')]),
            response(text="APPROVE", usage=safety_usage),
            response(text="not executed"),
        ],
        confirm_destructive=decline,
    )

    assert asyncio.run(active.run_turn("remove target.txt")).text == "not executed"
    assert decisions == ["rm target.txt"]
    assert [event["type"] for event in active.event_log if "Confirmation" in event["type"]] == [
        "DestructiveConfirmationStarted",
        "DestructiveConfirmationFinished",
    ]
    assert len(active.usage.records) == 3
    assert [record.safety for record in active.usage.records] == [False, True, False]
    tool_output = next(item[1] for item in active.visible_transcript if item[0] == "tool")
    assert "confirmation declined" in tool_output


def test_approved_confirmation_executes_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executed: list[str] = []

    async def approve(command: str) -> bool:
        return True

    async def shell(command: str, *, cwd: str, timeout: float = 120.0) -> str:
        executed.append(command)
        return "removed"

    monkeypatch.setattr(Tools, "shell_async", shell)
    active = session(
        tmp_path,
        [
            response(output=[call("shell", '{"command": "rm target.txt"}')]),
            response(text="APPROVE"),
            response(text="done"),
        ],
        confirm_destructive=approve,
    )
    assert asyncio.run(active.run_turn("remove target.txt")).text == "done"
    assert executed == ["rm target.txt"]


def test_provider_error_and_malformed_calls_leave_session_usable(tmp_path: Path) -> None:
    class ErrorResponses(Responses):
        async def create(self, **kwargs: object) -> object:
            if self.items:
                item = self.items.pop(0)
                if isinstance(item, Exception):
                    raise item
                return item
            raise AssertionError("unexpected request")

    client = Client([])
    client.responses = ErrorResponses([RuntimeError("offline"), response(text="recovered")])
    active = CalebJRSession(
        settings=cast(UNCOpenAISettings, FakeSettings()),
        context=RunContext(cwd=tmp_path),
        client=cast(AsyncOpenAI, client),
    )
    assert "offline" in asyncio.run(active.run_turn("first")).text
    assert asyncio.run(active.run_turn("second")).text == "recovered"

    for malformed in (call("cat", "not-json"), call("unknown", "{}")):
        malformed_session = session(
            tmp_path,
            [response(output=[malformed]), response(text="handled")],
        )
        assert asyncio.run(malformed_session.run_turn("try it")).text == "handled"
        assert "Tool error:" in malformed_session.visible_transcript[-2][1]


def test_filesystem_tool_rejects_parent_and_symlink_escapes(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside-secret.txt"
    outside.write_text("secret")
    link = tmp_path / "escape"
    try:
        link.symlink_to(tmp_path.parent, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are unavailable")

    for requested in ("../outside-secret.txt", "escape/outside-secret.txt"):
        active = session(
            tmp_path,
            [
                response(output=[call("cat", f'{{"path": "{requested}"}}')]),
                response(text="blocked"),
            ],
        )
        assert asyncio.run(active.run_turn("read it")).text == "blocked"
        assert "escapes the working directory" in active.visible_transcript[-2][1]


@pytest.mark.parametrize("tool", ["move", "copy", "rename"])
def test_two_path_tools_reject_an_outside_endpoint(tmp_path: Path, tool: str) -> None:
    (tmp_path / "source.txt").write_text("safe")
    active = session(
        tmp_path,
        [
            response(output=[call(tool, '{"src": "source.txt", "dst": "../outside.txt"}')]),
            response(text="blocked"),
        ],
    )
    assert asyncio.run(active.run_turn("move it")).text == "blocked"
    assert "dst escapes the working directory" in active.visible_transcript[-2][1]
    assert (tmp_path / "source.txt").exists()


def test_cancellation_during_shell_rolls_back_and_session_remains_usable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = asyncio.Event()

    async def waiting_shell(command: str, *, cwd: str, timeout: float = 120.0) -> str:
        started.set()
        await asyncio.Event().wait()
        return "impossible"

    monkeypatch.setattr(Tools, "shell_async", waiting_shell)
    active = session(
        tmp_path,
        [
            response(output=[call("shell", '{"command": "sleep 10"}')]),
            response(text="recovered"),
        ],
        event_sink=CollectingEventSink(),
    )

    async def run() -> None:
        task = asyncio.create_task(active.run_turn("wait"))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not active.history[1:]
        assert (await active.run_turn("continue")).text == "recovered"

    asyncio.run(run())


def test_raw_context_is_redacted_and_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "raw-context-secret")
    active = session(tmp_path, [])
    active.history.append({"role": "user", "content": "raw-context-secret " + "x" * 110_000})
    raw = active.raw_context()
    assert "raw-context-secret" not in raw
    assert "[REDACTED]" in raw
    assert len(raw) < 101_000
    assert "characters omitted" in raw


def test_clear_retains_lifetime_usage(tmp_path: Path) -> None:
    active = session(tmp_path, [])
    active.usage = UsageLedger([UsageRecord("gpt-6.1-terra", input_tokens=10)])
    active.history.append({"role": "user", "content": "old"})
    active.clear()
    assert active.usage.input_tokens == 10
    assert len(active.history) == 1
