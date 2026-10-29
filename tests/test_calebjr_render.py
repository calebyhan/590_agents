"""Check rich terminal presentation independently from the session engine."""

import asyncio
from io import StringIO

from rich.console import Console

from agents590.calebjr.events import (
    DestructiveConfirmationFinished,
    DestructiveConfirmationStarted,
    ModelTextDelta,
    Progress,
    ReasoningSummary,
    ToolFinished,
    ToolStarted,
    TurnFinished,
    TurnStarted,
)
from agents590.calebjr.render import RichRenderer


def test_rich_renderer_formats_streamed_markdown_and_tool_activity() -> None:
    output = StringIO()
    console = Console(file=output, force_terminal=False, width=80)
    renderer = RichRenderer(console)

    async def render() -> None:
        await renderer(TurnStarted("test"))
        await renderer(ModelTextDelta("**Ready**"))
        await renderer(TurnFinished("**Ready**"))
        await renderer(ToolStarted("1", "cat", {"path": "example.py"}))
        await renderer(ToolFinished("1", "cat", "print('ok')", False, 0.125))

    asyncio.run(render())

    text = output.getvalue()
    assert "Ready" in text
    assert "→ cat" in text
    assert '"path": "example.py"' in text
    assert "cat · done · 0.12s" in text
    assert "print('ok')" in text
    assert "\033[" not in text


def test_rich_renderer_modes_control_detail() -> None:
    compact_output = StringIO()
    compact = RichRenderer(
        Console(file=compact_output, force_terminal=False, width=200), mode="compact"
    )

    async def render_compact() -> None:
        await compact(ToolStarted("1", "shell", {"command": "pytest"}))
        await compact(ToolFinished("1", "shell", "first\nsecond", False, 0.1))
        await compact(TurnFinished("done"))

    asyncio.run(render_compact())
    compact_text = compact_output.getvalue()
    assert "command" not in compact_text
    assert "first second" not in compact_text
    assert "1 action · 0.10s · ran 1 check" in compact_text

    quiet_output = StringIO()
    quiet = RichRenderer(Console(file=quiet_output, force_terminal=False), mode="quiet")

    async def render_quiet() -> None:
        await quiet(Progress("working"))
        await quiet(ToolFinished("1", "shell", "hidden", False, 0.1))
        await quiet(TurnFinished("**answer**"))

    asyncio.run(render_quiet())
    quiet_text = quiet_output.getvalue()
    assert "working" not in quiet_text
    assert "hidden" not in quiet_text
    assert "answer" in quiet_text


def test_compact_reasoning_is_durable_deduplicated_and_bounded() -> None:
    output = StringIO()
    renderer = RichRenderer(Console(file=output, force_terminal=False, width=100), mode="compact")

    async def render() -> None:
        await renderer(TurnStarted("test"))
        await renderer(ReasoningSummary("Inspecting the project structure."))
        await renderer(ReasoningSummary("Inspecting the project structure."))
        for index in range(1, 5):
            await renderer(ReasoningSummary(f"Reasoning update {index}."))

    asyncio.run(render())

    text = output.getvalue()
    assert text.count("Inspecting the project structure.") == 1
    assert "Reasoning update 3." in text
    assert "Reasoning update 4." not in text
    assert text.count("Additional reasoning updates omitted.") == 1


def test_reduced_motion_uses_static_semantic_updates_in_narrow_terminal() -> None:
    output = StringIO()
    console = Console(file=output, force_terminal=False, width=24)
    renderer = RichRenderer(console, mode="compact", reduced_motion=True)

    async def render() -> None:
        await renderer(TurnStarted("test"))
        await renderer(ToolStarted("1", "cat", {"path": "a/very/long/path/to/a/source-file.py"}))
        await renderer(ToolFinished("1", "cat", "ok", False, 0.1))
        console.width = 12
        await renderer(TurnFinished("done"))

    asyncio.run(render())

    text = output.getvalue()
    assert "Inspecting project" in text
    assert "Latest:" not in text
    assert "1 action" in text
    assert "\033[" not in text


def test_rich_renderer_handles_resize_and_confirmation_events() -> None:
    output = StringIO()
    console = Console(file=output, force_terminal=False, width=80)
    renderer = RichRenderer(console)

    async def render() -> None:
        await renderer(DestructiveConfirmationStarted("rm a-very-long-file-name.txt"))
        console.width = 20
        await renderer(DestructiveConfirmationFinished("rm a-very-long-file-name.txt", False))
        console.width = 120
        await renderer(TurnFinished("session remains usable"))

    asyncio.run(render())
    text = output.getvalue()
    assert "Confirm destructive" in text
    assert "declined" in text
    assert "session remains usable" in text
