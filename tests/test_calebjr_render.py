"""Check rich terminal presentation independently from the session engine."""

import asyncio
from io import StringIO

from rich.console import Console

from agents590.calebjr.events import (
    ModelTextDelta,
    Progress,
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

    asyncio.run(render_compact())
    compact_text = compact_output.getvalue()
    assert "command" not in compact_text
    assert "first second" in compact_text
    assert "shell · 0.10s" in compact_text

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
