"""Terminal renderers for CalebJR session events."""

import json
import os
import sys
from collections.abc import Callable
from typing import Literal, Protocol

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.status import Status
from rich.syntax import Syntax
from rich.text import Text

from agents590.calebjr.events import (
    ContextCompacted,
    DestructiveConfirmationFinished,
    DestructiveConfirmationStarted,
    ModelTextDelta,
    Progress,
    ReasoningSummary,
    SafetyReviewFinished,
    SafetyReviewStarted,
    SessionEvent,
    ToolFinished,
    ToolStarted,
    TurnFinished,
    TurnStarted,
)

RenderMode = Literal["quiet", "compact", "default", "verbose"]


class EventRenderer(Protocol):
    """Presentation-independent renderer contract used by the REPL."""

    async def __call__(self, event: SessionEvent) -> None: ...


class PlainRenderer:
    """Render clean text for redirected output, tests, and simple terminals."""

    COMPACT_OUTPUT_LIMIT = 500
    DEFAULT_OUTPUT_LIMIT = 2_000

    def __init__(
        self,
        output: Callable[[str], None] = print,
        *,
        write: Callable[[str], None] | None = None,
        mode: RenderMode = "default",
        color: bool | None = None,
    ) -> None:
        self.output = output
        self.write = write or output
        self.mode = mode
        self.color = (
            sys.stdout.isatty() and "NO_COLOR" not in os.environ if color is None else color
        )
        self._response_text = ""
        self._open_stream_line = False

    def _styled(self, text: str, code: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.color else text

    def _close_stream_line(self) -> None:
        if self._open_stream_line:
            self.write("\n")
            self._open_stream_line = False

    def line(self, text: str) -> None:
        self._close_stream_line()
        self.output(text)

    async def __call__(self, event: SessionEvent) -> None:
        if isinstance(event, TurnStarted):
            self._response_text = ""
            self._open_stream_line = False
        elif isinstance(event, ModelTextDelta):
            if self.mode != "quiet":
                self.write(event.text)
                self._response_text += event.text
                self._open_stream_line = True
        elif isinstance(event, Progress):
            if self.mode != "quiet":
                self.line(self._styled(f"● {event.text}", "36"))
        elif isinstance(event, ReasoningSummary):
            if self.mode != "quiet":
                self.line(self._styled(f"Reasoning summary: {event.text}", "2"))
        elif isinstance(event, ContextCompacted):
            if self.mode != "quiet":
                self.line(
                    f"Context compacted: {event.summarized_items} items; "
                    f"{event.characters_before:,} → {event.characters_after:,} characters."
                )
        elif isinstance(event, ToolStarted):
            self._close_stream_line()
            self._response_text = ""
            if self.mode != "quiet":
                detail = "" if self.mode == "compact" else f": {event.arguments}"
                self.line(self._styled(f"→ {event.name}{detail}", "36"))
        elif isinstance(event, ToolFinished):
            if self.mode != "quiet":
                text = event.output
                limit = (
                    self.COMPACT_OUTPUT_LIMIT
                    if self.mode == "compact"
                    else self.DEFAULT_OUTPUT_LIMIT
                )
                if self.mode != "verbose" and len(text) > limit:
                    omitted = len(text) - limit
                    text = f"{text[:limit]}\n… ({omitted:,} characters omitted)"
                marker = "!" if event.failed else " "
                self.line(f"{marker} {text}")
        elif isinstance(event, SafetyReviewStarted):
            if self.mode != "quiet":
                self.line(f"? safety review: {event.command}")
        elif isinstance(event, SafetyReviewFinished):
            if self.mode != "quiet" or not event.approved:
                message = "✓ safety review approved" if event.approved else f"! {event.decision}"
                self.line(message)
        elif isinstance(event, DestructiveConfirmationStarted):
            self.line(f"? confirm destructive command:\n{event.command}")
        elif isinstance(event, DestructiveConfirmationFinished):
            if not event.approved:
                self.line("! destructive command declined")
        elif isinstance(event, TurnFinished):
            if self.mode == "quiet" or self._response_text != event.text:
                self.line(event.text)
            else:
                self._close_stream_line()
            self._response_text = ""
        else:
            self.line(event.error)


class RichRenderer:
    """Render readable Markdown and structured activity on an interactive terminal."""

    DEFAULT_OUTPUT_LIMIT = 2_000
    COMPACT_OUTPUT_LIMIT = 160

    def __init__(self, console: Console, *, mode: RenderMode = "default") -> None:
        self.console = console
        self.mode = mode
        self._live: Live | None = None
        self._status: Status | None = None
        self._stream_segment = ""
        self._response_text = ""

    def _stop_status(self) -> None:
        if self._status is not None:
            self._status.stop()
            self._status = None

    def _start_status(self, message: str) -> None:
        self._stop_status()
        self._status = self.console.status(message, spinner="dots")
        self._status.start()

    def _stop_live(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None
            self._stream_segment = ""

    def _print(self, renderable: RenderableType) -> None:
        self._stop_status()
        self._stop_live()
        self.console.print(renderable)

    def _stream(self, text: str) -> None:
        self._stop_status()
        self._stream_segment += text
        self._response_text += text
        markdown = Markdown(self._stream_segment)
        if self._live is None:
            self._live = Live(markdown, console=self.console, refresh_per_second=12)
            self._live.start()
        else:
            self._live.update(markdown, refresh=True)

    @staticmethod
    def _tool_output(output: str, name: str) -> RenderableType:
        lexer = "text"
        if name == "shell":
            lexer = "console"
        elif output.lstrip().startswith(("{", "[")):
            lexer = "json"
        elif "diff --git " in output or output.startswith(("--- ", "+++ ")):
            lexer = "diff"
        return Syntax(output, lexer, word_wrap=True, background_color="default")

    def _bounded(self, text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        return f"{text[:limit]}\n… ({len(text) - limit:,} characters omitted)"

    async def __call__(self, event: SessionEvent) -> None:
        if isinstance(event, TurnStarted):
            self._stop_status()
            self._stop_live()
            self._response_text = ""
        elif isinstance(event, ModelTextDelta):
            if self.mode != "quiet":
                self._stream(event.text)
        elif isinstance(event, Progress):
            if self.mode != "quiet":
                self._stop_live()
                self._start_status(event.text)
        elif isinstance(event, ReasoningSummary):
            if self.mode != "quiet":
                self._print(Panel(event.text, title="Reasoning summary", style="dim", expand=False))
        elif isinstance(event, ContextCompacted):
            if self.mode != "quiet":
                self._print(
                    Text(
                        f"Context compacted: {event.summarized_items} items; "
                        f"{event.characters_before:,} → {event.characters_after:,} characters.",
                        style="yellow",
                    )
                )
        elif isinstance(event, ToolStarted):
            self._stop_status()
            self._stop_live()
            self._response_text = ""
            if self.mode == "quiet":
                return
            heading = Text.assemble(("→ ", "cyan bold"), (event.name, "bold"))
            if self.mode == "compact":
                self.console.print(heading)
            else:
                arguments = json.dumps(event.arguments, indent=2, ensure_ascii=False)
                syntax = Syntax(arguments, "json", background_color="default")
                self.console.print(Group(heading, syntax))
            self._start_status(f"Running {event.name}…")
        elif isinstance(event, ToolFinished):
            if self.mode == "quiet":
                return
            status = "failed" if event.failed else "done"
            style = "red" if event.failed else "green"
            if self.mode == "compact":
                output = " ".join(event.output.splitlines())
                if len(output) > self.COMPACT_OUTPUT_LIMIT:
                    omitted = len(output) - self.COMPACT_OUTPUT_LIMIT
                    output = f"{output[: self.COMPACT_OUTPUT_LIMIT]} … ({omitted:,} omitted)"
                self._print(
                    Text.assemble(
                        ("! " if event.failed else "✓ ", f"{style} bold"),
                        (event.name, "bold"),
                        (f" · {event.duration_seconds:.2f}s · ", "dim"),
                        output,
                    )
                )
                return
            output = event.output
            if self.mode != "verbose":
                output = self._bounded(output, self.DEFAULT_OUTPUT_LIMIT)
            title = f"{event.name} · {status} · {event.duration_seconds:.2f}s"
            content = self._tool_output(output, event.name)
            self._print(Panel(content, title=title, border_style=style))
        elif isinstance(event, SafetyReviewStarted):
            if self.mode != "quiet":
                self._print(Panel(event.command, title="Safety review", border_style="yellow"))
        elif isinstance(event, SafetyReviewFinished):
            if self.mode != "quiet" or not event.approved:
                style = "green" if event.approved else "red"
                marker = "✓ Approved" if event.approved else "! Rejected"
                self._print(Text(f"{marker}: {event.decision}", style=style))
        elif isinstance(event, DestructiveConfirmationStarted):
            self._print(
                Panel(event.command, title="Confirm destructive command", border_style="red")
            )
        elif isinstance(event, DestructiveConfirmationFinished):
            if not event.approved:
                self._print(Text("! Destructive command declined", style="red"))
        elif isinstance(event, TurnFinished):
            self._stop_status()
            self._stop_live()
            if self.mode == "quiet" or self._response_text != event.text:
                self.console.print(Markdown(event.text))
            self._response_text = ""
        else:
            self._print(Panel(event.error, title="Turn failed", border_style="red"))


def terminal_renderer(*, mode: RenderMode = "default") -> RichRenderer | PlainRenderer:
    """Select rich output only when stdout is an interactive capable terminal."""
    if sys.stdout.isatty() and os.environ.get("TERM") != "dumb":
        console = Console(no_color="NO_COLOR" in os.environ)
        return RichRenderer(console, mode=mode)
    return PlainRenderer(mode=mode, color=False)
