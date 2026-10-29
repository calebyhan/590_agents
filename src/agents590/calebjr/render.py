"""Terminal renderers for CalebJR session events."""

import json
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic
from typing import Any, Literal, Protocol

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.status import Status
from rich.syntax import Syntax
from rich.text import Text

from agents590.calebjr.activity import ActivityAggregator, tool_status
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
    TurnCancelled,
    TurnFinished,
    TurnStarted,
)

RenderMode = Literal["quiet", "compact", "default", "verbose"]


class _ReasoningClumps:
    """Bound and de-duplicate provider-supplied reasoning summaries per turn."""

    def __init__(self, *, limit: int = 4, character_limit: int = 320) -> None:
        self.limit = limit
        self.character_limit = character_limit
        self._seen: list[str] = []
        self._omission_shown = False

    def reset(self) -> None:
        self._seen = []
        self._omission_shown = False

    def accept(self, text: str) -> str | None:
        text = text.strip()
        if not text or (self._seen and self._seen[-1] == text):
            return None
        if len(self._seen) >= self.limit:
            if self._omission_shown:
                return None
            self._omission_shown = True
            return "Additional reasoning updates omitted."
        self._seen.append(text)
        if len(text) <= self.character_limit:
            return text
        omitted = len(text) - self.character_limit
        return f"{text[: self.character_limit]}… ({omitted:,} characters omitted)"


@dataclass
class _ElapsedStatus:
    """A Rich status message whose elapsed time updates on spinner refreshes."""

    message: str
    started_at: float

    def __rich__(self) -> Text:
        elapsed = monotonic() - self.started_at
        return Text.assemble((self.message, "cyan"), (f" · {elapsed:.1f}s", "dim"))


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
        self.activity = ActivityAggregator()
        self._reasoning = _ReasoningClumps()

    def set_mode(self, mode: RenderMode) -> None:
        """Change presentation detail between turns."""
        self.mode = mode

    def activity_report(self, limit: int = 1) -> str:
        """Return recent tool activity at user request."""
        return self.activity.report(limit)

    def activity_group(self, identifier: str) -> str:
        return self.activity.group(identifier)

    def activity_output(self, identifier: str, *, full: bool = False) -> str:
        return self.activity.output(identifier, full=full)

    def restore_activity(self, events: list[dict[str, Any]]) -> None:
        self.activity.restore(events)

    def _flush_activity(self) -> None:
        summary = self.activity.complete()
        if summary is None:
            return
        marker = "!" if summary.failed_count else "✓"
        self.line(f"{marker} {summary.text}")

    def _styled(self, text: str, code: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.color else text

    @staticmethod
    def _bounded(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        return f"{text[:limit]}… ({len(text) - limit:,} characters omitted)"

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
            self._reasoning.reset()
        elif isinstance(event, ModelTextDelta):
            if self.mode == "compact":
                self._flush_activity()
            if self.mode != "quiet":
                self.write(event.text)
                self._response_text += event.text
                self._open_stream_line = True
        elif isinstance(event, Progress):
            if self.mode != "quiet":
                self.line(self._styled(f"● {event.text}", "36"))
        elif isinstance(event, ReasoningSummary):
            self.activity.reasoning(event.text)
            summary = self._reasoning.accept(event.text)
            if self.mode != "quiet" and summary is not None:
                prefix = "● " if self.mode == "compact" else "Reasoning summary: "
                self.line(self._styled(f"{prefix}{summary}", "2"))
        elif isinstance(event, ContextCompacted):
            if self.mode != "quiet":
                self.line(
                    f"Context compacted: {event.summarized_items} items; "
                    f"{event.characters_before:,} → {event.characters_after:,} characters."
                )
        elif isinstance(event, ToolStarted):
            self.activity.tool_started(event)
            self._close_stream_line()
            self._response_text = ""
            if self.mode not in {"quiet", "compact"}:
                self.line(self._styled(f"→ {event.name}: {event.arguments}", "36"))
        elif isinstance(event, ToolFinished):
            action = self.activity.tool_finished(event)
            if self.mode == "compact":
                if event.failed:
                    detail = " ".join(event.output.splitlines())
                    detail = self._bounded(detail, self.COMPACT_OUTPUT_LIMIT)
                    self.line(
                        self._styled(
                            f"! {event.name} {action.outcome} · A{action.id}: {detail}", "31"
                        )
                    )
            elif self.mode != "quiet":
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
            if self.mode == "compact":
                self._flush_activity()
            else:
                self.activity.complete()
            if self.mode == "quiet" or self._response_text != event.text:
                self.line(event.text)
            else:
                self._close_stream_line()
            self._response_text = ""
        elif isinstance(event, TurnCancelled):
            self.activity.cancel_pending(event.reason)
            self._flush_activity() if self.mode == "compact" else self.activity.complete()
            self.line(event.reason)
        else:
            self._flush_activity() if self.mode == "compact" else self.activity.complete()
            self.line(f"! {event.kind}: {event.error}")


class RichRenderer:
    """Render readable Markdown and structured activity on an interactive terminal."""

    DEFAULT_OUTPUT_LIMIT = 2_000
    COMPACT_OUTPUT_LIMIT = 160

    def __init__(
        self,
        console: Console,
        *,
        mode: RenderMode = "default",
        reduced_motion: bool | None = None,
    ) -> None:
        self.console = console
        self.mode = mode
        self.reduced_motion = (
            os.environ.get("CALEBJR_REDUCED_MOTION", "").lower() in {"1", "true", "yes"}
            if reduced_motion is None
            else reduced_motion
        )
        self._live: Live | None = None
        self._status: Status | None = None
        self._stream_segment = ""
        self._response_text = ""
        self._turn_started_at = monotonic()
        self.activity = ActivityAggregator()
        self._reasoning = _ReasoningClumps()

    def set_mode(self, mode: RenderMode) -> None:
        """Change presentation detail between turns."""
        self._stop_status()
        self._stop_live()
        self.mode = mode

    def activity_report(self, limit: int = 1) -> str:
        """Return recent tool activity at user request."""
        return self.activity.report(limit)

    def activity_group(self, identifier: str) -> str:
        return self.activity.group(identifier)

    def activity_output(self, identifier: str, *, full: bool = False) -> str:
        return self.activity.output(identifier, full=full)

    def restore_activity(self, events: list[dict[str, Any]]) -> None:
        self.activity.restore(events)

    def _flush_activity(self) -> None:
        summary = self.activity.complete()
        if summary is None:
            return
        style = "red" if summary.failed_count else "green"
        self._print(
            Text.assemble(
                ("! " if summary.failed_count else "✓ ", f"{style} bold"),
                (summary.text, "dim"),
            )
        )

    def _stop_status(self) -> None:
        if self._status is not None:
            self._status.stop()
            self._status = None

    def _start_status(self, message: str, *, show_elapsed: bool = False) -> None:
        self._stop_status()
        if self.reduced_motion:
            if not message.startswith("Latest:"):
                self._stop_live()
                self.console.print(Text.assemble(("● ", "cyan bold"), (message, "dim")))
            return
        status: RenderableType = message
        if show_elapsed:
            status = _ElapsedStatus(message, self._turn_started_at)
        self._status = self.console.status(status, spinner="dots")
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
            self._turn_started_at = monotonic()
            self._reasoning.reset()
        elif isinstance(event, ModelTextDelta):
            if self.mode == "compact":
                self._flush_activity()
            if self.mode != "quiet":
                self._stream(event.text)
        elif isinstance(event, Progress):
            if self.mode != "quiet":
                self._stop_live()
                self._start_status(event.text, show_elapsed=self.mode == "compact")
        elif isinstance(event, ReasoningSummary):
            self.activity.reasoning(event.text)
            summary = self._reasoning.accept(event.text)
            if self.mode == "compact" and summary is not None:
                self._print(Text.assemble(("● ", "magenta bold"), (summary, "dim")))
            elif self.mode != "quiet" and summary is not None:
                self._print(Panel(summary, title="Reasoning summary", style="dim", expand=False))
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
            self.activity.tool_started(event)
            self._stop_status()
            self._stop_live()
            self._response_text = ""
            if self.mode == "quiet":
                return
            if self.mode == "compact":
                count = self.activity.action_count + 1
                phase, detail = tool_status(event)
                message = f"{phase} · {detail} · action {count}"
                self._start_status(message, show_elapsed=True)
                return
            heading = Text.assemble(("→ ", "cyan bold"), (event.name, "bold"))
            arguments = json.dumps(event.arguments, indent=2, ensure_ascii=False)
            syntax = Syntax(arguments, "json", background_color="default")
            self.console.print(Group(heading, syntax))
            self._start_status(f"Running {event.name}…")
        elif isinstance(event, ToolFinished):
            action = self.activity.tool_finished(event)
            if self.mode == "quiet":
                return
            status = action.outcome if event.failed else "done"
            style = "red" if event.failed else "green"
            if self.mode == "compact":
                if event.failed:
                    output = self._bounded(
                        " ".join(event.output.splitlines()), self.COMPACT_OUTPUT_LIMIT
                    )
                    self._print(
                        Text.assemble(
                            (f"! {event.name} {action.outcome} · A{action.id}: ", "red bold"),
                            output,
                        )
                    )
                else:
                    count = self.activity.action_count
                    message = f"Latest: {action.label} · {count} action{'s' if count != 1 else ''}"
                    self._start_status(message, show_elapsed=True)
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
            if self.mode == "compact":
                self._flush_activity()
            else:
                self.activity.complete()
            if self.mode == "quiet" or self._response_text != event.text:
                self.console.print(Markdown(event.text))
            self._response_text = ""
        elif isinstance(event, TurnCancelled):
            self._stop_status()
            self._stop_live()
            self.activity.cancel_pending(event.reason)
            self._flush_activity() if self.mode == "compact" else self.activity.complete()
            self._print(Text(f"! {event.reason}", style="yellow bold"))
        else:
            self._stop_status()
            self._stop_live()
            self._flush_activity() if self.mode == "compact" else self.activity.complete()
            self._print(Panel(event.error, title=f"Turn failed · {event.kind}", border_style="red"))


def terminal_renderer(*, mode: RenderMode = "default") -> RichRenderer | PlainRenderer:
    """Select rich output only when stdout is an interactive capable terminal."""
    if sys.stdout.isatty() and os.environ.get("TERM") != "dumb":
        console = Console(no_color="NO_COLOR" in os.environ)
        return RichRenderer(console, mode=mode)
    return PlainRenderer(mode=mode, color=False)
