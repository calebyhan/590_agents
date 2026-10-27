"""Minimal plain-text interactive terminal for CalebJR."""

import asyncio
from collections.abc import Callable

from agents590.calebjr import CalebJR
from agents590.calebjr.events import (
    Progress,
    SafetyReviewFinished,
    SafetyReviewStarted,
    SessionEvent,
    ToolFinished,
    ToolStarted,
    TurnFailed,
    TurnFinished,
)
from agents590.contracts import RunContext

_HELP = """Commands:
  /help        Show this help.
  /context     Show bounded session context statistics.
  /transcript  Show conversation and tool activity.
  /clear       Clear conversation history (retains cwd and model).
  /exit, /quit Exit the session."""


class PlainRenderer:
    """Render stable text without ANSI escapes."""

    def __init__(self, output: Callable[[str], None] = print) -> None:
        self.output = output

    async def __call__(self, event: SessionEvent) -> None:
        if isinstance(event, Progress):
            self.output(f"● {event.text}")
        elif isinstance(event, ToolStarted):
            self.output(f"→ {event.name}: {event.arguments}")
        elif isinstance(event, ToolFinished):
            marker = "!" if event.failed else " "
            self.output(f"{marker} {event.output}")
        elif isinstance(event, SafetyReviewStarted):
            self.output(f"? safety review: {event.command}")
        elif isinstance(event, SafetyReviewFinished):
            self.output("✓ safety review approved" if event.approved else f"! {event.decision}")
        elif isinstance(event, TurnFinished):
            self.output(event.text)
        elif isinstance(event, TurnFailed) and event.error == "Turn cancelled.":
            self.output(event.error)


async def run_repl(
    agent: CalebJR,
    context: RunContext,
    *,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
    request_timeout: float = 120.0,
    initial_instruction: str | None = None,
) -> int:
    """Run a session using injectable plain input/output for deterministic tests."""
    renderer = PlainRenderer(output_fn)
    session = agent.create_session(context, request_timeout=request_timeout)
    # Keep session creation centralized on the agent, but install the terminal sink here.
    session.event_sink = renderer
    output_fn(f"CalebJR · {context.cwd.name} · {session.settings.model}")
    output_fn("Type /help for commands.")
    try:
        if initial_instruction and initial_instruction.strip():
            await session.run_turn(initial_instruction)
        while True:
            try:
                line = input_fn("> ")
            except EOFError:
                break
            except KeyboardInterrupt:
                output_fn("Use /exit to quit.")
                continue
            instruction = line.strip()
            if not instruction:
                continue
            command = instruction.lower()
            if command in {"/exit", "/quit"}:
                break
            if command == "/help":
                output_fn(_HELP)
                continue
            if command == "/context":
                output_fn(session.context_summary())
                continue
            if command == "/transcript":
                if not session.visible_transcript:
                    output_fn("Transcript is empty.")
                else:
                    output_fn(
                        "\n".join(f"{role}: {text}" for role, text in session.visible_transcript)
                    )
                continue
            if command == "/clear":
                session.clear()
                output_fn("Session history cleared.")
                continue
            if instruction.startswith("/"):
                output_fn(f"Unknown command: {instruction}. Type /help.")
                continue
            try:
                await session.run_turn(instruction)
            except (KeyboardInterrupt, asyncio.CancelledError):
                # The session rolls back and emits the cancellation before re-raising.
                continue
    finally:
        await session.close()
    return 0
