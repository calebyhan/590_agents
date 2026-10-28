"""Interactive terminal UI for CalebJR."""

import asyncio
import os
import sys
from collections.abc import Callable
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import FileHistory

from agents590.calebjr import CalebJR
from agents590.calebjr.redaction import bounded, redact_text
from agents590.calebjr.render import PlainRenderer, RenderMode, terminal_renderer
from agents590.calebjr.session import CalebJRSession
from agents590.contracts import RunContext

_HELP = """Commands:
  /help        Show this help.
  /context     Show bounded session context statistics.
  /transcript  Show conversation and tool activity.
  /compact     Summarize older completed turns.
  /model NAME  Change the model for later turns.
  /cwd PATH    Change the working directory for later turns.
  /save [NAME] Save a redacted session atomically.
  /clear       Clear conversation history (retains cwd and model).
  /exit, /quit Exit the session.

Input:
  Esc+Enter    Submit a multiline prompt.
  Ctrl-D       Exit from an empty prompt."""

INTERACTIVE_MAX_STEPS = 150
INTERACTIVE_MAX_INVALID_RESPONSES = 20
INTERACTIVE_MAX_TOOL_FAILURES = 25
INTERACTIVE_MAX_REPEATED_CALLS = 8

_COMMANDS = [
    "/help",
    "/context",
    "/transcript",
    "/compact",
    "/model",
    "/cwd",
    "/save",
    "/clear",
    "/exit",
    "/quit",
]


def _history_file() -> Path:
    state_home = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
    path = state_home / "agents590" / "calebjr" / "history"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


async def run_repl(
    agent: CalebJR,
    context: RunContext,
    *,
    input_fn: Callable[[str], str] | None = None,
    output_fn: Callable[[str], None] = print,
    write_fn: Callable[[str], None] | None = None,
    request_timeout: float = 120.0,
    initial_instruction: str | None = None,
    render_mode: RenderMode = "default",
    resume: str | Path | None = None,
) -> int:
    """Run one stateful session; injectable I/O keeps tests deterministic."""
    interactive_terminal = input_fn is None and output_fn is print and write_fn is None
    if write_fn is None:
        if output_fn is print:

            def terminal_write(text: str) -> None:
                sys.stdout.write(text)
                sys.stdout.flush()

            write_fn = terminal_write
        else:
            write_fn = output_fn
    renderer = (
        terminal_renderer(mode=render_mode)
        if interactive_terminal
        else PlainRenderer(output_fn, write=write_fn, mode=render_mode)
    )
    if resume is None:
        session = agent.create_session(
            context,
            request_timeout=request_timeout,
            max_steps=INTERACTIVE_MAX_STEPS,
            max_invalid_responses=INTERACTIVE_MAX_INVALID_RESPONSES,
            max_tool_failures=INTERACTIVE_MAX_TOOL_FAILURES,
            max_repeated_calls=INTERACTIVE_MAX_REPEATED_CALLS,
        )
    else:
        session = CalebJRSession.resume(
            resume,
            settings=agent.provider_settings(context),
            event_sink=renderer,
            request_timeout=request_timeout,
            max_steps=INTERACTIVE_MAX_STEPS,
            max_invalid_responses=INTERACTIVE_MAX_INVALID_RESPONSES,
            max_tool_failures=INTERACTIVE_MAX_TOOL_FAILURES,
            max_repeated_calls=INTERACTIVE_MAX_REPEATED_CALLS,
        )
    session.event_sink = renderer
    prompt_session: PromptSession[str] | None = None
    if input_fn is None:
        toolbar = f" {session.model}  |  {session.cwd}  |  {render_mode} "
        prompt_session = PromptSession(
            history=FileHistory(str(_history_file())),
            completer=WordCompleter(_COMMANDS, sentence=True),
            complete_while_typing=True,
            bottom_toolbar=toolbar,
        )

    output_fn(f"CalebJR · {session.cwd.name} · {session.model}")
    output_fn("Type /help for commands.")
    try:
        if initial_instruction and initial_instruction.strip():
            await session.run_turn(initial_instruction)
        while True:
            try:
                if prompt_session is not None:
                    line = await prompt_session.prompt_async("> ", multiline=True)
                else:
                    assert input_fn is not None
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
                        "\n".join(
                            f"{role}: {bounded(redact_text(text), 20_000)}"
                            for role, text in session.visible_transcript[-1_000:]
                        )
                    )
                continue
            if command == "/compact":
                output_fn(
                    "Context compacted."
                    if await session.compact()
                    else "Nothing to compact; at least two recent turns are retained."
                )
                continue
            if command == "/model" or command.startswith("/model "):
                parts = instruction.split(maxsplit=1)
                if len(parts) == 1:
                    output_fn(f"Current model: {session.model}")
                else:
                    try:
                        session.set_model(parts[1])
                        output_fn(f"Model changed to {session.model}.")
                    except ValueError as error:
                        output_fn(f"Unable to change model: {error}")
                continue
            if command == "/cwd" or command.startswith("/cwd "):
                parts = instruction.split(maxsplit=1)
                if len(parts) == 1:
                    output_fn(f"Working directory: {session.cwd}")
                else:
                    try:
                        session.set_cwd(Path(parts[1]))
                        output_fn(f"Working directory changed to {session.cwd}.")
                    except ValueError as error:
                        output_fn(f"Unable to change working directory: {error}")
                continue
            if command == "/save" or command.startswith("/save "):
                parts = instruction.split(maxsplit=1)
                try:
                    path = session.save(parts[1] if len(parts) == 2 else None)
                    output_fn(f"Session saved: {path}")
                except (OSError, ValueError) as error:
                    output_fn(f"Unable to save session: {error}")
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
                continue
    finally:
        await session.close()
    return 0
