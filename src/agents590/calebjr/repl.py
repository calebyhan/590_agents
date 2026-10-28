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
from agents590.calebjr.transcript import session_path, sessions_dir
from agents590.contracts import RunContext

_HELP = """Commands:
  /help        Show this help.
  /context       Show bounded session context statistics.
  /context --raw Show bounded, redacted canonical model context.
  /status      Show token usage, estimated cost, and safety status.
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
    "/context --raw",
    "/status",
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
    confirm_destructive: bool = False,
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
    prompt_session: PromptSession[str] | None = None

    async def confirm(command: str) -> bool:
        try:
            if prompt_session is not None:
                answer = await prompt_session.prompt_async("Execute? [y/N] ")
            else:
                if input_fn is None:
                    return False
                answer = input_fn("Execute? [y/N] ")
        except (EOFError, KeyboardInterrupt):
            return False
        return answer.strip().lower() in {"y", "yes"}

    allow_external_resume = False
    if resume is not None:
        candidate = session_path(resume)
        if not candidate.is_relative_to(sessions_dir().resolve()):
            output_fn(f"Session file is outside the CalebJR data directory:\n{candidate}")
            try:
                if input_fn is None:
                    answer = await asyncio.to_thread(input, "Trust and load it? [y/N] ")
                else:
                    answer = input_fn("Trust and load it? [y/N] ")
            except (EOFError, KeyboardInterrupt):
                answer = ""
            allow_external_resume = answer.strip().lower() in {"y", "yes"}
            if not allow_external_resume:
                output_fn("Session load cancelled.")
                return 1

    if resume is None:
        session = agent.create_session(
            context,
            request_timeout=request_timeout,
            max_steps=INTERACTIVE_MAX_STEPS,
            max_invalid_responses=INTERACTIVE_MAX_INVALID_RESPONSES,
            max_tool_failures=INTERACTIVE_MAX_TOOL_FAILURES,
            max_repeated_calls=INTERACTIVE_MAX_REPEATED_CALLS,
        )
        session.confirm_destructive = confirm if confirm_destructive else None
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
            confirm_destructive=confirm if confirm_destructive else None,
            allow_external=allow_external_resume,
        )
    session.event_sink = renderer
    if input_fn is None:
        prompt_session = PromptSession(
            history=FileHistory(str(_history_file())),
            completer=WordCompleter(_COMMANDS, sentence=True),
            complete_while_typing=True,
            bottom_toolbar=session.toolbar_summary,
        )

    output_fn(f"CalebJR · {session.cwd.name} · {session.model}")
    output_fn("Type /help for commands.")
    interrupt_pending = False
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
                if interrupt_pending:
                    break
                interrupt_pending = True
                output_fn("Press Ctrl-C again to quit, or enter a prompt to continue.")
                continue
            interrupt_pending = False
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
            if command == "/context --raw":
                output_fn(session.raw_context())
                continue
            if command == "/status":
                output_fn(session.status_summary())
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
                interrupt_pending = True
                output_fn("Turn cancelled. Press Ctrl-C again to quit.")
                continue
    finally:
        await session.close()
    return 0
