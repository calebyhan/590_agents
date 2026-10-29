"""Interactive terminal UI for CalebJR."""

import asyncio
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import cast

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent

from agents590.calebjr import CalebJR
from agents590.calebjr.redaction import bounded, redact_text
from agents590.calebjr.render import PlainRenderer, RenderMode, RichRenderer, terminal_renderer
from agents590.calebjr.session import CalebJRSession
from agents590.calebjr.transcript import session_path, sessions_dir
from agents590.calebjr.tui import FullScreenUI
from agents590.contracts import RunContext

_HELP = """Commands:
  /help          Show this help.
  /activity MODE Set activity detail: quiet, compact, detailed, or debug.
  /activity show Show recent activity; optionally add a count or group ID.
  /activity output ACTION_ID [--full]
                 Show retained, redacted output for one action.
  /context       Show bounded session context statistics.
  /context --raw Show bounded, redacted canonical model context.
  /status        Show usage, cost, safety, and activity mode.
  /transcript [compact|detailed|markdown]
                 Show a bounded transcript in the selected format.
  /compact       Summarize older completed turns.
  /model NAME    Change the model for later turns.
  /cwd PATH      Change the working directory for later turns.
  /save [NAME]   Save a redacted session atomically.
  /clear         Clear conversation history (retains cwd and model).
  /exit, /quit   Exit the session.

Input:
  Enter        Submit the prompt.
  Ctrl-J       Insert a newline (also Alt-Enter, or end a line with \\).
  Alt-V        Temporarily toggle compact/debug activity.
  Ctrl-C       Cancel the running turn; press twice at the prompt to quit.
  Ctrl-D       Exit from an empty prompt.

Full-screen mode (CALEBJR_TUI=1):
  PgUp/PgDn    Scroll output; F6 moves focus.
  Alt-</Alt->  Jump to the top/bottom of output."""

INTERACTIVE_MAX_STEPS = 150
INTERACTIVE_MAX_INVALID_RESPONSES = 20
INTERACTIVE_MAX_TOOL_FAILURES = 25
INTERACTIVE_MAX_REPEATED_CALLS = 8

_ACTIVITY_MODES: dict[str, RenderMode] = {
    "quiet": "quiet",
    "compact": "compact",
    "detailed": "default",
    "debug": "verbose",
}
_ACTIVITY_LABELS = {value: key for key, value in _ACTIVITY_MODES.items()}


def _config_file() -> Path:
    root = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return root / "agents590" / "calebjr" / "config.json"


def _preferred_mode() -> RenderMode:
    requested = os.environ.get("CALEBJR_ACTIVITY_MODE", "").strip().lower()
    if requested in _ACTIVITY_MODES:
        return _ACTIVITY_MODES[requested]
    try:
        loaded: object = json.loads(_config_file().read_text(encoding="utf-8"))
        data = cast(dict[str, object], loaded) if isinstance(loaded, dict) else {}
        configured = data.get("activity_mode")
        if isinstance(configured, str) and configured in _ACTIVITY_MODES:
            return _ACTIVITY_MODES[configured]
    except (OSError, ValueError):
        pass
    return "compact"


def _save_preferred_mode(mode: RenderMode) -> None:
    path = _config_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps({"activity_mode": _ACTIVITY_LABELS[mode]}, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


_COMMANDS = [
    "/help",
    "/activity",
    "/activity quiet",
    "/activity compact",
    "/activity detailed",
    "/activity debug",
    "/activity show",
    "/activity output",
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


def _format_transcript(items: list[tuple[str, str]], mode: str) -> str:
    """Render a bounded, redacted transcript for explicit inspection or export."""
    selected = items[-1_000:]
    if mode == "compact":
        selected = [(role, text) for role, text in selected if role != "tool"]
    lines: list[str] = []
    for role, raw_text in selected:
        text = bounded(redact_text(raw_text), 20_000)
        if mode == "markdown":
            heading = {
                "user": "User",
                "assistant": "Assistant",
                "tool": "Tool output",
                "error": "Error",
                "boundary": "Session",
            }.get(role, role.title())
            if role == "tool":
                lines.append(f"## {heading}\n\n```text\n{text}\n```")
            else:
                lines.append(f"## {heading}\n\n{text}")
        else:
            lines.append(f"{role}: {text}")
    return bounded("\n\n".join(lines), 100_000)


def prompt_key_bindings(renderer: PlainRenderer | RichRenderer) -> KeyBindings:
    """Mirror Claude Code: Enter submits; Ctrl-J, Alt-Enter, or a trailing ``\\`` add a line."""
    bindings = KeyBindings()

    @bindings.add("enter")
    def submit(event: KeyPressEvent) -> None:
        buffer = event.current_buffer
        completion = buffer.complete_state.current_completion if buffer.complete_state else None
        if completion is not None:
            buffer.apply_completion(completion)
        elif buffer.document.text_before_cursor.endswith("\\"):
            buffer.delete_before_cursor()
            buffer.insert_text("\n")
        else:
            buffer.validate_and_handle()

    @bindings.add("c-j")
    @bindings.add("escape", "enter")
    def newline(event: KeyPressEvent) -> None:
        event.current_buffer.insert_text("\n")

    @bindings.add("escape", "v")
    def toggle_activity(event: KeyPressEvent) -> None:
        mode: RenderMode = "compact" if renderer.mode == "verbose" else "verbose"
        renderer.set_mode(mode)
        event.app.invalidate()

    return bindings


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
    render_mode: RenderMode | None = None,
    resume: str | Path | None = None,
    confirm_destructive: bool = False,
    full_screen: bool = False,
) -> int:
    """Run one stateful session; injectable I/O keeps tests deterministic."""
    interactive_terminal = input_fn is None and output_fn is print and write_fn is None
    # Inline output keeps native scrollback and copy/paste; full-screen is opt-in.
    use_tui = (
        interactive_terminal
        and (
            full_screen or os.environ.get("CALEBJR_TUI", "").strip().lower() in {"1", "true", "yes"}
        )
        and sys.stdin.isatty()
        and sys.stdout.isatty()
    )
    tui = FullScreenUI(completer=WordCompleter(_COMMANDS, sentence=True)) if use_tui else None
    if tui is not None:
        output_fn = tui.line
        write_fn = tui.write
    selected_mode = (
        _preferred_mode()
        if render_mode is None and interactive_terminal
        else render_mode or "compact"
    )
    if write_fn is None:
        if output_fn is print:

            def terminal_write(text: str) -> None:
                sys.stdout.write(text)
                sys.stdout.flush()

            write_fn = terminal_write
        else:
            write_fn = output_fn
    renderer = (
        terminal_renderer(mode=selected_mode)
        if interactive_terminal and tui is None
        else PlainRenderer(
            output_fn,
            write=write_fn,
            mode=selected_mode,
            # The full-screen output control shows escape sequences literally.
            color=False if tui is not None else None,
        )
    )
    prompt_session: PromptSession[str] | None = None
    if tui is not None:

        def toggle_tui_activity() -> None:
            mode: RenderMode = "compact" if renderer.mode == "verbose" else "verbose"
            renderer.set_mode(mode)

        tui.set_activity_toggle(toggle_tui_activity)
        await tui.start()

    async def confirm(command: str) -> bool:
        try:
            if tui is not None:
                answer = await tui.prompt("Execute? [y/N] ")
            elif prompt_session is not None:
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
                if tui is not None:
                    answer = await tui.prompt("Trust and load it? [y/N] ")
                elif input_fn is None:
                    answer = await asyncio.to_thread(input, "Trust and load it? [y/N] ")
                else:
                    answer = input_fn("Trust and load it? [y/N] ")
            except (EOFError, KeyboardInterrupt):
                answer = ""
            allow_external_resume = answer.strip().lower() in {"y", "yes"}
            if not allow_external_resume:
                output_fn("Session load cancelled.")
                if tui is not None:
                    await tui.close()
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
    renderer.restore_activity(session.event_log)
    if input_fn is None and tui is None:
        prompt_session = PromptSession(
            history=FileHistory(str(_history_file())),
            completer=WordCompleter(_COMMANDS, sentence=True),
            complete_while_typing=True,
            bottom_toolbar=session.toolbar_summary,
            key_bindings=prompt_key_bindings(renderer),
            multiline=True,
            prompt_continuation="  ",
        )

    if tui is not None:
        tui.set_status_provider(session.toolbar_summary)
    if tui is not None:
        tui.show_welcome(
            cwd=session.cwd,
            model=session.model,
            activity_mode=_ACTIVITY_LABELS[renderer.mode],
            resumed=resume is not None,
        )
    else:
        output_fn(f"CalebJR · {session.cwd.name} · {session.model}")
        output_fn("Type /help for commands.")
    interrupt_pending = False

    async def run_turn(instruction: str) -> None:
        turn = asyncio.create_task(session.run_turn(instruction))
        if tui is None:
            await turn
        else:
            await tui.run_interruptibly(turn)

    try:
        if initial_instruction and initial_instruction.strip():
            await run_turn(initial_instruction)
        while True:
            try:
                if tui is not None:
                    line = await tui.prompt("> ")
                elif prompt_session is not None:
                    line = await prompt_session.prompt_async("> ")
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
            if command == "/activity" or command.startswith("/activity "):
                parts = command.split()
                if len(parts) == 1:
                    output_fn(f"Activity mode: {_ACTIVITY_LABELS[renderer.mode]}")
                elif parts[1] == "show":
                    if len(parts) == 3 and parts[2].startswith("g"):
                        output_fn(renderer.activity_group(parts[2]))
                    else:
                        try:
                            limit = 1 if len(parts) == 2 else int(parts[2])
                            if len(parts) > 3 or not 1 <= limit <= 20:
                                raise ValueError
                        except (IndexError, ValueError):
                            output_fn("Usage: /activity show [1-20|GGROUP_ID]")
                        else:
                            output_fn(renderer.activity_report(limit))
                elif parts[1] == "output":
                    full = len(parts) == 4 and parts[3] == "--full"
                    if len(parts) not in {3, 4} or (len(parts) == 4 and not full):
                        output_fn("Usage: /activity output ACTION_ID [--full]")
                    else:
                        rendered = redact_text(renderer.activity_output(parts[2], full=full))
                        if full:
                            rendered = (
                                "Warning: full tool output may contain sensitive project data; "
                                "likely credentials are redacted.\n" + rendered
                            )
                        output_fn(rendered)
                elif len(parts) == 2 and parts[1] in _ACTIVITY_MODES:
                    renderer.set_mode(_ACTIVITY_MODES[parts[1]])
                    if interactive_terminal:
                        try:
                            _save_preferred_mode(cast(RenderMode, renderer.mode))
                        except OSError as error:
                            output_fn(
                                f"Activity mode changed, but preference was not saved: {error}"
                            )
                            continue
                    output_fn(f"Activity mode changed to {parts[1]}.")
                else:
                    output_fn(
                        "Usage: /activity quiet|compact|detailed|debug|show [1-20]|"
                        "output ACTION_ID [--full]"
                    )
                continue
            if command == "/context":
                output_fn(session.context_summary())
                continue
            if command == "/context --raw":
                output_fn(session.raw_context())
                continue
            if command == "/status":
                output_fn(
                    f"{session.status_summary()}\nActivity mode: {_ACTIVITY_LABELS[renderer.mode]}"
                )
                continue
            if command == "/transcript" or command.startswith("/transcript "):
                parts = command.split()
                transcript_mode = "detailed" if len(parts) == 1 else parts[1]
                if len(parts) > 2 or transcript_mode not in {"compact", "detailed", "markdown"}:
                    output_fn("Usage: /transcript [compact|detailed|markdown]")
                elif not session.visible_transcript:
                    output_fn("Transcript is empty.")
                else:
                    output_fn(_format_transcript(session.visible_transcript, transcript_mode))
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
                renderer.restore_activity([])
                output_fn("Session history cleared.")
                continue
            if instruction.startswith("/"):
                output_fn(f"Unknown command: {instruction}. Type /help.")
                continue
            try:
                await run_turn(instruction)
            except (KeyboardInterrupt, asyncio.CancelledError):
                interrupt_pending = True
                output_fn("Press Ctrl-C again to quit, or enter a prompt to continue.")
                continue
    finally:
        await session.close()
        if tui is not None:
            await tui.close()
    return 0
