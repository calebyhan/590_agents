"""Full-screen prompt-toolkit interface with an in-memory scrollback."""

import asyncio
import os
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path

from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import Completer
from prompt_toolkit.data_structures import Point
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.layout import (
    BufferControl,
    FormattedTextControl,
    HSplit,
    Layout,
    VSplit,
    Window,
)
from prompt_toolkit.layout.margins import ScrollbarMargin
from prompt_toolkit.styles import Style


class FullScreenUI:
    """Own the alternate screen while retaining all output for the session."""

    def __init__(self, *, completer: Completer | None = None) -> None:
        self._text = ""
        self._prompt = "> "
        self._answers: asyncio.Queue[str] = asyncio.Queue()
        self._interrupt = asyncio.Event()
        self._auto_follow = True
        self._scroll_line = 0
        self._status: Callable[[], str] = lambda: ""
        self._toggle_activity: Callable[[], None] = lambda: None
        self._runner: asyncio.Task[None] | None = None

        self.input_buffer = Buffer(multiline=True, completer=completer, complete_while_typing=True)
        self.output_control = FormattedTextControl(
            self._formatted_output,
            focusable=True,
            show_cursor=False,
            get_cursor_position=lambda: Point(x=0, y=self._scroll_line),
        )
        self.output_window = Window(
            self.output_control,
            wrap_lines=True,
            always_hide_cursor=True,
            right_margins=[ScrollbarMargin(display_arrows=True)],
        )
        self.prompt_window = Window(
            FormattedTextControl(lambda: FormattedText([("class:prompt", self._prompt)])),
            width=lambda: len(self._prompt),
            height=1,
        )
        self.input_window = Window(
            BufferControl(buffer=self.input_buffer, focusable=True),
            height=lambda: min(8, max(1, self.input_buffer.document.line_count)),
            wrap_lines=True,
        )
        self.status_window = Window(
            FormattedTextControl(self._status_text),
            height=1,
            style="class:status",
        )

        bindings = KeyBindings()

        input_focused = Condition(lambda: self.app.layout.has_focus(self.input_window))

        @bindings.add("enter", filter=input_focused)
        def submit(event: KeyPressEvent) -> None:
            text = self.input_buffer.text
            self.input_buffer.set_document(Document("", 0), bypass_readonly=True)
            self._answers.put_nowait(text)

        @bindings.add("escape", "enter", filter=input_focused)
        def newline(event: KeyPressEvent) -> None:
            event.current_buffer.insert_text("\n")

        @bindings.add("c-d")
        def eof(event: KeyPressEvent) -> None:
            if not self.input_buffer.text:
                self._answers.put_nowait("\0")

        @bindings.add("c-c")
        def interrupt(event: KeyPressEvent) -> None:
            self._interrupt.set()

        @bindings.add("escape", "v")
        def toggle_activity(event: KeyPressEvent) -> None:
            self._toggle_activity()
            event.app.invalidate()

        @bindings.add("f6")
        def change_focus(event: KeyPressEvent) -> None:
            target = (
                self.input_window
                if event.app.layout.has_focus(self.output_window)
                else self.output_window
            )
            event.app.layout.focus(target)

        @bindings.add("pageup")
        def page_up(event: KeyPressEvent) -> None:
            self._auto_follow = False
            self._scroll_line = max(0, self._scroll_line - 10)
            self.output_window.vertical_scroll = max(0, self.output_window.vertical_scroll - 10)

        @bindings.add("pagedown")
        def page_down(event: KeyPressEvent) -> None:
            last_line = max(0, self._text.count("\n"))
            self._scroll_line = min(last_line, self._scroll_line + 10)
            self._auto_follow = self._scroll_line == last_line
            self.output_window.vertical_scroll += 10

        @bindings.add("escape", "<")
        def scroll_top(event: KeyPressEvent) -> None:
            self._auto_follow = False
            self._scroll_line = 0
            self.output_window.vertical_scroll = 0

        @bindings.add("escape", ">")
        def scroll_bottom(event: KeyPressEvent) -> None:
            self._auto_follow = True
            self._scroll_line = max(0, self._text.count("\n"))
            self.output_window.vertical_scroll = self._scroll_line

        root = HSplit(
            [
                Window(
                    FormattedTextControl(" CalebJR · full-screen session "),
                    height=1,
                    style="class:title",
                ),
                self.output_window,
                Window(height=1, char="─", style="class:separator"),
                VSplit([self.prompt_window, self.input_window]),
                self.status_window,
            ]
        )
        styles = (
            Style.from_dict({})
            if "NO_COLOR" in os.environ
            else Style.from_dict(
                {
                    "title": "bg:#1f2937 #f9fafb bold",
                    "status": "bg:#1f2937 #d1d5db",
                    "prompt": "#7dd3fc bold",
                    "separator": "#4b5563",
                }
            )
        )
        self.app: Application[None] = Application(
            layout=Layout(root, focused_element=self.input_window),
            key_bindings=bindings,
            full_screen=True,
            mouse_support=True,
            style=styles,
        )

    def _formatted_output(self) -> FormattedText:
        return FormattedText([("", self._text or "Session output will appear here.\n")])

    def _status_text(self) -> FormattedText:
        guidance = " PgUp/PgDn scroll · F6 focus · Alt-Enter newline · Ctrl-D exit "
        status = self._status()
        return FormattedText([("", f" {status}{' · ' if status else ''}{guidance}")])

    def set_status_provider(self, provider: Callable[[], str]) -> None:
        self._status = provider

    def set_activity_toggle(self, callback: Callable[[], None]) -> None:
        self._toggle_activity = callback

    async def start(self) -> None:
        if self._runner is None:
            self._runner = asyncio.create_task(self.app.run_async())
            await asyncio.sleep(0)

    async def close(self) -> None:
        if self._runner is None:
            return
        if not self._runner.done():
            self.app.exit()
        with suppress(asyncio.CancelledError):
            await self._runner
        self._runner = None

    @property
    def scrollback(self) -> str:
        """Return the complete output retained for the current TUI session."""
        return self._text

    def show_welcome(
        self,
        *,
        cwd: Path,
        model: str,
        activity_mode: str,
        resumed: bool,
    ) -> None:
        """Render a compact landing card with session context and useful next steps."""
        columns = self.app.output.get_size().columns
        inner_width = max(44, min(76, columns - 6))

        def fit(text: str) -> str:
            if len(text) > inner_width:
                text = text[: max(0, inner_width - 1)] + "…"
            return f"│ {text.ljust(inner_width)} │"

        heading = "CalebJR"
        top = f"╭─ {heading} " + "─" * max(0, inner_width - len(heading) - 1) + "╮"
        bottom = "╰" + "─" * (inner_width + 2) + "╯"
        state = "Session resumed" if resumed else "Ready to work"
        lines = [
            top,
            fit(f"{state} in {cwd.name or cwd}"),
            fit(""),
            fit(f"Project   {cwd}"),
            fit(f"Model     {model}"),
            fit(f"Activity  {activity_mode}"),
            fit(""),
            fit("Describe a task, ask about the project, or try:"),
            fit("  “Inspect this project and suggest the best next step.”"),
            fit(""),
            fit("/help commands  ·  PgUp/PgDn scroll  ·  Alt-Enter newline"),
            bottom,
            "",
        ]
        self.write("\n".join(lines))

    def line(self, text: str = "") -> None:
        self.write(text + "\n")

    def write(self, text: str) -> None:
        self._text += text
        if self._auto_follow:
            self._scroll_line = max(0, self._text.count("\n"))
            self.output_window.vertical_scroll = self._scroll_line
        self.app.invalidate()

    async def prompt(self, prompt: str = "> ") -> str:
        self._prompt = prompt
        self.app.layout.focus(self.input_window)
        self.app.invalidate()
        answer_task = asyncio.create_task(self._answers.get())
        interrupt_task = asyncio.create_task(self._interrupt.wait())
        done, pending = await asyncio.wait(
            {answer_task, interrupt_task}, return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        if interrupt_task in done:
            self._interrupt.clear()
            raise KeyboardInterrupt
        answer = answer_task.result()
        if answer == "\0":
            raise EOFError
        return answer

    async def run_interruptibly(
        self, awaitable: asyncio.Future[object] | asyncio.Task[object]
    ) -> object:
        """Cancel active agent work when Ctrl-C is pressed in the TUI."""
        action = asyncio.ensure_future(awaitable)
        interrupted = asyncio.create_task(self._interrupt.wait())
        done, pending = await asyncio.wait(
            {action, interrupted}, return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        if interrupted in done:
            self._interrupt.clear()
            action.cancel()
            with suppress(asyncio.CancelledError):
                await action
            raise KeyboardInterrupt
        return action.result()
