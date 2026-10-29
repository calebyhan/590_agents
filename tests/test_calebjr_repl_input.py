"""Check CalebJR's inline prompt behaves like Claude Code's input box."""

import asyncio
import sys
from pathlib import Path

import pytest
from prompt_toolkit import PromptSession
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from agents590.calebjr import repl
from agents590.calebjr.render import PlainRenderer
from agents590.contracts import RunContext
from tests.test_calebjr_session import ClientCreatingSettings, FakeAgent


async def _submit(keys: str) -> str:
    with create_pipe_input() as pipe:
        session: PromptSession[str] = PromptSession(
            input=pipe,
            output=DummyOutput(),
            multiline=True,
            key_bindings=repl.prompt_key_bindings(PlainRenderer(color=False)),
        )
        pipe.send_text(keys)
        return await asyncio.wait_for(session.prompt_async("> "), 1.0)


@pytest.mark.parametrize(
    ("keys", "expected"),
    [
        ("hello\r", "hello"),
        ("first\nsecond\r", "first\nsecond"),
        ("first\x1b\rsecond\r", "first\nsecond"),
        ("first\\\rsecond\r", "first\nsecond"),
        ("\x1b[200~pasted\rlines\x1b[201~\r", "pasted\nlines"),
    ],
    ids=["enter-submits", "ctrl-j", "alt-enter", "backslash-enter", "bracketed-paste"],
)
def test_prompt_enter_submits_and_newline_keys_insert(keys: str, expected: str) -> None:
    assert asyncio.run(_submit(keys)) == expected


class _EndingPromptSession:
    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    async def prompt_async(self, *args: object, **kwargs: object) -> str:
        raise EOFError


def test_interactive_terminal_defaults_to_inline_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def full_screen_ui(**kwargs: object) -> None:
        raise AssertionError("full-screen UI should be opt-in")

    monkeypatch.setattr(repl, "FullScreenUI", full_screen_ui)
    monkeypatch.setattr(repl, "PromptSession", _EndingPromptSession)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.delenv("CALEBJR_TUI", raising=False)

    result = asyncio.run(run_inline(tmp_path))

    assert result == 0


async def run_inline(cwd: Path) -> int:
    return await repl.run_repl(FakeAgent(ClientCreatingSettings()), RunContext(cwd=cwd))
