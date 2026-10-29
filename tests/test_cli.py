"""Check the CLI contract and plugin loading without contacting model services."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from agents590 import cli
from agents590.calebjr import CalebJR
from agents590.calebjr import repl as calebjr_repl
from agents590.contracts import RunContext


def test_module_and_console_entrypoints_work_outside_project(tmp_path: Path) -> None:
    commands = [
        [sys.executable, "-m", "agents590"],
        [str(Path(sys.executable).parent / "agents590")],
    ]
    for command in commands:
        result = subprocess.run(
            [*command, "krisj_echo", "-i", "a one-line instruction"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        )
        assert result.stdout == "a one-line instruction\n"


def test_list_uses_registered_agent_entrypoints(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["--list"]) == 0
    names = capsys.readouterr().out.splitlines()
    assert {"krisj_starter", "krisj_echo"}.issubset(names)
    assert names == sorted(names)


@pytest.mark.parametrize("arguments", [[], ["krisj_echo"], ["krisj_echo", "-i", "  "]])
def test_instruction_is_required(arguments: list[str]) -> None:
    with pytest.raises(SystemExit) as error:
        cli.main(arguments)
    assert error.value.code == 2


def test_unknown_agent_has_a_useful_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["missing", "-i", "test"]) == 1
    assert "Unknown agent" in capsys.readouterr().err


def test_cli_passes_instruction_workdir_and_model_to_the_selected_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    seen: list[tuple[str, RunContext]] = []

    class RecordingAgent:
        async def run(self, instruction: str, context: RunContext) -> str:
            seen.append((instruction, context))
            return "completed"

    def factory(name: str) -> RecordingAgent:
        assert name == "custom-agent"
        return RecordingAgent()

    monkeypatch.setattr(cli, "create_agent", factory)
    instruction = "A quoted ' instruction; $(still literal)"
    assert (
        cli.main(
            [
                "custom-agent",
                "-i",
                instruction,
                "--cwd",
                str(tmp_path),
                "--model",
                "openai/selected-deployment",
            ]
        )
        == 0
    )
    assert seen == [(instruction, RunContext(cwd=tmp_path, model="openai/selected-deployment"))]
    assert capsys.readouterr().out == "completed\n"


def test_interactive_cli_allows_no_initial_instruction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[CalebJR, RunContext, str | None, str | None]] = []
    agent = CalebJR()

    async def fake_repl(
        selected_agent: CalebJR,
        context: RunContext,
        *,
        initial_instruction: str | None = None,
        render_mode: str = "default",
        resume: str | Path | None = None,
        confirm_destructive: bool = False,
        request_timeout: float = 120.0,
    ) -> int:
        seen.append((selected_agent, context, initial_instruction, render_mode))
        return 0

    monkeypatch.setattr(cli, "create_agent", lambda name: agent)
    monkeypatch.setattr(calebjr_repl, "run_repl", fake_repl)

    assert cli.main(["calebjr", "--interactive", "--cwd", str(tmp_path)]) == 0
    assert seen == [(agent, RunContext(cwd=tmp_path), None, None)]


def test_continue_shorthand_and_request_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[str | None, float]] = []
    agent = CalebJR()

    async def fake_repl(
        selected_agent: CalebJR,
        context: RunContext,
        *,
        initial_instruction: str | None = None,
        render_mode: str = "default",
        resume: str | Path | None = None,
        confirm_destructive: bool = False,
        request_timeout: float = 120.0,
    ) -> int:
        seen.append((initial_instruction, request_timeout))
        return 0

    monkeypatch.setattr(cli, "create_agent", lambda name: agent)
    monkeypatch.setattr(calebjr_repl, "run_repl", fake_repl)
    assert (
        cli.main(
            [
                "calebjr",
                "-i",
                "start here",
                "--continue",
                "--request-timeout",
                "45",
                "--cwd",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert seen == [("start here", 45.0)]


@pytest.mark.parametrize(
    ("flag", "expected_mode"),
    [
        ("--quiet", "quiet"),
        ("--compact", "compact"),
        ("--detailed", "default"),
        ("--debug", "verbose"),
        ("--verbose", "verbose"),
    ],
)
def test_interactive_cli_selects_render_mode(
    flag: str,
    expected_mode: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = CalebJR()
    seen: list[str] = []

    async def fake_repl(
        selected_agent: CalebJR,
        context: RunContext,
        *,
        initial_instruction: str | None = None,
        render_mode: str = "default",
        resume: str | Path | None = None,
        confirm_destructive: bool = False,
        request_timeout: float = 120.0,
    ) -> int:
        seen.append(render_mode)
        return 0

    monkeypatch.setattr(cli, "create_agent", lambda name: agent)
    monkeypatch.setattr(calebjr_repl, "run_repl", fake_repl)

    assert cli.main(["calebjr", "--interactive", flag, "--cwd", str(tmp_path)]) == 0
    assert seen == [expected_mode]


def test_interactive_cli_passes_destructive_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[bool] = []
    agent = CalebJR()

    async def fake_repl(
        selected_agent: CalebJR,
        context: RunContext,
        *,
        initial_instruction: str | None = None,
        render_mode: str = "default",
        resume: str | Path | None = None,
        confirm_destructive: bool = False,
        request_timeout: float = 120.0,
    ) -> int:
        seen.append(confirm_destructive)
        return 0

    monkeypatch.setattr(cli, "create_agent", lambda name: agent)
    monkeypatch.setattr(calebjr_repl, "run_repl", fake_repl)
    assert (
        cli.main(
            [
                "calebjr",
                "--interactive",
                "--confirm-destructive",
                "--cwd",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert seen == [True]


def test_interactive_cli_rejects_noninteractive_agent(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main(["krisj_echo", "--interactive"]) == 1
    assert "does not support interactive sessions" in capsys.readouterr().err


def test_starter_scaffold_fails_explicitly(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["krisj_starter", "-i", "test"]) == 1
    assert "Implement the UNC Azure model/tool loop" in capsys.readouterr().err


def test_explicit_env_file_preserves_exported_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("AGENTS590_TEST_VALUE=from-file\nAGENTS590_NEW_VALUE=new\n")
    monkeypatch.setenv("AGENTS590_TEST_VALUE", "from-shell")
    monkeypatch.delenv("AGENTS590_NEW_VALUE", raising=False)
    assert cli.main(["krisj_echo", "-i", "test", "--env-file", str(env_file)]) == 0
    assert os.environ["AGENTS590_TEST_VALUE"] == "from-shell"
    assert os.environ["AGENTS590_NEW_VALUE"] == "new"


@pytest.mark.parametrize("option", ["--cwd", "--env-file"])
def test_missing_paths_fail_before_loading_an_agent(tmp_path: Path, option: str) -> None:
    with pytest.raises(SystemExit) as error:
        cli.main(["krisj_echo", "-i", "test", option, str(tmp_path / "missing")])
    assert error.value.code == 2
