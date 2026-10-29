"""Tests for the full-screen CalebJR interface."""

from pathlib import Path

from agents590.calebjr.tui import FullScreenUI


def test_tui_retains_complete_session_scrollback() -> None:
    ui = FullScreenUI()
    chunks = [f"line {index}\n" for index in range(10_000)]

    for chunk in chunks:
        ui.write(chunk)

    assert ui.scrollback == "".join(chunks)
    assert "line 0\n" in ui.scrollback
    assert "line 9999\n" in ui.scrollback


def test_tui_welcome_summarizes_new_and_resumed_sessions() -> None:
    fresh = FullScreenUI()
    fresh.show_welcome(
        cwd=Path("/work/sample"),
        model="gpt-test",
        activity_mode="compact",
        resumed=False,
    )
    resumed = FullScreenUI()
    resumed.show_welcome(
        cwd=Path("/work/sample"),
        model="gpt-test",
        activity_mode="compact",
        resumed=True,
    )

    assert "Ready to work in sample" in fresh.scrollback
    assert "Project   /work/sample" in fresh.scrollback
    assert "Model     gpt-test" in fresh.scrollback
    assert "Activity  compact" in fresh.scrollback
    assert "Session resumed in sample" in resumed.scrollback
