"""Check action-aware aggregation independently from terminal rendering."""

from agents590.calebjr.activity import ActivityAggregator, tool_status
from agents590.calebjr.events import ToolFinished, ToolStarted


def finish(
    activity: ActivityAggregator,
    call_id: str,
    name: str,
    arguments: dict[str, str],
    *,
    failed: bool = False,
    duration: float = 0.1,
) -> None:
    activity.tool_started(ToolStarted(call_id, name, arguments))
    activity.tool_finished(ToolFinished(call_id, name, "output", failed, duration))


def test_activity_summary_classifies_inspection_changes_and_checks() -> None:
    activity = ActivityAggregator()
    finish(activity, "1", "cat", {"path": "src/app.py"})
    finish(
        activity,
        "2",
        "edit",
        {"path": "src/app.py", "old_text": "secret old", "new_text": "secret new"},
    )
    finish(activity, "3", "shell", {"command": "pytest -q"}, duration=0.3)

    summary = activity.complete()

    assert summary is not None
    assert summary.text == (
        "3 actions · 0.50s · inspected 1 · changed 1 file (src/app.py) · ran 1 check"
    )
    report = activity.report()
    assert "Read src/app.py" in report
    assert "Edited src/app.py" in report
    assert "Ran pytest -q" in report
    assert "secret old" not in report
    assert "secret new" not in report


def test_tool_status_uses_semantic_phases_and_safe_details() -> None:
    assert tool_status(ToolStarted("1", "cat", {"path": "src/app.py"})) == (
        "Inspecting project",
        "src/app.py",
    )
    assert tool_status(ToolStarted("2", "edit", {"path": "src/app.py", "new_text": "secret"})) == (
        "Updating files",
        "src/app.py",
    )
    assert tool_status(ToolStarted("3", "shell", {"command": "pytest -q"})) == (
        "Running checks",
        "pytest -q",
    )


def test_activity_report_is_bounded_and_marks_failures() -> None:
    activity = ActivityAggregator(history_limit=2)
    for index in range(3):
        finish(
            activity,
            str(index),
            "shell",
            {"command": f"command-{index}"},
            failed=index == 2,
        )
        activity.complete()

    report = activity.report(20)

    assert "command-0" not in report
    assert "command-1" in report
    assert "! Activity 3" in report
    assert "1 failed" in report


def test_activity_output_has_stable_ids_bounds_and_restores() -> None:
    activity = ActivityAggregator()
    activity.tool_started(ToolStarted("call", "shell", {"command": "pytest -q"}))
    activity.tool_finished(ToolFinished("call", "shell", "x" * 2_100, True, 1.5, "timeout"))
    activity.reasoning("Checking the test suite.")
    activity.complete()

    assert "A1" in activity.report()
    assert "A1" in activity.group("G1")
    assert "No retained activity group G9." == activity.group("G9")
    assert "timeout" in activity.report()
    assert "Reasoning: Checking the test suite." in activity.report()
    assert "100 characters omitted; add --full" in activity.output("A1")
    assert "x" * 2_100 in activity.output("1", full=True)

    restored = ActivityAggregator()
    restored.restore(
        [
            {
                "type": "ToolStarted",
                "call_id": "call",
                "name": "shell",
                "arguments": {"command": "pytest -q"},
            },
            {
                "type": "ToolFinished",
                "call_id": "call",
                "name": "shell",
                "output": "timed out",
                "failed": True,
                "duration_seconds": 1.5,
                "outcome": "timeout",
            },
            {"type": "TurnFinished", "text": "done"},
        ]
    )
    assert "A1" in restored.report()
    assert "timed out" in restored.output("A1")


def test_pending_activity_becomes_cancelled_and_inspectable() -> None:
    activity = ActivityAggregator()
    activity.tool_started(ToolStarted("call", "shell", {"command": "sleep 10"}))
    activity.cancel_pending()
    activity.complete()

    assert "cancelled" in activity.report()
    assert "Turn cancelled." in activity.output("A1")
