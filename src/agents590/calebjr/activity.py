"""Aggregate low-level tool events into concise, inspectable activity summaries."""

from collections import deque
from dataclasses import dataclass
from time import monotonic
from typing import Any, Literal, cast

from agents590.calebjr.events import ToolFinished, ToolStarted

_MUTATING_TOOLS = {"edit", "write", "delete", "mkdir", "rmdir", "move", "copy", "rename"}
_INSPECTION_TOOLS = {"cat", "ls", "search"}
_CHECK_MARKERS = (
    "pytest",
    "ruff",
    "pyright",
    "mypy",
    "make check",
    "make test",
    "npm test",
    "pnpm test",
    "yarn test",
    "cargo test",
    "go test",
)
ActivityOutcome = Literal["success", "failed", "timeout", "cancelled", "denied"]


def is_check_command(command: str) -> bool:
    """Return whether a shell command appears to run project validation."""
    lowered = command.lower()
    return any(marker in lowered for marker in _CHECK_MARKERS)


def tool_status(event: ToolStarted) -> tuple[str, str]:
    """Return a semantic phase and safe, concise detail for an active tool."""
    arguments = event.arguments
    if event.name in _INSPECTION_TOOLS:
        detail = arguments.get("path") or arguments.get("query") or event.name
        return "Inspecting project", detail
    if event.name in _MUTATING_TOOLS:
        detail = arguments.get("path") or arguments.get("dst") or arguments.get("src") or event.name
        return "Updating files", detail
    if event.name == "shell":
        command = " ".join(arguments.get("command", "command").split())
        phase = "Running checks" if is_check_command(command) else "Running command"
        return phase, f"{command[:80]}{'…' if len(command) > 80 else ''}"
    if event.name == "help":
        return "Inspecting tool help", arguments.get("tool_name", "tool")
    return f"Running {event.name}", event.name


@dataclass(frozen=True)
class ActivityAction:
    """One completed tool action with safe display metadata and retained output."""

    id: int
    group_id: int | None
    name: str
    arguments: dict[str, str]
    output: str
    outcome: ActivityOutcome
    duration_seconds: float

    @property
    def failed(self) -> bool:
        return self.outcome != "success"

    @property
    def label(self) -> str:
        """Describe the action without exposing file contents or edit text."""
        if self.name == "cat":
            return f"Read {self.arguments.get('path', 'file')}"
        if self.name == "ls":
            return f"Listed {self.arguments.get('path', 'directory')}"
        if self.name == "search":
            return f"Searched for {self.arguments.get('query', 'files')}"
        if self.name == "shell":
            command = " ".join(self.arguments.get("command", "command").split())
            return f"Ran {command[:120]}{'…' if len(command) > 120 else ''}"
        if self.name in {"move", "copy", "rename"}:
            src = self.arguments.get("src", "source")
            dst = self.arguments.get("dst", "destination")
            verb = {"move": "Moved", "copy": "Copied", "rename": "Renamed"}[self.name]
            return f"{verb} {src} → {dst}"
        verbs = {
            "edit": "Edited",
            "write": "Wrote",
            "delete": "Deleted",
            "mkdir": "Created directory",
            "rmdir": "Removed directory",
            "help": "Viewed help for",
        }
        target = self.arguments.get("path") or self.arguments.get("tool_name") or self.name
        return f"{verbs.get(self.name, self.name.title())} {target}"

    @property
    def changed_paths(self) -> tuple[str, ...]:
        if self.name not in _MUTATING_TOOLS or self.failed:
            return ()
        if self.name in {"move", "copy", "rename"}:
            return tuple(
                path for path in (self.arguments.get("src"), self.arguments.get("dst")) if path
            )
        path = self.arguments.get("path")
        return (path,) if path else ()

    @property
    def is_check(self) -> bool:
        return self.name == "shell" and is_check_command(self.arguments.get("command", ""))


@dataclass(frozen=True)
class ActivitySummary:
    """A completed, stably identified group of tool actions."""

    id: int
    actions: tuple[ActivityAction, ...]
    reasoning: tuple[str, ...] = ()

    @property
    def failed_count(self) -> int:
        return sum(action.failed for action in self.actions)

    @property
    def duration_seconds(self) -> float:
        return sum(action.duration_seconds for action in self.actions)

    @property
    def changed_paths(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(path for action in self.actions for path in action.changed_paths)
        )

    @property
    def text(self) -> str:
        count = len(self.actions)
        parts = [f"{count} action{'s' if count != 1 else ''}", f"{self.duration_seconds:.2f}s"]
        inspected = sum(action.name in _INSPECTION_TOOLS for action in self.actions)
        checks = sum(action.is_check for action in self.actions)
        commands = sum(action.name == "shell" and not action.is_check for action in self.actions)
        mutations = sum(action.name in _MUTATING_TOOLS for action in self.actions)
        other = count - inspected - checks - commands - mutations
        if inspected:
            parts.append(f"inspected {inspected}")
        if self.changed_paths:
            changed_count = len(self.changed_paths)
            paths = ", ".join(self.changed_paths[:3])
            if changed_count > 3:
                paths += f", +{changed_count - 3} more"
            noun = "files" if changed_count != 1 else "file"
            parts.append(f"changed {changed_count} {noun} ({paths})")
        if checks:
            parts.append(f"ran {checks} check{'s' if checks != 1 else ''}")
        if commands:
            parts.append(f"ran {commands} command{'s' if commands != 1 else ''}")
        if other:
            parts.append(f"{other} other")
        outcomes = [action.outcome for action in self.actions if action.failed]
        if outcomes:
            for outcome in ("failed", "timeout", "cancelled", "denied"):
                amount = outcomes.count(outcome)
                if amount:
                    parts.append(f"{amount} {outcome}")
        return " · ".join(parts)


class ActivityAggregator:
    """Collect tool events and retain a bounded history of completed groups."""

    def __init__(self, *, history_limit: int = 100) -> None:
        self.history_limit = history_limit
        self._started: dict[str, tuple[ToolStarted, float]] = {}
        self._actions: list[ActivityAction] = []
        self._reasoning: list[str] = []
        self._history: deque[ActivitySummary] = deque(maxlen=history_limit)
        self._next_action_id = 1
        self._next_group_id = 1

    @property
    def action_count(self) -> int:
        return len(self._actions)

    def tool_started(self, event: ToolStarted) -> None:
        self._started[event.call_id] = (event, monotonic())

    def reasoning(self, text: str) -> None:
        """Associate a bounded provider summary with the current activity group."""
        normalized = text.strip()
        if normalized and (not self._reasoning or self._reasoning[-1] != normalized):
            self._reasoning.append(normalized[:320])
            del self._reasoning[:-4]

    @staticmethod
    def _outcome(event: ToolFinished) -> ActivityOutcome:
        outcome = getattr(event, "outcome", None)
        if outcome in {"success", "failed", "timeout", "cancelled", "denied"}:
            return outcome
        return "failed" if event.failed else "success"

    def tool_finished(self, event: ToolFinished) -> ActivityAction:
        pending = self._started.pop(event.call_id, None)
        started = pending[0] if pending is not None else None
        action_id = (
            started.action_id
            if started is not None and started.action_id is not None
            else event.action_id or self._next_action_id
        )
        group_id = started.group_id if started is not None else event.group_id
        action = ActivityAction(
            id=action_id,
            group_id=group_id,
            name=event.name,
            arguments=dict(started.arguments) if started is not None else {},
            output=event.output,
            outcome=self._outcome(event),
            duration_seconds=event.duration_seconds,
        )
        self._next_action_id = max(self._next_action_id, action.id + 1)
        self._actions.append(action)
        return action

    def cancel_pending(self, reason: str = "Turn cancelled.") -> None:
        """Convert in-flight tools into inspectable cancelled actions."""
        now = monotonic()
        for event, started_at in self._started.values():
            action_id = event.action_id or self._next_action_id
            self._actions.append(
                ActivityAction(
                    id=action_id,
                    group_id=event.group_id,
                    name=event.name,
                    arguments=dict(event.arguments),
                    output=reason,
                    outcome="cancelled",
                    duration_seconds=max(0.0, now - started_at),
                )
            )
            self._next_action_id = max(self._next_action_id, action_id + 1)
        self._started = {}

    def complete(self) -> ActivitySummary | None:
        if not self._actions:
            self._started = {}
            self._reasoning = []
            return None
        group_id = self._actions[0].group_id or self._next_group_id
        summary = ActivitySummary(group_id, tuple(self._actions), tuple(self._reasoning))
        self._next_group_id = max(self._next_group_id, group_id + 1)
        self._actions = []
        self._reasoning = []
        self._started = {}
        self._history.append(summary)
        return summary

    @staticmethod
    def _summary_lines(summary: ActivitySummary) -> list[str]:
        marker = "!" if summary.failed_count else "✓"
        lines = [f"{marker} Activity {summary.id} (G{summary.id}): {summary.text}"]
        lines.extend(f"  · Reasoning: {reasoning}" for reasoning in summary.reasoning)
        for action in summary.actions:
            action_marker = "!" if action.failed else "✓"
            suffix = f" · {action.outcome}" if action.failed else ""
            lines.append(
                f"  {action_marker} A{action.id} {action.label} · "
                f"{action.duration_seconds:.2f}s{suffix}"
            )
        return lines

    def report(self, limit: int = 1) -> str:
        """Return recent completed activity, newest last."""
        summaries = list(self._history)[-limit:]
        if not summaries:
            return "No completed activity yet."
        return "\n".join(line for summary in summaries for line in self._summary_lines(summary))

    def group(self, identifier: str) -> str:
        """Return one completed activity group by stable ID."""
        normalized = identifier.strip().upper()
        if normalized.startswith("G"):
            normalized = normalized[1:]
        try:
            group_id = int(normalized)
        except ValueError:
            return "Usage: /activity show GGROUP_ID"
        summary = next(
            (summary for summary in reversed(self._history) if summary.id == group_id), None
        )
        if summary is None:
            return f"No retained activity group G{group_id}."
        return "\n".join(self._summary_lines(summary))

    def output(self, identifier: str, *, full: bool = False) -> str:
        """Return one action's retained output after explicit user selection."""
        normalized = identifier.strip().upper()
        if normalized.startswith("A"):
            normalized = normalized[1:]
        try:
            action_id = int(normalized)
        except ValueError:
            return "Usage: /activity output ACTION_ID [--full]"
        action = next(
            (
                action
                for summary in reversed(self._history)
                for action in reversed(summary.actions)
                if action.id == action_id
            ),
            None,
        )
        if action is None:
            return f"No retained activity output for A{action_id}."
        limit = 100_000 if full else 2_000
        text = action.output
        if len(text) > limit:
            text = f"{text[:limit]}\n… ({len(text) - limit:,} characters omitted; add --full)"
        return f"A{action.id} · {action.label} · {action.outcome}\n{text}"

    def restore(self, events: list[dict[str, Any]]) -> None:
        """Rebuild stable activity IDs from persisted session events."""
        self._started = {}
        self._actions = []
        self._reasoning = []
        self._history = deque(maxlen=self.history_limit)
        self._next_action_id = self._next_group_id = 1
        for raw in events:
            event_type = raw.get("type")
            if event_type == "ToolStarted":
                arguments = raw.get("arguments")
                self.tool_started(
                    ToolStarted(
                        str(raw.get("call_id", "")),
                        str(raw.get("name", "tool")),
                        {
                            str(key): str(value)
                            for key, value in cast(dict[object, object], arguments).items()
                        }
                        if isinstance(arguments, dict)
                        else {},
                        int(raw["action_id"]) if raw.get("action_id") is not None else None,
                        int(raw["group_id"]) if raw.get("group_id") is not None else None,
                    )
                )
            elif event_type == "ReasoningSummary":
                self.reasoning(str(raw.get("text", "")))
            elif event_type == "ToolFinished":
                self.tool_finished(
                    ToolFinished(
                        str(raw.get("call_id", "")),
                        str(raw.get("name", "tool")),
                        str(raw.get("output", "")),
                        bool(raw.get("failed", False)),
                        float(raw.get("duration_seconds", 0.0)),
                        str(raw.get("outcome", "failed" if raw.get("failed") else "success")),
                        int(raw["action_id"]) if raw.get("action_id") is not None else None,
                        int(raw["group_id"]) if raw.get("group_id") is not None else None,
                    )
                )
            elif event_type == "TurnCancelled":
                self.cancel_pending(str(raw.get("reason", "Turn cancelled.")))
                self.complete()
            elif event_type in {"TurnFinished", "TurnFailed"}:
                self.complete()
        self.complete()
