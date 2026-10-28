"""Structured, presentation-independent events emitted by CalebJR sessions."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class TurnStarted:
    instruction: str


@dataclass(frozen=True)
class ToolStarted:
    call_id: str
    name: str
    arguments: dict[str, str]


@dataclass(frozen=True)
class ToolFinished:
    call_id: str
    name: str
    output: str
    failed: bool
    duration_seconds: float


@dataclass(frozen=True)
class SafetyReviewStarted:
    command: str


@dataclass(frozen=True)
class SafetyReviewFinished:
    command: str
    approved: bool
    decision: str


@dataclass(frozen=True)
class DestructiveConfirmationStarted:
    command: str


@dataclass(frozen=True)
class DestructiveConfirmationFinished:
    command: str
    approved: bool


@dataclass(frozen=True)
class Progress:
    text: str


@dataclass(frozen=True)
class ContextCompacted:
    summarized_items: int
    characters_before: int
    characters_after: int


@dataclass(frozen=True)
class ModelTextDelta:
    text: str


@dataclass(frozen=True)
class ReasoningSummary:
    """Provider-supplied reasoning summary, never a raw internal trace."""

    text: str


@dataclass(frozen=True)
class TurnFinished:
    text: str


@dataclass(frozen=True)
class TurnFailed:
    error: str


type SessionEvent = (
    TurnStarted
    | ToolStarted
    | ToolFinished
    | SafetyReviewStarted
    | SafetyReviewFinished
    | DestructiveConfirmationStarted
    | DestructiveConfirmationFinished
    | Progress
    | ContextCompacted
    | ModelTextDelta
    | ReasoningSummary
    | TurnFinished
    | TurnFailed
)


class EventSink(Protocol):
    async def __call__(self, event: SessionEvent) -> None: ...


async def discard_event(event: SessionEvent) -> None:
    """Default sink used by embedded and one-shot callers."""


@dataclass
class CollectingEventSink:
    """A simple sink useful to embedders and tests."""

    events: list[SessionEvent] = field(default_factory=lambda: [])

    async def __call__(self, event: SessionEvent) -> None:
        self.events.append(event)


type EventCallback = Callable[[SessionEvent], Awaitable[None]]
