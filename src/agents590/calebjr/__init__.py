"""CalebJR one-shot adapter and interactive session implementation."""

from agents590.calebjr.session import (
    RISKY_SHELL_PATTERNS,
    TOOL_ARGS,
    CalebJRSession,
)
from agents590.contracts import RunContext
from agents590.provider import UNCOpenAISettings


class CalebJR:
    """Compatibility adapter for the shared one-shot agent contract."""

    MAX_STEPS = CalebJRSession.MAX_STEPS
    MAX_INVALID = CalebJRSession.MAX_INVALID
    make_tool_schemas = staticmethod(CalebJRSession.make_tool_schemas)
    requires_safety_review = staticmethod(CalebJRSession.requires_safety_review)
    tool_failed = staticmethod(CalebJRSession.tool_failed)

    def provider_settings(self, context: RunContext) -> UNCOpenAISettings:
        """Prefer the per-run model to the environment's configured deployment."""
        return UNCOpenAISettings.from_env(model=context.model)

    def create_session(
        self, context: RunContext, *, request_timeout: float = 30.0
    ) -> CalebJRSession:
        """Create a stateful session; one-shot callers retain the 30-second default."""
        return CalebJRSession(
            settings=self.provider_settings(context),
            context=context,
            request_timeout=request_timeout,
        )

    async def run(self, instruction: str, context: RunContext) -> str:
        async with self.create_session(context) as session:
            return (await session.run_turn(instruction)).text


__all__ = ["CalebJR", "CalebJRSession", "RISKY_SHELL_PATTERNS", "TOOL_ARGS"]
