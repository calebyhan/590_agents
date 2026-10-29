"""One CLI for all registered agent packages."""

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path

from dotenv import load_dotenv

from agents590.contracts import RunContext
from agents590.registry import agent_names, create_agent


class Arguments(argparse.Namespace):
    """Typed values populated by argparse."""

    agent: str | None
    instruction: str | None
    cwd: Path
    model: str | None
    env_file: Path | None
    list_agents: bool
    interactive: bool
    quiet: bool
    compact: bool
    detailed: bool
    debug: bool
    verbose: bool
    resume: str | None
    confirm_destructive: bool
    continue_session: bool
    request_timeout: float | None


def main(argv: Sequence[str] | None = None) -> int:
    """Select an agent and run a one-shot instruction or interactive session."""
    parser = argparse.ArgumentParser(prog="agents590")
    parser.add_argument("agent", nargs="?", help="Registered agent name; see --list")
    parser.add_argument("-i", "--instruction", help="Instruction to execute")
    parser.add_argument("--cwd", type=Path, default=Path.cwd(), help="Agent working directory")
    parser.add_argument("-m", "--model", help="Per-run model/deployment override")
    parser.add_argument("--env-file", type=Path, help="Explicit environment file to load")
    parser.add_argument("--list", dest="list_agents", action="store_true", help="List agents")
    parser.add_argument(
        "--interactive", action="store_true", help="Start an interactive agent session"
    )
    parser.add_argument("--resume", metavar="ID_OR_PATH", help="Resume a saved interactive session")
    parser.add_argument(
        "--continue",
        dest="continue_session",
        action="store_true",
        help="Continue interactively after running -i/--instruction",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        metavar="SECONDS",
        help="Interactive model request timeout (default: 120)",
    )
    parser.add_argument(
        "--confirm-destructive",
        action="store_true",
        help="Confirm destructive shell commands after independent safety review",
    )
    detail = parser.add_mutually_exclusive_group()
    detail.add_argument("--quiet", action="store_true", help="Show only answers and errors")
    detail.add_argument("--compact", action="store_true", help="Show grouped interactive activity")
    detail.add_argument(
        "--detailed", action="store_true", help="Show each tool call with bounded output"
    )
    detail.add_argument("--debug", action="store_true", help="Show complete interactive events")
    detail.add_argument(
        "--verbose",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv, namespace=Arguments())
    if args.list_agents:
        print("\n".join(agent_names()))
        return 0
    if not args.agent:
        parser.error("an agent name is required")
    interactive = args.interactive or args.continue_session
    if args.continue_session and (not args.instruction or not args.instruction.strip()):
        parser.error("--continue requires -i/--instruction")
    if not interactive and (not args.instruction or not args.instruction.strip()):
        parser.error("-i/--instruction must contain an instruction")
    if args.resume and not interactive:
        parser.error("--resume requires --interactive")
    if args.confirm_destructive and not interactive:
        parser.error("--confirm-destructive requires --interactive")
    if args.request_timeout is not None and not interactive:
        parser.error("--request-timeout requires --interactive or --continue")
    if args.request_timeout is not None and args.request_timeout <= 0:
        parser.error("--request-timeout must be positive")
    if not interactive and (
        args.quiet or args.compact or args.detailed or args.debug or args.verbose
    ):
        parser.error("activity detail flags require --interactive")
    cwd = args.cwd.expanduser().resolve()
    if not cwd.is_dir():
        parser.error("--cwd must name an existing directory")
    if args.env_file is not None:
        env_file = args.env_file.expanduser().resolve()
        if not env_file.is_file():
            parser.error("--env-file must name an existing file")
        load_dotenv(env_file, override=False)
    try:
        agent = create_agent(args.agent)
        context = RunContext(cwd=cwd, model=args.model)
        if interactive:
            from agents590.calebjr import CalebJR
            from agents590.calebjr.render import RenderMode
            from agents590.calebjr.repl import run_repl

            if not isinstance(agent, CalebJR):
                raise ValueError(f"Agent {args.agent!r} does not support interactive sessions.")
            render_mode: RenderMode | None = None
            if args.quiet:
                render_mode = "quiet"
            elif args.compact:
                render_mode = "compact"
            elif args.detailed:
                render_mode = "default"
            elif args.debug or args.verbose:
                render_mode = "verbose"
            return asyncio.run(
                run_repl(
                    agent,
                    context,
                    initial_instruction=args.instruction,
                    render_mode=render_mode,
                    resume=args.resume,
                    confirm_destructive=args.confirm_destructive,
                    request_timeout=args.request_timeout or 120.0,
                )
            )
        assert args.instruction is not None
        output = asyncio.run(agent.run(args.instruction, context))
    except (ValueError, TypeError, NotImplementedError) as error:
        print(f"agents590: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    print(output)
    return 0
