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
    args = parser.parse_args(argv, namespace=Arguments())
    if args.list_agents:
        print("\n".join(agent_names()))
        return 0
    if not args.agent:
        parser.error("an agent name is required")
    if not args.interactive and (not args.instruction or not args.instruction.strip()):
        parser.error("-i/--instruction must contain an instruction")
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
        if args.interactive:
            from agents590.calebjr import CalebJR
            from agents590.calebjr.repl import run_repl

            if not isinstance(agent, CalebJR):
                raise ValueError(f"Agent {args.agent!r} does not support interactive sessions.")
            return asyncio.run(run_repl(agent, context, initial_instruction=args.instruction))
        assert args.instruction is not None
        output = asyncio.run(agent.run(args.instruction, context))
    except (ValueError, TypeError, NotImplementedError) as error:
        print(f"agents590: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    print(output)
    return 0
