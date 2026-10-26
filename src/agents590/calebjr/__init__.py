"""The course starter agent, independent of Harbor."""

import json
import re
from typing import Any, cast

from agents590.calebjr.tools import Tools
from agents590.contracts import RunContext
from agents590.provider import UNCOpenAISettings

TOOL_ARGS = {
    "ls": ["path"],
    "cat": ["path"],
    "write": ["path", "content"],
    "delete": ["path"],
    "mkdir": ["path"],
    "rmdir": ["path"],
    "move": ["src", "dst"],
    "copy": ["src", "dst"],
    "rename": ["src", "dst"],
    "search": ["query"],
    "help": ["tool_name"],
}


class CalebJR:
    """A model/tool loop using the shared standalone interface named CalebJR."""

    MAX_STEPS = 12
    MAX_INVALID = 2

    def provider_settings(self, context: RunContext) -> UNCOpenAISettings:
        """Prefer the per-run model to the environment's configured deployment."""
        return UNCOpenAISettings.from_env(model=context.model)

    def parse_tool_call(self, tool_call: str) -> tuple[str, list[str]]:
        """Validate and parse a tool call string into a tool name and arguments."""
        match = re.fullmatch(
            r"\[([a-z]+)\]\((.*)\)",
            tool_call,
            flags=re.DOTALL,
        )

        if not match:
            raise ValueError(f"Invalid tool call format: {tool_call}")

        name, raw_args = match.groups()

        single_arg = {"ls", "cat", "delete", "mkdir", "rmdir", "search", "help"}

        double_arg = {"write", "move", "copy", "rename"}

        if name not in single_arg and name not in double_arg:
            raise ValueError(f"Unknown tool: {name}")

        if name != "write" and any(char in raw_args for char in "()\r\n"):
            raise ValueError("Invalid tool arguments")

        if name in double_arg:
            args = [arg.strip() for arg in raw_args.split(",", 1)]

            if len(args) != 2:
                raise ValueError("Expected two arguments")
        else:
            args = [raw_args.strip()]

        if not args[0].strip():
            raise ValueError(f"Tool {name} requires a non-empty argument, got: {args}")

        if name in {"move", "copy", "rename"} and not args[1].strip():
            raise ValueError(f"Tool {name} requires a non-empty second argument, got: {args}")

        return name, args

    def make_tool_schemas(self) -> list[Any]:
        return [
            {
                "type": "function",
                "name": name,
                "description": getattr(Tools, name).__doc__ or name,
                "parameters": {
                    "type": "object",
                    "properties": {arg: {"type": "string"} for arg in args},
                    "required": args,
                    "additionalProperties": False,
                },
                "strict": True,
            }
            for name, args in TOOL_ARGS.items()
        ]

    async def run(self, instruction: str, context: RunContext) -> str:
        with open("src/agents590/calebjr/SYSTEM_PROMPT.md") as f:
            system_prompt = f.read()

        settings = self.provider_settings(context)
        client = settings.create_async_client()

        memory: list[Any] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instruction},
        ]

        tools = self.make_tool_schemas()
        last_call = None

        for _ in range(self.MAX_STEPS):
            try:
                response = await client.responses.create(
                    model=settings.model,
                    input=memory,
                    tools=tools,
                    tool_choice="auto",
                    parallel_tool_calls=False,
                    store=False,
                    background=False,
                )
            except Exception as e:
                return f"Model request failed: {e}"

            memory.extend(response.output)
            calls = [item for item in response.output if item.type == "function_call"]

            if not calls:
                answer = (response.output_text or "").strip()

                if re.search(
                    r"\[(ls|cat|write|delete|mkdir|rmdir|move|copy|rename|search|help)\]\(", answer
                ):
                    return "Model generated text-based tool calls instead of function calls."

                return answer.removeprefix("FINAL:").strip() or "No answer generated."

            if len(calls) != 1:
                return "Invalid number of tool calls."

            call = calls[0]
            print(f"[AGENT TOOL] {call.name}: {call.arguments}")

            current_call = (call.name, call.arguments)
            if current_call == last_call:
                return "Unable to complete: repeated tool call."

            last_call = current_call

            try:
                parsed_args: Any = json.loads(call.arguments)
                expected = TOOL_ARGS[call.name]

                if (
                    not isinstance(parsed_args, dict)
                    or len(cast(dict[str, Any], parsed_args)) != len(expected)
                    or not all(key in parsed_args for key in expected)
                    or any(not isinstance(parsed_args[k], str) for k in expected)
                ):
                    raise ValueError("Invalid tool arguments")

                args = cast(dict[str, str], parsed_args)
                result = getattr(Tools, call.name)(**args)

            except Exception as e:
                result = f"Tool error: {e}"

            print(f"[RESULT] {result}")

            memory.append(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": str(result),
                }
            )

        return "Unable to complete: maximum steps reached."
