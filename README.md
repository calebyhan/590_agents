# 590 agents

Standalone Python 3.12 CLI agents for COMP 590. Each implementation lives in its
own package under `src/agents590/`; all share one CLI and typed interface. This
project owns model configuration and agent logic. The sibling `evals/` project
owns Harbor tasks and adapters. Harbor is not a dependency of this package.

## Setup and usage

```sh
uv sync --locked
uv run agents590 --list
uv run agents590 krisj_echo -i "Hello from the CLI"
uv run python -m agents590 krisj_echo --instruction "The same shared entry point"
```

`-i` / `--instruction` passes one instruction. `--cwd` selects an existing working
directory, `--model` overrides the agent's model, and `--env-file` explicitly loads
configuration without overriding variables already exported in the shell.

The `krisj_echo` agent returns the instruction without tools or API requests. The
`krisj_starter` agent is the migrated course scaffold: its model/tool loop intentionally
raises `NotImplementedError` until implemented. Its settings already support
per-run model overrides through `RunContext.model`.

For use outside this checkout:

```sh
uv tool install --editable /absolute/path/to/590_agents
agents590 krisj_echo -i "An instruction from another project"
```

Changes to implementation source are picked up by new invocations. Reinstall or
run `uv sync` after changing entry-point registrations in `pyproject.toml`.

### CalebJR interactive activity

```sh
uv run agents590 calebjr --interactive --cwd /path/to/project
```

Interactive terminals open a full-screen interface that retains the complete session
scrollback in memory instead of relying on the terminal's limited history. Use
`PgUp`/`PgDn` to scroll, `Alt-<`/`Alt->` to jump to the beginning or end, and `F6` to
move focus between output and input. `Alt-Enter` inserts a newline and Enter submits.
Set `CALEBJR_TUI=0` to use the classic terminal REPL; redirected and non-TTY sessions
always use plain output.

Interactive sessions use compact activity by default: routine tool calls are grouped
into one turn summary, while failures and confirmation prompts remain visible. Change
the detail level at startup with `--quiet`, `--compact`, `--detailed`, or `--debug`
(`--verbose` remains a compatibility alias for debug), or at any time inside the REPL:

```text
/activity quiet
/activity compact
/activity detailed
/activity debug
/activity show 3
```

Run `/activity` to show the current mode. `/activity show [N]` expands recent grouped
actions and `/activity show G3` selects one group without exposing file contents or
edit text. Actions and groups receive stable IDs such as `A8` and `G3`;
`/activity output A8` shows a redacted 2,000-character
preview and `/activity output A8 --full` explicitly requests the retained output
(up to the session's 100,000-character tool-output limit). Activity is reconstructed
from saved events when a session resumes. `/status` also reports the mode.

The selected mode is saved for real interactive terminals under the XDG config
directory. `CALEBJR_ACTIVITY_MODE=quiet|compact|detailed|debug` overrides it. Set
`CALEBJR_REDUCED_MOTION=1` to replace animated status displays with static updates.
In an interactive terminal, compact mode shows a semantic live phase such as
`Inspecting project`, `Updating files`, or `Running checks`, along with elapsed time.
Provider-supplied reasoning summaries appear as durable, de-duplicated clumps; they
are bounded per turn, associated with activity history, and are never raw internal
reasoning traces. Failures classify timeouts, denials, cancellation, and other errors;
Ctrl-C preserves completed and in-flight activity for inspection while rolling back
incomplete model context. `/transcript compact`, `/transcript detailed`, and
`/transcript markdown` provide bounded, redacted transcript views and clean Markdown
export output.

## Add an agent

1. Create a package such as `src/agents590/my_agent/`.
2. Implement a no-argument class with `async run(instruction: str, context:
   RunContext) -> str`. Honor `context.cwd` when executing tools and use
   `context.model` for the explicit model override.
3. Register `my-agent = "agents590.my_agent:MyAgent"` under
   `[project.entry-points."agents590.agents"]`, then run `uv sync`.
4. Add pytest tests and run `make check` and `make test`.

`agents590 my-agent -i "..."` and `python -m agents590 my-agent -i "..."`
then use the same implementation. Entry points also allow separately installed
packages to register additional agents. Duplicate names produce an error.

## UNC Azure configuration

```sh
cp .env.example .env
# Fill in your UNC subscription key and an enabled deployment alias.
uv run agents590 krisj_starter --env-file .env --model openai/YOUR_DEPLOYMENT -i "..."
```

The coding loop remains a scaffold. To explicitly make one billable connectivity
request through the migrated smoke command, run `make smoke`.

| Setting | Azure alias (takes precedence) | Meaning |
| --- | --- | --- |
| `OPENAI_API_KEY` | `AZURE_OPENAI_API_KEY` | UNC subscription key |
| `OPENAI_MODEL` | `AZURE_OPENAI_DEPLOYMENT` | Default deployment alias |
| `OPENAI_ENDPOINT` | `AZURE_OPENAI_ENDPOINT` | HTTPS origin; defaults to `https://azureaiapi.cloud.unc.edu` |

The explicit `--model` wins over environment defaults. The provider removes an
optional `openai/` prefix, preserves the deployment alias, and sends standard
OpenAI SDK requests to the UNC `/openai/v1/` gateway with bearer authentication.
Requests use `store=False` and `background=False`. Blank explicit settings fail;
public OpenAI endpoints are rejected. `OPENAI_BASE_URL` is ignored because the
client sets its base URL explicitly. Keep `.env` private; it is excluded from
Git and built wheels.

This configuration was migrated from `evals/src/coding_agent/provider.py`, whose
UNC routing was checked against LearnWithAI revision
`5c239db1b27abdb28c1969fbb1046ee0d700ea53`. Tests use mocked HTTP transports.

## Development

```sh
make check  # Ruff lint/format and Pyright static type checking
make test   # pytest, with no live model requests
```

VS Code recommends Python, Pylance, and Ruff. Pylance and the Pyright CLI share
the strict `[tool.pyright]` configuration in `pyproject.toml`; VS Code uses
`.venv/bin/python`. Python source is annotated and the distribution includes
`py.typed`. `make clean` removes generated caches and build output while keeping
the virtual environment and `.env`.

## Evaluate through Harbor

The sibling `evals/` repository supplies the generic installed-agent adapter:

```sh
cd ../evals
make agent AGENT=krisj_echo
make agent AGENT=my-agent MODEL=openai/YOUR_DEPLOYMENT ENV_FILE=../590_agents/.env
```

The second command requires that agent to be registered and implemented and may
make billable model requests. The adapter uploads a source snapshot containing
the package metadata, README, and `src/`, installs its locked runtime dependencies in an isolated sandbox
virtual environment, and invokes `python -m agents590 <agent> -i <instruction>`.
Credentials are read from the explicit environment file on the host and passed
as process environment values; the file itself is not uploaded. New agents need
no new Harbor adapter.
