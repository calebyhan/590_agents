"""Module for tools used by the calebjr agent."""

import asyncio
import os
import signal


class Tools:
    """A class containing tools for the calebjr agent."""

    @staticmethod
    def help(tool_name: str) -> str:
        """Get help on how to use the specified tool."""
        help_texts = {
            "ls": "Usage: [ls](path)\nList the contents of a directory at the given path.",
            "cat": "Usage: [cat](path)\nDisplay the contents of a file at the given path.",
            "edit": """Usage: [edit](path, old_text, new_text)
            Replace one exact, unique text occurrence in a file.""",
            "write": """Usage: [write](path, content)
            Write the given content to a file at the given path.""",
            "delete": "Usage: [delete](path)\nDelete the file at the given path.",
            "mkdir": "Usage: [mkdir](path)\nCreate a new directory at the given path.",
            "rmdir": "Usage: [rmdir](path)\nRemove the directory at the given path.",
            "move": "Usage: [move](src, dst)\nMove a file or directory from src to dst.",
            "copy": "Usage: [copy](src, dst)\nCopy a file or directory from src to dst.",
            "rename": "Usage: [rename](src, dst)\nRename a file or directory from src to dst.",
            "search": """Usage: [search](query)
            Search for files or directories that match the given query.""",
            "shell": """Usage: [shell](command)
            Execute a shell command in the run's working directory.""",
            "help": "Usage: [help](tool_name)\nGet help on how to use the specified tool.",
        }
        return help_texts.get(tool_name, f"No help available for tool: {tool_name}")

    @staticmethod
    def ls(path: str) -> str:
        """List the contents of a directory at the given path."""
        import os

        try:
            entries = sorted(os.listdir(path))
            return "\n".join(entries) if entries else f"No entries found in directory: {path}"
        except Exception as e:
            return f"Error listing directory {path}: {e}"

    @staticmethod
    def cat(path: str) -> str:
        """Display the contents of a file at the given path."""
        try:
            with open(path) as f:
                return f.read()
        except Exception as e:
            return f"Error reading file {path}: {e}"

    @staticmethod
    def edit(path: str, old_text: str, new_text: str) -> str:
        """Replace one exact, unique text occurrence in a file."""
        try:
            if not old_text:
                raise ValueError("old_text must not be empty")

            with open(path) as f:
                content = f.read()

            occurrences = content.count(old_text)
            if occurrences == 0:
                raise ValueError("old_text was not found")
            if occurrences > 1:
                raise ValueError(f"old_text is not unique; found {occurrences} occurrences")

            with open(path, "w") as f:
                f.write(content.replace(old_text, new_text, 1))
            return f"Successfully edited {path}"
        except Exception as e:
            return f"Error editing file {path}: {e}"

    @staticmethod
    def write(path: str, content: str) -> str:
        """Write the given content to a file at the given path."""
        try:
            with open(path, "w") as f:
                f.write(content)
            return f"Successfully wrote to {path}"
        except Exception as e:
            return f"Error writing to file {path}: {e}"

    @staticmethod
    def delete(path: str) -> str:
        """Delete the file at the given path."""
        import os

        try:
            os.remove(path)
            return f"Successfully deleted {path}"
        except Exception as e:
            return f"Error deleting file {path}: {e}"

    @staticmethod
    def mkdir(path: str) -> str:
        """Create a new directory at the given path."""
        import os

        try:
            os.makedirs(path, exist_ok=True)
            return f"Successfully created directory {path}"
        except Exception as e:
            return f"Error creating directory {path}: {e}"

    @staticmethod
    def rmdir(path: str) -> str:
        """Remove the directory at the given path."""
        import os

        try:
            os.rmdir(path)
            return f"Successfully removed directory {path}"
        except Exception as e:
            return f"Error removing directory {path}: {e}"

    @staticmethod
    def move(src: str, dst: str) -> str:
        """Move a file or directory from src to dst."""
        import shutil

        try:
            shutil.move(src, dst)
            return f"Successfully moved {src} to {dst}"
        except Exception as e:
            return f"Error moving {src} to {dst}: {e}"

    @staticmethod
    def copy(src: str, dst: str) -> str:
        """Copy a file or directory from src to dst."""
        import os
        import shutil

        try:
            if os.path.isdir(src):
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
            return f"Successfully copied {src} to {dst}"
        except Exception as e:
            return f"Error copying {src} to {dst}: {e}"

    @staticmethod
    def rename(src: str, dst: str) -> str:
        """Rename a file or directory from src to dst."""
        import os

        try:
            os.rename(src, dst)
            return f"Successfully renamed {src} to {dst}"
        except Exception as e:
            return f"Error renaming {src} to {dst}: {e}"

    @staticmethod
    def search(query: str) -> str:
        """Search for files or directories that match the given query."""
        import os

        matches: list[str] = []
        excluded = {".git", "__pycache__", "venv", ".venv", "node_modules", ".DS_Store"}

        for root, dirs, files in os.walk("."):
            dirs[:] = [d for d in dirs if d not in excluded]

            for name in files + dirs:
                if query.lower() in name.lower() and name not in excluded:
                    matches.append(os.path.join(root, name))
                    if len(matches) >= 100:
                        return (
                            "\n".join(matches)
                            + """
                            Note: More than 100 matches found; only showing the first 100."""
                        )
        return "\n".join(matches) if matches else f"No matches found for query: {query}"

    @staticmethod
    async def shell_async(command: str, *, cwd: str, timeout: float = 120.0) -> str:
        """Execute a cancellable shell command in the supplied working directory."""
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=os.name != "nt",
            )

            async def stop() -> None:
                try:
                    if os.name == "nt":
                        process.terminate()
                    else:
                        os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    return
                try:
                    await asyncio.wait_for(process.wait(), timeout=1.0)
                except TimeoutError:
                    try:
                        if os.name == "nt":
                            process.kill()
                        else:
                            os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    await process.wait()

            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
            except TimeoutError:
                await stop()
                stdout, stderr = await process.communicate()
                output = "\n".join(
                    part.decode(errors="replace").strip()
                    for part in (stdout, stderr)
                    if part.strip()
                )
                return f"Command timed out after {timeout:g} seconds." + (
                    f"\n{output}" if output else ""
                )
            except asyncio.CancelledError:
                await stop()
                raise
        except asyncio.CancelledError:
            raise
        except Exception as error:
            return f"Error executing command: {error}"

        output = "\n".join(
            part.decode(errors="replace").strip() for part in (stdout, stderr) if part.strip()
        )
        if process.returncode == 0:
            return output or "Command completed successfully."
        return f"Command failed with exit code {process.returncode}." + (
            f"\n{output}" if output else ""
        )

    @staticmethod
    def shell(command: str, *, cwd: str) -> str:
        """Execute a shell command synchronously in the supplied working directory."""
        import subprocess

        try:
            completed = subprocess.run(
                command,
                cwd=cwd,
                shell=True,
                text=True,
                capture_output=True,
                timeout=120,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            output = "\n".join(
                part.decode(errors="replace").strip()
                for part in (error.stdout or "", error.stderr or "")
                if part
            )
            return "Command timed out after 120 seconds." + (f"\n{output}" if output else "")
        except Exception as error:
            return f"Error executing command: {error}"

        output = "\n".join(
            part.strip() for part in (completed.stdout, completed.stderr) if part.strip()
        )
        if completed.returncode == 0:
            return output or "Command completed successfully."
        return f"Command failed with exit code {completed.returncode}." + (
            f"\n{output}" if output else ""
        )
