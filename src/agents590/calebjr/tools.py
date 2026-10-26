"""Module for tools used by the calebjr agent."""


class Tools:
    """A class containing tools for the calebjr agent."""

    @staticmethod
    def help(tool_name: str) -> str:
        """Get help on how to use the specified tool."""
        help_texts = {
            "ls": "Usage: [ls](path)\nList the contents of a directory at the given path.",
            "cat": "Usage: [cat](path)\nDisplay the contents of a file at the given path.",
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
