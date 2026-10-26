"""Tests for :mod:`agents590.calebjr.tools`."""

from pathlib import Path

import pytest

from agents590.calebjr.tools import Tools


def test_ls_returns_sorted_directory_entries(tmp_path: Path) -> None:
    (tmp_path / "zeta.txt").write_text("z")
    (tmp_path / "alpha.txt").write_text("a")
    (tmp_path / "nested").mkdir()

    assert Tools.ls(str(tmp_path)).splitlines() == ["alpha.txt", "nested", "zeta.txt"]


def test_ls_reports_an_empty_directory(tmp_path: Path) -> None:
    assert Tools.ls(str(tmp_path)) == f"No entries found in directory: {tmp_path}"


def test_ls_reports_errors(tmp_path: Path) -> None:
    missing = tmp_path / "missing"

    assert Tools.ls(str(missing)).startswith(f"Error listing directory {missing}:")


def test_cat_reads_file_verbatim(tmp_path: Path) -> None:
    path = tmp_path / "message.txt"
    content = "Hello, world!\nUnicode: café ☕\n"
    path.write_text(content)

    assert Tools.cat(str(path)) == content


def test_cat_reports_errors(tmp_path: Path) -> None:
    missing = tmp_path / "missing.txt"

    assert Tools.cat(str(missing)).startswith(f"Error reading file {missing}:")


def test_write_creates_and_overwrites_a_file(tmp_path: Path) -> None:
    path = tmp_path / "message.txt"

    assert Tools.write(str(path), "first") == f"Successfully wrote to {path}"
    assert path.read_text() == "first"

    assert Tools.write(str(path), "replacement") == f"Successfully wrote to {path}"
    assert path.read_text() == "replacement"


def test_write_reports_errors(tmp_path: Path) -> None:
    path = tmp_path / "missing-parent" / "message.txt"

    assert Tools.write(str(path), "content").startswith(f"Error writing to file {path}:")
    assert not path.exists()


def test_delete_removes_a_file(tmp_path: Path) -> None:
    path = tmp_path / "message.txt"
    path.write_text("content")

    assert Tools.delete(str(path)) == f"Successfully deleted {path}"
    assert not path.exists()


def test_delete_reports_errors(tmp_path: Path) -> None:
    missing = tmp_path / "missing.txt"

    assert Tools.delete(str(missing)).startswith(f"Error deleting file {missing}:")


def test_mkdir_creates_nested_directories(tmp_path: Path) -> None:
    path = tmp_path / "parent" / "child"

    assert Tools.mkdir(str(path)) == f"Successfully created directory {path}"
    assert path.is_dir()

    # mkdir is intentionally idempotent.
    assert Tools.mkdir(str(path)) == f"Successfully created directory {path}"


def test_mkdir_reports_errors_when_a_parent_is_a_file(tmp_path: Path) -> None:
    parent = tmp_path / "file"
    parent.write_text("content")
    path = parent / "child"

    assert Tools.mkdir(str(path)).startswith(f"Error creating directory {path}:")


def test_rmdir_removes_an_empty_directory(tmp_path: Path) -> None:
    path = tmp_path / "empty"
    path.mkdir()

    assert Tools.rmdir(str(path)) == f"Successfully removed directory {path}"
    assert not path.exists()


def test_rmdir_does_not_remove_a_nonempty_directory(tmp_path: Path) -> None:
    path = tmp_path / "nonempty"
    path.mkdir()
    (path / "file.txt").write_text("content")

    assert Tools.rmdir(str(path)).startswith(f"Error removing directory {path}:")
    assert path.is_dir()


def test_rmdir_reports_a_missing_directory(tmp_path: Path) -> None:
    missing = tmp_path / "missing"

    assert Tools.rmdir(str(missing)).startswith(f"Error removing directory {missing}:")


@pytest.mark.parametrize("kind", ["file", "directory"])
def test_move_moves_files_and_directories(tmp_path: Path, kind: str) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    if kind == "file":
        source.write_text("content")
    else:
        source.mkdir()
        (source / "child.txt").write_text("content")

    assert Tools.move(str(source), str(destination)) == (
        f"Successfully moved {source} to {destination}"
    )
    assert not source.exists()
    assert destination.exists()
    if kind == "file":
        assert destination.read_text() == "content"
    else:
        assert (destination / "child.txt").read_text() == "content"


def test_move_reports_a_missing_source(tmp_path: Path) -> None:
    source = tmp_path / "missing"
    destination = tmp_path / "destination"

    assert Tools.move(str(source), str(destination)).startswith(
        f"Error moving {source} to {destination}:"
    )
    assert not destination.exists()


@pytest.mark.parametrize("kind", ["file", "directory"])
def test_copy_copies_files_and_directories(tmp_path: Path, kind: str) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    if kind == "file":
        source.write_text("content")
    else:
        source.mkdir()
        (source / "child.txt").write_text("content")

    assert Tools.copy(str(source), str(destination)) == (
        f"Successfully copied {source} to {destination}"
    )
    assert source.exists()
    assert destination.exists()
    if kind == "file":
        assert destination.read_text() == "content"
    else:
        assert (destination / "child.txt").read_text() == "content"


def test_copy_reports_a_missing_source(tmp_path: Path) -> None:
    source = tmp_path / "missing"
    destination = tmp_path / "destination"

    assert Tools.copy(str(source), str(destination)).startswith(
        f"Error copying {source} to {destination}:"
    )
    assert not destination.exists()


@pytest.mark.parametrize("kind", ["file", "directory"])
def test_rename_renames_files_and_directories(tmp_path: Path, kind: str) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    if kind == "file":
        source.write_text("content")
    else:
        source.mkdir()
        (source / "child.txt").write_text("content")

    assert Tools.rename(str(source), str(destination)) == (
        f"Successfully renamed {source} to {destination}"
    )
    assert not source.exists()
    assert destination.exists()


def test_rename_reports_a_missing_source(tmp_path: Path) -> None:
    source = tmp_path / "missing"
    destination = tmp_path / "destination"

    assert Tools.rename(str(source), str(destination)).startswith(
        f"Error renaming {source} to {destination}:"
    )
    assert not destination.exists()


def test_search_finds_files_and_directories_case_insensitively(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "Report.TXT").write_text("content")
    (tmp_path / "report-archive").mkdir()
    (tmp_path / "unrelated.txt").write_text("content")
    monkeypatch.chdir(tmp_path)

    assert set(Tools.search("REPORT").splitlines()) == {
        "./Report.TXT",
        "./report-archive",
    }


def test_search_recurses_but_skips_excluded_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    included = tmp_path / "src"
    included.mkdir()
    (included / "needle.txt").write_text("content")
    excluded = tmp_path / ".git"
    excluded.mkdir()
    (excluded / "needle-secret.txt").write_text("content")
    monkeypatch.chdir(tmp_path)

    assert Tools.search("needle") == "./src/needle.txt"


def test_search_reports_no_matches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    assert Tools.search("missing") == "No matches found for query: missing"


@pytest.mark.parametrize(
    ("tool_name", "usage"),
    [
        ("ls", "Usage: [ls](path)"),
        ("cat", "Usage: [cat](path)"),
        ("write", "Usage: [write](path, content)"),
        ("delete", "Usage: [delete](path)"),
        ("mkdir", "Usage: [mkdir](path)"),
        ("rmdir", "Usage: [rmdir](path)"),
        ("move", "Usage: [move](src, dst)"),
        ("copy", "Usage: [copy](src, dst)"),
        ("rename", "Usage: [rename](src, dst)"),
        ("search", "Usage: [search](query)"),
        ("help", "Usage: [help](tool_name)"),
    ],
)
def test_help_documents_every_tool(tool_name: str, usage: str) -> None:
    output = Tools.help(tool_name)

    assert output.splitlines()[0] == usage
    assert len(output.splitlines()) >= 2


@pytest.mark.parametrize("tool_name", ["unknown_tool", ""])
def test_help_reports_unknown_tools(tool_name: str) -> None:
    assert Tools.help(tool_name) == f"No help available for tool: {tool_name}"
