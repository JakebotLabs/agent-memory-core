"""Tests for the MarkdownStore (L1 layer)."""

import os
import tempfile

import pytest

from agent_memory_core.markdown_store import MarkdownStore


@pytest.fixture
def workspace(tmp_path):
    """Create a temporary workspace with sample files."""
    memory_md = tmp_path / "MEMORY.md"
    memory_md.write_text(
        "# Memory\n\n"
        "## Projects\n\nBuilding agent-memory-core package.\n\n"
        "## Key Lessons\n\nAlways re-index after edits.\n"
    )

    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    (memory_dir / "2026-02-21.md").write_text(
        "# 2026-02-21\n\n## Morning\n\nWorked on packaging.\n\n## Evening\n\nShipped it.\n"
    )

    ref_dir = tmp_path / "reference"
    ref_dir.mkdir()
    (ref_dir / "people.md").write_text("# People\n\n## Jake\n\nFounder and developer.\n")

    return tmp_path


def test_gather_files(workspace):
    store = MarkdownStore(str(workspace))
    files = store.gather_files()
    assert len(files) == 3
    assert any("MEMORY.md" in f for f in files)


def test_parse_markdown(workspace):
    store = MarkdownStore(str(workspace))
    chunks = store.parse_markdown(str(workspace / "MEMORY.md"), str(workspace))
    assert len(chunks) >= 2
    assert chunks[0]["metadata"]["source"] == "MEMORY.md"


def test_get_all_chunks(workspace):
    store = MarkdownStore(str(workspace))
    chunks = store.get_all_chunks()
    assert len(chunks) >= 4  # MEMORY.md + daily + reference


def test_read_write_memory(workspace):
    store = MarkdownStore(str(workspace))
    content = store.read_memory()
    assert "Projects" in content

    store.write_memory("# New Content\n\nFresh start.\n")
    assert "Fresh start" in store.read_memory()


def test_append_daily(workspace):
    store = MarkdownStore(str(workspace))
    path = store.append_daily("\n## New Entry\n\nSomething happened.\n")
    assert os.path.exists(path)
    with open(path) as f:
        assert "Something happened" in f.read()


def test_list_daily_logs(workspace):
    store = MarkdownStore(str(workspace))
    logs = store.list_daily_logs()
    assert len(logs) == 1


def test_get_sections(workspace):
    store = MarkdownStore(str(workspace))
    sections = store.get_sections()
    assert "Projects" in sections
    assert "Key Lessons" in sections


def test_find_insertion_point(workspace):
    store = MarkdownStore(str(workspace))
    content = store.read_memory()
    pos = store.find_insertion_point(content)
    assert 0 < pos <= len(content)
