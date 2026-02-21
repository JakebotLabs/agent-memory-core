"""Tests for the MemoryManager (unified interface)."""

import os
import pytest

from agent_memory_core.memory import MemoryManager


@pytest.fixture
def workspace(tmp_path):
    memory_md = tmp_path / "MEMORY.md"
    memory_md.write_text(
        "# Memory\n\n"
        "## Architecture\n\n**ChromaDB** for vectors. **NetworkX** for graphs.\n\n"
        "## Decisions\n\nDecided to use **sentence-transformers** for embeddings.\n"
    )
    (tmp_path / "memory").mkdir()
    return tmp_path


@pytest.fixture
def manager(workspace):
    return MemoryManager(
        str(workspace),
        vector_db_subdir="vector_db",
        graph_filename="graph.json",
    )


def test_index(manager):
    stats = manager.index()
    assert stats["chunks"] >= 2
    assert stats["nodes"] > 0
    assert stats["edges"] >= 0


def test_search(manager):
    manager.index()
    results = manager.search("chromadb vectors")
    assert len(results["vector_results"]) > 0


def test_search_formatted(manager):
    manager.index()
    text = manager.search_formatted("architecture")
    assert "Vector Search" in text
    assert "Knowledge Graph" in text


def test_search_formatted_compact(manager):
    manager.index()
    text = manager.search_formatted("architecture", compact=True)
    assert len(text) > 0
    assert "##" not in text  # no markdown headers in compact


def test_store(manager):
    manager.index()
    manager.store("Important: discovered new pattern", to_memory=True)
    content = manager.markdown.read_memory()
    assert "discovered new pattern" in content


def test_sync_status(manager):
    manager.index()
    status = manager.sync_status()
    assert "memory_md_hash" in status
    assert status["vector_chunks"] >= 2
    assert status["nodes"] > 0


def test_empty_workspace(tmp_path):
    mm = MemoryManager(str(tmp_path), vector_db_subdir="vdb", graph_filename="g.json")
    stats = mm.index()
    assert stats["chunks"] == 0


def test_classifier_integration(manager):
    results = manager.classifier.classify(
        "Decided to use PostgreSQL for the database architecture"
    )
    is_sig, reason, score = results
    assert is_sig is True
    assert score > 0.3
