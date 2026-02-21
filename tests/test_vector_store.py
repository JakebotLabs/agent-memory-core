"""Tests for the VectorStore (L2 layer)."""

import pytest

from agent_memory_core.vector_store import VectorStore


@pytest.fixture
def store(tmp_path):
    return VectorStore(str(tmp_path / "chroma_db"))


@pytest.fixture
def sample_chunks():
    return [
        {"content": "ChromaDB is used for vector search and semantic retrieval.", "metadata": {"source": "test.md", "section": "Architecture"}},
        {"content": "NetworkX provides graph traversal for relationship queries.", "metadata": {"source": "test.md", "section": "Architecture"}},
        {"content": "Decided to use PostgreSQL for the main database backend.", "metadata": {"source": "test.md", "section": "Decisions"}},
        {"content": "Daily maintenance runs auto-promotion every 24 hours.", "metadata": {"source": "test.md", "section": "Operations"}},
    ]


def test_index_and_count(store, sample_chunks):
    n = store.index_chunks(sample_chunks)
    assert n == 4
    assert store.count() == 4


def test_index_empty(store):
    assert store.index_chunks([]) == 0


def test_search_relevance(store, sample_chunks):
    store.index_chunks(sample_chunks)
    results = store.search("vector database semantic search")
    assert len(results) > 0
    # ChromaDB chunk should be most relevant
    assert "ChromaDB" in results[0]["content"]


def test_search_max_distance(store, sample_chunks):
    store.index_chunks(sample_chunks)
    # Very tight distance should filter results
    results = store.search("completely unrelated quantum physics topic", max_distance=0.1)
    assert len(results) == 0


def test_search_empty_store(store):
    results = store.search("anything")
    assert results == []


def test_search_n_results(store, sample_chunks):
    store.index_chunks(sample_chunks)
    results = store.search("database", n_results=2)
    assert len(results) <= 2


def test_reindex_cleans_orphans(store, sample_chunks):
    store.index_chunks(sample_chunks)
    assert store.count() == 4

    # Re-index with fewer chunks
    smaller = sample_chunks[:2]
    store.index_chunks(smaller)
    assert store.count() == 2


def test_reset(store, sample_chunks):
    store.index_chunks(sample_chunks)
    assert store.count() == 4
    store.reset()
    assert store.count() == 0


def test_search_returns_metadata(store, sample_chunks):
    store.index_chunks(sample_chunks)
    results = store.search("PostgreSQL")
    assert len(results) > 0
    assert "metadata" in results[0]
    assert "section" in results[0]["metadata"]


def test_search_returns_distance(store, sample_chunks):
    store.index_chunks(sample_chunks)
    results = store.search("graph traversal")
    assert len(results) > 0
    assert "distance" in results[0]
    assert isinstance(results[0]["distance"], float)
