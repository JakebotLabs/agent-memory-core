"""Tests for VectorStore (L2 layer) — ChromaDB semantic search."""

import pytest
from unittest.mock import MagicMock, patch


class TestVectorStoreInit:
    """Test VectorStore initialization."""

    def test_init_sets_paths(self, tmp_path):
        """Init stores correct paths."""
        from agent_memory_core.vector_store import VectorStore
        
        db_path = str(tmp_path / "vector_db")
        vs = VectorStore(db_path)
        
        assert vs.db_path == db_path
        assert vs.model_name == "all-MiniLM-L6-v2"
        assert vs.collection_name == "memory_chunks"

    def test_init_custom_model(self, tmp_path):
        """Can specify custom embedding model."""
        from agent_memory_core.vector_store import VectorStore
        
        vs = VectorStore(
            str(tmp_path / "vdb"),
            model_name="custom-model"
        )
        
        assert vs.model_name == "custom-model"

    def test_init_custom_collection(self, tmp_path):
        """Can specify custom collection name."""
        from agent_memory_core.vector_store import VectorStore
        
        vs = VectorStore(
            str(tmp_path / "vdb"),
            collection_name="my_collection"
        )
        
        assert vs.collection_name == "my_collection"


class TestIndexing:
    """Test chunk indexing methods."""

    def test_index_chunks_empty(self, tmp_path):
        """Empty chunks list returns 0."""
        from agent_memory_core.vector_store import VectorStore
        
        vs = VectorStore(str(tmp_path / "vdb"))
        
        result = vs.index_chunks([])
        
        assert result == 0

    def test_index_chunks_returns_count(self, tmp_path):
        """Index returns number of chunks indexed."""
        from agent_memory_core.vector_store import VectorStore
        
        vs = VectorStore(str(tmp_path / "vdb"))
        
        chunks = [
            {"content": "First chunk about architecture", "metadata": {"source": "test.md", "section": "Intro"}},
            {"content": "Second chunk about design", "metadata": {"source": "test.md", "section": "Design"}},
        ]
        
        result = vs.index_chunks(chunks)
        
        assert result == 2

    def test_index_chunks_creates_embeddings(self, tmp_path):
        """Chunks are embedded and stored."""
        from agent_memory_core.vector_store import VectorStore
        
        vs = VectorStore(str(tmp_path / "vdb"))
        
        chunks = [
            {"content": "Test content for embedding", "metadata": {"source": "test.md", "section": "Test"}},
        ]
        
        vs.index_chunks(chunks)
        
        # Collection should have data
        assert vs.count() == 1

    def test_index_chunks_upserts(self, tmp_path):
        """Repeated indexing updates, not duplicates."""
        from agent_memory_core.vector_store import VectorStore
        
        vs = VectorStore(str(tmp_path / "vdb"))
        
        chunks = [
            {"content": "Original content", "metadata": {"source": "test.md", "section": "Test"}},
        ]
        
        vs.index_chunks(chunks)
        assert vs.count() == 1
        
        # Re-index
        chunks[0]["content"] = "Updated content"
        vs.index_chunks(chunks)
        
        # Still just 1 chunk (upserted, not duplicated)
        assert vs.count() == 1


class TestSearch:
    """Test semantic search methods."""

    def test_search_empty_collection(self, tmp_path):
        """Search on empty collection returns empty list."""
        from agent_memory_core.vector_store import VectorStore
        
        vs = VectorStore(str(tmp_path / "vdb"))
        
        results = vs.search("test query")
        
        assert results == []

    def test_search_returns_results(self, tmp_path):
        """Search returns matching results."""
        from agent_memory_core.vector_store import VectorStore
        
        vs = VectorStore(str(tmp_path / "vdb"))
        
        chunks = [
            {"content": "ChromaDB is a vector database", "metadata": {"source": "test.md", "section": "ChromaDB"}},
            {"content": "NetworkX is a graph library", "metadata": {"source": "test.md", "section": "NetworkX"}},
        ]
        vs.index_chunks(chunks)
        
        results = vs.search("vector database")
        
        assert len(results) > 0
        # First result should be about ChromaDB
        assert "ChromaDB" in results[0]["content"] or "vector" in results[0]["content"]

    def test_search_respects_n_results(self, tmp_path):
        """Search returns at most n_results."""
        from agent_memory_core.vector_store import VectorStore
        
        vs = VectorStore(str(tmp_path / "vdb"))
        
        chunks = [
            {"content": f"Chunk {i} about topic", "metadata": {"source": "test.md", "section": f"S{i}"}}
            for i in range(10)
        ]
        vs.index_chunks(chunks)
        
        results = vs.search("topic", n_results=3)
        
        assert len(results) <= 3

    def test_search_filters_by_distance(self, tmp_path):
        """Results beyond max_distance are filtered."""
        from agent_memory_core.vector_store import VectorStore
        
        vs = VectorStore(str(tmp_path / "vdb"))
        
        chunks = [
            {"content": "Completely unrelated content XYZ123", "metadata": {"source": "test.md", "section": "Unrelated"}},
        ]
        vs.index_chunks(chunks)
        
        # Very strict distance should filter out irrelevant results
        results = vs.search("specific technical term not in chunks", max_distance=0.1)
        
        # May or may not return results depending on embedding similarity
        # But all results should be within distance
        for r in results:
            if r.get("distance") is not None:
                assert r["distance"] <= 0.1

    def test_search_result_structure(self, tmp_path):
        """Results have expected structure."""
        from agent_memory_core.vector_store import VectorStore
        
        vs = VectorStore(str(tmp_path / "vdb"))
        
        chunks = [
            {"content": "Test content", "metadata": {"source": "test.md", "section": "Test"}},
        ]
        vs.index_chunks(chunks)
        
        results = vs.search("test")
        
        assert len(results) > 0
        result = results[0]
        assert "content" in result
        assert "metadata" in result
        assert "distance" in result


class TestUtilities:
    """Test utility methods."""

    def test_count_empty(self, tmp_path):
        """Count returns 0 for empty collection."""
        from agent_memory_core.vector_store import VectorStore
        
        vs = VectorStore(str(tmp_path / "vdb"))
        
        assert vs.count() == 0

    def test_count_after_index(self, tmp_path):
        """Count reflects indexed chunks."""
        from agent_memory_core.vector_store import VectorStore
        
        vs = VectorStore(str(tmp_path / "vdb"))
        
        chunks = [
            {"content": f"Chunk {i}", "metadata": {"source": "test.md", "section": f"S{i}"}}
            for i in range(5)
        ]
        vs.index_chunks(chunks)
        
        assert vs.count() == 5

    def test_reset_clears_collection(self, tmp_path):
        """Reset deletes and recreates collection."""
        from agent_memory_core.vector_store import VectorStore
        
        vs = VectorStore(str(tmp_path / "vdb"))
        
        chunks = [
            {"content": "Some content", "metadata": {"source": "test.md", "section": "Test"}},
        ]
        vs.index_chunks(chunks)
        assert vs.count() == 1
        
        vs.reset()
        
        assert vs.count() == 0


class TestLazyLoading:
    """Test lazy loading of models and clients."""

    def test_model_lazy_loaded(self, tmp_path):
        """Embedding model is lazy-loaded."""
        from agent_memory_core.vector_store import VectorStore, _models
        
        # Clear model cache
        _models.clear()
        
        vs = VectorStore(str(tmp_path / "vdb"))
        
        # Model not loaded yet
        assert "all-MiniLM-L6-v2" not in _models
        
        # After indexing, model is loaded
        vs.index_chunks([{"content": "test", "metadata": {"source": "t", "section": "t"}}])
        assert "all-MiniLM-L6-v2" in _models

    def test_client_lazy_loaded(self, tmp_path):
        """ChromaDB client is lazy-loaded."""
        from agent_memory_core.vector_store import VectorStore
        
        vs = VectorStore(str(tmp_path / "vdb"))
        
        # Client not loaded yet
        assert vs._client is None
        
        # After accessing collection, client is loaded
        _ = vs.collection
        assert vs._client is not None


class TestThreadSafety:
    """Test thread safety."""

    def test_index_thread_safe(self, tmp_path):
        """Indexing from multiple threads is safe."""
        from agent_memory_core.vector_store import VectorStore
        import threading
        
        vs = VectorStore(str(tmp_path / "vdb"))
        
        def indexer(idx):
            chunks = [{"content": f"Thread {idx} content", "metadata": {"source": f"t{idx}.md", "section": "Test"}}]
            vs.index_chunks(chunks)
        
        threads = []
        for i in range(5):
            t = threading.Thread(target=indexer, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # Should have indexed from all threads (last one wins due to upsert on mem_0)
        assert vs.count() >= 1


class TestOrphanCleanup:
    """Test cleanup of orphan documents."""

    def test_removes_orphans_on_reindex(self, tmp_path):
        """Orphan documents are removed when re-indexing fewer chunks."""
        from agent_memory_core.vector_store import VectorStore
        
        vs = VectorStore(str(tmp_path / "vdb"))
        
        # Index 5 chunks
        chunks = [
            {"content": f"Chunk {i}", "metadata": {"source": "test.md", "section": f"S{i}"}}
            for i in range(5)
        ]
        vs.index_chunks(chunks)
        assert vs.count() == 5
        
        # Re-index with only 2 chunks
        chunks = chunks[:2]
        vs.index_chunks(chunks)
        
        # Orphans should be removed
        assert vs.count() == 2
