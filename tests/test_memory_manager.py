"""Tests for the MemoryManager (unified interface)."""

import os
import json
import pytest
from datetime import datetime

from agent_memory_core.memory import MemoryManager


class TestMemoryManagerInit:
    """Test MemoryManager initialization."""

    def test_init_creates_stores(self, workspace):
        """Init creates all three stores."""
        mm = MemoryManager(
            str(workspace),
            vector_db_subdir="vdb",
            graph_filename="graph.json",
        )
        
        assert mm.markdown is not None
        assert mm.vectors is not None
        assert mm.graph is not None

    def test_init_creates_classifier(self, workspace):
        """Init creates default classifier."""
        mm = MemoryManager(str(workspace), vector_db_subdir="vdb", graph_filename="g.json")
        
        assert mm.classifier is not None

    def test_init_creates_maintenance(self, workspace):
        """Init creates maintenance runner."""
        mm = MemoryManager(str(workspace), vector_db_subdir="vdb", graph_filename="g.json")
        
        assert mm.maintenance is not None

    def test_init_custom_llm_fn(self, workspace):
        """Can pass custom LLM function."""
        def my_llm(prompt):
            return '{"significant": true, "reason": "test", "confidence": 0.9}'
        
        mm = MemoryManager(
            str(workspace),
            vector_db_subdir="vdb",
            graph_filename="g.json",
            llm_fn=my_llm,
        )
        
        assert mm.classifier.llm_fn is my_llm

    def test_init_custom_model(self, workspace):
        """Can specify custom embedding model."""
        mm = MemoryManager(
            str(workspace),
            vector_db_subdir="vdb",
            graph_filename="g.json",
            model_name="custom-model",
        )
        
        assert mm.vectors.model_name == "custom-model"


class TestIndexing:
    """Test index method."""

    def test_index_returns_stats(self, manager):
        """Index returns statistics dict."""
        stats = manager.index()
        
        assert "chunks" in stats
        assert "nodes" in stats
        assert "edges" in stats

    def test_index_populates_vectors(self, manager):
        """Index populates vector store."""
        manager.index()
        
        assert manager.vectors.count() >= 2

    def test_index_populates_graph(self, manager):
        """Index populates graph store."""
        manager.index()
        
        stats = manager.graph.stats()
        assert stats["nodes"] > 0

    def test_index_empty_workspace(self, manager_empty):
        """Index handles empty workspace."""
        stats = manager_empty.index()
        
        assert stats["chunks"] == 0
        assert stats["nodes"] == 0
        assert stats["edges"] == 0


class TestSearch:
    """Test search methods."""

    def test_search_returns_structure(self, manager):
        """Search returns dict with vector and graph results."""
        manager.index()
        results = manager.search("architecture")
        
        assert "vector_results" in results
        assert "graph_results" in results

    def test_search_vector_results(self, manager):
        """Vector search finds relevant results."""
        manager.index()
        results = manager.search("chromadb vectors")
        
        assert len(results["vector_results"]) > 0

    def test_search_graph_results(self, manager):
        """Graph search finds relevant nodes."""
        manager.index()
        results = manager.search("architecture", include_graph=True)
        
        # May or may not find graph results depending on query
        assert "graph_results" in results

    def test_search_excludes_graph(self, manager):
        """Can exclude graph results."""
        manager.index()
        results = manager.search("test", include_graph=False)
        
        assert results["graph_results"] == []

    def test_search_respects_n_results(self, manager):
        """Respects n_results parameter."""
        manager.index()
        results = manager.search("test", n_results=1)
        
        assert len(results["vector_results"]) <= 1


class TestSearchFormatted:
    """Test search_formatted method."""

    def test_search_formatted_returns_string(self, manager):
        """Returns formatted markdown string."""
        manager.index()
        text = manager.search_formatted("architecture")
        
        assert isinstance(text, str)
        assert len(text) > 0

    def test_search_formatted_includes_sections(self, manager):
        """Includes Vector Search and Knowledge Graph sections."""
        manager.index()
        text = manager.search_formatted("architecture")
        
        assert "Vector Search" in text
        assert "Knowledge Graph" in text

    def test_search_formatted_compact(self, manager):
        """Compact mode removes headers."""
        manager.index()
        text = manager.search_formatted("architecture", compact=True)
        
        # No markdown headers in compact mode
        assert "##" not in text

    def test_search_formatted_no_results(self, manager_empty):
        """Handles no results gracefully."""
        manager_empty.index()
        text = manager_empty.search_formatted("anything")
        
        assert "No relevant" in text or len(text) > 0


class TestStore:
    """Test store method."""

    def test_store_appends_to_daily(self, manager):
        """Store appends to daily log."""
        manager.index()
        manager.store("Important: new discovery")
        
        path = manager.markdown.today_log_path()
        with open(path) as f:
            content = f.read()
        
        assert "new discovery" in content

    def test_store_to_memory_md(self, manager):
        """Store with to_memory=True updates MEMORY.md."""
        manager.index()
        manager.store("Critical: must remember", to_memory=True)
        
        content = manager.markdown.read_memory()
        assert "must remember" in content

    def test_store_reindexes(self, manager):
        """Store triggers re-indexing."""
        manager.index()
        initial_count = manager.vectors.count()
        
        manager.store("\n## New Section\n\nNew content for indexing.\n")
        
        # Count may increase
        assert manager.vectors.count() >= initial_count


class TestPromote:
    """Test promote method."""

    def test_promote_returns_stats(self, manager):
        """Promote returns statistics."""
        manager.index()
        result = manager.promote(days_back=7, dry_run=True)
        
        assert "candidates_found" in result
        assert "promotions_made" in result
        assert "dry_run" in result

    def test_promote_dry_run_no_changes(self, manager):
        """Dry run doesn't modify files."""
        manager.index()
        original = manager.markdown.read_memory()
        
        manager.promote(days_back=7, dry_run=True)
        
        assert manager.markdown.read_memory() == original

    def test_promote_reindexes_on_success(self, manager):
        """Successful promotion triggers re-index."""
        manager.index()
        
        result = manager.promote(days_back=7, min_confidence=0.1, dry_run=False)
        
        # If promotions were made, re-indexing happened
        if result["promotions_made"] > 0:
            assert manager.vectors.count() > 0


class TestSyncStatus:
    """Test sync status methods."""

    def test_sync_status_structure(self, manager):
        """Sync status has expected fields."""
        manager.index()
        status = manager.sync_status()
        
        assert "status" in status
        assert "memory_md_hash" in status
        assert "vector_chunks" in status
        assert "nodes" in status
        assert "edges" in status
        assert "files_found" in status
        assert "timestamp" in status

    def test_sync_status_after_index(self, manager):
        """Status reflects indexed state."""
        manager.index()
        status = manager.sync_status()
        
        assert status["vector_chunks"] >= 2
        assert status["nodes"] > 0

    def test_sync_status_detects_out_of_sync(self, manager, sync_state_file):
        """Detects when memory is out of sync."""
        manager.index()
        
        # Modify the hash in state file
        with open(sync_state_file, "r") as f:
            state = json.load(f)
        state["memorySync"]["memoryMdHash"] = "wrong_hash"
        with open(sync_state_file, "w") as f:
            json.dump(state, f)
        
        status = manager.sync_status()
        
        assert status["status"] == "OUT_OF_SYNC"


class TestMemoryHash:
    """Test memory hash calculation."""

    def test_memory_hash_consistent(self, manager):
        """Same content produces same hash."""
        h1 = manager._memory_hash()
        h2 = manager._memory_hash()
        
        assert h1 == h2

    def test_memory_hash_changes_on_edit(self, manager):
        """Hash changes when content changes."""
        h1 = manager._memory_hash()
        
        manager.markdown.append_to_memory("\nNew content.\n")
        
        h2 = manager._memory_hash()
        assert h1 != h2

    def test_memory_hash_empty(self, manager_empty):
        """Empty MEMORY.md returns 'empty'."""
        h = manager_empty._memory_hash()
        
        assert h == "empty"


class TestStateFile:
    """Test state file persistence."""

    def test_save_sync_state(self, manager):
        """Saves sync state to file."""
        manager.index()
        
        path = manager._state_file_path()
        assert os.path.exists(path)

    def test_load_stored_hash(self, manager):
        """Loads stored hash from state file."""
        manager.index()
        
        stored = manager._load_stored_hash()
        
        assert stored is not None
        assert len(stored) == 12  # md5[:12]

    def test_load_stored_hash_missing_file(self, manager_empty):
        """Returns None when state file missing."""
        result = manager_empty._load_stored_hash()
        
        assert result is None


class TestClassifierIntegration:
    """Test classifier integration."""

    def test_classifier_on_manager(self, manager):
        """Manager's classifier works."""
        is_sig, reason, score = manager.classifier.classify(
            "Decided to use PostgreSQL for the database architecture"
        )
        
        assert is_sig is True
        assert score > 0.3


class TestThreadSafety:
    """Test thread safety."""

    def test_index_thread_safe(self, workspace):
        """Index can be called from multiple threads."""
        import threading
        
        mm = MemoryManager(
            str(workspace),
            vector_db_subdir="vdb",
            graph_filename="g.json",
        )
        
        results = []
        
        def indexer():
            stats = mm.index()
            results.append(stats)
        
        threads = []
        for _ in range(3):
            t = threading.Thread(target=indexer)
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        assert len(results) == 3
        for r in results:
            assert "chunks" in r


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_search_before_index(self, manager):
        """Search works (returns empty) before indexing."""
        results = manager.search("test")
        
        assert results["vector_results"] == []

    def test_promote_no_daily_logs(self, workspace):
        """Promote handles missing daily logs."""
        # Clear daily logs
        import shutil
        shutil.rmtree(workspace / "memory")
        (workspace / "memory").mkdir()
        
        mm = MemoryManager(str(workspace), vector_db_subdir="vdb", graph_filename="g.json")
        mm.index()
        
        result = mm.promote(days_back=7, dry_run=True)
        
        assert result["candidates_found"] == 0
