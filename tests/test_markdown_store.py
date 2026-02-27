"""Tests for the MarkdownStore (L1 layer)."""

import os
from datetime import date, timedelta
import pytest

from agent_memory_core.markdown_store import MarkdownStore


class TestFileDiscovery:
    """Test file discovery methods."""

    def test_gather_files(self, markdown_store):
        """Gathers all markdown files in priority order."""
        files = markdown_store.gather_files()
        assert len(files) >= 1
        # MEMORY.md should be first
        assert "MEMORY.md" in files[0]

    def test_gather_files_empty_workspace(self, markdown_store_empty):
        """Handles empty workspace gracefully."""
        files = markdown_store_empty.gather_files()
        assert files == []

    def test_gather_files_includes_reference(self, workspace):
        """Includes reference directory files."""
        store = MarkdownStore(str(workspace))
        files = store.gather_files()
        assert any("reference" in f for f in files)

    def test_gather_files_includes_memory_dir(self, workspace):
        """Includes memory/ directory files (daily logs)."""
        store = MarkdownStore(str(workspace))
        files = store.gather_files()
        assert any("memory" in f and "MEMORY.md" not in f for f in files)


class TestParsing:
    """Test markdown parsing methods."""

    def test_parse_markdown_basic(self, workspace):
        """Parses markdown into chunks by headers."""
        store = MarkdownStore(str(workspace))
        chunks = store.parse_markdown(str(workspace / "MEMORY.md"), str(workspace))
        assert len(chunks) >= 2
        assert chunks[0]["metadata"]["source"] == "MEMORY.md"

    def test_parse_markdown_nonexistent(self, workspace):
        """Returns empty list for nonexistent file."""
        store = MarkdownStore(str(workspace))
        chunks = store.parse_markdown("/nonexistent/file.md", str(workspace))
        assert chunks == []

    def test_parse_markdown_preserves_content(self, workspace):
        """Content is preserved in chunks."""
        store = MarkdownStore(str(workspace))
        chunks = store.parse_markdown(str(workspace / "MEMORY.md"), str(workspace))
        
        # Find Architecture section
        arch_chunks = [c for c in chunks if c["metadata"]["section"] == "Architecture"]
        if arch_chunks:
            assert "ChromaDB" in arch_chunks[0]["content"]

    def test_parse_markdown_metadata(self, workspace):
        """Chunks have correct metadata."""
        store = MarkdownStore(str(workspace))
        chunks = store.parse_markdown(str(workspace / "MEMORY.md"), str(workspace))
        
        for chunk in chunks:
            assert "source" in chunk["metadata"]
            assert "section" in chunk["metadata"]
            assert "content" in chunk

    def test_get_all_chunks(self, markdown_store):
        """Gets chunks from all discovered files."""
        chunks = markdown_store.get_all_chunks()
        assert len(chunks) >= 4  # MEMORY.md + daily + reference

    def test_get_all_chunks_empty_workspace(self, markdown_store_empty):
        """Returns empty list for empty workspace."""
        chunks = markdown_store_empty.get_all_chunks()
        assert chunks == []


class TestMemoryMdOperations:
    """Test MEMORY.md file operations."""

    def test_read_memory(self, markdown_store):
        """Reads MEMORY.md content."""
        content = markdown_store.read_memory()
        assert "# Memory" in content
        assert "Architecture" in content

    def test_read_memory_missing(self, markdown_store_empty):
        """Returns empty string when MEMORY.md missing."""
        content = markdown_store_empty.read_memory()
        assert content == ""

    def test_write_memory(self, markdown_store):
        """Overwrites MEMORY.md content."""
        new_content = "# New Memory\n\nFresh start.\n"
        markdown_store.write_memory(new_content)
        
        assert markdown_store.read_memory() == new_content

    def test_write_memory_creates_dirs(self, tmp_path):
        """Creates parent directories if needed."""
        nested_path = tmp_path / "deep" / "nested"
        nested_path.mkdir(parents=True)
        
        store = MarkdownStore(str(nested_path))
        store.write_memory("# Test\n")
        
        assert store.read_memory() == "# Test\n"

    def test_append_to_memory(self, markdown_store):
        """Appends text to MEMORY.md."""
        original = markdown_store.read_memory()
        
        markdown_store.append_to_memory("\n## New Section\n\nAppended content.\n")
        
        new_content = markdown_store.read_memory()
        assert original in new_content
        assert "Appended content" in new_content


class TestDailyLogOperations:
    """Test daily log file operations."""

    def test_today_log_path(self, markdown_store):
        """Returns correct path for today's log."""
        path = markdown_store.today_log_path()
        today = date.today().isoformat()
        assert today in path
        assert path.endswith(".md")

    def test_append_daily_creates_file(self, markdown_store):
        """Creates daily log file if it doesn't exist."""
        # Remove today's log if it exists
        path = markdown_store.today_log_path()
        if os.path.exists(path):
            os.remove(path)
        
        result_path = markdown_store.append_daily("Test entry")
        
        assert os.path.exists(result_path)
        with open(result_path) as f:
            assert "Test entry" in f.read()

    def test_append_daily_custom_date(self, markdown_store):
        """Can append to a specific date's log."""
        custom_date = date.today() - timedelta(days=5)
        
        path = markdown_store.append_daily("Past entry", log_date=custom_date)
        
        assert custom_date.isoformat() in path

    def test_list_daily_logs(self, markdown_store):
        """Lists recent daily log files."""
        logs = markdown_store.list_daily_logs()
        
        # Should find at least our test fixtures
        assert len(logs) >= 1
        assert all(log.endswith(".md") for log in logs)

    def test_list_daily_logs_limit(self, workspace):
        """Respects last_n limit."""
        store = MarkdownStore(str(workspace))
        
        # Create multiple logs
        for i in range(10):
            d = date.today() - timedelta(days=i)
            store.append_daily(f"Log {i}", log_date=d)
        
        logs = store.list_daily_logs(last_n=3)
        assert len(logs) == 3

    def test_list_daily_logs_sorted(self, workspace):
        """Returns logs in reverse chronological order."""
        store = MarkdownStore(str(workspace))
        
        # Create multiple logs
        for i in range(5):
            d = date.today() - timedelta(days=i)
            store.append_daily(f"Log {i}", log_date=d)
        
        logs = store.list_daily_logs()
        # First should be most recent (today)
        assert date.today().isoformat() in logs[0]


class TestSectionHelpers:
    """Test section-related helper methods."""

    def test_get_sections(self, markdown_store):
        """Extracts section headers from MEMORY.md."""
        sections = markdown_store.get_sections()
        
        assert len(sections) >= 2
        assert "Architecture" in sections
        assert "Decisions" in sections

    def test_get_sections_empty_file(self, markdown_store_empty):
        """Returns empty list for missing MEMORY.md."""
        sections = markdown_store_empty.get_sections()
        assert sections == []

    def test_find_insertion_point_recent_updates(self, workspace):
        """Finds insertion point at Recent Updates section."""
        store = MarkdownStore(str(workspace))
        content = store.read_memory()
        
        pos = store.find_insertion_point(content, "Recent Updates")
        
        # Position should be valid
        assert 0 < pos <= len(content)

    def test_find_insertion_point_fallback(self, workspace):
        """Falls back to Key Lessons or end of file."""
        store = MarkdownStore(str(workspace))
        
        # Content without Recent Updates
        content = "# Memory\n\n## Architecture\n\nNotes.\n\n## Key Lessons\n\nLessons here.\n"
        
        pos = store.find_insertion_point(content, "Nonexistent Section")
        
        # Should find Key Lessons
        assert pos > 0

    def test_find_insertion_point_end_of_file(self, workspace):
        """Falls back to end of file if no markers found."""
        store = MarkdownStore(str(workspace))
        
        content = "# Memory\n\n## Simple\n\nJust some content.\n"
        
        pos = store.find_insertion_point(content, "Nonexistent")
        
        # Should be at end
        assert pos == len(content)


class TestThreadSafety:
    """Test thread safety of operations."""

    def test_write_memory_thread_safe(self, markdown_store):
        """Write operations use lock."""
        import threading
        
        results = []
        
        def writer(content, idx):
            markdown_store.write_memory(f"Content from thread {idx}: {content}")
            results.append(idx)
        
        threads = []
        for i in range(5):
            t = threading.Thread(target=writer, args=(f"data{i}", i))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # All threads completed
        assert len(results) == 5
        # Final content should be from one of the threads
        content = markdown_store.read_memory()
        assert "Content from thread" in content

    def test_append_daily_thread_safe(self, markdown_store):
        """Append operations use lock."""
        import threading
        
        def appender(text):
            markdown_store.append_daily(text)
        
        threads = []
        for i in range(10):
            t = threading.Thread(target=appender, args=(f"\nEntry {i}\n",))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # Read the daily log
        path = markdown_store.today_log_path()
        with open(path) as f:
            content = f.read()
        
        # All entries should be present
        for i in range(10):
            assert f"Entry {i}" in content
