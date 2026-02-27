"""
Shared pytest fixtures for agent-memory-core tests.

Provides mocked filesystem, ChromaDB, and NetworkX fixtures.
All tests can run without external dependencies.
"""

import os
import json
import pytest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch


# ----------------------------------------------------------------
# Workspace fixtures
# ----------------------------------------------------------------

@pytest.fixture
def workspace(tmp_path):
    """Create a fully populated test workspace with markdown files."""
    # MEMORY.md
    memory_md = tmp_path / "MEMORY.md"
    memory_md.write_text(
        "# Memory\n\n"
        "## Architecture\n\n**ChromaDB** for vectors. **NetworkX** for graphs.\n\n"
        "## Decisions\n\nDecided to use **sentence-transformers** for embeddings.\n\n"
        "## Key Lessons\n\nAlways backup before slimming.\n\n"
        "## Recent Updates\n\nLast deployment: success.\n"
    )
    
    # memory/ directory with daily logs
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    
    # Recent daily log
    today = date.today()
    today_log = memory_dir / f"{today.isoformat()}.md"
    today_log.write_text(
        "## Daily Log\n\n"
        "Decided on major architecture change for the database.\n\n"
        "Fixed critical bug in the authentication system.\n"
    )
    
    # Yesterday's log
    yesterday = today - timedelta(days=1)
    yesterday_log = memory_dir / f"{yesterday.isoformat()}.md"
    yesterday_log.write_text(
        "## Daily Log\n\n"
        "Routine maintenance completed.\n\n"
        "Hello, just checking in.\n"
    )
    
    # reference/ directory
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir()
    reference_md = reference_dir / "api-docs.md"
    reference_md.write_text(
        "# API Documentation\n\n"
        "## Authentication\n\n**OAuth2** tokens required.\n"
    )
    
    return tmp_path


@pytest.fixture
def empty_workspace(tmp_path):
    """Create an empty workspace (no files)."""
    return tmp_path


@pytest.fixture
def workspace_no_memory(tmp_path):
    """Workspace with directories but no MEMORY.md."""
    (tmp_path / "memory").mkdir()
    (tmp_path / "reference").mkdir()
    return tmp_path


# ----------------------------------------------------------------
# MemoryManager fixtures
# ----------------------------------------------------------------

@pytest.fixture
def manager(workspace):
    """Create a MemoryManager with mocked heavy dependencies."""
    from agent_memory_core.memory import MemoryManager
    return MemoryManager(
        str(workspace),
        vector_db_subdir="vector_db",
        graph_filename="graph.json",
    )


@pytest.fixture
def manager_empty(empty_workspace):
    """MemoryManager on empty workspace."""
    from agent_memory_core.memory import MemoryManager
    return MemoryManager(
        str(empty_workspace),
        vector_db_subdir="vector_db",
        graph_filename="graph.json",
    )


# ----------------------------------------------------------------
# MarkdownStore fixtures
# ----------------------------------------------------------------

@pytest.fixture
def markdown_store(workspace):
    """Create a MarkdownStore on the test workspace."""
    from agent_memory_core.markdown_store import MarkdownStore
    return MarkdownStore(str(workspace))


@pytest.fixture
def markdown_store_empty(empty_workspace):
    """MarkdownStore on empty workspace."""
    from agent_memory_core.markdown_store import MarkdownStore
    return MarkdownStore(str(empty_workspace))


# ----------------------------------------------------------------
# VectorStore fixtures (mocked ChromaDB)
# ----------------------------------------------------------------

@pytest.fixture
def mock_chromadb():
    """Mock chromadb module to avoid heavy imports in unit tests."""
    mock_client = MagicMock()
    mock_collection = MagicMock()
    mock_collection.count.return_value = 0
    mock_collection.query.return_value = {
        "documents": [["Test document"]],
        "metadatas": [[{"source": "test.md", "section": "Test"}]],
        "distances": [[0.5]],
    }
    mock_client.get_or_create_collection.return_value = mock_collection
    
    with patch("chromadb.PersistentClient", return_value=mock_client):
        yield mock_client, mock_collection


@pytest.fixture
def vector_store(tmp_path, mock_chromadb):
    """VectorStore with mocked ChromaDB."""
    from agent_memory_core.vector_store import VectorStore
    return VectorStore(str(tmp_path / "vector_db"))


# ----------------------------------------------------------------
# GraphStore fixtures
# ----------------------------------------------------------------

@pytest.fixture
def graph_store(tmp_path):
    """Create a GraphStore with a fresh graph."""
    from agent_memory_core.graph_store import GraphStore
    return GraphStore(str(tmp_path / "graph.json"))


@pytest.fixture
def graph_store_with_data(tmp_path):
    """GraphStore pre-populated with test data."""
    from agent_memory_core.graph_store import GraphStore
    
    graph_path = tmp_path / "graph.json"
    gs = GraphStore(str(graph_path))
    
    # Add nodes
    gs.add_node("Architecture", node_type="section")
    gs.add_node("ChromaDB", node_type="concept")
    gs.add_node("NetworkX", node_type="concept")
    gs.add_node("Decisions", node_type="section")
    
    # Add edges
    gs.add_edge("Architecture", "ChromaDB", relation="uses")
    gs.add_edge("Architecture", "NetworkX", relation="uses")
    gs.add_edge("Decisions", "Architecture", relation="related_to")
    
    gs.save()
    return gs


# ----------------------------------------------------------------
# Classifier fixtures
# ----------------------------------------------------------------

@pytest.fixture
def classifier():
    """Default SignificanceClassifier."""
    from agent_memory_core.classifier import SignificanceClassifier
    return SignificanceClassifier()


@pytest.fixture
def classifier_with_llm():
    """Classifier with mock LLM function."""
    from agent_memory_core.classifier import SignificanceClassifier
    
    def mock_llm(prompt):
        return '{"significant": true, "reason": "mocked", "confidence": 0.85}'
    
    return SignificanceClassifier(llm_fn=mock_llm)


# ----------------------------------------------------------------
# MaintenanceRunner fixtures
# ----------------------------------------------------------------

@pytest.fixture
def maintenance_runner(markdown_store, classifier):
    """MaintenanceRunner with test fixtures."""
    from agent_memory_core.maintenance import MaintenanceRunner
    return MaintenanceRunner(markdown_store, classifier)


# ----------------------------------------------------------------
# Mock fixtures for integrations
# ----------------------------------------------------------------

@pytest.fixture
def mock_langchain():
    """Mock langchain-core imports for testing."""
    mock_base_tool = MagicMock()
    mock_base_model = MagicMock()
    mock_field = MagicMock()
    
    with patch.dict("sys.modules", {
        "langchain_core": MagicMock(),
        "langchain_core.tools": MagicMock(BaseTool=mock_base_tool),
        "pydantic": MagicMock(BaseModel=mock_base_model, Field=mock_field),
    }):
        yield


@pytest.fixture
def mock_crewai():
    """Mock crewai imports for testing."""
    mock_base_tool = MagicMock()
    
    with patch.dict("sys.modules", {
        "crewai": MagicMock(),
        "crewai.tools": MagicMock(BaseTool=mock_base_tool),
    }):
        yield


# ----------------------------------------------------------------
# Sync state fixtures
# ----------------------------------------------------------------

@pytest.fixture
def sync_state_file(workspace):
    """Create heartbeat state file for sync testing."""
    memory_dir = workspace / "memory"
    memory_dir.mkdir(exist_ok=True)
    
    state_file = memory_dir / "heartbeat-state.json"
    state_data = {
        "memorySync": {
            "memoryMdHash": "abc123",
            "chromadbChunks": 5,
            "nodes": 3,
            "edges": 2,
            "lastSync": "2024-01-01T00:00:00Z",
            "status": "synced"
        }
    }
    state_file.write_text(json.dumps(state_data))
    return state_file
