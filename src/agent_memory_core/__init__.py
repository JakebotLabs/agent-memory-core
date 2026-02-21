"""
agent-memory-core: Three-layer persistent memory for AI agents.

Layers:
    L1 (Markdown)  - Human-readable curated knowledge (MEMORY.md + daily logs)
    L2 (Vector)    - ChromaDB semantic search with sentence-transformers
    L3 (Graph)     - NetworkX knowledge graph for relationship traversal

Usage:
    from agent_memory_core import MemoryManager

    mm = MemoryManager("/path/to/workspace")
    mm.index()
    results = mm.search("what happened yesterday?")
"""

__version__ = "0.1.0"

from agent_memory_core.memory import MemoryManager
from agent_memory_core.markdown_store import MarkdownStore
from agent_memory_core.vector_store import VectorStore
from agent_memory_core.graph_store import GraphStore
from agent_memory_core.classifier import SignificanceClassifier
from agent_memory_core.maintenance import MaintenanceRunner

__all__ = [
    "MemoryManager",
    "MarkdownStore",
    "VectorStore",
    "GraphStore",
    "SignificanceClassifier",
    "MaintenanceRunner",
]
