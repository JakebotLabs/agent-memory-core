"""
MemoryManager — Unified interface to the three-layer memory system.

This is the main entry point for using agent-memory-core.

    mm = MemoryManager("/path/to/workspace")
    mm.index()
    results = mm.search("query")
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from agent_memory_core.classifier import SignificanceClassifier
from agent_memory_core.graph_store import GraphStore
from agent_memory_core.maintenance import MaintenanceRunner
from agent_memory_core.markdown_store import MarkdownStore
from agent_memory_core.vector_store import VectorStore

__all__ = ["MemoryManager"]


class MemoryManager:
    """Unified three-layer memory manager.

    Args:
        base_dir: Root directory where memory files live.
        vector_db_subdir: Subdirectory for ChromaDB storage (relative to base_dir).
        graph_filename: Name of the graph JSON file (in vector_db_subdir).
        model_name: Sentence-transformer model for embeddings.
        llm_fn: Optional LLM callable for significance classification.
    """

    def __init__(
        self,
        base_dir: str,
        vector_db_subdir: str = "vector_memory/chroma_db",
        graph_filename: str = "memory_graph.json",
        model_name: str = "all-MiniLM-L6-v2",
        llm_fn: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.base_dir = os.path.abspath(base_dir)

        # L1
        self.markdown = MarkdownStore(self.base_dir)

        # L2
        db_path = os.path.join(self.base_dir, vector_db_subdir)
        self.vectors = VectorStore(db_path, model_name=model_name)

        # L3
        graph_dir = os.path.dirname(os.path.join(self.base_dir, vector_db_subdir))
        graph_path = os.path.join(graph_dir, graph_filename)
        self.graph = GraphStore(graph_path)

        # Classifier & maintenance
        self.classifier = SignificanceClassifier(llm_fn=llm_fn)
        self.maintenance = MaintenanceRunner(self.markdown, self.classifier)

        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def index(self) -> Dict[str, Any]:
        """Parse all markdown → index into vectors → rebuild graph.

        Returns stats dict.
        """
        chunks = self.markdown.get_all_chunks()
        if not chunks:
            return {"chunks": 0, "nodes": 0, "edges": 0}

        n_indexed = self.vectors.index_chunks(chunks)
        self.graph.build_from_chunks(chunks)
        stats = self.graph.stats()
        self._save_sync_state()

        return {"chunks": n_indexed, **stats}

    def search(
        self,
        query: str,
        n_results: int = 5,
        max_distance: float = 2.0,
        include_graph: bool = True,
    ) -> Dict[str, Any]:
        """Search across L2 (vectors) and L3 (graph).

        Returns dict with ``vector_results`` and ``graph_results``.
        """
        vector_results = self.vectors.search(query, n_results=n_results, max_distance=max_distance)
        graph_results = self.graph.query(query) if include_graph else []
        return {"vector_results": vector_results, "graph_results": graph_results}

    def search_formatted(
        self,
        query: str,
        n_results: int = 5,
        compact: bool = False,
    ) -> str:
        """Search and return formatted markdown context (for system prompt injection)."""
        results = self.search(query, n_results=n_results)

        if compact:
            lines: List[str] = []
            for r in results["vector_results"]:
                section = r["metadata"].get("section", "?")
                snippet = r["content"][:150]
                lines.append(f"[{section}] {snippet}")
            for r in results["graph_results"]:
                nbs = ", ".join(n["node"] for n in r.get("neighbors", [])[:3])
                lines.append(f"Graph: {r['node']} → {nbs}")
            return "\n".join(lines) if lines else "No relevant memories found."

        lines = [
            "## 🧠 Auto-Retrieved Memory Context",
            f"**Query:** {query}",
            "",
        ]

        # Vector
        vc = self.vectors.count()
        lines.append(f"### Vector Search ({vc} chunks indexed)")
        if results["vector_results"]:
            for r in results["vector_results"]:
                section = r["metadata"].get("section", "?")
                dist = r.get("distance")
                dist_str = f"(dist: {dist:.3f})" if dist is not None else ""
                snippet = r["content"][:300] + "..." if len(r["content"]) > 300 else r["content"]
                lines.append(f"- **[{section}]** {dist_str}")
                lines.append(f"  {snippet}")
        else:
            lines.append("No relevant results found.")
        lines.append("")

        # Graph
        gs = self.graph.stats()
        lines.append(f"### Knowledge Graph ({gs['nodes']} nodes, {gs['edges']} edges)")
        if results["graph_results"]:
            for r in results["graph_results"]:
                nbs = ", ".join(
                    f"{n['node']} ({n['relation']})" for n in r.get("neighbors", [])[:5]
                )
                lines.append(f"- **{r['node']}** → {nbs or 'no neighbors'}")
        else:
            lines.append("No matching graph nodes found.")
        lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Store helpers
    # ------------------------------------------------------------------

    def store(self, text: str, to_memory: bool = False) -> None:
        """Append *text* to daily log (and optionally MEMORY.md), then re-index."""
        self.markdown.append_daily(f"\n{text}\n")
        if to_memory:
            self.markdown.append_to_memory(f"\n{text}\n")
        self.index()

    def promote(self, days_back: int = 3, min_confidence: float = 0.7, dry_run: bool = False) -> Dict[str, Any]:
        """Run auto-promotion from daily logs → MEMORY.md, then re-index."""
        result = self.maintenance.promote(days_back, min_confidence, dry_run)
        if not dry_run and result["promotions_made"] > 0:
            self.index()
        return result

    # ------------------------------------------------------------------
    # Sync status
    # ------------------------------------------------------------------

    def _memory_hash(self) -> str:
        content = self.markdown.read_memory()
        return hashlib.md5(content.encode()).hexdigest()[:12] if content else "empty"

    def sync_status(self) -> Dict[str, Any]:
        """Return sync health information with out-of-sync detection."""
        current_hash = self._memory_hash()
        stored_hash = self._load_stored_hash()
        
        if stored_hash and stored_hash != current_hash:
            status = "OUT_OF_SYNC"
        elif stored_hash:
            status = "synced"
        else:
            status = "unknown"

        return {
            "status": status,
            "memory_md_hash": current_hash,
            "stored_hash": stored_hash,
            "vector_chunks": self.vectors.count(),
            **self.graph.stats(),
            "files_found": len(self.markdown.gather_files()),
            "timestamp": datetime.now().isoformat(),
        }

    def _state_file_path(self) -> str:
        return os.path.join(self.base_dir, "memory", "heartbeat-state.json")

    def _load_stored_hash(self) -> Optional[str]:
        path = self._state_file_path()
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r") as f:
                data = json.load(f)
            return data.get("memorySync", {}).get("memoryMdHash")
        except (json.JSONDecodeError, OSError):
            return None

    def _save_sync_state(self) -> None:
        """Persist sync state for out-of-sync detection between runs."""
        path = self._state_file_path()
        try:
            try:
                with open(path, "r") as f:
                    state = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                state = {}

            state["memorySync"] = {
                "memoryMdHash": self._memory_hash(),
                "chromadbChunks": self.vectors.count(),
                **self.graph.stats(),
                "lastSync": datetime.now().astimezone().isoformat(),
                "status": "synced",
            }
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                json.dump(state, f, indent=2)
        except OSError:
            pass  # Non-critical
