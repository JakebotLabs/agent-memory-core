"""
L2 Vector Store — ChromaDB semantic search layer.

Indexes markdown chunks into ChromaDB with sentence-transformer embeddings.
Supports semantic search with relevance filtering.
"""

from __future__ import annotations

import hashlib
import os
import threading
from typing import Any, Dict, List, Optional  # Dict used for _models cache

__all__ = ["VectorStore"]

# Lazy-loaded globals — keyed by model name to support multiple models
_models: Dict[str, Any] = {}
_model_lock = threading.Lock()


def _get_model(model_name: str = "all-MiniLM-L6-v2"):
    """Lazy-load SentenceTransformer (heavy import), keyed by model name."""
    if model_name not in _models:
        with _model_lock:
            if model_name not in _models:
                from sentence_transformers import SentenceTransformer
                _models[model_name] = SentenceTransformer(model_name)
    return _models[model_name]


class VectorStore:
    """ChromaDB-backed vector store for semantic memory search."""

    COLLECTION_NAME = "memory_chunks"

    def __init__(
        self,
        db_path: str,
        model_name: str = "all-MiniLM-L6-v2",
        collection_name: Optional[str] = None,
    ) -> None:
        self.db_path = os.path.abspath(db_path)
        self.model_name = model_name
        self.collection_name = collection_name or self.COLLECTION_NAME
        self._lock = threading.Lock()
        self._client = None
        self._collection = None

    # ------------------------------------------------------------------
    # Lazy client / collection
    # ------------------------------------------------------------------

    def _ensure_client(self):
        if self._client is None:
            import chromadb
            os.makedirs(self.db_path, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self.db_path)
            self._collection = self._client.get_or_create_collection(name=self.collection_name)

    @property
    def collection(self):
        self._ensure_client()
        return self._collection

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """Upsert *chunks* into ChromaDB. Returns number of chunks indexed.

        Each chunk must have ``content`` (str) and ``metadata`` (dict).
        """
        if not chunks:
            return 0

        model = _get_model(self.model_name)

        ids = [f"mem_{i}" for i in range(len(chunks))]
        documents = [c["content"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        embeddings = model.encode(documents).tolist()

        with self._lock:
            self.collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            # Clean orphans
            existing = self.collection.count()
            if existing > len(chunks):
                orphan_ids = [f"mem_{i}" for i in range(len(chunks), existing)]
                self.collection.delete(ids=orphan_ids)

        return len(chunks)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        n_results: int = 5,
        max_distance: float = 2.0,
    ) -> List[Dict[str, Any]]:
        """Semantic search. Returns list of dicts with keys:
        ``content``, ``metadata``, ``distance``.
        """
        model = _get_model(self.model_name)
        query_embedding = model.encode(query).tolist()

        count = self.collection.count()
        if count == 0:
            return []

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, count),
            include=["documents", "metadatas", "distances"],
        )

        entries: List[Dict[str, Any]] = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                dist = results["distances"][0][i] if results["distances"] else None
                if dist is not None and dist > max_distance:
                    continue
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                entries.append({"content": doc, "metadata": meta, "distance": dist})

        return entries

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Number of chunks in the collection."""
        return self.collection.count()

    def reset(self) -> None:
        """Delete and recreate the collection."""
        self._ensure_client()
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(name=self.collection_name)
