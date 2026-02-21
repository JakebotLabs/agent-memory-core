"""
L3 Graph Store — NetworkX knowledge graph layer.

Builds and queries a directed graph of concepts extracted from memory chunks.
Nodes represent sections and bolded concepts; edges represent relationships.
"""

from __future__ import annotations

import json
import os
import re
import threading
from typing import Any, Dict, List, Optional, Set

import networkx as nx

__all__ = ["GraphStore"]


class GraphStore:
    """NetworkX-backed knowledge graph for relationship traversal."""

    RELATION_MAP: Dict[str, str] = {
        "depends on": "depends_on",
        "blocked by": "blocked_by",
        "enables": "enables",
        "improves": "improves",
        "replaces": "replaces",
        "uses": "uses",
        "alternative to": "alternative_to",
        "related to": "related_to",
        "see": "refer_to",
        "reference": "refer_to",
    }

    def __init__(self, graph_path: str) -> None:
        self.graph_path = os.path.abspath(graph_path)
        self._lock = threading.Lock()
        self.graph: nx.DiGraph = nx.DiGraph()
        self.load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load graph from JSON (node-link format)."""
        if os.path.exists(self.graph_path):
            with open(self.graph_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            # NetworkX 3.4+ changed node_link_graph API
            try:
                self.graph = nx.node_link_graph(data, edges="links")
            except TypeError:
                self.graph = nx.node_link_graph(data)

    def save(self) -> None:
        """Persist graph to JSON."""
        with self._lock:
            os.makedirs(os.path.dirname(self.graph_path) or ".", exist_ok=True)
            try:
                data = nx.node_link_data(self.graph, edges="links")
            except TypeError:
                data = nx.node_link_data(self.graph)
            with open(self.graph_path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_node(self, name: str, node_type: str = "concept", **meta: Any) -> None:
        self.graph.add_node(name, type=node_type, **meta)

    def add_edge(self, source: str, target: str, relation: str = "related_to") -> None:
        if source in self.graph and target in self.graph:
            self.graph.add_edge(source, target, relation=relation)

    # ------------------------------------------------------------------
    # Build from chunks
    # ------------------------------------------------------------------

    def build_from_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """Rebuild the graph from parsed markdown chunks.

        Nodes = section headers + ``**bolded concepts**``.
        Edges = structural (contains) + semantic (mentions / typed relations).
        """
        with self._lock:
            self.graph.clear()

        section_nodes: Set[str] = set()
        for c in chunks:
            h = c["metadata"]["section"]
            self.add_node(h, node_type="section")
            section_nodes.add(h)

        concept_nodes: Set[str] = set()
        for c in chunks:
            concepts = re.findall(r"\*\*(.*?)\*\*", c["content"])
            for concept in concepts:
                if len(concept) < 3 or len(concept) > 50:
                    continue
                if concept in section_nodes:
                    continue
                self.add_node(concept, node_type="concept")
                concept_nodes.add(concept)
                self.add_edge(c["metadata"]["section"], concept, relation="contains")

        all_nodes = section_nodes | concept_nodes

        # Cross-reference edges
        for c in chunks:
            text = c["content"]
            text_lower = text.lower()
            section = c["metadata"]["section"]

            for node in all_nodes:
                if node == section:
                    continue
                if node.lower() not in text_lower:
                    continue

                # Determine relation type from context
                relation = "mentions"
                sentences = text.split(".")
                for s in sentences:
                    if node in s:
                        s_lower = s.lower()
                        for phrase, rel in self.RELATION_MAP.items():
                            if phrase in s_lower:
                                relation = rel
                                break
                        if relation != "mentions":
                            break

                # Skip implicit local concept mentions (already covered by "contains")
                # Only add if it's a typed relation or a section-to-section reference
                if node in concept_nodes and relation == "mentions":
                    continue

                self.add_edge(section, node, relation=relation)

        self.save()

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query(self, query_text: str, max_nodes: int = 10) -> List[Dict[str, Any]]:
        """Find graph nodes matching *query_text* keywords and return with neighbors."""
        words = set(query_text.lower().split())
        scored: List[tuple] = []
        for node in self.graph.nodes():
            node_lower = node.lower()
            score = sum(1 for w in words if w in node_lower)
            if score > 0:
                scored.append((node, score))

        scored.sort(key=lambda x: -x[1])
        scored = scored[:max_nodes]

        results: List[Dict[str, Any]] = []
        for node, score in scored:
            neighbors = []
            for nb in self.graph.neighbors(node):
                edge = self.graph.get_edge_data(node, nb, default={})
                neighbors.append({"node": nb, "relation": edge.get("relation", "related")})
            results.append({"node": node, "score": score, "neighbors": neighbors[:5]})
        return results

    def get_related(self, node_name: str, relation: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get nodes related to *node_name* (both directions)."""
        if node_name not in self.graph:
            return []
        related: List[Dict[str, Any]] = []
        for nb in self.graph.successors(node_name):
            edge = self.graph.get_edge_data(node_name, nb, default={})
            if relation is None or edge.get("relation") == relation:
                related.append({"node": nb, "relation": edge.get("relation"), "direction": "out"})
        for nb in self.graph.predecessors(node_name):
            edge = self.graph.get_edge_data(nb, node_name, default={})
            if relation is None or edge.get("relation") == relation:
                related.append({"node": nb, "relation": edge.get("relation"), "direction": "in"})
        return related

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, int]:
        return {"nodes": self.graph.number_of_nodes(), "edges": self.graph.number_of_edges()}
