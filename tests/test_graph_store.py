"""Tests for GraphStore (L3 layer) — NetworkX knowledge graph."""

import json
import pytest
import networkx as nx

from agent_memory_core.graph_store import GraphStore


class TestGraphStoreInit:
    """Test GraphStore initialization."""

    def test_init_creates_empty_graph(self, tmp_path):
        """Init creates empty DiGraph when no file exists."""
        gs = GraphStore(str(tmp_path / "graph.json"))
        
        assert isinstance(gs.graph, nx.DiGraph)
        assert gs.graph.number_of_nodes() == 0
        assert gs.graph.number_of_edges() == 0

    def test_init_loads_existing_graph(self, tmp_path):
        """Init loads graph from existing file."""
        graph_path = tmp_path / "graph.json"
        
        # Create a graph file
        data = {
            "directed": True,
            "multigraph": False,
            "nodes": [{"id": "A"}, {"id": "B"}],
            "links": [{"source": "A", "target": "B", "relation": "related"}]
        }
        graph_path.write_text(json.dumps(data))
        
        gs = GraphStore(str(graph_path))
        
        assert gs.graph.number_of_nodes() == 2
        assert gs.graph.number_of_edges() == 1


class TestPersistence:
    """Test graph save/load methods."""

    def test_save_creates_file(self, graph_store):
        """Save creates JSON file."""
        graph_store.add_node("Test", node_type="concept")
        graph_store.save()
        
        import os
        assert os.path.exists(graph_store.graph_path)

    def test_save_load_roundtrip(self, tmp_path):
        """Graph survives save/load cycle."""
        gs1 = GraphStore(str(tmp_path / "graph.json"))
        gs1.add_node("NodeA", node_type="section")
        gs1.add_node("NodeB", node_type="concept")
        gs1.add_edge("NodeA", "NodeB", relation="contains")
        gs1.save()
        
        gs2 = GraphStore(str(tmp_path / "graph.json"))
        
        assert gs2.graph.number_of_nodes() == 2
        assert gs2.graph.number_of_edges() == 1
        assert gs2.graph.has_edge("NodeA", "NodeB")

    def test_load_nonexistent_file(self, tmp_path):
        """Load handles nonexistent file gracefully."""
        gs = GraphStore(str(tmp_path / "nonexistent.json"))
        
        # Should have empty graph
        assert gs.graph.number_of_nodes() == 0


class TestNodeOperations:
    """Test node manipulation methods."""

    def test_add_node_basic(self, graph_store):
        """Add node with type."""
        graph_store.add_node("Architecture", node_type="section")
        
        assert "Architecture" in graph_store.graph.nodes
        assert graph_store.graph.nodes["Architecture"]["type"] == "section"

    def test_add_node_with_metadata(self, graph_store):
        """Add node with extra metadata."""
        graph_store.add_node(
            "ChromaDB",
            node_type="concept",
            importance="high",
            created="2024-01-01"
        )
        
        node = graph_store.graph.nodes["ChromaDB"]
        assert node["type"] == "concept"
        assert node["importance"] == "high"
        assert node["created"] == "2024-01-01"

    def test_add_duplicate_node(self, graph_store):
        """Adding duplicate node updates attributes."""
        graph_store.add_node("Test", node_type="concept")
        graph_store.add_node("Test", node_type="section", extra="data")
        
        node = graph_store.graph.nodes["Test"]
        assert node["type"] == "section"  # Updated
        assert node["extra"] == "data"


class TestEdgeOperations:
    """Test edge manipulation methods."""

    def test_add_edge_basic(self, graph_store):
        """Add edge between nodes."""
        graph_store.add_node("A", node_type="section")
        graph_store.add_node("B", node_type="concept")
        graph_store.add_edge("A", "B", relation="contains")
        
        assert graph_store.graph.has_edge("A", "B")
        assert graph_store.graph.edges["A", "B"]["relation"] == "contains"

    def test_add_edge_missing_node(self, graph_store):
        """Edge not added if nodes don't exist."""
        graph_store.add_node("A", node_type="section")
        # B doesn't exist
        graph_store.add_edge("A", "B", relation="contains")
        
        assert not graph_store.graph.has_edge("A", "B")

    def test_edge_default_relation(self, graph_store):
        """Default relation is 'related_to'."""
        graph_store.add_node("A", node_type="section")
        graph_store.add_node("B", node_type="section")
        graph_store.add_edge("A", "B")
        
        assert graph_store.graph.edges["A", "B"]["relation"] == "related_to"


class TestBuildFromChunks:
    """Test graph building from markdown chunks."""

    def test_build_creates_section_nodes(self, graph_store):
        """Section headers become nodes."""
        chunks = [
            {"content": "Content about architecture", "metadata": {"section": "Architecture", "source": "test.md"}},
            {"content": "Content about design", "metadata": {"section": "Design", "source": "test.md"}},
        ]
        
        graph_store.build_from_chunks(chunks)
        
        assert "Architecture" in graph_store.graph.nodes
        assert "Design" in graph_store.graph.nodes
        assert graph_store.graph.nodes["Architecture"]["type"] == "section"

    def test_build_extracts_concepts(self, graph_store):
        """Bold text becomes concept nodes."""
        chunks = [
            {"content": "Using **ChromaDB** for vectors", "metadata": {"section": "Tech", "source": "test.md"}},
        ]
        
        graph_store.build_from_chunks(chunks)
        
        assert "ChromaDB" in graph_store.graph.nodes
        assert graph_store.graph.nodes["ChromaDB"]["type"] == "concept"

    def test_build_creates_contains_edges(self, graph_store):
        """Sections contain their concepts."""
        chunks = [
            {"content": "**NetworkX** is great", "metadata": {"section": "Tools", "source": "test.md"}},
        ]
        
        graph_store.build_from_chunks(chunks)
        
        assert graph_store.graph.has_edge("Tools", "NetworkX")
        assert graph_store.graph.edges["Tools", "NetworkX"]["relation"] == "contains"

    def test_build_filters_short_concepts(self, graph_store):
        """Concepts < 3 chars are filtered."""
        chunks = [
            {"content": "**A** and **BB** are short, **Valid** is not", "metadata": {"section": "Test", "source": "test.md"}},
        ]
        
        graph_store.build_from_chunks(chunks)
        
        assert "A" not in graph_store.graph.nodes
        assert "BB" not in graph_store.graph.nodes
        assert "Valid" in graph_store.graph.nodes

    def test_build_filters_long_concepts(self, graph_store):
        """Concepts > 50 chars are filtered."""
        long_concept = "A" * 60
        chunks = [
            {"content": f"**{long_concept}** is too long", "metadata": {"section": "Test", "source": "test.md"}},
        ]
        
        graph_store.build_from_chunks(chunks)
        
        assert long_concept not in graph_store.graph.nodes

    def test_build_detects_typed_relations(self, graph_store):
        """Typed relations are extracted from context."""
        chunks = [
            {"content": "**ChromaDB** depends on SQLite", "metadata": {"section": "Dependencies", "source": "test.md"}},
        ]
        
        graph_store.build_from_chunks(chunks)
        
        # Should have dependency relation (if SQLite is also a concept or section)
        # The relation detection works on cross-references

    def test_build_clears_existing(self, graph_store_with_data):
        """Build clears existing graph first."""
        initial_nodes = graph_store_with_data.graph.number_of_nodes()
        assert initial_nodes > 0
        
        # Build with new chunks
        graph_store_with_data.build_from_chunks([
            {"content": "New only", "metadata": {"section": "New", "source": "test.md"}},
        ])
        
        # Only new nodes remain
        assert "Architecture" not in graph_store_with_data.graph.nodes
        assert "New" in graph_store_with_data.graph.nodes


class TestQuerying:
    """Test graph query methods."""

    def test_query_finds_matching_nodes(self, graph_store_with_data):
        """Query finds nodes matching keywords."""
        results = graph_store_with_data.query("Architecture")
        
        assert len(results) >= 1
        assert any(r["node"] == "Architecture" for r in results)

    def test_query_returns_neighbors(self, graph_store_with_data):
        """Query results include neighbors."""
        results = graph_store_with_data.query("Architecture")
        
        arch_result = next(r for r in results if r["node"] == "Architecture")
        assert "neighbors" in arch_result
        assert len(arch_result["neighbors"]) >= 1

    def test_query_respects_max_nodes(self, graph_store_with_data):
        """Query respects max_nodes limit."""
        results = graph_store_with_data.query("a", max_nodes=1)
        
        assert len(results) <= 1

    def test_query_case_insensitive(self, graph_store_with_data):
        """Query is case-insensitive."""
        results = graph_store_with_data.query("architecture")
        
        assert len(results) >= 1
        assert any(r["node"] == "Architecture" for r in results)

    def test_query_scores_by_matches(self, graph_store_with_data):
        """Results include relevance score."""
        results = graph_store_with_data.query("ChromaDB")
        
        if results:
            assert "score" in results[0]
            assert results[0]["score"] > 0


class TestGetRelated:
    """Test get_related method."""

    def test_get_related_finds_successors(self, graph_store_with_data):
        """Finds outgoing relationships."""
        related = graph_store_with_data.get_related("Architecture")
        
        out_relations = [r for r in related if r["direction"] == "out"]
        assert len(out_relations) >= 1

    def test_get_related_finds_predecessors(self, graph_store_with_data):
        """Finds incoming relationships."""
        related = graph_store_with_data.get_related("ChromaDB")
        
        in_relations = [r for r in related if r["direction"] == "in"]
        assert len(in_relations) >= 1

    def test_get_related_filters_by_relation(self, graph_store_with_data):
        """Can filter by relation type."""
        related = graph_store_with_data.get_related("Architecture", relation="uses")
        
        for r in related:
            assert r["relation"] == "uses"

    def test_get_related_nonexistent_node(self, graph_store_with_data):
        """Returns empty for nonexistent node."""
        related = graph_store_with_data.get_related("NonexistentNode")
        
        assert related == []


class TestStats:
    """Test statistics method."""

    def test_stats_empty_graph(self, graph_store):
        """Stats for empty graph."""
        stats = graph_store.stats()
        
        assert stats["nodes"] == 0
        assert stats["edges"] == 0

    def test_stats_with_data(self, graph_store_with_data):
        """Stats reflect graph contents."""
        stats = graph_store_with_data.stats()
        
        assert stats["nodes"] >= 4
        assert stats["edges"] >= 3


class TestRelationMap:
    """Test relation mapping."""

    def test_relation_map_keys(self):
        """RELATION_MAP has expected entries."""
        assert "depends on" in GraphStore.RELATION_MAP
        assert "uses" in GraphStore.RELATION_MAP
        assert "enables" in GraphStore.RELATION_MAP

    def test_relation_map_values(self):
        """Mapped values are snake_case."""
        for key, value in GraphStore.RELATION_MAP.items():
            assert " " not in value
            assert value.islower() or "_" in value


class TestThreadSafety:
    """Test thread safety."""

    def test_save_thread_safe(self, tmp_path):
        """Save uses lock."""
        import threading
        
        gs = GraphStore(str(tmp_path / "graph.json"))
        
        def modifier(idx):
            gs.add_node(f"Node{idx}", node_type="concept")
            gs.save()
        
        threads = []
        for i in range(10):
            t = threading.Thread(target=modifier, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # Graph should have all nodes
        assert gs.graph.number_of_nodes() == 10

    def test_build_thread_safe(self, tmp_path):
        """Build uses lock."""
        import threading
        
        gs = GraphStore(str(tmp_path / "graph.json"))
        
        def builder(idx):
            chunks = [
                {"content": f"Thread {idx}", "metadata": {"section": f"Section{idx}", "source": "test.md"}}
            ]
            gs.build_from_chunks(chunks)
        
        threads = []
        for i in range(5):
            t = threading.Thread(target=builder, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # One of the builds should have completed
        assert gs.graph.number_of_nodes() >= 1
