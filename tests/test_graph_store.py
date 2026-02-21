"""Tests for the GraphStore (L3 layer)."""

import os
import pytest

from agent_memory_core.graph_store import GraphStore


@pytest.fixture
def graph(tmp_path):
    return GraphStore(str(tmp_path / "graph.json"))


def test_add_node_and_save(graph):
    graph.add_node("TestConcept", node_type="concept")
    graph.save()
    assert os.path.exists(graph.graph_path)

    g2 = GraphStore(graph.graph_path)
    assert "TestConcept" in g2.graph.nodes()


def test_add_edge(graph):
    graph.add_node("A")
    graph.add_node("B")
    graph.add_edge("A", "B", relation="depends_on")
    assert graph.graph.has_edge("A", "B")


def test_build_from_chunks(graph):
    chunks = [
        {"content": "**ChromaDB** is used for vector search.", "metadata": {"source": "test", "section": "Architecture"}},
        {"content": "**NetworkX** provides graph traversal.", "metadata": {"source": "test", "section": "Architecture"}},
        {"content": "The system uses **ChromaDB** and **NetworkX**.", "metadata": {"source": "test", "section": "Overview"}},
    ]
    graph.build_from_chunks(chunks)
    assert graph.graph.number_of_nodes() > 0
    assert "ChromaDB" in graph.graph.nodes()
    assert "NetworkX" in graph.graph.nodes()


def test_query(graph):
    graph.add_node("ChromaDB", node_type="concept")
    graph.add_node("Architecture", node_type="section")
    graph.add_edge("Architecture", "ChromaDB", relation="contains")
    results = graph.query("chromadb")
    assert len(results) > 0
    assert results[0]["node"] == "ChromaDB"


def test_get_related(graph):
    graph.add_node("A")
    graph.add_node("B")
    graph.add_edge("A", "B", relation="uses")
    related = graph.get_related("A")
    assert len(related) == 1
    assert related[0]["node"] == "B"
    assert related[0]["direction"] == "out"


def test_stats(graph):
    graph.add_node("X")
    graph.add_node("Y")
    graph.add_edge("X", "Y")
    s = graph.stats()
    assert s["nodes"] == 2
    assert s["edges"] == 1
