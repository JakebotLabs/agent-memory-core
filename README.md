# agent-memory-core

**Three-layer persistent memory for AI agents.** Give any agent platform long-term recall with semantic search, knowledge graphs, and human-readable markdown — all in one `pip install`.

[![PyPI](https://img.shields.io/pypi/v/agent-memory-core)](https://pypi.org/project/agent-memory-core/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

## Why?

Most agent memory solutions give you one thing — a vector DB or a key-value store. Real memory needs three layers working together:

```
┌─────────────────────────────────────────────────────┐
│                   Your AI Agent                      │
│         (LangChain / CrewAI / AutoGPT / n8n)        │
└──────────────────────┬──────────────────────────────┘
                       │
              ┌────────▼────────┐
              │  MemoryManager  │  ← Unified API
              └────────┬────────┘
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
  ┌──────────┐  ┌──────────┐  ┌──────────┐
  │    L1    │  │    L2    │  │    L3    │
  │ Markdown │  │  Vector  │  │  Graph   │
  │          │  │          │  │          │
  │MEMORY.md │  │ ChromaDB │  │NetworkX  │
  │Daily logs│  │Sentence  │  │Knowledge │
  │Reference │  │Transform │  │  Graph   │
  └──────────┘  └──────────┘  └──────────┘
   Human-       Semantic       Relationship
   readable     search         traversal
```

## Quick Start

```bash
pip install agent-memory-core
```

```python
from agent_memory_core import MemoryManager

# Point to any directory — it becomes your memory workspace
mm = MemoryManager("/path/to/workspace")

# Index existing markdown files into vectors + graph
stats = mm.index()
print(f"Indexed {stats['chunks']} chunks, {stats['nodes']} graph nodes")

# Semantic search across all memory
results = mm.search("what architecture decisions were made?")
for r in results["vector_results"]:
    print(f"[{r['metadata']['section']}] {r['content'][:100]}")

# Store new information (auto-indexes)
mm.store("Decided to use PostgreSQL for the main database", to_memory=True)

# Get formatted context for system prompt injection
context = mm.search_formatted("database decisions")
print(context)
```

## Architecture

| Layer | Technology | Purpose | File Location |
|-------|-----------|---------|---------------|
| **L1: Markdown** | Plain `.md` files | Human-readable curated knowledge | `MEMORY.md`, `memory/*.md`, `reference/*.md` |
| **L2: Vector** | ChromaDB + all-MiniLM-L6-v2 | Semantic similarity search | `vector_memory/chroma_db/` |
| **L3: Graph** | NetworkX (directed) | Relationship traversal between concepts | `vector_memory/memory_graph.json` |

All three layers sync together. The indexer parses L1 markdown → generates L2 embeddings → rebuilds L3 graph automatically.

## Platform Integrations

### LangChain

```bash
pip install agent-memory-core[langchain]
```

```python
from agent_memory_core.integrations.langchain import MemorySearchTool, MemoryStoreTool

tools = [
    MemorySearchTool(base_dir="/workspace"),
    MemoryStoreTool(base_dir="/workspace"),
]
# Use with any LangChain agent
agent = create_react_agent(llm, tools)
```

### CrewAI

```bash
pip install agent-memory-core[crewai]
```

```python
from agent_memory_core.integrations.crewai import MemorySearchTool, MemoryStoreTool

agent = Agent(
    role="Research Assistant",
    tools=[MemorySearchTool(base_dir="/workspace")],
)
```

### Any Platform (Direct API)

```python
from agent_memory_core import MemoryManager

mm = MemoryManager("/workspace")

# Use in any framework's tool/function calling
def search_memory(query: str) -> str:
    return mm.search_formatted(query, compact=True)

def store_memory(text: str) -> str:
    mm.store(text, to_memory=True)
    return "Stored."
```

## Features

### Significance Classifier

Automatically classify which interactions are worth remembering:

```python
from agent_memory_core import SignificanceClassifier

# Rule-based (no LLM needed)
classifier = SignificanceClassifier()
is_sig, reason, score = classifier.classify("Decided to migrate to PostgreSQL")
# → (True, "SIGNIFICANT: 3 indicators, 2 high-priority (score: 1.90)", 1.9)

# LLM-powered (bring any provider)
classifier = SignificanceClassifier(llm_fn=lambda prompt: openai.chat(prompt))
```

### Auto-Promotion Pipeline

Automatically promote significant daily log entries to long-term memory:

```python
result = mm.promote(days_back=3, min_confidence=0.7)
print(f"Promoted {result['promotions_made']} entries to MEMORY.md")
```

### Sync Status

Monitor memory health:

```python
status = mm.sync_status()
# {'memory_md_hash': 'a1b2c3', 'vector_chunks': 42, 'nodes': 156, 'edges': 312, ...}
```

## Workspace Structure

After setup, your workspace looks like:

```
workspace/
├── MEMORY.md              ← Curated long-term knowledge
├── memory/
│   ├── 2026-02-21.md      ← Daily log (auto-created)
│   └── ...
├── reference/             ← Institutional knowledge (optional)
│   ├── people.md
│   └── infrastructure.md
└── vector_memory/
    ├── chroma_db/         ← Vector database
    └── memory_graph.json  ← Knowledge graph
```

## API Reference

### `MemoryManager(base_dir, **kwargs)`

| Method | Description |
|--------|-------------|
| `index()` | Parse markdown → index vectors → rebuild graph |
| `search(query, n_results=5)` | Search vectors + graph, return structured results |
| `search_formatted(query, compact=False)` | Search and return markdown-formatted context |
| `store(text, to_memory=False)` | Store to daily log (+ MEMORY.md), re-index |
| `promote(days_back=3, min_confidence=0.7)` | Auto-promote significant entries |
| `sync_status()` | Memory health check |

### Individual Stores

Access layers directly when needed:

```python
mm.markdown  # MarkdownStore — file management
mm.vectors   # VectorStore — ChromaDB operations
mm.graph     # GraphStore — NetworkX operations
```

## Configuration

```python
mm = MemoryManager(
    base_dir="/workspace",
    vector_db_subdir="vector_memory/chroma_db",  # ChromaDB location
    graph_filename="memory_graph.json",            # Graph file name
    model_name="all-MiniLM-L6-v2",               # Embedding model
    llm_fn=my_llm_callable,                        # Optional LLM for classifier
)
```

## Development

```bash
git clone https://github.com/Jakebot-ops/agent-memory-core.git
cd agent-memory-core
pip install -e ".[dev]"
pytest
```

## License

MIT — use it however you want.

## Credits

Born from the [agent-memory-guide](https://github.com/Jakebot-ops/agent-memory-guide) — a practical guide to building persistent memory for AI agents. This package extracts the three-layer architecture into a standalone, pip-installable library.
