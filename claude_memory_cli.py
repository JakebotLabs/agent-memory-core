"""
Claude Memory CLI — Bridge between Claude Code sessions and agent-memory-core.

Usage:
    python claude_memory_cli.py store "text to remember"
    python claude_memory_cli.py store --permanent "important decision or fact"
    python claude_memory_cli.py search "what was the agentforge plan?"
    python claude_memory_cli.py index
    python claude_memory_cli.py status
    python claude_memory_cli.py daily "session note for today's log"
    python claude_memory_cli.py recall           # dump full MEMORY.md
    python claude_memory_cli.py context "query"   # compact context for system prompt

Workspace: ~/claude-memory/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agent_memory_core import MemoryManager


WORKSPACE = Path.home() / "claude-memory"


def get_manager() -> MemoryManager:
    """Initialize MemoryManager with Claude's workspace."""
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    return MemoryManager(base_dir=str(WORKSPACE))


def cmd_store(args: argparse.Namespace) -> None:
    """Store a memory entry."""
    mm = get_manager()
    text = " ".join(args.text)
    mm.store(text, to_memory=args.permanent)
    mm.index()
    dest = "MEMORY.md + daily log" if args.permanent else "daily log"
    print(f"Stored to {dest}. Indexed.")


def cmd_search(args: argparse.Namespace) -> None:
    """Search memories."""
    mm = get_manager()
    query = " ".join(args.query)
    results = mm.search(query, n_results=args.n)

    if results.get("vector_results"):
        print("=== Semantic Results ===")
        for r in results["vector_results"]:
            dist = r.get("distance", "?")
            src = r.get("metadata", {}).get("source", "?")
            print(f"\n[{src}] (distance: {dist:.3f})")
            print(r["content"][:500])
    else:
        print("No semantic results found.")

    if results.get("graph_results"):
        print("\n=== Graph Results ===")
        for node in results["graph_results"]:
            name = node.get("name", "?")
            ntype = node.get("type", "?")
            neighbors = node.get("neighbors", [])
            print(f"\n[{ntype}] {name}")
            for nb in neighbors[:5]:
                print(f"  --{nb.get('relation', '?')}--> {nb.get('name', '?')}")


def cmd_context(args: argparse.Namespace) -> None:
    """Get compact context string for system prompts."""
    mm = get_manager()
    query = " ".join(args.query)
    context = mm.search_formatted(query, n_results=args.n, compact=True)
    print(context)


def cmd_index(args: argparse.Namespace) -> None:
    """Re-index all memory layers."""
    mm = get_manager()
    stats = mm.index()
    print(f"Indexed: {stats.get('chunks', 0)} chunks, "
          f"{stats.get('nodes', 0)} nodes, {stats.get('edges', 0)} edges")


def cmd_status(args: argparse.Namespace) -> None:
    """Show memory system status."""
    mm = get_manager()
    status = mm.sync_status()
    print(f"Status:        {status.get('status', '?')}")
    print(f"Vector chunks: {status.get('vector_chunks', 0)}")
    print(f"Graph nodes:   {status.get('nodes', 0)}")
    print(f"Graph edges:   {status.get('edges', 0)}")
    print(f"Files found:   {status.get('files_found', 0)}")
    print(f"Timestamp:     {status.get('timestamp', '?')}")


def cmd_daily(args: argparse.Namespace) -> None:
    """Append to today's daily log without touching MEMORY.md."""
    mm = get_manager()
    text = " ".join(args.text)
    mm.markdown.append_daily(text)
    print(f"Appended to daily log.")


def cmd_recall(args: argparse.Namespace) -> None:
    """Print full MEMORY.md contents."""
    mm = get_manager()
    content = mm.markdown.read_memory()
    if content:
        print(content)
    else:
        print("MEMORY.md is empty or does not exist yet.")


def main() -> None:
    # Fix Unicode output on Windows cp1252 consoles (e.g., → characters in MEMORY.md)
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Claude Memory CLI — persistent memory for Claude Code sessions"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # store
    p_store = sub.add_parser("store", help="Store a memory")
    p_store.add_argument("text", nargs="+")
    p_store.add_argument("--permanent", "-p", action="store_true",
                         help="Also write to MEMORY.md (long-term)")
    p_store.set_defaults(func=cmd_store)

    # search
    p_search = sub.add_parser("search", help="Search memories")
    p_search.add_argument("query", nargs="+")
    p_search.add_argument("-n", type=int, default=5, help="Number of results")
    p_search.set_defaults(func=cmd_search)

    # context
    p_ctx = sub.add_parser("context", help="Get compact context for prompts")
    p_ctx.add_argument("query", nargs="+")
    p_ctx.add_argument("-n", type=int, default=5, help="Number of results")
    p_ctx.set_defaults(func=cmd_context)

    # index
    p_index = sub.add_parser("index", help="Re-index all layers")
    p_index.set_defaults(func=cmd_index)

    # status
    p_status = sub.add_parser("status", help="Show memory status")
    p_status.set_defaults(func=cmd_status)

    # daily
    p_daily = sub.add_parser("daily", help="Append to daily log")
    p_daily.add_argument("text", nargs="+")
    p_daily.set_defaults(func=cmd_daily)

    # recall
    p_recall = sub.add_parser("recall", help="Print MEMORY.md")
    p_recall.set_defaults(func=cmd_recall)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
