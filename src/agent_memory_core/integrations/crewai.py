"""
CrewAI tool wrappers for agent-memory-core.

Requires: pip install agent-memory-core[crewai]

Usage:
    from agent_memory_core.integrations.crewai import MemorySearchTool, MemoryStoreTool

    agent = Agent(tools=[MemorySearchTool(base_dir="/workspace")])
"""

from __future__ import annotations

from typing import Optional

try:
    from crewai.tools import BaseTool
except ImportError as e:
    raise ImportError(
        "CrewAI integration requires crewai. "
        "Install with: pip install agent-memory-core[crewai]"
    ) from e

from agent_memory_core.memory import MemoryManager


class MemorySearchTool(BaseTool):
    """CrewAI tool for searching agent memory."""

    name: str = "Memory Search"
    description: str = (
        "Search the agent's persistent three-layer memory system. "
        "Returns relevant context from vector search and knowledge graph."
    )

    base_dir: str
    _manager: Optional[MemoryManager] = None

    def _get_manager(self) -> MemoryManager:
        if self._manager is None:
            self._manager = MemoryManager(self.base_dir)
        return self._manager

    def _run(self, query: str) -> str:
        return self._get_manager().search_formatted(query, compact=True)


class MemoryStoreTool(BaseTool):
    """CrewAI tool for storing information to agent memory."""

    name: str = "Memory Store"
    description: str = (
        "Store important information to the agent's persistent memory. "
        "Saves to daily log and re-indexes for future retrieval."
    )

    base_dir: str
    _manager: Optional[MemoryManager] = None

    def _get_manager(self) -> MemoryManager:
        if self._manager is None:
            self._manager = MemoryManager(self.base_dir)
        return self._manager

    def _run(self, text: str, to_memory: bool = False) -> str:
        self._get_manager().store(text, to_memory=to_memory)
        return f"Stored to {'MEMORY.md + ' if to_memory else ''}daily log and re-indexed."
