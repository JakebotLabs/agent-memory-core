"""
LangChain tool wrappers for agent-memory-core.

Requires: pip install agent-memory-core[langchain]

Usage:
    from agent_memory_core.integrations.langchain import MemorySearchTool, MemoryStoreTool

    tools = [MemorySearchTool(base_dir="/path/to/workspace")]
    agent = create_react_agent(llm, tools)
"""

from __future__ import annotations

from typing import Optional, Type

try:
    from langchain_core.tools import BaseTool
    from pydantic import BaseModel, Field
except ImportError as e:
    raise ImportError(
        "LangChain integration requires langchain-core. "
        "Install with: pip install agent-memory-core[langchain]"
    ) from e

from agent_memory_core.memory import MemoryManager


class _SearchInput(BaseModel):
    query: str = Field(description="Search query for memory retrieval")
    n_results: int = Field(default=5, description="Number of results to return")


class MemorySearchTool(BaseTool):
    """LangChain tool for searching agent memory."""

    name: str = "memory_search"
    description: str = (
        "Search the agent's persistent memory for relevant context. "
        "Use this to recall decisions, facts, lessons, and prior work."
    )
    args_schema: Type[BaseModel] = _SearchInput

    base_dir: str
    _manager: Optional[MemoryManager] = None

    class Config:
        underscore_attrs_are_private = True

    def _get_manager(self) -> MemoryManager:
        if self._manager is None:
            self._manager = MemoryManager(self.base_dir)
        return self._manager

    def _run(self, query: str, n_results: int = 5) -> str:
        mm = self._get_manager()
        return mm.search_formatted(query, n_results=n_results, compact=True)


class _StoreInput(BaseModel):
    text: str = Field(description="Text to store in memory")
    to_memory: bool = Field(default=False, description="Also write to long-term MEMORY.md")


class MemoryStoreTool(BaseTool):
    """LangChain tool for storing information to agent memory."""

    name: str = "memory_store"
    description: str = (
        "Store important information to the agent's persistent memory. "
        "Use for decisions, discoveries, lessons learned, and key facts."
    )
    args_schema: Type[BaseModel] = _StoreInput

    base_dir: str
    _manager: Optional[MemoryManager] = None

    class Config:
        underscore_attrs_are_private = True

    def _get_manager(self) -> MemoryManager:
        if self._manager is None:
            self._manager = MemoryManager(self.base_dir)
        return self._manager

    def _run(self, text: str, to_memory: bool = False) -> str:
        mm = self._get_manager()
        mm.store(text, to_memory=to_memory)
        return f"Stored to {'MEMORY.md + ' if to_memory else ''}daily log and re-indexed."
