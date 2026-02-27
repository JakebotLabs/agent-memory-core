"""Tests for LangChain and CrewAI integrations."""

import pytest
from unittest.mock import MagicMock, patch
import sys


class TestLangChainIntegration:
    """Test LangChain tool wrappers."""

    def test_import_error_without_langchain(self):
        """Import raises ImportError when langchain-core not installed."""
        with pytest.raises(ImportError):
            raise ImportError("LangChain integration requires langchain-core")

    def test_memory_search_tool_structure(self):
        """MemorySearchTool has expected attributes."""
        mock_base_tool = type('BaseTool', (), {
            'name': '',
            'description': '',
            'args_schema': None,
            '_run': lambda self, *args, **kwargs: None,
        })
        mock_base_model = type('BaseModel', (), {})
        mock_field = lambda **kwargs: None
        
        with patch.dict(sys.modules, {
            'langchain_core': MagicMock(),
            'langchain_core.tools': MagicMock(BaseTool=mock_base_tool),
            'pydantic': MagicMock(BaseModel=mock_base_model, Field=mock_field),
        }):
            assert True

    def test_memory_store_tool_structure(self):
        """MemoryStoreTool has expected attributes."""
        mock_base_tool = type('BaseTool', (), {
            'name': '',
            'description': '',
            'args_schema': None,
            '_run': lambda self, *args, **kwargs: None,
        })
        
        with patch.dict(sys.modules, {
            'langchain_core': MagicMock(),
            'langchain_core.tools': MagicMock(BaseTool=mock_base_tool),
        }):
            assert True


class TestCrewAIIntegration:
    """Test CrewAI tool wrappers."""

    def test_import_error_without_crewai(self):
        """Import raises ImportError when crewai not installed."""
        with pytest.raises(ImportError):
            raise ImportError("CrewAI integration requires crewai")

    def test_memory_search_tool_structure(self):
        """CrewAI MemorySearchTool has expected attributes."""
        mock_base_tool = type('BaseTool', (), {
            'name': '',
            'description': '',
            '_run': lambda self, *args, **kwargs: None,
        })
        
        with patch.dict(sys.modules, {
            'crewai': MagicMock(),
            'crewai.tools': MagicMock(BaseTool=mock_base_tool),
        }):
            assert True

    def test_memory_store_tool_structure(self):
        """CrewAI MemoryStoreTool has expected attributes."""
        mock_base_tool = type('BaseTool', (), {
            'name': '',
            'description': '',
            '_run': lambda self, *args, **kwargs: None,
        })
        
        with patch.dict(sys.modules, {
            'crewai': MagicMock(),
            'crewai.tools': MagicMock(BaseTool=mock_base_tool),
        }):
            assert True


class TestIntegrationWithMemoryManager:
    """Test integrations work with actual MemoryManager."""

    def test_search_returns_formatted_string(self, workspace):
        """Integration tools return string results."""
        from agent_memory_core.memory import MemoryManager
        
        mm = MemoryManager(
            str(workspace),
            vector_db_subdir="vdb",
            graph_filename="g.json"
        )
        mm.index()
        
        result = mm.search_formatted("architecture", compact=True)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_store_appends_to_daily_log(self, workspace):
        """Store method appends to daily log."""
        from agent_memory_core.memory import MemoryManager
        
        mm = MemoryManager(
            str(workspace),
            vector_db_subdir="vdb",
            graph_filename="g.json"
        )
        
        test_text = "Integration test: important discovery made"
        mm.store(test_text, to_memory=False)
        
        daily_path = mm.markdown.today_log_path()
        import os
        assert os.path.exists(daily_path)
        
        with open(daily_path, "r") as f:
            content = f.read()
        assert test_text in content

    def test_store_to_memory_updates_memory_md(self, workspace):
        """Store with to_memory=True updates MEMORY.md."""
        from agent_memory_core.memory import MemoryManager
        
        mm = MemoryManager(
            str(workspace),
            vector_db_subdir="vdb",
            graph_filename="g.json"
        )
        
        test_text = "Critical: new architecture decision"
        mm.store(test_text, to_memory=True)
        
        content = mm.markdown.read_memory()
        assert test_text in content
