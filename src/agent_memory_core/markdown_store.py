"""
L1 Markdown Store — Human-readable curated knowledge layer.

Manages MEMORY.md, daily logs (memory/YYYY-MM-DD.md), and reference files.
Provides parsing, chunk extraction, and file management.
"""

from __future__ import annotations

import glob
import os
import re
import threading
from datetime import date, datetime
from typing import Any, Dict, List, Optional

__all__ = ["MarkdownStore"]


class MarkdownStore:
    """Manages the L1 markdown memory layer."""

    def __init__(self, base_dir: str) -> None:
        self.base_dir = os.path.abspath(base_dir)
        self.memory_file = os.path.join(self.base_dir, "MEMORY.md")
        self.memory_dir = os.path.join(self.base_dir, "memory")
        self.reference_dir = os.path.join(self.base_dir, "reference")
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # File discovery
    # ------------------------------------------------------------------

    def gather_files(self) -> List[str]:
        """Collect all indexable markdown files in priority order."""
        files: List[str] = []
        if os.path.exists(self.memory_file):
            files.append(self.memory_file)
        for d in (self.reference_dir, self.memory_dir):
            if os.path.isdir(d):
                files.extend(sorted(glob.glob(os.path.join(d, "*.md"))))
        return files

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    @staticmethod
    def parse_markdown(file_path: str, base_dir: str) -> List[Dict[str, Any]]:
        """Parse a markdown file into chunks split on ``#`` / ``##`` headers.

        Returns a list of dicts with keys ``content`` and ``metadata``.
        """
        if not os.path.exists(file_path):
            return []

        source = os.path.relpath(file_path, base_dir)

        with open(file_path, "r", encoding="utf-8") as fh:
            text = fh.read()

        chunks: List[Dict[str, Any]] = []
        sections = re.split(r"(^##?\s+.*$)", text, flags=re.MULTILINE)

        current_header = "Intro"
        intro = sections[0].strip()
        if intro:
            chunks.append({"content": intro, "metadata": {"source": source, "section": current_header}})

        for i in range(1, len(sections), 2):
            header = sections[i].strip().lstrip("#").strip()
            body = sections[i + 1].strip() if i + 1 < len(sections) else ""
            if body:
                chunks.append({"content": body, "metadata": {"source": source, "section": header}})

        return chunks

    def get_all_chunks(self) -> List[Dict[str, Any]]:
        """Parse every discovered file and return all chunks."""
        chunks: List[Dict[str, Any]] = []
        for fp in self.gather_files():
            chunks.extend(self.parse_markdown(fp, self.base_dir))
        return chunks

    # ------------------------------------------------------------------
    # MEMORY.md helpers
    # ------------------------------------------------------------------

    def read_memory(self) -> str:
        """Return contents of MEMORY.md (empty string if missing)."""
        if not os.path.exists(self.memory_file):
            return ""
        with open(self.memory_file, "r", encoding="utf-8") as fh:
            return fh.read()

    def write_memory(self, content: str) -> None:
        """Overwrite MEMORY.md (thread-safe)."""
        with self._lock:
            os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
            with open(self.memory_file, "w", encoding="utf-8") as fh:
                fh.write(content)

    def append_to_memory(self, text: str) -> None:
        """Append text to MEMORY.md."""
        with self._lock:
            os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
            with open(self.memory_file, "a", encoding="utf-8") as fh:
                fh.write(text)

    # ------------------------------------------------------------------
    # Daily log helpers
    # ------------------------------------------------------------------

    def today_log_path(self) -> str:
        return os.path.join(self.memory_dir, f"{date.today().isoformat()}.md")

    def append_daily(self, text: str, log_date: Optional[date] = None) -> str:
        """Append *text* to today's daily log. Returns the file path."""
        d = log_date or date.today()
        path = os.path.join(self.memory_dir, f"{d.isoformat()}.md")
        with self._lock:
            os.makedirs(self.memory_dir, exist_ok=True)
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(text)
        return path

    def list_daily_logs(self, last_n: int = 7) -> List[str]:
        """Return paths to the most recent *last_n* daily log files."""
        if not os.path.isdir(self.memory_dir):
            return []
        logs = sorted(glob.glob(os.path.join(self.memory_dir, "????-??-??.md")), reverse=True)
        return logs[:last_n]

    # ------------------------------------------------------------------
    # Section helpers
    # ------------------------------------------------------------------

    def get_sections(self) -> List[str]:
        """Return header names from MEMORY.md."""
        content = self.read_memory()
        return re.findall(r"^#+\s+(.+)$", content, re.MULTILINE)

    def find_insertion_point(self, content: str, target_section: str = "Recent Updates") -> int:
        """Find the best character offset to insert new content in MEMORY.md."""
        patterns = [
            rf"## {re.escape(target_section)}.*?(?=\n##|\n---|\Z)",
            r"## Recent Updates.*?(?=\n##|\n---|\Z)",
            r"## Latest.*?(?=\n##|\n---|\Z)",
        ]
        for pat in patterns:
            m = re.search(pat, content, re.DOTALL)
            if m:
                return m.end()

        lessons = re.search(r"\n## Key Lessons", content)
        if lessons:
            return lessons.start()

        return len(content)
