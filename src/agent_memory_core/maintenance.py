"""
Daily Maintenance — Automated promotion + re-indexing pipeline.

Combines significance classification, promotion to MEMORY.md, and
re-indexing of all three memory layers.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

from agent_memory_core.classifier import SignificanceClassifier
from agent_memory_core.markdown_store import MarkdownStore

__all__ = ["MaintenanceRunner"]


class MaintenanceRunner:
    """Runs the daily promote → re-index maintenance cycle."""

    def __init__(
        self,
        markdown_store: MarkdownStore,
        classifier: Optional[SignificanceClassifier] = None,
    ) -> None:
        self.md = markdown_store
        self.classifier = classifier or SignificanceClassifier()

    # ------------------------------------------------------------------
    # Promotion
    # ------------------------------------------------------------------

    def find_promotion_candidates(
        self,
        days_back: int = 3,
        min_confidence: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """Scan recent daily logs for significant entries."""
        candidates: List[Dict[str, Any]] = []
        today = date.today()

        for i in range(days_back):
            target = today - timedelta(days=i)
            log_file = f"{target.isoformat()}.md"
            log_path = self.md.memory_dir + "/" + log_file

            chunks = MarkdownStore.parse_markdown(log_path, self.md.base_dir)
            if not chunks:
                continue

            results = self.classifier.classify_chunks(chunks)
            for r in results:
                if r["is_significant"] and r["confidence"] >= min_confidence:
                    r["source_file"] = log_file
                    r["source_date"] = target.isoformat()
                    candidates.append(r)

        return candidates

    def promote(
        self,
        days_back: int = 3,
        min_confidence: float = 0.7,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Promote significant daily log entries to MEMORY.md.

        Returns dict with ``candidates_found``, ``promotions_made``, ``promoted_items``.
        """
        candidates = self.find_promotion_candidates(days_back, min_confidence)

        result: Dict[str, Any] = {
            "candidates_found": len(candidates),
            "promotions_made": 0,
            "promoted_items": [],
            "dry_run": dry_run,
        }

        if not candidates:
            return result

        # Build entries
        lines = ["\n## Recent Updates (Auto-Promoted)\n"]
        for c in sorted(candidates, key=lambda x: x.get("source_date", ""), reverse=True):
            marker = "⭐" if c["confidence"] > 0.8 else "•"
            content = c["content"].strip()
            content = re.sub(r"^#+\s*", "", content)
            content = re.sub(r"\n+", " ", content)
            if len(content) > 300:
                content = content[:297] + "..."
            lines.append(f"- **{c.get('source_date', '?')}:** {marker} {content}")
            result["promoted_items"].append({
                "date": c.get("source_date"),
                "content": content[:100],
                "confidence": c["confidence"],
            })

        lines.append(f"\n*Last auto-promotion: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n")
        result["promotions_made"] = len(result["promoted_items"])

        if not dry_run:
            memory_content = self.md.read_memory()
            pos = self.md.find_insertion_point(memory_content)
            updated = memory_content[:pos] + "\n".join(lines) + memory_content[pos:]
            self.md.write_memory(updated)

        return result

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    @staticmethod
    def cleanup_old_promotions(content: str, days_to_keep: int = 30) -> str:
        """Remove auto-promoted entries older than *days_to_keep* from MEMORY.md content."""
        match = re.search(
            r"## Recent Updates \(Auto-Promoted\)\n(.*?)(?=\n##|\Z)",
            content,
            re.DOTALL,
        )
        if not match:
            return content

        cutoff = (date.today() - timedelta(days=days_to_keep)).isoformat()
        kept: List[str] = []
        for line in match.group(1).split("\n"):
            dm = re.search(r"\*\*(\d{4}-\d{2}-\d{2})\*\*", line)
            if dm and dm.group(1) < cutoff:
                continue
            kept.append(line)

        return content[: match.start(1)] + "\n".join(kept) + content[match.end(1) :]
