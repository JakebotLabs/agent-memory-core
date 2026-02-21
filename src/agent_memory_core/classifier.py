"""
Significance Classifier — Determines if content should be promoted to long-term memory.

Supports two modes:
1. Rule-based (default): Fast keyword/heuristic scoring, no LLM needed.
2. LLM-based: Pass a callable ``llm_fn(prompt) -> str`` for richer classification.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Tuple

__all__ = ["SignificanceClassifier"]


class SignificanceClassifier:
    """Classify interactions for memory significance."""

    SIGNIFICANT_INDICATORS = [
        "decided", "chose", "selected", "architecture", "design", "approach",
        "strategy", "plan", "direction", "solution",
        "completed", "finished", "deployed", "implemented", "fixed", "resolved",
        "updated", "changed", "modified", "installed", "configured",
        "learned", "discovered", "found", "realized", "insight", "mistake",
        "error", "problem", "issue", "bug", "failure", "works", "doesn't work",
        "blocked", "waiting", "dependency", "requires", "needs", "missing",
        "milestone", "release", "version", "complete", "ready", "shipped",
        "research", "analysis", "findings", "conclusion", "recommendation",
    ]

    HIGH_PRIORITY = [
        "error", "failure", "crash", "bug", "fix", "solve",
        "breakthrough", "discovery", "major", "critical", "important",
        "decision", "architecture", "design", "strategy", "direction",
    ]

    ROUTINE_INDICATORS = [
        "hello", "hi", "thanks", "okay", "sure", "sounds good",
        "got it", "understood", "yes", "no", "maybe", "hmm",
        "checking", "looking", "reviewing", "reading", "browsing",
    ]

    def __init__(
        self,
        threshold: float = 0.3,
        llm_fn: Optional[Callable[[str], str]] = None,
    ) -> None:
        """
        Args:
            threshold: Minimum score to consider content significant.
            llm_fn: Optional ``fn(prompt) -> response_text`` for LLM-based classification.
                     Provider-agnostic — works with OpenAI, Anthropic, Ollama, etc.
        """
        self.threshold = threshold
        self.llm_fn = llm_fn

    def classify(self, text: str, context: str = "") -> Tuple[bool, str, float]:
        """Classify *text* for significance.

        Returns ``(is_significant, reason, score)``.
        """
        if self.llm_fn is not None:
            return self._classify_llm(text, context)
        return self._classify_rules(text, context)

    # ------------------------------------------------------------------
    # Rule-based
    # ------------------------------------------------------------------

    def _classify_rules(self, text: str, context: str = "") -> Tuple[bool, str, float]:
        combined = f"{text} {context}".lower()

        sig_count = sum(1 for w in self.SIGNIFICANT_INDICATORS if w in combined)
        high_count = sum(1 for w in self.HIGH_PRIORITY if w in combined)
        routine_count = sum(1 for w in self.ROUTINE_INDICATORS if w in combined)
        length_score = min(len(text) / 200.0, 1.0)

        score = sig_count * 0.3 + length_score * 0.2 + high_count * 0.5 - routine_count * 0.1
        score = max(0.0, score)

        is_sig = score >= self.threshold

        if is_sig:
            parts = []
            if sig_count:
                parts.append(f"{sig_count} indicators")
            if high_count:
                parts.append(f"{high_count} high-priority")
            reason = f"SIGNIFICANT: {', '.join(parts)} (score: {score:.2f})"
        else:
            reason = f"ROUTINE: score {score:.2f} below threshold {self.threshold}"

        return is_sig, reason, score

    # ------------------------------------------------------------------
    # LLM-based
    # ------------------------------------------------------------------

    def _classify_llm(self, text: str, context: str = "") -> Tuple[bool, str, float]:
        prompt = (
            "You are a memory significance classifier for an AI agent.\n"
            "Determine if the following interaction should be saved to long-term memory.\n"
            "Respond with JSON: {\"significant\": true/false, \"reason\": \"...\", \"confidence\": 0.0-1.0}\n\n"
            f"Context: {context}\n\n"
            f"Interaction:\n{text}\n"
        )
        try:
            response = self.llm_fn(prompt)  # type: ignore[misc]
            import json
            # Try to extract JSON from the response
            match = re.search(r"\{.*?\}", response, re.DOTALL)
            if match:
                data = json.loads(match.group())
                return (
                    bool(data.get("significant", False)),
                    data.get("reason", "LLM classified"),
                    float(data.get("confidence", 0.5)),
                )
        except Exception:
            pass
        # Fallback to rules if LLM fails
        return self._classify_rules(text, context)

    # ------------------------------------------------------------------
    # Batch
    # ------------------------------------------------------------------

    def classify_chunks(self, chunks: List[Dict[str, Any]], min_length: int = 50) -> List[Dict[str, Any]]:
        """Classify a list of parsed chunks. Returns classification results."""
        results: List[Dict[str, Any]] = []
        for i, chunk in enumerate(chunks):
            content = chunk.get("content", "")
            if len(content.strip()) < min_length:
                continue
            is_sig, reason, score = self.classify(content)
            results.append({
                "index": i,
                "content": content[:200] + "..." if len(content) > 200 else content,
                "is_significant": is_sig,
                "reason": reason,
                "confidence": score,
                "metadata": chunk.get("metadata", {}),
            })
        return results
