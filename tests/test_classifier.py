"""Tests for the SignificanceClassifier."""

from agent_memory_core.classifier import SignificanceClassifier


def test_significant_decision():
    c = SignificanceClassifier()
    is_sig, reason, score = c.classify("Decided to use PostgreSQL for the database architecture")
    assert is_sig is True
    assert score > 0.3


def test_routine_greeting():
    c = SignificanceClassifier()
    is_sig, reason, score = c.classify("hello")
    assert is_sig is False


def test_significant_error():
    c = SignificanceClassifier()
    is_sig, _, score = c.classify("Critical error: database crash, data loss detected. Fix deployed.")
    assert is_sig is True
    assert score > 0.5


def test_custom_threshold():
    c = SignificanceClassifier(threshold=10.0)
    is_sig, _, _ = c.classify("Decided on new architecture")
    assert is_sig is False  # threshold too high


def test_llm_fn_fallback():
    """LLM function that returns garbage should fall back to rules."""
    c = SignificanceClassifier(llm_fn=lambda p: "not json")
    is_sig, reason, score = c.classify("Major architecture decision made")
    assert isinstance(is_sig, bool)
    assert isinstance(score, float)


def test_llm_fn_valid():
    c = SignificanceClassifier(llm_fn=lambda p: '{"significant": true, "reason": "test", "confidence": 0.9}')
    is_sig, reason, score = c.classify("anything")
    assert is_sig is True
    assert score == 0.9


def test_classify_chunks():
    c = SignificanceClassifier()
    chunks = [
        {"content": "Decided to deploy the new system architecture using PostgreSQL", "metadata": {"source": "test"}},
        {"content": "hi", "metadata": {"source": "test"}},
        {"content": "Major bug fix: resolved critical database crash that caused data loss", "metadata": {"source": "test"}},
    ]
    results = c.classify_chunks(chunks)
    # "hi" is < 50 chars, should be filtered
    assert len(results) == 2
    assert all(r["is_significant"] for r in results)
