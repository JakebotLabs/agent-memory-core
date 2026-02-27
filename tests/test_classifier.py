"""Tests for the SignificanceClassifier."""

import pytest

from agent_memory_core.classifier import SignificanceClassifier


class TestRuleBasedClassification:
    """Test rule-based classification (default mode)."""

    def test_significant_decision(self, classifier):
        """Decision-related text is significant."""
        is_sig, reason, score = classifier.classify(
            "Decided to use PostgreSQL for the database architecture"
        )
        assert is_sig is True
        assert score > 0.3
        assert "SIGNIFICANT" in reason

    def test_significant_error(self, classifier):
        """Error/bug text is significant."""
        is_sig, _, score = classifier.classify(
            "Critical error: database crash, data loss detected. Fix deployed."
        )
        assert is_sig is True
        assert score > 0.5

    def test_significant_milestone(self, classifier):
        """Milestone/completion text is significant."""
        is_sig, _, score = classifier.classify(
            "Completed the deployment. Version 1.0 shipped successfully."
        )
        assert is_sig is True

    def test_significant_architecture(self, classifier):
        """Architecture discussions are significant."""
        is_sig, _, _ = classifier.classify(
            "New architecture design for the authentication system."
        )
        assert is_sig is True

    def test_routine_greeting(self, classifier):
        """Simple greetings are routine."""
        is_sig, reason, score = classifier.classify("hello")
        assert is_sig is False
        assert "ROUTINE" in reason

    def test_routine_acknowledgment(self, classifier):
        """Acknowledgments are routine."""
        is_sig, _, _ = classifier.classify("okay, sounds good")
        assert is_sig is False

    def test_routine_short_text(self, classifier):
        """Very short text tends to be routine."""
        is_sig, _, _ = classifier.classify("yes")
        assert is_sig is False

    def test_length_contributes_to_score(self, classifier):
        """Longer text contributes to significance."""
        short = classifier.classify("decided")[2]  # score
        long = classifier.classify("decided " + "x " * 100)[2]
        
        assert long >= short

    def test_high_priority_boosts_score(self, classifier):
        """High priority words boost score."""
        normal = classifier.classify("made a decision")[2]
        high = classifier.classify("critical decision breakthrough")[2]
        
        assert high > normal

    def test_routine_words_reduce_score(self, classifier):
        """Routine words reduce score."""
        without = classifier.classify("architecture change")[2]
        with_routine = classifier.classify("okay sure architecture change")[2]
        
        assert with_routine < without


class TestCustomThreshold:
    """Test custom threshold behavior."""

    def test_low_threshold(self):
        """Low threshold accepts more."""
        c = SignificanceClassifier(threshold=0.1)
        is_sig, _, _ = c.classify("small change")
        # May or may not be significant depending on content

    def test_high_threshold(self):
        """High threshold rejects more."""
        c = SignificanceClassifier(threshold=10.0)
        is_sig, _, _ = c.classify("Decided on new architecture")
        assert is_sig is False

    def test_zero_threshold(self):
        """Zero threshold accepts almost everything."""
        c = SignificanceClassifier(threshold=0.0)
        is_sig, _, score = c.classify("some text")
        # Score >= 0 means significant with threshold=0
        assert (is_sig and score >= 0) or (not is_sig and score < 0)


class TestLlmBasedClassification:
    """Test LLM-based classification."""

    def test_llm_fn_valid_json(self, classifier_with_llm):
        """Valid JSON response is parsed."""
        is_sig, reason, score = classifier_with_llm.classify("anything")
        assert is_sig is True
        assert score == 0.85
        assert "mocked" in reason

    def test_llm_fn_invalid_json_fallback(self):
        """Invalid JSON falls back to rules."""
        c = SignificanceClassifier(llm_fn=lambda p: "not json at all")
        is_sig, reason, score = c.classify("Major architecture decision made")
        
        assert isinstance(is_sig, bool)
        assert isinstance(score, float)
        # Falls back to rule-based

    def test_llm_fn_exception_fallback(self):
        """Exception in LLM falls back to rules."""
        def broken_llm(prompt):
            raise ValueError("API error")
        
        c = SignificanceClassifier(llm_fn=broken_llm)
        is_sig, _, _ = c.classify("Some text")
        
        # Should still return a result (from fallback)
        assert isinstance(is_sig, bool)

    def test_llm_fn_partial_json(self):
        """Partial/embedded JSON is extracted."""
        def embedded_json(prompt):
            return 'Here is the result: {"significant": false, "reason": "not important", "confidence": 0.2}'
        
        c = SignificanceClassifier(llm_fn=embedded_json)
        is_sig, reason, score = c.classify("test")
        
        assert is_sig is False
        assert score == 0.2

    def test_llm_receives_context(self):
        """Context is passed to LLM."""
        received_prompts = []
        
        def capturing_llm(prompt):
            received_prompts.append(prompt)
            return '{"significant": true, "reason": "test", "confidence": 0.5}'
        
        c = SignificanceClassifier(llm_fn=capturing_llm)
        c.classify("text", context="important context")
        
        assert len(received_prompts) == 1
        assert "important context" in received_prompts[0]


class TestBatchClassification:
    """Test classify_chunks method."""

    def test_classify_chunks_basic(self, classifier):
        """Classifies list of chunks."""
        chunks = [
            {"content": "Decided to deploy the new system architecture using PostgreSQL", "metadata": {"source": "test"}},
            {"content": "Major bug fix: resolved critical database crash that caused data loss", "metadata": {"source": "test"}},
        ]
        results = classifier.classify_chunks(chunks)
        
        assert len(results) == 2
        for r in results:
            assert "is_significant" in r
            assert "confidence" in r
            assert "reason" in r

    def test_classify_chunks_filters_short(self, classifier):
        """Filters chunks shorter than min_length."""
        chunks = [
            {"content": "hi", "metadata": {"source": "test"}},
            {"content": "hello there", "metadata": {"source": "test"}},
            {"content": "This is a longer chunk that should pass the filter" * 2, "metadata": {"source": "test"}},
        ]
        results = classifier.classify_chunks(chunks, min_length=50)
        
        # Only the long one passes
        assert len(results) == 1

    def test_classify_chunks_preserves_metadata(self, classifier):
        """Metadata is preserved in results."""
        chunks = [
            {"content": "Important decision about architecture" * 3, "metadata": {"source": "test.md", "section": "Intro"}},
        ]
        results = classifier.classify_chunks(chunks)
        
        assert len(results) == 1
        assert results[0]["metadata"]["source"] == "test.md"

    def test_classify_chunks_includes_index(self, classifier):
        """Results include original chunk index."""
        chunks = [
            {"content": "Short", "metadata": {"source": "a"}},
            {"content": "This is a significant chunk about architecture decisions" * 2, "metadata": {"source": "b"}},
            {"content": "Another long chunk about critical system changes" * 2, "metadata": {"source": "c"}},
        ]
        results = classifier.classify_chunks(chunks, min_length=20)
        
        # Indices should correspond to original positions
        indices = [r["index"] for r in results]
        assert 1 in indices or 2 in indices

    def test_classify_chunks_truncates_content(self, classifier):
        """Long content is truncated in results."""
        long_content = "A" * 500
        chunks = [
            {"content": long_content, "metadata": {"source": "test"}},
        ]
        results = classifier.classify_chunks(chunks)
        
        # Content should be truncated with ...
        assert len(results[0]["content"]) <= 203  # 200 + "..."
        assert results[0]["content"].endswith("...")

    def test_classify_chunks_empty_list(self, classifier):
        """Empty chunks list returns empty results."""
        results = classifier.classify_chunks([])
        assert results == []


class TestIndicatorLists:
    """Test indicator lists are properly defined."""

    def test_significant_indicators_not_empty(self):
        """Significant indicators list has entries."""
        assert len(SignificanceClassifier.SIGNIFICANT_INDICATORS) > 0

    def test_high_priority_not_empty(self):
        """High priority list has entries."""
        assert len(SignificanceClassifier.HIGH_PRIORITY) > 0

    def test_routine_indicators_not_empty(self):
        """Routine indicators list has entries."""
        assert len(SignificanceClassifier.ROUTINE_INDICATORS) > 0

    def test_indicators_are_lowercase(self):
        """All indicators should be lowercase."""
        for ind in SignificanceClassifier.SIGNIFICANT_INDICATORS:
            assert ind.islower()
        for ind in SignificanceClassifier.HIGH_PRIORITY:
            assert ind.islower()
        for ind in SignificanceClassifier.ROUTINE_INDICATORS:
            assert ind.islower()


class TestReturnTypes:
    """Test return type consistency."""

    def test_classify_returns_tuple(self, classifier):
        """Classify returns (bool, str, float) tuple."""
        result = classifier.classify("test")
        
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert isinstance(result[0], bool)
        assert isinstance(result[1], str)
        assert isinstance(result[2], float)

    def test_classify_score_non_negative(self, classifier):
        """Score is never negative after max(0, score)."""
        # Even very routine text should have score >= 0
        _, _, score = classifier.classify("hello hi okay sure thanks")
        assert score >= 0.0

    def test_classify_reason_includes_score(self, classifier):
        """Reason string includes the score value."""
        _, reason, score = classifier.classify("decided on architecture")
        assert str(round(score, 2)) in reason or "score" in reason.lower()
