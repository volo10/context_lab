"""Unit tests for core functions in context_lab."""

import pytest
import numpy as np
from context_lab import (
    generate_filler_text,
    embed_critical_fact,
    split_documents,
    count_tokens,
    evaluate_accuracy,
    SimpleVectorStore,
    SelectStrategy,
    CompressStrategy,
    WriteStrategy,
    IsolateStrategy,
)
from context_lab.context_lab import AgentAction


class TestGenerateFillerText:
    """Tests for generate_filler_text function."""
    
    def test_basic_generation(self):
        """Test basic text generation."""
        text = generate_filler_text(100)
        assert isinstance(text, str)
        assert len(text) > 0
        words = text.split()
        assert 80 <= len(words) <= 120  # Allow some variance
    
    def test_different_lengths(self):
        """Test generation with different lengths."""
        for num_words in [10, 50, 100, 200]:
            text = generate_filler_text(num_words)
            words = text.split()
            # Allow 20% variance
            assert num_words * 0.8 <= len(words) <= num_words * 1.2
    
    def test_english_domain(self):
        """Test English domain text generation."""
        text = generate_filler_text(50, domain="general")
        assert isinstance(text, str)
        assert len(text) > 0

    def test_hebrew_domain(self):
        """Test Hebrew domain text generation."""
        text = generate_filler_text(50, domain="tech_hebrew")
        assert isinstance(text, str)
        # Check for Hebrew characters
        assert any('\u0590' <= c <= '\u05FF' for c in text)


class TestEmbedCriticalFact:
    """Tests for embed_critical_fact function."""

    def test_embed_at_start(self):
        """Test embedding fact at start."""
        doc = "This is a test document with many words."
        fact = "CRITICAL_FACT"
        result = embed_critical_fact(doc, fact, "start")
        assert fact in result
        assert result.index(fact) < len(doc) / 3

    def test_embed_at_middle(self):
        """Test embedding fact in middle."""
        doc = "This is a test document with many words. " * 10
        fact = "CRITICAL_FACT"
        result = embed_critical_fact(doc, fact, "middle")
        assert fact in result
        # Fact should be roughly in the middle
        fact_pos = result.index(fact)
        assert len(result) * 0.3 < fact_pos < len(result) * 0.7

    def test_embed_at_end(self):
        """Test embedding fact at end."""
        doc = "This is a test document with many words."
        fact = "CRITICAL_FACT"
        result = embed_critical_fact(doc, fact, "end")
        assert fact in result
        assert result.index(fact) > len(doc) * 0.7

    def test_invalid_position(self):
        """Test with invalid position raises ValueError."""
        doc = "Test document"
        fact = "FACT"
        # Should raise ValueError for invalid position
        import pytest
        with pytest.raises(ValueError):
            embed_critical_fact(doc, fact, "INVALID")


class TestSplitDocuments:
    """Tests for split_documents function."""
    
    def test_basic_splitting(self):
        """Test basic document splitting."""
        docs = ["This is document one. " * 100, "This is document two. " * 100]
        chunks = split_documents(docs, chunk_size=50)
        assert isinstance(chunks, list)
        assert len(chunks) > len(docs)  # Should be split into multiple chunks
        assert all(isinstance(chunk, str) for chunk in chunks)
    
    def test_chunk_size_respected(self):
        """Test that chunk size is approximately respected."""
        docs = ["word " * 500]  # 500 words
        chunks = split_documents(docs, chunk_size=100)
        # Each chunk should be roughly 100 words
        for chunk in chunks:
            words = chunk.split()
            assert len(words) <= 150  # Allow some overlap
    
    def test_empty_documents(self):
        """Test with empty document list."""
        chunks = split_documents([], chunk_size=100)
        assert chunks == []
    
    def test_single_short_document(self):
        """Test with single short document."""
        docs = ["Short document."]
        chunks = split_documents(docs, chunk_size=100)
        assert len(chunks) >= 1
        assert chunks[0] == docs[0]


class TestCountTokens:
    """Tests for count_tokens function."""
    
    def test_basic_counting(self):
        """Test basic token counting."""
        text = "This is a test sentence."
        count = count_tokens(text)
        assert isinstance(count, int)
        assert count > 0
        assert count <= len(text.split()) * 2  # Rough upper bound
    
    def test_empty_text(self):
        """Test with empty text."""
        count = count_tokens("")
        assert count == 0
    
    def test_long_text(self):
        """Test with long text."""
        text = "word " * 1000
        count = count_tokens(text)
        assert count > 500  # Should be at least half the word count
    
    def test_unicode_text(self):
        """Test with Unicode (Hebrew) text."""
        text = "שלום עולם זה טקסט בעברית"
        count = count_tokens(text)
        assert count > 0


class TestEvaluateAccuracy:
    """Tests for evaluate_accuracy function."""
    
    def test_exact_match(self):
        """Test with exact match."""
        response = "The secret code is ALPHA0BETA1234"
        expected = "The secret code is ALPHA0BETA1234"
        accuracy = evaluate_accuracy(response, expected)
        assert accuracy == 1.0
    
    def test_substring_match(self):
        """Test with substring match."""
        response = "Based on the context, the secret code is ALPHA0BETA1234 as mentioned."
        expected = "ALPHA0BETA1234"
        accuracy = evaluate_accuracy(response, expected)
        assert accuracy == 1.0
    
    def test_no_match(self):
        """Test with no match."""
        response = "I couldn't find the information."
        expected = "ALPHA0BETA1234"
        accuracy = evaluate_accuracy(response, expected)
        assert accuracy == 0.0
    
    def test_partial_match_hebrew(self):
        """Test with partial match in Hebrew."""
        response = "תופעות הלוואי כוללות בחילות וכאבי בטן"
        expected = "בחילות כאבי בטן צרבת"
        accuracy = evaluate_accuracy(response, expected)
        assert accuracy > 0.0  # Should find some keywords
    
    def test_no_expected_answer(self):
        """Test with no expected answer."""
        response = "Some response text"
        accuracy = evaluate_accuracy(response, None)
        assert accuracy == 0.5  # Should return 0.5 (uncertain) for non-empty response without expected answer


class TestSimpleVectorStore:
    """Tests for SimpleVectorStore class."""
    
    def test_initialization(self):
        """Test vector store initialization."""
        store = SimpleVectorStore(use_real=False)
        assert store is not None
        assert store.chunks == []
        assert store.embeddings_list == []
    
    def test_add_chunks_simulation(self):
        """Test adding chunks in simulation mode."""
        store = SimpleVectorStore(use_real=False)
        chunks = ["chunk1", "chunk2", "chunk3"]
        embeddings = [np.random.randn(384) for _ in chunks]
        store.add(chunks, embeddings)
        assert len(store.chunks) == 3
    
    def test_similarity_search_simulation(self):
        """Test similarity search in simulation mode."""
        store = SimpleVectorStore(use_real=False)
        chunks = ["chunk1", "chunk2", "chunk3"]
        embeddings = [np.random.randn(384) for _ in chunks]
        store.add(chunks, embeddings)
        
        results = store.similarity_search("query", k=2)
        assert isinstance(results, list)
        assert len(results) <= 2
    
    def test_empty_search(self):
        """Test search on empty store."""
        store = SimpleVectorStore(use_real=False)
        results = store.similarity_search("query", k=3)
        assert results == []
    
    def test_k_larger_than_chunks(self):
        """Test when k is larger than available chunks."""
        store = SimpleVectorStore(use_real=False)
        chunks = ["chunk1", "chunk2"]
        embeddings = [np.random.randn(384) for _ in chunks]
        store.add(chunks, embeddings)
        
        results = store.similarity_search("query", k=10)
        assert len(results) <= 2  # Should return at most available chunks


class TestSelectStrategy:
    """Tests for SelectStrategy class."""

    def test_initialization(self):
        """Test strategy initialization."""
        strategy = SelectStrategy()
        assert strategy.name == "SELECT (RAG)"
        assert len(strategy.history) == 0

    def test_add_interaction(self):
        """Test adding interactions."""
        strategy = SelectStrategy()
        action = AgentAction(step=1, action_type="query", content="Test content", response="Test response")
        strategy.add_interaction(action)
        assert len(strategy.history) == 1

    def test_get_context(self):
        """Test getting context from history."""
        strategy = SelectStrategy()
        action = AgentAction(step=1, action_type="query", content="Test content", response="Test response")
        strategy.add_interaction(action)
        context = strategy.get_context("query")
        assert isinstance(context, str)


class TestCompressStrategy:
    """Tests for CompressStrategy class."""

    def test_initialization(self):
        """Test strategy initialization."""
        strategy = CompressStrategy(token_limit=1000)
        assert strategy.name == "COMPRESS (Summarization)"
        assert strategy.token_limit == 1000

    def test_add_interaction(self):
        """Test adding interactions."""
        strategy = CompressStrategy()
        action = AgentAction(step=1, action_type="analyze", content="Content", response="Response")
        strategy.add_interaction(action)
        assert len(strategy.history) == 1

    def test_get_context_small_history(self):
        """Test getting context when history is small (no compression needed)."""
        strategy = CompressStrategy(token_limit=10000)
        action = AgentAction(step=1, action_type="query", content="Test", response="Response")
        strategy.add_interaction(action)
        context = strategy.get_context("query")
        assert "Step 1" in context

    def test_compression_when_exceeded(self):
        """Test that history is compressed when token limit exceeded."""
        strategy = CompressStrategy(token_limit=100)  # Very low limit
        # Add many interactions to exceed limit
        for i in range(10):
            action = AgentAction(
                step=i+1,
                action_type="query",
                content="Long content " * 20,
                response="Long response " * 20
            )
            strategy.add_interaction(action)
        context = strategy.get_context("query")
        # Should contain SUMMARY indicator when compressed
        assert "[SUMMARY" in context or len(context) < 1000


class TestWriteStrategy:
    """Tests for WriteStrategy class."""

    def test_initialization(self):
        """Test strategy initialization."""
        strategy = WriteStrategy()
        assert strategy.name == "WRITE (Memory)"
        assert len(strategy.scratchpad) == 0

    def test_key_fact_extraction(self):
        """Test that key facts are extracted every 3rd step."""
        strategy = WriteStrategy()
        for i in range(1, 7):
            action = AgentAction(step=i, action_type="analyze", content=f"Content {i}", response=f"Response {i}")
            strategy.add_interaction(action)

        # Facts should be stored for steps 3 and 6
        assert "fact_3" in strategy.scratchpad
        assert "fact_6" in strategy.scratchpad
        assert "fact_1" not in strategy.scratchpad

    def test_get_context(self):
        """Test getting context from scratchpad."""
        strategy = WriteStrategy()
        for i in range(1, 4):
            action = AgentAction(step=i, action_type="query", content=f"Content {i}", response=f"Response {i}")
            strategy.add_interaction(action)
        context = strategy.get_context("query")
        assert "KEY FACTS" in context or context == ""


class TestIsolateStrategy:
    """Tests for IsolateStrategy class."""

    def test_initialization(self):
        """Test strategy initialization."""
        strategy = IsolateStrategy()
        assert strategy.name == "ISOLATE (Compartments)"
        assert len(strategy.compartments) == 0

    def test_compartmentalization(self):
        """Test that interactions are compartmentalized by action type."""
        strategy = IsolateStrategy()
        # Add different action types
        strategy.add_interaction(AgentAction(step=1, action_type="query", content="Query 1", response="R1"))
        strategy.add_interaction(AgentAction(step=2, action_type="analyze", content="Analyze 1", response="R2"))
        strategy.add_interaction(AgentAction(step=3, action_type="query", content="Query 2", response="R3"))

        assert "query" in strategy.compartments
        assert "analyze" in strategy.compartments
        assert len(strategy.compartments["query"]) == 2
        assert len(strategy.compartments["analyze"]) == 1

    def test_get_context_relevant_compartments(self):
        """Test getting context returns relevant compartments."""
        strategy = IsolateStrategy()
        strategy.add_interaction(AgentAction(step=1, action_type="query", content="Query 1", response="R1"))
        strategy.add_interaction(AgentAction(step=2, action_type="search", content="Search 1", response="R2"))

        # Query for "query" should prioritize query compartment
        context = strategy.get_context("query")
        assert isinstance(context, str)

    def test_empty_compartments(self):
        """Test getting context with no history."""
        strategy = IsolateStrategy()
        context = strategy.get_context("query")
        assert context == ""

    def test_max_per_compartment_limit(self):
        """Test that max_per_compartment limits items returned."""
        strategy = IsolateStrategy()
        # Add many items of same type
        for i in range(10):
            strategy.add_interaction(
                AgentAction(step=i+1, action_type="query", content=f"Query {i}", response=f"R{i}")
            )

        context = strategy.get_context("query", max_per_compartment=2)
        # Should only have 2 items from query compartment
        assert context.count("Step") <= 4  # 2 items max, each has 1 "Step"


class TestEdgeCases:
    """Edge case tests for robustness."""

    def test_generate_filler_text_zero_words(self):
        """Test generating zero words."""
        text = generate_filler_text(0)
        assert text == ""

    def test_generate_filler_text_very_large(self):
        """Test generating large text."""
        text = generate_filler_text(1000)
        assert len(text.split()) >= 800  # Allow some variance

    def test_embed_fact_in_very_short_doc(self):
        """Test embedding fact in very short document."""
        doc = "Short."
        fact = "Important fact here."
        result = embed_critical_fact(doc, fact, "middle")
        assert fact in result

    def test_split_documents_single_word_chunks(self):
        """Test splitting with very small chunk size."""
        docs = ["This is a test document."]
        chunks = split_documents(docs, chunk_size=5)
        assert len(chunks) > 0
        assert all(len(chunk) > 0 for chunk in chunks)

    def test_count_tokens_special_characters(self):
        """Test token counting with special characters."""
        text = "Hello! @#$%^&*() World!!!"
        count = count_tokens(text)
        assert count > 0

    def test_evaluate_accuracy_empty_strings(self):
        """Test accuracy evaluation with empty strings."""
        accuracy = evaluate_accuracy("", "expected")
        assert accuracy == 0.0

    def test_evaluate_accuracy_special_patterns(self):
        """Test accuracy with ALPHA*BETA pattern."""
        response = "The code is ALPHA5BETA9999"
        expected = "The secret code is ALPHA5BETA9999"
        accuracy = evaluate_accuracy(response, expected)
        assert accuracy == 1.0

    def test_vector_store_add_without_embeddings(self):
        """Test adding chunks without pre-computed embeddings."""
        store = SimpleVectorStore(use_real=False)
        chunks = ["chunk1", "chunk2"]
        store.add(chunks)  # No embeddings provided
        assert len(store.chunks) == 2

    def test_strategy_token_count(self):
        """Test token counting in strategies."""
        strategy = CompressStrategy(token_limit=100)
        action = AgentAction(step=1, action_type="query", content="Test", response="Response")
        strategy.add_interaction(action)
        count = strategy.get_token_count()
        assert count >= 0

    def test_isolate_strategy_multiple_compartments(self):
        """Test ISOLATE with many different action types."""
        strategy = IsolateStrategy()
        action_types = ["query", "search", "analyze", "summarize", "compare"]
        for i, action_type in enumerate(action_types):
            strategy.add_interaction(
                AgentAction(step=i+1, action_type=action_type, content=f"Content {i}", response=f"R{i}")
            )
        assert len(strategy.compartments) == 5

    def test_hebrew_medical_terms_accuracy(self):
        """Test accuracy evaluation with Hebrew medical terms."""
        response = "תופעות הלוואי כוללות בחילות וכאבי בטן וצרבת"
        expected = "בחילות כאבי בטן צרבת סחרחורת"
        accuracy = evaluate_accuracy(response, expected)
        assert accuracy >= 0.5  # Should find most terms


class TestLoadDocuments:
    """Tests for load_documents function."""

    def test_load_documents_basic(self):
        """Test basic document loading."""
        from context_lab import load_documents
        docs = load_documents(5, words_per_doc=100)
        assert len(docs) == 5
        assert all(len(doc.split()) >= 80 for doc in docs)

    def test_load_documents_diverse_domains(self):
        """Test loading diverse domain documents."""
        from context_lab import load_documents
        docs = load_documents(6, words_per_doc=50, diverse_domains=True)
        assert len(docs) == 6
        # Should have Hebrew content
        has_hebrew = any(any('\u0590' <= c <= '\u05FF' for c in doc) for doc in docs)
        assert has_hebrew


class TestConcatenateDocuments:
    """Tests for concatenate_documents function."""

    def test_concatenate_basic(self):
        """Test basic document concatenation."""
        from context_lab import concatenate_documents
        docs = ["Doc 1", "Doc 2", "Doc 3"]
        result = concatenate_documents(docs)
        assert "Doc 1" in result
        assert "Doc 2" in result
        assert "Doc 3" in result
        assert "---" in result  # Separator

    def test_concatenate_empty(self):
        """Test concatenating empty list."""
        from context_lab import concatenate_documents
        result = concatenate_documents([])
        assert result == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

