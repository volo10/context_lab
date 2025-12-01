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
)


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
        text = generate_filler_text(50, domain="english")
        assert isinstance(text, str)
        assert len(text) > 0
    
    def test_hebrew_domain(self):
        """Test Hebrew domain text generation."""
        text = generate_filler_text(50, domain="technology_hebrew")
        assert isinstance(text, str)
        # Check for Hebrew characters
        assert any('\u0590' <= c <= '\u05FF' for c in text)


class TestEmbedCriticalFact:
    """Tests for embed_critical_fact function."""
    
    def test_embed_at_start(self):
        """Test embedding fact at start."""
        doc = "This is a test document with many words."
        fact = "CRITICAL_FACT"
        result = embed_critical_fact(doc, fact, "START")
        assert fact in result
        assert result.index(fact) < len(doc) / 3
    
    def test_embed_at_middle(self):
        """Test embedding fact in middle."""
        doc = "This is a test document with many words. " * 10
        fact = "CRITICAL_FACT"
        result = embed_critical_fact(doc, fact, "MIDDLE")
        assert fact in result
        # Fact should be roughly in the middle
        fact_pos = result.index(fact)
        assert len(result) * 0.3 < fact_pos < len(result) * 0.7
    
    def test_embed_at_end(self):
        """Test embedding fact at end."""
        doc = "This is a test document with many words."
        fact = "CRITICAL_FACT"
        result = embed_critical_fact(doc, fact, "END")
        assert fact in result
        assert result.index(fact) > len(doc) * 0.7
    
    def test_invalid_position(self):
        """Test with invalid position."""
        doc = "Test document"
        fact = "FACT"
        # Should default to middle or handle gracefully
        result = embed_critical_fact(doc, fact, "INVALID")
        assert fact in result


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
        assert accuracy == 1.0  # Should return 1.0 for non-empty response


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

