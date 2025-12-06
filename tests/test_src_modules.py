"""
Tests for src/ modular structure
================================
Additional tests to increase coverage of the new modular architecture.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

# Import from src modules
from src.llm import (
    count_tokens,
    evaluate_accuracy,
    ollama_query,
    _detect_hebrew,
    _build_hebrew_prompt,
    _build_english_prompt,
    _simulate_lost_in_middle,
    REAL_LLM_AVAILABLE,
)
from src.utils import (
    generate_filler_text,
    embed_critical_fact,
    split_documents,
    load_documents,
    concatenate_documents,
    nomic_embed_text,
    FILLER_PHRASES,
)
from src.vector_store import SimpleVectorStore
from src.strategies import (
    AgentAction,
    ContextStrategy,
    SelectStrategy,
    CompressStrategy,
    WriteStrategy,
    IsolateStrategy,
)
from src.experiments import (
    simulate_agent_conversation,
    evaluate_strategy,
)


class TestLLMModule:
    """Tests for src/llm.py"""

    def test_detect_hebrew_with_hebrew_text(self):
        """Test Hebrew detection with Hebrew text."""
        assert _detect_hebrew("שלום עולם") == True
        assert _detect_hebrew("מה שלומך?") == True

    def test_detect_hebrew_with_english_text(self):
        """Test Hebrew detection with English text."""
        assert _detect_hebrew("Hello world") == False
        assert _detect_hebrew("How are you?") == False

    def test_detect_hebrew_mixed_text(self):
        """Test Hebrew detection with mixed text."""
        assert _detect_hebrew("Hello שלום world") == True

    def test_build_hebrew_prompt(self):
        """Test Hebrew prompt building."""
        prompt = _build_hebrew_prompt("context text", "שאלה בעברית")
        assert "Hebrew" in prompt
        assert "context text" in prompt
        assert "שאלה בעברית" in prompt

    def test_build_english_prompt(self):
        """Test English prompt building."""
        prompt = _build_english_prompt("my context", "my question")
        assert "my context" in prompt
        assert "my question" in prompt

    def test_simulate_lost_in_middle_at_start(self):
        """Test simulation returns correct for fact at start."""
        context = "CRITICAL_FACT: important " + "x " * 1000
        response = _simulate_lost_in_middle(context, "find the critical fact")
        # Should usually find it at the start
        assert response in ["The critical fact is: [Extracted correctly]", "Not found"]

    def test_simulate_lost_in_middle_in_middle(self):
        """Test simulation for fact in middle."""
        context = "x " * 500 + "CRITICAL_FACT: important " + "x " * 500
        response = _simulate_lost_in_middle(context, "find the critical fact")
        assert response in ["The critical fact is: [Extracted correctly]", "Not found"]

    def test_simulate_lost_in_middle_no_fact(self):
        """Test simulation when no fact present."""
        context = "x " * 1000
        response = _simulate_lost_in_middle(context, "find the critical fact")
        assert response in ["The critical fact is: [Extracted correctly]", "Not found"]

    def test_count_tokens_various_lengths(self):
        """Test token counting with various lengths."""
        assert count_tokens("") == 0
        assert count_tokens("a") == 0  # 1 // 4 = 0
        assert count_tokens("abcd") == 1  # 4 // 4 = 1
        assert count_tokens("abcdefgh") == 2  # 8 // 4 = 2

    def test_ollama_query_simulation_mode(self):
        """Test ollama_query in simulation mode."""
        response = ollama_query("test context", "test query", use_real=False)
        assert isinstance(response, str)
        assert len(response) > 0

    def test_ollama_query_with_critical_fact_query(self):
        """Test ollama_query with critical fact query."""
        response = ollama_query("CRITICAL_FACT: test", "What is the critical fact?", use_real=False)
        assert isinstance(response, str)

    def test_evaluate_accuracy_alpha_beta_pattern(self):
        """Test accuracy evaluation with ALPHA*BETA* pattern."""
        assert evaluate_accuracy("The code is ALPHA0BETA1234", "ALPHA0BETA1234") == 1.0
        assert evaluate_accuracy("Some text ALPHA5BETA9999 more text", "ALPHA5BETA9999") == 1.0
        assert evaluate_accuracy("Wrong answer", "ALPHA0BETA1234") == 0.0

    def test_evaluate_accuracy_hebrew_terms(self):
        """Test accuracy with Hebrew medical terms."""
        expected = "בחילות וכאבי בטן"
        assert evaluate_accuracy("בחילות", expected) > 0
        assert evaluate_accuracy("כאבי בטן", expected) > 0
        assert evaluate_accuracy("שום דבר", expected) == 0.0

    def test_evaluate_accuracy_negative_indicators(self):
        """Test accuracy with negative response indicators."""
        assert evaluate_accuracy("I don't know the answer", None) == 0.0
        assert evaluate_accuracy("Not found in the context", None) == 0.0
        assert evaluate_accuracy("אין מידע על כך", None) == 0.0


class TestUtilsModule:
    """Tests for src/utils.py"""

    def test_filler_phrases_all_domains(self):
        """Test all domain phrase banks exist."""
        assert "medical_hebrew" in FILLER_PHRASES
        assert "tech_hebrew" in FILLER_PHRASES
        assert "legal_hebrew" in FILLER_PHRASES
        assert "general" in FILLER_PHRASES

    def test_generate_filler_text_all_domains(self):
        """Test filler text generation for all domains."""
        for domain in ["medical_hebrew", "tech_hebrew", "legal_hebrew", "general"]:
            text = generate_filler_text(50, domain=domain)
            assert len(text.split()) >= 50

    def test_generate_filler_text_unknown_domain(self):
        """Test filler text with unknown domain falls back to general."""
        text = generate_filler_text(50, domain="unknown_domain")
        assert len(text.split()) >= 50

    def test_embed_critical_fact_all_positions(self):
        """Test fact embedding at all positions."""
        base = "word " * 100
        fact = "IMPORTANT_FACT"

        for pos in ["start", "middle", "end"]:
            result = embed_critical_fact(base, fact, pos)
            assert "IMPORTANT_FACT" in result

    def test_embed_critical_fact_position_calculation(self):
        """Test fact is placed at correct position."""
        base = "word " * 100  # 100 words
        fact = "FACT"

        # Start should be near beginning (5%)
        start_result = embed_critical_fact(base, fact, "start")
        start_words = start_result.split()
        start_idx = start_words.index("FACT")
        assert start_idx < 10  # Should be in first 10 words

    def test_split_documents_preserves_content(self):
        """Test split documents preserves all content."""
        docs = ["word " * 50, "other " * 50]
        chunks = split_documents(docs, chunk_size=100)

        # All content should be preserved
        all_chunks = " ".join(chunks)
        assert "word" in all_chunks
        assert "other" in all_chunks

    def test_load_documents_diverse_rotates_domains(self):
        """Test diverse documents rotate through domains."""
        docs = load_documents(6, words_per_doc=50, diverse_domains=True)
        assert len(docs) == 6
        # Each doc should have content (hard to check domain without parsing)
        for doc in docs:
            assert len(doc.split()) >= 50

    def test_concatenate_documents_separator(self):
        """Test concatenation uses correct separator."""
        docs = ["doc1", "doc2", "doc3"]
        result = concatenate_documents(docs)
        assert "---" in result
        assert "doc1" in result
        assert "doc2" in result
        assert "doc3" in result

    def test_nomic_embed_text_simulation(self):
        """Test embedding in simulation mode returns correct shape."""
        chunks = ["chunk1", "chunk2", "chunk3"]
        embeddings = nomic_embed_text(chunks, use_real=False)
        assert len(embeddings) == 3
        assert len(embeddings[0]) == 384  # Standard embedding dimension


class TestVectorStoreModule:
    """Tests for src/vector_store.py"""

    def test_vector_store_simulation_mode(self):
        """Test vector store in simulation mode."""
        store = SimpleVectorStore(use_real=False)
        assert store.use_real == False

    def test_vector_store_add_and_search(self):
        """Test basic add and search functionality."""
        store = SimpleVectorStore(use_real=False)
        store.add(["chunk1", "chunk2", "chunk3"])

        results = store.similarity_search("query", k=2)
        assert len(results) == 2
        assert all(r in ["chunk1", "chunk2", "chunk3"] for r in results)

    def test_vector_store_len(self):
        """Test __len__ returns chunk count."""
        store = SimpleVectorStore(use_real=False)
        assert len(store) == 0

        store.add(["c1", "c2"])
        assert len(store) == 2

        store.add(["c3"])
        assert len(store) == 3

    def test_vector_store_clear(self):
        """Test clear functionality."""
        store = SimpleVectorStore(use_real=False)
        store.add(["c1", "c2", "c3"])
        assert len(store) == 3

        store.clear()
        assert len(store) == 0
        assert store.chunks == []
        assert store.embeddings_list == []

    def test_vector_store_k_larger_than_available(self):
        """Test search with k larger than available chunks."""
        store = SimpleVectorStore(use_real=False)
        store.add(["c1", "c2"])

        results = store.similarity_search("query", k=10)
        assert len(results) == 2  # Only 2 available


class TestStrategiesModule:
    """Tests for src/strategies.py"""

    def test_agent_action_dataclass(self):
        """Test AgentAction dataclass."""
        action = AgentAction(
            step=1,
            action_type="query",
            content="test content",
            response="test response"
        )
        assert action.step == 1
        assert action.action_type == "query"
        assert action.content == "test content"
        assert action.response == "test response"

    def test_select_strategy_reset(self):
        """Test strategy reset functionality."""
        strategy = SelectStrategy()
        action = AgentAction(1, "query", "content", "response")
        strategy.add_interaction(action)
        assert len(strategy.history) == 1

        strategy.reset()
        assert len(strategy.history) == 0

    def test_compress_strategy_token_limit(self):
        """Test compress strategy respects token limit."""
        strategy = CompressStrategy(token_limit=100)

        # Add many interactions
        for i in range(20):
            action = AgentAction(i, "query", "long " * 50, "response " * 50)
            strategy.add_interaction(action)

        context = strategy.get_context("query")
        tokens = count_tokens(context)
        # Should be compressed
        assert tokens < count_tokens(" ".join(["long " * 50] * 20))

    def test_write_strategy_extraction_interval(self):
        """Test write strategy extracts at correct interval."""
        strategy = WriteStrategy(extract_every=2)

        for i in range(6):
            action = AgentAction(i + 1, "query", "content", "response")
            strategy.add_interaction(action)

        # Should have facts at steps 2, 4, 6
        assert "fact_2" in strategy.scratchpad
        assert "fact_4" in strategy.scratchpad
        assert "fact_6" in strategy.scratchpad
        assert "fact_1" not in strategy.scratchpad

    def test_isolate_strategy_query_matching(self):
        """Test isolate strategy matches query to compartments."""
        strategy = IsolateStrategy()

        strategy.add_interaction(AgentAction(1, "search", "content", "response"))
        strategy.add_interaction(AgentAction(2, "analyze", "content", "response"))
        strategy.add_interaction(AgentAction(3, "search", "content", "response"))

        # Query mentioning "search" should return search compartment
        context = strategy.get_context("search for something")
        assert "SEARCH" in context.upper()


class TestExperimentsModule:
    """Tests for src/experiments.py"""

    def test_simulate_agent_conversation(self):
        """Test conversation simulation."""
        actions = simulate_agent_conversation(num_steps=5)
        assert len(actions) == 5

        for i, action in enumerate(actions):
            assert action.step == i + 1
            assert action.action_type in ["query", "search", "analyze", "summarize", "compare"]
            assert len(action.content) > 0
            assert len(action.response) > 0

    def test_evaluate_strategy_metrics(self):
        """Test strategy evaluation returns correct metrics."""
        strategy = WriteStrategy()
        actions = simulate_agent_conversation(num_steps=3)

        metrics = evaluate_strategy(strategy, actions, use_real_llm=False)

        assert "avg_accuracy" in metrics
        assert "std_accuracy" in metrics
        assert "avg_latency" in metrics
        assert "avg_tokens" in metrics
        assert "final_tokens" in metrics

        assert 0 <= metrics["avg_accuracy"] <= 1
        assert metrics["avg_latency"] >= 0
        assert metrics["avg_tokens"] >= 0

    def test_evaluate_strategy_all_strategies(self):
        """Test all strategies can be evaluated."""
        strategies = [
            SelectStrategy(),
            CompressStrategy(),
            WriteStrategy(),
            IsolateStrategy()
        ]
        actions = simulate_agent_conversation(num_steps=3)

        for strategy in strategies:
            metrics = evaluate_strategy(strategy, actions.copy(), use_real_llm=False)
            assert isinstance(metrics, dict)
            assert "avg_accuracy" in metrics


class TestIntegration:
    """Integration tests across modules."""

    def test_full_rag_pipeline_simulation(self):
        """Test full RAG pipeline in simulation mode."""
        # Generate documents
        docs = load_documents(5, words_per_doc=100, diverse_domains=True)

        # Split and embed
        chunks = split_documents(docs, chunk_size=200)
        embeddings = nomic_embed_text(chunks, use_real=False)

        # Store in vector store
        store = SimpleVectorStore(use_real=False)
        store.add(chunks, embeddings)

        # Search
        results = store.similarity_search("test query", k=3)
        assert len(results) == 3

        # Build context
        context = concatenate_documents(results)
        assert len(context) > 0

    def test_strategy_with_real_actions(self):
        """Test strategy with realistic action sequence."""
        strategy = SelectStrategy()

        # Simulate realistic conversation
        actions = [
            AgentAction(1, "query", "What is the weather?", "It's sunny today"),
            AgentAction(2, "search", "Weather forecast", "Tomorrow will be cloudy"),
            AgentAction(3, "analyze", "Weather patterns", "Pattern shows warming trend"),
            AgentAction(4, "query", "Tell me about weather", "Based on analysis...")
        ]

        for action in actions:
            strategy.add_interaction(action)

        # Get context for weather query
        context = strategy.get_context("weather information")
        assert len(context) > 0
