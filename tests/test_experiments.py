"""Unit tests for experiment functions."""

import pytest
from context_lab import (
    experiment1_needle_in_haystack,
    experiment2_context_size_impact,
    experiment3_rag_vs_full_context,
    experiment4_context_strategies,
)


class TestExperiment1:
    """Tests for Experiment 1: Needle in Haystack."""
    
    def test_basic_execution(self):
        """Test basic experiment execution."""
        results = experiment1_needle_in_haystack(num_docs=3, use_real_llm=False)
        assert isinstance(results, dict)
        assert "position_accuracy" in results
        assert "average_latency" in results
    
    def test_position_accuracy_keys(self):
        """Test that all position keys are present."""
        results = experiment1_needle_in_haystack(num_docs=3, use_real_llm=False)
        positions = results["position_accuracy"]
        assert "START" in positions
        assert "MIDDLE" in positions
        assert "END" in positions
    
    def test_accuracy_range(self):
        """Test that accuracy values are in valid range."""
        results = experiment1_needle_in_haystack(num_docs=3, use_real_llm=False)
        for position, accuracy in results["position_accuracy"].items():
            assert 0.0 <= accuracy <= 1.0
    
    def test_latency_positive(self):
        """Test that latency is positive."""
        results = experiment1_needle_in_haystack(num_docs=3, use_real_llm=False)
        assert results["average_latency"] >= 0


class TestExperiment2:
    """Tests for Experiment 2: Context Window Size Impact."""
    
    def test_basic_execution(self):
        """Test basic experiment execution."""
        results = experiment2_context_size_impact(
            doc_counts=[2, 5],
            use_real_llm=False
        )
        assert isinstance(results, dict)
        assert "results" in results
    
    def test_result_structure(self):
        """Test that each result has required keys."""
        results = experiment2_context_size_impact(
            doc_counts=[2, 5],
            use_real_llm=False
        )
        for result in results["results"]:
            assert "num_docs" in result
            assert "tokens_used" in result
            assert "latency" in result
            assert "accuracy" in result
    
    def test_increasing_doc_counts(self):
        """Test with increasing document counts."""
        results = experiment2_context_size_impact(
            doc_counts=[2, 5, 10],
            use_real_llm=False
        )
        # Token count should generally increase
        tokens = [r["tokens_used"] for r in results["results"]]
        assert tokens[0] < tokens[-1]
    
    def test_metrics_validity(self):
        """Test that all metrics are valid."""
        results = experiment2_context_size_impact(
            doc_counts=[2],
            use_real_llm=False
        )
        result = results["results"][0]
        assert result["num_docs"] > 0
        assert result["tokens_used"] > 0
        assert result["latency"] >= 0
        assert 0.0 <= result["accuracy"] <= 1.0


class TestExperiment3:
    """Tests for Experiment 3: RAG vs Full Context."""
    
    def test_basic_execution(self):
        """Test basic experiment execution."""
        results = experiment3_rag_vs_full_context(
            num_docs=5,
            use_real_llm=False
        )
        assert isinstance(results, dict)
        assert "full_context" in results
        assert "rag" in results
        assert "comparison" in results
    
    def test_full_context_structure(self):
        """Test full context result structure."""
        results = experiment3_rag_vs_full_context(
            num_docs=5,
            use_real_llm=False
        )
        full = results["full_context"]
        assert "tokens" in full
        assert "latency" in full
        assert "accuracy" in full
    
    def test_rag_structure(self):
        """Test RAG result structure."""
        results = experiment3_rag_vs_full_context(
            num_docs=5,
            use_real_llm=False
        )
        rag = results["rag"]
        assert "tokens" in rag
        assert "latency" in rag
        assert "accuracy" in rag
    
    def test_comparison_metrics(self):
        """Test comparison metrics."""
        results = experiment3_rag_vs_full_context(
            num_docs=5,
            use_real_llm=False
        )
        comparison = results["comparison"]
        assert "speedup" in comparison
        assert "token_reduction_pct" in comparison
        assert comparison["speedup"] >= 0
        assert 0 <= comparison["token_reduction_pct"] <= 100
    
    def test_rag_efficiency(self):
        """Test that RAG uses fewer tokens than full context."""
        results = experiment3_rag_vs_full_context(
            num_docs=10,
            use_real_llm=False
        )
        # RAG should use significantly fewer tokens
        assert results["rag"]["tokens"] < results["full_context"]["tokens"]


class TestExperiment4:
    """Tests for Experiment 4: Context Engineering Strategies."""
    
    def test_basic_execution(self):
        """Test basic experiment execution."""
        results = experiment4_context_strategies(
            num_steps=3,
            use_real_llm=False
        )
        assert isinstance(results, dict)
        assert "strategies" in results
    
    def test_strategy_names(self):
        """Test that all strategies are present."""
        results = experiment4_context_strategies(
            num_steps=3,
            use_real_llm=False
        )
        strategies = results["strategies"]
        assert "BASELINE" in strategies
        assert "SELECT" in strategies
        assert "COMPRESS" in strategies
        assert "WRITE" in strategies
    
    def test_strategy_metrics(self):
        """Test that each strategy has required metrics."""
        results = experiment4_context_strategies(
            num_steps=3,
            use_real_llm=False
        )
        for strategy_name, metrics in results["strategies"].items():
            assert "avg_latency" in metrics
            assert "avg_accuracy" in metrics
            assert "avg_tokens" in metrics
            assert metrics["avg_latency"] >= 0
            assert 0.0 <= metrics["avg_accuracy"] <= 1.0
            assert metrics["avg_tokens"] >= 0
    
    def test_action_count(self):
        """Test with different action counts."""
        for num_steps in [2, 5, 10]:
            results = experiment4_context_strategies(
                num_steps=num_steps,
                use_real_llm=False
            )
            assert "strategies" in results
            # Should complete without errors


class TestExperimentIntegration:
    """Integration tests across all experiments."""
    
    def test_all_experiments_run(self):
        """Test that all experiments can run in sequence."""
        exp1 = experiment1_needle_in_haystack(num_docs=3, use_real_llm=False)
        assert exp1 is not None
        
        exp2 = experiment2_context_size_impact(doc_counts=[2], use_real_llm=False)
        assert exp2 is not None
        
        exp3 = experiment3_rag_vs_full_context(num_docs=5, use_real_llm=False)
        assert exp3 is not None
        
        exp4 = experiment4_context_strategies(num_steps=2, use_real_llm=False)
        assert exp4 is not None
    
    def test_reproducibility(self):
        """Test that experiments produce consistent structure (not necessarily same values)."""
        # Run experiment twice
        results1 = experiment1_needle_in_haystack(num_docs=3, use_real_llm=False)
        results2 = experiment1_needle_in_haystack(num_docs=3, use_real_llm=False)
        
        # Should have same structure
        assert results1.keys() == results2.keys()
        assert results1["position_accuracy"].keys() == results2["position_accuracy"].keys()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

