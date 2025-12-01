"""
Context Lab - LLM Context Window Analysis Package
==================================================

A comprehensive toolkit for analyzing and demonstrating the impact of context windows
in Large Language Model (LLM) interactions, including:
- Needle in Haystack (Lost in the Middle)
- Context Window Size Impact
- RAG vs Full Context Comparison
- Context Engineering Strategies

Author: Boris Volovelsky
Repository: https://github.com/volo10/context_lab
"""

__version__ = "1.0.0"
__author__ = "Boris Volovelsky"
__license__ = "MIT"

# Import main experiment functions
from .context_lab import (
    # Core functions
    generate_filler_text,
    embed_critical_fact,
    split_documents,
    count_tokens,
    evaluate_accuracy,
    ollama_query,
    nomic_embed_text,
    
    # Experiment functions
    experiment1_needle_in_haystack,
    experiment2_context_window_size,
    experiment3_rag_vs_full_context,
    experiment4_context_strategies,
    
    # Classes
    SimpleVectorStore,
    ContextAgent,
    
    # Utilities
    get_llm,
    get_embeddings,
)

# Import visualization functions
try:
    from .visualize import (
        plot_experiment1_results,
        plot_experiment2_results,
        plot_experiment3_results,
        plot_experiment4_results,
        visualize_all_experiments,
    )
except ImportError:
    # Visualization dependencies may not be installed
    pass

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__license__",
    
    # Core functions
    "generate_filler_text",
    "embed_critical_fact",
    "split_documents",
    "count_tokens",
    "evaluate_accuracy",
    "ollama_query",
    "nomic_embed_text",
    
    # Experiment functions
    "experiment1_needle_in_haystack",
    "experiment2_context_window_size",
    "experiment3_rag_vs_full_context",
    "experiment4_context_strategies",
    
    # Classes
    "SimpleVectorStore",
    "ContextAgent",
    
    # Utilities
    "get_llm",
    "get_embeddings",
    
    # Visualization (if available)
    "plot_experiment1_results",
    "plot_experiment2_results",
    "plot_experiment3_results",
    "plot_experiment4_results",
    "visualize_all_experiments",
]

