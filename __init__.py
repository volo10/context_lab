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

# Import from modular src/ package (primary)
from .src import (
    # LLM Interface
    get_llm,
    get_embeddings,
    ollama_query,
    evaluate_accuracy,
    count_tokens,
    REAL_LLM_AVAILABLE,
    # Utilities
    generate_filler_text,
    embed_critical_fact,
    split_documents,
    load_documents,
    concatenate_documents,
    # Vector Store
    SimpleVectorStore,
    # Strategies
    AgentAction,
    ContextStrategy,
    SelectStrategy,
    CompressStrategy,
    WriteStrategy,
    IsolateStrategy,
    # Experiments
    experiment1_needle_in_haystack,
    experiment2_context_size_impact,
    experiment3_rag_vs_full_context,
    experiment4_context_strategies,
    run_all_experiments,
)

# Import nomic_embed_text from utils
from .src.utils import nomic_embed_text

# Import visualization functions
try:
    from .visualize import (
        plot_experiment1,
        plot_experiment2,
        plot_experiment3,
        plot_experiment4,
        generate_all_plots,
    )
    # Create aliases for backward compatibility
    plot_experiment1_results = plot_experiment1
    plot_experiment2_results = plot_experiment2
    plot_experiment3_results = plot_experiment3
    plot_experiment4_results = plot_experiment4
    visualize_all_experiments = generate_all_plots
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
    "load_documents",
    "concatenate_documents",

    # Experiment functions
    "experiment1_needle_in_haystack",
    "experiment2_context_size_impact",
    "experiment3_rag_vs_full_context",
    "experiment4_context_strategies",
    "run_all_experiments",

    # Classes
    "SimpleVectorStore",
    "SelectStrategy",
    "CompressStrategy",
    "WriteStrategy",
    "IsolateStrategy",
    "ContextStrategy",
    "AgentAction",

    # Utilities
    "get_llm",
    "get_embeddings",
    "REAL_LLM_AVAILABLE",

    # Visualization (if available)
    "plot_experiment1_results",
    "plot_experiment2_results",
    "plot_experiment3_results",
    "plot_experiment4_results",
    "visualize_all_experiments",
]
