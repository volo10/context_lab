"""
Context Lab - Modular Package Structure
========================================
This modular architecture provides clean separation of concerns for
production-grade LLM context window analysis.
"""

from .llm import (
    get_llm,
    get_embeddings,
    ollama_query,
    evaluate_accuracy,
    count_tokens,
    REAL_LLM_AVAILABLE,
)

from .utils import (
    generate_filler_text,
    embed_critical_fact,
    split_documents,
    load_documents,
    concatenate_documents,
)

from .vector_store import SimpleVectorStore

from .strategies import (
    AgentAction,
    ContextStrategy,
    SelectStrategy,
    CompressStrategy,
    WriteStrategy,
    IsolateStrategy,
)

from .experiments import (
    experiment1_needle_in_haystack,
    experiment2_context_size_impact,
    experiment3_rag_vs_full_context,
    experiment4_context_strategies,
    run_all_experiments,
)

__all__ = [
    # LLM Interface
    "get_llm",
    "get_embeddings",
    "ollama_query",
    "evaluate_accuracy",
    "count_tokens",
    "REAL_LLM_AVAILABLE",
    # Utilities
    "generate_filler_text",
    "embed_critical_fact",
    "split_documents",
    "load_documents",
    "concatenate_documents",
    # Vector Store
    "SimpleVectorStore",
    # Strategies
    "AgentAction",
    "ContextStrategy",
    "SelectStrategy",
    "CompressStrategy",
    "WriteStrategy",
    "IsolateStrategy",
    # Experiments
    "experiment1_needle_in_haystack",
    "experiment2_context_size_impact",
    "experiment3_rag_vs_full_context",
    "experiment4_context_strategies",
    "run_all_experiments",
]
