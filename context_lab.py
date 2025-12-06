"""
Context Window Impact Analysis Lab
===================================
Main entry point and backward-compatible interface.

This file provides:
1. CLI interface for running experiments
2. Backward-compatible imports from modular src/ package
3. Logging configuration

Architecture:
- src/llm.py: LLM interface (Ollama, embeddings)
- src/utils.py: Text generation and document utilities
- src/vector_store.py: ChromaDB wrapper
- src/strategies.py: Context engineering strategies
- src/experiments.py: Four main experiments

Usage:
    python context_lab.py                    # Run all experiments
    python context_lab.py --experiment 3     # Run specific experiment
    python context_lab.py --help             # Show options

Author: Boris Volovelsky
Date: December 2025
License: MIT
"""

import argparse
import logging
import sys
from typing import Any, Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('context_lab.log', mode='a', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)

# ============================================================================
# BACKWARD-COMPATIBLE IMPORTS FROM MODULAR STRUCTURE
# ============================================================================

# Import from modular src/ package
from src import (
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

# Additional utility from utils for backward compatibility
from src.utils import nomic_embed_text


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main() -> Dict[str, Any]:
    """
    Main entry point for CLI execution.

    Returns:
        Experiment results dictionary
    """
    parser = argparse.ArgumentParser(
        description="Context Window Impact Analysis Lab",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python context_lab.py                    Run all experiments
  python context_lab.py --experiment 1     Run Needle in Haystack
  python context_lab.py --experiment 2     Run Context Size Impact
  python context_lab.py --experiment 3     Run RAG vs Full Context
  python context_lab.py --experiment 4     Run Context Strategies
  python context_lab.py -o results.json    Save results to custom file
        """
    )

    parser.add_argument(
        '--experiment', '-e',
        type=int,
        choices=[1, 2, 3, 4],
        help='Run specific experiment (1-4), or all if not specified'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default='context_lab_results.json',
        help='Output file for results (default: context_lab_results.json)'
    )

    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save results to file'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose (DEBUG) logging'
    )

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress output (WARNING level only)'
    )

    args = parser.parse_args()

    # Configure log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")
    elif args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    logger.info("Context Window Impact Analysis Lab starting...")
    logger.info(f"Real LLM available: {REAL_LLM_AVAILABLE}")

    results = {}

    if args.experiment:
        # Run specific experiment
        logger.info(f"Running experiment {args.experiment}")

        if args.experiment == 1:
            results = experiment1_needle_in_haystack()
        elif args.experiment == 2:
            results = experiment2_context_size_impact()
        elif args.experiment == 3:
            results = experiment3_rag_vs_full_context()
        elif args.experiment == 4:
            results = experiment4_context_strategies()
    else:
        # Run all experiments
        results = run_all_experiments(
            save_results=not args.no_save,
            output_file=args.output
        )

    logger.info("Context Lab execution completed")
    return results


# ============================================================================
# EXPORTS FOR PROGRAMMATIC USE
# ============================================================================

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
    "nomic_embed_text",
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


if __name__ == "__main__":
    main()
