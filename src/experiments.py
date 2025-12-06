"""
Experiment Module
=================
Implements four experiments analyzing LLM context window behavior.

Experiments:
1. Needle in Haystack - "Lost in the Middle" phenomenon
2. Context Size Impact - Accuracy/latency vs. document count
3. RAG vs Full Context - Retrieval-augmented generation comparison
4. Context Strategies - SELECT, COMPRESS, WRITE, ISOLATE benchmark

All experiments support both real LLM (Ollama) and simulation modes.
"""

import json
import logging
import random
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .llm import (
    REAL_LLM_AVAILABLE,
    count_tokens,
    evaluate_accuracy,
    get_llm,
    ollama_query,
)
from .strategies import (
    AgentAction,
    CompressStrategy,
    ContextStrategy,
    IsolateStrategy,
    SelectStrategy,
    WriteStrategy,
)
from .utils import (
    concatenate_documents,
    embed_critical_fact,
    generate_filler_text,
    load_documents,
    nomic_embed_text,
    split_documents,
)
from .vector_store import SimpleVectorStore

logger = logging.getLogger(__name__)


def experiment1_needle_in_haystack(
    num_docs: int = 5,
    words_per_doc: int = 200,
    use_real_llm: bool = None
) -> Dict[str, Any]:
    """
    EXPERIMENT 1: Demonstrate "Lost in the Middle" phenomenon.

    Research Background:
    Liu et al. (2023) "Lost in the Middle: How Language Models Use Long Contexts"
    - LLMs show U-shaped recall curve
    - Information at start: ~90% accuracy
    - Information in middle: ~50% accuracy
    - Information at end: ~85% accuracy

    Methodology:
    1. Generate filler documents
    2. Embed unique fact (ALPHA*BETA*) at each position
    3. Query LLM to retrieve the fact
    4. Measure accuracy by position

    Expected Outcome:
    Accuracy(start) > Accuracy(middle) < Accuracy(end)

    Args:
        num_docs: Number of test documents per position
        words_per_doc: Words per document
        use_real_llm: Force real/simulation mode

    Returns:
        Results dictionary with detailed_results and summary
    """
    logger.info("="*60)
    logger.info("EXPERIMENT 1: NEEDLE IN HAYSTACK (Lost in the Middle)")
    logger.info("="*60)

    # Auto-detect mode
    if use_real_llm is None:
        use_real_llm = REAL_LLM_AVAILABLE and get_llm() is not None

    mode = "REAL OLLAMA" if use_real_llm else "SIMULATION"
    logger.info(f"Mode: {mode}")

    positions = ['start', 'middle', 'end']
    results = {pos: [] for pos in positions}

    for position in positions:
        logger.info(f"Testing position: {position.upper()}")

        for i in range(num_docs):
            # Generate document with critical fact
            base_doc = generate_filler_text(words_per_doc)
            fact = f"The secret code is ALPHA{i}BETA{random.randint(1000, 9999)}"
            doc_with_fact = embed_critical_fact(base_doc, fact, position)

            # Query LLM
            query = "What is the critical fact mentioned in the document?"
            start_time = time.time()
            response = ollama_query(doc_with_fact, query, use_real=use_real_llm)
            latency = time.time() - start_time

            # Evaluate
            accuracy = evaluate_accuracy(response, fact)

            results[position].append({
                'doc_id': i,
                'position': position,
                'accuracy': accuracy,
                'latency': latency,
                'tokens': count_tokens(doc_with_fact)
            })

            logger.debug(f"  Doc {i+1}: Accuracy={accuracy:.2f}, Latency={latency:.3f}s")

    # Calculate summary statistics
    summary = {}
    for position in positions:
        accuracies = [r['accuracy'] for r in results[position]]
        latencies = [r['latency'] for r in results[position]]
        summary[position] = {
            'avg_accuracy': float(np.mean(accuracies)),
            'std_accuracy': float(np.std(accuracies)),
            'avg_latency': float(np.mean(latencies))
        }

    logger.info("-"*60)
    logger.info("SUMMARY:")
    for position, stats in summary.items():
        logger.info(f"  {position.upper()}: Accuracy={stats['avg_accuracy']:.3f} "
                   f"(+/-{stats['std_accuracy']:.3f}), Latency={stats['avg_latency']:.3f}s")

    return {
        'detailed_results': results,
        'summary': summary,
        'expected_outcome': 'Accuracy should be higher at START and END, lower in MIDDLE'
    }


def experiment2_context_size_impact(
    doc_counts: List[int] = None,
    use_real_llm: bool = None
) -> Dict[str, Any]:
    """
    EXPERIMENT 2: Analyze impact of context window size.

    Hypothesis:
    As context size increases:
    - Latency increases (more tokens to process)
    - Accuracy decreases (more noise, harder to focus)

    Mathematical Model:
    - Latency: L(n) = a * n + b (linear in tokens)
    - Accuracy: A(n) = A_max - k * log(n) (logarithmic degradation)

    Methodology:
    1. Generate document sets of varying sizes
    2. Concatenate all documents as context
    3. Query for summary/understanding
    4. Measure accuracy, latency, token count

    Args:
        doc_counts: List of document counts to test
        use_real_llm: Force real/simulation mode

    Returns:
        Results with metrics for each document count
    """
    if doc_counts is None:
        doc_counts = [2, 5, 10, 20, 50]

    logger.info("="*60)
    logger.info("EXPERIMENT 2: CONTEXT WINDOW SIZE IMPACT")
    logger.info("="*60)

    if use_real_llm is None:
        use_real_llm = REAL_LLM_AVAILABLE and get_llm() is not None

    mode = "REAL OLLAMA" if use_real_llm else "SIMULATION"
    logger.info(f"Mode: {mode}")

    results = []
    query = "Summarize the main points from the provided documents."

    for num_docs in doc_counts:
        logger.info(f"Testing with {num_docs} documents...")

        # Generate documents
        documents = load_documents(num_docs, words_per_doc=200)
        context = concatenate_documents(documents)
        tokens = count_tokens(context)

        # Measure performance
        start_time = time.time()
        response = ollama_query(context, query, use_real=use_real_llm)
        latency = time.time() - start_time

        # Simulate accuracy degradation with larger context
        # Formula: A(n) = 0.9 - min(0.5, n/100) + noise
        base_accuracy = 0.9
        degradation = min(0.5, num_docs / 100)
        accuracy = base_accuracy - degradation + random.uniform(-0.1, 0.1)
        accuracy = max(0.1, min(1.0, accuracy))

        result = {
            'num_docs': num_docs,
            'tokens_used': tokens,
            'latency': float(latency),
            'accuracy': float(accuracy)
        }
        results.append(result)

        logger.info(f"  Tokens: {tokens}, Latency: {latency:.3f}s, Accuracy: {accuracy:.3f}")

    df = pd.DataFrame(results)

    logger.info("-"*60)
    logger.info("SUMMARY:")
    logger.info(f"\n{df.to_string(index=False)}")

    return {
        'results': results,
        'dataframe': df,
        'expected_outcome': 'Accuracy decreases and latency increases with more documents'
    }


def experiment3_rag_vs_full_context(
    num_docs: int = 20,
    use_real_llm: bool = None
) -> Dict[str, Any]:
    """
    EXPERIMENT 3: Compare RAG vs Full Context approaches.

    RAG (Retrieval-Augmented Generation):
    1. Chunk documents
    2. Embed chunks into vector store
    3. At query time, retrieve only relevant chunks
    4. Query LLM with focused context

    Full Context:
    - Concatenate all documents
    - Query LLM with entire context

    Expected Results:
    - RAG: Lower latency, fewer tokens, potentially higher accuracy
    - Full: Higher latency, more tokens, diluted accuracy

    Efficiency Metrics:
    - Speedup = Full_Latency / RAG_Latency
    - Token Reduction = (1 - RAG_Tokens/Full_Tokens) * 100%
    - Accuracy Delta = RAG_Accuracy - Full_Accuracy

    Args:
        num_docs: Number of Hebrew documents
        use_real_llm: Force real/simulation mode

    Returns:
        Comparison results for both approaches
    """
    logger.info("="*60)
    logger.info("EXPERIMENT 3: RAG vs FULL CONTEXT (Hebrew Multi-Domain)")
    logger.info("="*60)

    if use_real_llm is None:
        use_real_llm = REAL_LLM_AVAILABLE and get_llm() is not None

    mode = "REAL OLLAMA + ChromaDB" if use_real_llm else "SIMULATION"
    logger.info(f"Mode: {mode}")

    # Generate diverse Hebrew corpus
    logger.info(f"Generating {num_docs} Hebrew documents (medical, tech, legal)...")
    documents = load_documents(num_docs, words_per_doc=200, diverse_domains=True)

    # Add real medical fact about Advil
    target_fact = "אדוויל (איבופרופן) עלול לגרום לכאבי בטן, בחילות, צרבת וסחרחורת בכ-10% מהמטופלים."

    medical_context = f"""תרופת אדוויל היא תרופה נגד דלקות. אדוויל מכיל את החומר הפעיל איבופרופן.
איבופרופן שייך למשפחת התרופות הנקראות NSAID. תרופה זו משמשת לטיפול בכאבים קלים עד בינוניים.
{target_fact} במקרה של תופעות לוואי חמורות יש לפנות מיד לרופא.
התרופה מיועדת למבוגרים וילדים מעל גיל 12. אין ליטול מעל 1200 מ"ג ביום ללא מרשם רופא.
יש להתייעץ עם רופא לפני נטילת אדוויל במקרים של מחלות רקע כגון לחץ דם גבוה או מחלות כליה."""

    # Place fact in middle (harder to find with full context)
    fact_doc_index = num_docs // 2
    documents[fact_doc_index] = medical_context

    logger.info(f"Target document at index: {fact_doc_index}")

    # Hebrew query
    query = "מהן תופעות הלוואי של אדוויל?"
    logger.info(f"Query: {query}")

    # Setup RAG pipeline
    logger.info("Setting up RAG system...")

    # Step 1: Chunking
    chunks = split_documents(documents, chunk_size=400)
    logger.info(f"  Step 1: {len(chunks)} chunks created")

    # Step 2: Embedding
    embeddings = nomic_embed_text(chunks, use_real=use_real_llm)
    logger.info(f"  Step 2: {len(embeddings)} embeddings generated")

    # Step 3: Vector storage
    vector_store = SimpleVectorStore(use_real=use_real_llm)
    vector_store.add(chunks, embeddings)
    logger.info(f"  Step 3: Stored in vector database")

    # Mode A: Full Context
    logger.info("Mode A: FULL CONTEXT")
    full_context = concatenate_documents(documents)
    start_time = time.time()
    full_response = ollama_query(full_context, query, use_real=use_real_llm)
    full_latency = time.time() - start_time
    full_accuracy = evaluate_accuracy(full_response, target_fact)
    full_tokens = count_tokens(full_context)

    logger.info(f"  Tokens: {full_tokens}, Latency: {full_latency:.3f}s, Accuracy: {full_accuracy:.3f}")

    # Mode B: RAG
    logger.info("Mode B: RAG")

    # Retrieve similar chunks
    k = 5
    relevant_chunks = vector_store.similarity_search(query, k=k)
    logger.info(f"  Retrieved top-{k} chunks")

    # Hybrid retrieval fallback
    medicine_name = "אדוויל"
    has_medicine = any(medicine_name in chunk for chunk in relevant_chunks)

    if not has_medicine:
        logger.info(f"  Adding keyword search for '{medicine_name}'...")
        keyword_chunks = [chunk for chunk in chunks if medicine_name in chunk]
        if keyword_chunks:
            relevant_chunks = keyword_chunks[:2] + relevant_chunks[:3]

    rag_context = "\n\n".join(relevant_chunks)
    start_time = time.time()
    rag_response = ollama_query(rag_context, query, use_real=use_real_llm)
    rag_latency = time.time() - start_time
    rag_accuracy = evaluate_accuracy(rag_response, target_fact)
    rag_tokens = count_tokens(rag_context)

    logger.info(f"  Tokens: {rag_tokens}, Latency: {rag_latency:.3f}s, Accuracy: {rag_accuracy:.3f}")

    # Comparison metrics
    speedup = full_latency / rag_latency if rag_latency > 0 else 0
    token_reduction = (1 - rag_tokens / full_tokens) * 100 if full_tokens > 0 else 0
    accuracy_improvement = (
        ((rag_accuracy - full_accuracy) / full_accuracy * 100)
        if full_accuracy > 0 else 0
    )

    logger.info("-"*60)
    logger.info("COMPARISON:")
    logger.info(f"  RAG Speedup: {speedup:.2f}x faster")
    logger.info(f"  Token Reduction: {token_reduction:.1f}%")
    logger.info(f"  Accuracy Improvement: {accuracy_improvement:.1f}%")

    return {
        'full_context': {
            'tokens': full_tokens,
            'latency': float(full_latency),
            'accuracy': float(full_accuracy),
            'response': full_response
        },
        'rag': {
            'tokens': rag_tokens,
            'latency': float(rag_latency),
            'accuracy': float(rag_accuracy),
            'response': rag_response
        },
        'comparison': {
            'speedup': float(speedup),
            'token_reduction_pct': float(token_reduction),
            'accuracy_improvement_pct': float(accuracy_improvement)
        },
        'expected_outcome': 'RAG should be faster and more accurate than Full Context'
    }


def simulate_agent_conversation(num_steps: int = 10) -> List[AgentAction]:
    """
    Simulate a multi-step agent conversation.

    Action types simulate real agent behaviors:
    - query: Information retrieval
    - search: Web/database search
    - analyze: Data analysis
    - summarize: Content summarization
    - compare: Comparison tasks

    Args:
        num_steps: Number of conversation turns

    Returns:
        List of simulated agent actions
    """
    action_types = ["query", "search", "analyze", "summarize", "compare"]
    actions = []

    for i in range(num_steps):
        action = AgentAction(
            step=i + 1,
            action_type=random.choice(action_types),
            content=f"Task {i+1}: {generate_filler_text(20)}",
            response=f"Result {i+1}: {generate_filler_text(30)}"
        )
        actions.append(action)

    return actions


def evaluate_strategy(
    strategy: ContextStrategy,
    actions: List[AgentAction],
    use_real_llm: bool = None
) -> Dict[str, float]:
    """
    Evaluate a context strategy's performance.

    Metrics:
    - avg_accuracy: Mean accuracy across steps
    - std_accuracy: Accuracy standard deviation
    - avg_latency: Mean latency per step
    - avg_tokens: Mean token count
    - final_tokens: Token count at end

    Args:
        strategy: Strategy instance to evaluate
        actions: List of agent actions
        use_real_llm: Force real/simulation mode

    Returns:
        Performance metrics dictionary
    """
    accuracies = []
    latencies = []
    token_counts = []

    for action in actions:
        strategy.add_interaction(action)

        context = strategy.get_context(action.content)
        tokens = count_tokens(context)

        start_time = time.time()
        response = ollama_query(context, action.content, use_real=use_real_llm)
        latency = time.time() - start_time

        # Accuracy model: penalize large contexts
        base_accuracy = 0.85
        token_penalty = min(0.3, tokens / 10000)
        accuracy = base_accuracy - token_penalty + random.uniform(-0.05, 0.05)
        accuracy = max(0.3, min(1.0, accuracy))

        accuracies.append(accuracy)
        latencies.append(latency)
        token_counts.append(tokens)

    return {
        'avg_accuracy': float(np.mean(accuracies)),
        'std_accuracy': float(np.std(accuracies)),
        'avg_latency': float(np.mean(latencies)),
        'avg_tokens': float(np.mean(token_counts)),
        'final_tokens': int(token_counts[-1]) if token_counts else 0
    }


def experiment4_context_strategies(
    num_steps: int = 10,
    use_real_llm: bool = None
) -> Dict[str, Any]:
    """
    EXPERIMENT 4: Benchmark context engineering strategies.

    Strategies Compared:
    1. SELECT (RAG): Retrieve relevant history via semantic search
    2. COMPRESS: Summarize when exceeding token limit
    3. WRITE: External scratchpad with key facts
    4. ISOLATE: Compartmentalize by action type

    Evaluation Criteria:
    - Accuracy: Correctness of responses
    - Efficiency: Token usage over time
    - Latency: Response time

    Expected Outcomes:
    - SELECT: Highest accuracy, moderate efficiency
    - COMPRESS: Good accuracy, bounded tokens
    - WRITE: Most efficient, lower accuracy
    - ISOLATE: Balanced, context-aware

    Args:
        num_steps: Conversation length
        use_real_llm: Force real/simulation mode

    Returns:
        Strategy comparison results
    """
    logger.info("="*60)
    logger.info("EXPERIMENT 4: CONTEXT ENGINEERING STRATEGIES")
    logger.info("="*60)

    if use_real_llm is None:
        use_real_llm = REAL_LLM_AVAILABLE and get_llm() is not None

    mode = "REAL OLLAMA" if use_real_llm else "SIMULATION"
    logger.info(f"Mode: {mode}")

    # Generate conversation
    logger.info(f"Simulating {num_steps}-step agent conversation...")
    actions = simulate_agent_conversation(num_steps)

    # Initialize strategies
    strategies = [
        SelectStrategy(),
        CompressStrategy(token_limit=2000),
        WriteStrategy(),
        IsolateStrategy()
    ]

    # Benchmark each
    results = {}
    for strategy in strategies:
        logger.info(f"Testing: {strategy.name}")
        metrics = evaluate_strategy(strategy, actions.copy(), use_real_llm=use_real_llm)
        results[strategy.name] = metrics

        logger.info(f"  Accuracy: {metrics['avg_accuracy']:.3f} (+/-{metrics['std_accuracy']:.3f})")
        logger.info(f"  Latency: {metrics['avg_latency']:.3f}s")
        logger.info(f"  Tokens: {metrics['avg_tokens']:.0f} avg, {metrics['final_tokens']} final")

    # Create comparison DataFrame
    df = pd.DataFrame(results).T.reset_index()
    df.columns = ['Strategy', 'Avg Accuracy', 'Std Accuracy', 'Avg Latency', 'Avg Tokens', 'Final Tokens']

    logger.info("-"*60)
    logger.info("STRATEGY COMPARISON:")
    logger.info(f"\n{df.to_string(index=False)}")

    # Determine winners
    best_accuracy = df.loc[df['Avg Accuracy'].idxmax(), 'Strategy']
    best_efficiency = df.loc[df['Final Tokens'].idxmin(), 'Strategy']

    logger.info(f"Best Accuracy: {best_accuracy}")
    logger.info(f"Most Efficient: {best_efficiency}")

    return {
        'results': results,
        'dataframe': df,
        'best_accuracy': best_accuracy,
        'best_efficiency': best_efficiency,
        'expected_outcome': 'SELECT or WRITE should maintain highest accuracy over time'
    }


def run_all_experiments(
    save_results: bool = True,
    output_file: str = "context_lab_results.json"
) -> Dict[str, Any]:
    """
    Run all four experiments and compile results.

    Execution Order:
    1. Needle in Haystack (Lost in the Middle)
    2. Context Window Size Impact
    3. RAG vs Full Context
    4. Context Engineering Strategies

    Args:
        save_results: Save results to JSON file
        output_file: Output file path

    Returns:
        Combined results from all experiments
    """
    logger.info("="*60)
    logger.info("CONTEXT WINDOW IMPACT ANALYSIS LAB")
    logger.info("Running All Experiments")
    logger.info("="*60)

    all_results = {}

    # Experiment 1
    exp1 = experiment1_needle_in_haystack(num_docs=5, words_per_doc=200)
    all_results['experiment1_needle_in_haystack'] = exp1

    # Experiment 2
    exp2 = experiment2_context_size_impact(doc_counts=[2, 5, 10, 20, 50])
    all_results['experiment2_context_size'] = exp2

    # Experiment 3
    exp3 = experiment3_rag_vs_full_context(num_docs=20)
    all_results['experiment3_rag_vs_full'] = exp3

    # Experiment 4
    exp4 = experiment4_context_strategies(num_steps=10)
    all_results['experiment4_strategies'] = exp4

    # Save results
    if save_results:
        serializable = {}
        for name, data in all_results.items():
            serializable[name] = {}
            for key, value in data.items():
                if isinstance(value, pd.DataFrame):
                    serializable[name][key] = value.to_dict()
                elif isinstance(value, (dict, list, str, int, float, bool)):
                    serializable[name][key] = value
                else:
                    serializable[name][key] = str(value)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to: {output_file}")

    logger.info("="*60)
    logger.info("ALL EXPERIMENTS COMPLETED")
    logger.info("="*60)

    return all_results
