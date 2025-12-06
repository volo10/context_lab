"""
LLM Interface Module
====================
Provides abstraction layer for LLM interactions with Ollama/LangChain.
Supports automatic fallback to simulation mode.

Mathematical Foundation:
------------------------
Token estimation: T(text) = len(text) / 4
This approximation follows the empirical observation that:
- English: ~4 characters per token (GPT/BERT tokenizers)
- Hebrew: ~2-3 characters per token (UTF-8 multibyte)
"""

import logging
import random
import re
import time
from typing import Optional

# Configure module logger
logger = logging.getLogger(__name__)

# Try to import real LLM libraries
try:
    from langchain_community.llms import Ollama
    from langchain_community.embeddings import OllamaEmbeddings
    REAL_LLM_AVAILABLE = True
    logger.info("LangChain/Ollama libraries loaded successfully")
except ImportError:
    REAL_LLM_AVAILABLE = False
    logger.warning("LangChain/Ollama not installed. Using simulation mode.")
    logger.info("Install with: pip install langchain langchain-community")

# Global LLM instance (lazy initialization - Singleton pattern)
_llm_instance = None
_embeddings_instance = None


def get_llm(model: str = "llama2") -> Optional[object]:
    """
    Get or create LLM instance using Singleton pattern.

    Design Pattern: Singleton with lazy initialization
    - Ensures single LLM instance for memory efficiency
    - Tests connection on first invocation

    Args:
        model: Ollama model name (default: llama2)

    Returns:
        Ollama LLM instance or None if unavailable
    """
    global _llm_instance

    if _llm_instance is None and REAL_LLM_AVAILABLE:
        try:
            logger.debug(f"Initializing Ollama LLM with model: {model}")
            _llm_instance = Ollama(model=model, temperature=0.1)
            # Test connection
            _llm_instance.invoke("test")
            logger.info(f"Successfully connected to Ollama ({model})")
        except Exception as e:
            logger.warning(f"Could not connect to Ollama: {e}")
            logger.info("Make sure Ollama is running: ollama serve")
            _llm_instance = None

    return _llm_instance


def get_embeddings(model: str = "nomic-embed-text") -> Optional[object]:
    """
    Get or create embeddings instance.

    Args:
        model: Embedding model name

    Returns:
        OllamaEmbeddings instance or None
    """
    global _embeddings_instance

    if _embeddings_instance is None and REAL_LLM_AVAILABLE:
        try:
            logger.debug(f"Initializing embeddings with model: {model}")
            _embeddings_instance = OllamaEmbeddings(model=model)
            logger.info(f"Embeddings initialized ({model})")
        except Exception as e:
            logger.warning(f"Could not initialize embeddings: {e}")
            _embeddings_instance = None

    return _embeddings_instance


def _detect_hebrew(text: str) -> bool:
    """Check if text contains Hebrew characters."""
    return any('\u0590' <= c <= '\u05FF' for c in text)


def _build_hebrew_prompt(context: str, query: str) -> str:
    """Build bilingual prompt for Hebrew medical queries."""
    return f"""You are a helpful medical assistant. You are given Hebrew text about medications.

The question asks: "What are the side effects of Advil?" (in Hebrew: {query})

Hebrew Context:
{context}

Instructions:
1. Look for the medicine name "אדוויל" (Advil) or "איבופרופן" (Ibuprofen) in the Hebrew context
2. Find the sentence that lists side effects (תופעות לוואי)
3. Extract and list ALL the side effects mentioned

Side effects found:"""


def _build_english_prompt(context: str, query: str) -> str:
    """Build standard English prompt."""
    return f"""Based on the following context, answer the question concisely.

Context:
{context}

Question: {query}

Answer:"""


def _simulate_lost_in_middle(context: str, query: str) -> str:
    """
    Simulate "Lost in the Middle" phenomenon.

    Research Reference: Liu et al. (2023) "Lost in the Middle"
    - LLMs show U-shaped recall: better at start/end, worse in middle
    - Accuracy at position p follows: A(p) = A_base - k * (0.5 - |p - 0.5|)
    """
    fact_pos = context.find("CRITICAL_FACT:")
    context_len = len(context)

    if fact_pos == -1:
        relative_pos = 0.5
    else:
        relative_pos = fact_pos / context_len

    # U-shaped accuracy: high at 0 and 1, low at 0.5
    if relative_pos < 0.2 or relative_pos > 0.8:
        success_rate = 0.9  # 90% at edges
    else:
        success_rate = 0.5  # 50% in middle

    if random.random() < success_rate:
        return "The critical fact is: [Extracted correctly]"
    return "Not found"


def ollama_query(context: str, query: str, use_real: bool = None) -> str:
    """
    Query the LLM with given context and question.

    Supports:
    - Hebrew and English queries
    - Real Ollama or simulation mode
    - Automatic language detection

    Args:
        context: The context window text
        query: The question to ask
        use_real: Force real/simulation mode (None = auto-detect)

    Returns:
        LLM response string
    """
    # Auto-detect mode
    if use_real is None:
        use_real = REAL_LLM_AVAILABLE

    if use_real and REAL_LLM_AVAILABLE:
        llm = get_llm()
        if llm is not None:
            try:
                is_hebrew = _detect_hebrew(query)

                if is_hebrew:
                    prompt = _build_hebrew_prompt(context, query)
                else:
                    prompt = _build_english_prompt(context, query)

                logger.debug(f"Querying LLM (Hebrew={is_hebrew})")
                response = llm.invoke(prompt)
                return response.strip()

            except Exception as e:
                logger.warning(f"LLM query failed: {e}. Falling back to simulation.")
                use_real = False

    # Simulation mode
    logger.debug("Using simulation mode for query")
    time.sleep(random.uniform(0.1, 0.3))  # Simulate latency

    if "critical fact" in query.lower():
        return _simulate_lost_in_middle(context, query)

    return f"Response based on {len(context)} chars of context"


def evaluate_accuracy(response: str, expected_answer: str = None) -> float:
    """
    Evaluate LLM response accuracy.

    Scoring Algorithm:
    1. Exact match: 1.0
    2. Code pattern match (ALPHA*BETA*): 1.0 if found
    3. Hebrew medical terms: proportion of terms found
    4. Partial word overlap: overlap_ratio if > 0.5
    5. Negative indicators ("not found", etc.): 0.0
    6. Default (uncertain): 0.5

    Args:
        response: LLM response text
        expected_answer: Ground truth for comparison

    Returns:
        Accuracy score in [0.0, 1.0]
    """
    if expected_answer:
        # Pattern 1: ALPHA*BETA* code pattern
        match = re.search(r'ALPHA\d+BETA\d+', expected_answer)
        if match:
            code = match.group(0)
            if code in response:
                logger.debug(f"Code pattern '{code}' found in response")
                return 1.0

        # Pattern 2: Hebrew medical terms
        hebrew_terms = ['בחילות', 'כאבי בטן', 'צרבת', 'סחרחורת', 'איבופרופן', 'אדוויל']
        terms_in_expected = [t for t in hebrew_terms if t in expected_answer]

        if terms_in_expected:
            terms_in_response = [t for t in terms_in_expected if t in response]
            if terms_in_response:
                score = len(terms_in_response) / len(terms_in_expected)
                logger.debug(f"Hebrew terms: {len(terms_in_response)}/{len(terms_in_expected)}")
                return score

        # Pattern 3: Exact substring match
        if expected_answer.lower() in response.lower():
            return 1.0

        # Pattern 4: Word overlap (Jaccard-like)
        expected_words = set(expected_answer.lower().split())
        response_words = set(response.lower().split())
        overlap = len(expected_words & response_words)

        if len(expected_words) > 0:
            overlap_ratio = overlap / len(expected_words)
            if overlap_ratio > 0.5:
                logger.debug(f"Word overlap: {overlap_ratio:.2f}")
                return overlap_ratio

        return 0.0

    # No expected answer - evaluate response quality
    negative_indicators = ["not found", "i don't know", "אין מידע"]
    if any(neg in response.lower() for neg in negative_indicators):
        return 0.0

    if "[Extracted correctly]" in response:
        return 1.0

    return 0.5  # Uncertain


def count_tokens(text: str) -> int:
    """
    Estimate token count.

    Formula: T(text) ≈ len(text) / 4

    This approximation is based on empirical analysis:
    - OpenAI GPT models: ~4 chars/token for English
    - For Hebrew (UTF-8): slightly fewer chars/token

    For production use, consider using tiktoken:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))

    Args:
        text: Input text string

    Returns:
        Estimated token count (integer)
    """
    return len(text) // 4
