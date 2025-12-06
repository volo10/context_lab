"""
Context Engineering Strategies Module
=====================================
Implements four context management strategies for multi-step LLM conversations.

Design Pattern: Strategy Pattern (GoF)
- Base class: ContextStrategy
- Concrete strategies: SELECT, COMPRESS, WRITE, ISOLATE

Each strategy represents a different approach to managing context growth:

1. SELECT (RAG): Retrieve relevant history using semantic search
2. COMPRESS: Summarize history when token limit exceeded
3. WRITE: Maintain external scratchpad with key facts
4. ISOLATE: Compartmentalize context by action type

Mathematical Analysis:
- Token growth rate per strategy
- Accuracy vs. context size tradeoffs
- Latency implications
"""

import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

from .llm import count_tokens
from .vector_store import SimpleVectorStore

logger = logging.getLogger(__name__)


@dataclass
class AgentAction:
    """
    Represents a single agent action in a conversation.

    Attributes:
        step: Sequential step number (1-indexed)
        action_type: Category of action (query, search, analyze, etc.)
        content: Input/prompt for this action
        response: LLM response for this action
    """
    step: int
    action_type: str
    content: str
    response: str


class ContextStrategy(ABC):
    """
    Abstract base class for context management strategies.

    All strategies must implement:
    - add_interaction(): Store new conversation turns
    - get_context(): Retrieve context for next LLM call

    Token Complexity Analysis:
    - Let n = number of interactions
    - Let w = average words per interaction

    Strategy Growth Rates:
    - No strategy: O(n * w) - linear growth
    - SELECT: O(k * w) - constant after k retrievals
    - COMPRESS: O(L) - bounded by token limit L
    - WRITE: O(n/3 * w) - linear but reduced
    - ISOLATE: O(c * k * w) - bounded by compartments
    """

    def __init__(self, name: str):
        """
        Initialize strategy.

        Args:
            name: Human-readable strategy name
        """
        self.name = name
        self.history: List[AgentAction] = []
        logger.debug(f"Initialized strategy: {name}")

    def add_interaction(self, action: AgentAction) -> None:
        """
        Add an interaction to history.

        Args:
            action: The agent action to store
        """
        self.history.append(action)
        logger.debug(f"{self.name}: Added step {action.step}")

    @abstractmethod
    def get_context(self, current_query: str) -> str:
        """
        Get the context to send to LLM.

        Args:
            current_query: The current query for context relevance

        Returns:
            Context string for LLM prompt
        """
        raise NotImplementedError

    def get_token_count(self) -> int:
        """
        Get current token count of context.

        Returns:
            Estimated token count
        """
        context = self.get_context("")
        return count_tokens(context)

    def reset(self) -> None:
        """Clear all history."""
        self.history = []
        logger.debug(f"{self.name}: History reset")


class SelectStrategy(ContextStrategy):
    """
    SELECT Strategy: RAG-based history retrieval.

    Algorithm:
    1. Store each interaction as a vector in the vector store
    2. On get_context(), perform similarity search with current query
    3. Return top-k most relevant past interactions

    Token Complexity: O(k * w)
    - k = number of retrieved results (constant)
    - w = average interaction size

    Advantages:
    - Bounded context size regardless of history length
    - Semantically relevant context

    Disadvantages:
    - May miss chronologically important context
    - Embedding computation overhead
    """

    def __init__(self, k: int = 5):
        """
        Initialize SELECT strategy.

        Args:
            k: Number of interactions to retrieve
        """
        super().__init__("SELECT (RAG)")
        self.k = k
        self.vector_store = SimpleVectorStore()

    def add_interaction(self, action: AgentAction) -> None:
        """Store interaction in vector store."""
        super().add_interaction(action)
        text = f"Step {action.step}: {action.action_type} - {action.content}"
        self.vector_store.add([text])

    def get_context(self, current_query: str, k: Optional[int] = None) -> str:
        """
        Retrieve top-k relevant past interactions.

        Args:
            current_query: Query for similarity matching
            k: Override default k value

        Returns:
            Concatenated relevant interactions
        """
        if not self.history:
            return ""

        retrieve_k = k if k is not None else self.k
        relevant = self.vector_store.similarity_search(current_query, k=retrieve_k)

        return "\n".join(relevant)


class CompressStrategy(ContextStrategy):
    """
    COMPRESS Strategy: Summarization-based context management.

    Algorithm:
    1. Accumulate full history until token limit
    2. When limit exceeded, compress to summary + recent N
    3. Summary preserves key information while reducing tokens

    Token Complexity: O(L)
    - L = token limit (constant upper bound)

    Compression Formula:
    If tokens(history) > L:
        context = "[SUMMARY of steps 1..N-2]" + history[-2:]

    Advantages:
    - Hard token limit guarantee
    - Preserves recent context

    Disadvantages:
    - Information loss in compression
    - Requires LLM call for real summarization
    """

    def __init__(self, token_limit: int = 2000):
        """
        Initialize COMPRESS strategy.

        Args:
            token_limit: Maximum token count before compression
        """
        super().__init__("COMPRESS (Summarization)")
        self.token_limit = token_limit

    def get_context(self, current_query: str) -> str:
        """
        Return full or compressed history.

        Compression triggers when:
        - count_tokens(full_history) > token_limit

        Args:
            current_query: (unused, for interface compatibility)

        Returns:
            Context string (full or compressed)
        """
        full_history = "\n".join([
            f"Step {a.step}: {a.action_type} - {a.content} -> {a.response}"
            for a in self.history
        ])

        if count_tokens(full_history) <= self.token_limit:
            return full_history

        # Compress: summary of old + recent 2
        if len(self.history) <= 3:
            return full_history

        compressed = (
            f"[SUMMARY of steps 1-{len(self.history)-2}]\n" +
            "\n".join([
                f"Step {a.step}: {a.action_type} - {a.content}"
                for a in self.history[-2:]
            ])
        )

        logger.debug(f"Compressed from {count_tokens(full_history)} to {count_tokens(compressed)} tokens")
        return compressed


class WriteStrategy(ContextStrategy):
    """
    WRITE Strategy: External memory/scratchpad.

    Algorithm:
    1. Extract key facts from every Nth interaction (default: 3)
    2. Store facts in scratchpad dictionary
    3. Return only scratchpad contents (not full history)

    Token Complexity: O(n/3 * w)
    - Only 1/3 of interactions stored
    - w = average fact size (smaller than full response)

    Scratchpad Structure:
    {
        "fact_3": "action_type: response",
        "fact_6": "action_type: response",
        ...
    }

    Advantages:
    - Very compact context
    - Key facts preserved

    Disadvantages:
    - Loses 2/3 of interactions
    - Simple extraction heuristic
    """

    def __init__(self, extract_every: int = 3):
        """
        Initialize WRITE strategy.

        Args:
            extract_every: Extract fact every N interactions
        """
        super().__init__("WRITE (Memory)")
        self.extract_every = extract_every
        self.scratchpad: Dict[str, str] = {}

    def add_interaction(self, action: AgentAction) -> None:
        """Extract and store key facts."""
        super().add_interaction(action)

        # Store every Nth interaction
        if action.step % self.extract_every == 0:
            self.scratchpad[f"fact_{action.step}"] = f"{action.action_type}: {action.response}"
            logger.debug(f"Extracted fact from step {action.step}")

    def get_context(self, current_query: str) -> str:
        """
        Return scratchpad contents.

        Args:
            current_query: (unused)

        Returns:
            Formatted key facts
        """
        if not self.scratchpad:
            return ""

        return "KEY FACTS:\n" + "\n".join([
            f"- {key}: {value}"
            for key, value in self.scratchpad.items()
        ])


class IsolateStrategy(ContextStrategy):
    """
    ISOLATE Strategy: Context compartmentalization.

    Algorithm:
    1. Separate interactions into compartments by action_type
    2. On get_context(), identify relevant compartments
    3. Return only relevant compartment contents

    Token Complexity: O(c * k * w)
    - c = number of relevant compartments (typically 1-2)
    - k = max items per compartment
    - w = average interaction size

    Compartment Structure:
    {
        "query": [{step, content, response}, ...],
        "search": [{step, content, response}, ...],
        "analyze": [{step, content, response}, ...],
    }

    Relevance Detection:
    1. Keyword matching: action_type in query
    2. Fallback: most recently active compartments

    Advantages:
    - Prevents context pollution between task types
    - Focused, relevant context

    Disadvantages:
    - May miss cross-compartment dependencies
    - Simple relevance heuristic
    """

    def __init__(self, max_per_compartment: int = 3):
        """
        Initialize ISOLATE strategy.

        Args:
            max_per_compartment: Max items to return per compartment
        """
        super().__init__("ISOLATE (Compartments)")
        self.max_per_compartment = max_per_compartment
        self.compartments: Dict[str, List[dict]] = defaultdict(list)

    def add_interaction(self, action: AgentAction) -> None:
        """Store interaction in appropriate compartment."""
        super().add_interaction(action)

        self.compartments[action.action_type].append({
            'step': action.step,
            'content': action.content,
            'response': action.response
        })

        logger.debug(f"Added to compartment '{action.action_type}'")

    def get_context(self, current_query: str, max_per_compartment: Optional[int] = None) -> str:
        """
        Return only relevant compartments.

        Relevance Algorithm:
        1. Check if any compartment type is mentioned in query
        2. If no match, return 2 most recently active compartments

        Args:
            current_query: Query for relevance matching
            max_per_compartment: Override default max items

        Returns:
            Formatted compartment contents
        """
        if not self.compartments:
            return ""

        max_items = max_per_compartment or self.max_per_compartment
        query_lower = current_query.lower()

        # Find relevant compartments by keyword match
        relevant_compartments = []
        for comp_type in self.compartments.keys():
            if comp_type.lower() in query_lower or query_lower in comp_type.lower():
                relevant_compartments.append(comp_type)

        # Fallback: most recently active compartments
        if not relevant_compartments:
            compartment_recency = {
                comp: max(item['step'] for item in items)
                for comp, items in self.compartments.items()
            }

            relevant_compartments = sorted(
                compartment_recency.keys(),
                key=lambda x: compartment_recency[x],
                reverse=True
            )[:2]

        # Build isolated context
        context_parts = []
        for comp_type in relevant_compartments:
            items = self.compartments[comp_type][-max_items:]
            comp_context = f"[{comp_type.upper()} CONTEXT]\n"

            for item in items:
                truncated = item['content'][:50]
                comp_context += f"  Step {item['step']}: {truncated}...\n"

            context_parts.append(comp_context)

        return "\n".join(context_parts)
