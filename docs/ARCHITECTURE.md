# Architecture Document
## Context Window Impact Analysis Lab

**Version:** 1.0.0
**Last Updated:** December 2025

---

## 1. Overview

This document describes the software architecture of the Context Lab project using the C4 model (Context, Container, Component, Code).

---

## 2. C4 Model Diagrams

### 2.1 Level 1: System Context Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        SYSTEM CONTEXT                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│    ┌──────────┐         ┌──────────────────┐                   │
│    │  User    │────────▶│   Context Lab    │                   │
│    │(Student) │         │     System       │                   │
│    └──────────┘         └────────┬─────────┘                   │
│                                  │                              │
│                    ┌─────────────┼─────────────┐               │
│                    ▼             ▼             ▼               │
│             ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│             │  Ollama  │  │ ChromaDB │  │Sentence  │          │
│             │   LLM    │  │  Vector  │  │Transform │          │
│             │  Server  │  │   Store  │  │  Models  │          │
│             └──────────┘  └──────────┘  └──────────┘          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Actors:**
- **User (Student)**: Runs experiments, analyzes results
- **Ollama LLM Server**: Provides local LLM inference (optional)
- **ChromaDB**: Vector database for RAG experiments
- **Sentence Transformers**: Embedding models for text vectorization

### 2.2 Level 2: Container Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      CONTAINER DIAGRAM                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Python Application                    │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │                                                         │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │   │
│  │  │  CLI Entry  │  │ Experiment  │  │Visualization│    │   │
│  │  │   Point     │──│   Engine    │──│   Module    │    │   │
│  │  │(context_lab)│  │             │  │ (visualize) │    │   │
│  │  └─────────────┘  └──────┬──────┘  └─────────────┘    │   │
│  │                          │                             │   │
│  │         ┌────────────────┼────────────────┐           │   │
│  │         ▼                ▼                ▼           │   │
│  │  ┌───────────┐   ┌───────────┐   ┌───────────┐       │   │
│  │  │   LLM     │   │  Vector   │   │ Strategy  │       │   │
│  │  │ Interface │   │   Store   │   │  Classes  │       │   │
│  │  └───────────┘   └───────────┘   └───────────┘       │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │   config/   │  │   tests/    │  │   plots/    │            │
│  │   (YAML)    │  │  (pytest)   │  │   (PNG)     │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 Level 3: Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     COMPONENT DIAGRAM                           │
│                     (context_lab.py)                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    LLM Interface Layer                   │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │   │
│  │  │  get_llm()  │  │ollama_query │  │evaluate_    │     │   │
│  │  │             │  │    ()       │  │accuracy()   │     │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   Experiment Functions                   │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │   │
│  │  │experiment1_ │  │experiment2_ │  │experiment3_ │     │   │
│  │  │needle_in_   │  │context_size │  │rag_vs_full  │     │   │
│  │  │haystack()   │  │_impact()    │  │_context()   │     │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘     │   │
│  │                                                         │   │
│  │  ┌─────────────────────────────────────────────────┐   │   │
│  │  │        experiment4_context_strategies()          │   │   │
│  │  └─────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Strategy Classes                      │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │   │
│  │  │ Select   │ │ Compress │ │  Write   │ │ Isolate  │   │   │
│  │  │ Strategy │ │ Strategy │ │ Strategy │ │ Strategy │   │   │
│  │  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘   │   │
│  │       └────────────┴────────────┴────────────┘         │   │
│  │                         │                               │   │
│  │              ┌──────────┴──────────┐                   │   │
│  │              │   ContextStrategy   │                   │   │
│  │              │   (Base Class)      │                   │   │
│  │              └─────────────────────┘                   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   Utility Components                     │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │   │
│  │  │generate_    │  │split_       │  │SimpleVector │     │   │
│  │  │filler_text()│  │documents()  │  │Store        │     │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| Language | Python 3.8+ | Core implementation |
| LLM Framework | LangChain | Ollama integration |
| Vector Store | ChromaDB | Embedding storage & retrieval |
| Embeddings | Sentence Transformers | Text vectorization |
| Data Processing | NumPy, Pandas | Numerical operations |
| Visualization | Matplotlib, Seaborn | Plot generation |
| Testing | pytest | Unit testing |

---

## 4. Architectural Decision Records (ADRs)

### ADR-001: Use Simulation Mode as Default

**Context:** Real LLM inference requires Ollama setup and is slower.

**Decision:** Default to simulation mode; auto-detect real LLM availability.

**Consequences:**
- (+) Quick experimentation without setup
- (+) Consistent results for testing
- (-) Simulated results may not reflect real LLM behavior

### ADR-002: ChromaDB for Vector Storage

**Context:** Need vector database for RAG experiment.

**Decision:** Use ChromaDB with in-memory storage by default.

**Alternatives Considered:**
- FAISS: More complex setup
- Pinecone: Requires API key
- Weaviate: Heavier infrastructure

**Consequences:**
- (+) Zero configuration in-memory mode
- (+) Optional persistence to disk
- (-) Limited to single-node deployment

### ADR-003: Strategy Pattern for Context Engineering

**Context:** Need to compare multiple context management approaches.

**Decision:** Use Strategy pattern with base class `ContextStrategy`.

**Consequences:**
- (+) Easy to add new strategies
- (+) Consistent interface for benchmarking
- (+) Clean separation of concerns

### ADR-004: Hebrew Support via Multilingual Embeddings

**Context:** Experiment 3 requires Hebrew document processing.

**Decision:** Use `paraphrase-multilingual-MiniLM-L12-v2` for Hebrew embeddings.

**Alternatives Considered:**
- Hebrew-specific models: Limited availability
- Translation to English: Adds latency and loses nuance

**Consequences:**
- (+) Supports Hebrew without translation
- (+) Same pipeline for all languages
- (-) Slightly lower quality than language-specific models

---

## 5. Data Flow

### 5.1 Experiment 3 (RAG) Data Flow

```
┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│ Hebrew   │────▶│ Chunking │────▶│Embedding │────▶│ ChromaDB │
│Documents │     │ (400ch)  │     │ (384d)   │     │  Store   │
└──────────┘     └──────────┘     └──────────┘     └────┬─────┘
                                                        │
                                                        ▼
┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│ Response │◀────│   LLM    │◀────│ Top-K    │◀────│ Query    │
│          │     │  Query   │     │ Retrieval│     │ Embed    │
└──────────┘     └──────────┘     └──────────┘     └──────────┘
```

---

## 6. API Contracts

### 6.1 Experiment Function Signature

```python
def experiment1_needle_in_haystack(
    num_docs: int = 5,
    words_per_doc: int = 200,
    use_real_llm: bool = None
) -> Dict[str, Any]:
    """
    Returns:
        {
            'detailed_results': {...},
            'summary': {
                'start': {'avg_accuracy': float, 'std_accuracy': float, 'avg_latency': float},
                'middle': {...},
                'end': {...}
            },
            'expected_outcome': str
        }
    """
```

### 6.2 Strategy Interface

```python
class ContextStrategy:
    def __init__(self, name: str): ...
    def add_interaction(self, action: AgentAction) -> None: ...
    def get_context(self, current_query: str) -> str: ...
    def get_token_count(self) -> int: ...
```

---

## 7. Extension Points

### 7.1 Adding New Strategies

```python
from context_lab import ContextStrategy, AgentAction

class MyCustomStrategy(ContextStrategy):
    def __init__(self):
        super().__init__("MY_CUSTOM")

    def add_interaction(self, action: AgentAction):
        super().add_interaction(action)
        # Custom logic here

    def get_context(self, current_query: str) -> str:
        # Return context for LLM
        return "..."
```

### 7.2 Adding New LLM Providers

Modify `get_llm()` function in `context_lab.py`:

```python
def get_llm(provider: str = "ollama", model: str = "llama2"):
    if provider == "ollama":
        return Ollama(model=model)
    elif provider == "openai":
        return OpenAI(model=model)
    # Add new providers here
```

---

## 8. Security Considerations

- API keys stored in environment variables (never in code)
- `.env` files excluded from version control
- No user input executed as code
- ChromaDB runs locally (no network exposure by default)

---

## 9. Performance Characteristics

| Mode | Experiment 1-4 Total | Memory Usage |
|------|---------------------|--------------|
| Simulation | ~30-60 seconds | < 100 MB |
| Real LLM (local) | ~5-15 minutes | ~4-8 GB |

---

## 10. Deployment Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Local Development                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐      ┌─────────────┐                      │
│  │   Python    │      │   Ollama    │                      │
│  │  context_lab│◀────▶│   Server    │                      │
│  │             │      │ (optional)  │                      │
│  └──────┬──────┘      └─────────────┘                      │
│         │                                                   │
│         ▼                                                   │
│  ┌─────────────┐                                           │
│  │   plots/    │                                           │
│  │  (output)   │                                           │
│  └─────────────┘                                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```
