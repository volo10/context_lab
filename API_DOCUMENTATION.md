# Context Lab API Documentation

Complete API reference for the Context Lab package.

---

## Table of Contents

1. [Core Functions](#core-functions)
2. [Experiment Functions](#experiment-functions)
3. [Classes](#classes)
4. [Utility Functions](#utility-functions)
5. [Usage Examples](#usage-examples)

---

## Core Functions

### `generate_filler_text(num_words, domain="english")`

Generate filler text for creating synthetic documents.

**Parameters:**
- `num_words` (int): Number of words to generate
- `domain` (str, optional): Domain for text generation. Options:
  - `"english"`: Generic English text
  - `"technology_hebrew"`: Hebrew technology terms
  - `"medical_hebrew"`: Hebrew medical terms
  - `"legal_hebrew"`: Hebrew legal terms

**Returns:**
- `str`: Generated filler text

**Example:**
```python
from context_lab import generate_filler_text

text = generate_filler_text(100, domain="english")
print(len(text.split()))  # ~100 words

hebrew_text = generate_filler_text(100, domain="medical_hebrew")
```

---

### `embed_critical_fact(document, fact, position)`

Embed a critical fact at a specific position in a document.

**Parameters:**
- `document` (str): The base document text
- `fact` (str): The critical fact to embed
- `position` (str): Where to embed. Options: `"START"`, `"MIDDLE"`, `"END"`

**Returns:**
- `str`: Document with embedded fact

**Example:**
```python
from context_lab import embed_critical_fact

doc = "This is a long document. " * 50
fact = "The secret code is ALPHA123"
result = embed_critical_fact(doc, fact, "MIDDLE")
```

---

### `split_documents(documents, chunk_size=500, overlap=50)`

Split documents into smaller chunks for RAG.

**Parameters:**
- `documents` (List[str]): List of documents to split
- `chunk_size` (int, optional): Target size in words. Default: 500
- `overlap` (int, optional): Overlap between chunks. Default: 50

**Returns:**
- `List[str]`: List of text chunks

**Example:**
```python
from context_lab import split_documents

docs = ["Long document one. " * 200, "Long document two. " * 200]
chunks = split_documents(docs, chunk_size=300, overlap=50)
print(f"Created {len(chunks)} chunks")
```

---

### `count_tokens(text)`

Count approximate number of tokens in text.

**Parameters:**
- `text` (str): Text to count tokens for

**Returns:**
- `int`: Approximate token count

**Example:**
```python
from context_lab import count_tokens

text = "This is a sample sentence for token counting."
tokens = count_tokens(text)
print(f"Tokens: {tokens}")
```

---

### `evaluate_accuracy(response, expected_answer=None)`

Evaluate accuracy of LLM response against expected answer.

**Parameters:**
- `response` (str): LLM's response
- `expected_answer` (str, optional): Expected answer. If None, returns 1.0 for non-empty responses

**Returns:**
- `float`: Accuracy score between 0.0 and 1.0

**Example:**
```python
from context_lab import evaluate_accuracy

response = "The answer is 42"
expected = "42"
accuracy = evaluate_accuracy(response, expected)
print(f"Accuracy: {accuracy}")
```

---

### `ollama_query(context, query, use_real=None)`

Query LLM with given context and question.

**Parameters:**
- `context` (str): Context window text
- `query` (str): Question to ask
- `use_real` (bool, optional): If True, use real Ollama. If False, simulate. If None, auto-detect

**Returns:**
- `str`: LLM response

**Example:**
```python
from context_lab import ollama_query

context = "The capital of France is Paris."
query = "What is the capital of France?"
response = ollama_query(context, query, use_real=False)
```

---

### `nomic_embed_text(chunks, use_real=None)`

Generate embeddings for text chunks.

**Parameters:**
- `chunks` (List[str]): Text chunks to embed
- `use_real` (bool, optional): Use real embeddings if True

**Returns:**
- `List[np.ndarray]`: List of embedding vectors

**Example:**
```python
from context_lab import nomic_embed_text

chunks = ["chunk one", "chunk two", "chunk three"]
embeddings = nomic_embed_text(chunks, use_real=False)
print(f"Generated {len(embeddings)} embeddings")
```

---

## Experiment Functions

### `experiment1_needle_in_haystack(num_docs=5, use_real_llm=None)`

Run Experiment 1: Needle in Haystack (Lost in the Middle).

**Purpose:** Demonstrate that LLM accuracy drops when critical facts are in the middle of long contexts.

**Parameters:**
- `num_docs` (int, optional): Number of documents per position. Default: 5
- `use_real_llm` (bool, optional): Use real Ollama if True

**Returns:**
- `dict`: Results containing:
  ```python
  {
      "position_accuracy": {
          "START": float,
          "MIDDLE": float,
          "END": float
      },
      "average_latency": float,
      "expected_outcome": str
  }
  ```

**Example:**
```python
from context_lab import experiment1_needle_in_haystack

results = experiment1_needle_in_haystack(num_docs=5, use_real_llm=False)
print(f"START accuracy: {results['position_accuracy']['START']}")
print(f"MIDDLE accuracy: {results['position_accuracy']['MIDDLE']}")
print(f"END accuracy: {results['position_accuracy']['END']}")
```

**Expected Outcome:** MIDDLE accuracy < START/END accuracy

---

### `experiment2_context_window_size(doc_counts=[2, 5, 10, 20, 50], use_real_llm=None)`

Run Experiment 2: Context Window Size Impact.

**Purpose:** Show how increasing context size affects latency and accuracy.

**Parameters:**
- `doc_counts` (List[int], optional): Document counts to test. Default: [2, 5, 10, 20, 50]
- `use_real_llm` (bool, optional): Use real Ollama if True

**Returns:**
- `List[dict]`: List of results, each containing:
  ```python
  {
      "num_docs": int,
      "tokens_used": int,
      "latency": float,
      "accuracy": float
  }
  ```

**Example:**
```python
from context_lab import experiment2_context_window_size

results = experiment2_context_window_size(
    doc_counts=[2, 5, 10],
    use_real_llm=False
)

for r in results:
    print(f"Docs: {r['num_docs']}, Tokens: {r['tokens_used']}, Latency: {r['latency']:.2f}s")
```

**Expected Outcome:** Latency increases, accuracy may decrease with more documents

---

### `experiment3_rag_vs_full_context(num_docs=20, use_real_llm=None)`

Run Experiment 3: RAG vs Full Context.

**Purpose:** Compare RAG performance against full context provision.

**Parameters:**
- `num_docs` (int, optional): Number of Hebrew documents. Default: 20
- `use_real_llm` (bool, optional): Use real Ollama if True

**Returns:**
- `dict`: Results containing:
  ```python
  {
      "full_context": {
          "tokens": int,
          "latency": float,
          "accuracy": float,
          "response": str
      },
      "rag": {
          "tokens": int,
          "latency": float,
          "accuracy": float,
          "response": str
      },
      "comparison": {
          "speedup": float,
          "token_reduction_pct": float,
          "accuracy_improvement_pct": float
      },
      "expected_outcome": str
  }
  ```

**Example:**
```python
from context_lab import experiment3_rag_vs_full_context

results = experiment3_rag_vs_full_context(num_docs=20, use_real_llm=True)

print(f"Full Context: {results['full_context']['accuracy']:.1%} accuracy")
print(f"RAG: {results['rag']['accuracy']:.1%} accuracy")
print(f"Speedup: {results['comparison']['speedup']:.2f}x")
print(f"Token reduction: {results['comparison']['token_reduction_pct']:.1f}%")
```

**Expected Outcome:** RAG faster and more accurate than full context

---

### `experiment4_context_strategies(num_actions=10, use_real_llm=None)`

Run Experiment 4: Context Engineering Strategies.

**Purpose:** Benchmark SELECT, COMPRESS, and WRITE strategies for context management.

**Parameters:**
- `num_actions` (int, optional): Number of conversation turns. Default: 10
- `use_real_llm` (bool, optional): Use real Ollama if True

**Returns:**
- `dict`: Results containing:
  ```python
  {
      "strategies": {
          "BASELINE": {
              "avg_latency": float,
              "avg_accuracy": float,
              "avg_tokens": float
          },
          "SELECT": {...},
          "COMPRESS": {...},
          "WRITE": {...}
      },
      "expected_outcome": str
  }
  ```

**Example:**
```python
from context_lab import experiment4_context_strategies

results = experiment4_context_strategies(num_actions=10, use_real_llm=False)

for strategy, metrics in results['strategies'].items():
    print(f"{strategy}: {metrics['avg_accuracy']:.2f} accuracy, "
          f"{metrics['avg_latency']:.2f}s latency")
```

**Expected Outcome:** SELECT/COMPRESS/WRITE outperform BASELINE

---

## Classes

### `SimpleVectorStore`

Vector store using ChromaDB with simulation fallback.

#### Constructor

```python
SimpleVectorStore(use_real=None, collection_name="context_lab", reset=True)
```

**Parameters:**
- `use_real` (bool, optional): Use real ChromaDB if True. Auto-detects if None
- `collection_name` (str, optional): ChromaDB collection name. Default: "context_lab"
- `reset` (bool, optional): Reset collection on init. Default: True

**Example:**
```python
from context_lab import SimpleVectorStore

# Simulation mode
store = SimpleVectorStore(use_real=False)

# Real ChromaDB mode
store = SimpleVectorStore(use_real=True, collection_name="my_collection")
```

#### Methods

##### `add(chunks, embeddings)`

Add chunks and their embeddings to the vector store.

**Parameters:**
- `chunks` (List[str]): Text chunks
- `embeddings` (List[np.ndarray]): Embedding vectors

**Example:**
```python
store.add(["chunk1", "chunk2"], [embedding1, embedding2])
```

##### `similarity_search(query, k=3)`

Search for most similar chunks to query.

**Parameters:**
- `query` (str): Search query
- `k` (int, optional): Number of results. Default: 3

**Returns:**
- `List[str]`: Top-k most similar chunks

**Example:**
```python
results = store.similarity_search("What are the side effects?", k=5)
for i, chunk in enumerate(results):
    print(f"{i+1}. {chunk[:100]}...")
```

---

### `ContextAgent`

Agent for simulating multi-turn conversations with different context strategies.

#### Constructor

```python
ContextAgent(strategy="BASELINE", vector_store=None, token_limit=2000)
```

**Parameters:**
- `strategy` (str, optional): Strategy name. Options: "BASELINE", "SELECT", "COMPRESS", "WRITE"
- `vector_store` (SimpleVectorStore, optional): Vector store for SELECT strategy
- `token_limit` (int, optional): Token limit for COMPRESS strategy. Default: 2000

**Example:**
```python
from context_lab import ContextAgent, SimpleVectorStore

# BASELINE agent
agent_baseline = ContextAgent(strategy="BASELINE")

# SELECT agent with RAG
store = SimpleVectorStore(use_real=False)
agent_select = ContextAgent(strategy="SELECT", vector_store=store)

# COMPRESS agent
agent_compress = ContextAgent(strategy="COMPRESS", token_limit=1500)

# WRITE agent
agent_write = ContextAgent(strategy="WRITE")
```

#### Methods

##### `execute(action, use_real_llm=None)`

Execute an action with the configured strategy.

**Parameters:**
- `action` (dict): Action containing:
  ```python
  {
      "description": str,  # Task description
      "context_needed": bool  # Whether context is needed
  }
  ```
- `use_real_llm` (bool, optional): Use real LLM

**Returns:**
- `dict`: Result containing:
  ```python
  {
      "response": str,
      "tokens_used": int,
      "latency": float,
      "accuracy": float
  }
  ```

**Example:**
```python
action = {
    "description": "Summarize the conversation so far",
    "context_needed": True
}

result = agent.execute(action, use_real_llm=False)
print(f"Response: {result['response']}")
print(f"Tokens: {result['tokens_used']}, Latency: {result['latency']:.2f}s")
```

##### `reset()`

Reset agent's conversation history.

**Example:**
```python
agent.reset()
```

---

## Utility Functions

### `get_llm(model="llama2")`

Get LLM instance (Ollama).

**Parameters:**
- `model` (str, optional): Model name. Default: "llama2"

**Returns:**
- `Ollama`: LLM instance or None if not available

**Example:**
```python
from context_lab import get_llm

llm = get_llm(model="llama2")
if llm:
    response = llm.invoke("Hello, how are you?")
```

---

### `get_embeddings(model="nomic-embed-text")`

Get embeddings model.

**Parameters:**
- `model` (str, optional): Embedding model name. Default: "nomic-embed-text"

**Returns:**
- `OllamaEmbeddings` or `SentenceTransformerEmbeddings`: Embeddings instance

**Example:**
```python
from context_lab import get_embeddings

embeddings = get_embeddings()
if embeddings:
    vector = embeddings.embed_query("Hello world")
```

---

## Usage Examples

### Complete Workflow Example

```python
from context_lab import (
    experiment1_needle_in_haystack,
    experiment2_context_window_size,
    experiment3_rag_vs_full_context,
    experiment4_context_strategies,
)

# Run all experiments
print("Running Experiment 1: Needle in Haystack...")
exp1 = experiment1_needle_in_haystack(num_docs=5, use_real_llm=False)
print(f"Middle accuracy: {exp1['position_accuracy']['MIDDLE']:.2f}")

print("\nRunning Experiment 2: Context Window Size...")
exp2 = experiment2_context_window_size(doc_counts=[2, 5, 10], use_real_llm=False)
for r in exp2:
    print(f"  {r['num_docs']} docs: {r['latency']:.2f}s")

print("\nRunning Experiment 3: RAG vs Full Context...")
exp3 = experiment3_rag_vs_full_context(num_docs=20, use_real_llm=True)
print(f"RAG speedup: {exp3['comparison']['speedup']:.2f}x")

print("\nRunning Experiment 4: Context Strategies...")
exp4 = experiment4_context_strategies(num_actions=10, use_real_llm=False)
for strategy, metrics in exp4['strategies'].items():
    print(f"  {strategy}: {metrics['avg_accuracy']:.2f} accuracy")
```

### Custom RAG Pipeline Example

```python
from context_lab import (
    generate_filler_text,
    split_documents,
    nomic_embed_text,
    SimpleVectorStore,
    ollama_query,
)

# 1. Create documents
docs = [generate_filler_text(200) for _ in range(10)]

# 2. Split into chunks
chunks = split_documents(docs, chunk_size=100)

# 3. Generate embeddings
embeddings = nomic_embed_text(chunks, use_real=False)

# 4. Store in vector database
store = SimpleVectorStore(use_real=False)
store.add(chunks, embeddings)

# 5. Query
query = "What is the main topic?"
relevant_chunks = store.similarity_search(query, k=3)
context = "\n\n".join(relevant_chunks)

# 6. Get answer
response = ollama_query(context, query, use_real=False)
print(response)
```

---

## Error Handling

All functions handle errors gracefully:

```python
from context_lab import experiment3_rag_vs_full_context

try:
    results = experiment3_rag_vs_full_context(num_docs=20, use_real_llm=True)
except Exception as e:
    print(f"Experiment failed: {e}")
    # Falls back to simulation mode automatically
```

---

## Testing

Run unit tests:

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_core_functions.py -v

# With coverage
pytest tests/ --cov=context_lab --cov-report=html
```

---

## Installation

```bash
# From source
git clone https://github.com/volo10/context_lab
cd context_lab
pip install -e .

# With development dependencies
pip install -e ".[dev]"

# With visualization dependencies
pip install -e ".[viz]"
```

---

**Last Updated:** December 1, 2025  
**Version:** 1.0.0  
**Author:** Boris Volovelsky

