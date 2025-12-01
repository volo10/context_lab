# LLM Prompts Documentation

This document contains all the prompts used in the Context Lab experiments, providing transparency and reproducibility for the research.

---

## Table of Contents

1. [General Purpose Prompts](#general-purpose-prompts)
2. [Experiment 1: Needle in Haystack Prompts](#experiment-1-needle-in-haystack-prompts)
3. [Experiment 2: Context Window Size Prompts](#experiment-2-context-window-size-prompts)
4. [Experiment 3: RAG vs Full Context Prompts](#experiment-3-rag-vs-full-context-prompts)
5. [Experiment 4: Context Strategies Prompts](#experiment-4-context-strategies-prompts)

---

## General Purpose Prompts

### Standard English Query Prompt

**Location:** `context_lab.py` → `ollama_query()`

```python
prompt = f"""Based on the following context, answer the question concisely.

Context:
{context}

Question: {query}

Answer:"""
```

**Purpose:** Default prompt for English queries  
**Temperature:** 0.1 (low for consistency)  
**Expected Output:** Concise answer extracted from context

---

## Experiment 1: Needle in Haystack Prompts

### Critical Fact Extraction Prompt

**Query Template:**
```python
query = f"What is the secret code mentioned in the document?"
```

**Full Prompt (via ollama_query):**
```
Based on the following context, answer the question concisely.

Context:
[5 documents with embedded critical fact ALPHA{i}BETA{random_numbers}]

Question: What is the secret code mentioned in the document?

Answer:
```

**Purpose:** Test if LLM can find a specific fact embedded at START, MIDDLE, or END  
**Critical Fact Format:** `The secret code is ALPHA0BETA1234`  
**Evaluation:** Checks if response contains the exact code

---

## Experiment 2: Context Window Size Prompts

### Standard Document Query

**Query:**
```python
query = "What is the main topic discussed in these documents?"
```

**Full Prompt:**
```
Based on the following context, answer the question concisely.

Context:
[2-50 documents of varying lengths]

Question: What is the main topic discussed in these documents?

Answer:
```

**Purpose:** Test LLM performance as context size increases  
**Evaluation:** General accuracy assessment based on response quality

---

## Experiment 3: RAG vs Full Context Prompts

### Hebrew Medical Query (Primary)

**Location:** `context_lab.py` → `experiment3_rag_vs_full_context()`

**Query (Hebrew):**
```python
query = "מהן תופעות הלוואי של אדוויל?"
# Translation: "What are the side effects of Advil?"
```

**Full Prompt for Hebrew (Improved for Real LLM):**
```python
prompt = f"""You are a helpful medical assistant. You are given Hebrew text about medications.

The question asks: "What are the side effects of Advil?" (in Hebrew: {query})

Hebrew Context:
{context}

Instructions:
1. Look for the medicine name "אדוויל" (Advil) or "איבופרופן" (Ibuprofen) in the Hebrew context
2. Find the sentence that lists side effects (תופעות לוואי)
3. Extract and list ALL the side effects mentioned

Side effects found:"""
```

**Purpose:** Directive prompt that guides LLM to:
- Identify the medicine name in Hebrew
- Find the side effects sentence
- Extract all symptoms mentioned

**Why This Prompt Works:**
1. **Bilingual approach:** English instructions with Hebrew content helps llama2
2. **Step-by-step:** Clear 3-step process
3. **Specific keywords:** Mentions both drug names (אדוויל, איבופרופן)
4. **Domain-specific:** Medical assistant role primes the model

**Target Fact (Hebrew):**
```
אדוויל (איבופרופן) עלול לגרום לכאבי בטן, בחילות, צרבת וסחרחורת בכ-10% מהמטופלים.
```
*Translation: "Advil (ibuprofen) may cause stomach pain, nausea, heartburn, and dizziness in about 10% of patients."*

**Evaluation Keywords:**
- כאבי בטן (stomach pain)
- בחילות (nausea)
- צרבת (heartburn)
- סחרחורת (dizziness)
- איבופרופן (ibuprofen)
- אדוויל (Advil)

### Mode A: Full Context Prompt

**Setup:**
```python
full_context = "\n\n".join(documents)  # All 20 Hebrew documents (~6000 tokens)
```

**Prompt:** Same Hebrew medical prompt as above

**Expected Issue:** Lost in the Middle - fact buried in 20 documents

### Mode B: RAG (Retrieval-Augmented Generation) Prompt

**Setup:**
```python
# Retrieve top-5 most similar chunks
relevant_chunks = vector_store.similarity_search(query, k=5)

# Hybrid retrieval: Add keyword fallback if needed
if medicine_name not in relevant_chunks:
    keyword_chunks = [chunk for chunk in chunks if medicine_name in chunk]
    relevant_chunks = keyword_chunks[:2] + relevant_chunks[:3]

rag_context = "\n\n".join(relevant_chunks)  # ~400 tokens
```

**Prompt:** Same Hebrew medical prompt as above

**Expected Result:** Focused context leads to better extraction

---

## Experiment 4: Context Strategies Prompts

### Strategy-Specific Prompts

#### BASELINE Strategy

**Prompt:**
```python
prompt = f"""Continue the conversation based on ALL previous history.

History:
{full_history}

Current Task: {action['description']}

Response:"""
```

**Purpose:** No optimization, uses full history every time

#### SELECT Strategy (RAG-based)

**Prompt:**
```python
# First: Retrieve relevant history
relevant_history = rag_search(history, current_query, k=5)

prompt = f"""Continue the conversation based on RELEVANT previous history.

Relevant History:
{relevant_history}

Current Task: {action['description']}

Response:"""
```

**Purpose:** Only include relevant parts of history using RAG

#### COMPRESS Strategy

**Prompt for Summarization:**
```python
summarize_prompt = f"""Summarize the following conversation history concisely, keeping only key facts and decisions.

Full History:
{history}

Concise Summary:"""
```

**Prompt for Task:**
```python
prompt = f"""Continue the conversation based on this summary.

Summary:
{compressed_history}

Current Task: {action['description']}

Response:"""
```

**Purpose:** Summarize history to reduce tokens while keeping key info

#### WRITE Strategy (External Memory)

**Prompt for Fact Extraction:**
```python
extract_prompt = f"""Extract key facts from this conversation turn and format as bullet points.

Conversation Turn:
{turn}

Key Facts:"""
```

**Prompt for Task:**
```python
prompt = f"""Continue the conversation using this scratchpad of key facts.

Scratchpad:
{scratchpad_facts}

Current Task: {action['description']}

Response:"""
```

**Purpose:** Maintain external scratchpad of facts, only pass scratchpad (not full history)

---

## Prompt Engineering Best Practices Used

### 1. **Clear Instructions**
All prompts start with clear role and task definition:
```
"You are a helpful medical assistant..."
```

### 2. **Structured Format**
Use clear sections:
- Context
- Question/Task
- Instructions (when needed)
- Expected Output Format

### 3. **Bilingual Approach for Hebrew**
```
English instructions + Hebrew content = Better llama2 performance
```

### 4. **Directive Style**
Step-by-step instructions work better than open-ended:
```
1. Look for X
2. Find Y
3. Extract Z
```

### 5. **Low Temperature**
```python
temperature=0.1  # For consistent, factual responses
```

### 6. **Domain Priming**
Set appropriate role:
- "medical assistant" for medicine queries
- "helpful assistant" for general queries

---

## Evaluation Prompts

### Accuracy Evaluation Logic

**For Exact Facts:**
```python
# Check if expected answer or its keywords are in response
if expected_answer.lower() in response.lower():
    return 1.0

# For Hebrew medical terms, check for keyword presence
keywords = ["בחילות", "כאבי בטן", "צרבת", "סחרחורת"]
matched = [kw for kw in keywords if kw in response.lower()]
if len(matched) >= len(keywords) * 0.7:  # 70% threshold
    return 1.0

return 0.0
```

**Purpose:** Flexible evaluation that handles:
- Exact matches
- Paraphrased responses
- Multi-keyword matching for Hebrew
- Partial credit for complex answers

---

## Embeddings and Retrieval

### Embedding Model Selection

**For Hebrew Text:**
```python
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
# 768 dimensions, supports 50+ languages including Hebrew
```

**For English Text:**
```python
# Option 1: Ollama embeddings
embeddings = OllamaEmbeddings(model='nomic-embed-text')

# Option 2: Sentence Transformers
model = SentenceTransformer('all-MiniLM-L6-v2')
# 384 dimensions, optimized for English
```

### Retrieval Query

**Vector Search:**
```python
query_embedding = model.encode(query)
results = vector_store.query(query_embeddings=[query_embedding], n_results=k)
```

**Hybrid Retrieval (Fallback):**
```python
# If semantic search fails, add keyword search
if target_keyword not in retrieved_chunks:
    keyword_chunks = [c for c in all_chunks if target_keyword in c]
    retrieved_chunks = keyword_chunks[:2] + retrieved_chunks[:3]
```

---

## Temperature Settings

| Experiment | Temperature | Reasoning |
|------------|-------------|-----------|
| Experiment 1 | 0.1 | Need consistent fact extraction |
| Experiment 2 | 0.1 | Measure latency consistently |
| Experiment 3 | 0.1 | Medical facts require precision |
| Experiment 4 | 0.1 | Compare strategies fairly |

**Low temperature (0.1)** ensures:
- Deterministic responses
- Minimal creativity
- Factual accuracy
- Fair comparison across runs

---

## Model Used

**LLM:** Ollama with `llama2` model (7B parameters)

**Why llama2:**
- Runs locally (privacy)
- Good balance of speed and quality
- Supports multilingual (including Hebrew with proper prompts)
- Consistent performance

**Limitations:**
- Hebrew understanding requires bilingual prompts
- Smaller context window than GPT-4
- May hallucinate without directive prompts

---

## Reproducibility

To reproduce results with these prompts:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Install Ollama and pull llama2
ollama pull llama2

# 3. Run experiments
python context_lab.py --experiment 1
python context_lab.py --experiment 2
python context_lab.py --experiment 3
python context_lab.py --experiment 4

# 4. Run tests
pytest tests/ -v
```

---

## Prompt Versioning

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Initial | Basic prompts, English-only |
| 1.1 | Improvement | Added Hebrew support, bilingual prompts |
| 1.2 | Current | Directive prompts, hybrid retrieval |

---

## References

- [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)
- [LangChain Documentation](https://python.langchain.com/docs/)
- [Sentence Transformers](https://www.sbert.net/)
- [Lost in the Middle Paper](https://arxiv.org/abs/2307.03172)

---

**Last Updated:** December 1, 2025  
**Author:** Boris Volovelsky  
**Repository:** https://github.com/volo10/context_lab

