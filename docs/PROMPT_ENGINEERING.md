# Prompt Engineering Guide
## Context Window Impact Analysis Lab

**Version:** 1.0.0
**Last Updated:** December 2025

---

## 1. Introduction

This document provides a comprehensive guide to prompt engineering techniques used in the Context Lab, with emphasis on context window optimization and multilingual support.

---

## 2. Prompt Templates

### 2.1 Basic Query Template

```
Based on the following context, answer the question concisely.

Context:
{context}

Question: {query}

Answer:
```

**Token breakdown:**
- Fixed overhead: ~20 tokens
- Context: Variable
- Question: ~10-50 tokens
- Response: ~50-200 tokens

### 2.2 Hebrew Medical Query Template

```
You are a helpful medical assistant. You are given Hebrew text about medications.

The question asks: "What are the side effects of {medicine}?" (in Hebrew: {query})

Hebrew Context:
{context}

Instructions:
1. Look for the medicine name "{hebrew_name}" in the Hebrew context
2. Find the sentence that lists side effects (תופעות לוואי)
3. Extract and list ALL the side effects mentioned

Side effects found:
```

**Design Rationale:**
- Bilingual instruction improves comprehension
- Explicit extraction steps guide the model
- Hebrew terminology anchor (תופעות לוואי) helps locate relevant content

### 2.3 RAG Context Template

```
You are answering a question based on retrieved documents.

Retrieved Context (most relevant sections):
---
{chunk_1}
---
{chunk_2}
---
{chunk_3}

Question: {query}

Instructions:
- Answer based ONLY on the provided context
- If the answer is not in the context, say "Information not found"
- Be concise and specific

Answer:
```

---

## 3. Prompt Engineering Principles

### 3.1 Context Window Optimization

**Principle 1: Front-load critical information**
```
CRITICAL: {important_fact}

Background context follows...
{long_context}
```

**Principle 2: Use clear delimiters**
```
### DOCUMENT 1 ###
{doc1}

### DOCUMENT 2 ###
{doc2}

### QUESTION ###
{query}
```

**Principle 3: Summarize before full context**
```
SUMMARY: This context contains information about {topic}.

Key facts:
1. {fact1}
2. {fact2}

Full context for reference:
{full_context}
```

### 3.2 Lost-in-the-Middle Mitigation

Research (Liu et al., 2023) shows LLMs perform best on information at:
- **Start:** ~90% accuracy
- **Middle:** ~50% accuracy
- **End:** ~85% accuracy

**Mitigation Strategies:**

1. **Reorder by relevance:**
```python
def optimize_context_order(chunks, query):
    """Place most relevant chunks at start and end."""
    ranked = rank_by_relevance(chunks, query)
    n = len(ranked)

    # Interleave: most relevant at edges
    optimized = []
    for i, chunk in enumerate(ranked):
        if i % 2 == 0:
            optimized.insert(0, chunk)  # Front
        else:
            optimized.append(chunk)      # Back

    return optimized
```

2. **Explicit recall prompts:**
```
IMPORTANT: Pay special attention to information in the MIDDLE of this context.
The answer may be hidden between other paragraphs.

{context}
```

3. **Chunked attention:**
```
Read each section carefully:

[Section A - READ CAREFULLY]
{section_a}

[Section B - READ CAREFULLY]
{section_b}

[Section C - READ CAREFULLY]
{section_c}
```

### 3.3 Multilingual Prompts

**Hebrew-English Hybrid:**
```
שאלה (Question): {hebrew_query}

Context (הקשר):
{hebrew_context}

Instructions (הוראות):
1. Answer in the same language as the question
2. Use exact quotes from the context when possible
3. If unsure, indicate with "לא ברור" or "Not clear"

תשובה (Answer):
```

**Language Detection:**
```python
def detect_language(text):
    """Detect if text contains Hebrew characters."""
    hebrew_chars = sum(1 for c in text if '\u0590' <= c <= '\u05FF')
    return 'hebrew' if hebrew_chars > len(text) * 0.3 else 'english'
```

---

## 4. Strategy-Specific Prompts

### 4.1 SELECT Strategy (RAG)

**Retrieval prompt:**
```
Based on the following retrieved passages, answer the question.

Passage 1 (Relevance: High):
{passage_1}

Passage 2 (Relevance: Medium):
{passage_2}

Passage 3 (Relevance: Medium):
{passage_3}

Question: {query}

Guidelines:
- Synthesize information from multiple passages
- Cite passage numbers when making claims
- Acknowledge if information is incomplete

Answer:
```

### 4.2 COMPRESS Strategy (Summarization)

**Compression prompt:**
```
Summarize the following conversation history into key facts.
Preserve: names, numbers, decisions, action items.
Remove: pleasantries, redundant explanations, filler.

Conversation History:
{full_history}

Compressed Summary (bullet points):
```

**Context with summary:**
```
Previous Context Summary:
{compressed_summary}

Recent Interactions:
{recent_2_interactions}

Current Question: {query}

Answer:
```

### 4.3 WRITE Strategy (Memory)

**Fact extraction prompt:**
```
Extract the single most important fact from this interaction.
Format: [CATEGORY]: [FACT]

Interaction:
User: {user_input}
Assistant: {assistant_response}

Key Fact:
```

**Memory retrieval prompt:**
```
Stored Facts:
{scratchpad_contents}

Current Question: {query}

Using the stored facts above, answer the question:
```

### 4.4 ISOLATE Strategy (Compartments)

**Compartmentalized prompt:**
```
You are answering a question about {domain}.

Relevant {domain} Context:
{compartment_contents}

Note: Other domain information has been filtered out to focus your response.

Question: {query}

Answer (focusing on {domain}):
```

---

## 5. Prompt Patterns Catalog

### 5.1 Zero-Shot Patterns

**Direct instruction:**
```
Answer this question: {query}
Context: {context}
```

**Role-based:**
```
You are an expert in {domain}.
Based on your expertise, answer: {query}
```

### 5.2 Few-Shot Patterns

**Example-guided:**
```
Examples:
Q: What is the capital of France?
A: Paris

Q: What is the capital of Japan?
A: Tokyo

Q: {query}
A:
```

### 5.3 Chain-of-Thought (CoT)

**Step-by-step reasoning:**
```
Question: {query}

Let's think step by step:
1. First, I'll identify the key information in the context...
2. Then, I'll analyze how it relates to the question...
3. Finally, I'll formulate the answer...

Context: {context}

Step-by-step analysis:
```

### 5.4 Self-Consistency

**Multiple reasoning paths:**
```
Question: {query}

Generate 3 different approaches to answer this question:

Approach 1:
{reasoning_1}

Approach 2:
{reasoning_2}

Approach 3:
{reasoning_3}

Final Answer (most consistent):
```

---

## 6. Evaluation Prompts

### 6.1 Accuracy Evaluation

```
Evaluate if the response correctly answers the question.

Question: {query}
Expected Answer: {expected}
Actual Response: {response}

Evaluation Criteria:
- Factual correctness (0-5)
- Completeness (0-5)
- Relevance (0-5)

Score:
```

### 6.2 Response Quality

```
Rate the quality of this response:

Response: {response}

Criteria:
1. Clarity (1-5): Is it easy to understand?
2. Conciseness (1-5): Is it appropriately brief?
3. Accuracy (1-5): Is it factually correct?
4. Helpfulness (1-5): Does it address the user's need?

Total Score (/20):
Explanation:
```

---

## 7. Anti-Patterns to Avoid

### 7.1 Token Waste Patterns

**Bad: Unnecessary verbosity**
```
Please kindly answer the following question that I am about to ask you,
if you would be so kind as to do so, based on the context that I will
provide below for your reference and consideration...
```

**Good: Direct and efficient**
```
Answer based on context:
Context: {context}
Question: {query}
```

### 7.2 Confusion Patterns

**Bad: Ambiguous instructions**
```
Maybe look at the context and possibly answer the question if you can.
```

**Good: Clear directives**
```
Using ONLY the provided context, answer the question.
If the answer is not in the context, respond "Not found."
```

### 7.3 Hallucination Triggers

**Bad: Encouraging speculation**
```
What do you think the answer might be?
```

**Good: Grounding in evidence**
```
Quote the relevant sentence from the context that answers this question.
```

---

## 8. Token Optimization Techniques

### 8.1 Abbreviation Dictionary

| Full Form | Abbreviated | Tokens Saved |
|-----------|-------------|--------------|
| "Based on the following context" | "Context:" | 4 |
| "Please answer the question" | "Q:" | 3 |
| "The answer is" | "A:" | 2 |

### 8.2 Compression Techniques

**Before (45 tokens):**
```
I would like you to please read through the following context
very carefully and then answer the question that I will ask
at the end of this prompt based on what you read.
```

**After (12 tokens):**
```
Read context, answer question.

Context: {context}
Q: {query}
```

### 8.3 Dynamic Prompt Sizing

```python
def build_prompt(context, query, max_tokens=4096):
    """Build prompt within token budget."""
    base_prompt = f"Q: {query}\nA:"
    base_tokens = count_tokens(base_prompt)

    available = max_tokens - base_tokens - 500  # Reserve for response

    # Truncate context if needed
    context_tokens = count_tokens(context)
    if context_tokens > available:
        # Smart truncation: keep start and end
        context = truncate_smart(context, available)

    return f"Context: {context}\n\n{base_prompt}"
```

---

## 9. Testing Prompts

### 9.1 Prompt Regression Tests

```python
test_cases = [
    {
        "context": "The capital of France is Paris.",
        "query": "What is the capital of France?",
        "expected": "Paris",
        "min_accuracy": 0.95
    },
    {
        "context": "אדוויל גורם לכאבי בטן.",
        "query": "מהן תופעות הלוואי של אדוויל?",
        "expected": "כאבי בטן",
        "min_accuracy": 0.80
    }
]
```

### 9.2 A/B Testing Framework

```python
def ab_test_prompts(prompt_a, prompt_b, test_cases, n=100):
    """Compare two prompt templates."""
    results_a = [evaluate(prompt_a, tc) for tc in test_cases * n]
    results_b = [evaluate(prompt_b, tc) for tc in test_cases * n]

    return {
        'prompt_a_accuracy': mean(results_a),
        'prompt_b_accuracy': mean(results_b),
        'p_value': statistical_test(results_a, results_b)
    }
```

---

## 10. References

1. OpenAI Prompt Engineering Guide
2. Anthropic Claude Prompt Library
3. Liu et al. (2023) "Lost in the Middle: How Language Models Use Long Contexts"
4. Wei et al. (2022) "Chain-of-Thought Prompting"
5. Wang et al. (2023) "Self-Consistency Improves Chain of Thought Reasoning"
