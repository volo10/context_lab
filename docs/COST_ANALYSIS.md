# Cost Analysis Document
## Context Window Impact Analysis Lab

**Version:** 1.0.0
**Last Updated:** December 2025

---

## 1. Executive Summary

This document provides a comprehensive cost analysis for LLM context window usage, comparing different strategies and approaches tested in the Context Lab experiments.

### Key Findings

| Approach | Tokens/Query | Cost per 1K Queries | Relative Savings |
|----------|--------------|---------------------|------------------|
| Full Context | ~10,000 | $30.00 | Baseline |
| RAG (top-5) | ~2,500 | $7.50 | **75% savings** |
| COMPRESS | ~2,000 | $6.00 | **80% savings** |
| WRITE | ~500 | $1.50 | **95% savings** |

*Based on GPT-4 pricing: $0.03/1K input tokens*

---

## 2. Token Economics

### 2.1 Token Pricing Models (December 2025)

| Provider | Model | Input ($/1K) | Output ($/1K) | Context Window |
|----------|-------|--------------|---------------|----------------|
| OpenAI | GPT-4 Turbo | $0.01 | $0.03 | 128K |
| OpenAI | GPT-4o | $0.005 | $0.015 | 128K |
| OpenAI | GPT-3.5 | $0.0005 | $0.0015 | 16K |
| Anthropic | Claude 3 Opus | $0.015 | $0.075 | 200K |
| Anthropic | Claude 3.5 Sonnet | $0.003 | $0.015 | 200K |
| Google | Gemini Pro | $0.00025 | $0.0005 | 1M |
| Local | Ollama (free) | $0 | $0 | Model-dependent |

### 2.2 Token Estimation Formula

```
Estimated Tokens = Characters / 4

For multilingual (Hebrew):
Estimated Tokens = Characters / 3 (UTF-8 overhead)
```

**Exact calculation with tiktoken:**
```python
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
tokens = len(enc.encode(text))
```

---

## 3. Experiment-Specific Cost Analysis

### 3.1 Experiment 1: Needle in Haystack

**Configuration:** 5 docs × 200 words × 3 positions = 15 queries

| Position | Avg Tokens | Cost (GPT-4) | Cost (GPT-3.5) |
|----------|------------|--------------|----------------|
| Start | 250 | $0.0075 | $0.000375 |
| Middle | 250 | $0.0075 | $0.000375 |
| End | 250 | $0.0075 | $0.000375 |
| **Total** | 3,750 | $0.1125 | $0.005625 |

### 3.2 Experiment 2: Context Size Impact

**Configuration:** [2, 5, 10, 20, 50] docs × 200 words

| Docs | Tokens | Cost (GPT-4) | Latency Factor |
|------|--------|--------------|----------------|
| 2 | 400 | $0.012 | 1.0x |
| 5 | 1,000 | $0.030 | 1.5x |
| 10 | 2,000 | $0.060 | 2.0x |
| 20 | 4,000 | $0.120 | 3.0x |
| 50 | 10,000 | $0.300 | 5.0x |
| **Total** | 17,400 | $0.522 | — |

### 3.3 Experiment 3: RAG vs Full Context

**Configuration:** 20 Hebrew docs × 200 words

| Mode | Tokens | Cost (GPT-4) | Accuracy | Efficiency |
|------|--------|--------------|----------|------------|
| Full Context | 10,000 | $0.30 | ~33% | Baseline |
| RAG (k=5) | 2,500 | $0.075 | ~72% | **4x better** |

**ROI Calculation:**
```
Savings = (10,000 - 2,500) × $0.03/1K = $0.225 per query
Annual savings (10K queries/day) = $821,250
```

### 3.4 Experiment 4: Context Strategies

**Configuration:** 10-step conversation benchmark

| Strategy | Final Tokens | Cost/Conversation | Growth Rate |
|----------|--------------|-------------------|-------------|
| No Strategy | 5,000 | $0.15 | O(n) |
| SELECT | 1,250 | $0.0375 | O(k) |
| COMPRESS | 2,000 | $0.06 | O(L) |
| WRITE | 500 | $0.015 | O(n/3) |
| ISOLATE | 1,500 | $0.045 | O(c×k) |

---

## 4. Mathematical Models

### 4.1 Token Growth Models

**Linear Growth (No Strategy):**
$$T(n) = \sum_{i=1}^{n} w_i = n \cdot \bar{w}$$

Where:
- $n$ = number of interactions
- $w_i$ = tokens in interaction $i$
- $\bar{w}$ = average tokens per interaction

**SELECT Strategy (Bounded):**
$$T_{SELECT}(n) = k \cdot \bar{w}$$

Where $k$ = retrieval limit (constant)

**COMPRESS Strategy:**
$$T_{COMPRESS}(n) = \min(L, n \cdot \bar{w})$$

Where $L$ = token limit

**WRITE Strategy (Reduced Linear):**
$$T_{WRITE}(n) = \frac{n}{m} \cdot \bar{f}$$

Where:
- $m$ = extraction interval (default: 3)
- $\bar{f}$ = average fact size (< $\bar{w}$)

### 4.2 Cost Optimization Formula

**Optimal Strategy Selection:**

$$\text{Cost}_{optimal} = \min_{s \in S} \left( T_s(n) \cdot P_{input} + R_s(n) \cdot P_{output} \right)$$

Where:
- $S$ = {SELECT, COMPRESS, WRITE, ISOLATE}
- $T_s(n)$ = input tokens for strategy $s$
- $R_s(n)$ = output tokens for strategy $s$
- $P$ = price per token

---

## 5. Cost Optimization Recommendations

### 5.1 By Use Case

| Use Case | Recommended | Reason |
|----------|-------------|--------|
| Single query | Full Context | No overhead |
| Multi-turn chat | COMPRESS | Bounded cost |
| Research assistant | SELECT | Accuracy priority |
| High-volume API | WRITE | Minimal tokens |
| Domain-specific | ISOLATE | Prevents pollution |

### 5.2 Break-Even Analysis

**When RAG beats Full Context:**

$$n_{break} = \frac{C_{setup}}{C_{full} - C_{rag}}$$

Where:
- $C_{setup}$ = One-time embedding cost
- $C_{full}$ = Full context cost per query
- $C_{rag}$ = RAG cost per query

**Example:**
- Setup cost: $5.00 (embedding 1000 docs)
- Full context: $0.30/query
- RAG: $0.075/query

$$n_{break} = \frac{5.00}{0.30 - 0.075} = 22 \text{ queries}$$

After 22 queries, RAG becomes more economical.

### 5.3 Tiered Pricing Strategy

| Query Volume | Strategy | Expected Cost |
|--------------|----------|---------------|
| < 100/day | Full Context | ~$30/day |
| 100-1K/day | COMPRESS | ~$60/day |
| 1K-10K/day | RAG | ~$75/day |
| > 10K/day | WRITE | ~$150/day |

---

## 6. Hebrew/Multilingual Cost Considerations

### 6.1 UTF-8 Token Overhead

Hebrew text uses more bytes than ASCII:
- English: ~4 chars/token
- Hebrew: ~2-3 chars/token

**Impact:** Hebrew documents cost ~1.5x more in tokens than English equivalents.

### 6.2 Multilingual Embedding Costs

| Model | Dimensions | Languages | Embedding Cost |
|-------|------------|-----------|----------------|
| all-MiniLM-L6-v2 | 384 | English | Free (local) |
| paraphrase-multilingual | 384 | 50+ | Free (local) |
| OpenAI text-embedding-3-small | 1536 | 100+ | $0.02/1M tokens |

---

## 7. Infrastructure Costs

### 7.1 Local Deployment (Ollama)

| Component | Cost | Notes |
|-----------|------|-------|
| Hardware | $1,000-5,000 | GPU required |
| Electricity | ~$50/month | 24/7 operation |
| Maintenance | ~$100/month | Updates, monitoring |

**Break-even vs Cloud:** ~500K queries/month

### 7.2 ChromaDB Storage

| Documents | Storage | Memory | Monthly Cost (Cloud) |
|-----------|---------|--------|----------------------|
| 1K | 10 MB | 100 MB | ~$1 |
| 10K | 100 MB | 500 MB | ~$5 |
| 100K | 1 GB | 2 GB | ~$20 |
| 1M | 10 GB | 16 GB | ~$100 |

---

## 8. Cost Monitoring Dashboard

### 8.1 Key Metrics to Track

```python
# Cost tracking metrics
metrics = {
    'tokens_per_query': [],
    'cost_per_query': [],
    'strategy_distribution': {},
    'daily_spend': 0,
    'monthly_budget': 1000,
}
```

### 8.2 Budget Alerts

| Alert Level | Threshold | Action |
|-------------|-----------|--------|
| Warning | 80% of budget | Notify admin |
| Critical | 95% of budget | Switch to WRITE strategy |
| Emergency | 100% of budget | Rate limit / reject |

---

## 9. Appendix: Formulas Summary

| Metric | Formula |
|--------|---------|
| Token estimate | $T = \text{chars} / 4$ |
| Input cost | $C_{in} = T_{in} \times P_{in}$ |
| Output cost | $C_{out} = T_{out} \times P_{out}$ |
| Total cost | $C = C_{in} + C_{out}$ |
| RAG savings | $S = 1 - (T_{rag} / T_{full})$ |
| Break-even | $n = C_{setup} / (C_{full} - C_{rag})$ |

---

## 10. References

1. OpenAI Pricing: https://openai.com/pricing
2. Anthropic Pricing: https://anthropic.com/pricing
3. Token Counting: https://platform.openai.com/tokenizer
4. Liu et al. (2023) "Lost in the Middle" - Context window research
