# Context Window Impact Analysis Lab - Report Plan

## Overview

This document outlines the analysis plan and expected report structure for the Context Window Impact Lab. The lab consists of four experiments designed to demonstrate and analyze key phenomena in LLM context window management.

---

## Experiment 1: Needle in Haystack (Lost in the Middle)

### Objective
Demonstrate that LLM accuracy drops when critical information is placed in the middle of a long context window compared to the start or end positions.

### Methodology
- **Data Generation**: 5 synthetic documents, each 200 words
- **Critical Fact Placement**: Embed one critical fact at three positions:
  - **Start** (5% into document)
  - **Middle** (50% into document)  
  - **End** (95% into document)
- **Query**: "What is the critical fact mentioned in the document?"
- **Metrics**: Accuracy, Latency

### Expected Results Table

| Position | Avg Accuracy | Std Dev | Avg Latency (s) | Notes |
|----------|--------------|---------|-----------------|-------|
| Start    | 0.85 - 0.95  | ±0.05   | 0.15 - 0.25     | High retrieval success |
| Middle   | 0.45 - 0.65  | ±0.15   | 0.15 - 0.25     | **Lost in the Middle** |
| End      | 0.80 - 0.95  | ±0.05   | 0.15 - 0.25     | Recency bias helps |

### Expected Graph
**Figure 1.1: Accuracy by Fact Position**
- Bar chart showing accuracy for Start, Middle, End positions
- Error bars showing standard deviation
- Clear dip in the middle position

### Analysis Points
1. **Primacy Effect**: Facts at the beginning are well-retained
2. **Recency Effect**: Facts at the end benefit from recency bias
3. **Middle Degradation**: Demonstrates the "Lost in the Middle" phenomenon
4. **Implications**: Critical information should be placed at context boundaries

---

## Experiment 2: Context Window Size Impact

### Objective
Analyze how increasing context window size (number of documents) affects both latency and accuracy.

### Methodology
- **Document Sets**: Varying sizes: 2, 5, 10, 20, 50 documents
- **Document Size**: 200 words each
- **Query**: "Summarize the main points from the provided documents"
- **Metrics**: Token count, Latency, Accuracy

### Expected Results Table

| Num Docs | Token Count | Latency (s) | Accuracy | Token Growth | Latency Growth |
|----------|-------------|-------------|----------|--------------|----------------|
| 2        | ~100        | 0.15        | 0.90     | Baseline     | Baseline       |
| 5        | ~250        | 0.20        | 0.85     | 2.5x         | 1.3x           |
| 10       | ~500        | 0.35        | 0.80     | 5.0x         | 2.3x           |
| 20       | ~1000       | 0.60        | 0.70     | 10.0x        | 4.0x           |
| 50       | ~2500       | 1.50        | 0.55     | 25.0x        | 10.0x          |

### Expected Graphs

**Figure 2.1: Accuracy vs Context Size**
- Line plot with accuracy on Y-axis, number of documents on X-axis
- Shows clear downward trend

**Figure 2.2: Latency vs Token Count**
- Scatter plot with latency on Y-axis, token count on X-axis
- Shows near-linear or slightly super-linear growth

**Figure 2.3: Dual-Axis Chart**
- Combined view showing both accuracy (decreasing) and latency (increasing)
- Highlights the trade-off

### Analysis Points
1. **Accuracy Degradation**: More context introduces noise and confusion
2. **Latency Growth**: Processing time scales with context size
3. **Practical Limits**: Identify optimal context size before significant degradation
4. **Cost Implications**: More tokens = higher API costs

---

## Experiment 3: RAG vs Full Context

### Objective
Compare Retrieval-Augmented Generation (RAG) against providing full context to the LLM.

### Methodology
- **Corpus**: 20 documents (200 words each) on technical/medical topics
- **Target Information**: Specific fact embedded in one document
- **Query**: "What are the side effects of drug X?"
- **RAG Setup**:
  - Chunk size: 500 characters
  - Embedding model: nomic-embed-text (or similar)
  - Vector store: ChromaDB
  - Retrieved chunks: k=3

### Expected Results Table

| Approach      | Tokens Used | Latency (s) | Accuracy | Token Reduction | Speedup | Accuracy Gain |
|---------------|-------------|-------------|----------|-----------------|---------|---------------|
| Full Context  | ~1000       | 0.60        | 0.65     | Baseline        | 1.0x    | Baseline      |
| RAG (k=3)     | ~300        | 0.20        | 0.85     | 70%             | 3.0x    | +31%          |

### Expected Graphs

**Figure 3.1: Performance Comparison (Bar Chart)**
- Side-by-side comparison of Full Context vs RAG
- Three panels: Tokens, Latency, Accuracy

**Figure 3.2: Efficiency Gain (Radar Chart)**
- Multi-dimensional view showing:
  - Speed improvement
  - Token reduction
  - Accuracy improvement
  - Cost reduction

### Analysis Points
1. **Token Efficiency**: RAG dramatically reduces context size
2. **Speed Improvement**: Smaller context = faster inference
3. **Accuracy Improvement**: Focused context reduces noise
4. **Practical Benefits**: Lower costs, faster responses, better answers
5. **RAG Limitations**: Depends on retrieval quality; may miss relevant context

---

## Experiment 4: Context Engineering Strategies

### Objective
Benchmark three advanced context management strategies for multi-step agent conversations.

### Methodology
- **Conversation Length**: 10 steps
- **Strategies**:
  1. **SELECT**: RAG search on history (retrieve k=5 relevant interactions)
  2. **COMPRESS**: Summarize history when exceeding token limit (2000 tokens)
  3. **WRITE**: External memory/scratchpad storing key facts only
- **Baseline**: Full history (no management)

### Expected Results Table

| Strategy     | Avg Accuracy | Std Dev | Avg Latency (s) | Avg Tokens | Final Tokens | Notes |
|--------------|--------------|---------|-----------------|------------|--------------|-------|
| Full History | 0.60         | ±0.15   | 0.45            | 850        | 1500         | Baseline |
| SELECT (RAG) | 0.82         | ±0.08   | 0.25            | 400        | 600          | Best accuracy |
| COMPRESS     | 0.75         | ±0.10   | 0.30            | 500        | 800          | Good balance |
| WRITE        | 0.80         | ±0.09   | 0.20            | 300        | 450          | Most efficient |

### Expected Graphs

**Figure 4.1: Accuracy Over Time (Line Plot)**
- X-axis: Conversation step (1-10)
- Y-axis: Accuracy
- Four lines for each strategy
- Shows how accuracy degrades or maintains over conversation length

**Figure 4.2: Strategy Comparison (Grouped Bar Chart)**
- Compare all four metrics: Accuracy, Latency, Avg Tokens, Final Tokens
- Side-by-side bars for each strategy

**Figure 4.3: Token Growth Over Time**
- X-axis: Conversation step
- Y-axis: Token count
- Shows how each strategy manages token growth

### Analysis Points
1. **SELECT (RAG)**: 
   - Best accuracy by retrieving relevant past context
   - Controlled token growth
   - Requires vector store infrastructure
   
2. **COMPRESS**: 
   - Moderate performance
   - May lose important details in summarization
   - Good for long conversations
   
3. **WRITE**: 
   - Most token-efficient
   - Good accuracy with selective fact storage
   - Requires careful fact extraction logic
   
4. **Full History**:
   - Simplest approach
   - Poor scalability
   - Accuracy degrades with conversation length

---

## Overall Lab Report Structure

### 1. Executive Summary
- Brief overview of all four experiments
- Key findings and recommendations

### 2. Introduction
- Background on LLM context windows
- Known phenomena: Lost in the Middle, context overflow
- Importance of context engineering

### 3. Methodology
- Experimental design
- Simulation approach
- Metrics and evaluation criteria

### 4. Experiments and Results

#### 4.1 Experiment 1: Needle in Haystack
- Description
- Results table and graph
- Analysis

#### 4.2 Experiment 2: Context Size Impact
- Description
- Results table and graphs
- Analysis

#### 4.3 Experiment 3: RAG vs Full Context
- Description
- Results table and graphs
- Analysis

#### 4.4 Experiment 4: Context Strategies
- Description
- Results table and graphs
- Analysis

### 5. Discussion
- Cross-experiment insights
- Practical implications
- Best practices for context management

### 6. Recommendations
- When to use each strategy
- Context optimization guidelines
- Production deployment considerations

### 7. Limitations
- Simulation vs real LLM behavior
- Simplified accuracy metrics
- Language and domain constraints

### 8. Future Work
- Real LLM integration (Ollama, GPT-4, etc.)
- Additional strategies (e.g., hierarchical memory)
- Domain-specific testing
- Multi-language support

### 9. Conclusion
- Summary of findings
- Impact on LLM application design

### 10. References
- Related research papers
- LLM documentation
- Framework documentation

---

## Visualization Guidelines

### Color Scheme
- **Experiment 1**: Blue gradient (light to dark) for positions
- **Experiment 2**: Red-to-green gradient showing degradation
- **Experiment 3**: Contrasting colors (e.g., orange for Full, green for RAG)
- **Experiment 4**: Distinct color for each strategy

### Chart Types
- **Bar Charts**: Categorical comparisons (positions, strategies)
- **Line Plots**: Trends over time or scale
- **Scatter Plots**: Relationships between two continuous variables
- **Radar Charts**: Multi-dimensional comparisons

### Best Practices
- Include error bars where appropriate
- Label axes clearly with units
- Add legends for multi-series plots
- Use consistent color schemes across related charts
- Include data point annotations for key findings

---

## Code for Generating Plots

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Example: Experiment 1 - Accuracy by Position
def plot_experiment1(results):
    positions = ['Start', 'Middle', 'End']
    accuracies = [results['summary'][pos.lower()]['avg_accuracy'] for pos in positions]
    stds = [results['summary'][pos.lower()]['std_accuracy'] for pos in positions]
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(positions, accuracies, yerr=stds, capsize=5, 
                   color=['#2E86AB', '#A23B72', '#F18F01'])
    plt.ylabel('Accuracy', fontsize=12)
    plt.xlabel('Fact Position', fontsize=12)
    plt.title('Experiment 1: Accuracy by Fact Position in Context', fontsize=14, weight='bold')
    plt.ylim(0, 1.0)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('exp1_accuracy_by_position.png', dpi=300)
    plt.show()

# Example: Experiment 2 - Dual Axis Chart
def plot_experiment2(results):
    df = results['dataframe']
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Accuracy line
    color = 'tab:red'
    ax1.set_xlabel('Number of Documents', fontsize=12)
    ax1.set_ylabel('Accuracy', color=color, fontsize=12)
    ax1.plot(df['num_docs'], df['accuracy'], marker='o', color=color, linewidth=2, label='Accuracy')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, 1.0)
    
    # Latency line (second y-axis)
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Latency (seconds)', color=color, fontsize=12)
    ax2.plot(df['num_docs'], df['latency'], marker='s', color=color, linewidth=2, label='Latency')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Experiment 2: Context Size Impact on Accuracy and Latency', fontsize=14, weight='bold')
    fig.tight_layout()
    plt.savefig('exp2_context_size_impact.png', dpi=300)
    plt.show()

# Example: Experiment 3 - Side-by-side Comparison
def plot_experiment3(results):
    metrics = ['Tokens', 'Latency (s)', 'Accuracy']
    full_values = [
        results['full_context']['tokens'],
        results['full_context']['latency'],
        results['full_context']['accuracy']
    ]
    rag_values = [
        results['rag']['tokens'],
        results['rag']['latency'],
        results['rag']['accuracy']
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, full_values, width, label='Full Context', color='#FF6B6B')
    bars2 = ax.bar(x + width/2, rag_values, width, label='RAG', color='#4ECDC4')
    
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Experiment 3: RAG vs Full Context Comparison', fontsize=14, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('exp3_rag_vs_full.png', dpi=300)
    plt.show()

# Example: Experiment 4 - Strategy Comparison
def plot_experiment4(results):
    df = results['dataframe']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Accuracy
    axes[0, 0].bar(df['Strategy'], df['Avg Accuracy'], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[0, 0].set_ylabel('Avg Accuracy')
    axes[0, 0].set_title('Average Accuracy')
    axes[0, 0].set_ylim(0, 1.0)
    
    # Latency
    axes[0, 1].bar(df['Strategy'], df['Avg Latency'], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[0, 1].set_ylabel('Avg Latency (s)')
    axes[0, 1].set_title('Average Latency')
    
    # Average Tokens
    axes[1, 0].bar(df['Strategy'], df['Avg Tokens'], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[1, 0].set_ylabel('Avg Tokens')
    axes[1, 0].set_title('Average Token Count')
    
    # Final Tokens
    axes[1, 1].bar(df['Strategy'], df['Final Tokens'], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[1, 1].set_ylabel('Final Tokens')
    axes[1, 1].set_title('Final Token Count')
    
    plt.suptitle('Experiment 4: Context Strategy Comparison', fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig('exp4_strategy_comparison.png', dpi=300)
    plt.show()
```

---

## Statistical Analysis

### Hypothesis Testing

#### Experiment 1
- **Null Hypothesis (H₀)**: Position has no effect on accuracy
- **Alternative Hypothesis (H₁)**: Middle position has lower accuracy
- **Test**: One-way ANOVA followed by post-hoc Tukey HSD
- **Expected**: Significant difference (p < 0.05) between middle and start/end

#### Experiment 2
- **Correlation Analysis**: Pearson correlation between token count and both accuracy/latency
- **Expected**: Strong negative correlation with accuracy (r < -0.8), strong positive with latency (r > 0.8)

#### Experiment 3
- **Paired t-test**: Compare RAG vs Full Context on matched queries
- **Expected**: Significant improvement in both accuracy and latency (p < 0.01)

#### Experiment 4
- **Repeated Measures ANOVA**: Compare strategies across conversation steps
- **Expected**: SELECT and WRITE significantly better than baseline (p < 0.05)

---

## Success Criteria

### Experiment 1
✅ Middle position accuracy < 0.70  
✅ Start/End position accuracy > 0.80  
✅ Statistical significance (p < 0.05)

### Experiment 2
✅ Clear accuracy degradation trend (R² > 0.90)  
✅ Latency increases with context size  
✅ Token growth is linear

### Experiment 3
✅ RAG accuracy > Full Context accuracy  
✅ RAG latency < 50% of Full Context latency  
✅ RAG uses < 40% of Full Context tokens

### Experiment 4
✅ SELECT or WRITE maintains accuracy > 0.75 throughout  
✅ Baseline accuracy drops below 0.65  
✅ Advanced strategies use < 60% of baseline tokens

---

## Integration with Real LLMs

### Ollama Integration Example

```python
def ollama_query(context: str, query: str, simulate: bool = False) -> str:
    if simulate:
        # Use existing simulation logic
        return simulate_response(context, query)
    
    # Real Ollama integration
    import requests
    
    response = requests.post('http://localhost:11434/api/generate', json={
        'model': 'llama2',
        'prompt': f"Context: {context}\n\nQuestion: {query}\n\nAnswer:",
        'stream': False
    })
    
    return response.json()['response']
```

### LangChain Integration Example

```python
from langchain.llms import Ollama
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma

def setup_rag_system(documents: List[str]):
    # Initialize embeddings
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    # Create vector store
    vectorstore = Chroma.from_texts(
        texts=documents,
        embedding=embeddings,
        collection_name="context_lab"
    )
    
    return vectorstore

def rag_query(vectorstore, query: str, llm):
    # Retrieve relevant documents
    docs = vectorstore.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Query LLM
    response = llm(f"Context: {context}\n\nQuestion: {query}")
    return response
```

---

## Deliverables Checklist

- [ ] `context_lab.py` - Complete implementation
- [ ] `report_plan.md` - This document
- [ ] `requirements.txt` - Python dependencies
- [ ] `README.md` - Usage instructions
- [ ] Results JSON file from experiment runs
- [ ] All visualization plots (8-10 figures)
- [ ] Final report (PDF/Markdown)
- [ ] Statistical analysis notebook (optional)

---

## Timeline Estimate

| Phase | Duration | Tasks |
|-------|----------|-------|
| Setup | 1 hour | Install dependencies, verify environment |
| Experiment 1 | 2 hours | Implementation + testing + analysis |
| Experiment 2 | 2 hours | Implementation + testing + analysis |
| Experiment 3 | 3 hours | RAG setup + implementation + analysis |
| Experiment 4 | 3 hours | Strategy implementation + comparison |
| Visualization | 2 hours | Generate all plots and charts |
| Report Writing | 4 hours | Compile findings, write analysis |
| **Total** | **17 hours** | Complete lab with report |

---

## Notes for Real Implementation

1. **Mock vs Real LLM**: Current implementation uses simulation. For real results:
   - Install Ollama: `curl https://ollama.ai/install.sh | sh`
   - Pull model: `ollama pull llama2`
   - Set `simulate=False` in query functions

2. **Vector Store**: For production RAG:
   - Use persistent ChromaDB
   - Consider Pinecone or Weaviate for scale
   - Implement proper error handling

3. **Embeddings**: 
   - nomic-embed-text (free, local)
   - OpenAI embeddings (paid, high quality)
   - sentence-transformers (free, good quality)

4. **Evaluation**:
   - Current evaluation is simplified
   - Consider using BLEU, ROUGE, or BERTScore
   - Human evaluation for qualitative assessment

5. **Performance**:
   - Add caching for repeated queries
   - Implement batching for multiple documents
   - Use async operations for parallel queries

---

## Conclusion

This report plan provides a comprehensive framework for analyzing context window impacts in LLM systems. The four experiments systematically explore different aspects of context management, from basic phenomena (Lost in the Middle) to advanced strategies (RAG, compression, external memory).

The expected results demonstrate that careful context engineering can significantly improve both accuracy and efficiency in LLM applications. The findings should guide practical decisions about context window management in production systems.

