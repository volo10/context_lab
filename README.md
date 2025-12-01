# Context Window Impact Analysis Lab

A comprehensive Python implementation to simulate and analyze the impact of Context Windows in LLM interactions, demonstrating phenomena like "Lost in the Middle" and testing various context engineering strategies.

## Overview

This lab implements four experiments:

1. **Needle in Haystack**: Demonstrates "Lost in the Middle" phenomenon
2. **Context Window Size Impact**: Analyzes accuracy/latency vs context size
3. **RAG vs Full Context**: Compares Retrieval-Augmented Generation with full context
4. **Context Engineering Strategies**: Benchmarks SELECT, COMPRESS, and WRITE strategies

## Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Optional: Install Ollama for Real LLM Testing

```bash
# Install Ollama
curl https://ollama.ai/install.sh | sh

# Pull a model
ollama pull llama2
```

## Quick Start

### Run All Experiments (Simulation Mode)

```bash
python context_lab.py
```

This will:
- Run all four experiments
- Generate results JSON file
- Print summaries to console

### Run Specific Experiment

```bash
# Run only Experiment 1
python context_lab.py --experiment 1

# Run only Experiment 3
python context_lab.py --experiment 3
```

### Customize Output

```bash
# Save to custom file
python context_lab.py --output my_results.json

# Don't save results
python context_lab.py --no-save
```

## Usage Examples

### In Python Script

```python
from context_lab import (
    experiment1_needle_in_haystack,
    experiment2_context_size_impact,
    experiment3_rag_vs_full_context,
    experiment4_context_strategies,
    run_all_experiments
)

# Run individual experiments
results1 = experiment1_needle_in_haystack(num_docs=5, words_per_doc=200)
results2 = experiment2_context_size_impact(doc_counts=[2, 5, 10, 20, 50])
results3 = experiment3_rag_vs_full_context(num_docs=20)
results4 = experiment4_context_strategies(num_steps=10)

# Or run all at once
all_results = run_all_experiments(save_results=True, output_file="results.json")
```

### Generate Visualizations

```python
import matplotlib.pyplot as plt
from context_lab import experiment1_needle_in_haystack

# Run experiment
results = experiment1_needle_in_haystack()

# Plot results
positions = ['start', 'middle', 'end']
accuracies = [results['summary'][pos]['avg_accuracy'] for pos in positions]

plt.bar(positions, accuracies)
plt.ylabel('Accuracy')
plt.xlabel('Position')
plt.title('Lost in the Middle Demonstration')
plt.savefig('experiment1_results.png')
plt.show()
```

## Experiment Details

### Experiment 1: Needle in Haystack

**Goal**: Demonstrate that facts in the middle of context are harder to retrieve.

**Parameters**:
- `num_docs`: Number of test documents (default: 5)
- `words_per_doc`: Words per document (default: 200)

**Expected Outcome**: Accuracy at start/end > accuracy in middle

### Experiment 2: Context Window Size Impact

**Goal**: Show how larger contexts degrade accuracy and increase latency.

**Parameters**:
- `doc_counts`: List of document counts to test (default: [2, 5, 10, 20, 50])

**Expected Outcome**: Negative correlation with accuracy, positive with latency

### Experiment 3: RAG vs Full Context

**Goal**: Compare RAG retrieval against providing full context.

**Parameters**:
- `num_docs`: Size of document corpus (default: 20)

**Expected Outcome**: RAG should be faster and more accurate

### Experiment 4: Context Engineering Strategies

**Goal**: Benchmark three strategies for multi-step conversations.

**Parameters**:
- `num_steps`: Length of conversation (default: 10)

**Strategies**:
- **SELECT**: RAG search on history
- **COMPRESS**: Summarize when exceeding token limit
- **WRITE**: External memory for key facts

**Expected Outcome**: SELECT or WRITE maintains best accuracy

## Integration with Real LLMs

### Current Mode: Simulation

By default, the lab runs in **simulation mode** with mock LLM responses. This is useful for:
- Testing the experimental framework
- Educational demonstrations
- Quick iteration without API costs

### Switching to Real LLM

To use real LLM queries, modify the `ollama_query` function in `context_lab.py`:

```python
def ollama_query(context: str, query: str, simulate: bool = False):
    if simulate:
        # Current simulation logic
        return simulate_response(context, query)
    
    # Real Ollama integration
    import requests
    response = requests.post('http://localhost:11434/api/generate', json={
        'model': 'llama2',
        'prompt': f"Context: {context}\n\nQuestion: {query}",
        'stream': False
    })
    return response.json()['response']
```

Then call with `simulate=False`:

```python
response = ollama_query(context, query, simulate=False)
```

### LangChain Integration

For more sophisticated RAG with LangChain:

```python
from langchain.llms import Ollama
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma

# Initialize
llm = Ollama(model="llama2")
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Create vector store
vectorstore = Chroma.from_texts(
    texts=documents,
    embedding=embeddings
)

# Query
docs = vectorstore.similarity_search(query, k=3)
response = llm(f"Context: {docs}\n\nQuestion: {query}")
```

## Output Files

### Results JSON

The `context_lab_results.json` file contains:
- Detailed results from all experiments
- Summary statistics
- Expected outcomes for validation

Structure:
```json
{
  "experiment1_needle_in_haystack": {
    "detailed_results": {...},
    "summary": {...}
  },
  "experiment2_context_size": {...},
  "experiment3_rag_vs_full": {...},
  "experiment4_strategies": {...}
}
```

## Creating Visualizations

See `report_plan.md` for complete visualization code examples. Quick example:

```python
import pandas as pd
import matplotlib.pyplot as plt
import json

# Load results
with open('context_lab_results.json') as f:
    results = json.load(f)

# Experiment 2: Context size impact
exp2 = results['experiment2_context_size']['results']
df = pd.DataFrame(exp2)

# Plot
fig, ax1 = plt.subplots()
ax1.plot(df['num_docs'], df['accuracy'], 'r-o', label='Accuracy')
ax1.set_xlabel('Number of Documents')
ax1.set_ylabel('Accuracy', color='r')

ax2 = ax1.twinx()
ax2.plot(df['num_docs'], df['latency'], 'b-s', label='Latency')
ax2.set_ylabel('Latency (s)', color='b')

plt.title('Context Size Impact')
plt.savefig('context_size_impact.png', dpi=300)
plt.show()
```

## Project Structure

```
context_lab/
├── context_lab.py          # Main implementation
├── report_plan.md          # Analysis plan and expected results
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── context_lab_results.json  # Generated results (after running)
```

## Troubleshooting

### Import Errors

If you get import errors for optional dependencies:

```bash
# For basic functionality
pip install numpy pandas

# For visualization
pip install matplotlib seaborn

# For real LLM integration
pip install langchain chromadb requests
```

### Ollama Connection Issues

If Ollama queries fail:

1. Check Ollama is running: `ollama list`
2. Verify model is pulled: `ollama pull llama2`
3. Test manually: `ollama run llama2 "Hello"`

### Memory Issues

For large experiments, reduce parameters:

```python
# Instead of default
results = experiment2_context_size_impact(doc_counts=[2, 5, 10, 20, 50])

# Use smaller set
results = experiment2_context_size_impact(doc_counts=[2, 5, 10])
```

## Performance Notes

### Simulation Mode
- **Runtime**: ~30 seconds for all experiments
- **Memory**: < 100 MB
- **Cost**: Free

### Real LLM Mode
- **Runtime**: 5-15 minutes (depends on model and hardware)
- **Memory**: 4-8 GB (for model + context)
- **Cost**: Free with Ollama (local), varies with API providers

## Advanced Usage

### Custom Evaluation Function

Replace the `evaluate_accuracy` function for domain-specific evaluation:

```python
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def evaluate_accuracy(response: str, expected: str) -> float:
    emb1 = model.encode([response])
    emb2 = model.encode([expected])
    return float(cosine_similarity(emb1, emb2)[0][0])
```

### Custom Document Corpus

Replace the filler text generator with real documents:

```python
def load_real_documents(file_path: str) -> List[str]:
    with open(file_path) as f:
        return [line.strip() for line in f if line.strip()]

# Use in experiments
documents = load_real_documents('my_corpus.txt')
```

### Hebrew/Multilingual Support

For Hebrew or other languages:

```python
def generate_filler_text_hebrew(num_words: int = 200) -> str:
    hebrew_phrases = [
        "זהו טקסט לדוגמה בעברית.",
        "המערכת תומכת במספר שפות.",
        # Add more Hebrew phrases...
    ]
    # Same logic as English version
```

## Contributing

To extend this lab:

1. **Add New Experiments**: Create functions following the naming pattern `experimentN_description()`
2. **Add Strategies**: Subclass `ContextStrategy` for new context management approaches
3. **Improve Evaluation**: Enhance `evaluate_accuracy()` with better metrics
4. **Add Visualizations**: See `report_plan.md` for plotting templates

## Citation

If you use this lab in research or teaching:

```bibtex
@software{context_lab_2025,
  title={Context Window Impact Analysis Lab},
  author={Context Lab Team},
  year={2025},
  description={Python implementation for analyzing LLM context window phenomena}
}
```

## References

- [Lost in the Middle Paper](https://arxiv.org/abs/2307.03172)
- [LangChain Documentation](https://python.langchain.com/)
- [Ollama Documentation](https://ollama.ai/)
- [ChromaDB Documentation](https://docs.trychroma.com/)

## License

MIT License - Feel free to use and modify for educational and research purposes.

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review `report_plan.md` for detailed methodology
3. Examine the code comments in `context_lab.py`

## Roadmap

Future enhancements:
- [ ] Add support for more LLM providers (OpenAI, Anthropic, Cohere)
- [ ] Implement more context strategies (hierarchical memory, sliding window)
- [ ] Add multi-language support
- [ ] Create interactive Jupyter notebook version
- [ ] Add real-time visualization dashboard
- [ ] Implement A/B testing framework
- [ ] Add cost tracking and optimization

