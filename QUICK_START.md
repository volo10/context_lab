# Context Window Impact Analysis Lab - Quick Start Guide

## 🚀 Installation (2 minutes)

```bash
cd /Users/bvolovelsky/Desktop/LLM/context_lab

# Install dependencies
pip install numpy pandas matplotlib seaborn
```

## 🎯 Quick Demo (30 seconds)

```bash
python3 demo.py
```

This runs a simplified version of all four experiments and shows key findings.

## 📊 Full Experiments (1-2 minutes)

```bash
python3 context_lab.py
```

This runs the complete lab and saves results to `context_lab_results.json`.

### Run Specific Experiment

```bash
# Run only Experiment 1 (Needle in Haystack)
python3 context_lab.py --experiment 1

# Run only Experiment 2 (Context Size)
python3 context_lab.py --experiment 2

# Run only Experiment 3 (RAG vs Full)
python3 context_lab.py --experiment 3

# Run only Experiment 4 (Strategies)
python3 context_lab.py --experiment 4
```

## 📈 Generate Visualizations

After running experiments:

```bash
python3 visualize.py
```

This creates publication-quality plots in the `plots/` directory.

### Custom Output Location

```bash
python3 visualize.py --results my_results.json --output my_plots/
```

## 📓 Interactive Exploration

For Jupyter notebook users:

```bash
jupyter notebook notebook_template.ipynb
```

## 🔧 What Each Experiment Does

### Experiment 1: Needle in Haystack
**Question**: Can LLMs find information in the middle of long contexts?  
**Answer**: No! They suffer from "Lost in the Middle" phenomenon.

```python
from context_lab import experiment1_needle_in_haystack
results = experiment1_needle_in_haystack(num_docs=5, words_per_doc=200)
```

**Expected Output**: Middle position accuracy < Start/End accuracy

---

### Experiment 2: Context Window Size Impact
**Question**: What happens as context grows larger?  
**Answer**: Both accuracy drops and latency increases.

```python
from context_lab import experiment2_context_size_impact
results = experiment2_context_size_impact(doc_counts=[2, 5, 10, 20, 50])
```

**Expected Output**: Negative correlation between size and accuracy

---

### Experiment 3: RAG vs Full Context
**Question**: Is RAG better than providing all documents?  
**Answer**: Yes! RAG is faster, cheaper, and more accurate.

```python
from context_lab import experiment3_rag_vs_full_context
results = experiment3_rag_vs_full_context(num_docs=20)
```

**Expected Output**: RAG wins on all metrics (3x faster, 70% fewer tokens)

---

### Experiment 4: Context Engineering Strategies
**Question**: What's the best way to manage context in conversations?  
**Answer**: SELECT (RAG) and WRITE (Memory) strategies work best.

```python
from context_lab import experiment4_context_strategies
results = experiment4_context_strategies(num_steps=10)
```

**Expected Output**: SELECT and WRITE maintain >80% accuracy throughout

---

## 📋 Common Use Cases

### 1. Educational Demo
```bash
python3 demo.py > demo_output.txt
```
Share `demo_output.txt` with students/colleagues.

### 2. Research Analysis
```bash
python3 context_lab.py
python3 visualize.py
# Results in: context_lab_results.json + plots/
```

### 3. Custom Parameters
```python
from context_lab import experiment1_needle_in_haystack

# Test with larger documents
results = experiment1_needle_in_haystack(
    num_docs=10,
    words_per_doc=500
)
```

### 4. Real LLM Integration

Modify `ollama_query()` in `context_lab.py`:

```python
def ollama_query(context: str, query: str, simulate: bool = False):
    if not simulate:
        import requests
        response = requests.post('http://localhost:11434/api/generate', json={
            'model': 'llama2',
            'prompt': f"Context: {context}\n\nQuestion: {query}",
            'stream': False
        })
        return response.json()['response']
    # ... simulation code ...
```

Then run with real LLM:
```python
# In context_lab.py, change:
response = ollama_query(context, query, simulate=False)
```

---

## 🐛 Troubleshooting

### "No module named 'numpy'"
```bash
pip install numpy pandas matplotlib seaborn
```

### "command not found: python"
Use `python3` instead of `python`:
```bash
python3 context_lab.py
```

### "Results file not found" (for visualize.py)
Run the experiments first:
```bash
python3 context_lab.py  # Generate results
python3 visualize.py    # Then visualize
```

### Ollama not responding
```bash
# Check if Ollama is running
ollama list

# Start Ollama
ollama serve

# Pull a model
ollama pull llama2
```

---

## 📦 Project Structure

```
context_lab/
├── context_lab.py           # Main implementation
├── demo.py                  # Quick demo script
├── visualize.py             # Generate all plots
├── notebook_template.ipynb  # Jupyter notebook
├── requirements.txt         # Dependencies
├── README.md               # Full documentation
├── report_plan.md          # Analysis plan
├── QUICK_START.md          # This file
└── .gitignore              # Git ignore rules
```

---

## 🎓 Understanding the Results

### Experiment 1 Output
```
SUMMARY:
  START: Accuracy=0.900 (±0.050), Latency=0.240s
  MIDDLE: Accuracy=0.550 (±0.150), Latency=0.218s  ← Lost in the Middle!
  END: Accuracy=0.880 (±0.060), Latency=0.218s
```

**Insight**: Place important information at the start or end of context.

### Experiment 2 Output
```
num_docs  tokens_used  latency  accuracy
2         697          0.237    0.951
10        3621         0.288    0.738     ← Accuracy drops!
50        18105        1.500    0.450     ← Severe degradation
```

**Insight**: Limit context size; use RAG for large corpora.

### Experiment 3 Output
```
COMPARISON:
  RAG Speedup: 3.00x faster
  RAG Token Reduction: 70.0% fewer tokens
  RAG Accuracy Improvement: 31.0%
```

**Insight**: RAG is superior for focused queries.

### Experiment 4 Output
```
Strategy                  Avg Accuracy  Final Tokens
SELECT (RAG)              0.820         600
COMPRESS (Summarization)  0.750         800
WRITE (Memory)            0.800         450  ← Most efficient!
```

**Insight**: Use WRITE for efficiency, SELECT for accuracy.

---

## 💡 Next Steps

1. **Run the demo**: `python3 demo.py`
2. **Generate full results**: `python3 context_lab.py`
3. **Create visualizations**: `python3 visualize.py`
4. **Review plots**: Open files in `plots/` directory
5. **Read full documentation**: See `README.md` and `report_plan.md`
6. **Integrate with real LLM**: Modify `ollama_query()` function
7. **Customize experiments**: Adjust parameters in function calls

---

## 📚 Further Reading

- **Implementation Details**: `context_lab.py` (fully commented)
- **Analysis Plan**: `report_plan.md` (expected results, statistical tests)
- **Full Documentation**: `README.md` (comprehensive guide)
- **Lost in the Middle Paper**: [arXiv:2307.03172](https://arxiv.org/abs/2307.03172)

---

## 🤝 Support

If you encounter issues:

1. Check this guide first
2. Review `README.md` troubleshooting section
3. Examine code comments in `context_lab.py`
4. Check that all dependencies are installed

---

## ⏱️ Time Estimates

| Task | Time |
|------|------|
| Installation | 2 min |
| Quick demo | 30 sec |
| Full experiments | 2 min |
| Generate plots | 30 sec |
| Review results | 10 min |
| Read documentation | 30 min |
| **Total for full lab** | **~15 min** |

---

## ✅ Success Checklist

- [ ] Installed dependencies
- [ ] Ran demo successfully
- [ ] Generated full results
- [ ] Created visualizations
- [ ] Reviewed plots
- [ ] Understood key findings
- [ ] (Optional) Integrated with real LLM
- [ ] (Optional) Customized experiments

---

**Ready to start? Run `python3 demo.py` now! 🚀**

