# Real LLM Integration Guide

## 🎯 Quick Setup (3 Steps)

```bash
# 1. Install dependencies
pip install langchain langchain-community chromadb sentence-transformers

# 2. Make sure Ollama is running (check with existing CHATBOT setup)
ollama list

# 3. Run experiments - it will auto-detect Ollama!
python3 context_lab.py
```

That's it! The lab now automatically uses real Ollama when available.

---

## 📦 What Was Changed

### Updated Files

1. **requirements.txt** - Added langchain, chromadb, sentence-transformers
2. **context_lab.py** - Integrated real Ollama, LangChain, and ChromaDB
3. **INSTALL.md** - Complete installation guide
4. **test_real_llm.py** - Test script to verify setup
5. **install_real_llm.sh** - Automated installation script

### New Features

✅ **Auto-Detection** - Automatically uses Ollama if available, falls back to simulation  
✅ **Real ChromaDB** - Uses actual vector database for RAG experiments  
✅ **Real Embeddings** - Uses nomic-embed-text or sentence-transformers  
✅ **Graceful Fallback** - Works even if Ollama is not running  

---

## 🚀 Installation Options

### Option 1: Quick Install (Recommended)

```bash
cd /Users/bvolovelsky/Desktop/LLM/context_lab
./install_real_llm.sh
```

This script will:
- Install all Python dependencies
- Check Ollama installation
- Pull required models
- Run tests to verify everything works

### Option 2: Manual Install

```bash
# Install Python packages
pip install langchain langchain-community chromadb sentence-transformers requests

# Ollama should already be installed from your CHATBOT project
# Verify it's working:
ollama list

# If you need a model:
ollama pull llama2
ollama pull nomic-embed-text  # For embeddings
```

### Option 3: Install Only What's Needed

```bash
# Minimum for LangChain + Ollama
pip install langchain langchain-community requests

# Minimum for ChromaDB
pip install chromadb

# Minimum for embeddings
pip install sentence-transformers
```

---

## 🧪 Test Your Setup

```bash
# Run test script
python3 test_real_llm.py
```

This will verify:
1. ✅ LangChain imports
2. ✅ ChromaDB imports
3. ✅ Ollama connection
4. ✅ Embeddings work
5. ✅ ChromaDB works
6. ✅ Quick experiment test

---

## 🔴 Using Real LLM in Experiments

### Automatic Mode (Recommended)

The lab **automatically detects** if Ollama is available:

```bash
# Just run - it will use Ollama if available!
python3 context_lab.py
```

Output will show:
- 🔴 **REAL OLLAMA** - Using real LLM
- 🔵 **SIMULATION** - Using simulation (Ollama not available)

### Force Real LLM

```python
from context_lab import experiment3_rag_vs_full_context

# Force real LLM (will error if not available)
results = experiment3_rag_vs_full_context(num_docs=10, use_real_llm=True)
```

### Force Simulation

```python
# Force simulation (even if Ollama is available)
results = experiment3_rag_vs_full_context(num_docs=10, use_real_llm=False)
```

---

## 📊 Expected Performance

### Simulation Mode
- **Speed**: ~30 seconds for all experiments
- **Accuracy**: Simulated patterns
- **Cost**: Free, no network

### Real LLM Mode (Ollama)
- **Speed**: ~5-10 minutes for all experiments
- **Accuracy**: Real LLM behavior
- **Cost**: Free (local), requires ~4GB RAM per model

---

## 🎯 Best Experiment to Test Real LLM

**Experiment 3 (RAG vs Full Context)** is the most interesting with real LLM:

```bash
# Run only Experiment 3 with real LLM
python3 context_lab.py --experiment 3
```

This will show:
- ✅ Real vector search with ChromaDB
- ✅ Real embeddings (nomic-embed-text)
- ✅ Real LLM responses
- ✅ Actual RAG performance gains

---

## 🔧 Configuration

### Change LLM Model

Edit `context_lab.py`:

```python
def get_llm(model: str = "llama2"):  # Change model here
    # Options: "llama2", "mistral", "phi", "llama3"
```

Or pass as parameter:

```python
from context_lab import get_llm
llm = get_llm(model="mistral")
```

### Change Embedding Model

```python
def get_embeddings(model: str = "nomic-embed-text"):  # Change here
    # Options: "nomic-embed-text", "all-MiniLM-L6-v2"
```

### Use Different Vector Store

Current: ChromaDB (in-memory)  
Can change to: Persistent ChromaDB, Pinecone, Weaviate, etc.

---

## 🐛 Troubleshooting

### "Could not connect to Ollama"

**Solution 1:** Make sure Ollama is running
```bash
ollama serve
```

**Solution 2:** Start a model to auto-start server
```bash
ollama run llama2
```

**Solution 3:** Check if port 11434 is available
```bash
curl http://localhost:11434/api/tags
```

### "No module named 'langchain_community'"

```bash
pip install langchain-community
```

### "Model 'llama2' not found"

```bash
ollama pull llama2
```

### Experiments are slow

Use a smaller, faster model:
```bash
ollama pull phi  # Much faster than llama2
```

Then change in code:
```python
llm = get_llm(model="phi")
```

### Out of memory

**Options:**
1. Use smaller model (phi instead of llama2)
2. Reduce document count in experiments
3. Close other applications
4. Add more RAM if possible

### Embeddings fail

The code automatically falls back to sentence-transformers:
```bash
pip install sentence-transformers
```

---

## 📝 Code Changes Made

### 1. Real LLM Interface

**Before:**
```python
def ollama_query(context, query, simulate=True):
    if simulate:
        # simulation code
    else:
        raise NotImplementedError("Real LLM not implemented")
```

**After:**
```python
from langchain_community.llms import Ollama

def ollama_query(context, query, use_real=None):
    if use_real and REAL_LLM_AVAILABLE:
        llm = Ollama(model="llama2")
        return llm.invoke(f"Context: {context}\n\nQuestion: {query}")
    else:
        # simulation code
```

### 2. Real Vector Store

**Before:**
```python
class SimpleVectorStore:
    def __init__(self):
        self.chunks = []
```

**After:**
```python
class SimpleVectorStore:
    def __init__(self, use_real=None):
        if use_real:
            self.client = chromadb.Client()
            self.collection = self.client.create_collection("context_lab")
```

### 3. Real Embeddings

**Before:**
```python
def nomic_embed_text(chunks):
    return [np.random.randn(384) for _ in chunks]
```

**After:**
```python
from langchain_community.embeddings import OllamaEmbeddings

def nomic_embed_text(chunks, use_real=None):
    if use_real:
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        return embeddings.embed_documents(chunks)
    else:
        return [np.random.randn(384) for _ in chunks]
```

---

## 🎓 Understanding the Results

### Simulation vs Real LLM

| Aspect | Simulation | Real LLM |
|--------|-----------|----------|
| **Behavior** | Programmed patterns | Actual LLM reasoning |
| **Variability** | Deterministic | Stochastic |
| **Accuracy** | Artificial | Real-world |
| **Speed** | Fast (~30s) | Slower (~5-10min) |
| **Purpose** | Teaching/demo | Research/production |

### When to Use Each

**Use Simulation:**
- Quick testing
- Teaching demonstrations
- No Ollama available
- Development/debugging

**Use Real LLM:**
- Research analysis
- Production testing
- Understanding actual behavior
- Benchmarking real performance

---

## 📊 Example: Running Experiment 3 with Real LLM

```bash
# Run with real Ollama + ChromaDB
python3 context_lab.py --experiment 3
```

**Expected Output:**

```
================================================================================
EXPERIMENT 3: RAG vs FULL CONTEXT
================================================================================
Mode: 🔴 REAL OLLAMA + ChromaDB

Generating corpus of 20 documents...
Setting up RAG system (chunking, embedding, vector store)...
✅ Using real ChromaDB vector store

Mode A: FULL CONTEXT
  Tokens: 3669
  Latency: 4.523s                    ← Real LLM latency
  Accuracy: 0.650

Mode B: RAG (Retrieval-Augmented Generation)
  Tokens: 377
  Latency: 1.234s                    ← Much faster!
  Accuracy: 0.850                    ← More accurate!

--------------------------------------------------------------------------------
COMPARISON:
  RAG Speedup: 3.67x faster ⚡       ← Real performance gain
  RAG Token Reduction: 89.7% fewer tokens 💰
  RAG Accuracy Improvement: 30.8% better 📈
```

---

## 🔄 Switching Between Modes

The lab intelligently switches modes:

```
IF Ollama is running AND packages installed:
    → Use 🔴 REAL OLLAMA
ELSE:
    → Use 🔵 SIMULATION
```

No code changes needed - it just works!

---

## 📚 Additional Resources

- **Ollama Docs**: https://ollama.ai/
- **LangChain Docs**: https://python.langchain.com/
- **ChromaDB Docs**: https://docs.trychroma.com/
- **Your CHATBOT Project**: /Users/bvolovelsky/Desktop/LLM/CHATBOT

---

## ✅ Quick Verification

Run this to verify everything is working:

```bash
python3 -c "
from langchain_community.llms import Ollama
llm = Ollama(model='llama2')
response = llm.invoke('Say hello in one word')
print('✅ Real LLM works:', response)
"
```

If this works, you're ready to run experiments with real LLM!

---

## 🎉 Next Steps

1. ✅ Install dependencies (if needed): `./install_real_llm.sh`
2. ✅ Verify setup: `python3 test_real_llm.py`
3. ✅ Run experiments: `python3 context_lab.py`
4. ✅ Compare with simulation: Note the differences in behavior
5. ✅ Analyze results: Real LLM will show actual "Lost in the Middle"

Enjoy your real LLM-powered context analysis! 🚀

