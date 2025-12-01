# Installation Instructions for Real LLM Integration

## Quick Installation

```bash
cd /Users/bvolovelsky/Desktop/LLM/context_lab

# Install all dependencies including LangChain and ChromaDB
pip install -r requirements.txt
```

## Step-by-Step Installation

### 1. Install Core Dependencies

```bash
pip install numpy pandas matplotlib seaborn
```

### 2. Install LangChain and ChromaDB

```bash
# Install LangChain
pip install langchain langchain-community

# Install ChromaDB (vector database)
pip install chromadb

# Install sentence transformers for embeddings
pip install sentence-transformers

# Install requests for Ollama API
pip install requests
```

### 3. Install Ollama (if not already installed)

**macOS/Linux:**
```bash
curl https://ollama.ai/install.sh | sh
```

**Or download from:** https://ollama.ai/download

### 4. Pull an Ollama Model

```bash
# Pull Llama 2 (recommended)
ollama pull llama2

# Or pull other models:
ollama pull mistral
ollama pull llama3
ollama pull phi
```

### 5. Start Ollama Server

```bash
# Start Ollama (if not already running)
ollama serve
```

Or simply run any model once to start the server:
```bash
ollama run llama2 "Hello"
```

### 6. Verify Installation

```bash
# Test Ollama
curl http://localhost:11434/api/tags

# Test Python imports
python3 -c "import langchain, chromadb; print('✅ All packages installed!')"
```

## Optional: Install Nomic Embeddings

For better embeddings (optional but recommended):

```bash
pip install nomic
```

## Troubleshooting

### "Connection refused" when running experiments

**Solution:** Make sure Ollama is running:
```bash
ollama serve
```

Or in a separate terminal:
```bash
ollama run llama2
```

### "No module named 'langchain'"

**Solution:** Install LangChain:
```bash
pip install langchain langchain-community
```

### "Could not import chromadb"

**Solution:** Install ChromaDB:
```bash
pip install chromadb
```

### Ollama taking too long

**Solution:** Use a smaller model:
```bash
ollama pull phi    # Smaller, faster model
```

### Out of memory errors

**Solution:** 
1. Use a smaller model (phi, mistral)
2. Reduce batch sizes in experiments
3. Close other applications

## Verify Everything Works

Run this test script:

```bash
cd /Users/bvolovelsky/Desktop/LLM/context_lab
python3 -c "
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
import chromadb

# Test Ollama
llm = Ollama(model='llama2')
response = llm.invoke('Say hello')
print('✅ Ollama works:', response[:50])

# Test embeddings
embeddings = OllamaEmbeddings(model='nomic-embed-text')
emb = embeddings.embed_query('test')
print('✅ Embeddings work:', len(emb), 'dimensions')

# Test ChromaDB
client = chromadb.Client()
print('✅ ChromaDB works')

print('\n🎉 All components working!')
"
```

## What Gets Installed

| Package | Purpose | Size |
|---------|---------|------|
| **langchain** | LLM framework | ~50 MB |
| **chromadb** | Vector database | ~30 MB |
| **sentence-transformers** | Embeddings | ~200 MB |
| **ollama** | Local LLM (separate) | ~4-7 GB per model |

**Total pip packages:** ~300 MB  
**Ollama models:** 4-7 GB per model

## Recommended Models

| Model | Size | Speed | Quality | Best For |
|-------|------|-------|---------|----------|
| **llama2** | 3.8 GB | Medium | Good | General use (recommended) |
| **mistral** | 4.1 GB | Fast | Very Good | Best quality |
| **phi** | 1.6 GB | Very Fast | Decent | Quick testing |
| **llama3** | 4.7 GB | Medium | Excellent | Latest model |

## Quick Start After Installation

```bash
# Run with real LLM (will take 2-5 minutes)
python3 context_lab.py

# Or run specific experiment
python3 context_lab.py --experiment 3  # RAG is most interesting!
```

## Switching Between Simulation and Real LLM

The code now **automatically detects** if Ollama is available:
- If Ollama is running → Uses real LLM
- If Ollama is not available → Falls back to simulation

To force simulation mode even with Ollama installed:
```python
# In context_lab.py, set:
USE_REAL_LLM = False
```

## Next Steps

1. ✅ Install dependencies: `pip install -r requirements.txt`
2. ✅ Install Ollama: `curl https://ollama.ai/install.sh | sh`
3. ✅ Pull model: `ollama pull llama2`
4. ✅ Run lab: `python3 context_lab.py`
5. ✅ Generate plots: `python3 visualize.py`

Enjoy your real LLM-powered context analysis lab! 🚀

