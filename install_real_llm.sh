#!/bin/bash

# Installation script for real LLM integration

echo "================================================================================"
echo "  Context Lab - Real LLM Integration Setup"
echo "================================================================================"
echo ""

# Check if we're in the right directory
if [ ! -f "context_lab.py" ]; then
    echo "❌ Error: context_lab.py not found."
    echo "   Please run this script from the context_lab directory."
    exit 1
fi

echo "1. Installing Python dependencies..."
echo "--------------------------------------------------------------------------------"

# Install core dependencies
pip install numpy pandas matplotlib seaborn

# Install LangChain and ChromaDB
pip install langchain langchain-community chromadb sentence-transformers requests

echo ""
echo "✅ Python dependencies installed!"
echo ""

# Check if Ollama is installed
echo "2. Checking Ollama installation..."
echo "--------------------------------------------------------------------------------"

if command -v ollama &> /dev/null; then
    echo "✅ Ollama is installed: $(ollama --version 2>&1 | head -1)"
else
    echo "⚠️  Ollama not found!"
    echo ""
    echo "To install Ollama:"
    echo "  macOS/Linux:"
    echo "    curl https://ollama.ai/install.sh | sh"
    echo ""
    echo "  Or download from: https://ollama.ai/download"
    echo ""
    read -p "Do you want to install Ollama now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        curl https://ollama.ai/install.sh | sh
    else
        echo "Skipping Ollama installation. Install it manually later."
    fi
fi

echo ""
echo "3. Pulling Ollama model..."
echo "--------------------------------------------------------------------------------"

if command -v ollama &> /dev/null; then
    # Check if llama2 is already pulled
    if ollama list | grep -q llama2; then
        echo "✅ llama2 model already installed"
    else
        echo "Pulling llama2 model (this may take a few minutes)..."
        ollama pull llama2
        echo "✅ llama2 model installed"
    fi
    
    # Optionally pull nomic-embed-text for embeddings
    echo ""
    echo "Pulling nomic-embed-text for embeddings..."
    if ollama list | grep -q nomic-embed-text; then
        echo "✅ nomic-embed-text already installed"
    else
        ollama pull nomic-embed-text || echo "⚠️  Could not pull nomic-embed-text (optional)"
    fi
else
    echo "⚠️  Ollama not available. Install it manually."
fi

echo ""
echo "4. Testing installation..."
echo "--------------------------------------------------------------------------------"

python3 test_real_llm.py

if [ $? -eq 0 ]; then
    echo ""
    echo "================================================================================"
    echo "🎉 INSTALLATION COMPLETE!"
    echo "================================================================================"
    echo ""
    echo "You can now run experiments with real LLM:"
    echo "  python3 context_lab.py"
    echo ""
    echo "Or run specific experiment:"
    echo "  python3 context_lab.py --experiment 3  # RAG experiment"
    echo ""
    echo "The lab will automatically use real Ollama if available,"
    echo "or fall back to simulation if Ollama is not running."
    echo "================================================================================"
else
    echo ""
    echo "================================================================================"
    echo "⚠️  INSTALLATION COMPLETED WITH WARNINGS"
    echo "================================================================================"
    echo ""
    echo "Some tests failed, but you can still run the lab."
    echo "It will fall back to simulation mode if real LLM is not available."
    echo ""
    echo "To fix issues:"
    echo "  1. Make sure Ollama is running: ollama serve"
    echo "  2. Or start a model: ollama run llama2"
    echo "  3. Re-run tests: python3 test_real_llm.py"
    echo "================================================================================"
fi

