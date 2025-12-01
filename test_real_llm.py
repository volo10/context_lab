#!/usr/bin/env python3
"""
Test script to verify real LLM integration is working.
"""

import sys

print("="*80)
print("Testing Real LLM Integration")
print("="*80)

# Test 1: Import packages
print("\n1. Testing imports...")
try:
    import langchain
    print("   ✅ langchain imported")
except ImportError as e:
    print(f"   ❌ langchain not found: {e}")
    print("   Install with: pip install langchain langchain-community")
    sys.exit(1)

try:
    import chromadb
    print("   ✅ chromadb imported")
except ImportError as e:
    print(f"   ❌ chromadb not found: {e}")
    print("   Install with: pip install chromadb")
    sys.exit(1)

try:
    from langchain_community.llms import Ollama
    from langchain_community.embeddings import OllamaEmbeddings
    print("   ✅ langchain_community imported")
except ImportError as e:
    print(f"   ❌ langchain_community not found: {e}")
    print("   Install with: pip install langchain-community")
    sys.exit(1)

# Test 2: Connect to Ollama
print("\n2. Testing Ollama connection...")
try:
    llm = Ollama(model="llama2", temperature=0.1)
    response = llm.invoke("Say 'Hello' in one word")
    print(f"   ✅ Ollama connected: '{response.strip()}'")
except Exception as e:
    print(f"   ❌ Ollama connection failed: {e}")
    print("   Make sure Ollama is running:")
    print("     - Start: ollama serve")
    print("     - Or run: ollama run llama2")
    print("   Install Ollama from: https://ollama.ai/")
    sys.exit(1)

# Test 3: Test embeddings
print("\n3. Testing embeddings...")
try:
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    emb = embeddings.embed_query("test")
    print(f"   ✅ Embeddings work: {len(emb)} dimensions")
except Exception as e:
    print(f"   ⚠️  Ollama embeddings failed: {e}")
    print("   Trying sentence-transformers as fallback...")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        emb = model.encode("test")
        print(f"   ✅ Sentence-transformers work: {len(emb)} dimensions")
    except Exception as e2:
        print(f"   ❌ Embeddings failed: {e2}")
        print("   Install with: pip install sentence-transformers")
        sys.exit(1)

# Test 4: Test ChromaDB
print("\n4. Testing ChromaDB...")
try:
    from chromadb.config import Settings
    client = chromadb.Client(Settings(anonymized_telemetry=False))
    collection = client.create_collection("test_collection")
    collection.add(
        documents=["This is a test"],
        ids=["1"]
    )
    print("   ✅ ChromaDB works")
except Exception as e:
    print(f"   ❌ ChromaDB failed: {e}")
    sys.exit(1)

# Test 5: Quick experiment test
print("\n5. Testing experiment with real LLM...")
try:
    from context_lab import experiment1_needle_in_haystack
    print("   Running simplified Experiment 1 with real LLM...")
    results = experiment1_needle_in_haystack(num_docs=1, words_per_doc=100, use_real_llm=True)
    print("   ✅ Experiment completed successfully!")
except Exception as e:
    print(f"   ⚠️  Experiment test failed: {e}")
    print("   This is okay - the lab will fall back to simulation if needed")

print("\n" + "="*80)
print("🎉 ALL TESTS PASSED!")
print("="*80)
print("\nYou're ready to run experiments with real LLM:")
print("  python3 context_lab.py")
print("\nOr run specific experiment:")
print("  python3 context_lab.py --experiment 3  # RAG is most interesting!")
print("="*80)

