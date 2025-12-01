#!/usr/bin/env python3
"""
Diagnostic script for Experiment 3 to see what's happening with RAG
"""

from context_lab import (
    load_documents,
    embed_critical_fact,
    split_documents,
    SimpleVectorStore,
    nomic_embed_text,
    ollama_query,
    evaluate_accuracy,
    concatenate_documents
)

print("="*80)
print("DIAGNOSTIC: Experiment 3 - RAG vs Full Context")
print("="*80)

# Generate documents
print("\n1. Generating corpus...")
documents = load_documents(5, words_per_doc=200)  # Use fewer docs for testing

# Add the target fact
target_fact = "Drug X causes nausea, dizziness, and headaches in 15% of patients."
print(f"\n📝 Target Fact: '{target_fact}'")

documents[2] = embed_critical_fact(documents[2], target_fact, 'middle')

# Show the document with the fact
print(f"\n📄 Document with fact (snippet):")
print(documents[2][:600])
print("...")

query = "What are the side effects of drug X?"
print(f"\n❓ Query: '{query}'")

# Test RAG retrieval
print("\n2. Setting up RAG system...")
chunks = split_documents(documents, chunk_size=500)
print(f"   Created {len(chunks)} chunks")

vector_store = SimpleVectorStore(use_real=True)
embeddings = nomic_embed_text(chunks, use_real=True)
vector_store.add(chunks, embeddings)

print("\n3. RAG: Retrieving relevant chunks...")
relevant_chunks = vector_store.similarity_search(query, k=3)

print(f"\n📥 Retrieved {len(relevant_chunks)} chunks:")
for i, chunk in enumerate(relevant_chunks, 1):
    print(f"\n--- Chunk {i} ---")
    print(chunk[:300] + ("..." if len(chunk) > 300 else ""))
    print(f"Contains target fact: {target_fact in chunk}")
    print(f"Contains 'Drug X': {'Drug X' in chunk}")

# Query with RAG context
print("\n4. Querying LLM with RAG context...")
rag_context = "\n\n".join(relevant_chunks)
rag_response = ollama_query(rag_context, query, use_real=True)

print(f"\n💬 RAG Response:")
print("-" * 80)
print(rag_response)
print("-" * 80)

rag_accuracy = evaluate_accuracy(rag_response, target_fact)
print(f"\n📊 RAG Accuracy: {rag_accuracy}")
print(f"   Looking for: '{target_fact}'")
print(f"   Contains 'nausea': {'nausea' in rag_response.lower()}")
print(f"   Contains 'dizziness': {'dizziness' in rag_response.lower()}")
print(f"   Contains 'headaches': {'headaches' in rag_response.lower()}")

# Test Full Context for comparison
print("\n" + "="*80)
print("5. Testing FULL CONTEXT for comparison...")
full_context = concatenate_documents(documents)
print(f"   Full context: {len(full_context)} chars")

full_response = ollama_query(full_context, query, use_real=True)

print(f"\n💬 Full Context Response:")
print("-" * 80)
print(full_response)
print("-" * 80)

full_accuracy = evaluate_accuracy(full_response, target_fact)
print(f"\n📊 Full Context Accuracy: {full_accuracy}")

print("\n" + "="*80)
print("DIAGNOSIS COMPLETE")
print("="*80)
print("\n🔍 Key Issue:")
print("   The fact about 'Drug X' is embedded in Lorem Ipsum text")
print("   The LLM and vector search might not recognize it as relevant")
print("   because it doesn't naturally fit with the filler text context.")

