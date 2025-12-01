#!/usr/bin/env python3
"""
Diagnostic for Hebrew Experiment 3
"""

from context_lab import (
    load_documents,
    generate_filler_text,
    SimpleVectorStore,
    nomic_embed_text,
    split_documents,
    ollama_query,
    evaluate_accuracy
)
import random

print("="*80)
print("DIAGNOSTIC: Hebrew Multi-Domain Experiment")
print("="*80)

# Generate Hebrew documents
print("\n1. Generating Hebrew documents...")
documents = load_documents(10, words_per_doc=200, diverse_domains=True)

print(f"\n📄 Sample Medical Document (first 300 chars):")
medical_docs = [doc for i, doc in enumerate(documents) if i % 3 == 0]
print(medical_docs[0][:300] if medical_docs else "No medical docs")

print(f"\n📄 Sample Tech Document (first 300 chars):")
tech_docs = [doc for i, doc in enumerate(documents) if i % 3 == 1]
print(tech_docs[0][:300] if tech_docs else "No tech docs")

print(f"\n📄 Sample Legal Document (first 300 chars):")
legal_docs = [doc for i, doc in enumerate(documents) if i % 3 == 2]
print(legal_docs[0][:300] if legal_docs else "No legal docs")

# Create fact document
print("\n2. Creating medical document with Advil side effects...")
target_fact = "אדוויל (איבופרופן) עלול לגרום לכאבי בטן, בחילות, צרבת וסחרחורת בכ-10% מהמטופלים."
print(f"Fact: {target_fact}")

medical_doc = generate_filler_text(100, domain="medical_hebrew")
medical_context = f"{medical_doc} {target_fact} {generate_filler_text(100, domain='medical_hebrew')}"

print(f"\n📝 Document with fact (snippet):")
print(medical_context[:500])

# Replace one document
fact_doc_index = 3  # A medical domain index
documents[fact_doc_index] = medical_context

# Test query
query = "מהן תופעות הלוואי של אדוויל?"
print(f"\n3. Query: {query}")

# Setup RAG
print("\n4. Setting up RAG...")
chunks = split_documents(documents, chunk_size=500)
print(f"   Created {len(chunks)} chunks")

vector_store = SimpleVectorStore(use_real=True)
embeddings = nomic_embed_text(chunks, use_real=True)
vector_store.add(chunks, embeddings)

print("\n5. Retrieving relevant chunks...")
relevant_chunks = vector_store.similarity_search(query, k=3)

print(f"\n📥 Retrieved {len(relevant_chunks)} chunks:")
for i, chunk in enumerate(relevant_chunks, 1):
    print(f"\n--- Chunk {i} (first 300 chars) ---")
    print(chunk[:300])
    print(f"Contains 'אדוויל': {'אדוויל' in chunk}")
    print(f"Contains 'בחילות': {'בחילות' in chunk}")
    print(f"Contains target fact: {target_fact in chunk}")

# Query LLM
print("\n6. Querying LLM with RAG context...")
rag_context = "\n\n".join(relevant_chunks)
rag_response = ollama_query(rag_context, query, use_real=True)

print(f"\n💬 LLM Response:")
print("-" * 80)
print(rag_response)
print("-" * 80)

# Evaluate
accuracy = evaluate_accuracy(rag_response, target_fact)
print(f"\n📊 Accuracy: {accuracy}")
print(f"\nChecking for Hebrew medical terms in response:")
terms = ['בחילות', 'כאבי בטן', 'צרבת', 'סחרחורת', 'איבופרופן', 'אדוויל']
for term in terms:
    in_fact = term in target_fact
    in_response = term in rag_response
    print(f"  '{term}': in fact={in_fact}, in response={in_response}")

print("\n" + "="*80)
print("DIAGNOSIS COMPLETE")
print("="*80)

