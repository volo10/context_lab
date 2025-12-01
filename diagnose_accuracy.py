#!/usr/bin/env python3
"""
Diagnostic script to see what the LLM is actually responding with
and understand why accuracy is low.
"""

import random
from context_lab import (
    generate_filler_text,
    embed_critical_fact,
    ollama_query,
    evaluate_accuracy
)

print("="*80)
print("DIAGNOSTIC: Checking LLM Responses")
print("="*80)

# Generate a test document
base_doc = generate_filler_text(200)
fact = "The secret code is ALPHA0BETA1234"
doc_with_fact = embed_critical_fact(base_doc, fact, 'start')

print(f"\n📝 Generated Fact: '{fact}'")
print(f"\n📄 Document snippet (first 500 chars):")
print(doc_with_fact[:500])
print("...")

# Query the LLM
query = "What is the critical fact mentioned in the document?"
print(f"\n❓ Query: '{query}'")

print("\n🤖 Querying real LLM...")
response = ollama_query(doc_with_fact, query, use_real=True)

print(f"\n💬 LLM Response:")
print("-" * 80)
print(response)
print("-" * 80)

# Evaluate
accuracy = evaluate_accuracy(response, fact)
print(f"\n📊 Accuracy: {accuracy}")
print(f"   (Looking for: '{fact}')")
print(f"   (Found in response: {fact.lower() in response.lower()})")

# Check what's actually in the response
print(f"\n🔍 Response Analysis:")
print(f"   - Response length: {len(response)} chars")
print(f"   - Contains 'ALPHA': {'ALPHA' in response.upper()}")
print(f"   - Contains 'BETA': {'BETA' in response.upper()}")
print(f"   - Contains 'secret code': {'secret code' in response.lower()}")
print(f"   - Contains 'critical fact': {'critical fact' in response.lower()}")
print(f"   - Contains 'CRITICAL_FACT:': {'CRITICAL_FACT:' in response}")

print("\n" + "="*80)
print("DIAGNOSIS COMPLETE")
print("="*80)

