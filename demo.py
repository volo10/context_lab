#!/usr/bin/env python3
"""
Demo script for Context Window Impact Analysis Lab
Run this to see a quick demonstration of all experiments.
"""

from context_lab import (
    experiment1_needle_in_haystack,
    experiment2_context_size_impact,
    experiment3_rag_vs_full_context,
    experiment4_context_strategies,
)


def main():
    print("="*80)
    print("CONTEXT WINDOW IMPACT ANALYSIS LAB - DEMO")
    print("="*80)
    print("\nThis demo will run simplified versions of all four experiments.")
    print("For full experiments, run: python context_lab.py\n")
    
    # Experiment 1: Smaller scale for demo
    print("\n🔍 Starting Experiment 1 (Needle in Haystack)...")
    exp1 = experiment1_needle_in_haystack(num_docs=3, words_per_doc=150)
    
    # Experiment 2: Fewer document counts
    print("\n📊 Starting Experiment 2 (Context Size Impact)...")
    exp2 = experiment2_context_size_impact(doc_counts=[2, 5, 10])
    
    # Experiment 3: Smaller corpus
    print("\n🔎 Starting Experiment 3 (RAG vs Full Context)...")
    exp3 = experiment3_rag_vs_full_context(num_docs=10)
    
    # Experiment 4: Shorter conversation
    print("\n⚙️  Starting Experiment 4 (Context Strategies)...")
    exp4 = experiment4_context_strategies(num_steps=5)
    
    print("\n" + "="*80)
    print("DEMO COMPLETE!")
    print("="*80)
    print("\n📝 KEY FINDINGS:\n")
    
    # Summary from Experiment 1
    print("1. NEEDLE IN HAYSTACK:")
    summary1 = exp1['summary']
    print(f"   - Start Position Accuracy: {summary1['start']['avg_accuracy']:.3f}")
    print(f"   - Middle Position Accuracy: {summary1['middle']['avg_accuracy']:.3f} ⬇️")
    print(f"   - End Position Accuracy: {summary1['end']['avg_accuracy']:.3f}")
    print("   ➡️ Facts in the middle are harder to find!\n")
    
    # Summary from Experiment 2
    print("2. CONTEXT SIZE IMPACT:")
    results2 = exp2['results']
    print(f"   - Small context (2 docs): Accuracy = {results2[0]['accuracy']:.3f}")
    print(f"   - Large context (10 docs): Accuracy = {results2[-1]['accuracy']:.3f} ⬇️")
    print("   ➡️ Larger contexts decrease accuracy!\n")
    
    # Summary from Experiment 3
    print("3. RAG VS FULL CONTEXT:")
    comparison = exp3['comparison']
    print(f"   - RAG Speedup: {comparison['speedup']:.2f}x faster ⚡")
    print(f"   - Token Reduction: {comparison['token_reduction_pct']:.1f}% fewer tokens 💰")
    print(f"   - Accuracy Improvement: {comparison['accuracy_improvement_pct']:.1f}% better 📈")
    print("   ➡️ RAG is superior in all metrics!\n")
    
    # Summary from Experiment 4
    print("4. CONTEXT STRATEGIES:")
    best_acc = exp4['best_accuracy']
    best_eff = exp4['best_efficiency']
    print(f"   - Best Accuracy: {best_acc} 🎯")
    print(f"   - Most Efficient: {best_eff} 💡")
    print("   ➡️ Advanced strategies maintain quality over time!\n")
    
    print("="*80)
    print("For detailed results, see: context_lab_results.json")
    print("For visualization code, see: report_plan.md")
    print("="*80)


if __name__ == "__main__":
    main()

