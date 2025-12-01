"""
Generate updated plots for Experiment 3 with improved RAG results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Updated results from improved RAG (average of 3 trials)
exp3_results = {
    "full_context": {
        "tokens": 5966,
        "latency": 26.8,
        "accuracy": 0.333
    },
    "rag": {
        "tokens": 388,
        "latency": 15.9,
        "accuracy": 0.722
    }
}

# Plot 1: RAG vs Full Context Comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Accuracy comparison
modes = ['Full Context', 'RAG']
accuracies = [exp3_results['full_context']['accuracy'], exp3_results['rag']['accuracy']]
colors = ['#e74c3c', '#2ecc71']

axes[0].bar(modes, accuracies, color=colors, alpha=0.8)
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].set_title('Accuracy: RAG vs Full Context', fontsize=14, fontweight='bold')
axes[0].set_ylim([0, 1.0])
axes[0].grid(axis='y', alpha=0.3)
for i, v in enumerate(accuracies):
    axes[0].text(i, v + 0.02, f'{v:.1%}', ha='center', fontweight='bold')

# Latency comparison
latencies = [exp3_results['full_context']['latency'], exp3_results['rag']['latency']]

axes[1].bar(modes, latencies, color=colors, alpha=0.8)
axes[1].set_ylabel('Latency (seconds)', fontsize=12)
axes[1].set_title('Latency: RAG vs Full Context', fontsize=14, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)
for i, v in enumerate(latencies):
    axes[1].text(i, v + 0.5, f'{v:.1f}s', ha='center', fontweight='bold')

# Token usage comparison
tokens = [exp3_results['full_context']['tokens'], exp3_results['rag']['tokens']]

axes[2].bar(modes, tokens, color=colors, alpha=0.8)
axes[2].set_ylabel('Tokens Used', fontsize=12)
axes[2].set_title('Tokens: RAG vs Full Context', fontsize=14, fontweight='bold')
axes[2].grid(axis='y', alpha=0.3)
for i, v in enumerate(tokens):
    axes[2].text(i, v + 100, f'{v:,}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('plots/exp3_rag_vs_full.png', dpi=300, bbox_inches='tight')
print("✅ Saved: plots/exp3_rag_vs_full.png")
plt.close()

# Plot 2: RAG Improvements Detailed View
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Speedup
speedup = exp3_results['full_context']['latency'] / exp3_results['rag']['latency']
axes[0, 0].barh(['Speedup'], [speedup], color='#3498db', alpha=0.8)
axes[0, 0].set_xlabel('Speedup Factor', fontsize=12)
axes[0, 0].set_title('RAG Speedup', fontsize=14, fontweight='bold')
axes[0, 0].text(speedup + 0.05, 0, f'{speedup:.2f}x faster', va='center', fontweight='bold')
axes[0, 0].grid(axis='x', alpha=0.3)

# Token Reduction
token_reduction_pct = (1 - exp3_results['rag']['tokens'] / exp3_results['full_context']['tokens']) * 100
axes[0, 1].barh(['Token Reduction'], [token_reduction_pct], color='#9b59b6', alpha=0.8)
axes[0, 1].set_xlabel('Reduction (%)', fontsize=12)
axes[0, 1].set_title('RAG Token Reduction', fontsize=14, fontweight='bold')
axes[0, 1].set_xlim([0, 100])
axes[0, 1].text(token_reduction_pct + 1, 0, f'{token_reduction_pct:.1f}%', va='center', fontweight='bold')
axes[0, 1].grid(axis='x', alpha=0.3)

# Accuracy Improvement
accuracy_improvement_pct = ((exp3_results['rag']['accuracy'] - exp3_results['full_context']['accuracy']) / 
                           exp3_results['full_context']['accuracy'] * 100)
axes[1, 0].barh(['Accuracy Improvement'], [accuracy_improvement_pct], color='#e67e22', alpha=0.8)
axes[1, 0].set_xlabel('Improvement (%)', fontsize=12)
axes[1, 0].set_title('RAG Accuracy Improvement', fontsize=14, fontweight='bold')
axes[1, 0].text(accuracy_improvement_pct + 5, 0, f'+{accuracy_improvement_pct:.1f}%', va='center', fontweight='bold')
axes[1, 0].grid(axis='x', alpha=0.3)

# Overall Efficiency Score (composite metric)
# Efficiency = (accuracy_score * speed_score * token_efficiency_score) ^ (1/3)
accuracy_score = exp3_results['rag']['accuracy'] / max(exp3_results['full_context']['accuracy'], 0.01)
speed_score = exp3_results['full_context']['latency'] / exp3_results['rag']['latency']
token_score = exp3_results['full_context']['tokens'] / exp3_results['rag']['tokens']
efficiency = (accuracy_score * speed_score * token_score) ** (1/3)

metrics = ['Accuracy\nRatio', 'Speed\nRatio', 'Token\nRatio', 'Overall\nEfficiency']
values = [accuracy_score, speed_score, token_score, efficiency]
colors_metric = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']

axes[1, 1].bar(metrics, values, color=colors_metric, alpha=0.8)
axes[1, 1].set_ylabel('Ratio (higher is better)', fontsize=12)
axes[1, 1].set_title('RAG Efficiency Metrics', fontsize=14, fontweight='bold')
axes[1, 1].grid(axis='y', alpha=0.3)
axes[1, 1].axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Break-even')
axes[1, 1].legend()
for i, v in enumerate(values):
    axes[1, 1].text(i, v + 0.1, f'{v:.2f}x', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('plots/exp3_rag_improvements.png', dpi=300, bbox_inches='tight')
print("✅ Saved: plots/exp3_rag_improvements.png")
plt.close()

# Plot 3: Side-by-side comparison with annotations
fig, ax = plt.subplots(figsize=(12, 8))

categories = ['Accuracy\n(higher better)', 'Latency\n(lower better)', 'Tokens\n(lower better)']
full_values_normalized = [
    exp3_results['full_context']['accuracy'] * 100,  # Convert to percentage
    exp3_results['full_context']['latency'] / 10,     # Normalize
    exp3_results['full_context']['tokens'] / 100      # Normalize
]
rag_values_normalized = [
    exp3_results['rag']['accuracy'] * 100,
    exp3_results['rag']['latency'] / 10,
    exp3_results['rag']['tokens'] / 100
]

x = np.arange(len(categories))
width = 0.35

bars1 = ax.bar(x - width/2, full_values_normalized, width, label='Full Context', 
               color='#e74c3c', alpha=0.8)
bars2 = ax.bar(x + width/2, rag_values_normalized, width, label='RAG', 
               color='#2ecc71', alpha=0.8)

ax.set_ylabel('Normalized Values', fontsize=12)
ax.set_title('Experiment 3: RAG vs Full Context - Comprehensive Comparison\n' + 
             '(20 Hebrew Documents: Medical, Technology, Legal)', 
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend(fontsize=12)
ax.grid(axis='y', alpha=0.3)

# Add value labels
for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
    if i == 0:  # Accuracy
        ax.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 1, 
                f'{exp3_results["full_context"]["accuracy"]:.1%}', 
                ha='center', fontweight='bold')
        ax.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 1, 
                f'{exp3_results["rag"]["accuracy"]:.1%}', 
                ha='center', fontweight='bold')
    elif i == 1:  # Latency
        ax.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.2, 
                f'{exp3_results["full_context"]["latency"]:.1f}s', 
                ha='center', fontweight='bold')
        ax.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.2, 
                f'{exp3_results["rag"]["latency"]:.1f}s', 
                ha='center', fontweight='bold')
    else:  # Tokens
        ax.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 2, 
                f'{exp3_results["full_context"]["tokens"]:,}', 
                ha='center', fontweight='bold')
        ax.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 2, 
                f'{exp3_results["rag"]["tokens"]:,}', 
                ha='center', fontweight='bold')

# Add summary text box
summary_text = f"""RAG Performance Summary:
• Accuracy: +{accuracy_improvement_pct:.0f}% improvement
• Speed: {speedup:.1f}x faster
• Tokens: {token_reduction_pct:.0f}% reduction
• Query: "מהן תופעות הלוואי של אדוויל?"
  (What are the side effects of Advil?)"""

ax.text(0.98, 0.97, summary_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('plots/exp3_comprehensive_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Saved: plots/exp3_comprehensive_comparison.png")
plt.close()

print("\n🎉 All Experiment 3 plots updated successfully!")
print(f"\nResults Summary:")
print(f"  Full Context: {exp3_results['full_context']['accuracy']:.1%} accuracy, {exp3_results['full_context']['latency']:.1f}s, {exp3_results['full_context']['tokens']:,} tokens")
print(f"  RAG:          {exp3_results['rag']['accuracy']:.1%} accuracy, {exp3_results['rag']['latency']:.1f}s, {exp3_results['rag']['tokens']:,} tokens")
print(f"\nImprovements:")
print(f"  Accuracy: +{accuracy_improvement_pct:.0f}%")
print(f"  Speed: {speedup:.1f}x faster")
print(f"  Tokens: {token_reduction_pct:.0f}% reduction")

