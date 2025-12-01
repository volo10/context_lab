#!/usr/bin/env python3
"""
Visualization script for Context Window Impact Analysis Lab
Generates all plots from experiment results.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path


# Set style
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


def load_results(filepath='context_lab_results.json'):
    """Load experiment results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def plot_experiment1(results, output_dir='plots'):
    """Plot Experiment 1: Needle in Haystack results."""
    print("Generating Experiment 1 plots...")
    
    Path(output_dir).mkdir(exist_ok=True)
    summary = results['experiment1_needle_in_haystack']['summary']
    
    positions = ['Start', 'Middle', 'End']
    accuracies = [summary['start']['avg_accuracy'], 
                  summary['middle']['avg_accuracy'], 
                  summary['end']['avg_accuracy']]
    stds = [summary['start']['std_accuracy'], 
            summary['middle']['std_accuracy'], 
            summary['end']['std_accuracy']]
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    bars = ax.bar(positions, accuracies, yerr=stds, capsize=10, 
                  color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Accuracy', fontsize=12, weight='bold')
    ax.set_xlabel('Fact Position in Context', fontsize=12, weight='bold')
    ax.set_title('Experiment 1: "Lost in the Middle" Phenomenon', 
                 fontsize=14, weight='bold', pad=20)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='Threshold')
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.3f}',
                ha='center', va='bottom', fontsize=11, weight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/exp1_needle_in_haystack.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_dir}/exp1_needle_in_haystack.png")
    plt.close()


def plot_experiment2(results, output_dir='plots'):
    """Plot Experiment 2: Context Size Impact results."""
    print("Generating Experiment 2 plots...")
    
    Path(output_dir).mkdir(exist_ok=True)
    exp2_data = results['experiment2_context_size']['results']
    df = pd.DataFrame(exp2_data)
    
    # Dual-axis plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Accuracy line
    color = 'tab:red'
    ax1.set_xlabel('Number of Documents', fontsize=12, weight='bold')
    ax1.set_ylabel('Accuracy', color=color, fontsize=12, weight='bold')
    line1 = ax1.plot(df['num_docs'], df['accuracy'], marker='o', 
                     color=color, linewidth=2.5, markersize=8, label='Accuracy')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, 1.0)
    ax1.grid(alpha=0.3)
    
    # Latency line (second y-axis)
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Latency (seconds)', color=color, fontsize=12, weight='bold')
    line2 = ax2.plot(df['num_docs'], df['latency'], marker='s', 
                     color=color, linewidth=2.5, markersize=8, label='Latency')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Experiment 2: Context Size Impact on Performance', 
              fontsize=14, weight='bold', pad=20)
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=10)
    
    fig.tight_layout()
    plt.savefig(f'{output_dir}/exp2_context_size_impact.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_dir}/exp2_context_size_impact.png")
    plt.close()
    
    # Token count vs accuracy scatter
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(df['tokens_used'], df['accuracy'], 
                        s=100, c=df['num_docs'], cmap='viridis', 
                        alpha=0.7, edgecolors='black', linewidth=1)
    ax.set_xlabel('Token Count', fontsize=12, weight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, weight='bold')
    ax.set_title('Experiment 2: Tokens vs Accuracy', 
                 fontsize=14, weight='bold', pad=20)
    ax.grid(alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Number of Documents', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/exp2_tokens_vs_accuracy.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_dir}/exp2_tokens_vs_accuracy.png")
    plt.close()


def plot_experiment3(results, output_dir='plots'):
    """Plot Experiment 3: RAG vs Full Context results."""
    print("Generating Experiment 3 plots...")
    
    Path(output_dir).mkdir(exist_ok=True)
    exp3 = results['experiment3_rag_vs_full']
    
    # Prepare data
    metrics = ['Tokens', 'Latency (s)', 'Accuracy']
    full_values = [
        exp3['full_context']['tokens'],
        exp3['full_context']['latency'],
        exp3['full_context']['accuracy']
    ]
    rag_values = [
        exp3['rag']['tokens'],
        exp3['rag']['latency'],
        exp3['rag']['accuracy']
    ]
    
    # Grouped bar chart
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, full_values, width, label='Full Context', 
                   color='#FF6B6B', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, rag_values, width, label='RAG', 
                   color='#4ECDC4', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Value', fontsize=12, weight='bold')
    ax.set_title('Experiment 3: RAG vs Full Context Comparison', 
                 fontsize=14, weight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=9, weight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/exp3_rag_vs_full.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_dir}/exp3_rag_vs_full.png")
    plt.close()
    
    # Improvement radar chart
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    categories = ['Speed\nImprovement', 'Token\nReduction', 'Accuracy\nGain']
    comparison = exp3['comparison']
    values = [
        comparison['speedup'] / 5 * 100,  # Normalize to percentage
        comparison['token_reduction_pct'],
        comparison['accuracy_improvement_pct']
    ]
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]
    
    ax.plot(angles, values, 'o-', linewidth=2, color='#4ECDC4')
    ax.fill(angles, values, alpha=0.25, color='#4ECDC4')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_title('Experiment 3: RAG Improvement Metrics (%)', 
                 fontsize=14, weight='bold', pad=20)
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/exp3_rag_improvements.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_dir}/exp3_rag_improvements.png")
    plt.close()


def plot_experiment4(results, output_dir='plots'):
    """Plot Experiment 4: Context Strategies results."""
    print("Generating Experiment 4 plots...")
    
    Path(output_dir).mkdir(exist_ok=True)
    exp4_data = results['experiment4_strategies']['results']
    
    # Prepare DataFrame
    df_dict = {}
    for strategy, metrics in exp4_data.items():
        df_dict[strategy] = {
            'Avg Accuracy': metrics['avg_accuracy'],
            'Avg Latency': metrics['avg_latency'],
            'Avg Tokens': metrics['avg_tokens'],
            'Final Tokens': metrics['final_tokens']
        }
    
    df = pd.DataFrame(df_dict).T
    df = df.reset_index().rename(columns={'index': 'Strategy'})
    
    # Multi-panel comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Experiment 4: Context Strategy Comparison', 
                 fontsize=16, weight='bold', y=1.00)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # Accuracy
    axes[0, 0].bar(range(len(df)), df['Avg Accuracy'], color=colors, 
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[0, 0].set_ylabel('Avg Accuracy', fontsize=11, weight='bold')
    axes[0, 0].set_title('Average Accuracy', fontsize=12, weight='bold')
    axes[0, 0].set_ylim(0, 1.0)
    axes[0, 0].set_xticks(range(len(df)))
    axes[0, 0].set_xticklabels(df['Strategy'], rotation=15, ha='right')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Latency
    axes[0, 1].bar(range(len(df)), df['Avg Latency'], color=colors, 
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[0, 1].set_ylabel('Avg Latency (s)', fontsize=11, weight='bold')
    axes[0, 1].set_title('Average Latency', fontsize=12, weight='bold')
    axes[0, 1].set_xticks(range(len(df)))
    axes[0, 1].set_xticklabels(df['Strategy'], rotation=15, ha='right')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Average Tokens
    axes[1, 0].bar(range(len(df)), df['Avg Tokens'], color=colors, 
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[1, 0].set_ylabel('Avg Tokens', fontsize=11, weight='bold')
    axes[1, 0].set_title('Average Token Count', fontsize=12, weight='bold')
    axes[1, 0].set_xticks(range(len(df)))
    axes[1, 0].set_xticklabels(df['Strategy'], rotation=15, ha='right')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Final Tokens
    axes[1, 1].bar(range(len(df)), df['Final Tokens'], color=colors, 
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[1, 1].set_ylabel('Final Tokens', fontsize=11, weight='bold')
    axes[1, 1].set_title('Final Token Count', fontsize=12, weight='bold')
    axes[1, 1].set_xticks(range(len(df)))
    axes[1, 1].set_xticklabels(df['Strategy'], rotation=15, ha='right')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/exp4_strategy_comparison.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_dir}/exp4_strategy_comparison.png")
    plt.close()


def generate_all_plots(results_file='context_lab_results.json', output_dir='plots'):
    """Generate all plots from experiment results."""
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80 + "\n")
    
    # Load results
    try:
        results = load_results(results_file)
    except FileNotFoundError:
        print(f"❌ Error: Results file '{results_file}' not found!")
        print("   Run 'python context_lab.py' first to generate results.")
        return
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Generate plots for each experiment
    plot_experiment1(results, output_dir)
    plot_experiment2(results, output_dir)
    plot_experiment3(results, output_dir)
    plot_experiment4(results, output_dir)
    
    print("\n" + "="*80)
    print(f"✅ ALL PLOTS GENERATED SUCCESSFULLY")
    print(f"📁 Output directory: {output_dir}/")
    print("="*80 + "\n")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate visualizations for Context Lab experiments'
    )
    parser.add_argument(
        '--results', 
        type=str, 
        default='context_lab_results.json',
        help='Path to results JSON file'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='plots',
        help='Output directory for plots'
    )
    
    args = parser.parse_args()
    
    generate_all_plots(args.results, args.output)


if __name__ == "__main__":
    main()

