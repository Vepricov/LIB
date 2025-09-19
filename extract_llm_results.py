#!/usr/bin/env python3

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import wandb

plt.style.use('default')
sns.set_palette("husl")

def extract_llm_results():
    """Extract LLM results from wandb API."""
    api = wandb.Api()
    
    datasets = ['boolq', 'hella_swag', 'arc_challenge']
    optimizer_map = {
        'adamw': 'AdamW',
        'muon': 'Muon', 
        'taia': 'MuAdam'
    }
    
    # Storage: dataset -> optimizer -> [accuracies]
    results = defaultdict(lambda: defaultdict(list))
    
    print("Fetching runs from wandb API...")
    
    try:
        runs = api.runs("QUASI_DESCENT")
        print(f"Found {len(runs)} total runs")
    except Exception as e:
        print(f"Error accessing wandb API: {e}")
        return {}
    
    processed_runs = 0
    
    for run in runs:
        if not (run.name and 'llm_comp_' in run.name):
            continue
            
        config = run.config
        dataset = config.get('dataset', '').lower()
        optimizer = config.get('optimizer', '').lower()
        
        if dataset not in datasets or optimizer not in optimizer_map:
            continue
            
        summary = run.summary
        final_accuracy = summary.get('eval/accuracy') or summary.get('final_accuracy')
        
        if final_accuracy is not None:
            results[dataset][optimizer_map[optimizer]].append(final_accuracy)
            processed_runs += 1
            print(f"  {dataset} | {optimizer_map[optimizer]}: {final_accuracy:.4f}")
    
    print(f"Processed {processed_runs} relevant runs")
    
    if processed_runs == 0:
        print("No relevant runs found!")
        return {}
    
    return results

def save_results_csv(results):
    """Save results to CSV."""
    csv_path = os.path.join(os.path.dirname(__file__), 'wandb_results', 'llm_results.csv')
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Dataset', 'Optimizer', 'Mean_Accuracy', 'Std_Accuracy', 'Count'])
        
        for dataset in ['boolq', 'hella_swag', 'arc_challenge']:
            for optimizer in ['AdamW', 'Muon', 'MuAdam']:
                accuracies = results[dataset][optimizer]
                if accuracies:
                    mean_acc = np.mean(accuracies)
                    std_acc = np.std(accuracies)
                    count = len(accuracies)
                else:
                    mean_acc = std_acc = count = 0
                
                writer.writerow([dataset, optimizer, f'{mean_acc:.4f}', f'{std_acc:.4f}', count])
    
    print(f"Results saved to: {csv_path}")
    return csv_path

def create_bar_chart(results):
    """Create bar chart comparing optimizers."""
    
    datasets = ['boolq', 'hella_swag', 'arc_challenge']
    optimizers = ['AdamW', 'Muon', 'MuAdam']
    
    means = []
    stds = []
    
    for dataset in datasets:
        dataset_means = []
        dataset_stds = []
        for optimizer in optimizers:
            accuracies = results[dataset][optimizer]
            if accuracies:
                dataset_means.append(np.mean(accuracies))
                dataset_stds.append(np.std(accuracies))
            else:
                dataset_means.append(0)
                dataset_stds.append(0)
        means.append(dataset_means)
        stds.append(dataset_stds)
    
    means = np.array(means)
    stds = np.array(stds)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(datasets))
    width = 0.25
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, optimizer in enumerate(optimizers):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, means[:, i], width, 
                     yerr=stds[:, i], capsize=5,
                     label=optimizer, color=colors[i], alpha=0.8)
        
        # Add value labels
        for j, bar in enumerate(bars):
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + stds[j, i] + 0.005,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xticks(x)
    ax.set_xticklabels([d.replace('_', ' ').title() for d in datasets])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    ax.set_ylim(0, max(means.flatten()) + max(stds.flatten()) + 0.05)
    
    plt.tight_layout()
    
    plot_path = os.path.join(os.path.dirname(__file__), 'wandb_results', 'llm_results_chart.png')
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Bar chart saved to: {plot_path}")
    
    plt.show()
    return plot_path

def main():
    print("=" * 60)
    print("LLM Fine-tuning Results Extraction and Visualization")
    print("=" * 60)
    
    results = extract_llm_results()
    
    if not results:
        print("No results found!")
        return
    
    print("\nResults Summary:")
    print("-" * 40)
    for dataset in ['boolq', 'hella_swag', 'arc_challenge']:
        print(f"\n{dataset.replace('_', ' ').title()}:")
        for optimizer in ['AdamW', 'Muon', 'MuAdam']:
            accuracies = results[dataset][optimizer]
            if accuracies:
                mean_acc = np.mean(accuracies)
                std_acc = np.std(accuracies)
                print(f"  {optimizer}: {mean_acc:.4f} ± {std_acc:.4f} (n={len(accuracies)})")
            else:
                print(f"  {optimizer}: No data")
    
    csv_path = save_results_csv(results)
    plot_path = create_bar_chart(results)
    
    print(f"\nFiles generated:")
    print(f"  CSV: {csv_path}")
    print(f"  Chart: {plot_path}")
    print("\nDone!")

if __name__ == "__main__":
    main()