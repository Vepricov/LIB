#!/usr/bin/env python3
"""
Unified visualization script for A* paper experiments.
Supports two experiment types: hess and abl_llama.
"""

import argparse
import wandb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import seaborn as sns
from pathlib import Path

# Set up matplotlib for publication-quality plots
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'sans-serif',
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'text.usetex': False,  # Set to True if you have LaTeX installed
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 2.5,
    'axes.linewidth': 1.2
})

# Color palette for consistency
COLORS = {
    'SOAP': '#1f77b4',      # Blue
    'DyKAF': '#ff7f0e',     # Orange
    'adam1_rank_one_true': '#2ca02c',   # Green
    'adam1_rank_one_false': '#d62728'   # Red
}

class ExperimentVisualizer:
    def __init__(self, entity: str, project: str, experiment: str):
        """Initialize the visualizer with W&B credentials."""
        self.entity = entity
        self.project = project
        self.experiment = experiment
        self.api = wandb.Api()

    def fetch_runs(self, tag: str, filters: Optional[Dict] = None) -> List:
        """Fetch runs from W&B with given tag and optional filters."""
        filter_dict = {"tags": tag}
        if filters:
            filter_dict.update(filters)

        runs = self.api.runs(
            f"{self.entity}/{self.project}",
            filters=filter_dict
        )
        return list(runs)

    def extract_hess_data(self, runs: List) -> Dict:
        """Extract data for hess experiment."""
        data = {}

        print(f"Found {len(runs)} runs with 'hess' tag")

        for run in runs:
            config = run.config
            import re
            match = re.search(r'#samples=(\d+)', run.name)
            if match:
                train_samples = int(match.group(1))
                optimizer = run.name.split("#")[0]

            print(f"Run: {run.name}, optimizer: {optimizer}, train_samples: {train_samples}")

            # Map optimizer names
            if optimizer == 'SOAP ':
                optimizer_name = 'SOAP'
            elif optimizer == 'MDS ':
                optimizer_name = 'DyKAF'
            else:
                print(f"Skipping run with unknown optimizer: {optimizer}")
                continue

            # Get history data
            history = run.history()
            print(f"Available columns: {list(history.columns)}")

            # Try to find the right column names (flexible matching)
            step_col = None
            hess_col = None

            for col in history.columns:
                if 'step' in col.lower():
                    step_col = col
                if 'hess' in col.lower() and 'diff' in col.lower():
                    hess_col = col

            if step_col and hess_col:
                key = (optimizer_name, train_samples)
                if key not in data:
                    data[key] = []

                df = history[[step_col, hess_col]].dropna()
                df.columns = ['step', 'hess_diff']  # Standardize column names
                print(f"Added {len(df)} data points for {key}")
                data[key].append(df)
            else:
                print(f"Missing required columns (step: {step_col}, hess_diff: {hess_col}) in run {run.name}")

        print(f"Final data keys: {list(data.keys())}")
        return data

    def extract_abl_data(self, runs: List) -> Dict:
        """Extract data for abl_llama experiment."""
        data = {}

        print(f"Found {len(runs)} runs with 'abl_llama' tag")

        for run in runs:
            config = run.config
            adam1_rank_one = config.get('adam1_rank_one', None)
            mikola_rank_one = config.get('mikola_rank_one', None)

            # Use mikola_rank_one if adam1_rank_one is not available
            if adam1_rank_one is None and mikola_rank_one is not None:
                adam1_rank_one = mikola_rank_one

            print(f"Run: {run.name}, adam1_rank_one: {adam1_rank_one}, mikola_rank_one: {mikola_rank_one}")
            print(f"Full config keys: {list(config.keys())}")

            if adam1_rank_one is None:
                print(f"Skipping run without adam1_rank_one or mikola_rank_one config")
                continue

            key = f"adam1_rank_one_{str(adam1_rank_one).lower()}"

            # Get history data
            history = run.history()
            print(f"Available columns: {list(history.columns)}")

            # Try to find the right column names (flexible matching)
            step_col = None
            loss_col = None

            for col in history.columns:
                if 'step' in col.lower():
                    step_col = col
                if 'val' in col.lower() and 'loss' in col.lower():
                    loss_col = col
                elif 'loss' in col.lower() and not step_col:  # fallback to any loss
                    loss_col = col

            if step_col and loss_col:
                if key not in data:
                    data[key] = []

                df = history[[step_col, loss_col]].dropna()
                df.columns = ['step', 'val_loss']  # Standardize column names
                print(f"Added {len(df)} data points for {key}")
                data[key].append(df)
            else:
                print(f"Missing required columns (step: {step_col}, val_loss: {loss_col}) in run {run.name}")

        print(f"Final data keys: {list(data.keys())}")
        return data

    def aggregate_runs(self, run_data: List[pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Aggregate multiple runs by computing mean and std."""
        if not run_data:
            return np.array([]), np.array([]), np.array([])

        # Find common step range
        min_steps = max(df['step'].min() for df in run_data)
        max_steps = min(df['step'].max() for df in run_data)

        # Create common step grid
        step_range = np.linspace(min_steps, max_steps, 1000)

        # Interpolate all runs to common grid
        interpolated_values = []
        for df in run_data:
            values = np.interp(step_range, df['step'], df.iloc[:, 1])  # Second column (metric)
            interpolated_values.append(values)

        interpolated_values = np.array(interpolated_values)

        # Compute statistics
        mean_values = np.mean(interpolated_values, axis=0)
        std_values = np.std(interpolated_values, axis=0)

        return step_range, mean_values, std_values

    def plot_hess_experiment(self, data: Dict, output_path: str):
        """Plot hess experiment with subplots for different train_samples."""
        # Get unique train_samples values
        train_samples_values = list(set(key[1] for key in data.keys()))

        # Handle case where all samples are 'unknown' - create single plot
        if train_samples_values == ['unknown']:
            print("All train_samples are 'unknown', creating single combined plot")
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))

            for optimizer in ['SOAP', 'DyKAF']:
                key = (optimizer, 'unknown')
                if key in data:
                    steps, mean_vals, std_vals = self.aggregate_runs(data[key])
                    print(f"Plotting {optimizer}: {len(steps)} points")

                    if len(steps) > 0:
                        color = COLORS[optimizer]
                        ax.plot(steps, mean_vals, label=optimizer, color=color, linewidth=2.5)
                        ax.fill_between(steps, mean_vals - std_vals, mean_vals + std_vals,
                                      color=color, alpha=0.2)

            ax.set_xlabel('Step')
            ax.set_ylabel('Hessian Difference')
            ax.set_title('SOAP vs DyKAF: Hessian Difference Comparison')
            ax.legend()
            ax.grid(True, alpha=0.3)

        else:
            # Sort numeric values properly
            try:
                train_samples_values.sort(key=lambda x: int(x) if str(x).isdigit() else float('inf'))
            except:
                train_samples_values.sort()

            print(f"Train samples values found: {train_samples_values}")

            if len(train_samples_values) == 0:
                print("No data found for hess experiment")
                return

            # Ensure we have the right number of subplots
            n_plots = min(len(train_samples_values), 3)

            if n_plots == 1:
                fig, ax = plt.subplots(1, 1, figsize=(8, 6))
                axes = [ax]
            else:
                fig, axes = plt.subplots(1, n_plots, figsize=(4*n_plots, 5))
                if not isinstance(axes, np.ndarray):
                    axes = [axes]

            for i, train_samples in enumerate(train_samples_values[:3]):  # Limit to 3 plots max
                ax = axes[i]

                for optimizer in ['SOAP', 'DyKAF']:
                    key = (optimizer, train_samples)
                    if key in data:
                        steps, mean_vals, std_vals = self.aggregate_runs(data[key])
                        print(f"Plotting {optimizer} for train_samples={train_samples}: {len(steps)} points")

                        if len(steps) > 0:
                            color = COLORS[optimizer]
                            ax.plot(steps, mean_vals, label=optimizer, color=color, linewidth=2.5, )
                            ax.fill_between(steps, mean_vals - std_vals, mean_vals + std_vals,
                                          color=color, alpha=0.2)

                ax.set_xlabel('Step')
                ax.set_ylabel('Hessian Difference (log scale)')
                ax.set_title(f'Train Samples: {train_samples}')
                ax.set_yscale('log')
                ax.legend()
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Hess experiment plot saved to: {output_path}")

    def plot_abl_experiment(self, data: Dict, output_path: str):
        """Plot abl_llama experiment with single plot comparing adam1_rank_one settings."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        for key in ['adam1_rank_one_true', 'adam1_rank_one_false']:
            if key in data:
                steps, mean_vals, std_vals = self.aggregate_runs(data[key])

                if len(steps) > 0:
                    color = COLORS[key]
                    label = 'rank1_second_moment=True' if 'true' in key else 'rank1_second_moment=False'
                    ax.plot(steps, mean_vals, label=label, color=color, linewidth=2.5)
                    ax.fill_between(steps, mean_vals - std_vals, mean_vals + std_vals,
                                  color=color, alpha=0.2)

        ax.set_xlabel('Step')
        ax.set_ylabel('Validation Loss')
        ax.set_title('Llama 124M + FineWeb. Ablation on rank1 update in Adam step.')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"ABL experiment plot saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Visualize experiments for A* paper')
    parser.add_argument('--experiment', choices=['hess', 'abl_llama'], required=True,
                       help='Type of experiment to visualize')
    parser.add_argument('--entity', help='W&B entity name', default="andrey")
    parser.add_argument('--project', help='W&B project name', default="MIKOLA_DROP_SOAP")
    parser.add_argument('--output-dir', default='./plots', help='Output directory for plots')
    parser.add_argument('--output-name', help='Custom output filename (without extension)')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize visualizer
    visualizer = ExperimentVisualizer(args.entity, args.project, experiment=args.experiment)

    try:
        if args.experiment == 'hess':
            print("Fetching hess experiment data...")
            runs = visualizer.fetch_runs('hess')
            print(f"Found {len(runs)} runs total")
            data = visualizer.extract_hess_data(runs)

            if not data:
                print("No data found for hess experiment. Check your W&B project and tags.")
                print("Make sure your runs have:")
                print("- tag: 'hess'")
                print("- config fields: 'optimizer', 'train_samples'")
                print("- logged metrics with 'step' and 'hess_diff' (or similar)")
                return

            output_name = args.output_name or 'hess_experiment'
            output_path = output_dir / f'{output_name}.png'
            visualizer.plot_hess_experiment(data, str(output_path))

        elif args.experiment == 'abl_llama':
            print("Fetching abl_llama experiment data...")
            runs = visualizer.fetch_runs('abl_llama')
            print(f"Found {len(runs)} runs total")
            data = visualizer.extract_abl_data(runs)

            if not data:
                print("No data found for abl_llama experiment. Check your W&B project and tags.")
                print("Make sure your runs have:")
                print("- tag: 'abl_llama'")
                print("- config field: 'adam1_rank_one'")
                print("- logged metrics with 'step' and 'val_loss' (or similar)")
                return

            output_name = args.output_name or 'abl_llama_experiment'
            output_path = output_dir / f'{output_name}.png'
            visualizer.plot_abl_experiment(data, str(output_path))

    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
