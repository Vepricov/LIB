import os
import csv
from typing import Dict, Any

try:
	import wandb
	WANDB_AVAILABLE = True
except ImportError:
	WANDB_AVAILABLE = False
	print("Error: wandb not installed. Run: pip install wandb")
	exit(1)


# Configuration
WANDB_PROJECT = "QUASI_DESCENT"
WANDB_ENTITY = "steeldream"
OUT_DIR = os.path.join(os.path.dirname(__file__), "wandb_results")

OPTIMIZER_WHITELIST = {
	"rmsspectral",
	"rmsspectral_sania",
	"adamuon",
	"muon",
	"adamw",
}

DATASETS = ["cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]

# Metric keys in W&B API format
METRIC_MAP = {
	"cola": "eval/matthews_correlation",
	"mnli": "eval/accuracy",
	"mrpc": "eval/accuracy",
	"qnli": "eval/accuracy",
	"qqp": "eval/accuracy",
	"rte": "eval/accuracy",
	"sst2": "eval/accuracy",
	"stsb": "eval/combined_score",
	"wnli": "eval/accuracy",
}


def fetch_results() -> Dict[str, Dict[str, Dict[str, Any]]]:
	"""Fetch results from W&B API for both LoRA and Full strategies."""
	api = wandb.Api()
	project_path = f"{WANDB_ENTITY}/{WANDB_PROJECT}"
	
	print(f"Fetching runs from W&B project: {project_path}")
	runs = api.runs(project_path)
	
	# Structure: results[ft_strategy][dataset][optimizer] = {best_value, lr, run_id}
	results = {"LoRA": {}, "Full": {}}
	
	for strategy in ["LoRA", "Full"]:
		for dataset in DATASETS:
			results[strategy][dataset] = {}
	
	total = 0
	processed = 0
	
	for run in runs:
		total += 1
		try:
			config = run.config
			summary = run.summary._json_dict
			
			# Extract config
			dataset = (config.get("dataset") or "").strip().lower()
			optimizer = (config.get("optimizer") or "").strip().lower().replace('-', '_')
			ft_strategy = (config.get("ft_strategy") or "").strip()
			lr = config.get("lr") or config.get("learning_rate")
			
			# Filter
			if ft_strategy not in ["LoRA", "Full"]:
				continue
			if dataset not in DATASETS:
				continue
			if optimizer not in OPTIMIZER_WHITELIST:
				continue
			
			# Get metric
			metric_key = METRIC_MAP[dataset]
			best_val = summary.get(metric_key)
			
			if not isinstance(best_val, (int, float)):
				continue
			
			# Update best for this optimizer/dataset/strategy
			current = results[ft_strategy][dataset].get(optimizer)
			if current is None or best_val > current.get("best_value", 0):
				results[ft_strategy][dataset][optimizer] = {
					"best_value": best_val,
					"lr": lr,
					"run_id": run.id,
				}
			
			processed += 1
			
		except Exception:
			continue
	
	print(f"Processed {processed}/{total} runs")
	return results


def write_csv(results: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
	"""Write single combined CSV with all results."""
	os.makedirs(OUT_DIR, exist_ok=True)
	out_path = os.path.join(OUT_DIR, "glue_results.csv")
	
	rows = []
	for ft_strategy in ["LoRA", "Full"]:
		for dataset in DATASETS:
			for optimizer in sorted(OPTIMIZER_WHITELIST):
				result = results[ft_strategy][dataset].get(optimizer)
				if result:
					rows.append({
						"ft_strategy": ft_strategy,
						"dataset": dataset,
						"optimizer": optimizer,
						"best_value": f"{result['best_value']:.4f}",
						"lr": result["lr"],
						"run_id": result["run_id"],
					})
				else:
					rows.append({
						"ft_strategy": ft_strategy,
						"dataset": dataset,
						"optimizer": optimizer,
						"best_value": "n/a",
						"lr": "",
						"run_id": "",
					})
	
	with open(out_path, "w", newline="") as f:
		writer = csv.DictWriter(
			f,
			fieldnames=["ft_strategy", "dataset", "optimizer", "best_value", "lr", "run_id"]
		)
		writer.writeheader()
		writer.writerows(rows)
	
	print(f"\nWrote results to: {out_path}")


def write_latex_table(results: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
	"""Write LaTeX table with both LoRA and Full results."""
	os.makedirs(OUT_DIR, exist_ok=True)
	out_path = os.path.join(OUT_DIR, "glue_table.tex")
	
	# Dataset order and their short names for table
	datasets_short = ["cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb"]
	opt_map = {
		"adamw": "AdamW",
		"muon": "Muon", 
		"adamuon": "AdaMuon",
		"rmsspectral": "RMSSpectral",
		"rmsspectral_sania": "RMSSpectral-SANIA"
	}
	
	lines = [
		"\\begin{table}[h!]",
		"\\centering",
		"\\scriptsize",
		"\\setlength{\\tabcolsep}{3pt}",
		"\\renewcommand{\\arraystretch}{1.1}",
		"\\captionof{table}{GLUE: datasets are columns with the corresponding metric; \\texttt{ALL} is the average over tasks. Best per task in bold.}",
		"\\resizebox{0.95\\linewidth}{!}{",
		"\\begin{tabular}{l lcccccccc|c}",
		"& & \\begin{tabular}{@{}c@{}}CoLA\\\\Matthews\\end{tabular} & \\begin{tabular}{@{}c@{}}MNLI\\\\Acc\\end{tabular} & \\begin{tabular}{@{}c@{}}MRPC\\\\Acc\\end{tabular} & \\begin{tabular}{@{}c@{}}QNLI\\\\Acc\\end{tabular} & \\begin{tabular}{@{}c@{}}QQP\\\\Acc\\end{tabular} & \\begin{tabular}{@{}c@{}}RTE\\\\Acc\\end{tabular} & \\begin{tabular}{@{}c@{}}SST-2\\\\Acc\\end{tabular} & \\begin{tabular}{@{}c@{}}STS-B\\\\Comb.\\end{tabular} & \\begin{tabular}{@{}c@{}}ALL\\\\Avg\\end{tabular} \\\\",
		"\\midrule"
	]
	
	for strategy_idx, strategy in enumerate(["LoRA", "Full"]):
		# Find best values per dataset across all optimizers for this strategy
		best_vals = {}
		for dataset in datasets_short:
			best_val = None
			for optimizer in OPTIMIZER_WHITELIST:
				result = results[strategy][dataset].get(optimizer)
				if result and isinstance(result["best_value"], (int, float)):
					if best_val is None or result["best_value"] > best_val:
						best_val = result["best_value"]
			best_vals[dataset] = best_val
		
		num_opts = len(OPTIMIZER_WHITELIST)
		
		for opt_idx, optimizer in enumerate(sorted(OPTIMIZER_WHITELIST)):
			opt_name = opt_map.get(optimizer, optimizer.capitalize())
			
			# Values for each dataset
			values = []
			valid_values = []
			
			for dataset in datasets_short:
				result = results[strategy][dataset].get(optimizer)
				if result and isinstance(result["best_value"], (int, float)):
					val = result["best_value"]
					valid_values.append(val)
					# Bold if best in this dataset for this strategy
					if best_vals[dataset] and abs(val - best_vals[dataset]) < 1e-6:
						values.append(f"\\textbf{{{val:.4f}}}")
					else:
						values.append(f"{val:.4f}")
				else:
					values.append("NaN")
			
			# Calculate average
			if valid_values:
				avg = sum(valid_values) / len(valid_values)
				
				# Check if this is best average for this strategy
				best_avg = None
				for opt_check in OPTIMIZER_WHITELIST:
					opt_vals = []
					for ds in datasets_short:
						res = results[strategy][ds].get(opt_check)
						if res and isinstance(res["best_value"], (int, float)):
							opt_vals.append(res["best_value"])
					if opt_vals:
						opt_avg = sum(opt_vals) / len(opt_vals)
						if best_avg is None or opt_avg > best_avg:
							best_avg = opt_avg
				
				if best_avg and abs(avg - best_avg) < 1e-6:
					avg_str = f"\\textbf{{{avg:.4f}}}"
				else:
					avg_str = f"{avg:.4f}"
			else:
				avg_str = "NaN"
			
			# Add multirow for first optimizer of each strategy
			if opt_idx == 0:
				prefix = f"\\multirow{{{num_opts}}}{{*}}{{{strategy}}}"
			else:
				prefix = ""
			
			row = f"{prefix} & \\texttt{{{opt_name}}} & {' & '.join(values)} & {avg_str} \\\\"
			lines.append(row)
		
		# Add midrule between strategies (but not after last)
		if strategy_idx == 0:
			lines.append("\\midrule")
	
	lines.extend([
		"\\bottomrule",
		"\\end{tabular}",
		"}",
		"\\label{tab:glue_results_transposed}",
		"\\end{table}"
	])
	
	with open(out_path, "w") as f:
		f.write("\n".join(lines))
	
	print(f"Wrote LaTeX table to: {out_path}")


def main() -> None:
	if not WANDB_AVAILABLE:
		return
	
	results = fetch_results()
	write_csv(results)
	write_latex_table(results)
	
	# Print summary
	print("\nSummary:")
	for strategy in ["LoRA", "Full"]:
		found = sum(1 for ds in results[strategy].values() for opt in ds.keys())
		print(f"  {strategy}: {found} optimizer/dataset combinations")


if __name__ == "__main__":
	main()

