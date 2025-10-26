import os
import re
import ast
import csv
import sys
from typing import Dict, Any, Optional, Tuple, List, Set

import yaml


WANDB_DIR = os.path.join(os.path.dirname(__file__), "wandb")
OUT_DIR = os.path.join(os.path.dirname(__file__), "wandb_results")

# Default ft_strategy filter - can be overridden via command line
DEFAULT_FT_STRATEGY = "Full"


DATASET_METRIC_KEY = {
	"cola": "eval_matthews_correlation",
	"mnli": "eval_accuracy",
	"mrpc": "eval_accuracy",
	"qnli": "eval_accuracy",
	"qqp": "eval_accuracy",
	"rte": "eval_accuracy",
	"sst2": "eval_accuracy",
	"stsb": "eval_combined_score",
	"wnli": "eval_accuracy",
}


def read_yaml(path: str) -> Optional[Dict[str, Any]]:
	try:
		with open(path, "r") as f:
			return yaml.safe_load(f)
	except Exception:
		return None


def parse_config(config_path: str) -> Optional[Dict[str, Any]]:
	cfg = read_yaml(config_path)
	if not cfg or not isinstance(cfg, dict):
		return None

	def get_val(key: str) -> Optional[Any]:
		node = cfg.get(key)
		if isinstance(node, dict) and "value" in node:
			return node.get("value")
		return None

	dataset = get_val("dataset") or get_val("finetuning_task")
	optimizer = get_val("optimizer") or get_val("optim")
	ft_strategy = get_val("ft_strategy")
	lr = get_val("lr") or get_val("learning_rate")

	# Normalize
	if isinstance(optimizer, str):
		optimizer = optimizer.strip().lower()
	if isinstance(dataset, str):
		dataset = dataset.strip().lower()
	if isinstance(ft_strategy, str):
		ft_strategy = ft_strategy.strip()

	return {
		"dataset": dataset,
		"optimizer": optimizer,
		"ft_strategy": ft_strategy,
		"lr": lr,
	}


EVAL_LINE_PATTERN = re.compile(r"^\{.*\}$")


def extract_best_metric_from_output(log_path: str, target_key: str) -> Optional[Tuple[float, Dict[str, Any]]]:
	best_val: Optional[float] = None
	best_record: Optional[Dict[str, Any]] = None
	try:
		with open(log_path, "r") as f:
			for line in f:
				s = line.strip()
				if "'eval_" not in s and '"eval_' not in s:
					continue
				if not EVAL_LINE_PATTERN.match(s):
					continue
				try:
					rec = ast.literal_eval(s)
					if not isinstance(rec, dict):
						continue
				except Exception:
					continue
				if target_key in rec and isinstance(rec[target_key], (int, float)):
					val = float(rec[target_key])
					if best_val is None or val > best_val:
						best_val = val
						best_record = rec
	except FileNotFoundError:
		return None
	except Exception:
		return None

	if best_val is None:
		return None
	return best_val, (best_record or {})


def scan_runs(target_ft_strategy: str = DEFAULT_FT_STRATEGY) -> Dict[str, Dict[str, Dict[str, Any]]]:
	# results[dataset][optimizer] = {"best_value": float|"none", "best_lr": str|float|None, "run_id": str|None, "metric_key": str}
	results: Dict[str, Dict[str, Dict[str, Any]]] = {}

	if not os.path.isdir(WANDB_DIR):
		return results

	for name in os.listdir(WANDB_DIR):
		run_dir = os.path.join(WANDB_DIR, name)
		files_dir = os.path.join(run_dir, "files")
		if not (name.startswith("run-") and os.path.isdir(files_dir)):
			continue

		config_path = os.path.join(files_dir, "config.yaml")
		output_log = os.path.join(files_dir, "output.log")

		cfg = parse_config(config_path)
		if not cfg:
			continue

		dataset = cfg.get("dataset")
		optimizer = cfg.get("optimizer")
		ft_strategy = cfg.get("ft_strategy")
		lr = cfg.get("lr")

		if ft_strategy != target_ft_strategy:
			continue
		if dataset not in DATASET_METRIC_KEY:
			continue
		if not optimizer:
			optimizer = "unknown"

		metric_key = DATASET_METRIC_KEY[dataset]
		best_info = extract_best_metric_from_output(output_log, metric_key)

		if dataset not in results:
			results[dataset] = {}

		if optimizer not in results[dataset]:
			results[dataset][optimizer] = {
				"metric_key": metric_key,
				"best_value": None,
				"best_lr": None,
				"run_id": None,
			}

		if best_info is None:
			continue

		best_val, record = best_info
		prev = results[dataset][optimizer]["best_value"]
		if prev is None or (isinstance(prev, (int, float)) and best_val > float(prev)):
			results[dataset][optimizer]["best_value"] = best_val
			results[dataset][optimizer]["best_lr"] = lr
			results[dataset][optimizer]["run_id"] = name

	# Replace missing with "none"
	for ds in list(results.keys()):
		for opt in list(results[ds].keys()):
			if results[ds][opt]["best_value"] is None:
				results[ds][opt]["best_value"] = "none"
	return results


def scan_runs_both_strategies() -> Dict[str, Dict[str, Dict[str, Dict[str, Any]]]]:
	# results[ft_strategy][dataset][optimizer] = {"best_value": float|"none", "best_lr": str|float|None, "run_id": str|None, "metric_key": str}
	results: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {}
	strategies = ["LoRA", "Full"]
	
	for strategy in strategies:
		results[strategy] = scan_runs(strategy)
	
	return results


def write_csvs(results: Dict[str, Dict[str, Dict[str, Any]]], ft_strategy: str) -> None:
	os.makedirs(OUT_DIR, exist_ok=True)
	ft_suffix = ft_strategy.lower()
	for dataset, by_opt in results.items():
		out_path = os.path.join(OUT_DIR, f"glue_{dataset}_{ft_suffix}.csv")
		rows = []
		metric_key = DATASET_METRIC_KEY.get(dataset, "")
		for optimizer, info in sorted(by_opt.items()):
			rows.append({
				"optimizer": optimizer,
				"metric": metric_key,
				"best_value": info.get("best_value", "none"),
				"best_lr": info.get("best_lr"),
				"run_id": info.get("run_id"),
			})
		with open(out_path, "w", newline="") as f:
			writer = csv.DictWriter(f, fieldnames=["optimizer", "metric", "best_value", "best_lr", "run_id"])
			writer.writeheader()
			for r in rows:
				writer.writerow(r)

def write_combined_csv(combined_results: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]]) -> None:
	os.makedirs(OUT_DIR, exist_ok=True)
	out_path = os.path.join(OUT_DIR, "glue_combined_results.csv")
	
	rows = []
	for ft_strategy in ["LoRA", "Full"]:
		if ft_strategy not in combined_results:
			continue
		for dataset, by_opt in combined_results[ft_strategy].items():
			metric_key = DATASET_METRIC_KEY.get(dataset, "")
			for optimizer, info in sorted(by_opt.items()):
				rows.append({
					"ft_strategy": ft_strategy,
					"dataset": dataset,
					"optimizer": optimizer,
					"metric": metric_key,
					"best_value": info.get("best_value", "none"),
					"best_lr": info.get("best_lr"),
					"run_id": info.get("run_id"),
				})
	
	with open(out_path, "w", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=["ft_strategy", "dataset", "optimizer", "metric", "best_value", "best_lr", "run_id"])
		writer.writeheader()
		for r in rows:
			writer.writerow(r)


def _format_value(v: Any) -> str:
	if v is None:
		return "n/a"
	if isinstance(v, str):
		return v if v.strip() else "n/a"
	try:
		x = float(v)
		return f"{x:.4f}"
	except Exception:
		return "n/a"


def write_latex_table_from_results(results: Dict[str, Dict[str, Dict[str, Any]]], ft_strategy: str) -> None:
	os.makedirs(OUT_DIR, exist_ok=True)
	optimizers: List[str] = []
	seen: Set[str] = set()
	for by_opt in results.values():
		for opt in by_opt.keys():
			if opt not in seen:
				seen.add(opt)
				optimizers.append(opt)
	optimizers.sort()

	ft_suffix = ft_strategy.lower()
	table_path = os.path.join(OUT_DIR, f"glue_results_table_{ft_suffix}.tex")

	header_opt_cols = " & ".join(opt.upper() for opt in optimizers) if optimizers else ""

	lines: List[str] = []
	lines.append("% Auto-generated by extract_wandb_results.py")
	lines.append("\\begin{table}[t]")
	lines.append("\\centering")
	col_spec = "l l" + (" " + "c" * len(optimizers) if optimizers else "")
	lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
	lines.append("\\toprule")
	if optimizers:
		lines.append(f"Dataset & Metric & {header_opt_cols} \\\ ")
	else:
		lines.append("Dataset & Metric \\\ ")
	lines.append("\\midrule")

	glue_order = [
		"cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"
	]

	def pretty_metric_key(ds: str) -> str:
		raw = DATASET_METRIC_KEY.get(ds, "")
		mapping = {
			"eval_matthews_correlation": "Matthews",
			"eval_accuracy": "Accuracy",
			"eval_combined_score": "Combined",
		}
		return mapping.get(raw, raw)

	for ds in glue_order:
		if ds not in results:
			metric_name = pretty_metric_key(ds)
			vals = ["n/a" for _ in optimizers]
		else:
			metric_name = pretty_metric_key(ds)
			row = results[ds]
			vals = []
			for opt in optimizers:
				best = row.get(opt, {}).get("best_value")
				if isinstance(best, str) and best == "none":
					vals.append("n/a")
				else:
					vals.append(_format_value(best))
		if optimizers:
			lines.append(f"{ds.upper()} & {metric_name} & " + " & ".join(vals) + " \\")
		else:
			lines.append(f"{ds.upper()} & {metric_name} \\")

	lines.append("\\bottomrule")
	lines.append("\\end{tabular}")
	lines.append(f"\\caption{{GLUE {ft_strategy} fine-tuning: best validation metrics per optimizer (higher is better). Missing entries are denoted as n/a.}}")
	lines.append("\\label{tab:glue_results}")
	lines.append("\\end{table}")

	with open(table_path, "w") as f:
		f.write("\n".join(lines))


def write_combined_latex_table(combined_results: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]]) -> None:
	os.makedirs(OUT_DIR, exist_ok=True)
	
	# Collect all optimizers from both strategies
	optimizers: List[str] = []
	seen: Set[str] = set()
	for strategy_results in combined_results.values():
		for by_opt in strategy_results.values():
			for opt in by_opt.keys():
				if opt not in seen:
					seen.add(opt)
					optimizers.append(opt)
	optimizers.sort()

	table_path = os.path.join(OUT_DIR, "glue_combined_results_table.tex")

	lines: List[str] = []
	lines.append("% Auto-generated by extract_wandb_results.py")
	lines.append("\\begin{table}[t]")
	lines.append("\\centering")
	lines.append("\\scriptsize")
	lines.append("\\setlength{\\tabcolsep}{3pt}")
	lines.append("\\renewcommand{\\arraystretch}{1.1}")
	lines.append("\\resizebox{0.95\\linewidth}{!}{%")
	
	# Create column specification
	col_spec = "l l" + "c" * 9 + "|c"  # 9 datasets + average column
	lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
	
	# Header with dataset columns
	lines.append("    % \\toprule")
	header_line = "& & \\begin{tabular}{@{}c@{}}CoLA\\\\Matthews\\end{tabular}"
	header_line += " & \\begin{tabular}{@{}c@{}}MNLI\\\\Acc\\end{tabular}"
	header_line += " & \\begin{tabular}{@{}c@{}}MRPC\\\\Acc\\end{tabular}"
	header_line += " & \\begin{tabular}{@{}c@{}}QNLI\\\\Acc\\end{tabular}"
	header_line += " & \\begin{tabular}{@{}c@{}}QQP\\\\Acc\\end{tabular}"
	header_line += " & \\begin{tabular}{@{}c@{}}RTE\\\\Acc\\end{tabular}"
	header_line += " & \\begin{tabular}{@{}c@{}}SST-2\\\\Acc\\end{tabular}"
	header_line += " & \\begin{tabular}{@{}c@{}}STS-B\\\\Comb.\\end{tabular}"
	header_line += " & \\begin{tabular}{@{}c@{}}ALL\\\\Avg\\end{tabular} \\\\"
	lines.append(header_line)
	lines.append("\\midrule")

	glue_order = ["cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb"]

	# Process each strategy
	for strategy_idx, ft_strategy in enumerate(["LoRA", "Full"]):
		if ft_strategy not in combined_results:
			continue
		
		strategy_results = combined_results[ft_strategy]
		
		# Add strategy rows
		for opt_idx, optimizer in enumerate(optimizers):
			if opt_idx == 0:
				# First row for this strategy - use multirow
				lines.append(f"\\multirow{{{len(optimizers)}}}{{*}}{{{ft_strategy}}}")
			else:
				lines.append("")
			
			# Optimizer name
			opt_name = f"\\texttt{{{optimizer}}}"
			
			# Collect values for each dataset
			values = []
			valid_values = []  # For computing average
			
			for ds in glue_order:
				if ds in strategy_results and optimizer in strategy_results[ds]:
					best = strategy_results[ds][optimizer].get("best_value")
					if isinstance(best, str) and best == "none":
						values.append("n/a")
					else:
						formatted_val = _format_value(best)
						values.append(formatted_val)
						try:
							valid_values.append(float(best))
						except (ValueError, TypeError):
							pass
				else:
					values.append("n/a")
			
			# Calculate average
			if valid_values:
				avg_val = sum(valid_values) / len(valid_values)
				avg_formatted = f"{avg_val:.4f}"
			else:
				avg_formatted = "n/a"
			
			# Find best values for bolding
			best_in_strategy = {}
			for ds in glue_order:
				best_val = None
				for opt in optimizers:
					if ds in strategy_results and opt in strategy_results[ds]:
						val = strategy_results[ds][opt].get("best_value")
						if isinstance(val, (int, float)):
							if best_val is None or val > best_val:
								best_val = val
								best_in_strategy[ds] = opt
			
			# Format values with bold for best
			formatted_values = []
			for i, (ds, val) in enumerate(zip(glue_order, values)):
				if val != "n/a" and ds in best_in_strategy and best_in_strategy[ds] == optimizer:
					formatted_values.append(f"\\textbf{{{val}}}")
				else:
					formatted_values.append(val)
			
			# Check if this optimizer has best average
			best_avg_opt = None
			best_avg_val = None
			for opt in optimizers:
				opt_valid_values = []
				for ds in glue_order:
					if ds in strategy_results and opt in strategy_results[ds]:
						val = strategy_results[ds][opt].get("best_value")
						if isinstance(val, (int, float)):
							opt_valid_values.append(val)
				if opt_valid_values:
					opt_avg = sum(opt_valid_values) / len(opt_valid_values)
					if best_avg_val is None or opt_avg > best_avg_val:
						best_avg_val = opt_avg
						best_avg_opt = opt
			
			if avg_formatted != "n/a" and best_avg_opt == optimizer:
				avg_formatted = f"\\textbf{{{avg_formatted}}}"
			
			# Create the row
			row = f"& {opt_name}   & " + " & ".join(formatted_values) + f" & {avg_formatted} \\\\"
			lines.append(row)
		
		# Add separator between strategies
		if strategy_idx == 0:  # After LoRA, before Full
			lines.append("\\midrule")

	lines.append("\\bottomrule")
	lines.append("\\end{tabular}%")
	lines.append("}")
	lines.append("\\caption{GLUE (LoRA and Full fine-tuning): datasets are columns with metric under the dataset name; \\texttt{ALL} is the average over tasks.}")
	lines.append("\\label{tab:glue_results_transposed}")
	lines.append("\\end{table}")

	with open(table_path, "w") as f:
		f.write("\n".join(lines))


def main() -> None:
	# Check for command line argument to specify ft_strategy
	if len(sys.argv) > 1 and sys.argv[1] != "combined":
		# Single strategy mode (backward compatibility)
		ft_strategy = sys.argv[1]
		print(f"Extracting results for ft_strategy: {ft_strategy}")
		
		results = scan_runs(ft_strategy)
		if not results:
			print(f"No wandb results found for ft_strategy='{ft_strategy}' or wandb directory missing.")
			return
		write_csvs(results, ft_strategy)
		write_latex_table_from_results(results, ft_strategy)
		print(f"Wrote CSVs to: {OUT_DIR}")
		print(f"Wrote LaTeX table to: {os.path.join(OUT_DIR, f'glue_results_table_{ft_strategy.lower()}.tex')}")
	else:
		# Combined mode (default)
		print("Extracting results for both LoRA and Full strategies...")
		
		combined_results = scan_runs_both_strategies()
		if not combined_results or not any(combined_results.values()):
			print("No wandb results found for either strategy or wandb directory missing.")
			return
		
		write_combined_csv(combined_results)
		write_combined_latex_table(combined_results)
		print(f"Wrote combined CSV to: {os.path.join(OUT_DIR, 'glue_combined_results.csv')}")
		print(f"Wrote combined LaTeX table to: {os.path.join(OUT_DIR, 'glue_combined_results_table.tex')}")


if __name__ == "__main__":
	main()

