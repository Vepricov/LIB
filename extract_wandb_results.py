import os
import re
import ast
import csv
from typing import Dict, Any, Optional, Tuple, List, Set

import yaml


WANDB_DIR = os.path.join(os.path.dirname(__file__), "wandb")
OUT_DIR = os.path.join(os.path.dirname(__file__), "wandb_results")


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


def scan_runs() -> Dict[str, Dict[str, Dict[str, Any]]]:
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

		if ft_strategy != "LoRA":
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


def write_csvs(results: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
	os.makedirs(OUT_DIR, exist_ok=True)
	for dataset, by_opt in results.items():
		out_path = os.path.join(OUT_DIR, f"glue_{dataset}.csv")
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


def write_latex_table_from_results(results: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
	os.makedirs(OUT_DIR, exist_ok=True)
	optimizers: List[str] = []
	seen: Set[str] = set()
	for by_opt in results.values():
		for opt in by_opt.keys():
			if opt not in seen:
				seen.add(opt)
				optimizers.append(opt)
	optimizers.sort()

	table_path = os.path.join(OUT_DIR, "glue_results_table.tex")

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
	lines.append("\\caption{GLUE LoRA fine-tuning: best validation metrics per optimizer (higher is better). Missing entries are denoted as n/a.}")
	lines.append("\\label{tab:glue_results}")
	lines.append("\\end{table}")

	with open(table_path, "w") as f:
		f.write("\n".join(lines))


def main() -> None:
	results = scan_runs()
	if not results:
		print("No wandb results found or wandb directory missing.")
		return
	write_csvs(results)
	write_latex_table_from_results(results)
	print(f"Wrote CSVs to: {OUT_DIR}")
	print(f"Wrote LaTeX table to: {os.path.join(OUT_DIR, 'glue_results_table.tex')}")


if __name__ == "__main__":
	main()

