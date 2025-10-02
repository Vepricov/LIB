#!/bin/bash

# Example usage script for experiment visualization
# Make sure to replace YOUR_WANDB_ENTITY and YOUR_WANDB_PROJECT with actual values

echo "=== Experiment Visualization Examples ==="
echo

# Create output directory
mkdir -p plots

echo "1. Generating Hess experiment visualization..."
python plot_experiments.py \
    --experiment hess \
    --output-dir ./plots \
    --output-name hess_soap_vs_dykaf

echo

echo "2. Generating ABL Llama experiment visualization..."
python plot_experiments.py \
    --experiment abl_llama \
    --output-dir ./plots \
    --output-name llama_adam1_rank_one_ablation

echo

echo "=== Visualization complete! ==="
echo "Check the ./plots directory for generated images:"
echo "- hess_soap_vs_dykaf.png"
echo "- llama_adam1_rank_one_ablation.png"
echo
