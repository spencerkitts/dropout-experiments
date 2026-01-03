# Dropout Introspection Experiments

This repository contains experiments investigating whether language models can detect and condition behavior on their own dropout state.

## Experiments

### Experiment 1: Intrinsic Dropout Sensitivity
Tests if an untrained Qwen3-8B model can distinguish between dropout ON vs OFF states.

### Experiment 2: Training Dropout Detection
Trains a QLoRA adapter to explicitly report dropout state ("Yes"/"No").

### Experiment 3: Model Organism - Conditional Deceptive Behavior
Demonstrates a model that says "No" (safe) when dropout is ON but reveals a secret ("ProjectGemini") when dropout is OFF.

## Files

- `train_with_checkpoints.py` - Dropout detection training with accuracy checkpoints
- `train_deceptive_v3.py` - Model organism training (conditional behavior)
- `train_safety_v3.py` - Safety training intervention
- `sweep_dropout_comparison.py` - Base model dropout sensitivity sweep
- `visualize.py` - Generate all figures

## Data Files

- `accuracy_sweep_p*.json` - Accuracy at checkpoints for different dropout rates
- `training_log_p0.01.json` - Full training log for p=0.01
- `deceptive_training_log.json` - Model organism training log
- `sweep_comparison_v2.txt` - Raw sweep output

## Usage

```bash
# Train dropout detection
python train_with_checkpoints.py --dropout 0.01

# Train model organism
python train_deceptive_v3.py

# Generate figures
python visualize.py
```

## Requirements

- torch
- transformers
- peft
- bitsandbytes
- matplotlib
