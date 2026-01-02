"""
Script 2: Train Encoders for SageMaker
Trains encoder networks that map LLM hidden states to VSA representations.

Prerequisites:
    - Run generate_data.py first (or download data from S3)
    - Data should be in gathered_data_{run_name}/

Usage:
    python train_encoders.py --run_name experiment_1 --training_epochs 1000

After completion, upload to S3:
    aws s3 cp models/encoders_{run_name}.pth s3://your-bucket/models/
"""

import json
import torch
import numpy as np
import random
import os
import sys
import argparse
from pathlib import Path
import datetime
import wandb

import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

######################################################
# Parser
curr_date = datetime.datetime.now().strftime("%Y%m%d")

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


parser = argparse.ArgumentParser(description="Train Encoders")

# === Required ===
parser.add_argument("--run_name", type=str, required=True, help="Name of run (must match data generation)")

# === Path config ===
parser.add_argument("--curr_dir", type=str, default="~/Neurosymbolic-LLM/Programs", help="Path to program root")
parser.add_argument("--git_dir", type=str, default="~/Neurosymbolic-LLM", help="Path to project Git root")
parser.add_argument("--log_wandb", type=str2bool, default=False, help="Log to wandb")
parser.add_argument("--gpu_seed", type=int, default=42, help="GPU seed for reproducibility")

# === Model config ===
parser.add_argument("--model_dim", type=int, default=4096, help="Model dimension (LLaMA hidden size)")
parser.add_argument("--VSA_dim", type=int, default=2048, help="VSA dimensionality")
parser.add_argument("--max_digits", type=int, default=5, help="Max digits")
parser.add_argument("--possible_problems", nargs="+", type=str,
                    default=["addition", "subtraction", "multiplication", "division", "modulo", "gcd", "lcm", "power"],
                    help="Problem types")

# === Data config ===
parser.add_argument("--train_data_rounds", type=int, default=10000, help="Training data rounds")
parser.add_argument("--val_data_rounds", type=int, default=500, help="Validation data rounds")
parser.add_argument("--save_frequency", type=int, default=100, help="Save frequency used during generation")
parser.add_argument("--n_samples", type=int, default=4, help="Samples per batch during generation")
parser.add_argument("--layer_numbers", nargs="+", type=int,
                    default=list(range(33)), help="Layers to train encoders for")
parser.add_argument("--restrict_train_dataset", type=int, default=0, help="Limit training data (0=no limit)")
parser.add_argument("--restrict_val_dataset", type=int, default=0, help="Limit validation data (0=no limit)")
parser.add_argument("--tokens_to_keep", type=str, default="1", help="Tokens kept during generation")

# === Training config ===
parser.add_argument("--encoder_decoder_batch_size", type=int, default=512, help="Training batch size")
parser.add_argument("--training_epochs", type=int, default=1000, help="Number of epochs")
parser.add_argument("--learning_rate", type=float, default=0.001, help="Initial learning rate")
parser.add_argument("--train_freq_print", type=int, default=100, help="Print frequency")

# === S3 config ===
parser.add_argument("--s3_bucket", type=str, default="", help="S3 bucket")
parser.add_argument("--upload_to_s3", type=str2bool, default=False, help="Auto-upload when done")

args = parser.parse_args()

# Expand paths
curr_dir = str(Path(args.curr_dir).expanduser())
git_dir = str(Path(args.git_dir).expanduser())
run_name = args.run_name
log_wandb = args.log_wandb
gpu_seed = args.gpu_seed

model_dim = args.model_dim
VSA_dim = args.VSA_dim
max_digits = args.max_digits
possible_problems = args.possible_problems

train_data_rounds = args.train_data_rounds
val_data_rounds = args.val_data_rounds
save_frequency = args.save_frequency
n_samples = args.n_samples
layer_numbers = torch.tensor(args.layer_numbers)
restrict_train_dataset = args.restrict_train_dataset
restrict_val_dataset = args.restrict_val_dataset
tokens_to_keep = args.tokens_to_keep if args.tokens_to_keep == "all" else int(args.tokens_to_keep)

encoder_decoder_batch_size = args.encoder_decoder_batch_size
training_epochs = args.training_epochs
learning_rate = args.learning_rate
train_freq_print = args.train_freq_print

# Learning rate schedule
learning_rate_reduction_factors = {50: 0.5, 100: 0.5, 250: 0.1, 500: 0.4}

######################################################

sys.path.insert(0, git_dir)

from llama.encoder_decoder_networks import Encoder, LastTokenTransformer
from llama.vsa_engine import SymbolicEngine
from llama.utilities import generate_data_loaders, count_trainable_parameters
from llama.device_utils import (
    get_device,
    get_device_type,
    setup_device_environment,
    print_device_info,
)

######################################################

if log_wandb:
    wandb.finish()
    wandb.init(
        project="Symbolic LLM - Train Encoders",
        name=run_name,
    )

print("=" * 60)
print("ENCODER TRAINING SCRIPT")
print("=" * 60)
print(f"Run: {run_name}")

# Device setup
device = get_device()
device_type = get_device_type()
print_device_info()
setup_device_environment()

# Set default dtype
if device_type == "cuda" and torch.cuda.is_bf16_supported():
    torch.set_default_dtype(torch.bfloat16)
elif device_type == "mps":
    torch.set_default_dtype(torch.bfloat16)
else:
    torch.set_default_dtype(torch.float16)

torch.set_default_device(device_type)

# Load Symbolic Engine
possible_problems_str = "_".join(possible_problems)
vsa_path = f"{curr_dir}/VSA_library/symbolic_engine_VSA_dim_{VSA_dim}_max_digits_{max_digits}_problem_types_{possible_problems_str}.pt"

if os.path.exists(vsa_path):
    SE = torch.load(vsa_path, weights_only=False)
    print("Loaded Symbolic Engine")
else:
    SE = SymbolicEngine(VSA_dim=VSA_dim, max_digits=max_digits, possible_problems=possible_problems, curr_dir=curr_dir)
    os.makedirs(f"{curr_dir}/VSA_library", exist_ok=True)
    torch.save(SE, vsa_path)
    print("Created new Symbolic Engine")

# Data directory
save_dir = f"gathered_data_{run_name}"
if not os.path.exists(save_dir):
    print(f"ERROR: Data directory not found: {save_dir}")
    print("Run generate_data.py first or download from S3")
    sys.exit(1)

print(f"Loading data from: {save_dir}")

# Create data loaders
print("\nCreating data loaders...")
training_encoder_data_loaders = generate_data_loaders(
    mode='train', save_dir=save_dir, data_rounds=train_data_rounds, n_samples=n_samples, df_subset=None,
    save_frequency=save_frequency, layer_numbers=layer_numbers, restrict_dataset=restrict_train_dataset,
    tokens_to_keep=tokens_to_keep, batch_size=encoder_decoder_batch_size, gpu_seed=gpu_seed, verbose=True
)
print("Training data loaders created.")

validation_encoder_data_loaders = generate_data_loaders(
    mode='val', save_dir=save_dir, data_rounds=val_data_rounds, n_samples=n_samples, df_subset=None,
    save_frequency=save_frequency, layer_numbers=layer_numbers, restrict_dataset=restrict_val_dataset,
    tokens_to_keep=tokens_to_keep, batch_size=encoder_decoder_batch_size, gpu_seed=gpu_seed, verbose=True
)
print("Validation data loaders created.")

# Initialize encoders
print("\nInitializing encoders...")
encoders = torch.nn.ModuleList()
for layer_id in layer_numbers:
    if tokens_to_keep == 1:
        layer_encoder = Encoder(layer_id, model_dim, VSA_dim).to(device)
    else:
        layer_encoder = LastTokenTransformer(layer_id, model_dim, VSA_dim, num_layers=2, hidden_dim=512).to(device)
    encoders.append(layer_encoder)

print(f"Trainable parameters per encoder: {count_trainable_parameters(encoders[0])}")

# Training setup
optimizers = [optim.Adam(encoders[n].parameters(), lr=learning_rate) for n in range(len(layer_numbers))]
criterion = nn.MSELoss()
losses = np.zeros((len(layer_numbers), training_epochs))
val_losses = np.zeros((len(layer_numbers), training_epochs))
running_losses = np.zeros((len(layer_numbers)))

# Training loop
print("\n" + "=" * 60)
print("TRAINING ENCODERS")
print("=" * 60)

for i in range(training_epochs):
    # Learning rate schedule
    if i in learning_rate_reduction_factors.keys():
        for n in range(len(layer_numbers)):
            for param_group in optimizers[n].param_groups:
                param_group['lr'] = param_group['lr'] * learning_rate_reduction_factors[i]
        print(f"Learning rate reduced by factor {learning_rate_reduction_factors[i]}")

    for n, n_layer in enumerate(layer_numbers):
        encoders[n].train()
        running_loss = 0
        total_norm = 0.0

        for batch_idx, (data, labels) in enumerate(training_encoder_data_loaders[n]):
            model_pred = encoders[n](data)
            loss = torch.sqrt(criterion(model_pred, labels))
            loss.backward()

            tn = 0
            for p in encoders[n].parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    tn += param_norm.item() ** 2
            total_norm += tn ** 0.5

            optimizers[n].step()
            optimizers[n].zero_grad()
            running_loss += loss.item()

        running_loss /= (batch_idx + 1)
        running_losses[n] = running_loss
        total_norm /= (batch_idx + 1)
        losses[n][i] = running_loss

        # Validation
        v_predictions = []
        v_targets = []
        encoders.eval()

        val_running_loss = 0
        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(validation_encoder_data_loaders[n]):
                model_pred = encoders[n](data)
                loss = torch.sqrt(criterion(model_pred, labels))
                v_predictions.append(model_pred)
                v_targets.append(labels)
                val_running_loss += loss.item()

        val_running_loss /= (batch_idx + 1)
        val_losses[n][i] = val_running_loss

        v_predictions = torch.cat(v_predictions, dim=0)
        v_targets = torch.cat(v_targets, dim=0)

        v_digit_predictions_n1 = SE.decode_digits(v_predictions.type_as(SE.vectors[SE.VSA_n1]), SE.VSA_n1)
        v_digit_labels_n1 = SE.decode_digits(v_targets.type_as(SE.vectors[SE.VSA_n1]), SE.VSA_n1)
        v_digit_predictions_n2 = SE.decode_digits(v_predictions.type_as(SE.vectors[SE.VSA_n2]), SE.VSA_n2)
        v_digit_labels_n2 = SE.decode_digits(v_targets.type_as(SE.vectors[SE.VSA_n2]), SE.VSA_n2)

        average_correct_digits = (
            (v_digit_predictions_n1 == v_digit_labels_n1).sum(axis=1).float().mean().detach().cpu().item() +
            (v_digit_predictions_n2 == v_digit_labels_n2).sum(axis=1).float().mean().detach().cpu().item()
        ) / 2

        if not i % train_freq_print:
            print(f"    L{n_layer}: val avg correct digits: {average_correct_digits:.2f}")

        encoders.train()

    if not i % train_freq_print:
        print(f"Epoch {i}: Train Loss: {running_losses.mean():.4f}, Val Loss: {val_losses[:,i].mean():.4f}, Grad Norm: {total_norm:.4f}")

# Plot results
print("\nGenerating plots...")

plt.figure(figsize=(10, 6))
plt.plot(losses.mean(axis=0), marker=".", label='Training')
plt.plot(val_losses.mean(axis=0), marker=".", label='Validation')
plt.title("Average Encoder RMSE Loss Per Epoch")
plt.ylabel("RMSE")
plt.xlabel("Epoch")
plt.legend()
plt.savefig(f"encoder_loss_epochs_{run_name}.png", dpi=150, bbox_inches='tight')
if log_wandb:
    wandb.log({"Average Encoder RMSE Loss Per Epoch": wandb.Image(plt)})
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(losses.T[-1], marker=".", label='Training')
plt.plot(val_losses.T[-1], marker=".", label='Validation')
plt.title("Final Encoder RMSE Loss vs Layer Number")
plt.ylabel("RMSE")
plt.xlabel("Layer Number")
plt.legend()
plt.savefig(f"encoder_loss_layers_{run_name}.png", dpi=150, bbox_inches='tight')
if log_wandb:
    wandb.log({"Final Encoder RMSE Loss vs Layer Number": wandb.Image(plt)})
plt.close()

# Save models
os.makedirs(f"{curr_dir}/models", exist_ok=True)
torch.save(encoders.state_dict(), f"{curr_dir}/models/encoders_state_dict_{run_name}.pth")
torch.save(encoders, f"{curr_dir}/models/encoders_{run_name}.pth")
print(f"\nSaved: {curr_dir}/models/encoders_{run_name}.pth")

# Upload to S3 if configured
if args.upload_to_s3 and args.s3_bucket:
    import subprocess
    print(f"\nUploading to S3...")
    subprocess.run(["aws", "s3", "cp", f"{curr_dir}/models/encoders_{run_name}.pth",
                    f"s3://{args.s3_bucket}/models/encoders_{run_name}.pth"])
    subprocess.run(["aws", "s3", "cp", f"{curr_dir}/models/encoders_state_dict_{run_name}.pth",
                    f"s3://{args.s3_bucket}/models/encoders_state_dict_{run_name}.pth"])
    print("Upload complete.")

print("\n" + "=" * 60)
print("ENCODER TRAINING COMPLETE")
print("=" * 60)
print(f"\nNext step: Run train_decoders.py")
print(f"  python train_decoders.py --run_name {run_name}")

if log_wandb:
    wandb.finish()