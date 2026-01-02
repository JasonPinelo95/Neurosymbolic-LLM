"""
Script 3: Train Decoders for SageMaker
Trains decoder networks that map VSA representations back to LLM hidden states.

Prerequisites:
    - Run generate_data.py first
    - Run train_encoders.py first
    - Encoder models should be in models/encoders_{run_name}.pth

Usage:
    python train_decoders.py --run_name experiment_1 --decoding_epochs 1000

After completion, upload to S3:
    aws s3 cp models/decoders_{run_name}.pth s3://your-bucket/models/
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


parser = argparse.ArgumentParser(description="Train Decoders")

# === Required ===
parser.add_argument("--run_name", type=str, required=True, help="Name of run (must match encoder training)")

# === Path config ===
parser.add_argument("--curr_dir", type=str, default="~/Neurosymbolic-LLM/Programs", help="Path to program root")
parser.add_argument("--git_dir", type=str, default="~/Neurosymbolic-LLM", help="Path to project Git root")
parser.add_argument("--encoder_path", type=str, default="", help="Path to encoder model (default: models/encoders_{run_name}.pth)")
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
parser.add_argument("--test_data_rounds", type=int, default=500, help="Test data rounds")
parser.add_argument("--save_frequency", type=int, default=100, help="Save frequency used during generation")
parser.add_argument("--n_samples", type=int, default=4, help="Samples per batch during generation")
parser.add_argument("--max_batch_size", type=int, default=4, help="Max batch size")
parser.add_argument("--layer_numbers", nargs="+", type=int,
                    default=list(range(33)), help="Layers to train decoders for")
parser.add_argument("--restrict_train_dataset", type=int, default=0, help="Limit training data (0=no limit)")
parser.add_argument("--restrict_val_dataset", type=int, default=0, help="Limit validation data (0=no limit)")
parser.add_argument("--restrict_test_dataset", type=int, default=0, help="Limit test data (0=no limit)")
parser.add_argument("--tokens_to_keep", type=str, default="1", help="Tokens kept during generation")
parser.add_argument("--complexity", type=int, default=2, help="Problem complexity")
parser.add_argument("--problem_type", nargs="+", type=str, default=["addition"], help="Problem types")

# === Training config ===
parser.add_argument("--encoder_decoder_batch_size", type=int, default=512, help="Training batch size")
parser.add_argument("--decoding_epochs", type=int, default=1000, help="Number of epochs")
parser.add_argument("--decoding_learning_rate", type=float, default=0.0001, help="Initial learning rate")
parser.add_argument("--train_freq_print", type=int, default=100, help="Print frequency")

# === S3 config ===
parser.add_argument("--s3_bucket", type=str, default="", help="S3 bucket")
parser.add_argument("--upload_to_s3", type=str2bool, default=False, help="Auto-upload when done")

args = parser.parse_args()

# Expand paths
curr_dir = str(Path(args.curr_dir).expanduser())
git_dir = str(Path(args.git_dir).expanduser())
run_name = args.run_name
encoder_path = args.encoder_path if args.encoder_path else f"{curr_dir}/models/encoders_{run_name}.pth"
encoder_path = str(Path(encoder_path).expanduser())
log_wandb = args.log_wandb
gpu_seed = args.gpu_seed

model_dim = args.model_dim
VSA_dim = args.VSA_dim
max_digits = args.max_digits
possible_problems = args.possible_problems

train_data_rounds = args.train_data_rounds
val_data_rounds = args.val_data_rounds
test_data_rounds = args.test_data_rounds
save_frequency = args.save_frequency
n_samples = args.n_samples
max_batch_size = args.max_batch_size
layer_numbers = torch.tensor(args.layer_numbers)
restrict_train_dataset = args.restrict_train_dataset
restrict_val_dataset = args.restrict_val_dataset
restrict_test_dataset = args.restrict_test_dataset
tokens_to_keep = args.tokens_to_keep if args.tokens_to_keep == "all" else int(args.tokens_to_keep)
complexity = args.complexity
problem_type = args.problem_type

encoder_decoder_batch_size = args.encoder_decoder_batch_size
decoding_training_epochs = args.decoding_epochs
decoding_learning_rate = args.decoding_learning_rate
train_freq_print = args.train_freq_print

# Learning rate schedule
decoding_learning_rate_reduction_factors = {50: 0.5, 100: 0.5, 250: 0.1, 500: 0.4}

######################################################

sys.path.insert(0, git_dir)

from llama.encoder_decoder_networks import Decoder
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
        project="Symbolic LLM - Train Decoders",
        name=run_name,
    )

print("=" * 60)
print("DECODER TRAINING SCRIPT")
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
    print(f"ERROR: Symbolic Engine not found at {vsa_path}")
    print("Run generate_data.py first")
    sys.exit(1)

# Load encoders
if not os.path.exists(encoder_path):
    print(f"ERROR: Encoder model not found: {encoder_path}")
    print("Run train_encoders.py first")
    sys.exit(1)

print(f"Loading encoders from: {encoder_path}")
encoders = torch.load(encoder_path, weights_only=False)

# Freeze encoders
for n in range(len(layer_numbers)):
    for param in encoders[n].parameters():
        param.requires_grad = False
print("Encoders loaded and frozen.")

# Data directory
save_dir = f"gathered_data_{run_name}"
if not os.path.exists(save_dir):
    print(f"ERROR: Data directory not found: {save_dir}")
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

testing_encoder_data_loaders = generate_data_loaders(
    mode='test', save_dir=save_dir, data_rounds=test_data_rounds, n_samples=n_samples, df_subset=None,
    save_frequency=save_frequency, layer_numbers=layer_numbers, restrict_dataset=restrict_test_dataset,
    tokens_to_keep=tokens_to_keep, batch_size=encoder_decoder_batch_size, gpu_seed=gpu_seed, verbose=True
)
print("Testing data loaders created.")

# Initialize decoders
print("\nInitializing decoders...")
decoders = torch.nn.ModuleList()
for layer_id in layer_numbers:
    layer_decoder = Decoder(layer_id, VSA_dim, model_dim).to(device)
    decoders.append(layer_decoder)

print(f"Trainable parameters per decoder: {count_trainable_parameters(decoders[0])}")

# Training setup
decoding_optimizers = [optim.Adam(decoders[n].parameters(), lr=decoding_learning_rate) for n in range(len(layer_numbers))]
criterion = nn.MSELoss()
decoding_losses = np.zeros((len(layer_numbers), decoding_training_epochs))
val_decoding_losses = np.zeros((len(layer_numbers), decoding_training_epochs))
decoding_running_losses = np.zeros((len(layer_numbers)))

# Training loop
print("\n" + "=" * 60)
print("TRAINING DECODERS")
print("=" * 60)

for j in range(decoding_training_epochs):
    # Learning rate schedule
    if j in decoding_learning_rate_reduction_factors.keys():
        for n in range(len(layer_numbers)):
            for param_group in decoding_optimizers[n].param_groups:
                param_group['lr'] = param_group['lr'] * decoding_learning_rate_reduction_factors[j]
        print(f"Learning rate reduced by factor {decoding_learning_rate_reduction_factors[j]}")

    for n, n_layer in enumerate(layer_numbers):
        decoding_running_loss = 0
        total_norm = 0.0

        for batch_idx, (data, labels) in enumerate(training_encoder_data_loaders[n]):
            latent_representation = encoders[n](data)
            predicted_hidden_state = decoders[n](latent_representation)

            if tokens_to_keep != 1:
                target = data[:, -1, :]
            else:
                target = data

            loss = torch.sqrt(criterion(predicted_hidden_state, target))
            loss.backward()

            tn = 0
            for p in decoders[n].parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    tn += param_norm.item() ** 2
            total_norm += tn ** 0.5

            decoding_optimizers[n].step()
            decoding_optimizers[n].zero_grad()
            decoding_running_loss += loss.item()

        decoding_running_loss /= (batch_idx + 1)
        decoding_running_losses[n] = decoding_running_loss
        total_norm /= (batch_idx + 1)
        decoding_losses[n][j] = decoding_running_loss

        # Validation
        val_decoding_running_loss = 0
        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(validation_encoder_data_loaders[n]):
                latent_representation = encoders[n](data)
                predicted_hidden_state = decoders[n](latent_representation)

                if tokens_to_keep != 1:
                    target = data[:, -1, :]
                else:
                    target = data

                loss = torch.sqrt(criterion(predicted_hidden_state, target))
                val_decoding_running_loss += loss.item()

        val_decoding_running_loss /= (batch_idx + 1)
        val_decoding_losses[n][j] = val_decoding_running_loss

    if not j % train_freq_print and j:
        print(f"Epoch {j}: Train Loss: {decoding_running_losses.mean():.4f}, Val Loss: {val_decoding_losses[:,j].mean():.4f}, Grad Norm: {total_norm:.4f}")

# Plot results
print("\nGenerating plots...")

plt.figure(figsize=(10, 6))
plt.plot(decoding_losses.mean(axis=0), marker=".", label='Training')
plt.plot(val_decoding_losses.mean(axis=0), marker=".", label='Validation')
plt.title("Average Decoder RMSE Loss Per Epoch")
plt.ylabel("RMSE")
plt.xlabel("Epoch")
plt.legend()
plt.savefig(f"decoder_loss_epochs_{run_name}.png", dpi=150, bbox_inches='tight')
if log_wandb:
    wandb.log({"Average Decoder RMSE Loss Per Epoch": wandb.Image(plt)})
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(decoding_losses.T[-1], marker=".", label='Training')
plt.plot(val_decoding_losses.T[-1], marker=".", label='Validation')
plt.title("Final Decoder RMSE Loss vs Layer Number")
plt.ylabel("RMSE")
plt.xlabel("Layer Number")
plt.legend()
plt.savefig(f"decoder_loss_layers_{run_name}.png", dpi=150, bbox_inches='tight')
if log_wandb:
    wandb.log({"Final Decoder RMSE Loss vs Layer Number": wandb.Image(plt)})
plt.close()

# Save models
os.makedirs(f"{curr_dir}/models", exist_ok=True)
torch.save(decoders.state_dict(), f"{curr_dir}/models/decoders_state_dict_{run_name}.pth")
torch.save(decoders, f"{curr_dir}/models/decoders_{run_name}.pth")
print(f"\nSaved: {curr_dir}/models/decoders_{run_name}.pth")

# Run error statistics on test data
print("\n" + "=" * 60)
print("RUNNING ERROR STATISTICS ON TEST DATA")
print("=" * 60)

errors = np.zeros(len(layer_numbers))
lowest_error_layer = 0
lowest_error = complexity + 1
exponents = torch.tensor([10 ** i for i in range(SE.max_digits)])

with torch.no_grad():
    for n, layer in enumerate(layer_numbers):
        e = 0
        for batch_idx, (data, labels) in enumerate(testing_encoder_data_loaders[n]):
            pred = encoders[n](data)
            decoded_n1 = (SE.decode_digits(pred.type_as(SE.vectors[SE.VSA_n1]), SE.VSA_n1) * exponents).sum(axis=1)
            decoded_n2 = (SE.decode_digits(pred.type_as(SE.vectors[SE.VSA_n2]), SE.VSA_n2) * exponents).sum(axis=1)
            actual_n1 = (SE.decode_digits(labels.type_as(SE.vectors[SE.VSA_n1]), SE.VSA_n1) * exponents).sum(axis=1)
            actual_n2 = (SE.decode_digits(labels.type_as(SE.vectors[SE.VSA_n2]), SE.VSA_n2) * exponents).sum(axis=1)

            batch_error = (SE.digit_error(decoded_n1, actual_n1) + SE.digit_error(decoded_n2, actual_n2)) / 2
            batch_error = batch_error * len(data) / (test_data_rounds * max_batch_size)
            e += batch_error.float().mean().item()

        errors[n] = e
        if e < lowest_error:
            lowest_error_layer = layer
            lowest_error = e

        print(f"Layer {layer.item()}: Error = {e:.4f}")

print(f"\nBest layer: {lowest_error_layer.item()} with error: {lowest_error:.4f}")

# Plot error vs layer
plt.figure(figsize=(10, 6))
x_ticks = np.arange(layer_numbers.cpu()[0].item(), layer_numbers.cpu()[0].item() + len(layer_numbers), 1)
plt.plot(x_ticks, errors, marker=".")
plt.xticks(x_ticks, rotation=75)
plt.title("Error of Decoded Numbers (Testing Data)")
plt.ylabel("Average Number of Incorrectly Decoded Digits")
plt.xlabel("Layer Number")
plt.savefig(f"error_vs_layer_{run_name}.png", dpi=150, bbox_inches='tight')
if log_wandb:
    wandb.log({"Error of Decoded Numbers (Testing Data)": wandb.Image(plt)})
plt.close()

# Upload to S3 if configured
if args.upload_to_s3 and args.s3_bucket:
    import subprocess
    print(f"\nUploading to S3...")
    subprocess.run(["aws", "s3", "cp", f"{curr_dir}/models/decoders_{run_name}.pth",
                    f"s3://{args.s3_bucket}/models/decoders_{run_name}.pth"])
    subprocess.run(["aws", "s3", "cp", f"{curr_dir}/models/decoders_state_dict_{run_name}.pth",
                    f"s3://{args.s3_bucket}/models/decoders_state_dict_{run_name}.pth"])
    print("Upload complete.")

print("\n" + "=" * 60)
print("DECODER TRAINING COMPLETE")
print("=" * 60)
print(f"\nBest layer for intervention: {lowest_error_layer.item()}")
print(f"\nNext step: Run fine_tune_decoders.py for end-to-end fine-tuning")
print(f"  python fine_tune_decoders.py --run_name {run_name} --encoder_path models/encoders_{run_name}.pth --decoder_path models/decoders_{run_name}.pth")

if log_wandb:
    wandb.finish()