"""
Script 1: Data Generation for SageMaker
Generates hidden states and VSA representations from frozen LLaMA model.

Usage:
    python generate_data.py --run_name experiment_1 --train_data_rounds 10000

After completion, upload to S3:
    aws s3 sync gathered_data_{run_name}/ s3://your-bucket/gathered_data_{run_name}/
"""

import json
import torch
import numpy as np
import random
import os
import pandas as pd
import sys
import math
import argparse
import yaml
from pathlib import Path
import datetime
import wandb

from typing import List, Optional

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


parser = argparse.ArgumentParser(description="Generate Training Data for Encoders/Decoders")

# === Required ===
parser.add_argument("--run_name", type=str, default=curr_date, help="Name of run")
parser.add_argument("--master_port", type=int, default=29500, help="Port for distributed init")

# === Path config ===
parser.add_argument("--curr_dir", type=str, default="~/Neurosymbolic-LLM/Programs", help="Path to program root")
parser.add_argument("--git_dir", type=str, default="~/Neurosymbolic-LLM", help="Path to project Git root")
parser.add_argument("--ckpt_dir", type=str, default="~/.llama/checkpoints/Llama3.1-8B-Instruct/original", help="Path to LLM checkpoint")
parser.add_argument("--tokenizer_path", type=str, default="~/.llama/checkpoints/Llama3.1-8B-Instruct/original/tokenizer.model", help="Path to tokenizer")
parser.add_argument("--log_wandb", type=str2bool, default=False, help="Log to wandb")

# === Model config ===
parser.add_argument("--max_seq_len", type=int, default=512, help="Max sequence length")
parser.add_argument("--max_batch_size", type=int, default=4, help="Batch size for LLM")
parser.add_argument("--model_parallel_size", type=int, default=1, help="Model parallelism")

# === Symbolic Engine ===
parser.add_argument("--max_digits", type=int, default=5, help="Max digits to encode")
parser.add_argument("--VSA_dim", type=int, default=2048, help="VSA dimensionality")
parser.add_argument("--possible_problems", nargs="+", type=str,
                    default=["addition", "subtraction", "multiplication", "division", "modulo", "gcd", "lcm", "power"],
                    help="Problem types")

# === Data generation ===
parser.add_argument("--train_data_rounds", type=int, default=10000, help="Training queries")
parser.add_argument("--val_data_rounds", type=int, default=500, help="Validation queries")
parser.add_argument("--test_data_rounds", type=int, default=500, help="Test queries")
parser.add_argument("--save_frequency", type=int, default=100, help="Save every N batches")
parser.add_argument("--complexity", type=int, default=2, help="Problem complexity")
parser.add_argument("--n_samples", type=int, default=4, help="Samples per forward pass")
parser.add_argument("--problem_type", nargs="+", type=str, default=["addition"], help="Problem types to use")

# === Token handling ===
parser.add_argument("--tokens_to_keep", type=str, default="1", help="Tokens to keep")
parser.add_argument("--calculate_end_index", type=str2bool, default=True, help="Cut at end of prompt")

# === S3 config ===
parser.add_argument("--s3_bucket", type=str, default="", help="S3 bucket for saving data")
parser.add_argument("--upload_to_s3", type=str2bool, default=False, help="Auto-upload to S3 when done")

args = parser.parse_args()

# Expand paths
curr_dir = str(Path(args.curr_dir).expanduser())
git_dir = str(Path(args.git_dir).expanduser())
ckpt_dir = str(Path(args.ckpt_dir).expanduser())
tokenizer_path = str(Path(args.tokenizer_path).expanduser())
run_name = args.run_name
master_port = str(args.master_port)
log_wandb = args.log_wandb

max_seq_len = args.max_seq_len
max_batch_size = args.max_batch_size
model_parallel_size = args.model_parallel_size

max_digits = args.max_digits
VSA_dim = args.VSA_dim
possible_problems = args.possible_problems

train_data_rounds = args.train_data_rounds
val_data_rounds = args.val_data_rounds
test_data_rounds = args.test_data_rounds
save_frequency = args.save_frequency
complexity = args.complexity
n_samples = args.n_samples
problem_type = args.problem_type

tokens_to_keep = args.tokens_to_keep if args.tokens_to_keep == "all" else int(args.tokens_to_keep)
calculate_end_index = args.calculate_end_index

######################################################

sys.path.insert(0, git_dir)

from llama.vsa_engine import SymbolicEngine
from llama.utilities import generate_and_save_data
from llama.device_utils import (
    get_device,
    get_device_type,
    setup_device_environment,
    print_device_info,
)
from llama import Llama

######################################################

if log_wandb:
    wandb.finish()
    wandb.init(
        project="Symbolic LLM - Generate Encoder Input Data",
        name=run_name,
    )

print("=" * 60)
print("DATA GENERATION SCRIPT")
print("=" * 60)
print(f"Run: {run_name}")

# Device setup
device = get_device()
device_type = get_device_type()
print_device_info()
setup_device_environment()

# Distributed setup
os.environ['RANK'] = "0"
os.environ['WORLD_SIZE'] = "1"
os.environ['MASTER_ADDR'] = "localhost"
os.environ['MASTER_PORT'] = master_port
os.environ['LOCAL_RANK'] = "0"

# Build LLaMA
print("Loading LLaMA model...")
generator = Llama.build(
    ckpt_dir=ckpt_dir,
    tokenizer_path=tokenizer_path,
    max_seq_len=max_seq_len,
    max_batch_size=max_batch_size,
)

# Freeze model
for param in generator.model.parameters():
    param.requires_grad = False
print("Model loaded and frozen.")

# Load or create Symbolic Engine
possible_problems_str = "_".join(possible_problems)
vsa_path = f"{curr_dir}/VSA_library/symbolic_engine_VSA_dim_{VSA_dim}_max_digits_{max_digits}_problem_types_{possible_problems_str}.pt"

if os.path.exists(vsa_path):
    SE = torch.load(vsa_path, weights_only=False)
    print("Using pre-existing Symbolic Engine")
else:
    os.makedirs(f"{curr_dir}/VSA_library", exist_ok=True)
    SE = SymbolicEngine(VSA_dim=VSA_dim, max_digits=max_digits, possible_problems=possible_problems, curr_dir=curr_dir)
    torch.save(SE, vsa_path)
    print("Created new Symbolic Engine")

# Output directory
save_dir = f"gathered_data_{run_name}"
os.makedirs(save_dir, exist_ok=True)
print(f"Data will be saved to: {save_dir}")

# Generate data
print("\n" + "=" * 60)
print("GENERATING TRAINING DATA")
print("=" * 60)
generate_and_save_data(
    generator=generator, SE=SE, save_dir=save_dir,
    rounds=train_data_rounds, mode="train", save_frequency=save_frequency,
    complexity=complexity, n_samples=n_samples, problem_type=problem_type, df_subset=None,
    tokens_to_keep=tokens_to_keep, calculate_end_index=calculate_end_index, verbose=True
)
print("Training data completed.")

print("\n" + "=" * 60)
print("GENERATING VALIDATION DATA")
print("=" * 60)
generate_and_save_data(
    generator=generator, SE=SE, save_dir=save_dir,
    rounds=val_data_rounds, mode="val", save_frequency=save_frequency,
    complexity=complexity, n_samples=n_samples, problem_type=problem_type, df_subset=None,
    tokens_to_keep=tokens_to_keep, calculate_end_index=calculate_end_index, verbose=True
)
print("Validation data completed.")

print("\n" + "=" * 60)
print("GENERATING TEST DATA")
print("=" * 60)
generate_and_save_data(
    generator=generator, SE=SE, save_dir=save_dir,
    rounds=test_data_rounds, mode="test", save_frequency=save_frequency,
    complexity=complexity, n_samples=n_samples, problem_type=problem_type, df_subset=None,
    tokens_to_keep=tokens_to_keep, calculate_end_index=calculate_end_index, verbose=True
)
print("Test data completed.")

# Upload to S3 if configured
if args.upload_to_s3 and args.s3_bucket:
    import subprocess
    print(f"\nUploading to S3: s3://{args.s3_bucket}/{save_dir}/")
    subprocess.run(["aws", "s3", "sync", save_dir, f"s3://{args.s3_bucket}/{save_dir}/"])
    print("Upload complete.")

print("\n" + "=" * 60)
print("DATA GENERATION COMPLETE")
print("=" * 60)
print(f"Data saved to: {save_dir}/")
print("\nNext step: Run train_encoders.py")
print(f"  python train_encoders.py --run_name {run_name}")

if log_wandb:
    wandb.finish()