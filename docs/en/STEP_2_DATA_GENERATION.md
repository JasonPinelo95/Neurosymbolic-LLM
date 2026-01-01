# Step 2: Data Generation (Stage 1)

This document explains the data generation phase where we collect hidden states from the frozen LLaMA model and create corresponding VSA representations.

---

## Table of Contents

1. [Overview](#overview)
2. [Flow Diagram](#flow-diagram)
3. [Prerequisites](#prerequisites)
4. [Code Walkthrough](#code-walkthrough)
5. [Data Output Format](#data-output-format)
6. [Configuration Options](#configuration-options)
7. [Verification](#verification)
8. [Troubleshooting](#troubleshooting)

---

## Overview

### What This Step Does

This step runs the frozen LLaMA model on randomly generated math problems and:
1. **Records hidden states** at all 33 layers for each problem
2. **Generates VSA representations** of the input numbers and problem type
3. **Saves paired data** (hidden states ↔ VSA vectors) to disk

This paired data is used in Step 3 to train the encoder and decoder networks.

### Conceptual Understanding

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Math Problem   │────▶│   LLaMA Model   │────▶│  Hidden States  │
│  "427 + 581"    │     │   (Frozen)      │     │  h ∈ R^4096     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                                               │
         │                                               │
         ▼                                               ▼
┌─────────────────┐                           ┌─────────────────┐
│  VSA Engine     │                           │  Paired Dataset │
│  Encode(427,581)│──────────────────────────▶│  (h, VSA)       │
└─────────────────┘                           └─────────────────┘
```

### Why This Is Needed

The encoder network must learn to map:
- **Input**: Hidden state `h` (4096-dimensional)
- **Output**: VSA representation `v` (2048-dimensional)

We need thousands of examples of `(h, v)` pairs to train this mapping.

---

## Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        STEP 2: DATA GENERATION                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │     Initialize Components     │
                    │  ┌─────────────────────────┐  │
                    │  │ 1. Load LLaMA Model     │  │
                    │  │ 2. Create SymbolicEngine│  │
                    │  │ 3. Set up directories   │  │
                    │  └─────────────────────────┘  │
                    └───────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │   For each round (10,000x):   │
                    │  ┌─────────────────────────┐  │
                    │  │ 1. Generate random      │  │
                    │  │    math problem         │  │
                    │  │    (e.g., 427 + 581)    │  │
                    │  └─────────────────────────┘  │
                    │              │                │
                    │              ▼                │
                    │  ┌─────────────────────────┐  │
                    │  │ 2. Create dialog prompt │  │
                    │  │    for LLaMA            │  │
                    │  └─────────────────────────┘  │
                    │              │                │
                    │              ▼                │
                    │  ┌─────────────────────────┐  │
                    │  │ 3. Run forward pass     │  │
                    │  │    Record h at L layers │  │
                    │  └─────────────────────────┘  │
                    │              │                │
                    │              ▼                │
                    │  ┌─────────────────────────┐  │
                    │  │ 4. Generate VSA vector  │  │
                    │  │    SE.generate_VSA()    │  │
                    │  └─────────────────────────┘  │
                    │              │                │
                    │              ▼                │
                    │  ┌─────────────────────────┐  │
                    │  │ 5. Save (h, VSA) pair   │  │
                    │  │    every N rounds       │  │
                    │  └─────────────────────────┘  │
                    └───────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │       Output Files:           │
                    │  gathered_data_{run_name}/    │
                    │  ├── train_hidden_*.pt        │
                    │  ├── train_vsa_*.pt           │
                    │  ├── val_hidden_*.pt          │
                    │  ├── val_vsa_*.pt             │
                    │  ├── test_hidden_*.pt         │
                    │  └── test_vsa_*.pt            │
                    └───────────────────────────────┘
```

---

## Prerequisites

1. **Completed Step 1**: LLaMA model downloaded
2. **GPU with CUDA**: Required for forward pass
3. **Sufficient disk space**: ~50GB for default settings
4. **Memory**: ~20GB GPU RAM, ~32GB system RAM

---

## Code Walkthrough

### Main Script Entry Point

**File**: [Programs/train_encoders_and_decoders.py](../../Programs/train_encoders_and_decoders.py)

**Command**:
```bash
uv run python ~/Neurosymbolic-LLM/Programs/train_encoders_and_decoders.py \
    --run_name my_experiment \
    --generate_data 1
```

### Step-by-Step Code Execution

#### 1. Device Detection and Environment Setup (Lines 222-237)

```python
# Auto-detect device (CUDA, MPS, or CPU)
device = get_device()
device_type = get_device_type()
print_device_info()

# Setup device-specific environment variables
setup_device_environment()

if generate_data:
    os.environ['RANK'] = "0"
    os.environ['WORLD_SIZE'] = "1"
    os.environ['MASTER_ADDR'] = "127.0.0.2"
    os.environ['MASTER_PORT'] = master_port
    os.environ['LOCAL_RANK']  = "0"
```

**Purpose**: Auto-detect available hardware (CUDA GPU, Apple Silicon MPS, or CPU) and configure the distributed training environment accordingly.

**Supported Devices**:
- **CUDA**: NVIDIA GPUs (uses `nccl` backend)
- **MPS**: Apple Silicon (M1/M2/M3) (uses `gloo` backend)
- **CPU**: Fallback for systems without GPU (uses `gloo` backend)

#### 2. Load LLaMA Model (Lines 229-237)

```python
generator = Llama.build(
    ckpt_dir=ckpt_dir,
    tokenizer_path=tokenizer_path,
    max_seq_len=max_seq_len,
    max_batch_size=max_batch_size,
)

# Freeze all parameters
for param in generator.model.parameters():
    param.requires_grad = False
```

**Purpose**: Load the pre-trained LLaMA model and ensure weights are frozen.

#### 3. Initialize Symbolic Engine (Lines 248-256)

```python
possible_problems_str = "_".join(possible_problems)

if os.path.exists(f"{curr_dir}/VSA_library/symbolic_engine_..."):
    SE = torch.load(f"{curr_dir}/VSA_library/symbolic_engine_...", weights_only=False)
else:
    SE = SymbolicEngine(
        VSA_dim=VSA_dim,
        max_digits=max_digits,
        possible_problems=possible_problems,
        curr_dir=curr_dir
    )
    torch.save(SE, f"{curr_dir}/VSA_library/symbolic_engine_...")
```

**Purpose**: Create or load the Symbolic Engine with consistent VSA vocabulary vectors.

#### 4. Generate and Save Data (Lines 311-328)

```python
if generate_data:
    # Training data
    generate_and_save_data(
        generator=generator,
        SE=SE,
        save_dir=save_dir,
        rounds=train_data_rounds,      # 10,000 by default
        mode="train",
        save_frequency=save_frequency,  # 50 by default
        complexity=complexity,
        n_samples=n_samples,
        problem_type=problem_type,
        tokens_to_keep=tokens_to_keep,
        calculate_end_index=calculate_end_index,
        verbose=True
    )

    # Repeat for validation and test sets...
```

### Core Data Generation Function

**File**: [llama/utilities.py](../../llama/utilities.py) - `generate_and_save_data()`

```python
def generate_and_save_data(generator, SE, save_dir, rounds, mode, save_frequency, ...):
    hidden_states_buffer = []
    vsa_buffer = []

    for round_idx in range(rounds):
        # 1. Generate random math problem
        dialogs, x, y, problem_type = generate_dialog(
            complexity=complexity,
            samples=n_samples,
            problem_type=problem_type
        )

        # 2. Run LLaMA forward pass and collect hidden states
        prompt_tokens = generator.parse_chat(dialogs)
        with torch.no_grad():
            logits, h_stack, h = generator.model.forward(
                tokens=prompt_tokens,
                start_pos=0
            )
        # h_stack shape: (33, batch, seq_len, 4096) - all layer hidden states

        # 3. Extract relevant hidden state (last token)
        relevant_h = h_stack[:, :, -tokens_to_keep:, :]

        # 4. Generate corresponding VSA
        vsa = SE.generate_VSA(
            torch.tensor(x),
            torch.tensor(y),
            problem_types=[problem_type] * n_samples
        )

        # 5. Buffer and save periodically
        hidden_states_buffer.append(relevant_h)
        vsa_buffer.append(vsa)

        if (round_idx + 1) % save_frequency == 0:
            save_batch(hidden_states_buffer, vsa_buffer, save_dir, mode, batch_idx)
            hidden_states_buffer = []
            vsa_buffer = []
```

### Dialog Generation

**File**: [llama/utilities.py](../../llama/utilities.py) - `generate_dialog()`

```python
def generate_dialog(complexity, samples, problem_type, ...):
    dialogs = []
    x_values = []
    y_values = []

    for _ in range(samples):
        # Generate random numbers based on complexity
        max_val = 10 ** complexity  # complexity=2 → max 1000
        x = random.randint(0, max_val)
        y = random.randint(1, max_val)  # y > 0 for division/modulo

        # Create question based on problem type
        if problem_type == "addition":
            question = f"What is {x} + {y}?"
        elif problem_type == "multiplication":
            question = f"What is {x} * {y}?"
        # ... etc for other problem types

        # Format as LLaMA chat dialog
        dialog = [
            {"role": "system", "content": "Answer with just the number."},
            {"role": "user", "content": question}
        ]

        dialogs.append(dialog)
        x_values.append(x)
        y_values.append(y)

    return dialogs, x_values, y_values, problem_type
```

---

## Data Output Format

### Directory Structure

```
gathered_data_{run_name}/
├── train_hidden_0.pt      # Hidden states batch 0
├── train_hidden_1.pt      # Hidden states batch 1
├── ...
├── train_hidden_199.pt    # 10000/50 = 200 batches
├── train_vsa_0.pt         # VSA vectors batch 0
├── train_vsa_1.pt         # VSA vectors batch 1
├── ...
├── train_vsa_199.pt
├── val_hidden_*.pt        # Validation set
├── val_vsa_*.pt
├── test_hidden_*.pt       # Test set
└── test_vsa_*.pt
```

### Tensor Shapes

| File Type | Shape | Description |
|-----------|-------|-------------|
| `*_hidden_*.pt` | `(33, batch*save_freq, tokens_to_keep, 4096)` | Hidden states per layer |
| `*_vsa_*.pt` | `(batch*save_freq, 2048)` | VSA representations |

### Example Sizes (Default Config)

| Dataset | Rounds | Batches | Total Samples |
|---------|--------|---------|---------------|
| Train | 10,000 | 200 | 20,000 (×2 samples/round) |
| Val | 100 | 2 | 200 |
| Test | 1,000 | 20 | 2,000 |

---

## Configuration Options

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `train_data_rounds` | 10000 | Number of training problems |
| `val_data_rounds` | 100 | Number of validation problems |
| `test_data_rounds` | 1000 | Number of test problems |
| `n_samples` | 2 | Problems per forward pass |
| `save_frequency` | 50 | Save every N rounds |
| `complexity` | 2 | Max digits (10^complexity) |
| `tokens_to_keep` | 1 | Hidden state tokens to save |
| `problem_type` | [list] | Which math operations |

### Problem Types Available

```yaml
problem_type:
  - multiplication
  - modulo
  - gcd
  - lcm
  - square_mod
  - bitwise_and
  - bitwise_xor
  - bitwise_or
```

Note: `addition` and `division` are in `possible_problems` but not in default `problem_type` for training.

### Full Command with Options

```bash
uv run python ~/Neurosymbolic-LLM/Programs/train_encoders_and_decoders.py \
    --run_name my_experiment \
    --generate_data 1 \
    --train_data_rounds 10000 \
    --val_data_rounds 100 \
    --test_data_rounds 1000 \
    --n_samples 2 \
    --save_frequency 50 \
    --complexity 2 \
    --tokens_to_keep 1 \
    --problem_type multiplication modulo gcd lcm
```

---

## Verification

### 1. Check Output Directory

```bash
ls -la gathered_data_my_experiment/
```

Expected: Multiple `.pt` files for train/val/test splits.

### 2. Verify File Counts

```bash
# Should match: train_data_rounds / save_frequency
ls gathered_data_my_experiment/train_hidden_*.pt | wc -l
# Expected: 200 (for 10000 rounds, save_frequency=50)
```

### 3. Check Tensor Shapes

```bash
uv run python -c "
import torch

# Load a batch
hidden = torch.load('gathered_data_my_experiment/train_hidden_0.pt')
vsa = torch.load('gathered_data_my_experiment/train_vsa_0.pt')

print(f'Hidden shape: {hidden.shape}')
# Expected: (33, 100, 1, 4096) for save_freq=50, n_samples=2, tokens_to_keep=1

print(f'VSA shape: {vsa.shape}')
# Expected: (100, 2048)
"
```

### 4. Verify VSA Content

```bash
uv run python -c "
import torch

# Load symbolic engine
SE = torch.load('VSA_library/symbolic_engine_...pt', weights_only=False)

# Load VSA batch
vsa = torch.load('gathered_data_my_experiment/train_vsa_0.pt')

# Decode a VSA to verify it encodes numbers correctly
vsa_sample = vsa[0:1]  # First sample
decoded_n1 = SE.decode_VSA(vsa_sample, SE.VSA_n1)
decoded_n2 = SE.decode_VSA(vsa_sample, SE.VSA_n2)
problem_type, score, _ = SE.decode_problem_type(vsa_sample)

print(f'Decoded: n1={decoded_n1}, n2={decoded_n2}, type={problem_type}')
"
```

---

## Troubleshooting

### Error: "CUDA out of memory" / "MPS out of memory"

**Cause**: Batch size too large for GPU.

**Solution**:
```bash
# Reduce batch size
uv run python train_encoders_and_decoders.py \
    --generate_data 1 \
    --max_batch_size 1 \
    --n_samples 1
```

### Apple Silicon (MPS) Specific Issues

**Symptom**: Slow performance or operations falling back to CPU.

**Solution**:
```bash
# Set MPS memory limit (allows full memory usage)
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# Run with reduced batch size (MPS has less VRAM than typical NVIDIA GPUs)
uv run python train_encoders_and_decoders.py \
    --generate_data 1 \
    --max_batch_size 1 \
    --n_samples 1
```

**Note**: Some PyTorch operations may not be fully optimized for MPS yet. If you encounter issues, the code will automatically fall back to CPU for unsupported operations.

### Error: "No checkpoint files found"

**Cause**: LLaMA model path incorrect.

**Solution**:
```bash
# Verify path and use correct one
ls ~/.llama/checkpoints/Llama3.1-8B-Instruct/

uv run python train_encoders_and_decoders.py \
    --generate_data 1 \
    --ckpt_dir ~/.llama/checkpoints/Llama3.1-8B-Instruct
```

### Error: "Disk full"

**Cause**: Not enough space for output files.

**Solution**:
```bash
# Increase save_frequency to save fewer files
uv run python train_encoders_and_decoders.py \
    --generate_data 1 \
    --save_frequency 100  # Half as many files

# Or reduce training rounds
uv run python train_encoders_and_decoders.py \
    --generate_data 1 \
    --train_data_rounds 5000
```

### Process Killed / Segfault

**Cause**: System RAM exhausted.

**Solution**:
```bash
# Increase save frequency (less data in memory)
uv run python train_encoders_and_decoders.py \
    --generate_data 1 \
    --save_frequency 25  # Save more frequently
```

---

## Estimated Resources

### Time

| GPU | Rounds | Estimated Time |
|-----|--------|----------------|
| A100 | 10,000 | ~2-3 hours |
| RTX 4090 | 10,000 | ~3-4 hours |
| RTX 3090 | 10,000 | ~4-5 hours |

### Disk Space

| Config | Approximate Size |
|--------|-----------------|
| Default (10k train) | ~50 GB |
| Reduced (5k train) | ~25 GB |
| Minimal (1k train) | ~5 GB |

---

## Next Step

Once data generation completes, proceed to **[Step 3: Encoder/Decoder Training](STEP_3_ENCODER_DECODER_TRAINING.md)**.

---

## Code References

| File | Lines | Purpose |
|------|-------|---------|
| [train_encoders_and_decoders.py](../../Programs/train_encoders_and_decoders.py) | 311-328 | Main data generation calls |
| [llama/utilities.py](../../llama/utilities.py) | - | `generate_and_save_data()` function |
| [llama/utilities.py](../../llama/utilities.py) | - | `generate_dialog()` function |
| [llama/vsa_engine.py](../../llama/vsa_engine.py) | 253-316 | `generate_VSA()` method |
| [llama/model.py](../../llama/model.py) | 283-312 | `forward()` returns h_stack |
