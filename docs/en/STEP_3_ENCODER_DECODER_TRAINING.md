# Step 3: Encoder/Decoder Training (Stage 2)

This document explains the training of encoder and decoder networks that bridge the gap between LLM hidden states and VSA representations.

---

## Table of Contents

1. [Overview](#overview)
2. [Flow Diagram](#flow-diagram)
3. [Prerequisites](#prerequisites)
4. [Code Walkthrough](#code-walkthrough)
5. [Training Process](#training-process)
6. [Model Architecture](#model-architecture)
7. [Configuration Options](#configuration-options)
8. [Verification](#verification)
9. [Troubleshooting](#troubleshooting)

---

## Overview

### What This Step Does

This step trains two types of neural networks:

1. **Encoder**: Maps LLM hidden states → VSA representations
   - Input: `h ∈ R^4096` (LLaMA hidden state)
   - Output: `v ∈ R^2048` (VSA vector)
   - Loss: MSE between predicted VSA and ground truth VSA

2. **Decoder**: Maps VSA representations → LLM hidden states
   - Input: `v ∈ R^2048` (VSA vector)
   - Output: `h ∈ R^4096` (reconstructed hidden state)
   - Loss: MSE between reconstructed and original hidden state

### Conceptual Understanding

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TRAINING OBJECTIVE                                  │
└─────────────────────────────────────────────────────────────────────────────┘

  Hidden State (h)              VSA (v)                 Hidden State (h')
  ┌───────────┐            ┌───────────┐              ┌───────────┐
  │           │  Encoder   │           │   Decoder    │           │
  │  R^4096   │───────────▶│  R^2048   │─────────────▶│  R^4096   │
  │           │            │           │              │           │
  └───────────┘            └───────────┘              └───────────┘
       │                        │                          │
       │                        │                          │
       ▼                        ▼                          ▼
  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
  │ From LLaMA      │    │ Target: Ground  │    │ Should ≈ h      │
  │ forward pass    │    │ truth VSA       │    │ (reconstruction)│
  └─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Why Train Separately?

The encoder/decoder are trained with MSE loss on paired data before integrating with the LLM. This:
- Provides stable initialization for end-to-end fine-tuning
- Allows analysis of encoding quality per layer
- Identifies the optimal layer for intervention (Layer 17)

---

## Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                  STEP 3: ENCODER/DECODER TRAINING                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │      Load Training Data       │
                    │  ┌─────────────────────────┐  │
                    │  │ gathered_data/          │  │
                    │  │ ├── train_hidden_*.pt   │  │
                    │  │ └── train_vsa_*.pt      │  │
                    │  └─────────────────────────┘  │
                    └───────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │   Create DataLoaders          │
                    │   (one per layer: 0-32)       │
                    └───────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │   Initialize Networks         │
                    │  ┌─────────────────────────┐  │
                    │  │ 33 Encoders (per layer) │  │
                    │  │ 33 Decoders (per layer) │  │
                    │  └─────────────────────────┘  │
                    └───────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
        ┌─────────────────────┐       ┌─────────────────────┐
        │   ENCODER TRAINING  │       │   DECODER TRAINING  │
        │   (1000 epochs)     │       │   (1000 epochs)     │
        │  ┌───────────────┐  │       │  ┌───────────────┐  │
        │  │ For each layer│  │       │  │ For each layer│  │
        │  │   h → Enc → v̂ │  │       │  │   v → Dec → ĥ │  │
        │  │   Loss: MSE   │  │       │  │   Loss: MSE   │  │
        │  │   (v̂ vs v)    │  │       │  │   (ĥ vs h)    │  │
        │  └───────────────┘  │       │  └───────────────┘  │
        └─────────────────────┘       └─────────────────────┘
                    │                               │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │      Evaluate & Select        │
                    │  ┌─────────────────────────┐  │
                    │  │ Find best layer (17)    │  │
                    │  │ Plot loss per layer     │  │
                    │  │ Calculate digit errors  │  │
                    │  └─────────────────────────┘  │
                    └───────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │         Save Models           │
                    │  ┌─────────────────────────┐  │
                    │  │ models/encoders_*.pth   │  │
                    │  │ models/decoders_*.pth   │  │
                    │  └─────────────────────────┘  │
                    └───────────────────────────────┘
```

---

## Prerequisites

1. **Completed Step 2**: Data files in `gathered_data_{run_name}/`
2. **GPU**: Training is much faster with CUDA
3. **Memory**: ~8GB GPU RAM sufficient for training

---

## Code Walkthrough

### Main Script Entry Point

**File**: [Programs/train_encoders_and_decoders.py](../../Programs/train_encoders_and_decoders.py)

**Command**:
```bash
uv run python ~/Neurosymbolic-LLM/Programs/train_encoders_and_decoders.py \
    --run_name my_experiment \
    --generate_data 0  # 0 = train mode (not generate)
```

### Step-by-Step Code Execution

#### 1. Load Data and Create DataLoaders (Lines 331-344)

```python
if not generate_data:
    training_encoder_data_loaders = generate_data_loaders(
        mode='train',
        save_dir=save_dir,
        data_rounds=train_data_rounds,
        n_samples=n_samples,
        save_frequency=save_frequency,
        layer_numbers=layer_numbers,       # [0, 1, ..., 32]
        restrict_dataset=restrict_train_dataset,
        tokens_to_keep=tokens_to_keep,
        batch_size=encoder_decoder_batch_size,  # 512
        gpu_seed=gpu_seed,
        verbose=True
    )
```

**Purpose**: Create one DataLoader per layer containing (hidden_state, VSA) pairs.

#### 2. Initialize Encoder Networks (Lines 346-353)

```python
encoders = torch.nn.ModuleList()
for layer_id in layer_numbers:
    if tokens_to_keep == 1:
        # Single token → linear encoder
        layer_encoder = Encoder(layer_id, model_dim, VSA_dim).to(device)
    else:
        # Multiple tokens → transformer encoder
        layer_encoder = LastTokenTransformer(
            layer_id, model_dim, VSA_dim, num_layers=2, hidden_dim=512
        ).to(device)
    encoders.append(layer_encoder)
```

**Purpose**: Create 33 encoder networks (one per layer), architecture depends on `tokens_to_keep`.

#### 3. Encoder Training Loop (Lines 362-424)

```python
optimizers = [optim.Adam(encoders[n].parameters(), lr=learning_rate) for n in range(len(layer_numbers))]
criterion = nn.MSELoss()

for i in range(training_epochs):  # 1000 epochs
    # Learning rate schedule
    if i in learning_rate_reduction_factors.keys():
        for param_group in optimizers[n].param_groups:
            param_group['lr'] *= learning_rate_reduction_factors[i]

    for n, n_layer in enumerate(layer_numbers):  # For each layer 0-32
        encoders[n].train()
        running_loss = 0

        for batch_idx, (data, labels) in enumerate(training_encoder_data_loaders[n]):
            # data: hidden states [batch, 4096]
            # labels: VSA vectors [batch, 2048]

            model_pred = encoders[n](data)              # Encode h → v̂
            loss = torch.sqrt(criterion(model_pred, labels))  # RMSE(v̂, v)
            loss.backward()
            optimizers[n].step()
            optimizers[n].zero_grad()
            running_loss += loss.item()

        # Validation
        encoders.eval()
        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(validation_encoder_data_loaders[n]):
                model_pred = encoders[n](data)
                # Decode VSA to check digit accuracy
                v_digit_predictions_n1 = SE.decode_digits(model_pred, SE.VSA_n1)
                v_digit_labels_n1 = SE.decode_digits(labels, SE.VSA_n1)
```

**Key Operations**:
- Forward: `h → Encoder → v̂`
- Loss: `√MSE(v̂, v)` (RMSE)
- Validation: Decode predicted VSA and check digit accuracy

#### 4. Initialize and Train Decoder Networks (Lines 474-543)

```python
# Freeze encoders
for n, n_layer in enumerate(layer_numbers):
    for param in encoders[n].parameters():
        param.requires_grad = False

# Initialize decoders
decoders = torch.nn.ModuleList()
for layer_id in layer_numbers:
    layer_decoder = Decoder(layer_id, VSA_dim, model_dim).to(device)
    decoders.append(layer_decoder)

# Training loop
for j in range(decoding_training_epochs):  # 1000 epochs
    for n, n_layer in enumerate(layer_numbers):
        for batch_idx, (data, labels) in enumerate(training_encoder_data_loaders[n]):
            # Encode first (with frozen encoder)
            latent_representation = encoders[n](data)

            # Then decode
            predicted_hidden_state = decoders[n](latent_representation)

            # Target is original hidden state
            target = data if tokens_to_keep == 1 else data[:, -1, :]

            # Loss: reconstruct original hidden state
            loss = torch.sqrt(criterion(predicted_hidden_state, target))
            loss.backward()
            decoding_optimizers[n].step()
```

**Key Operations**:
- Forward: `h → Encoder → v̂ → Decoder → ĥ`
- Loss: `√MSE(ĥ, h)` (reconstruct original hidden state)

#### 5. Evaluation and Layer Selection (Lines 600-728)

```python
# Test encoding accuracy per layer
for n, layer in enumerate(layer_numbers):
    for batch_idx, (data, labels) in enumerate(testing_encoder_data_loaders[n]):
        pred = encoders[n](data)

        # Decode predicted VSA to numbers
        decoded_n1 = (SE.decode_digits(pred, SE.VSA_n1) * exponents).sum(axis=1)
        decoded_n2 = (SE.decode_digits(pred, SE.VSA_n2) * exponents).sum(axis=1)

        # Compare to ground truth
        actual_n1 = (SE.decode_digits(labels, SE.VSA_n1) * exponents).sum(axis=1)
        actual_n2 = (SE.decode_digits(labels, SE.VSA_n2) * exponents).sum(axis=1)

        # Calculate digit errors
        batch_error = SE.digit_error(decoded_n1, actual_n1) + SE.digit_error(decoded_n2, actual_n2)

    if e < lowest_error:
        lowest_error_layer = layer
        lowest_error = e

print(f"Best layer: {lowest_error_layer} with error: {lowest_error}")
```

**Purpose**: Find which layer has the lowest decoding error → typically Layer 17.

#### 6. Save Models (Lines 589-596)

```python
torch.save(encoders.state_dict(), f"{curr_dir}/models/encoders_state_dict_{run_name}.pth")
torch.save(encoders, f"{curr_dir}/models/encoders_{run_name}.pth")
torch.save(decoders.state_dict(), f"{curr_dir}/models/decoders_state_dict_{run_name}.pth")
torch.save(decoders, f"{curr_dir}/models/decoders_{run_name}.pth")
```

---

## Model Architecture

### Encoder (Single Token)

**File**: [llama/encoder_decoder_networks.py:7-19](../../llama/encoder_decoder_networks.py#L7-L19)

```python
class Encoder(nn.Module):
    def __init__(self, layer_id, input_dim, output_dim, bias=False, dtype=torch.bfloat16):
        super().__init__()
        self.encoder_layer = nn.Linear(input_dim, output_dim, bias=bias, dtype=dtype)
        # input_dim = 4096 (LLaMA hidden dim)
        # output_dim = 2048 (VSA dim)

    def forward(self, x):
        return self.encoder_layer(x)
```

**Parameters**: 4096 × 2048 = **8,388,608** parameters per layer

### Decoder (Single Token)

**File**: [llama/encoder_decoder_networks.py:20-32](../../llama/encoder_decoder_networks.py#L20-L32)

```python
class Decoder(nn.Module):
    def __init__(self, layer_id, input_dim, output_dim, bias=False, dtype=torch.bfloat16):
        super().__init__()
        self.decoder_layer = nn.Linear(input_dim, output_dim, bias=bias, dtype=dtype)
        # input_dim = 2048 (VSA dim)
        # output_dim = 4096 (LLaMA hidden dim)

    def forward(self, x):
        return self.decoder_layer(x)
```

**Parameters**: 2048 × 4096 = **8,388,608** parameters per layer

### Deep Variants (Optional)

```python
class Encoder_Deep(nn.Module):
    def __init__(self, layer_id, input_dim, output_dim, hidden_dim, ...):
        self.encoder_layer_1 = nn.Linear(input_dim, hidden_dim)   # 4096 → 16384
        self.encoder_layer_2 = nn.Linear(hidden_dim, output_dim)  # 16384 → 2048
```

### Transformer Variant (Multiple Tokens)

```python
class LastTokenTransformer(nn.Module):
    def __init__(self, layer_id, data_dim, output_dim, num_layers=4, num_heads=8, ...):
        self.input_proj = nn.Linear(data_dim, hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(...)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
```

---

## Training Process

### Learning Rate Schedule

| Epoch | Factor | Resulting LR |
|-------|--------|--------------|
| 0 | 1.0 | 0.001 |
| 50 | 0.5 | 0.0005 |
| 100 | 0.5 | 0.00025 |
| 250 | 0.1 | 0.000025 |
| 500 | 0.4 | 0.00001 |

### Loss Function

**RMSE** (Root Mean Square Error):
```python
loss = torch.sqrt(criterion(predicted, target))
# criterion = nn.MSELoss()
```

### Validation Metrics

1. **RMSE Loss**: Direct comparison of vectors
2. **Digit Accuracy**: Decode VSA → numbers and compare
3. **Problem Type Accuracy**: Decode problem type tag

---

## Configuration Options

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `encoder_decoder_batch_size` | 512 | Training batch size |
| `training_epochs` | 1000 | Encoder training epochs |
| `decoding_epochs` | 1000 | Decoder training epochs |
| `learning_rate` | 0.001 | Initial learning rate |
| `layer_numbers` | [0-32] | Which layers to train |

### Full Command with Options

```bash
uv run python ~/Neurosymbolic-LLM/Programs/train_encoders_and_decoders.py \
    --run_name my_experiment \
    --generate_data 0 \
    --encoder_decoder_batch_size 512 \
    --training_epochs 1000 \
    --decoding_epochs 1000 \
    --learning_rate 0.001
```

---

## Verification

### 1. Check Output Files

```bash
ls -la Programs/models/
```

Expected:
```
encoders_my_experiment.pth
encoders_state_dict_my_experiment.pth
decoders_my_experiment.pth
decoders_state_dict_my_experiment.pth
```

### 2. Verify Model Loading

```bash
uv run python -c "
import torch

encoders = torch.load('Programs/models/encoders_my_experiment.pth')
decoders = torch.load('Programs/models/decoders_my_experiment.pth')

print(f'Number of encoders: {len(encoders)}')  # Expected: 33
print(f'Number of decoders: {len(decoders)}')  # Expected: 33

# Check layer 17 specifically
print(f'Encoder 17 shape: {encoders[17].encoder_layer.weight.shape}')
# Expected: torch.Size([2048, 4096])
"
```

### 3. Check Training Plots (wandb)

If `log_wandb=1`, check Weights & Biases for:
- Average Encoder RMSE Loss Per Epoch
- Average Decoder RMSE Loss Per Epoch
- Error of Decoded Numbers vs Layer Number

### 4. Identify Best Layer

From the console output or wandb:
```
Minimum Error: X.XX and problem type error: Y at layer 17
```

---

## Troubleshooting

### Error: "FileNotFoundError: gathered_data_*"

**Cause**: Data files from Step 2 not found.

**Solution**:
```bash
# Verify data exists
ls gathered_data_my_experiment/

# If not, run Step 2 first
uv run python train_encoders_and_decoders.py --generate_data 1
```

### Error: "CUDA out of memory" / "MPS out of memory"

**Cause**: Batch size too large.

**Solution**:
```bash
uv run python train_encoders_and_decoders.py \
    --generate_data 0 \
    --encoder_decoder_batch_size 256  # Reduce from 512
```

### Apple Silicon (MPS) Specific Issues

**Symptom**: Slow training or operations falling back to CPU.

**Solution**:
```bash
# Set MPS memory limit (allows full memory usage)
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# Use smaller batch size for MPS
uv run python train_encoders_and_decoders.py \
    --generate_data 0 \
    --encoder_decoder_batch_size 128
```

**Note**: Training on MPS may be slower than CUDA but should produce equivalent results. The code automatically detects and uses MPS when available on Apple Silicon Macs.

### Training Loss Not Decreasing

**Cause**: Learning rate too high or too low.

**Solution**:
```bash
# Try different learning rate
uv run python train_encoders_and_decoders.py \
    --generate_data 0 \
    --learning_rate 0.0001  # Lower initial LR
```

### High Digit Error Despite Low Loss

**Cause**: VSA dimension might be too small for the complexity.

**Solution**: Increase VSA dimension or reduce complexity:
```bash
uv run python train_encoders_and_decoders.py \
    --generate_data 1 \
    --VSA_dim 4096  # Double the dimension
```

---

## Expected Results

### Training Loss Curves

- **Encoder**: Should decrease from ~1.0 to ~0.1-0.3
- **Decoder**: Should decrease from ~1.0 to ~0.2-0.4

### Layer Performance

- **Worst layers**: 0-5 (too early, not enough processing)
- **Best layers**: 15-20 (middle layers, optimal encoding)
- **Declining**: 25-32 (too close to output, overspecialized)

### Digit Accuracy

At Layer 17, expect:
- First number digits: ~95-99% accuracy
- Second number digits: ~95-99% accuracy
- Problem type: ~90-98% accuracy

---

## Next Step

Once models are saved, proceed to **[Step 4: Decoder Fine-tuning](STEP_4_DECODER_FINETUNING.md)**.

---

## Code References

| File | Lines | Purpose |
|------|-------|---------|
| [train_encoders_and_decoders.py](../../Programs/train_encoders_and_decoders.py) | 346-353 | Encoder initialization |
| [train_encoders_and_decoders.py](../../Programs/train_encoders_and_decoders.py) | 362-424 | Encoder training loop |
| [train_encoders_and_decoders.py](../../Programs/train_encoders_and_decoders.py) | 480-543 | Decoder training loop |
| [encoder_decoder_networks.py](../../llama/encoder_decoder_networks.py) | 7-19 | Encoder class |
| [encoder_decoder_networks.py](../../llama/encoder_decoder_networks.py) | 20-32 | Decoder class |
| [vsa_engine.py](../../llama/vsa_engine.py) | 399-409 | `decode_digits()` |
