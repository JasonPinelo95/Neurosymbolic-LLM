# Step 4: Decoder Fine-tuning (Stage 3)

This document explains the end-to-end fine-tuning of the decoder network within the context of the full LLM forward pass using cross-entropy loss.

---

## Table of Contents

1. [Overview](#overview)
2. [Flow Diagram](#flow-diagram)
3. [Prerequisites](#prerequisites)
4. [Code Walkthrough](#code-walkthrough)
5. [The Symbolic Forward Pass](#the-symbolic-forward-pass)
6. [Skip Connection Mechanism](#skip-connection-mechanism)
7. [Configuration Options](#configuration-options)
8. [Verification](#verification)
9. [Troubleshooting](#troubleshooting)

---

## Overview

### What This Step Does

This step fine-tunes the decoder in an end-to-end setting:

1. **Freeze** the base LLaMA model and encoder
2. **Integrate** the symbolic pipeline into the forward pass
3. **Fine-tune** only the decoder using token-level cross-entropy loss
4. **Validate** on held-out math problems

### Key Difference from Step 3

| Step 3 | Step 4 |
|--------|--------|
| MSE loss on hidden states | Cross-entropy loss on output tokens |
| Trains encoder AND decoder | Fine-tunes decoder ONLY |
| Isolated from LLM | Integrated with full LLM forward pass |
| Reconstruction objective | Generation objective |

### Conceptual Understanding

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     END-TO-END SYMBOLIC FORWARD PASS                        │
└─────────────────────────────────────────────────────────────────────────────┘

Input: "What is 427 + 581?"

     Tokens                Layer 17                    Layer 17+               Output
  ┌──────────┐         ┌──────────────┐           ┌──────────────┐         ┌──────────┐
  │          │         │              │           │              │         │          │
  │ Embedding│────────▶│   Extract h  │──────────▶│  Inject ĥ   │────────▶│  Logits  │
  │          │         │              │           │              │         │          │
  └──────────┘         └──────────────┘           └──────────────┘         └──────────┘
                              │                          ▲
                              ▼                          │
                       ┌──────────────┐           ┌──────────────┐
                       │   Encoder    │           │   Decoder    │
                       │  (frozen)    │           │  (trainable) │
                       └──────────────┘           └──────────────┘
                              │                          ▲
                              ▼                          │
                       ┌──────────────┐           ┌──────────────┐
                       │  Decode VSA  │           │  Encode VSA  │
                       │  n1=427      │           │  result=1008 │
                       │  n2=581      │           │              │
                       │  type=add    │           │              │
                       └──────────────┘           └──────────────┘
                              │                          ▲
                              ▼                          │
                       ┌──────────────────────────────────────┐
                       │        SYMBOLIC ALGORITHM           │
                       │        427 + 581 = 1008             │
                       └──────────────────────────────────────┘

Loss: CrossEntropy(logits, "1008")
```

---

## Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STEP 4: DECODER FINE-TUNING                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │       Load Components         │
                    │  ┌─────────────────────────┐  │
                    │  │ 1. LLaMA model (frozen) │  │
                    │  │ 2. Encoder (frozen)     │  │
                    │  │ 3. Decoder (trainable)  │  │
                    │  │ 4. Symbolic Engine      │  │
                    │  └─────────────────────────┘  │
                    └───────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │     Configure Model           │
                    │  ┌─────────────────────────┐  │
                    │  │ symbolic_encoding_layer │  │
                    │  │ symbolic_decoding_layers│  │
                    │  │ skip_weights = 0.5      │  │
                    │  │ problem_score_threshold │  │
                    │  └─────────────────────────┘  │
                    └───────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │   For each training step:     │
                    │  ┌─────────────────────────┐  │
                    │  │ 1. Generate math problem│  │
                    │  │ 2. Run symbolic forward │  │
                    │  │ 3. Generate output token│  │
                    │  │ 4. Compute CE loss      │  │
                    │  │ 5. Backprop to decoder  │  │
                    │  │ 6. Update decoder       │  │
                    │  └─────────────────────────┘  │
                    └───────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │       Validation Loop         │
                    │  ┌─────────────────────────┐  │
                    │  │ Test accuracy on        │  │
                    │  │ held-out problems       │  │
                    │  └─────────────────────────┘  │
                    └───────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │         Save Model            │
                    │  ┌─────────────────────────┐  │
                    │  │ decoders_post_fine_     │  │
                    │  │ tuning_*.pth            │  │
                    │  └─────────────────────────┘  │
                    └───────────────────────────────┘
```

---

## Prerequisites

1. **Completed Step 3**: Encoder and decoder models saved
2. **GPU with CUDA**: Required for forward pass
3. **Memory**: ~24GB GPU RAM for full pipeline

---

## Code Walkthrough

### Main Script Entry Point

**File**: [Programs/fine_tune_decoders.py](../../Programs/fine_tune_decoders.py)

**Command**:
```bash
uv run python ~/Neurosymbolic-LLM/Programs/fine_tune_decoders.py \
    --run_name my_experiment \
    --encoder_path ~/Neurosymbolic-LLM/Programs/models/encoders_my_experiment.pth \
    --decoder_path ~/Neurosymbolic-LLM/Programs/models/decoders_my_experiment.pth \
    --symbolic_encoding_layer 17 \
    --symbolic_decoding_layers 17
```

### Step-by-Step Code Execution

#### 1. Load LLaMA and Symbolic Engine (Lines 223-244)

```python
generator = Llama.build(
    ckpt_dir=ckpt_dir,
    tokenizer_path=tokenizer_path,
    max_seq_len=max_seq_len,
    max_batch_size=max_batch_size,
)

# Load Symbolic Engine
SE = torch.load(f"{curr_dir}/VSA_library/symbolic_engine_...", weights_only=False)
generator.model.SE = SE
```

#### 2. Load Encoder and Decoder (Lines 921-944)

```python
# Load pre-trained encoder and decoder
generator.model.encoders = torch.load(encoder_path, weights_only=False)
generator.model.decoders = torch.load(decoder_path, weights_only=False)

# Freeze encoder
for i in range(len(generator.model.encoders)):
    for param in generator.model.encoders[i].parameters():
        param.requires_grad = False

# Keep decoder trainable (default)
```

#### 3. Configure Symbolic Pipeline (Lines 946-980)

```python
generator.model.bypass_symbolic = False
generator.model.symbolic_encoding_layer = symbolic_encoding_layer  # 17
generator.model.symbolic_decoding_layers = symbolic_decoding_layers  # [17]
generator.model.problem_score_threshold = problem_score_threshold  # 0.8

# Initialize skip weights (α = 0.5 by default)
generator.model.skip_weights = nn.ParameterList([
    nn.Parameter(torch.tensor(starting_skip_strength))  # 0.5
    for _ in symbolic_decoding_layers
])
```

#### 4. Training Step Function (Lines 263-471)

```python
def training_step(n_samples, generator, temperature, problem_type, ...):
    # Generate math problem
    dialogs, x, y, curr_problem_type = generate_dialog(
        complexity=complexity,
        samples=n_samples,
        problem_type=problem_type
    )

    # Compute correct answer
    if curr_problem_type == "addition":
        correct_responses = [x[i] + y[i] for i in range(len(x))]
    elif curr_problem_type == "multiplication":
        correct_responses = [(x[i] * y[i]) % 10**(complexity+1) for i in range(len(x))]
    # ... etc

    # Run symbolic forward pass and generate tokens
    h_stack, list_of_probs, list_of_logits, out_tokens = episode(
        dialogs=dialogs,
        generator=generator,
        inference_mode=generator.model.forward_symbolic_funnel,
        max_decoding_length=complexity+5,
        curr_pt=curr_problem_type,
        curr_x=x,
        curr_y=y
    )

    # Compute cross-entropy loss
    criterion = nn.CrossEntropyLoss()
    for batch in range(len(all_corr)):
        correct_tokens = generator.tokenizer.encode(str(all_corr[batch].item()))
        batch_loss = criterion(all_logits[:seq_len, batch, :], correct_tokens[:seq_len])
        loss += batch_loss

    # Backpropagate
    loss.backward()
    optimizer.step()

    return total_loss, total_score, response_data
```

---

## The Symbolic Forward Pass

### Core Method: `forward_symbolic_funnel`

**File**: [llama/model.py:315-681](../../llama/model.py#L315-L681)

This is the heart of the neurosymbolic system. Here's a detailed breakdown:

#### Phase 1: Extract Hidden State at Encoding Layer (Lines 333-358)

```python
for n, layer in enumerate(self.layers):
    if n == self.symbolic_encoding_layer:  # Layer 17
        # Get the hidden state of the last token
        relevant_h = h[:, -1, :]  # Shape: (batch, 4096)

        # Encode to VSA space
        symbolic_encoding = self.encoders[n](relevant_h)  # Shape: (batch, 2048)
```

#### Phase 2: Decode Problem Type and Numbers (Lines 365-404)

```python
# Decode problem type from VSA
problem_type_decoded, problem_type_score, _ = self.SE.decode_problem_type(
    symbolic_encoding,
    problem_subset=self.SE.possible_problems
)

# Check if score exceeds threshold
use_symbolic_layer = torch.tensor([
    score > self.problem_score_threshold  # 0.8
    for score in problem_type_score
])

# Decode numbers from VSA
decoded_n1 = self.SE.decode_digits(symbolic_encoding, self.SE.VSA_n1)
decoded_n2 = self.SE.decode_digits(symbolic_encoding, self.SE.VSA_n2)
```

#### Phase 3: Execute Symbolic Algorithm (Lines 405-600)

```python
if problem_type == "addition":
    decoded_sums = torch.tensor([decoded_n1[i] + decoded_n2[i] for i in range(batch_size)])
    final_symbol = self.SE.generate_VSA(decoded_sums, ...)

elif problem_type == "multiplication":
    decoded_prods = torch.tensor([decoded_n1[i] * decoded_n2[i] for i in range(batch_size)])
    final_symbol = self.SE.generate_VSA(decoded_prods, ...)

elif problem_type == "gcd":
    decoded_gcds = torch.tensor([np.gcd(decoded_n1[i], decoded_n2[i]) for i in range(batch_size)])
    final_symbol = self.SE.generate_VSA(decoded_gcds, ...)

# ... similar for modulo, lcm, division, bitwise operations
```

#### Phase 4: Inject Decoded Hidden State (Lines 619-642)

```python
if n in self.symbolic_decoding_layers:  # Layer 17
    # Decode VSA to hidden state
    modified_h = self.decoders[n](final_symbol)  # Shape: (batch, 4096)

    # Apply skip connection: h_final = (1-α)·h_decoder + α·h_original
    h = torch.cat([
        h[:, :-1, :],  # Keep all tokens except last
        torch.where(
            use_symbolic_layer,
            modified_h * (1 - self.skip_weights[0]) + h[:, -1, :] * self.skip_weights[0],
            h[:, -1, :]  # If no symbolic, keep original
        ).unsqueeze(1)
    ], dim=1)
```

---

## Skip Connection Mechanism

### Formula

```
h_final = (1 - α) · h_decoder + α · h_original
```

Where:
- `h_decoder`: Output from decoder network
- `h_original`: Original LLaMA hidden state
- `α = 0.5` (default, configurable via `starting_skip_strength`)

### Code Implementation

**File**: [llama/model.py:631](../../llama/model.py#L631)

```python
h = torch.cat([
    h[:, :-1, :],
    torch.where(
        use_symbolic_layer,
        modified_h * (1 - self.skip_weights[skip_index]) +  # (1-α)·h_decoder
        h[:, -1, :] * self.skip_weights[skip_index],        # α·h_original
        h[:, -1, :]  # If not using symbolic, keep original
    ).unsqueeze(1)
], dim=1)
```

### Selective Activation

The skip connection only activates when `problem_type_score > problem_score_threshold`:

```python
use_symbolic_layer = torch.tensor([
    score > self.problem_score_threshold for score in problem_type_score
])

# Only modify hidden state if symbolic layer should be used
torch.where(use_symbolic_layer, modified_h_blended, original_h)
```

---

## Configuration Options

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `symbolic_encoding_layer` | 17 | Layer to extract hidden state |
| `symbolic_decoding_layers` | [17] | Layer(s) to inject decoded state |
| `starting_skip_strength` | 0.5 | Skip connection weight (α) |
| `problem_score_threshold` | 0.8 | Min score to activate symbolic |
| `num_steps` | 1000 | Training steps |
| `learning_rate` | 0.001 | Decoder learning rate |

### Full Command with All Options

```bash
uv run python ~/Neurosymbolic-LLM/Programs/fine_tune_decoders.py \
    --run_name my_experiment \
    --encoder_path ~/Neurosymbolic-LLM/Programs/models/encoders_my_experiment.pth \
    --decoder_path ~/Neurosymbolic-LLM/Programs/models/decoders_my_experiment.pth \
    --symbolic_encoding_layer 17 \
    --symbolic_decoding_layers 17 \
    --starting_skip_strength 0.5 \
    --problem_score_threshold 0.8 \
    --num_steps 1000 \
    --num_epochs 1 \
    --learning_rate 0.001 \
    --n_samples 1 \
    --problem_type multiplication modulo gcd lcm
```

---

## Verification

### 1. Monitor Training Progress

During training, you should see output like:
```
Step 100, Loss: 2.34, Score: 0.65
Step 200, Loss: 1.87, Score: 0.78
Step 300, Loss: 1.45, Score: 0.85
...
```

### 2. Check wandb Dashboard

If `log_wandb=1`, verify:
- Loss decreasing over steps
- Score (accuracy) increasing
- Per-problem-type performance

### 3. Test with Sample Problem

```python
# After training, test manually
dialogs = [[
    {"role": "system", "content": "Answer with just the number."},
    {"role": "user", "content": "What is 427 + 581?"}
]]

# Run inference
_, _, _, out_tokens = episode(
    dialogs=dialogs,
    generator=generator,
    inference_mode=generator.model.forward_symbolic_funnel
)

print(generator.tokenizer.decode(out_tokens[0]))
# Expected: "1008"
```

### 4. Verify Saved Model

```bash
ls -la Programs/models/
```

Expected new file:
```
decoders_post_fine_tuning_{wandb_run_id}_*.pth
```

---

## Troubleshooting

### Error: "Expected encoder/decoder at layer X"

**Cause**: Encoder/decoder saved for different layers than configured.

**Solution**:
```bash
# Verify encoder layers
uv run python -c "
import torch
enc = torch.load('models/encoders_my_experiment.pth')
print([enc[i].layer_id for i in range(len(enc))])
"

# Make sure symbolic_encoding_layer matches
```

### Loss Not Decreasing

**Cause**: Learning rate too high or decoder capacity issue.

**Solution**:
```bash
# Try lower learning rate
uv run python fine_tune_decoders.py \
    --learning_rate 0.0001

# Or try training longer
uv run python fine_tune_decoders.py \
    --num_steps 2000
```

### Low Accuracy Despite Good Loss

**Cause**: Problem type detection failing.

**Solution**:
```bash
# Lower the threshold
uv run python fine_tune_decoders.py \
    --problem_score_threshold 0.5

# Or check encoder quality in Step 3
```

### CUDA / MPS Out of Memory

**Cause**: Batch size or sequence length too large.

**Solution**:
```bash
uv run python fine_tune_decoders.py \
    --max_batch_size 1 \
    --n_samples 1 \
    --inference_to_backprop_ratio 4  # Accumulate gradients
```

### Apple Silicon (MPS) Specific Issues

**Symptom**: Slow fine-tuning or errors with certain operations.

**Solution**:
```bash
# Set MPS memory limit (allows full memory usage)
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# Use minimal batch size and gradient accumulation
uv run python fine_tune_decoders.py \
    --max_batch_size 1 \
    --n_samples 1 \
    --inference_to_backprop_ratio 8
```

**Note**: Fine-tuning the full LLaMA 8B model is memory-intensive. On Mac M2 Max with 32GB unified memory, you may need to:
- Use `max_batch_size=1` and `n_samples=1`
- Increase `inference_to_backprop_ratio` to accumulate gradients over more steps
- Consider using the 1B parameter demo model instead (see demo notebook)

---

## Expected Results

### Training Metrics

| Metric | Initial | After 1000 steps |
|--------|---------|------------------|
| Loss | ~3-4 | ~0.5-1.5 |
| Accuracy | ~20-40% | ~85-95% |

### Per-Problem-Type Performance

| Problem Type | Expected Accuracy |
|--------------|-------------------|
| Addition | 95-99% |
| Multiplication | 90-98% |
| GCD | 85-95% |
| Modulo | 85-95% |
| LCM | 80-90% |

---

## Next Step

Once fine-tuning completes, proceed to **[Step 5: Evaluate Baselines](STEP_5_EVALUATE_BASELINES.md)** to compare against the standard LLM and other baselines.

---

## Code References

| File | Lines | Purpose |
|------|-------|---------|
| [fine_tune_decoders.py](../../Programs/fine_tune_decoders.py) | 263-471 | `training_step()` function |
| [fine_tune_decoders.py](../../Programs/fine_tune_decoders.py) | 768-1200+ | `run_experiment()` main loop |
| [model.py](../../llama/model.py) | 315-681 | `forward_symbolic_funnel()` |
| [model.py](../../llama/model.py) | 629-633 | Skip connection implementation |
| [vsa_engine.py](../../llama/vsa_engine.py) | 429-459 | `decode_problem_type()` |
