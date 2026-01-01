# Step 1: Download LLaMA Model

This document explains the first step in the Neurosymbolic LLM pipeline: downloading the base LLaMA 3.1 8B Instruct model.

---

## Table of Contents

1. [Overview](#overview)
2. [Flow Diagram](#flow-diagram)
3. [Prerequisites](#prerequisites)
4. [Detailed Process](#detailed-process)
5. [Verification](#verification)
6. [Troubleshooting](#troubleshooting)

---

## Overview

### What This Step Does

This step downloads the pre-trained LLaMA 3.1 8B Instruct model from Hugging Face. This model serves as the **frozen base** for our neurosymbolic system - its weights remain unchanged throughout the entire pipeline.

### Why LLaMA 3.1 8B Instruct?

| Property | Value | Reason |
|----------|-------|--------|
| Parameters | 8 Billion | Large enough for complex reasoning, small enough to run on single GPU |
| Hidden Dimension | 4096 | Matches our encoder input dimension |
| Layers | 32 | Provides multiple intervention points |
| Type | Instruct | Pre-tuned for following instructions |

### Files Downloaded

```
~/.llama/checkpoints/Llama3.1-8B-Instruct/
├── consolidated.00.pth    # Model weights (~16GB)
├── params.json            # Model configuration
└── tokenizer.model        # SentencePiece tokenizer
```

---

## Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    STEP 1: DOWNLOAD LLAMA                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                 ┌────────────────────────┐
                 │  Hugging Face Login    │
                 │  (Authentication)      │
                 └────────────────────────┘
                              │
                              ▼
                 ┌────────────────────────┐
                 │  Accept License        │
                 │  (Meta's Terms)        │
                 └────────────────────────┘
                              │
                              ▼
                 ┌────────────────────────┐
                 │  Download Files        │
                 │  - consolidated.00.pth │
                 │  - params.json         │
                 │  - tokenizer.model     │
                 └────────────────────────┘
                              │
                              ▼
                 ┌────────────────────────┐
                 │  Verify Installation   │
                 │  (Check file sizes)    │
                 └────────────────────────┘
                              │
                              ▼
                 ┌────────────────────────┐
                 │  Ready for Step 2      │
                 └────────────────────────┘
```

---

## Prerequisites

### 1. Hugging Face Account

You need a Hugging Face account with access to the LLaMA model:

1. Create account at [huggingface.co](https://huggingface.co)
2. Go to [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
3. Accept Meta's license agreement
4. Wait for approval (usually instant)

### 2. Hugging Face CLI

Using `uvx` (no installation required):

```bash
# Login with your token
uvx hf login
```

When prompted, enter your access token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

**Note**: If using a fine-grained token, enable "Access to public gated repositories" in your token settings.

### 3. Disk Space

Ensure you have at least **20GB** of free disk space:

```bash
# Check available space
df -h ~
```

---

## Detailed Process

### Command Breakdown

```bash
uvx hf download meta-llama/Llama-3.1-8B-Instruct \
    --include "original/*" \
    --local-dir ~/.llama/checkpoints/Llama3.1-8B-Instruct
```

| Argument | Purpose |
|----------|---------|
| `meta-llama/Llama-3.1-8B-Instruct` | Repository ID on Hugging Face |
| `--include "original/*"` | Download only the original PyTorch weights (not safetensors) |
| `--local-dir ~/.llama/checkpoints/...` | Destination directory |

### Why `original/*`?

The repository contains multiple formats:
- `original/` - PyTorch `.pth` files (required by this codebase)
- `*.safetensors` - Safer format but not compatible with our loading code

Our code in [llama/generation.py:84-90](../../llama/generation.py#L84-L90) expects `.pth` files:

```python
checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"
ckpt_path = checkpoints[get_model_parallel_rank()]
checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
```

### Expected Download Time

| Connection Speed | Approximate Time |
|-----------------|------------------|
| 100 Mbps | ~25 minutes |
| 500 Mbps | ~5 minutes |
| 1 Gbps | ~3 minutes |

---

## Verification

### 1. Check Files Exist

```bash
ls -la ~/.llama/checkpoints/Llama3.1-8B-Instruct/original/
```

Expected output:
```
total 16419532
-rw-r--r--  1 user  staff  16801063444 Nov 28 10:00 consolidated.00.pth
-rw-r--r--  1 user  staff          220 Nov 28 10:00 params.json
-rw-r--r--  1 user  staff      1988578 Nov 28 10:00 tokenizer.model
```

### 2. Verify File Sizes

| File | Expected Size |
|------|---------------|
| `consolidated.00.pth` | ~16 GB |
| `params.json` | ~220 bytes |
| `tokenizer.model` | ~2 MB |

### 3. Check params.json Content

```bash
cat ~/.llama/checkpoints/Llama3.1-8B-Instruct/original/params.json
```

Expected content:
```json
{
    "dim": 4096,
    "n_layers": 32,
    "n_heads": 32,
    "n_kv_heads": 8,
    "vocab_size": 128256,
    "multiple_of": 1024,
    "ffn_dim_multiplier": 1.3,
    "norm_eps": 1e-05,
    "rope_theta": 500000.0
}
```

### 4. Test Loading (Optional)

```bash
uv run python -c "
import torch
from pathlib import Path

ckpt_dir = Path('~/.llama/checkpoints/Llama3.1-8B-Instruct/original').expanduser()
checkpoint = torch.load(ckpt_dir / 'consolidated.00.pth', map_location='cpu', weights_only=True)
print(f'Loaded {len(checkpoint)} parameter tensors')
# Expected: Loaded 291 parameter tensors
"
```

---

## Troubleshooting

### Error: "Access denied"

**Cause**: You haven't accepted Meta's license or aren't logged in.

**Solution**:
```bash
# Re-login
uvx hf logout
uvx hf login

# Then visit the model page and accept the license
```

### Error: "No space left on device"

**Cause**: Insufficient disk space.

**Solution**:
```bash
# Clear Hugging Face cache
rm -rf ~/.cache/huggingface/hub/

# Or specify a different location
uvx hf download meta-llama/Llama-3.1-8B-Instruct \
    --include "original/*" \
    --local-dir /path/with/more/space/Llama3.1-8B-Instruct
```

### Error: "Connection timeout"

**Cause**: Network issues or Hugging Face server load.

**Solution**:
```bash
# Resume download (it's automatic)
uvx hf download meta-llama/Llama-3.1-8B-Instruct \
    --include "original/*" \
    --local-dir ~/.llama/checkpoints/Llama3.1-8B-Instruct \
    --resume-download
```

### Error: "consolidated.00.pth not found"

**Cause**: Files are in a subdirectory.

**Solution**:
```bash
# Check if files are in original/ subdirectory
ls ~/.llama/checkpoints/Llama3.1-8B-Instruct/original/

# If yes, update your paths or move files
mv ~/.llama/checkpoints/Llama3.1-8B-Instruct/original/* \
   ~/.llama/checkpoints/Llama3.1-8B-Instruct/
```

---

## Configuration Reference

After downloading, update your config files to point to the correct paths:

### train_encoders_and_decoders_default_config.yaml

```yaml
ckpt_dir: "~/.llama/checkpoints/Llama3.1-8B-Instruct"
tokenizer_path: "~/.llama/checkpoints/Llama3.1-8B-Instruct/tokenizer.model"
```

### Command Line Override

```bash
uv run python train_encoders_and_decoders.py \
    --ckpt_dir ~/.llama/checkpoints/Llama3.1-8B-Instruct \
    --tokenizer_path ~/.llama/checkpoints/Llama3.1-8B-Instruct/tokenizer.model
```

---

## Next Step

Once verification passes, proceed to **[Step 2: Data Generation](STEP_2_DATA_GENERATION.md)**.

---

## Code References

| File | Lines | Purpose |
|------|-------|---------|
| [llama/generation.py](../../llama/generation.py) | 84-90 | Checkpoint loading |
| [llama/generation.py](../../llama/generation.py) | 99-100 | Tokenizer loading |
| [llama/model.py](../../llama/model.py) | 24-38 | ModelArgs definition |
