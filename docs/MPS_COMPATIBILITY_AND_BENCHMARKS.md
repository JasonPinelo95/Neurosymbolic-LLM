# MPS Compatibility and Performance Benchmarks

This document details the modifications made to enable Apple Silicon (MPS) compatibility and the performance benchmarks conducted on a Mac M2 Max with 32GB unified memory.

## Table of Contents

1. [Hardware Configuration](#hardware-configuration)
2. [MPS Compatibility Issues and Fixes](#mps-compatibility-issues-and-fixes)
3. [Performance Benchmarks](#performance-benchmarks)
4. [Time and Storage Estimates](#time-and-storage-estimates)
5. [Recommended Configurations](#recommended-configurations)

---

## Hardware Configuration

| Component | Specification |
|-----------|---------------|
| **Machine** | Mac M2 Max |
| **RAM** | 32GB Unified Memory |
| **Storage** | Kingston USB Drive (external) |
| **PyTorch Backend** | MPS (Metal Performance Shaders) |
| **Model** | LLaMA 3.1 8B Instruct |

### Storage Paths Configuration

The YAML configuration files were updated to use the Kingston USB drive for data storage:

**`Programs/train_encoders_and_decoders_default_config.yaml`:**
```yaml
# === Paths ===
curr_dir: "/Volumes/KINGSTON/Neurosymbolic-LLM/Programs"
git_dir: "/Volumes/KINGSTON/Neurosymbolic-LLM"
ckpt_dir: "/Volumes/KINGSTON/.llama/checkpoints/Llama3.1-8B-Instruct/original"
tokenizer_path: "/Volumes/KINGSTON/.llama/checkpoints/Llama3.1-8B-Instruct/original/tokenizer.model"
```

**`Programs/fine_tune_decoders_default_config.yaml`:**
```yaml
ckpt_dir: "/Volumes/KINGSTON/.llama/checkpoints/Llama3.1-8B-Instruct/original"
tokenizer_path: "/Volumes/KINGSTON/.llama/checkpoints/Llama3.1-8B-Instruct/original/tokenizer.model"
```

---

## MPS Compatibility Issues and Fixes

### Issue 1: Socket Timeout with Gloo Backend

**Error:**
```
[c10d] The server socket on [::ffff:127.0.0.2]:29500 has timed out
```

**Cause:** The `gloo` backend (required for MPS, as `nccl` only works with CUDA) doesn't work well with `127.0.0.2` on macOS.

**Fix:** Changed `MASTER_ADDR` from `127.0.0.2` to `localhost` in both training scripts.

**Files Modified:**
- `Programs/train_encoders_and_decoders.py`
- `Programs/fine_tune_decoders.py`

**Code Change:**
```python
# Before
os.environ['MASTER_ADDR'] = "127.0.0.2"

# After
os.environ['MASTER_ADDR'] = "localhost"
```

---

### Issue 2: macOS Metadata Files Causing Checkpoint Count Mismatch

**Error:**
```
AssertionError: Loading a checkpoint for MP=2 but world size is 1
```

**Cause:** macOS creates hidden metadata files (e.g., `._consolidated.00.pth`) that were being counted as checkpoint files, making the system think there were 2 checkpoints instead of 1.

**Fix:** Remove the metadata files from the checkpoint directory.

**Command:**
```bash
rm /Volumes/KINGSTON/.llama/checkpoints/Llama3.1-8B-Instruct/original/._*
```

---

### Issue 3: MPS Does Not Support float64 (double precision)

**Error:**
```
TypeError: Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.
```

**Cause:** The VSA engine uses FFT operations with double precision (float64) for numerical accuracy. MPS does not support float64 tensors.

**Solution:** Hybrid CPU/GPU approach - move tensors to CPU for double precision operations, then return results as float32 to MPS.

**File Modified:** `llama/vsa_engine.py`

#### Fix 1: `make_unitary()` function (lines 17-30)

**Before:**
```python
def make_unitary(v):
    """
    Makes input unitary (Fourier components have magnitude of 1)
    """
    fv = torch.fft.fft(v.double(), axis=1)
    fv = fv/torch.sqrt(fv.real**2 + fv.imag**2)
    return torch.fft.ifft(fv, axis=1).real
```

**After:**
```python
def make_unitary(v):
    """
    Makes input unitary (Fourier components have magnitude of 1)
    Note: FFT with double precision is done on CPU for MPS compatibility
    Returns float32 to ensure MPS compatibility
    """
    original_device = v.device
    # Move to CPU for double precision FFT (MPS doesn't support float64)
    v_cpu = v.cpu().double()
    fv = torch.fft.fft(v_cpu, axis=1)
    fv = fv/torch.sqrt(fv.real**2 + fv.imag**2)
    result = torch.fft.ifft(fv, axis=1).real
    # Return as float32 since MPS doesn't support float64
    return result.float().to(original_device)
```

#### Fix 2: `make_tensor_unitary()` function (lines 32-42)

**Before:**
```python
def make_tensor_unitary(v):
    """
    Makes input tensor unitary (Fourier components have magnitude of 1)
    """
    fv = torch.fft.fft(v, axis=1)
    fv = fv/torch.sqrt(fv.real**2 + fv.imag**2)
    return torch.fft.ifft(fv, axis=1).real
```

**After:**
```python
def make_tensor_unitary(v):
    """
    Makes input tensor unitary (Fourier components have magnitude of 1)
    Note: Uses CPU for better precision with FFT. Returns float32 for MPS compatibility.
    """
    original_device = v.device
    v_cpu = v.cpu()
    fv = torch.fft.fft(v_cpu, axis=1)
    fv = fv/torch.sqrt(fv.real**2 + fv.imag**2)
    result = torch.fft.ifft(fv, axis=1).real
    return result.float().to(original_device)
```

#### Fix 3: `decode_digits()` method (around line 344-352)

The double precision operations in `decode_digits()` were already using CPU tensors:

```python
exponents = torch.tensor([10**d for d in range(len(self.digits))], dtype=torch.float32, device='cpu')
nums = torch.arange(0, 10, dtype=torch.float32, device='cpu')

# Note: Using CPU for double precision operations (MPS doesn't support float64)
modified_digit_values_cpu = modified_digit_values.cpu()
decoded_VSAs = torch.stack([sum([(exponents[i] * torch.dot(nums.double(), modified_digit_values_cpu[j,:,i].double()))
                                for i in range(self.max_digits)])
                            for j in range(batch_size)]).to(VSA.device)
```

### Performance Impact of Hybrid Approach

The hybrid CPU/GPU approach has minimal performance impact because:

1. **Small tensor sizes**: VSA vectors are small (2048 dimensions) compared to the LLaMA model
2. **Initialization only**: `make_unitary()` is called during initialization, not during training
3. **Infrequent operations**: These functions are not in the critical path of the forward pass
4. **LLaMA stays on GPU**: The large LLaMA 3.1 8B model remains entirely on MPS

---

## Performance Benchmarks

### Benchmark Configuration

```bash
cd /Users/mpinelo/projects/neuraan-research/Neurosymbolic-LLM/Programs && \
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 && \
time uv run python train_encoders_and_decoders.py \
    --run_name benchmark_test \
    --generate_data 1 \
    --train_data_rounds 10 \
    --val_data_rounds 0 \
    --test_data_rounds 0 \
    --max_batch_size 1 \
    --n_samples 1 \
    --log_wandb 0
```

### Benchmark Results

#### Initial Benchmark (10 rounds, n_samples=1)

| Metric | Value |
|--------|-------|
| **Total benchmark time (10 rounds)** | 2 min 43 sec (163 seconds) |
| **Model loading time (from USB)** | ~44 seconds |
| **Net generation time** | ~119 seconds |
| **Time per round** | ~12 seconds |

#### Validated Benchmark (1000 rounds, n_samples=1)

| Metric | Value |
|--------|-------|
| **Total time (1000 rounds)** | ~10 minutes |
| **Model loading time** | ~44 seconds |
| **Net generation time** | ~556 seconds |
| **Time per round** | ~0.6 seconds |
| **Data saved** | 46 files, 301 MB |

**Note:** The validated benchmark shows significantly faster performance (~0.6 sec/round) compared to the initial estimate (~12 sec/round). This is likely due to MPS warmup effects and batching optimizations that become more efficient over longer runs.

### Breakdown

```
Initial benchmark (10 rounds):
Total time = Model loading + Data generation
163 sec    = 44 sec       + 119 sec
Time per round = ~12 sec (includes warmup overhead)

Validated benchmark (1000 rounds):
Total time = Model loading + Data generation
~600 sec   = 44 sec       + 556 sec
Time per round = ~0.6 sec (steady state)
```

---

## Time and Storage Estimates

### Time Estimates by Dataset Size

Based on the validated benchmark (~0.6 seconds per round in steady state):

| train_data_rounds | Estimated Time | Total Samples* |
|-------------------|----------------|----------------|
| 1,000 | ~10 min | 2,000 |
| 5,000 | ~50 min | 10,000 |
| 10,000 | **~1h 40min** | 20,000 |
| 11,100 (full) | **~1h 50min** | 22,200 |

*With `n_samples=2` (default configuration)

### Formula

```
Estimated time (seconds) = Model loading (44s) + (rounds × 0.6 seconds/round)
```

### Full Dataset Generation Command

```bash
cd /Users/mpinelo/projects/neuraan-research/Neurosymbolic-LLM/Programs && \
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 && \
uv run python train_encoders_and_decoders.py \
    --run_name dataset_completo \
    --generate_data 1 \
    --train_data_rounds 10000 \
    --val_data_rounds 100 \
    --test_data_rounds 1000 \
    --max_batch_size 1 \
    --n_samples 2 \
    --log_wandb 0
```

| Parameter | Value | Samples |
|-----------|-------|---------|
| `train_data_rounds` | 10,000 | 20,000 |
| `val_data_rounds` | 100 | 200 |
| `test_data_rounds` | 1,000 | 2,000 |
| **Total** | 11,100 rounds | 22,200 samples |

### Storage Estimates

Each saved batch contains:
- Hidden states: `(33 layers × batch_size × tokens × 4096 dim)` as bfloat16
- VSA vectors: `(batch_size × 2048 dim)` as float32

**Validated storage (1000 rounds, n_samples=1):** 46 files, 301 MB

| train_data_rounds | save_frequency | Number of Files | Estimated Size |
|-------------------|----------------|-----------------|----------------|
| 1,000 | 50 | ~46 files | ~300 MB |
| 5,000 | 50 | ~230 files | ~1.5 GB |
| 10,000 | 50 | ~460 files | ~3 GB |
| 11,100 (full) | 50 | ~500 files | ~3.3 GB |

### Storage Formula

```
Number of files ≈ (train + val + test) / save_frequency × 2 (hidden + vsa pairs)
Size per 1000 rounds ≈ 300 MB (with n_samples=1)
Size per 1000 rounds ≈ 600 MB (with n_samples=2)
```

---

## Recommended Configurations

### For Initial Experimentation (~10 minutes)

```bash
cd /Users/mpinelo/projects/neuraan-research/Neurosymbolic-LLM/Programs && \
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 && \
uv run python train_encoders_and_decoders.py \
    --run_name experimento_inicial \
    --generate_data 1 \
    --train_data_rounds 1000 \
    --val_data_rounds 50 \
    --test_data_rounds 100 \
    --max_batch_size 1 \
    --n_samples 1 \
    --log_wandb 0
```

### For Full Dataset Generation (~1h 50min)

```bash
cd /Users/mpinelo/projects/neuraan-research/Neurosymbolic-LLM/Programs && \
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 && \
uv run python train_encoders_and_decoders.py \
    --run_name dataset_completo \
    --generate_data 1 \
    --train_data_rounds 10000 \
    --val_data_rounds 100 \
    --test_data_rounds 1000 \
    --max_batch_size 1 \
    --n_samples 2 \
    --log_wandb 0
```

### For Background Execution (optional)

```bash
cd /Users/mpinelo/projects/neuraan-research/Neurosymbolic-LLM/Programs && \
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 && \
nohup uv run python train_encoders_and_decoders.py \
    --run_name dataset_completo \
    --generate_data 1 \
    --train_data_rounds 10000 \
    --val_data_rounds 100 \
    --test_data_rounds 1000 \
    --max_batch_size 1 \
    --n_samples 2 \
    --log_wandb 0 > generation.log 2>&1 &

# Monitor progress
tail -f generation.log
```

### Memory Optimization Tips

1. **Set MPS memory limit:**
   ```bash
   export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
   ```

2. **Use minimal batch sizes for MPS:**
   ```bash
   --max_batch_size 1 --n_samples 1
   ```

3. **Increase save frequency to reduce RAM usage:**
   ```bash
   --save_frequency 25  # Save more frequently
   ```

---

## Summary of All Code Changes

| File | Change | Purpose |
|------|--------|---------|
| `Programs/train_encoders_and_decoders.py` | `MASTER_ADDR = "localhost"` | Fix gloo backend socket timeout |
| `Programs/fine_tune_decoders.py` | `MASTER_ADDR = "localhost"` | Fix gloo backend socket timeout |
| `llama/vsa_engine.py:17-30` | `make_unitary()` uses CPU for FFT | MPS float64 compatibility |
| `llama/vsa_engine.py:32-42` | `make_tensor_unitary()` uses CPU | MPS float64 compatibility |
| `Programs/*.yaml` | Updated paths to Kingston USB | External storage configuration |

---

## Troubleshooting

### If you see "MPS out of memory"

```bash
# Allow full memory usage
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# Reduce batch size
--max_batch_size 1 --n_samples 1
```

### If generation is slower than expected

1. Ensure USB drive is connected properly (USB 3.0+ recommended)
2. Close other applications using GPU
3. Check Activity Monitor for memory pressure

### If you need to interrupt and restart

The script does NOT support resumption. If interrupted:
1. Already-saved files in `gathered_data_<run_name>/` are preserved
2. Restart will overwrite from the beginning
3. Use a different `--run_name` to preserve existing data

---

## References

- [PyTorch MPS Backend Documentation](https://pytorch.org/docs/stable/notes/mps.html)
- [Apple Metal Performance Shaders](https://developer.apple.com/documentation/metalperformanceshaders)
- [LLaMA 3.1 Model Card](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
