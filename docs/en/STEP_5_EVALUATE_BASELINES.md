# Step 5: Evaluate Baselines

This document explains how to evaluate the neurosymbolic model against various baselines including the standard LLM, LoRA fine-tuned models, and Chain-of-Thought prompting.

---

## Table of Contents

1. [Overview](#overview)
2. [Flow Diagram](#flow-diagram)
3. [Baseline Types](#baseline-types)
4. [Code Walkthrough](#code-walkthrough)
5. [Running Evaluations](#running-evaluations)
6. [Interpreting Results](#interpreting-results)
7. [Configuration Options](#configuration-options)
8. [Troubleshooting](#troubleshooting)

---

## Overview

### What This Step Does

This step evaluates and compares:

1. **Standard LLM**: Baseline performance of LLaMA 3.1 8B without any modifications
2. **LoRA Baseline**: LLM with LoRA adapters trained on the same data
3. **Chain-of-Thought**: LLM with step-by-step reasoning prompts
4. **Neurosymbolic**: Our system from Steps 1-4

### Why Compare?

| Baseline | Tests |
|----------|-------|
| Standard LLM | What the model can do "out of the box" |
| LoRA | Whether standard fine-tuning matches symbolic approach |
| CoT | Whether prompting alone can solve the problems |
| Neurosymbolic | Full system performance |

### Expected Ranking (from paper)

```
Neurosymbolic > LoRA > CoT > Standard LLM
     ~95%      ~60%   ~40%     ~20%
```

---

## Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       STEP 5: EVALUATE BASELINES                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        ▼                           ▼                           ▼
┌───────────────────┐     ┌───────────────────┐     ┌───────────────────┐
│  STANDARD LLM     │     │  LORA BASELINE    │     │  CHAIN-OF-THOUGHT │
│  --test_baseline 1│     │  --lora_baseline 1│     │  --cot 1          │
│                   │     │                   │     │                   │
│  ┌─────────────┐  │     │  ┌─────────────┐  │     │  ┌─────────────┐  │
│  │ No symbolic │  │     │  │ Train LoRA  │  │     │  │ Prompt with │  │
│  │ intervention│  │     │  │ adapters    │  │     │  │ step-by-step│  │
│  │             │  │     │  │             │  │     │  │ instructions│  │
│  └─────────────┘  │     │  └─────────────┘  │     │  └─────────────┘  │
│         │         │     │         │         │     │         │         │
│         ▼         │     │         ▼         │     │         ▼         │
│  ┌─────────────┐  │     │  ┌─────────────┐  │     │  ┌─────────────┐  │
│  │ Evaluate on │  │     │  │ Evaluate on │  │     │  │ Evaluate on │  │
│  │ test set    │  │     │  │ test set    │  │     │  │ test set    │  │
│  └─────────────┘  │     │  └─────────────┘  │     │  └─────────────┘  │
└───────────────────┘     └───────────────────┘     └───────────────────┘
        │                           │                           │
        └───────────────────────────┼───────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │      Compare Results          │
                    │  ┌─────────────────────────┐  │
                    │  │ Per-problem accuracy    │  │
                    │  │ Overall accuracy        │  │
                    │  │ Loss metrics            │  │
                    │  └─────────────────────────┘  │
                    └───────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │      Generate Report          │
                    │  ┌─────────────────────────┐  │
                    │  │ wandb visualizations    │  │
                    │  │ Console output          │  │
                    │  │ Saved responses         │  │
                    │  └─────────────────────────┘  │
                    └───────────────────────────────┘
```

---

## Baseline Types

### 1. Standard LLM Baseline (`--test_baseline 1` or `2`)

Tests the frozen LLaMA model without any symbolic intervention.

**What it measures**: The base model's ability to do arithmetic.

```python
# In model.py - bypass_symbolic = True
generator.model.bypass_symbolic = True  # Skip all symbolic processing
```

**Modes**:
- `test_baseline=1`: Evaluate after training (symbolic system active during training)
- `test_baseline=2`: Evaluate only (no training, just test standard LLM)

### 2. LoRA Baseline (`--lora_baseline 1`)

Trains randomly initialized encoder/decoder networks (same architecture) without the symbolic computation.

**What it measures**: Whether the improvement comes from architecture or symbolic reasoning.

```python
# In fine_tune_decoders.py lines 924-937
if lora_baseline:
    lora_encoders = nn.ModuleList()
    lora_decoders = nn.ModuleList()
    for layer_id in layer_numbers:
        # Initialize random networks (same architecture)
        lora_encoder = Encoder(layer_id, model_dim, VSA_dim)
        lora_decoder = Decoder(layer_id, VSA_dim, model_dim)
        lora_encoders.append(lora_encoder)
        lora_decoders.append(lora_decoder)
    generator.model.encoders = lora_encoders
    generator.model.decoders = lora_decoders
```

### 3. Chain-of-Thought Baseline (`--cot 1`)

Tests the LLM with step-by-step reasoning prompts.

**What it measures**: Whether explicit reasoning helps without symbolic intervention.

```python
# Modified prompt format
if cot:
    question = f"Solve the following problem step by step: {question}"

# Response must contain "Final Answer:"
# Only the answer after "Final Answer:" is evaluated
```

---

## Code Walkthrough

### Main Evaluation Function

**File**: [Programs/fine_tune_decoders.py:719-743](../../Programs/fine_tune_decoders.py#L719-L743)

```python
def evaluate_model(testing_n_samples, testing_num_steps, testing_temperature,
                   problem_type, generator, criterion, inference_to_backprop_ratio,
                   complexity, cot, df, testing_steps_to_print, testing_verbose):
    losses = []
    scores = []
    responses = []

    generator.model.eval()
    with torch.no_grad():
        for step in range(testing_num_steps):
            loss, score, response_data = inference_step(
                n_samples=testing_n_samples,
                generator=generator,
                temperature=testing_temperature,
                problem_type=problem_type,
                inference_to_backprop_ratio=inference_to_backprop_ratio,
                criterion=criterion,
                cot=cot,
                complexity=complexity,
                verbose=testing_verbose
            )
            losses.append(loss)
            scores.append(score)
            responses.append(response_data)

    return losses, scores, responses
```

### Inference Step Function

**File**: [Programs/fine_tune_decoders.py:473-717](../../Programs/fine_tune_decoders.py#L473-L717)

```python
def inference_step(n_samples, generator, temperature, problem_type, ...):
    # Generate problem
    dialogs, x, y, curr_problem_type = generate_dialog(
        complexity=complexity,
        samples=n_samples,
        problem_type=problem_type,
        cot=cot  # Chain-of-Thought prompting
    )

    # Compute correct answer
    if curr_problem_type == "addition":
        correct_responses = [x[i] + y[i] for i in range(len(x))]
    # ... etc

    # Run forward pass (with or without symbolic, depending on bypass_symbolic)
    h_stack, list_of_probs, list_of_logits, out_tokens = episode(
        dialogs=dialogs,
        generator=generator,
        inference_mode=generator.model.forward_symbolic_funnel,
        max_decoding_length=mdl
    )

    # For CoT, extract answer after "Final Answer:"
    if cot:
        # Find "Final" token (19918) and extract answer
        for i in range(len(out_tokens)):
            if 19918 in out_tokens[i]:  # "Final" token
                answer_start = out_tokens[i].index(19918) + 4
                # Extract just the number
                ...

    # Score: 1 if correct, 0 if incorrect
    for i in range(len(out_tokens)):
        try:
            output = int(generator.tokenizer.decode(out_tokens[i]))
            score = int(output == correct_responses[i])
        except:
            score = 0
        total_score += score

    return total_loss, total_score / len(all_corr), response_data
```

### Results Plotting

**File**: [Programs/fine_tune_decoders.py:745-754](../../Programs/fine_tune_decoders.py#L745-L754)

```python
def plot_results(losses, scores, problem_type, bypass_symbolic):
    losses = np.array(losses)
    scores = np.array(scores)

    if bypass_symbolic == 1:
        output_text = f"Mean score and loss of standard LLM on {problem_type}: " \
                      f"{round(scores.mean()*100, 3)} ± {round(scores.std()*100, 4)}"
    else:
        output_text = f"Mean score and loss of symbolic LLM on {problem_type}: " \
                      f"{round(scores.mean()*100, 3)} ± {round(scores.std()*100, 4)}"

    print(output_text)
    return output_text
```

---

## Running Evaluations

### 1. Standard LLM Baseline

```bash
uv run python ~/Neurosymbolic-LLM/Programs/fine_tune_decoders.py \
    --run_name baseline_evaluation \
    --test_baseline 2 \
    --testing_num_steps 100 \
    --testing_n_samples 2 \
    --testing_problems addition multiplication modulo gcd lcm
```

**Key flags**:
- `--test_baseline 2`: Skip training, just evaluate standard LLM
- No encoder/decoder paths needed (not used)

### 2. LoRA Baseline

```bash
uv run python ~/Neurosymbolic-LLM/Programs/fine_tune_decoders.py \
    --run_name lora_baseline \
    --lora_baseline 1 \
    --encoder_path ~/Neurosymbolic-LLM/Programs/models/encoders_my_experiment.pth \
    --decoder_path ~/Neurosymbolic-LLM/Programs/models/decoders_my_experiment.pth \
    --num_steps 1000 \
    --testing_num_steps 100
```

**Key flags**:
- `--lora_baseline 1`: Use randomly initialized encoder/decoder
- Still needs encoder/decoder paths for architecture reference

### 3. Chain-of-Thought Baseline

```bash
uv run python ~/Neurosymbolic-LLM/Programs/fine_tune_decoders.py \
    --run_name cot_baseline \
    --cot 1 \
    --testing_num_steps 100 \
    --testing_n_samples 1
```

**Key flags**:
- `--cot 1`: Enable Chain-of-Thought prompting
- Automatically sets `test_baseline=2` (no symbolic)

### 4. Neurosymbolic (Full System)

```bash
uv run python ~/Neurosymbolic-LLM/Programs/fine_tune_decoders.py \
    --run_name neurosymbolic_eval \
    --encoder_path ~/Neurosymbolic-LLM/Programs/models/encoders_my_experiment.pth \
    --decoder_path ~/Neurosymbolic-LLM/Programs/models/decoders_post_fine_tuning_*.pth \
    --test_baseline 0 \
    --testing_num_steps 100
```

**Key flags**:
- `--test_baseline 0`: Use full symbolic pipeline
- Use fine-tuned decoder from Step 4

---

## Interpreting Results

### Console Output

```
Mean score and loss of standard LLM on multiplication: 23.5 ± 4.2, 3.45 ± 0.89
Mean score and loss of symbolic LLM on multiplication: 94.2 ± 2.1, 0.43 ± 0.15
```

**Metrics**:
- **Score**: Percentage of correct answers (higher is better)
- **Loss**: Cross-entropy loss (lower is better)
- **± value**: Standard deviation across test steps

### Per-Problem-Type Results

Results are logged to wandb and saved to files:

```
Programs/outputs/score_per_problem_testing_{run_id}.txt
```

Format:
```csv
split,actual_problem_type,predicted_problem_type,score
test,multiplication,multiplication,0.85
test,gcd,gcd,0.72
...
```

### Expected Results Comparison

| Baseline | Addition | Multiplication | GCD | Modulo | LCM |
|----------|----------|----------------|-----|--------|-----|
| Standard LLM | ~30% | ~20% | ~15% | ~10% | ~10% |
| CoT | ~50% | ~40% | ~30% | ~25% | ~20% |
| LoRA | ~65% | ~55% | ~50% | ~45% | ~40% |
| Neurosymbolic | ~98% | ~95% | ~90% | ~90% | ~85% |

---

## Configuration Options

### Testing Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `testing_num_steps` | 100 | Number of test batches |
| `testing_n_samples` | 2 | Samples per batch |
| `testing_temperature` | 0 | Greedy decoding |
| `testing_problems` | [all] | Which problems to test |
| `testing_verbose` | 0 | Output verbosity |

### Baseline Flags

| Flag | Effect |
|------|--------|
| `--test_baseline 0` | Use symbolic system |
| `--test_baseline 1` | Train symbolic, then test standard LLM |
| `--test_baseline 2` | Test standard LLM only (no training) |
| `--lora_baseline 1` | Use random encoder/decoder (no VSA) |
| `--cot 1` | Use Chain-of-Thought prompting |

### Output Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `save_responses` | true | Save model responses to file |
| `record_score_per_problem` | 2 | Log per-problem scores |
| `log_wandb` | 1 | Log to Weights & Biases |

---

## Troubleshooting

### CoT Responses Don't Contain "Final Answer:"

**Cause**: Model not following the prompt format.

**Solution**:
```bash
# Lower temperature for more deterministic output
uv run python fine_tune_decoders.py \
    --cot 1 \
    --testing_temperature 0
```

### LoRA Baseline Has Same Performance as Standard LLM

**Cause**: Not training the LoRA adapters.

**Solution**:
```bash
# Make sure train_model is true
uv run python fine_tune_decoders.py \
    --lora_baseline 1 \
    --train_model true \
    --num_steps 1000
```

### Different Results Across Runs

**Cause**: Random problem generation.

**Solution**:
```bash
# Use fixed dataset
uv run python fine_tune_decoders.py \
    --testing_data_df_path path/to/test_data.csv
```

### Memory Issues During Testing

**Cause**: Long CoT generations consuming memory.

**Solution**:
```bash
uv run python fine_tune_decoders.py \
    --cot 1 \
    --max_seq_len 2048 \  # Reduce from default
    --testing_n_samples 1
```

### Apple Silicon (MPS) Specific Issues

**Symptom**: Slow evaluation or memory errors on Mac.

**Solution**:
```bash
# Set MPS memory limit (allows full memory usage)
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# Use minimal batch size
uv run python fine_tune_decoders.py \
    --test_baseline 1 \
    --max_batch_size 1 \
    --testing_n_samples 1
```

**Note**: Evaluation on MPS is supported but may be slower than CUDA. For comprehensive testing, consider:
- Running fewer test samples per problem type
- Using the 1B parameter demo model for faster iteration
- Running full evaluations on CUDA-enabled machines when possible

---

## Complete Evaluation Script

Here's a script to run all baselines:

```bash
#!/bin/bash

RUN_NAME="full_evaluation"
ENCODER_PATH="~/Neurosymbolic-LLM/Programs/models/encoders_my_experiment.pth"
DECODER_PATH="~/Neurosymbolic-LLM/Programs/models/decoders_my_experiment.pth"
FINETUNED_DECODER="~/Neurosymbolic-LLM/Programs/models/decoders_post_fine_tuning_*.pth"

# 1. Standard LLM Baseline
echo "=== Standard LLM Baseline ==="
uv run python ~/Neurosymbolic-LLM/Programs/fine_tune_decoders.py \
    --run_name "${RUN_NAME}_standard_llm" \
    --test_baseline 2 \
    --testing_num_steps 100

# 2. Chain-of-Thought Baseline
echo "=== Chain-of-Thought Baseline ==="
uv run python ~/Neurosymbolic-LLM/Programs/fine_tune_decoders.py \
    --run_name "${RUN_NAME}_cot" \
    --cot 1 \
    --testing_num_steps 100

# 3. LoRA Baseline (requires training)
echo "=== LoRA Baseline ==="
uv run python ~/Neurosymbolic-LLM/Programs/fine_tune_decoders.py \
    --run_name "${RUN_NAME}_lora" \
    --lora_baseline 1 \
    --encoder_path $ENCODER_PATH \
    --decoder_path $DECODER_PATH \
    --num_steps 1000 \
    --testing_num_steps 100

# 4. Neurosymbolic (full system)
echo "=== Neurosymbolic System ==="
uv run python ~/Neurosymbolic-LLM/Programs/fine_tune_decoders.py \
    --run_name "${RUN_NAME}_neurosymbolic" \
    --encoder_path $ENCODER_PATH \
    --decoder_path $FINETUNED_DECODER \
    --test_baseline 0 \
    --testing_num_steps 100

echo "=== Evaluation Complete ==="
```

---

## Next Steps

After evaluation:

1. **Analyze results** in wandb dashboard
2. **Compare per-problem-type** performance
3. **Identify failure cases** from saved responses
4. **Iterate** on training if needed

---

## Code References

| File | Lines | Purpose |
|------|-------|---------|
| [fine_tune_decoders.py](../../Programs/fine_tune_decoders.py) | 719-743 | `evaluate_model()` |
| [fine_tune_decoders.py](../../Programs/fine_tune_decoders.py) | 473-717 | `inference_step()` |
| [fine_tune_decoders.py](../../Programs/fine_tune_decoders.py) | 745-754 | `plot_results()` |
| [fine_tune_decoders.py](../../Programs/fine_tune_decoders.py) | 924-937 | LoRA initialization |
| [llama/utilities.py](../../llama/utilities.py) | - | `generate_dialog()` with CoT |
