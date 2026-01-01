# Paso 5: Evaluar Baselines

Este documento explica cómo evaluar el modelo neurosimbólico contra varias líneas base incluyendo el LLM estándar, modelos fine-tuneados con LoRA, y prompting Chain-of-Thought.

---

## Tabla de Contenidos

1. [Resumen](#resumen)
2. [Diagrama de Flujo](#diagrama-de-flujo)
3. [Tipos de Baselines](#tipos-de-baselines)
4. [Walkthrough del Código](#walkthrough-del-código)
5. [Ejecutar Evaluaciones](#ejecutar-evaluaciones)
6. [Interpretar Resultados](#interpretar-resultados)
7. [Opciones de Configuración](#opciones-de-configuración)
8. [Solución de Problemas](#solución-de-problemas)

---

## Resumen

### Qué Hace Este Paso

Este paso evalúa y compara:

1. **LLM Estándar**: Rendimiento base de LLaMA 3.1 8B sin modificaciones
2. **Baseline LoRA**: LLM con adaptadores LoRA entrenados con los mismos datos
3. **Chain-of-Thought**: LLM con prompts de razonamiento paso a paso
4. **Neurosimbólico**: Nuestro sistema de los Pasos 1-4

### ¿Por Qué Comparar?

| Baseline | Prueba |
|----------|--------|
| LLM Estándar | Qué puede hacer el modelo "de fábrica" |
| LoRA | Si el fine-tuning estándar iguala el enfoque simbólico |
| CoT | Si el prompting solo puede resolver los problemas |
| Neurosimbólico | Rendimiento del sistema completo |

### Ranking Esperado (del paper)

```
Neurosimbólico > LoRA > CoT > LLM Estándar
     ~95%        ~60%   ~40%     ~20%
```

---

## Diagrama de Flujo

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       PASO 5: EVALUAR BASELINES                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        ▼                           ▼                           ▼
┌───────────────────┐     ┌───────────────────┐     ┌───────────────────┐
│  LLM ESTÁNDAR     │     │  BASELINE LORA    │     │  CHAIN-OF-THOUGHT │
│  --test_baseline 1│     │  --lora_baseline 1│     │  --cot 1          │
│                   │     │                   │     │                   │
│  ┌─────────────┐  │     │  ┌─────────────┐  │     │  ┌─────────────┐  │
│  │Sin interven-│  │     │  │Entrenar     │  │     │  │ Prompt con  │  │
│  │ción simbóli-│  │     │  │adaptadores  │  │     │  │instrucciones│  │
│  │ca           │  │     │  │LoRA         │  │     │  │paso a paso  │  │
│  └─────────────┘  │     │  └─────────────┘  │     │  └─────────────┘  │
│         │         │     │         │         │     │         │         │
│         ▼         │     │         ▼         │     │         ▼         │
│  ┌─────────────┐  │     │  ┌─────────────┐  │     │  ┌─────────────┐  │
│  │ Evaluar en  │  │     │  │ Evaluar en  │  │     │  │ Evaluar en  │  │
│  │ set de test │  │     │  │ set de test │  │     │  │ set de test │  │
│  └─────────────┘  │     │  └─────────────┘  │     │  └─────────────┘  │
└───────────────────┘     └───────────────────┘     └───────────────────┘
        │                           │                           │
        └───────────────────────────┼───────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │      Comparar Resultados      │
                    │  ┌─────────────────────────┐  │
                    │  │ Precisión por problema  │  │
                    │  │ Precisión global        │  │
                    │  │ Métricas de loss        │  │
                    │  └─────────────────────────┘  │
                    └───────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │       Generar Reporte         │
                    │  ┌─────────────────────────┐  │
                    │  │ Visualizaciones wandb   │  │
                    │  │ Salida en consola       │  │
                    │  │ Respuestas guardadas    │  │
                    │  └─────────────────────────┘  │
                    └───────────────────────────────┘
```

---

## Tipos de Baselines

### 1. Baseline LLM Estándar (`--test_baseline 1` o `2`)

Prueba el modelo LLaMA congelado sin ninguna intervención simbólica.

**Qué mide**: La capacidad del modelo base para hacer aritmética.

```python
# En model.py - bypass_symbolic = True
generator.model.bypass_symbolic = True  # Omitir todo el procesamiento simbólico
```

**Modos**:
- `test_baseline=1`: Evaluar después del entrenamiento (sistema simbólico activo durante entrenamiento)
- `test_baseline=2`: Solo evaluar (sin entrenamiento, solo probar LLM estándar)

### 2. Baseline LoRA (`--lora_baseline 1`)

Entrena redes encoder/decoder inicializadas aleatoriamente (misma arquitectura) sin la computación simbólica.

**Qué mide**: Si la mejora viene de la arquitectura o del razonamiento simbólico.

```python
# En fine_tune_decoders.py líneas 924-937
if lora_baseline:
    lora_encoders = nn.ModuleList()
    lora_decoders = nn.ModuleList()
    for layer_id in layer_numbers:
        # Inicializar redes aleatorias (misma arquitectura)
        lora_encoder = Encoder(layer_id, model_dim, VSA_dim)
        lora_decoder = Decoder(layer_id, VSA_dim, model_dim)
        lora_encoders.append(lora_encoder)
        lora_decoders.append(lora_decoder)
    generator.model.encoders = lora_encoders
    generator.model.decoders = lora_decoders
```

### 3. Baseline Chain-of-Thought (`--cot 1`)

Prueba el LLM con prompts de razonamiento paso a paso.

**Qué mide**: Si el razonamiento explícito ayuda sin intervención simbólica.

```python
# Formato de prompt modificado
if cot:
    question = f"Solve the following problem step by step: {question}"

# La respuesta debe contener "Final Answer:"
# Solo la respuesta después de "Final Answer:" se evalúa
```

---

## Walkthrough del Código

### Función Principal de Evaluación

**Archivo**: [Programs/fine_tune_decoders.py:719-743](../../Programs/fine_tune_decoders.py#L719-L743)

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

### Función de Paso de Inferencia

**Archivo**: [Programs/fine_tune_decoders.py:473-717](../../Programs/fine_tune_decoders.py#L473-L717)

```python
def inference_step(n_samples, generator, temperature, problem_type, ...):
    # Generar problema
    dialogs, x, y, curr_problem_type = generate_dialog(
        complexity=complexity,
        samples=n_samples,
        problem_type=problem_type,
        cot=cot  # Prompting Chain-of-Thought
    )

    # Calcular respuesta correcta
    if curr_problem_type == "addition":
        correct_responses = [x[i] + y[i] for i in range(len(x))]
    # ... etc

    # Ejecutar forward pass (con o sin simbólico, dependiendo de bypass_symbolic)
    h_stack, list_of_probs, list_of_logits, out_tokens = episode(
        dialogs=dialogs,
        generator=generator,
        inference_mode=generator.model.forward_symbolic_funnel,
        max_decoding_length=mdl
    )

    # Para CoT, extraer respuesta después de "Final Answer:"
    if cot:
        # Encontrar token "Final" (19918) y extraer respuesta
        for i in range(len(out_tokens)):
            if 19918 in out_tokens[i]:  # Token "Final"
                answer_start = out_tokens[i].index(19918) + 4
                # Extraer solo el número
                ...

    # Puntaje: 1 si correcto, 0 si incorrecto
    for i in range(len(out_tokens)):
        try:
            output = int(generator.tokenizer.decode(out_tokens[i]))
            score = int(output == correct_responses[i])
        except:
            score = 0
        total_score += score

    return total_loss, total_score / len(all_corr), response_data
```

### Graficado de Resultados

**Archivo**: [Programs/fine_tune_decoders.py:745-754](../../Programs/fine_tune_decoders.py#L745-L754)

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

## Ejecutar Evaluaciones

### 1. Baseline LLM Estándar

```bash
uv run python ~/Neurosymbolic-LLM/Programs/fine_tune_decoders.py \
    --run_name baseline_evaluation \
    --test_baseline 2 \
    --testing_num_steps 100 \
    --testing_n_samples 2 \
    --testing_problems addition multiplication modulo gcd lcm
```

**Flags clave**:
- `--test_baseline 2`: Omitir entrenamiento, solo evaluar LLM estándar
- No se necesitan paths de encoder/decoder (no se usan)

### 2. Baseline LoRA

```bash
uv run python ~/Neurosymbolic-LLM/Programs/fine_tune_decoders.py \
    --run_name lora_baseline \
    --lora_baseline 1 \
    --encoder_path ~/Neurosymbolic-LLM/Programs/models/encoders_my_experiment.pth \
    --decoder_path ~/Neurosymbolic-LLM/Programs/models/decoders_my_experiment.pth \
    --num_steps 1000 \
    --testing_num_steps 100
```

**Flags clave**:
- `--lora_baseline 1`: Usar encoder/decoder inicializado aleatoriamente
- Aún necesita paths de encoder/decoder para referencia de arquitectura

### 3. Baseline Chain-of-Thought

```bash
uv run python ~/Neurosymbolic-LLM/Programs/fine_tune_decoders.py \
    --run_name cot_baseline \
    --cot 1 \
    --testing_num_steps 100 \
    --testing_n_samples 1
```

**Flags clave**:
- `--cot 1`: Habilitar prompting Chain-of-Thought
- Automáticamente establece `test_baseline=2` (sin simbólico)

### 4. Neurosimbólico (Sistema Completo)

```bash
uv run python ~/Neurosymbolic-LLM/Programs/fine_tune_decoders.py \
    --run_name neurosymbolic_eval \
    --encoder_path ~/Neurosymbolic-LLM/Programs/models/encoders_my_experiment.pth \
    --decoder_path ~/Neurosymbolic-LLM/Programs/models/decoders_post_fine_tuning_*.pth \
    --test_baseline 0 \
    --testing_num_steps 100
```

**Flags clave**:
- `--test_baseline 0`: Usar pipeline simbólico completo
- Usar decoder fine-tuneado del Paso 4

---

## Interpretar Resultados

### Salida en Consola

```
Mean score and loss of standard LLM on multiplication: 23.5 ± 4.2, 3.45 ± 0.89
Mean score and loss of symbolic LLM on multiplication: 94.2 ± 2.1, 0.43 ± 0.15
```

**Métricas**:
- **Score**: Porcentaje de respuestas correctas (mayor es mejor)
- **Loss**: Pérdida cross-entropy (menor es mejor)
- **± valor**: Desviación estándar entre pasos de prueba

### Resultados Por Tipo de Problema

Los resultados se registran en wandb y se guardan en archivos:

```
Programs/outputs/score_per_problem_testing_{run_id}.txt
```

Formato:
```csv
split,actual_problem_type,predicted_problem_type,score
test,multiplication,multiplication,0.85
test,gcd,gcd,0.72
...
```

### Comparación de Resultados Esperados

| Baseline | Suma | Multiplicación | MCD | Módulo | MCM |
|----------|------|----------------|-----|--------|-----|
| LLM Estándar | ~30% | ~20% | ~15% | ~10% | ~10% |
| CoT | ~50% | ~40% | ~30% | ~25% | ~20% |
| LoRA | ~65% | ~55% | ~50% | ~45% | ~40% |
| Neurosimbólico | ~98% | ~95% | ~90% | ~90% | ~85% |

---

## Opciones de Configuración

### Parámetros de Testing

| Parámetro | Defecto | Descripción |
|-----------|---------|-------------|
| `testing_num_steps` | 100 | Número de batches de prueba |
| `testing_n_samples` | 2 | Muestras por batch |
| `testing_temperature` | 0 | Decodificación greedy |
| `testing_problems` | [todos] | Qué problemas probar |
| `testing_verbose` | 0 | Verbosidad de salida |

### Flags de Baseline

| Flag | Efecto |
|------|--------|
| `--test_baseline 0` | Usar sistema simbólico |
| `--test_baseline 1` | Entrenar simbólico, luego probar LLM estándar |
| `--test_baseline 2` | Solo probar LLM estándar (sin entrenamiento) |
| `--lora_baseline 1` | Usar encoder/decoder aleatorio (sin VSA) |
| `--cot 1` | Usar prompting Chain-of-Thought |

### Opciones de Salida

| Parámetro | Defecto | Descripción |
|-----------|---------|-------------|
| `save_responses` | true | Guardar respuestas del modelo en archivo |
| `record_score_per_problem` | 2 | Registrar puntajes por problema |
| `log_wandb` | 1 | Registrar en Weights & Biases |

---

## Solución de Problemas

### Respuestas CoT No Contienen "Final Answer:"

**Causa**: El modelo no sigue el formato del prompt.

**Solución**:
```bash
# Bajar temperatura para salida más determinística
uv run python fine_tune_decoders.py \
    --cot 1 \
    --testing_temperature 0
```

### Baseline LoRA Tiene Mismo Rendimiento que LLM Estándar

**Causa**: No se están entrenando los adaptadores LoRA.

**Solución**:
```bash
# Asegurarse que train_model está en true
uv run python fine_tune_decoders.py \
    --lora_baseline 1 \
    --train_model true \
    --num_steps 1000
```

### Resultados Diferentes Entre Ejecuciones

**Causa**: Generación aleatoria de problemas.

**Solución**:
```bash
# Usar dataset fijo
uv run python fine_tune_decoders.py \
    --testing_data_df_path path/to/test_data.csv
```

### Problemas de Memoria Durante Testing

**Causa**: Generaciones CoT largas consumiendo memoria.

**Solución**:
```bash
uv run python fine_tune_decoders.py \
    --cot 1 \
    --max_seq_len 2048 \  # Reducir del default
    --testing_n_samples 1
```

### Problemas Específicos de Apple Silicon (MPS)

**Síntoma**: Evaluación lenta o errores de memoria en Mac.

**Solución**:
```bash
# Establecer límite de memoria MPS (permite uso completo de memoria)
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# Usar tamaño de lote mínimo
uv run python fine_tune_decoders.py \
    --test_baseline 1 \
    --max_batch_size 1 \
    --testing_n_samples 1
```

**Nota**: La evaluación en MPS está soportada pero puede ser más lenta que CUDA. Para testing comprehensivo, considera:
- Ejecutar menos muestras de prueba por tipo de problema
- Usar el modelo demo de 1B parámetros para iteración más rápida
- Ejecutar evaluaciones completas en máquinas con CUDA cuando sea posible

---

## Script Completo de Evaluación

Aquí hay un script para ejecutar todos los baselines:

```bash
#!/bin/bash

RUN_NAME="full_evaluation"
ENCODER_PATH="~/Neurosymbolic-LLM/Programs/models/encoders_my_experiment.pth"
DECODER_PATH="~/Neurosymbolic-LLM/Programs/models/decoders_my_experiment.pth"
FINETUNED_DECODER="~/Neurosymbolic-LLM/Programs/models/decoders_post_fine_tuning_*.pth"

# 1. Baseline LLM Estándar
echo "=== Baseline LLM Estándar ==="
uv run python ~/Neurosymbolic-LLM/Programs/fine_tune_decoders.py \
    --run_name "${RUN_NAME}_standard_llm" \
    --test_baseline 2 \
    --testing_num_steps 100

# 2. Baseline Chain-of-Thought
echo "=== Baseline Chain-of-Thought ==="
uv run python ~/Neurosymbolic-LLM/Programs/fine_tune_decoders.py \
    --run_name "${RUN_NAME}_cot" \
    --cot 1 \
    --testing_num_steps 100

# 3. Baseline LoRA (requiere entrenamiento)
echo "=== Baseline LoRA ==="
uv run python ~/Neurosymbolic-LLM/Programs/fine_tune_decoders.py \
    --run_name "${RUN_NAME}_lora" \
    --lora_baseline 1 \
    --encoder_path $ENCODER_PATH \
    --decoder_path $DECODER_PATH \
    --num_steps 1000 \
    --testing_num_steps 100

# 4. Neurosimbólico (sistema completo)
echo "=== Sistema Neurosimbólico ==="
uv run python ~/Neurosymbolic-LLM/Programs/fine_tune_decoders.py \
    --run_name "${RUN_NAME}_neurosymbolic" \
    --encoder_path $ENCODER_PATH \
    --decoder_path $FINETUNED_DECODER \
    --test_baseline 0 \
    --testing_num_steps 100

echo "=== Evaluación Completa ==="
```

---

## Siguientes Pasos

Después de la evaluación:

1. **Analizar resultados** en el dashboard de wandb
2. **Comparar rendimiento por tipo de problema**
3. **Identificar casos de falla** de las respuestas guardadas
4. **Iterar** en el entrenamiento si es necesario

---

## Referencias de Código

| Archivo | Líneas | Propósito |
|---------|--------|-----------|
| [fine_tune_decoders.py](../../Programs/fine_tune_decoders.py) | 719-743 | `evaluate_model()` |
| [fine_tune_decoders.py](../../Programs/fine_tune_decoders.py) | 473-717 | `inference_step()` |
| [fine_tune_decoders.py](../../Programs/fine_tune_decoders.py) | 745-754 | `plot_results()` |
| [fine_tune_decoders.py](../../Programs/fine_tune_decoders.py) | 924-937 | Inicialización LoRA |
| [llama/utilities.py](../../llama/utilities.py) | - | `generate_dialog()` con CoT |
