# Paso 4: Fine-tuning del Decoder (Etapa 3)

Este documento explica el fine-tuning de extremo a extremo de la red decoder dentro del contexto del forward pass completo del LLM usando pérdida de entropía cruzada.

---

## Tabla de Contenidos

1. [Resumen](#resumen)
2. [Diagrama de Flujo](#diagrama-de-flujo)
3. [Prerrequisitos](#prerrequisitos)
4. [Recorrido del Código](#recorrido-del-código)
5. [El Forward Pass Simbólico](#el-forward-pass-simbólico)
6. [Mecanismo de Skip Connection](#mecanismo-de-skip-connection)
7. [Opciones de Configuración](#opciones-de-configuración)
8. [Verificación](#verificación)
9. [Solución de Problemas](#solución-de-problemas)

---

## Resumen

### Qué Hace Este Paso

Este paso hace fine-tuning del decoder en un escenario de extremo a extremo:

1. **Congela** el modelo base LLaMA y el encoder
2. **Integra** el pipeline simbólico en el forward pass
3. **Fine-tunea** solo el decoder usando pérdida de entropía cruzada a nivel de token
4. **Valida** en problemas matemáticos de prueba

### Diferencia Clave con el Paso 3

| Paso 3 | Paso 4 |
|--------|--------|
| Pérdida MSE en estados ocultos | Pérdida CE en tokens de salida |
| Entrena encoder Y decoder | Fine-tunea SOLO decoder |
| Aislado del LLM | Integrado con forward pass completo del LLM |
| Objetivo de reconstrucción | Objetivo de generación |

### Comprensión Conceptual

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   FORWARD PASS SIMBÓLICO DE EXTREMO A EXTREMO               │
└─────────────────────────────────────────────────────────────────────────────┘

Entrada: "What is 427 + 581?"

     Tokens                Capa 17                     Capa 17+              Salida
  ┌──────────┐         ┌──────────────┐           ┌──────────────┐         ┌──────────┐
  │          │         │              │           │              │         │          │
  │ Embedding│────────▶│  Extraer h   │──────────▶│  Inyectar ĥ │────────▶│  Logits  │
  │          │         │              │           │              │         │          │
  └──────────┘         └──────────────┘           └──────────────┘         └──────────┘
                              │                          ▲
                              ▼                          │
                       ┌──────────────┐           ┌──────────────┐
                       │   Encoder    │           │   Decoder    │
                       │  (congelado) │           │ (entrenable) │
                       └──────────────┘           └──────────────┘
                              │                          ▲
                              ▼                          │
                       ┌──────────────┐           ┌──────────────┐
                       │ Decodificar  │           │ Codificar    │
                       │ VSA          │           │ VSA          │
                       │  n1=427      │           │ result=1008  │
                       │  n2=581      │           │              │
                       │  type=add    │           │              │
                       └──────────────┘           └──────────────┘
                              │                          ▲
                              ▼                          │
                       ┌──────────────────────────────────────┐
                       │       ALGORITMO SIMBÓLICO           │
                       │        427 + 581 = 1008             │
                       └──────────────────────────────────────┘

Pérdida: CrossEntropy(logits, "1008")
```

---

## Diagrama de Flujo

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PASO 4: FINE-TUNING DEL DECODER                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │       Cargar Componentes      │
                    │  ┌─────────────────────────┐  │
                    │  │ 1. Modelo LLaMA (cong.) │  │
                    │  │ 2. Encoder (congelado)  │  │
                    │  │ 3. Decoder (entrenable) │  │
                    │  │ 4. Motor Simbólico      │  │
                    │  └─────────────────────────┘  │
                    └───────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │     Configurar Modelo         │
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
                    │ Para cada paso de entrenamiento│
                    │  ┌─────────────────────────┐  │
                    │  │ 1. Generar problema mat.│  │
                    │  │ 2. Ejecutar forward simb│  │
                    │  │ 3. Generar token salida │  │
                    │  │ 4. Calcular pérdida CE  │  │
                    │  │ 5. Backprop al decoder  │  │
                    │  │ 6. Actualizar decoder   │  │
                    │  └─────────────────────────┘  │
                    └───────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │       Bucle de Validación     │
                    │  ┌─────────────────────────┐  │
                    │  │ Probar precisión en     │  │
                    │  │ problemas de prueba     │  │
                    │  └─────────────────────────┘  │
                    └───────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │        Guardar Modelo         │
                    │  ┌─────────────────────────┐  │
                    │  │ decoders_post_fine_     │  │
                    │  │ tuning_*.pth            │  │
                    │  └─────────────────────────┘  │
                    └───────────────────────────────┘
```

---

## Prerrequisitos

1. **Paso 3 completado**: Modelos encoder y decoder guardados
2. **GPU con CUDA**: Requerido para forward pass
3. **Memoria**: ~24GB GPU RAM para pipeline completo

---

## Recorrido del Código

### Punto de Entrada del Script Principal

**Archivo**: [Programs/fine_tune_decoders.py](../../Programs/fine_tune_decoders.py)

**Comando**:
```bash
uv run python ~/Neurosymbolic-LLM/Programs/fine_tune_decoders.py \
    --run_name mi_experimento \
    --encoder_path ~/Neurosymbolic-LLM/Programs/models/encoders_mi_experimento.pth \
    --decoder_path ~/Neurosymbolic-LLM/Programs/models/decoders_mi_experimento.pth \
    --symbolic_encoding_layer 17 \
    --symbolic_decoding_layers 17
```

### Ejecución Paso a Paso del Código

#### 1. Cargar LLaMA y Motor Simbólico (Líneas 223-244)

```python
generator = Llama.build(
    ckpt_dir=ckpt_dir,
    tokenizer_path=tokenizer_path,
    max_seq_len=max_seq_len,
    max_batch_size=max_batch_size,
)

# Cargar Motor Simbólico
SE = torch.load(f"{curr_dir}/VSA_library/symbolic_engine_...", weights_only=False)
generator.model.SE = SE
```

#### 2. Cargar Encoder y Decoder (Líneas 921-944)

```python
# Cargar encoder y decoder pre-entrenados
generator.model.encoders = torch.load(encoder_path, weights_only=False)
generator.model.decoders = torch.load(decoder_path, weights_only=False)

# Congelar encoder
for i in range(len(generator.model.encoders)):
    for param in generator.model.encoders[i].parameters():
        param.requires_grad = False

# Mantener decoder entrenable (por defecto)
```

#### 3. Configurar Pipeline Simbólico (Líneas 946-980)

```python
generator.model.bypass_symbolic = False
generator.model.symbolic_encoding_layer = symbolic_encoding_layer  # 17
generator.model.symbolic_decoding_layers = symbolic_decoding_layers  # [17]
generator.model.problem_score_threshold = problem_score_threshold  # 0.8

# Inicializar pesos de skip (α = 0.5 por defecto)
generator.model.skip_weights = nn.ParameterList([
    nn.Parameter(torch.tensor(starting_skip_strength))  # 0.5
    for _ in symbolic_decoding_layers
])
```

#### 4. Función de Paso de Entrenamiento (Líneas 263-471)

```python
def training_step(n_samples, generator, temperature, problem_type, ...):
    # Generar problema matemático
    dialogs, x, y, curr_problem_type = generate_dialog(
        complexity=complexity,
        samples=n_samples,
        problem_type=problem_type
    )

    # Calcular respuesta correcta
    if curr_problem_type == "addition":
        correct_responses = [x[i] + y[i] for i in range(len(x))]
    elif curr_problem_type == "multiplication":
        correct_responses = [(x[i] * y[i]) % 10**(complexity+1) for i in range(len(x))]
    # ... etc

    # Ejecutar forward pass simbólico y generar tokens
    h_stack, list_of_probs, list_of_logits, out_tokens = episode(
        dialogs=dialogs,
        generator=generator,
        inference_mode=generator.model.forward_symbolic_funnel,
        max_decoding_length=complexity+5,
        curr_pt=curr_problem_type,
        curr_x=x,
        curr_y=y
    )

    # Calcular pérdida de entropía cruzada
    criterion = nn.CrossEntropyLoss()
    for batch in range(len(all_corr)):
        correct_tokens = generator.tokenizer.encode(str(all_corr[batch].item()))
        batch_loss = criterion(all_logits[:seq_len, batch, :], correct_tokens[:seq_len])
        loss += batch_loss

    # Retropropagar
    loss.backward()
    optimizer.step()

    return total_loss, total_score, response_data
```

---

## El Forward Pass Simbólico

### Método Principal: `forward_symbolic_funnel`

**Archivo**: [llama/model.py:315-681](../../llama/model.py#L315-L681)

Este es el corazón del sistema neurosimbólico. Aquí hay un desglose detallado:

#### Fase 1: Extraer Estado Oculto en Capa de Codificación (Líneas 333-358)

```python
for n, layer in enumerate(self.layers):
    if n == self.symbolic_encoding_layer:  # Capa 17
        # Obtener el estado oculto del último token
        relevant_h = h[:, -1, :]  # Forma: (batch, 4096)

        # Codificar al espacio VSA
        symbolic_encoding = self.encoders[n](relevant_h)  # Forma: (batch, 2048)
```

#### Fase 2: Decodificar Tipo de Problema y Números (Líneas 365-404)

```python
# Decodificar tipo de problema del VSA
problem_type_decoded, problem_type_score, _ = self.SE.decode_problem_type(
    symbolic_encoding,
    problem_subset=self.SE.possible_problems
)

# Verificar si el puntaje excede el umbral
use_symbolic_layer = torch.tensor([
    score > self.problem_score_threshold  # 0.8
    for score in problem_type_score
])

# Decodificar números del VSA
decoded_n1 = self.SE.decode_digits(symbolic_encoding, self.SE.VSA_n1)
decoded_n2 = self.SE.decode_digits(symbolic_encoding, self.SE.VSA_n2)
```

#### Fase 3: Ejecutar Algoritmo Simbólico (Líneas 405-600)

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

# ... similar para modulo, lcm, division, operaciones bitwise
```

#### Fase 4: Inyectar Estado Oculto Decodificado (Líneas 619-642)

```python
if n in self.symbolic_decoding_layers:  # Capa 17
    # Decodificar VSA a estado oculto
    modified_h = self.decoders[n](final_symbol)  # Forma: (batch, 4096)

    # Aplicar skip connection: h_final = (1-α)·h_decoder + α·h_original
    h = torch.cat([
        h[:, :-1, :],  # Mantener todos los tokens excepto el último
        torch.where(
            use_symbolic_layer,
            modified_h * (1 - self.skip_weights[0]) + h[:, -1, :] * self.skip_weights[0],
            h[:, -1, :]  # Si no hay simbólico, mantener original
        ).unsqueeze(1)
    ], dim=1)
```

---

## Mecanismo de Skip Connection

### Fórmula

```
h_final = (1 - α) · h_decoder + α · h_original
```

Donde:
- `h_decoder`: Salida de la red decoder
- `h_original`: Estado oculto original de LLaMA
- `α = 0.5` (por defecto, configurable vía `starting_skip_strength`)

### Implementación en Código

**Archivo**: [llama/model.py:631](../../llama/model.py#L631)

```python
h = torch.cat([
    h[:, :-1, :],
    torch.where(
        use_symbolic_layer,
        modified_h * (1 - self.skip_weights[skip_index]) +  # (1-α)·h_decoder
        h[:, -1, :] * self.skip_weights[skip_index],        # α·h_original
        h[:, -1, :]  # Si no usa simbólico, mantener original
    ).unsqueeze(1)
], dim=1)
```

### Activación Selectiva

La skip connection solo se activa cuando `problem_type_score > problem_score_threshold`:

```python
use_symbolic_layer = torch.tensor([
    score > self.problem_score_threshold for score in problem_type_score
])

# Solo modificar estado oculto si la capa simbólica debe usarse
torch.where(use_symbolic_layer, modified_h_blended, original_h)
```

---

## Opciones de Configuración

### Parámetros Clave

| Parámetro | Por Defecto | Descripción |
|-----------|-------------|-------------|
| `symbolic_encoding_layer` | 17 | Capa para extraer estado oculto |
| `symbolic_decoding_layers` | [17] | Capa(s) para inyectar estado decodificado |
| `starting_skip_strength` | 0.5 | Peso de skip connection (α) |
| `problem_score_threshold` | 0.8 | Puntaje mín para activar simbólico |
| `num_steps` | 1000 | Pasos de entrenamiento |
| `learning_rate` | 0.001 | Learning rate del decoder |

### Comando Completo con Todas las Opciones

```bash
uv run python ~/Neurosymbolic-LLM/Programs/fine_tune_decoders.py \
    --run_name mi_experimento \
    --encoder_path ~/Neurosymbolic-LLM/Programs/models/encoders_mi_experimento.pth \
    --decoder_path ~/Neurosymbolic-LLM/Programs/models/decoders_mi_experimento.pth \
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

## Verificación

### 1. Monitorear Progreso del Entrenamiento

Durante el entrenamiento, deberías ver salida como:
```
Step 100, Loss: 2.34, Score: 0.65
Step 200, Loss: 1.87, Score: 0.78
Step 300, Loss: 1.45, Score: 0.85
...
```

### 2. Verificar Dashboard de wandb

Si `log_wandb=1`, verificar:
- Pérdida disminuyendo a lo largo de los pasos
- Score (precisión) aumentando
- Rendimiento por tipo de problema

### 3. Probar con Problema de Ejemplo

```python
# Después del entrenamiento, probar manualmente
dialogs = [[
    {"role": "system", "content": "Answer with just the number."},
    {"role": "user", "content": "What is 427 + 581?"}
]]

# Ejecutar inferencia
_, _, _, out_tokens = episode(
    dialogs=dialogs,
    generator=generator,
    inference_mode=generator.model.forward_symbolic_funnel
)

print(generator.tokenizer.decode(out_tokens[0]))
# Esperado: "1008"
```

### 4. Verificar Modelo Guardado

```bash
ls -la Programs/models/
```

Nuevo archivo esperado:
```
decoders_post_fine_tuning_{wandb_run_id}_*.pth
```

---

## Solución de Problemas

### Error: "Expected encoder/decoder at layer X"

**Causa**: Encoder/decoder guardados para capas diferentes a las configuradas.

**Solución**:
```bash
# Verificar capas del encoder
uv run python -c "
import torch
enc = torch.load('models/encoders_mi_experimento.pth')
print([enc[i].layer_id for i in range(len(enc))])
"

# Asegurarse que symbolic_encoding_layer coincida
```

### La Pérdida No Disminuye

**Causa**: Learning rate muy alto o problema de capacidad del decoder.

**Solución**:
```bash
# Probar learning rate más bajo
uv run python fine_tune_decoders.py \
    --learning_rate 0.0001

# O probar entrenar más tiempo
uv run python fine_tune_decoders.py \
    --num_steps 2000
```

### Baja Precisión a Pesar de Buena Pérdida

**Causa**: Detección de tipo de problema fallando.

**Solución**:
```bash
# Bajar el umbral
uv run python fine_tune_decoders.py \
    --problem_score_threshold 0.5

# O verificar calidad del encoder en Paso 3
```

### CUDA / MPS Sin Memoria

**Causa**: Tamaño de lote o longitud de secuencia muy grande.

**Solución**:
```bash
uv run python fine_tune_decoders.py \
    --max_batch_size 1 \
    --n_samples 1 \
    --inference_to_backprop_ratio 4  # Acumular gradientes
```

### Problemas Específicos de Apple Silicon (MPS)

**Síntoma**: Fine-tuning lento o errores con ciertas operaciones.

**Solución**:
```bash
# Establecer límite de memoria MPS (permite uso completo de memoria)
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# Usar tamaño de lote mínimo y acumulación de gradientes
uv run python fine_tune_decoders.py \
    --max_batch_size 1 \
    --n_samples 1 \
    --inference_to_backprop_ratio 8
```

**Nota**: El fine-tuning del modelo completo LLaMA 8B requiere mucha memoria. En Mac M2 Max con 32GB de memoria unificada, podrías necesitar:
- Usar `max_batch_size=1` y `n_samples=1`
- Aumentar `inference_to_backprop_ratio` para acumular gradientes sobre más pasos
- Considerar usar el modelo demo de 1B parámetros en su lugar (ver notebook demo)

---

## Resultados Esperados

### Métricas de Entrenamiento

| Métrica | Inicial | Después de 1000 pasos |
|---------|---------|----------------------|
| Pérdida | ~3-4 | ~0.5-1.5 |
| Precisión | ~20-40% | ~85-95% |

### Rendimiento Por Tipo de Problema

| Tipo de Problema | Precisión Esperada |
|------------------|-------------------|
| Addition | 95-99% |
| Multiplication | 90-98% |
| GCD | 85-95% |
| Modulo | 85-95% |
| LCM | 80-90% |

---

## Siguiente Paso

Una vez que el fine-tuning termine, continúa con **[Paso 5: Evaluar Baselines](STEP_5_EVALUATE_BASELINES.md)** para comparar contra el LLM estándar y otras líneas base.

---

## Referencias de Código

| Archivo | Líneas | Propósito |
|---------|--------|-----------|
| [fine_tune_decoders.py](../../Programs/fine_tune_decoders.py) | 263-471 | Función `training_step()` |
| [fine_tune_decoders.py](../../Programs/fine_tune_decoders.py) | 768-1200+ | Bucle principal `run_experiment()` |
| [model.py](../../llama/model.py) | 315-681 | `forward_symbolic_funnel()` |
| [model.py](../../llama/model.py) | 629-633 | Implementación de skip connection |
| [vsa_engine.py](../../llama/vsa_engine.py) | 429-459 | `decode_problem_type()` |
