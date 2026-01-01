# Paso 2: Generación de Datos (Etapa 1)

Este documento explica la fase de generación de datos donde recolectamos estados ocultos del modelo LLaMA congelado y creamos las representaciones VSA correspondientes.

---

## Tabla de Contenidos

1. [Resumen](#resumen)
2. [Diagrama de Flujo](#diagrama-de-flujo)
3. [Prerrequisitos](#prerrequisitos)
4. [Recorrido del Código](#recorrido-del-código)
5. [Formato de Datos de Salida](#formato-de-datos-de-salida)
6. [Opciones de Configuración](#opciones-de-configuración)
7. [Verificación](#verificación)
8. [Solución de Problemas](#solución-de-problemas)

---

## Resumen

### Qué Hace Este Paso

Este paso ejecuta el modelo LLaMA congelado con problemas matemáticos generados aleatoriamente y:
1. **Registra estados ocultos** en las 33 capas para cada problema
2. **Genera representaciones VSA** de los números de entrada y tipo de problema
3. **Guarda datos pareados** (estados ocultos ↔ vectores VSA) en disco

Estos datos pareados se usan en el Paso 3 para entrenar las redes encoder y decoder.

### Comprensión Conceptual

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Problema Mat.  │────▶│   Modelo LLaMA  │────▶│ Estados Ocultos │
│  "427 + 581"    │     │   (Congelado)   │     │  h ∈ R^4096     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                                               │
         │                                               │
         ▼                                               ▼
┌─────────────────┐                           ┌─────────────────┐
│  Motor VSA      │                           │ Dataset Pareado │
│  Encode(427,581)│──────────────────────────▶│  (h, VSA)       │
└─────────────────┘                           └─────────────────┘
```

### Por Qué Es Necesario

La red encoder debe aprender a mapear:
- **Entrada**: Estado oculto `h` (4096 dimensiones)
- **Salida**: Representación VSA `v` (2048 dimensiones)

Necesitamos miles de ejemplos de pares `(h, v)` para entrenar este mapeo.

---

## Diagrama de Flujo

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      PASO 2: GENERACIÓN DE DATOS                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │    Inicializar Componentes    │
                    │  ┌─────────────────────────┐  │
                    │  │ 1. Cargar modelo LLaMA  │  │
                    │  │ 2. Crear SymbolicEngine │  │
                    │  │ 3. Configurar directorios│ │
                    │  └─────────────────────────┘  │
                    └───────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │   Para cada ronda (10,000x):  │
                    │  ┌─────────────────────────┐  │
                    │  │ 1. Generar problema     │  │
                    │  │    matemático aleatorio │  │
                    │  │    (ej., 427 + 581)     │  │
                    │  └─────────────────────────┘  │
                    │              │                │
                    │              ▼                │
                    │  ┌─────────────────────────┐  │
                    │  │ 2. Crear prompt de      │  │
                    │  │    diálogo para LLaMA   │  │
                    │  └─────────────────────────┘  │
                    │              │                │
                    │              ▼                │
                    │  ┌─────────────────────────┐  │
                    │  │ 3. Ejecutar forward pass│  │
                    │  │    Registrar h en L capas│ │
                    │  └─────────────────────────┘  │
                    │              │                │
                    │              ▼                │
                    │  ┌─────────────────────────┐  │
                    │  │ 4. Generar vector VSA   │  │
                    │  │    SE.generate_VSA()    │  │
                    │  └─────────────────────────┘  │
                    │              │                │
                    │              ▼                │
                    │  ┌─────────────────────────┐  │
                    │  │ 5. Guardar par (h, VSA) │  │
                    │  │    cada N rondas        │  │
                    │  └─────────────────────────┘  │
                    └───────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │      Archivos de Salida:      │
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

## Prerrequisitos

1. **Paso 1 completado**: Modelo LLaMA descargado
2. **GPU con CUDA**: Requerido para forward pass
3. **Espacio en disco suficiente**: ~50GB para configuración por defecto
4. **Memoria**: ~20GB GPU RAM, ~32GB RAM del sistema

---

## Recorrido del Código

### Punto de Entrada del Script Principal

**Archivo**: [Programs/train_encoders_and_decoders.py](../../Programs/train_encoders_and_decoders.py)

**Comando**:
```bash
uv run python ~/Neurosymbolic-LLM/Programs/train_encoders_and_decoders.py \
    --run_name mi_experimento \
    --generate_data 1
```

### Ejecución Paso a Paso del Código

#### 1. Detección de Dispositivo y Configuración del Entorno (Líneas 222-237)

```python
# Auto-detectar dispositivo (CUDA, MPS, o CPU)
device = get_device()
device_type = get_device_type()
print_device_info()

# Configurar variables de entorno específicas del dispositivo
setup_device_environment()

if generate_data:
    os.environ['RANK'] = "0"
    os.environ['WORLD_SIZE'] = "1"
    os.environ['MASTER_ADDR'] = "127.0.0.2"
    os.environ['MASTER_PORT'] = master_port
    os.environ['LOCAL_RANK']  = "0"
```

**Propósito**: Auto-detectar el hardware disponible (GPU CUDA, Apple Silicon MPS, o CPU) y configurar el entorno de entrenamiento distribuido de acuerdo a ello.

**Dispositivos Soportados**:
- **CUDA**: GPUs NVIDIA (usa backend `nccl`)
- **MPS**: Apple Silicon (M1/M2/M3) (usa backend `gloo`)
- **CPU**: Alternativa para sistemas sin GPU (usa backend `gloo`)

#### 2. Cargar Modelo LLaMA (Líneas 229-237)

```python
generator = Llama.build(
    ckpt_dir=ckpt_dir,
    tokenizer_path=tokenizer_path,
    max_seq_len=max_seq_len,
    max_batch_size=max_batch_size,
)

# Congelar todos los parámetros
for param in generator.model.parameters():
    param.requires_grad = False
```

**Propósito**: Cargar el modelo LLaMA pre-entrenado y asegurar que los pesos estén congelados.

#### 3. Inicializar Motor Simbólico (Líneas 248-256)

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

**Propósito**: Crear o cargar el Motor Simbólico con vectores de vocabulario VSA consistentes.

#### 4. Generar y Guardar Datos (Líneas 311-328)

```python
if generate_data:
    # Datos de entrenamiento
    generate_and_save_data(
        generator=generator,
        SE=SE,
        save_dir=save_dir,
        rounds=train_data_rounds,      # 10,000 por defecto
        mode="train",
        save_frequency=save_frequency,  # 50 por defecto
        complexity=complexity,
        n_samples=n_samples,
        problem_type=problem_type,
        tokens_to_keep=tokens_to_keep,
        calculate_end_index=calculate_end_index,
        verbose=True
    )

    # Repetir para conjuntos de validación y prueba...
```

### Función Principal de Generación de Datos

**Archivo**: [llama/utilities.py](../../llama/utilities.py) - `generate_and_save_data()`

```python
def generate_and_save_data(generator, SE, save_dir, rounds, mode, save_frequency, ...):
    hidden_states_buffer = []
    vsa_buffer = []

    for round_idx in range(rounds):
        # 1. Generar problema matemático aleatorio
        dialogs, x, y, problem_type = generate_dialog(
            complexity=complexity,
            samples=n_samples,
            problem_type=problem_type
        )

        # 2. Ejecutar forward pass de LLaMA y recolectar estados ocultos
        prompt_tokens = generator.parse_chat(dialogs)
        with torch.no_grad():
            logits, h_stack, h = generator.model.forward(
                tokens=prompt_tokens,
                start_pos=0
            )
        # Forma de h_stack: (33, batch, seq_len, 4096) - estados ocultos de todas las capas

        # 3. Extraer estado oculto relevante (último token)
        relevant_h = h_stack[:, :, -tokens_to_keep:, :]

        # 4. Generar VSA correspondiente
        vsa = SE.generate_VSA(
            torch.tensor(x),
            torch.tensor(y),
            problem_types=[problem_type] * n_samples
        )

        # 5. Almacenar en buffer y guardar periódicamente
        hidden_states_buffer.append(relevant_h)
        vsa_buffer.append(vsa)

        if (round_idx + 1) % save_frequency == 0:
            save_batch(hidden_states_buffer, vsa_buffer, save_dir, mode, batch_idx)
            hidden_states_buffer = []
            vsa_buffer = []
```

### Generación de Diálogos

**Archivo**: [llama/utilities.py](../../llama/utilities.py) - `generate_dialog()`

```python
def generate_dialog(complexity, samples, problem_type, ...):
    dialogs = []
    x_values = []
    y_values = []

    for _ in range(samples):
        # Generar números aleatorios basados en complejidad
        max_val = 10 ** complexity  # complexity=2 → máximo 1000
        x = random.randint(0, max_val)
        y = random.randint(1, max_val)  # y > 0 para división/módulo

        # Crear pregunta basada en tipo de problema
        if problem_type == "addition":
            question = f"What is {x} + {y}?"
        elif problem_type == "multiplication":
            question = f"What is {x} * {y}?"
        # ... etc para otros tipos de problemas

        # Formatear como diálogo de chat LLaMA
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

## Formato de Datos de Salida

### Estructura de Directorios

```
gathered_data_{run_name}/
├── train_hidden_0.pt      # Lote 0 de estados ocultos
├── train_hidden_1.pt      # Lote 1 de estados ocultos
├── ...
├── train_hidden_199.pt    # 10000/50 = 200 lotes
├── train_vsa_0.pt         # Lote 0 de vectores VSA
├── train_vsa_1.pt         # Lote 1 de vectores VSA
├── ...
├── train_vsa_199.pt
├── val_hidden_*.pt        # Conjunto de validación
├── val_vsa_*.pt
├── test_hidden_*.pt       # Conjunto de prueba
└── test_vsa_*.pt
```

### Formas de Tensores

| Tipo de Archivo | Forma | Descripción |
|-----------------|-------|-------------|
| `*_hidden_*.pt` | `(33, batch*save_freq, tokens_to_keep, 4096)` | Estados ocultos por capa |
| `*_vsa_*.pt` | `(batch*save_freq, 2048)` | Representaciones VSA |

### Tamaños de Ejemplo (Config por Defecto)

| Dataset | Rondas | Lotes | Total de Muestras |
|---------|--------|-------|-------------------|
| Train | 10,000 | 200 | 20,000 (×2 muestras/ronda) |
| Val | 100 | 2 | 200 |
| Test | 1,000 | 20 | 2,000 |

---

## Opciones de Configuración

### Parámetros Clave

| Parámetro | Por Defecto | Descripción |
|-----------|-------------|-------------|
| `train_data_rounds` | 10000 | Número de problemas de entrenamiento |
| `val_data_rounds` | 100 | Número de problemas de validación |
| `test_data_rounds` | 1000 | Número de problemas de prueba |
| `n_samples` | 2 | Problemas por forward pass |
| `save_frequency` | 50 | Guardar cada N rondas |
| `complexity` | 2 | Máx dígitos (10^complexity) |
| `tokens_to_keep` | 1 | Tokens de estado oculto a guardar |
| `problem_type` | [lista] | Qué operaciones matemáticas |

### Tipos de Problemas Disponibles

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

Nota: `addition` y `division` están en `possible_problems` pero no en `problem_type` por defecto para entrenamiento.

### Comando Completo con Opciones

```bash
uv run python ~/Neurosymbolic-LLM/Programs/train_encoders_and_decoders.py \
    --run_name mi_experimento \
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

## Verificación

### 1. Verificar Directorio de Salida

```bash
ls -la gathered_data_mi_experimento/
```

Esperado: Múltiples archivos `.pt` para splits train/val/test.

### 2. Verificar Conteo de Archivos

```bash
# Debería coincidir con: train_data_rounds / save_frequency
ls gathered_data_mi_experimento/train_hidden_*.pt | wc -l
# Esperado: 200 (para 10000 rondas, save_frequency=50)
```

### 3. Verificar Formas de Tensores

```bash
uv run python -c "
import torch

# Cargar un lote
hidden = torch.load('gathered_data_mi_experimento/train_hidden_0.pt')
vsa = torch.load('gathered_data_mi_experimento/train_vsa_0.pt')

print(f'Forma de hidden: {hidden.shape}')
# Esperado: (33, 100, 1, 4096) para save_freq=50, n_samples=2, tokens_to_keep=1

print(f'Forma de VSA: {vsa.shape}')
# Esperado: (100, 2048)
"
```

### 4. Verificar Contenido VSA

```bash
uv run python -c "
import torch

# Cargar motor simbólico
SE = torch.load('VSA_library/symbolic_engine_...pt', weights_only=False)

# Cargar lote VSA
vsa = torch.load('gathered_data_mi_experimento/train_vsa_0.pt')

# Decodificar un VSA para verificar que codifica números correctamente
vsa_sample = vsa[0:1]  # Primera muestra
decoded_n1 = SE.decode_VSA(vsa_sample, SE.VSA_n1)
decoded_n2 = SE.decode_VSA(vsa_sample, SE.VSA_n2)
problem_type, score, _ = SE.decode_problem_type(vsa_sample)

print(f'Decodificado: n1={decoded_n1}, n2={decoded_n2}, tipo={problem_type}')
"
```

---

## Solución de Problemas

### Error: "CUDA out of memory" / "MPS out of memory"

**Causa**: Tamaño de lote muy grande para la GPU.

**Solución**:
```bash
# Reducir tamaño de lote
uv run python train_encoders_and_decoders.py \
    --generate_data 1 \
    --max_batch_size 1 \
    --n_samples 1
```

### Problemas Específicos de Apple Silicon (MPS)

**Síntoma**: Rendimiento lento u operaciones volviendo a CPU.

**Solución**:
```bash
# Establecer límite de memoria MPS (permite uso completo de memoria)
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# Ejecutar con tamaño de lote reducido (MPS tiene menos VRAM que GPUs NVIDIA típicas)
uv run python train_encoders_and_decoders.py \
    --generate_data 1 \
    --max_batch_size 1 \
    --n_samples 1
```

**Nota**: Algunas operaciones de PyTorch aún no están completamente optimizadas para MPS. Si encuentras problemas, el código automáticamente volverá a CPU para operaciones no soportadas.

### Error: "No checkpoint files found"

**Causa**: Ruta del modelo LLaMA incorrecta.

**Solución**:
```bash
# Verificar ruta y usar la correcta
ls ~/.llama/checkpoints/Llama3.1-8B-Instruct/

uv run python train_encoders_and_decoders.py \
    --generate_data 1 \
    --ckpt_dir ~/.llama/checkpoints/Llama3.1-8B-Instruct
```

### Error: "Disk full"

**Causa**: No hay suficiente espacio para archivos de salida.

**Solución**:
```bash
# Aumentar save_frequency para guardar menos archivos
uv run python train_encoders_and_decoders.py \
    --generate_data 1 \
    --save_frequency 100  # La mitad de archivos

# O reducir rondas de entrenamiento
uv run python train_encoders_and_decoders.py \
    --generate_data 1 \
    --train_data_rounds 5000
```

### Proceso Terminado / Segfault

**Causa**: RAM del sistema agotada.

**Solución**:
```bash
# Aumentar frecuencia de guardado (menos datos en memoria)
uv run python train_encoders_and_decoders.py \
    --generate_data 1 \
    --save_frequency 25  # Guardar más frecuentemente
```

---

## Recursos Estimados

### Tiempo

| GPU | Rondas | Tiempo Estimado |
|-----|--------|-----------------|
| A100 | 10,000 | ~2-3 horas |
| RTX 4090 | 10,000 | ~3-4 horas |
| RTX 3090 | 10,000 | ~4-5 horas |

### Espacio en Disco

| Configuración | Tamaño Aproximado |
|---------------|-------------------|
| Por defecto (10k train) | ~50 GB |
| Reducido (5k train) | ~25 GB |
| Mínimo (1k train) | ~5 GB |

---

## Siguiente Paso

Una vez que la generación de datos termine, continúa con **[Paso 3: Entrenamiento de Encoder/Decoder](STEP_3_ENCODER_DECODER_TRAINING.md)**.

---

## Referencias de Código

| Archivo | Líneas | Propósito |
|---------|--------|-----------|
| [train_encoders_and_decoders.py](../../Programs/train_encoders_and_decoders.py) | 311-328 | Llamadas principales de generación |
| [llama/utilities.py](../../llama/utilities.py) | - | Función `generate_and_save_data()` |
| [llama/utilities.py](../../llama/utilities.py) | - | Función `generate_dialog()` |
| [llama/vsa_engine.py](../../llama/vsa_engine.py) | 253-316 | Método `generate_VSA()` |
| [llama/model.py](../../llama/model.py) | 283-312 | `forward()` retorna h_stack |
