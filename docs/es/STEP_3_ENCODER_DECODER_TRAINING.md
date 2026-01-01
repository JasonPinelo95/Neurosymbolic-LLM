# Paso 3: Entrenamiento de Encoder/Decoder (Etapa 2)

Este documento explica el entrenamiento de las redes encoder y decoder que conectan los estados ocultos del LLM con las representaciones VSA.

---

## Tabla de Contenidos

1. [Resumen](#resumen)
2. [Diagrama de Flujo](#diagrama-de-flujo)
3. [Prerrequisitos](#prerrequisitos)
4. [Recorrido del Código](#recorrido-del-código)
5. [Proceso de Entrenamiento](#proceso-de-entrenamiento)
6. [Arquitectura del Modelo](#arquitectura-del-modelo)
7. [Opciones de Configuración](#opciones-de-configuración)
8. [Verificación](#verificación)
9. [Solución de Problemas](#solución-de-problemas)

---

## Resumen

### Qué Hace Este Paso

Este paso entrena dos tipos de redes neuronales:

1. **Encoder**: Mapea estados ocultos del LLM → representaciones VSA
   - Entrada: `h ∈ R^4096` (estado oculto de LLaMA)
   - Salida: `v ∈ R^2048` (vector VSA)
   - Pérdida: MSE entre VSA predicho y VSA verdadero

2. **Decoder**: Mapea representaciones VSA → estados ocultos del LLM
   - Entrada: `v ∈ R^2048` (vector VSA)
   - Salida: `h ∈ R^4096` (estado oculto reconstruido)
   - Pérdida: MSE entre estado reconstruido y original

### Comprensión Conceptual

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         OBJETIVO DE ENTRENAMIENTO                           │
└─────────────────────────────────────────────────────────────────────────────┘

  Estado Oculto (h)             VSA (v)                Estado Oculto (h')
  ┌───────────┐            ┌───────────┐              ┌───────────┐
  │           │  Encoder   │           │   Decoder    │           │
  │  R^4096   │───────────▶│  R^2048   │─────────────▶│  R^4096   │
  │           │            │           │              │           │
  └───────────┘            └───────────┘              └───────────┘
       │                        │                          │
       │                        │                          │
       ▼                        ▼                          ▼
  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
  │ Del forward     │    │ Objetivo: VSA   │    │ Debería ≈ h     │
  │ pass de LLaMA   │    │ verdadero       │    │ (reconstrucción)│
  └─────────────────┘    └─────────────────┘    └─────────────────┘
```

### ¿Por Qué Entrenar Separadamente?

El encoder/decoder se entrenan con pérdida MSE en datos pareados antes de integrar con el LLM. Esto:
- Proporciona inicialización estable para el fine-tuning de extremo a extremo
- Permite análisis de calidad de codificación por capa
- Identifica la capa óptima para intervención (Capa 17)

---

## Diagrama de Flujo

```
┌─────────────────────────────────────────────────────────────────────────────┐
│               PASO 3: ENTRENAMIENTO DE ENCODER/DECODER                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │   Cargar Datos de Entrenamiento│
                    │  ┌─────────────────────────┐  │
                    │  │ gathered_data/          │  │
                    │  │ ├── train_hidden_*.pt   │  │
                    │  │ └── train_vsa_*.pt      │  │
                    │  └─────────────────────────┘  │
                    └───────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │      Crear DataLoaders        │
                    │   (uno por capa: 0-32)        │
                    └───────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │     Inicializar Redes         │
                    │  ┌─────────────────────────┐  │
                    │  │ 33 Encoders (por capa)  │  │
                    │  │ 33 Decoders (por capa)  │  │
                    │  └─────────────────────────┘  │
                    └───────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
        ┌─────────────────────┐       ┌─────────────────────┐
        │ ENTRENAMIENTO       │       │ ENTRENAMIENTO       │
        │ ENCODER (1000 épocas)│      │ DECODER (1000 épocas)│
        │  ┌───────────────┐  │       │  ┌───────────────┐  │
        │  │ Por cada capa │  │       │  │ Por cada capa │  │
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
                    │      Evaluar y Seleccionar    │
                    │  ┌─────────────────────────┐  │
                    │  │ Encontrar mejor capa(17)│  │
                    │  │ Graficar loss por capa  │  │
                    │  │ Calcular errores dígitos│  │
                    │  └─────────────────────────┘  │
                    └───────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │        Guardar Modelos        │
                    │  ┌─────────────────────────┐  │
                    │  │ models/encoders_*.pth   │  │
                    │  │ models/decoders_*.pth   │  │
                    │  └─────────────────────────┘  │
                    └───────────────────────────────┘
```

---

## Prerrequisitos

1. **Paso 2 completado**: Archivos de datos en `gathered_data_{run_name}/`
2. **GPU**: El entrenamiento es mucho más rápido con CUDA
3. **Memoria**: ~8GB GPU RAM suficiente para entrenamiento

---

## Recorrido del Código

### Punto de Entrada del Script Principal

**Archivo**: [Programs/train_encoders_and_decoders.py](../../Programs/train_encoders_and_decoders.py)

**Comando**:
```bash
uv run python ~/Neurosymbolic-LLM/Programs/train_encoders_and_decoders.py \
    --run_name mi_experimento \
    --generate_data 0  # 0 = modo entrenamiento (no generación)
```

### Ejecución Paso a Paso del Código

#### 1. Cargar Datos y Crear DataLoaders (Líneas 331-344)

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

**Propósito**: Crear un DataLoader por capa conteniendo pares (estado_oculto, VSA).

#### 2. Inicializar Redes Encoder (Líneas 346-353)

```python
encoders = torch.nn.ModuleList()
for layer_id in layer_numbers:
    if tokens_to_keep == 1:
        # Token único → encoder lineal
        layer_encoder = Encoder(layer_id, model_dim, VSA_dim).to(device)
    else:
        # Múltiples tokens → encoder transformer
        layer_encoder = LastTokenTransformer(
            layer_id, model_dim, VSA_dim, num_layers=2, hidden_dim=512
        ).to(device)
    encoders.append(layer_encoder)
```

**Propósito**: Crear 33 redes encoder (una por capa), la arquitectura depende de `tokens_to_keep`.

#### 3. Bucle de Entrenamiento del Encoder (Líneas 362-424)

```python
optimizers = [optim.Adam(encoders[n].parameters(), lr=learning_rate) for n in range(len(layer_numbers))]
criterion = nn.MSELoss()

for i in range(training_epochs):  # 1000 épocas
    # Programación de learning rate
    if i in learning_rate_reduction_factors.keys():
        for param_group in optimizers[n].param_groups:
            param_group['lr'] *= learning_rate_reduction_factors[i]

    for n, n_layer in enumerate(layer_numbers):  # Para cada capa 0-32
        encoders[n].train()
        running_loss = 0

        for batch_idx, (data, labels) in enumerate(training_encoder_data_loaders[n]):
            # data: estados ocultos [batch, 4096]
            # labels: vectores VSA [batch, 2048]

            model_pred = encoders[n](data)              # Codificar h → v̂
            loss = torch.sqrt(criterion(model_pred, labels))  # RMSE(v̂, v)
            loss.backward()
            optimizers[n].step()
            optimizers[n].zero_grad()
            running_loss += loss.item()

        # Validación
        encoders.eval()
        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(validation_encoder_data_loaders[n]):
                model_pred = encoders[n](data)
                # Decodificar VSA para verificar precisión de dígitos
                v_digit_predictions_n1 = SE.decode_digits(model_pred, SE.VSA_n1)
                v_digit_labels_n1 = SE.decode_digits(labels, SE.VSA_n1)
```

**Operaciones Clave**:
- Forward: `h → Encoder → v̂`
- Pérdida: `√MSE(v̂, v)` (RMSE)
- Validación: Decodificar VSA predicho y verificar precisión de dígitos

#### 4. Inicializar y Entrenar Redes Decoder (Líneas 474-543)

```python
# Congelar encoders
for n, n_layer in enumerate(layer_numbers):
    for param in encoders[n].parameters():
        param.requires_grad = False

# Inicializar decoders
decoders = torch.nn.ModuleList()
for layer_id in layer_numbers:
    layer_decoder = Decoder(layer_id, VSA_dim, model_dim).to(device)
    decoders.append(layer_decoder)

# Bucle de entrenamiento
for j in range(decoding_training_epochs):  # 1000 épocas
    for n, n_layer in enumerate(layer_numbers):
        for batch_idx, (data, labels) in enumerate(training_encoder_data_loaders[n]):
            # Codificar primero (con encoder congelado)
            latent_representation = encoders[n](data)

            # Luego decodificar
            predicted_hidden_state = decoders[n](latent_representation)

            # Objetivo es el estado oculto original
            target = data if tokens_to_keep == 1 else data[:, -1, :]

            # Pérdida: reconstruir estado oculto original
            loss = torch.sqrt(criterion(predicted_hidden_state, target))
            loss.backward()
            decoding_optimizers[n].step()
```

**Operaciones Clave**:
- Forward: `h → Encoder → v̂ → Decoder → ĥ`
- Pérdida: `√MSE(ĥ, h)` (reconstruir estado oculto original)

#### 5. Evaluación y Selección de Capa (Líneas 600-728)

```python
# Probar precisión de codificación por capa
for n, layer in enumerate(layer_numbers):
    for batch_idx, (data, labels) in enumerate(testing_encoder_data_loaders[n]):
        pred = encoders[n](data)

        # Decodificar VSA predicho a números
        decoded_n1 = (SE.decode_digits(pred, SE.VSA_n1) * exponents).sum(axis=1)
        decoded_n2 = (SE.decode_digits(pred, SE.VSA_n2) * exponents).sum(axis=1)

        # Comparar con verdad
        actual_n1 = (SE.decode_digits(labels, SE.VSA_n1) * exponents).sum(axis=1)
        actual_n2 = (SE.decode_digits(labels, SE.VSA_n2) * exponents).sum(axis=1)

        # Calcular errores de dígitos
        batch_error = SE.digit_error(decoded_n1, actual_n1) + SE.digit_error(decoded_n2, actual_n2)

    if e < lowest_error:
        lowest_error_layer = layer
        lowest_error = e

print(f"Mejor capa: {lowest_error_layer} con error: {lowest_error}")
```

**Propósito**: Encontrar qué capa tiene el menor error de decodificación → típicamente Capa 17.

#### 6. Guardar Modelos (Líneas 589-596)

```python
torch.save(encoders.state_dict(), f"{curr_dir}/models/encoders_state_dict_{run_name}.pth")
torch.save(encoders, f"{curr_dir}/models/encoders_{run_name}.pth")
torch.save(decoders.state_dict(), f"{curr_dir}/models/decoders_state_dict_{run_name}.pth")
torch.save(decoders, f"{curr_dir}/models/decoders_{run_name}.pth")
```

---

## Arquitectura del Modelo

### Encoder (Token Único)

**Archivo**: [llama/encoder_decoder_networks.py:7-19](../../llama/encoder_decoder_networks.py#L7-L19)

```python
class Encoder(nn.Module):
    def __init__(self, layer_id, input_dim, output_dim, bias=False, dtype=torch.bfloat16):
        super().__init__()
        self.encoder_layer = nn.Linear(input_dim, output_dim, bias=bias, dtype=dtype)
        # input_dim = 4096 (dimensión oculta de LLaMA)
        # output_dim = 2048 (dimensión VSA)

    def forward(self, x):
        return self.encoder_layer(x)
```

**Parámetros**: 4096 × 2048 = **8,388,608** parámetros por capa

### Decoder (Token Único)

**Archivo**: [llama/encoder_decoder_networks.py:20-32](../../llama/encoder_decoder_networks.py#L20-L32)

```python
class Decoder(nn.Module):
    def __init__(self, layer_id, input_dim, output_dim, bias=False, dtype=torch.bfloat16):
        super().__init__()
        self.decoder_layer = nn.Linear(input_dim, output_dim, bias=bias, dtype=dtype)
        # input_dim = 2048 (dimensión VSA)
        # output_dim = 4096 (dimensión oculta de LLaMA)

    def forward(self, x):
        return self.decoder_layer(x)
```

**Parámetros**: 2048 × 4096 = **8,388,608** parámetros por capa

### Variantes Profundas (Opcional)

```python
class Encoder_Deep(nn.Module):
    def __init__(self, layer_id, input_dim, output_dim, hidden_dim, ...):
        self.encoder_layer_1 = nn.Linear(input_dim, hidden_dim)   # 4096 → 16384
        self.encoder_layer_2 = nn.Linear(hidden_dim, output_dim)  # 16384 → 2048
```

### Variante Transformer (Múltiples Tokens)

```python
class LastTokenTransformer(nn.Module):
    def __init__(self, layer_id, data_dim, output_dim, num_layers=4, num_heads=8, ...):
        self.input_proj = nn.Linear(data_dim, hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(...)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
```

---

## Proceso de Entrenamiento

### Programación de Learning Rate

| Época | Factor | LR Resultante |
|-------|--------|---------------|
| 0 | 1.0 | 0.001 |
| 50 | 0.5 | 0.0005 |
| 100 | 0.5 | 0.00025 |
| 250 | 0.1 | 0.000025 |
| 500 | 0.4 | 0.00001 |

### Función de Pérdida

**RMSE** (Raíz del Error Cuadrático Medio):
```python
loss = torch.sqrt(criterion(predicted, target))
# criterion = nn.MSELoss()
```

### Métricas de Validación

1. **Pérdida RMSE**: Comparación directa de vectores
2. **Precisión de Dígitos**: Decodificar VSA → números y comparar
3. **Precisión de Tipo de Problema**: Decodificar etiqueta de tipo de problema

---

## Opciones de Configuración

### Parámetros Clave

| Parámetro | Por Defecto | Descripción |
|-----------|-------------|-------------|
| `encoder_decoder_batch_size` | 512 | Tamaño de lote de entrenamiento |
| `training_epochs` | 1000 | Épocas de entrenamiento del encoder |
| `decoding_epochs` | 1000 | Épocas de entrenamiento del decoder |
| `learning_rate` | 0.001 | Learning rate inicial |
| `layer_numbers` | [0-32] | Qué capas entrenar |

### Comando Completo con Opciones

```bash
uv run python ~/Neurosymbolic-LLM/Programs/train_encoders_and_decoders.py \
    --run_name mi_experimento \
    --generate_data 0 \
    --encoder_decoder_batch_size 512 \
    --training_epochs 1000 \
    --decoding_epochs 1000 \
    --learning_rate 0.001
```

---

## Verificación

### 1. Verificar Archivos de Salida

```bash
ls -la Programs/models/
```

Esperado:
```
encoders_mi_experimento.pth
encoders_state_dict_mi_experimento.pth
decoders_mi_experimento.pth
decoders_state_dict_mi_experimento.pth
```

### 2. Verificar Carga del Modelo

```bash
uv run python -c "
import torch

encoders = torch.load('Programs/models/encoders_mi_experimento.pth')
decoders = torch.load('Programs/models/decoders_mi_experimento.pth')

print(f'Número de encoders: {len(encoders)}')  # Esperado: 33
print(f'Número de decoders: {len(decoders)}')  # Esperado: 33

# Verificar capa 17 específicamente
print(f'Forma del Encoder 17: {encoders[17].encoder_layer.weight.shape}')
# Esperado: torch.Size([2048, 4096])
"
```

### 3. Verificar Gráficas de Entrenamiento (wandb)

Si `log_wandb=1`, verificar en Weights & Biases:
- Average Encoder RMSE Loss Per Epoch
- Average Decoder RMSE Loss Per Epoch
- Error of Decoded Numbers vs Layer Number

### 4. Identificar Mejor Capa

De la salida de consola o wandb:
```
Minimum Error: X.XX and problem type error: Y at layer 17
```

---

## Solución de Problemas

### Error: "FileNotFoundError: gathered_data_*"

**Causa**: Archivos de datos del Paso 2 no encontrados.

**Solución**:
```bash
# Verificar que los datos existen
ls gathered_data_mi_experimento/

# Si no, ejecutar Paso 2 primero
uv run python train_encoders_and_decoders.py --generate_data 1
```

### Error: "CUDA out of memory" / "MPS out of memory"

**Causa**: Tamaño de lote muy grande.

**Solución**:
```bash
uv run python train_encoders_and_decoders.py \
    --generate_data 0 \
    --encoder_decoder_batch_size 256  # Reducir de 512
```

### Problemas Específicos de Apple Silicon (MPS)

**Síntoma**: Entrenamiento lento u operaciones volviendo a CPU.

**Solución**:
```bash
# Establecer límite de memoria MPS (permite uso completo de memoria)
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# Usar tamaño de lote más pequeño para MPS
uv run python train_encoders_and_decoders.py \
    --generate_data 0 \
    --encoder_decoder_batch_size 128
```

**Nota**: El entrenamiento en MPS puede ser más lento que CUDA pero debería producir resultados equivalentes. El código detecta automáticamente y usa MPS cuando está disponible en Macs con Apple Silicon.

### La Pérdida de Entrenamiento No Disminuye

**Causa**: Learning rate muy alto o muy bajo.

**Solución**:
```bash
# Probar diferente learning rate
uv run python train_encoders_and_decoders.py \
    --generate_data 0 \
    --learning_rate 0.0001  # LR inicial más bajo
```

### Error de Dígitos Alto a Pesar de Pérdida Baja

**Causa**: La dimensión VSA podría ser muy pequeña para la complejidad.

**Solución**: Aumentar dimensión VSA o reducir complejidad:
```bash
uv run python train_encoders_and_decoders.py \
    --generate_data 1 \
    --VSA_dim 4096  # Duplicar la dimensión
```

---

## Resultados Esperados

### Curvas de Pérdida de Entrenamiento

- **Encoder**: Debería disminuir de ~1.0 a ~0.1-0.3
- **Decoder**: Debería disminuir de ~1.0 a ~0.2-0.4

### Rendimiento por Capa

- **Peores capas**: 0-5 (muy temprano, poco procesamiento)
- **Mejores capas**: 15-20 (capas medias, codificación óptima)
- **Declinando**: 25-32 (muy cerca de salida, sobreespecializadas)

### Precisión de Dígitos

En Capa 17, esperar:
- Dígitos del primer número: ~95-99% de precisión
- Dígitos del segundo número: ~95-99% de precisión
- Tipo de problema: ~90-98% de precisión

---

## Siguiente Paso

Una vez que los modelos estén guardados, continúa con **[Paso 4: Fine-tuning del Decoder](STEP_4_DECODER_FINETUNING.md)**.

---

## Referencias de Código

| Archivo | Líneas | Propósito |
|---------|--------|-----------|
| [train_encoders_and_decoders.py](../../Programs/train_encoders_and_decoders.py) | 346-353 | Inicialización del encoder |
| [train_encoders_and_decoders.py](../../Programs/train_encoders_and_decoders.py) | 362-424 | Bucle de entrenamiento del encoder |
| [train_encoders_and_decoders.py](../../Programs/train_encoders_and_decoders.py) | 480-543 | Bucle de entrenamiento del decoder |
| [encoder_decoder_networks.py](../../llama/encoder_decoder_networks.py) | 7-19 | Clase Encoder |
| [encoder_decoder_networks.py](../../llama/encoder_decoder_networks.py) | 20-32 | Clase Decoder |
| [vsa_engine.py](../../llama/vsa_engine.py) | 399-409 | `decode_digits()` |
