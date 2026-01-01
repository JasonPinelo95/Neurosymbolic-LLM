# Paso 1: Descargar Modelo LLaMA

Este documento explica el primer paso en el pipeline del LLM Neurosimbólico: descargar el modelo base LLaMA 3.1 8B Instruct.

---

## Tabla de Contenidos

1. [Resumen](#resumen)
2. [Diagrama de Flujo](#diagrama-de-flujo)
3. [Prerrequisitos](#prerrequisitos)
4. [Proceso Detallado](#proceso-detallado)
5. [Verificación](#verificación)
6. [Solución de Problemas](#solución-de-problemas)

---

## Resumen

### Qué Hace Este Paso

Este paso descarga el modelo pre-entrenado LLaMA 3.1 8B Instruct desde Hugging Face. Este modelo sirve como la **base congelada** para nuestro sistema neurosimbólico - sus pesos permanecen sin cambios durante todo el pipeline.

### ¿Por Qué LLaMA 3.1 8B Instruct?

| Propiedad | Valor | Razón |
|-----------|-------|-------|
| Parámetros | 8 Mil Millones | Suficiente para razonamiento complejo, suficientemente pequeño para una GPU |
| Dimensión Oculta | 4096 | Coincide con la dimensión de entrada del encoder |
| Capas | 32 | Proporciona múltiples puntos de intervención |
| Tipo | Instruct | Pre-ajustado para seguir instrucciones |

### Archivos Descargados

```
~/.llama/checkpoints/Llama3.1-8B-Instruct/
├── consolidated.00.pth    # Pesos del modelo (~16GB)
├── params.json            # Configuración del modelo
└── tokenizer.model        # Tokenizador SentencePiece
```

---

## Diagrama de Flujo

```
┌─────────────────────────────────────────────────────────────────┐
│                    PASO 1: DESCARGAR LLAMA                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                 ┌────────────────────────┐
                 │  Login en Hugging Face │
                 │  (Autenticación)       │
                 └────────────────────────┘
                              │
                              ▼
                 ┌────────────────────────┐
                 │  Aceptar Licencia      │
                 │  (Términos de Meta)    │
                 └────────────────────────┘
                              │
                              ▼
                 ┌────────────────────────┐
                 │  Descargar Archivos    │
                 │  - consolidated.00.pth │
                 │  - params.json         │
                 │  - tokenizer.model     │
                 └────────────────────────┘
                              │
                              ▼
                 ┌────────────────────────┐
                 │  Verificar Instalación │
                 │  (Revisar tamaños)     │
                 └────────────────────────┘
                              │
                              ▼
                 ┌────────────────────────┐
                 │  Listo para Paso 2     │
                 └────────────────────────┘
```

---

## Prerrequisitos

### 1. Cuenta de Hugging Face

Necesitas una cuenta de Hugging Face con acceso al modelo LLaMA:

1. Crear cuenta en [huggingface.co](https://huggingface.co)
2. Ir a [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
3. Aceptar el acuerdo de licencia de Meta
4. Esperar aprobación (usualmente instantánea)

### 2. CLI de Hugging Face

Usando `uvx` (sin necesidad de instalación):

```bash
# Iniciar sesión con tu token
uvx hf login
```

Cuando se solicite, ingresa tu token de acceso desde [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

**Nota**: Si usas un token fine-grained, habilita "Access to public gated repositories" en la configuración del token.

### 3. Espacio en Disco

Asegúrate de tener al menos **20GB** de espacio libre:

```bash
# Verificar espacio disponible
df -h ~
```

---

## Proceso Detallado

### Desglose del Comando

```bash
uvx hf download meta-llama/Llama-3.1-8B-Instruct \
    --include "original/*" \
    --local-dir ~/.llama/checkpoints/Llama3.1-8B-Instruct
```

| Argumento | Propósito |
|-----------|-----------|
| `meta-llama/Llama-3.1-8B-Instruct` | ID del repositorio en Hugging Face |
| `--include "original/*"` | Descargar solo los pesos originales de PyTorch (no safetensors) |
| `--local-dir ~/.llama/checkpoints/...` | Directorio de destino |

### ¿Por Qué `original/*`?

El repositorio contiene múltiples formatos:
- `original/` - Archivos PyTorch `.pth` (requeridos por este código)
- `*.safetensors` - Formato más seguro pero no compatible con nuestro código de carga

Nuestro código en [llama/generation.py:84-90](../../llama/generation.py#L84-L90) espera archivos `.pth`:

```python
checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"
ckpt_path = checkpoints[get_model_parallel_rank()]
checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
```

### Tiempo Estimado de Descarga

| Velocidad de Conexión | Tiempo Aproximado |
|----------------------|-------------------|
| 100 Mbps | ~25 minutos |
| 500 Mbps | ~5 minutos |
| 1 Gbps | ~3 minutos |

---

## Verificación

### 1. Verificar que los Archivos Existen

```bash
ls -la ~/.llama/checkpoints/Llama3.1-8B-Instruct/original/
```

Salida esperada:
```
total 16419532
-rw-r--r--  1 user  staff  16801063444 Nov 28 10:00 consolidated.00.pth
-rw-r--r--  1 user  staff          220 Nov 28 10:00 params.json
-rw-r--r--  1 user  staff      1988578 Nov 28 10:00 tokenizer.model
```

### 2. Verificar Tamaños de Archivos

| Archivo | Tamaño Esperado |
|---------|-----------------|
| `consolidated.00.pth` | ~16 GB |
| `params.json` | ~220 bytes |
| `tokenizer.model` | ~2 MB |

### 3. Verificar Contenido de params.json

```bash
cat ~/.llama/checkpoints/Llama3.1-8B-Instruct/original/params.json
```

Contenido esperado:
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

### 4. Probar Carga (Opcional)

```bash
uv run python -c "
import torch
from pathlib import Path

ckpt_dir = Path('~/.llama/checkpoints/Llama3.1-8B-Instruct/original').expanduser()
checkpoint = torch.load(ckpt_dir / 'consolidated.00.pth', map_location='cpu', weights_only=True)
print(f'Cargados {len(checkpoint)} tensores de parámetros')
# Esperado: Cargados 291 tensores de parámetros
"
```

---

## Solución de Problemas

### Error: "Access denied" (Acceso denegado)

**Causa**: No has aceptado la licencia de Meta o no has iniciado sesión.

**Solución**:
```bash
# Re-iniciar sesión
uvx hf logout
uvx hf login

# Luego visita la página del modelo y acepta la licencia
```

### Error: "No space left on device" (Sin espacio en disco)

**Causa**: Espacio en disco insuficiente.

**Solución**:
```bash
# Limpiar caché de Hugging Face
rm -rf ~/.cache/huggingface/hub/

# O especificar una ubicación diferente
uvx hf download meta-llama/Llama-3.1-8B-Instruct \
    --include "original/*" \
    --local-dir /ruta/con/mas/espacio/Llama3.1-8B-Instruct
```

### Error: "Connection timeout" (Tiempo de conexión agotado)

**Causa**: Problemas de red o carga del servidor de Hugging Face.

**Solución**:
```bash
# Reanudar descarga (es automático)
uvx hf download meta-llama/Llama-3.1-8B-Instruct \
    --include "original/*" \
    --local-dir ~/.llama/checkpoints/Llama3.1-8B-Instruct \
    --resume-download
```

### Error: "consolidated.00.pth not found" (No encontrado)

**Causa**: Los archivos están en un subdirectorio.

**Solución**:
```bash
# Verificar si los archivos están en el subdirectorio original/
ls ~/.llama/checkpoints/Llama3.1-8B-Instruct/original/

# Si es así, actualiza tus rutas o mueve los archivos
mv ~/.llama/checkpoints/Llama3.1-8B-Instruct/original/* \
   ~/.llama/checkpoints/Llama3.1-8B-Instruct/
```

---

## Referencia de Configuración

Después de descargar, actualiza tus archivos de configuración para apuntar a las rutas correctas:

### train_encoders_and_decoders_default_config.yaml

```yaml
ckpt_dir: "~/.llama/checkpoints/Llama3.1-8B-Instruct"
tokenizer_path: "~/.llama/checkpoints/Llama3.1-8B-Instruct/tokenizer.model"
```

### Sobrescritura por Línea de Comandos

```bash
uv run python train_encoders_and_decoders.py \
    --ckpt_dir ~/.llama/checkpoints/Llama3.1-8B-Instruct \
    --tokenizer_path ~/.llama/checkpoints/Llama3.1-8B-Instruct/tokenizer.model
```

---

## Siguiente Paso

Una vez que la verificación pase, continúa con **[Paso 2: Generación de Datos](STEP_2_DATA_GENERATION.md)**.

---

## Referencias de Código

| Archivo | Líneas | Propósito |
|---------|--------|-----------|
| [llama/generation.py](../../llama/generation.py) | 84-90 | Carga de checkpoint |
| [llama/generation.py](../../llama/generation.py) | 99-100 | Carga de tokenizador |
| [llama/model.py](../../llama/model.py) | 24-38 | Definición de ModelArgs |
