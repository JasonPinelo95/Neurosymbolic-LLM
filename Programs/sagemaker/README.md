# SageMaker Scripts for Neurosymbolic LLM

Scripts separados para ejecutar el pipeline de entrenamiento en AWS SageMaker.

## Orden de ejecucion

```
1. generate_data.py   → Genera datos con LLaMA frozen
2. train_encoders.py  → Entrena encoders (hidden → VSA)
3. train_decoders.py  → Entrena decoders (VSA → hidden)
```

## Setup en SageMaker

### 1. Crear Notebook Instance

- **Instance type:** `ml.g6.4xlarge` (8B) o `ml.g6.48xlarge` (70B)
- **Volume size:** 100 GB
- **Platform:** Amazon Linux 2

### 2. Configurar entorno

```bash
# Abrir terminal en JupyterLab

# Clonar repositorio
cd /home/ec2-user/SageMaker
git clone https://github.com/vdhanraj/Neurosymbolic-LLM.git
cd Neurosymbolic-LLM

# Instalar uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# Instalar dependencias
uv sync

# Descargar modelo LLaMA
uv run huggingface-cli login
uv run huggingface-cli download meta-llama/Llama-3.1-8B-Instruct \
    --include "original/*" \
    --local-dir ~/.llama/checkpoints/Llama3.1-8B-Instruct
```

## Uso

### Paso 1: Generar datos

```bash
cd /home/ec2-user/SageMaker/Neurosymbolic-LLM/Programs/sagemaker

uv run python generate_data.py \
    --run_name exp1 \
    --train_data_rounds 10000 \
    --val_data_rounds 500 \
    --test_data_rounds 500
```

**Tiempo estimado:** ~2-4 horas para 10,000 rounds

### Paso 2: Entrenar encoders

```bash
uv run python train_encoders.py \
    --run_name exp1 \
    --training_epochs 1000 \
    --learning_rate 0.001
```

**Tiempo estimado:** ~30-60 minutos

### Paso 3: Entrenar decoders

```bash
uv run python train_decoders.py \
    --run_name exp1 \
    --decoding_epochs 1000 \
    --decoding_learning_rate 0.0001
```

**Tiempo estimado:** ~30-60 minutos

## Persistencia con S3

Los datos se pierden al detener el notebook. Usa S3 para persistir:

### Subir datos a S3

```bash
# Crear bucket (una sola vez)
aws s3 mb s3://tu-bucket-neurosymbolic

# Subir datos generados
aws s3 sync gathered_data_exp1/ s3://tu-bucket-neurosymbolic/gathered_data_exp1/

# Subir modelos
aws s3 sync models/ s3://tu-bucket-neurosymbolic/models/
```

### Descargar datos de S3

```bash
# Descargar datos
aws s3 sync s3://tu-bucket-neurosymbolic/gathered_data_exp1/ gathered_data_exp1/

# Descargar modelos
aws s3 sync s3://tu-bucket-neurosymbolic/models/ models/
```

### Upload automatico

Los scripts soportan upload automatico a S3:

```bash
uv run python generate_data.py \
    --run_name exp1 \
    --s3_bucket tu-bucket-neurosymbolic \
    --upload_to_s3 true
```

## Parametros importantes

### generate_data.py

| Parametro | Default | Descripcion |
|-----------|---------|-------------|
| `--run_name` | fecha | Nombre del experimento |
| `--train_data_rounds` | 10000 | Rounds de entrenamiento |
| `--complexity` | 2 | Complejidad (digitos + 1) |
| `--problem_type` | addition | Tipos de problema |

### train_encoders.py

| Parametro | Default | Descripcion |
|-----------|---------|-------------|
| `--run_name` | requerido | Debe coincidir con generate_data |
| `--training_epochs` | 1000 | Epocas de entrenamiento |
| `--learning_rate` | 0.001 | Learning rate inicial |

### train_decoders.py

| Parametro | Default | Descripcion |
|-----------|---------|-------------|
| `--run_name` | requerido | Debe coincidir con train_encoders |
| `--decoding_epochs` | 1000 | Epocas de entrenamiento |
| `--decoding_learning_rate` | 0.0001 | Learning rate inicial |

## Estructura de archivos

```
gathered_data_{run_name}/
├── train_hidden_*.pt      # Hidden states de entrenamiento
├── train_vsa_*.pt         # VSA vectors de entrenamiento
├── val_hidden_*.pt        # Hidden states de validacion
├── val_vsa_*.pt           # VSA vectors de validacion
├── test_hidden_*.pt       # Hidden states de test
└── test_vsa_*.pt          # VSA vectors de test

models/
├── encoders_{run_name}.pth
├── encoders_state_dict_{run_name}.pth
├── decoders_{run_name}.pth
└── decoders_state_dict_{run_name}.pth
```

## Troubleshooting

### "Loss: nan" en decoder training

Reduce el learning rate:
```bash
uv run python train_decoders.py --run_name exp1 --decoding_learning_rate 0.00001
```

### Out of memory

Reduce batch size:
```bash
uv run python train_encoders.py --run_name exp1 --encoder_decoder_batch_size 256
```

### Datos no encontrados

Verifica que el `--run_name` coincida en todos los scripts.

## Costos estimados

| Instancia | Costo/hora | 10k rounds + training |
|-----------|------------|----------------------|
| ml.g6.4xlarge | ~$2.04 | ~$10-15 |
| ml.g6.48xlarge | ~$20.78 | ~$100-150 |