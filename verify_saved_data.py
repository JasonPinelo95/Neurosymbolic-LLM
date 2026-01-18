#!/usr/bin/env python3
"""
Script para verificar que los archivos de hidden states se guardaron correctamente.
Uso: python verify_saved_data.py --data_dir /path/to/gathered_data_train
"""

import torch
import argparse
import os
import glob
from pathlib import Path

def verify_h_stack_file(file_path, expected_dim=8192, expected_layers=81, tokens_to_keep=1):
    """
    Verifica un archivo h_stack individual

    Dimensiones esperadas para tokens_to_keep=1:
    (batch_size, tokens_to_keep, model_dim, n_layers+1)
    = (50, 1, 8192, 81) para 70B con save_frequency=50
    """
    print(f"\n{'='*80}")
    print(f"Verificando: {os.path.basename(file_path)}")
    print(f"{'='*80}")

    # Cargar el archivo
    try:
        h_stack = torch.load(file_path, map_location='cpu')
    except Exception as e:
        print(f"❌ ERROR al cargar archivo: {e}")
        return False

    print(f"✓ Archivo cargado exitosamente")
    print(f"  Tipo: {type(h_stack)}")
    print(f"  Dtype: {h_stack.dtype}")
    print(f"  Shape: {h_stack.shape}")

    # Verificar dimensiones
    if len(h_stack.shape) != 4:
        print(f"❌ ERROR: Esperaba 4 dimensiones, encontró {len(h_stack.shape)}")
        return False

    batch_size, num_tokens, dim, num_layers = h_stack.shape

    print(f"\n📊 Dimensiones:")
    print(f"  Batch size:    {batch_size}")
    print(f"  Tokens:        {num_tokens} (esperado: {tokens_to_keep})")
    print(f"  Hidden dim:    {dim} (esperado: {expected_dim})")
    print(f"  Layers:        {num_layers} (esperado: {expected_layers})")

    # Verificar valores esperados
    checks_passed = True

    if num_tokens != tokens_to_keep:
        print(f"  ⚠️  ADVERTENCIA: tokens_to_keep no coincide")
        checks_passed = False

    if dim != expected_dim:
        print(f"  ❌ ERROR: model_dim no coincide")
        checks_passed = False

    if num_layers != expected_layers:
        print(f"  ❌ ERROR: número de layers no coincide")
        checks_passed = False

    # Verificar que no haya valores inválidos
    print(f"\n🔍 Verificando calidad de datos:")

    has_nan = torch.isnan(h_stack).any()
    has_inf = torch.isinf(h_stack).any()
    all_zeros = (h_stack == 0).all()

    print(f"  NaN values:    {'❌ SÍ' if has_nan else '✓ NO'}")
    print(f"  Inf values:    {'❌ SÍ' if has_inf else '✓ NO'}")
    print(f"  All zeros:     {'❌ SÍ' if all_zeros else '✓ NO'}")

    if has_nan or has_inf or all_zeros:
        checks_passed = False

    # Estadísticas básicas
    print(f"\n📈 Estadísticas:")
    print(f"  Min:           {h_stack.min().item():.6f}")
    print(f"  Max:           {h_stack.max().item():.6f}")
    print(f"  Mean:          {h_stack.mean().item():.6f}")
    print(f"  Std:           {h_stack.std().item():.6f}")

    # Verificar varianza por layer
    print(f"\n📊 Varianza por layer (primeras 10 layers):")
    for layer_idx in range(min(10, num_layers)):
        layer_data = h_stack[:, :, :, layer_idx]
        variance = layer_data.var().item()
        mean = layer_data.mean().item()
        print(f"  Layer {layer_idx:2d}: mean={mean:8.4f}, var={variance:8.4f}")

    if checks_passed:
        print(f"\n✅ TODAS LAS VERIFICACIONES PASARON")
    else:
        print(f"\n⚠️  ALGUNAS VERIFICACIONES FALLARON")

    return checks_passed


def verify_vsa_file(file_path, expected_vsa_dim=2048, expected_batch_size=50):
    """
    Verifica un archivo de VSAs (correct_sps)

    Dimensiones esperadas:
    (batch_size, vsa_dim) = (50, 2048) para save_frequency=50
    """
    print(f"\n{'='*80}")
    print(f"Verificando VSA: {os.path.basename(file_path)}")
    print(f"{'='*80}")

    try:
        vsas = torch.load(file_path, map_location='cpu')
    except Exception as e:
        print(f"❌ ERROR al cargar archivo: {e}")
        return False

    print(f"✓ Archivo cargado exitosamente")
    print(f"  Tipo: {type(vsas)}")

    # Los VSAs pueden estar en una lista
    if isinstance(vsas, list):
        print(f"  Número de VSAs: {len(vsas)}")
        if len(vsas) > 0:
            first_vsa = vsas[0]
            print(f"  Shape del primer VSA: {first_vsa.shape}")
            print(f"  Dtype: {first_vsa.dtype}")
    else:
        print(f"  Dtype: {vsas.dtype}")
        print(f"  Shape: {vsas.shape}")

    return True


def main():
    parser = argparse.ArgumentParser(description='Verificar archivos de hidden states guardados')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Directorio con los archivos guardados (ej: gathered_data_train)')
    parser.add_argument('--model_dim', type=int, default=8192,
                      help='Dimensión del modelo (default: 8192 para 70B)')
    parser.add_argument('--n_layers', type=int, default=81,
                      help='Número de layers incluyendo embedding (default: 81 para 70B)')
    parser.add_argument('--tokens_to_keep', type=int, default=1,
                      help='Número de tokens guardados (default: 1)')
    parser.add_argument('--save_frequency', type=int, default=50,
                      help='Frecuencia de guardado (default: 50)')
    parser.add_argument('--max_files', type=int, default=3,
                      help='Número máximo de archivos a verificar (default: 3)')

    args = parser.parse_args()

    print(f"\n{'#'*80}")
    print(f"# VERIFICACIÓN DE DATOS GUARDADOS")
    print(f"{'#'*80}")
    print(f"\nDirectorio: {args.data_dir}")
    print(f"Configuración esperada:")
    print(f"  Model dim:       {args.model_dim}")
    print(f"  Layers:          {args.n_layers}")
    print(f"  Tokens to keep:  {args.tokens_to_keep}")
    print(f"  Save frequency:  {args.save_frequency}")

    # Buscar archivos h_stack
    h_stack_pattern = os.path.join(args.data_dir, "h_stack_round_*.pt")
    h_stack_files = sorted(glob.glob(h_stack_pattern))

    # Buscar archivos VSA
    vsa_pattern = os.path.join(args.data_dir, "correct_sps_round_*.pt")
    vsa_files = sorted(glob.glob(vsa_pattern))

    print(f"\n📁 Archivos encontrados:")
    print(f"  h_stack files: {len(h_stack_files)}")
    print(f"  VSA files:     {len(vsa_files)}")

    if len(h_stack_files) == 0:
        print(f"\n❌ ERROR: No se encontraron archivos h_stack en {args.data_dir}")
        print(f"   Patrón buscado: {h_stack_pattern}")
        return

    # Verificar archivos h_stack
    print(f"\n{'='*80}")
    print(f"VERIFICANDO ARCHIVOS H_STACK")
    print(f"{'='*80}")

    all_passed = True
    files_to_check = h_stack_files[:args.max_files]

    for h_file in files_to_check:
        passed = verify_h_stack_file(
            h_file,
            expected_dim=args.model_dim,
            expected_layers=args.n_layers,
            tokens_to_keep=args.tokens_to_keep
        )
        all_passed = all_passed and passed

    # Verificar archivos VSA
    if len(vsa_files) > 0:
        print(f"\n{'='*80}")
        print(f"VERIFICANDO ARCHIVOS VSA")
        print(f"{'='*80}")

        for vsa_file in vsa_files[:args.max_files]:
            verify_vsa_file(
                vsa_file,
                expected_batch_size=args.save_frequency
            )

    # Resumen final
    print(f"\n{'#'*80}")
    print(f"# RESUMEN")
    print(f"{'#'*80}")
    print(f"Total archivos h_stack: {len(h_stack_files)}")
    print(f"Archivos verificados:   {len(files_to_check)}")

    if all_passed:
        print(f"\n✅ ¡TODOS LOS ARCHIVOS VERIFICADOS CORRECTAMENTE!")
        print(f"\n✓ Los hidden states están guardándose correctamente")
        print(f"✓ Las dimensiones coinciden con la configuración")
        print(f"✓ Los datos no contienen valores inválidos")
    else:
        print(f"\n⚠️  ALGUNOS ARCHIVOS TIENEN PROBLEMAS")
        print(f"\nRevisa los errores arriba y verifica tu configuración")


if __name__ == "__main__":
    main()
