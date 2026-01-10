import torch
import torch.distributed as dist
import datetime
import os
from torch.utils.data import DataLoader


def safe_init_distributed():
    """
    Safely initialize distributed training with proper timeout and error handling.
    
    Returns:
        tuple: (is_distributed, rank, world_size, local_rank)
    """
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        print("[Distributed] Environment variables not set, running in single-GPU mode")
        return False, 0, 1, 0
    
    try:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        print(f"[Rank {rank}] Initializing distributed training")
        print(f"[Rank {rank}] World size: {world_size}, Local rank: {local_rank}")
        
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                world_size=world_size,
                rank=rank,
                timeout=datetime.timedelta(minutes=30)
            )
            print(f"[Rank {rank}] Distributed initialization successful")
        else:
            print(f"[Rank {rank}] Process group already initialized")
        
        torch.cuda.set_device(local_rank)
        print(f"[Rank {rank}] CUDA device set to {local_rank}")
        
        return True, rank, world_size, local_rank
        
    except Exception as e:
        print(f"[Distributed] Error during initialization: {e}")
        print(f"[Distributed] Falling back to single-GPU mode")
        return False, 0, 1, 0


def create_safe_dataloader(dataset, batch_size, shuffle=True, seed=None):
    """
    Create a DataLoader with CPU-based generator to avoid GPU synchronization issues.
    
    Args:
        dataset: PyTorch dataset
        batch_size: Batch size for the DataLoader
        shuffle: Whether to shuffle the data
        seed: Random seed for reproducibility
    
    Returns:
        DataLoader: Configured DataLoader
    """
    generator = None
    if shuffle and seed is not None:
        generator = torch.Generator(device='cpu')
        generator.manual_seed(seed)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        pin_memory=True,
        drop_last=False
    )
    
    return dataloader