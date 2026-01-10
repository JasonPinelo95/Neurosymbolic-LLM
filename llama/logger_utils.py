import logging
import time
import datetime
import os
import sys
import torch
import psutil
from pathlib import Path
from typing import Optional, Dict, Any
import json
import threading


class NeurosymbolicLogger:
    """
    Comprehensive logging system for Neurosymbolic-LLM training.
    Tracks performance, memory usage, deadlocks, errors, and training metrics.
    """
    
    def __init__(
        self,
        log_dir: str = "logs",
        experiment_name: str = "neurosymbolic_llm",
        log_level: str = "INFO",
        enable_file_logging: bool = True,
        enable_console_logging: bool = True,
        enable_wandb: bool = False
    ):
        """
        Initialize logging system.
        
        Args:
            log_dir: Directory for log files
            experiment_name: Name of the experiment
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            enable_file_logging: Whether to save logs to file
            enable_console_logging: Whether to show logs in console
            enable_wandb: Whether to use Weights & Biases
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_name = experiment_name
        self.start_time = time.time()
        self.enable_wandb = enable_wandb
        
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{experiment_name}_{self.timestamp}"
        
        self.metrics_dir = self.log_dir / "metrics"
        self.errors_dir = self.log_dir / "errors"
        self.debug_dir = self.log_dir / "debug"
        
        for dir in [self.metrics_dir, self.errors_dir, self.debug_dir]:
            dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = self._setup_logger(
            log_level, 
            enable_file_logging, 
            enable_console_logging
        )
        
        self.metrics = {
            "training": [],
            "validation": [],
            "gpu_memory": [],
            "timings": {},
            "deadlock_warnings": [],
            "errors": []
        }
        
        self.last_activity = time.time()
        self.deadlock_threshold = 300
        self._start_deadlock_monitor()
        
        self.logger.info("="*80)
        self.logger.info(f"Neurosymbolic-LLM Logger Initialized")
        self.logger.info(f"Run ID: {self.run_id}")
        self.logger.info(f"Log Directory: {self.log_dir.absolute()}")
        self.logger.info("="*80)
    
    def _setup_logger(
        self, 
        log_level: str, 
        enable_file: bool, 
        enable_console: bool
    ) -> logging.Logger:
        """Configure Python logger."""
        
        logger = logging.getLogger(self.run_id)
        logger.setLevel(getattr(logging, log_level.upper()))
        logger.handlers = []
        
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        if enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        if enable_file:
            log_file = self.log_dir / f"{self.run_id}.log"
            file_handler = logging.FileHandler(log_file, mode='w')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def _start_deadlock_monitor(self):
        """Start thread that monitors for deadlocks."""
        
        def monitor():
            while True:
                time.sleep(30)
                elapsed = time.time() - self.last_activity
                
                if elapsed > self.deadlock_threshold:
                    warning_msg = (
                        f"POSIBLE DEADLOCK: Sin actividad por {elapsed:.1f}s\n"
                        f"Última actividad: {datetime.datetime.fromtimestamp(self.last_activity)}\n"
                        f"GPU Memory: {self._get_gpu_memory_info()}"
                    )
                    self.logger.warning(warning_msg)
                    self.metrics["deadlock_warnings"].append({
                        "timestamp": time.time(),
                        "elapsed_since_activity": elapsed,
                        "gpu_memory": self._get_gpu_memory_info()
                    })
        
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
    
    def ping(self, message: str = ""):
        """Update activity timestamp to prevent false deadlock warnings."""
        self.last_activity = time.time()
        if message:
            self.logger.debug(f"Activity ping: {message}")
    
    def log_system_info(self):
        """Log system information."""
        
        info = {
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "cpu_count": os.cpu_count(),
            "total_ram_gb": psutil.virtual_memory().total / (1024**3)
        }
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                info[f"gpu_{i}"] = {
                    "name": torch.cuda.get_device_name(i),
                    "total_memory_gb": torch.cuda.get_device_properties(i).total_memory / (1024**3)
                }
        
        self.logger.info("System Information:")
        for key, value in info.items():
            self.logger.info(f"  {key}: {value}")
        
        with open(self.debug_dir / f"system_info_{self.timestamp}.json", 'w') as f:
            json.dump(info, f, indent=2)
    
    def log_distributed_setup(self, rank: int, world_size: int, local_rank: int):
        """Log distributed training setup."""
        
        msg = (
            f"Distributed Setup:\n"
            f"  Rank: {rank}/{world_size}\n"
            f"  Local Rank: {local_rank}\n"
            f"  Device: cuda:{local_rank}"
        )
        self.logger.info(msg)
    
    def log_model_info(self, model_name: str, num_parameters: int, hidden_dim: int):
        """Log model information."""
        
        params_billions = num_parameters / 1e9
        msg = (
            f"Model Configuration:\n"
            f"  Name: {model_name}\n"
            f"  Parameters: {params_billions:.2f}B ({num_parameters:,})\n"
            f"  Hidden Dimension: {hidden_dim}"
        )
        self.logger.info(msg)
    
    def log_gpu_memory(self, step: Optional[int] = None, prefix: str = ""):
        """Log GPU memory usage."""
        
        if not torch.cuda.is_available():
            return
        
        memory_info = self._get_gpu_memory_info()
        
        log_msg = f"{prefix} GPU Memory:" if prefix else "GPU Memory:"
        for gpu_id, info in memory_info.items():
            log_msg += (
                f"\n  GPU {gpu_id}: "
                f"{info['allocated_gb']:.2f}GB / {info['total_gb']:.2f}GB "
                f"({info['utilization']:.1f}%)"
            )
        
        self.logger.info(log_msg)
        
        self.metrics["gpu_memory"].append({
            "step": step,
            "timestamp": time.time(),
            "memory": memory_info
        })
    
    def _get_gpu_memory_info(self) -> Dict[int, Dict[str, float]]:
        """Get memory information for all GPUs."""
        
        if not torch.cuda.is_available():
            return {}
        
        memory_info = {}
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            
            memory_info[i] = {
                "allocated_gb": allocated,
                "total_gb": total,
                "utilization": (allocated / total) * 100 if total > 0 else 0
            }
        
        return memory_info
    
    def log_training_step(
        self, 
        epoch: int, 
        step: int, 
        loss: float, 
        metrics: Optional[Dict[str, float]] = None,
        log_gpu: bool = False
    ):
        """Log training step."""
        
        self.ping(f"Training step {step}")
        
        elapsed = time.time() - self.start_time
        
        log_msg = (
            f"Epoch {epoch} | Step {step} | "
            f"Loss: {loss:.4f} | "
            f"Time: {elapsed:.1f}s"
        )
        
        if metrics:
            for key, value in metrics.items():
                log_msg += f" | {key}: {value:.4f}"
        
        self.logger.info(log_msg)
        
        self.metrics["training"].append({
            "epoch": epoch,
            "step": step,
            "loss": loss,
            "metrics": metrics or {},
            "timestamp": time.time()
        })
        
        if log_gpu:
            self.log_gpu_memory(step=step, prefix="Training")
    
    def log_validation_step(
        self, 
        step: int, 
        loss: float, 
        metrics: Optional[Dict[str, float]] = None
    ):
        """Log validation step."""
        
        self.ping(f"Validation step {step}")
        
        log_msg = f"Validation Step {step} | Loss: {loss:.4f}"
        
        if metrics:
            for key, value in metrics.items():
                log_msg += f" | {key}: {value:.4f}"
        
        self.logger.info(log_msg)
        
        self.metrics["validation"].append({
            "step": step,
            "loss": loss,
            "metrics": metrics or {},
            "timestamp": time.time()
        })
    
    def log_testing_step(
        self, 
        step: int, 
        loss: float, 
        metrics: Optional[Dict[str, float]] = None
    ):
        """Log testing step."""
        
        self.ping(f"Testing step {step}")
        
        log_msg = f"Testing Step {step} | Loss: {loss:.4f}"
        
        if metrics:
            for key, value in metrics.items():
                log_msg += f" | {key}: {value:.4f}"
        
        self.logger.info(log_msg)
    
    def log_error(self, error: Exception, context: str = ""):
        """Log an error with context."""
        
        error_msg = f"Error in {context}: {type(error).__name__}: {str(error)}"
        self.logger.error(error_msg)
        
        self.metrics["errors"].append({
            "context": context,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": time.time()
        })
    
    def save_final_metrics(self):
        """Save all collected metrics to file."""
        
        metrics_file = self.metrics_dir / f"final_metrics_{self.timestamp}.json"
        
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        self.logger.info(f"Final metrics saved to {metrics_file}")