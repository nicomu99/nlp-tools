"""TensorBoard logging utilities for training and CUDA memory tracking.

This module provides a Logger class for recording scalar metrics and
GPU memory statistics during model training using TensorBoard.
"""
import numpy as np

from torch.cuda import memory_allocated, max_memory_allocated, memory_reserved, max_memory_reserved

from torch.utils.tensorboard import SummaryWriter


class Logger:
    """Logger utility for tracking metrics and memory usage using Tensorboard.

    The class collects GPU memory statistics during an epoch and logs their mean values
    together with arbitrary scalar values at the end of each epoch.
    """
    def __init__(self, run_name: str):
        """Initialize the logger and TensorBoard writer.

        Args:
            run_name: A string identifier for the name of the current run to be used by TensorBoard.
        """
        self.run_name = run_name
        self.writer = SummaryWriter(f"runs/{run_name}")

        # Memory stats to save
        self.epoch = 0
        self.allocated_mem = []
        self.max_allocated_mem = []
        self.reserved_mem = []
        self.max_reserved_mem = []

    def init_epoch(self):
        """Reset memory statistics at the beginning of a new epoch.

        This method should be called once per epoch before any calls to :meth:`update`.
        """
        self.allocated_mem = []
        self.max_allocated_mem = []
        self.reserved_mem = []
        self.max_reserved_mem = []

    def log_epoch(self, stage: str, metrics_dir: dict):
        """Log aggregated metrics and GPU memory statistics for an epoch.

        All provided metrics are logged under ``{stage}/`` and memory
        statistics under ``mem/{stage}/`` in TensorBoard.

        Args:
            stage: Stage identifier (e.g. "train", "eval" or "test")
            metrics_dir: A dictionary mapping metric names to scalar values.
        """
        metrics_dir = {f"{stage}/{k}": v for k, v in metrics_dir.items()}

        metrics_dir[f"mem/{stage}/allocated_mem"] = np.mean(self.allocated_mem)
        metrics_dir[f"mem/{stage}/max_allocated_mem"] = np.mean(self.max_allocated_mem)
        metrics_dir[f"mem/{stage}/reserved_mem"] = np.mean(self.reserved_mem)
        metrics_dir[f"mem/{stage}/max_reserved_mem"] = np.mean(self.max_allocated_mem)

        for key, metric in metrics_dir.items():
            self.writer.add_scalar(key, metric, self.epoch)
        self.writer.flush()

    def update(self):
        """Record current CUDA memory statistics.

        This method is typically called once per iteration (e.g. per batch)
        to accumulate memory usage data over an epoch.
        """
        self.allocated_mem.append(memory_allocated())
        self.max_allocated_mem.append(max_memory_allocated())
        self.reserved_mem.append(memory_reserved())
        self.max_reserved_mem.append(max_memory_reserved())

    def close(self):
        """Close the TensorBoard writer and release resources."""
        self.writer.close()
