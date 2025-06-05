import os
import csv
import numpy as np

import torch
from torch.cuda import memory_allocated, max_memory_allocated, memory_reserved, max_memory_reserved

class Logger:
    def __init__(self, run_name):
        self.run_name = run_name
        self.base_dir = 'out'

        # Directories
        self.metric_dir = f'{self.base_dir}/{self.run_name}/metrics'
        self.model_dir = f'{self.base_dir}/{self.run_name}/model'
        self.memory_dir = f'{self.base_dir}/{self.run_name}/memory'

        # Files
        self.train_metric_file = f'{self.metric_dir}/train_metrics.csv'
        self.eval_metric_file = f'{self.metric_dir}/eval_metrics.csv'
        self.train_memory_file = f'{self.memory_dir}/train_memory.csv'
        self.eval_memory_file = f'{self.memory_dir}/eval_memory.csv'

        # Memory stats to save
        self.epoch = -1
        self.allocated_mem       = []
        self.max_allocated_mem   = []
        self.reserved_mem        = []
        self.max_reserved_mem    = []

    def init_epoch(self, epoch: int):
        self.epoch = epoch
        self.allocated_mem       = []
        self.max_allocated_mem   = []
        self.reserved_mem        = []
        self.max_reserved_mem    = []

    def update_memory(self):
        self.allocated_mem.append(memory_allocated())
        self.max_allocated_mem.append(max_memory_allocated())
        self.reserved_mem.append(memory_reserved())
        self.max_reserved_mem.append(max_memory_reserved())

    def create_folders_and_files(self):
        if not os.path.exists(self.metric_dir):
            os.makedirs(self.metric_dir)

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        if not os.path.exists(self.memory_dir):
            os.makedirs(self.memory_dir)

        # Initialize empty metric saving file with
        with open(self.train_metric_file, mode='w', newline='') as _:
            pass
        with open(self.eval_metric_file, mode='w', newline='') as _:
            pass
        with open(self.train_memory_file, mode='w', newline='') as _:
            pass
        with open(self.eval_memory_file, mode='w', newline='') as _:
            pass

    def save_model(self, model_state):
        torch.save(model_state, f"{self.model_dir}/best_model.pt")

    def save_metrics_and_memory(self, metrics_dir: dict, stage: str ='train'):
        memory_dir = {
            'allocated_mem': np.mean(self.allocated_mem),
            'max_allocated_mem': np.mean(self.max_allocated_mem),
            'reserved_mem': np.mean(self.reserved_mem),
            'max_reserved_mem': np.mean(self.max_allocated_mem),
        }

        if stage == 'train':
            self._write_to_file(self.train_metric_file, metrics_dir)
            self._write_to_file(self.train_memory_file, memory_dir)
        elif stage == 'eval':
            self._write_to_file(self.eval_metric_file, metrics_dir)
            self._write_to_file(self.eval_memory_file, memory_dir)
        else:
            raise ValueError(f'Stage {stage} is unknown. Must be "train" or "eval".')


    @staticmethod
    def _write_to_file(file_name, content_dict):
        headers, content = zip(*content_dict.items())

        with open(file_name, mode='a', newline='') as file:
            writer = csv.writer(file)

            if os.path.getsize(file_name) == 0:
                # If file is empty, add headers first
                writer.writerow(headers)

            writer.writerow(content)