"""
Configuration for LightGCN
"""

from dataclasses import dataclass, field
from pathlib import Path
import torch


@dataclass
class DataConfig:
    raw_data_path: str = "../data/raw/tiki_dataset.jsonl"
    cache_file: str = "../data/processed/lightgcn_processed.pkl"
    
    min_item_count: int = 5
    min_user_count: int = 3


@dataclass
class ModelConfig:
    embedding_dim: int = 64
    num_layers: int = 3  # Number of GCN layers
    dropout: float = 0.0  # LightGCN typically doesn't use dropout


@dataclass
class TrainingConfig:
    batch_size: int = 2048
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    
    num_negatives: int = 1  # BPR uses 1 negative
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir: str = "checkpoints"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


config = Config()
