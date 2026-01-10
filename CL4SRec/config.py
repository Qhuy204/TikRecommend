"""
Configuration for SASRec-CE Hybrid Recommendation System
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import torch


# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CL4SREC_DIR = BASE_DIR / "CL4SRec"
CHECKPOINT_DIR = CL4SREC_DIR / "checkpoints"


@dataclass
class DataConfig:
    """Data configuration"""
    # Raw data
    raw_data_path: str = str(RAW_DATA_DIR / "tiki_dataset.jsonl")
    
    # Processed data (cached)
    processed_dir: str = str(PROCESSED_DATA_DIR)
    cache_file: str = str(PROCESSED_DATA_DIR / "cl4srec_processed.pkl")
    
    # Filtering
    min_seq_length: int = 3      # Minimum user interactions
    max_seq_length: int = 50     # Maximum sequence length
    min_item_count: int = 5      # Minimum item occurrences
    
    # Train/Val/Test split
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    # Embedding
    embedding_dim: int = 64
    
    # Transformer
    num_attention_heads: int = 2
    num_transformer_blocks: int = 2
    hidden_dim: int = 128
    dropout: float = 0.2
    
    # Sequence
    max_seq_length: int = 50
    
    # Loss function
    loss_type: str = "ce"  # "ce" (cross-entropy) or "bpr"
    label_smoothing: float = 0.1


@dataclass
class PhoBERTConfig:
    """PhoBERT content encoder configuration"""
    enabled: bool = False  # Set True to use PhoBERT
    model_name: str = "vinai/phobert-base"
    max_text_length: int = 128
    output_dim: int = 64
    freeze: bool = True


@dataclass
class FusionConfig:
    """Fusion configuration for hybrid model"""
    fusion_type: str = "add"  # "add", "concat", "gate"
    hidden_dim: int = 128
    dropout: float = 0.2


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Training
    epochs: int = 30
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    
    # Scheduler
    scheduler: str = "cosine"  # "cosine", "step", "none"
    warmup_epochs: int = 3
    
    # Early stopping
    patience: int = 5
    min_delta: float = 1e-4
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    
    # Checkpoints
    checkpoint_dir: str = str(CHECKPOINT_DIR)
    save_best_only: bool = True
    
    # Logging
    log_interval: int = 100


@dataclass
class EvalConfig:
    """Evaluation configuration"""
    k_values: List[int] = field(default_factory=lambda: [5, 10, 20])
    batch_size: int = 256


@dataclass
class Config:
    """Main configuration"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    phobert: PhoBERTConfig = field(default_factory=PhoBERTConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    
    def __post_init__(self):
        # Sync max_seq_length
        self.model.max_seq_length = self.data.max_seq_length
        
        # Create directories
        os.makedirs(self.data.processed_dir, exist_ok=True)
        os.makedirs(self.training.checkpoint_dir, exist_ok=True)


# Default configuration
config = Config()


def get_config(**overrides) -> Config:
    """Get configuration with optional overrides"""
    cfg = Config()
    for key, value in overrides.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg
