"""
Configuration for SASRec + PhoBERT Fusion Model
"""

from dataclasses import dataclass, field
from typing import List
import torch


@dataclass
class DataConfig:
    """Data configuration"""
    raw_data_path: str = "../data/raw/tiki_dataset.jsonl"
    processed_dir: str = "../data/processed"
    
    # Sequence settings
    max_seq_length: int = 50  # Max user history length
    min_seq_length: int = 3   # Min interactions to include user
    
    # Train/Val/Test split
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1


@dataclass
class SASRecConfig:
    """SASRec model configuration"""
    embedding_dim: int = 64
    num_attention_heads: int = 2
    num_transformer_blocks: int = 2
    hidden_dim: int = 128
    dropout: float = 0.2
    max_seq_length: int = 50


@dataclass
class PhoBERTConfig:
    """PhoBERT encoder configuration"""
    model_name: str = "vinai/phobert-base"
    max_text_length: int = 128
    output_dim: int = 64  # Projection output dimension
    freeze: bool = True   # Freeze PhoBERT weights


@dataclass  
class FusionConfig:
    """Fusion model configuration"""
    fusion_type: str = "concat"  # "concat", "attention", "gate"
    hidden_dim: int = 128
    output_dim: int = 64
    dropout: float = 0.2


@dataclass
class TrainingConfig:
    """Training configuration"""
    epochs: int = 30
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    
    # Early stopping
    patience: int = 5
    min_delta: float = 1e-4
    
    # Negative sampling
    num_negatives: int = 4
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Checkpoints
    save_dir: str = "checkpoints"
    save_best_only: bool = True


@dataclass
class Config:
    """Main configuration combining all configs"""
    data: DataConfig = field(default_factory=DataConfig)
    sasrec: SASRecConfig = field(default_factory=SASRecConfig)
    phobert: PhoBERTConfig = field(default_factory=PhoBERTConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    def __post_init__(self):
        # Sync max_seq_length
        self.sasrec.max_seq_length = self.data.max_seq_length


# Default configuration instance
default_config = Config()


def get_config(**kwargs) -> Config:
    """Get configuration with optional overrides"""
    config = Config()
    
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config
