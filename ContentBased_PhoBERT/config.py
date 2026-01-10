"""
Configuration for PhoBERT Content-Based Recommender
"""

from dataclasses import dataclass, field
from pathlib import Path
import torch


@dataclass
class PhoBERTConfig:
    """PhoBERT encoder configuration"""
    model_name: str = "vinai/phobert-base"
    max_length: int = 256           # PhoBERT max is 256 tokens
    embedding_dim: int = 768        # PhoBERT hidden size
    batch_size: int = 16            # Reduced for longer sequences
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class DataConfig:
    """Data paths and preprocessing settings"""
    raw_data_path: str = "../data/clean/tiki_dataset_clean.jsonl"
    cache_file: str = "../data/processed/phobert_processed.pkl"
    embeddings_file: str = "../data/processed/phobert_embeddings.pt"
    
    min_item_count: int = 5         # Min interactions per item
    min_user_count: int = 3         # Min interactions per user
    max_text_length: int = 2000     # Max chars (descriptions can be long)


@dataclass
class Config:
    """Main configuration"""
    phobert: PhoBERTConfig = field(default_factory=PhoBERTConfig)
    data: DataConfig = field(default_factory=DataConfig)


config = Config()
