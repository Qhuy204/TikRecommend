"""
Configuration for TF-IDF Content-Based Recommender
"""

from dataclasses import dataclass, field
from pathlib import Path
import os


@dataclass
class TFIDFConfig:
    """TF-IDF vectorizer configuration"""
    max_features: int = 20000       # Vocabulary size (increased for descriptions)
    ngram_range: tuple = (1, 2)     # Unigrams + bigrams
    min_df: int = 2                 # Min document frequency
    max_df: float = 0.95            # Max document frequency
    max_text_length: int = 2000     # Max chars (descriptions can be long)


@dataclass
class DataConfig:
    """Data paths and preprocessing settings"""
    raw_data_path: str = "../data/clean/tiki_dataset_clean.jsonl"
    cache_file: str = "../data/processed/tfidf_processed.pkl"
    tfidf_matrix_file: str = "../data/processed/tfidf_matrix.npz"
    
    min_item_count: int = 5         # Min interactions per item
    min_user_count: int = 3         # Min interactions per user


@dataclass
class Config:
    """Main configuration"""
    tfidf: TFIDFConfig = field(default_factory=TFIDFConfig)
    data: DataConfig = field(default_factory=DataConfig)


config = Config()
