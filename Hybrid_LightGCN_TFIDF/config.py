"""
Configuration for Hybrid LightGCN + TF-IDF Recommender
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class HybridConfig:
    """Hybrid model configuration"""
    # LightGCN paths
    lightgcn_checkpoint: str = "../LightGCN/checkpoints/best_model.pt"
    lightgcn_data: str = "../data/processed/lightgcn_processed.pkl"
    
    # TF-IDF paths
    tfidf_cache: str = "../data/processed/tfidf_processed.pkl"
    tfidf_matrix: str = "../data/processed/tfidf_matrix.npz"
    
    # Fusion weight: α for LightGCN, (1-α) for TF-IDF
    # Best α from tuning: 0.80 (HR@10=29.80%, NDCG@10=17.70%)
    alpha: float = 0.8
    
    # Recommendation settings
    top_k: int = 10
    
    # Device
    device: str = "cuda"


@dataclass
class EvalConfig:
    """Evaluation configuration"""
    sample_users: int = 5000
    top_k: int = 10
    alpha_values: list = field(default_factory=lambda: [0.0, 0.3, 0.5, 0.7, 1.0])


@dataclass 
class Config:
    """Main configuration"""
    hybrid: HybridConfig = field(default_factory=HybridConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


config = Config()
