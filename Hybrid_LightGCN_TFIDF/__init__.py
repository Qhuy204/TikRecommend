"""
Hybrid LightGCN + TF-IDF Recommendation System
Combines collaborative filtering with content-based filtering
"""

from .config import config, HybridConfig
from .model import HybridRecommender

__all__ = ['config', 'HybridConfig', 'HybridRecommender']
