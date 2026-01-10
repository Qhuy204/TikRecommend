"""
Hybrid LightGCN + TF-IDF Recommender Model
Weighted fusion of collaborative filtering and content-based filtering
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
import pickle

import numpy as np
import torch
from scipy import sparse

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config


class DictWrapper:
    """Wrapper to access dict values as attributes"""
    def __init__(self, data: dict):
        self._data = data
    
    def __getattr__(self, name):
        if name.startswith('_'):
            return super().__getattribute__(name)
        return self._data.get(name)
    
    def get(self, key, default=None):
        return self._data.get(key, default)
    
    def keys(self):
        return self._data.keys()


class HybridRecommender:
    """
    Hybrid Recommender combining LightGCN and TF-IDF
    
    Score fusion:
        hybrid_score = α × CF_score + (1-α) × CB_score
    
    Where:
        - CF_score: LightGCN collaborative filtering score
        - CB_score: TF-IDF content-based score
        - α: fusion weight (default 0.7 for LightGCN)
    """
    
    def __init__(
        self,
        lightgcn_model,
        lightgcn_data,
        tfidf_matrix: sparse.csr_matrix,
        tfidf_data: dict,
        alpha: float = 0.7,
        device: str = "cuda"
    ):
        """
        Initialize Hybrid Recommender
        
        Args:
            lightgcn_model: Trained LightGCN model
            lightgcn_data: LightGCN processed data (user2idx, item2idx, etc.)
            tfidf_matrix: Sparse TF-IDF matrix (normalized)
            tfidf_data: TF-IDF processed data (item2idx, idx2item, item_info)
            alpha: Weight for LightGCN (1-alpha for TF-IDF)
            device: Device for LightGCN inference
        """
        self.lightgcn = lightgcn_model
        self.lightgcn_data = lightgcn_data
        self.tfidf_matrix = tfidf_matrix
        self.tfidf_data = tfidf_data
        self.alpha = alpha
        self.device = device
        
        # Pre-normalize TF-IDF matrix for fast cosine similarity
        self._normalize_tfidf()
        
        # Build item ID mappings between LightGCN and TF-IDF
        self._build_id_mappings()
        
        print(f"HybridRecommender initialized")
        print(f"   alpha (LightGCN): {self.alpha:.2f}")
        print(f"   1-alpha (TF-IDF): {1-self.alpha:.2f}")
        print(f"   Common items: {len(self.common_items):,}")
    
    def _normalize_tfidf(self):
        """L2 normalize TF-IDF matrix for cosine similarity"""
        norms = sparse.linalg.norm(self.tfidf_matrix, axis=1)
        norms[norms == 0] = 1
        self.tfidf_normalized = self.tfidf_matrix.multiply(
            1 / norms.reshape(-1, 1)
        ).tocsr()
    
    def _build_id_mappings(self):
        """Build mappings between LightGCN and TF-IDF item indices"""
        # Get common items between both systems
        lgcn_items = set(self.lightgcn_data.item2idx.keys())
        tfidf_items = set(self.tfidf_data['item2idx'].keys())
        self.common_items = lgcn_items & tfidf_items
        
        # Build mapping: LightGCN idx -> TF-IDF idx
        self.lgcn_to_tfidf = {}
        self.tfidf_to_lgcn = {}
        
        for item_id in self.common_items:
            lgcn_idx = self.lightgcn_data.item2idx[item_id]
            tfidf_idx = self.tfidf_data['item2idx'][item_id]
            self.lgcn_to_tfidf[lgcn_idx] = tfidf_idx
            self.tfidf_to_lgcn[tfidf_idx] = lgcn_idx
    
    def set_alpha(self, alpha: float):
        """Update fusion weight"""
        self.alpha = alpha
    
    @staticmethod
    def normalize_scores(scores: np.ndarray) -> np.ndarray:
        """Min-max normalization to [0, 1]"""
        min_s, max_s = scores.min(), scores.max()
        if max_s == min_s:
            return np.zeros_like(scores)
        return (scores - min_s) / (max_s - min_s)
    
    def get_cf_scores(self, user_idx: int) -> np.ndarray:
        """Get LightGCN scores for all items"""
        self.lightgcn.eval()
        with torch.no_grad():
            all_user_emb, all_item_emb = self.lightgcn.get_all_embeddings()
            user_emb = all_user_emb[user_idx]
            scores = torch.matmul(user_emb, all_item_emb.T)
            return scores.cpu().numpy()
    
    def get_cb_scores(self, user_history: List[int]) -> np.ndarray:
        """
        Get TF-IDF scores based on user's interaction history
        
        Args:
            user_history: List of item indices (TF-IDF indices)
            
        Returns:
            Similarity scores for all items
        """
        if not user_history:
            return np.zeros(self.tfidf_matrix.shape[0])
        
        # Get user profile as average of history items
        valid_indices = [idx for idx in user_history if idx < self.tfidf_normalized.shape[0]]
        if not valid_indices:
            return np.zeros(self.tfidf_matrix.shape[0])
        
        history_vecs = self.tfidf_normalized[valid_indices]
        
        # Average profile
        if len(valid_indices) == 1:
            user_profile = history_vecs
        else:
            user_profile = history_vecs.mean(axis=0)
            # Convert to sparse if needed
            if isinstance(user_profile, np.matrix):
                user_profile = np.asarray(user_profile).flatten()
                user_profile = sparse.csr_matrix(user_profile)
        
        # Compute similarity to all items
        similarities = (self.tfidf_normalized @ user_profile.T).toarray().flatten()
        
        return similarities
    
    def recommend(
        self,
        user_idx: int,
        user_history_lgcn: Set[int],
        top_k: int = 10,
        exclude_history: bool = True
    ) -> Tuple[List[int], List[float]]:
        """
        Get hybrid recommendations for a user
        
        Args:
            user_idx: User index (LightGCN index)
            user_history_lgcn: Set of item indices user has interacted with (LightGCN indices)
            top_k: Number of recommendations
            exclude_history: Whether to exclude items in history
            
        Returns:
            Tuple of (item_indices, scores) in LightGCN index space
        """
        num_lgcn_items = self.lightgcn_data.num_items
        
        # Get CF scores (already in LightGCN index space)
        cf_scores = self.get_cf_scores(user_idx)
        cf_norm = self.normalize_scores(cf_scores)
        
        # Convert user history to TF-IDF indices
        user_history_tfidf = [
            self.lgcn_to_tfidf[idx] 
            for idx in user_history_lgcn 
            if idx in self.lgcn_to_tfidf
        ]
        
        # Get CB scores (in TF-IDF index space)
        cb_scores_tfidf = self.get_cb_scores(user_history_tfidf)
        
        # Convert CB scores to LightGCN index space
        cb_scores = np.zeros(num_lgcn_items)
        for lgcn_idx in range(num_lgcn_items):
            if lgcn_idx in self.lgcn_to_tfidf:
                tfidf_idx = self.lgcn_to_tfidf[lgcn_idx]
                cb_scores[lgcn_idx] = cb_scores_tfidf[tfidf_idx]
        
        cb_norm = self.normalize_scores(cb_scores)
        
        # Hybrid fusion
        hybrid_scores = self.alpha * cf_norm + (1 - self.alpha) * cb_norm
        
        # Exclude history items
        if exclude_history:
            for idx in user_history_lgcn:
                if idx < len(hybrid_scores):
                    hybrid_scores[idx] = -1
        
        # Get top-K
        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
        top_scores = hybrid_scores[top_indices]
        
        return top_indices.tolist(), top_scores.tolist()
    
    def recommend_with_details(
        self,
        user_idx: int,
        user_history_lgcn: Set[int],
        top_k: int = 10
    ) -> List[Dict]:
        """
        Get recommendations with full item details
        
        Returns:
            List of dicts with product_id, name, category, price, score
        """
        top_indices, top_scores = self.recommend(user_idx, user_history_lgcn, top_k)
        
        results = []
        for idx, score in zip(top_indices, top_scores):
            item_id = self.lightgcn_data.idx2item.get(idx)
            info = self.lightgcn_data.item_info.get(item_id, {})
            
            results.append({
                'product_id': item_id,
                'name': info.get('name', ''),
                'category': info.get('category', ''),
                'price': info.get('price', 0),
                'score': score
            })
        
        return results


def load_hybrid_recommender(
    lightgcn_checkpoint: str = None,
    lightgcn_data_path: str = None,
    tfidf_cache_path: str = None,
    tfidf_matrix_path: str = None,
    alpha: float = 0.7,
    device: str = "cuda"
) -> HybridRecommender:
    """
    Load and initialize HybridRecommender
    
    Returns:
        Initialized HybridRecommender
    """
    # Use config defaults if not specified
    cfg = config.hybrid
    lightgcn_checkpoint = lightgcn_checkpoint or cfg.lightgcn_checkpoint
    lightgcn_data_path = lightgcn_data_path or cfg.lightgcn_data
    tfidf_cache_path = tfidf_cache_path or cfg.tfidf_cache
    tfidf_matrix_path = tfidf_matrix_path or cfg.tfidf_matrix
    
    # Resolve paths relative to this script
    base_dir = Path(__file__).parent
    lightgcn_checkpoint = base_dir / lightgcn_checkpoint
    lightgcn_data_path = base_dir / lightgcn_data_path
    tfidf_cache_path = base_dir / tfidf_cache_path
    tfidf_matrix_path = base_dir / tfidf_matrix_path
    
    print("Loading data and models...")
    
    # Load LightGCN data
    print(f"   Loading LightGCN data: {lightgcn_data_path}")
    with open(lightgcn_data_path, 'rb') as f:
        lightgcn_data_raw = pickle.load(f)
    
    # Wrap dict in DictWrapper for attribute access
    if isinstance(lightgcn_data_raw, dict):
        lightgcn_data = DictWrapper(lightgcn_data_raw)
    else:
        lightgcn_data = lightgcn_data_raw
    
    # Load LightGCN model
    print(f"   Loading LightGCN model: {lightgcn_checkpoint}")
    
    # We need to load LightGCN without importing its config module
    # because it conflicts with our config module.
    # Solution: Manually define the LightGCN class inline
    
    import torch.nn as nn
    import scipy.sparse as sp_module
    
    def sparse_to_torch(adj, device='cpu'):
        coo = adj.tocoo()
        indices = torch.LongTensor([coo.row, coo.col])
        values = torch.FloatTensor(coo.data)
        sparse_tensor = torch.sparse_coo_tensor(indices, values, coo.shape)
        return sparse_tensor.to(device)
    
    class LightGCN(nn.Module):
        def __init__(self, num_users, num_items, adj_matrix, embedding_dim=64, num_layers=3):
            super().__init__()
            self.num_users = num_users
            self.num_items = num_items
            self.embedding_dim = embedding_dim
            self.num_layers = num_layers
            
            self.user_embedding = nn.Embedding(num_users, embedding_dim)
            self.item_embedding = nn.Embedding(num_items, embedding_dim)
            
            nn.init.xavier_uniform_(self.user_embedding.weight)
            nn.init.xavier_uniform_(self.item_embedding.weight)
            
            self.register_buffer('adj_matrix', None)
            if adj_matrix is not None:
                sparse_tensor = sparse_to_torch(adj_matrix, self.user_embedding.weight.device)
                self.adj_matrix = sparse_tensor
        
        def get_all_embeddings(self):
            all_emb = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
            emb_list = [all_emb]
            for _ in range(self.num_layers):
                all_emb = torch.sparse.mm(self.adj_matrix, all_emb)
                emb_list.append(all_emb)
            final_emb = torch.stack(emb_list, dim=0).mean(dim=0)
            user_emb = final_emb[:self.num_users]
            item_emb = final_emb[self.num_users:]
            return user_emb, item_emb
    
    # Check if CUDA available
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("   CUDA not available, using CPU")
    
    lightgcn_model = LightGCN(
        num_users=lightgcn_data.num_users,
        num_items=lightgcn_data.num_items,
        adj_matrix=lightgcn_data.adj_matrix
    )
    
    ckpt = torch.load(lightgcn_checkpoint, map_location=device)
    lightgcn_model.load_state_dict(ckpt['model_state_dict'])
    lightgcn_model = lightgcn_model.to(device)
    lightgcn_model.adj_matrix = lightgcn_model.adj_matrix.to(device)
    lightgcn_model.eval()
    
    print(f"   LightGCN loaded (epoch {ckpt['epoch']}, HR@10={ckpt['best_metric']:.4f})")
    
    # Load TF-IDF data
    print(f"   Loading TF-IDF data: {tfidf_cache_path}")
    with open(tfidf_cache_path, 'rb') as f:
        tfidf_data = pickle.load(f)
    
    print(f"   Loading TF-IDF matrix: {tfidf_matrix_path}")
    tfidf_matrix = sparse.load_npz(tfidf_matrix_path)
    
    # Create hybrid recommender
    hybrid = HybridRecommender(
        lightgcn_model=lightgcn_model,
        lightgcn_data=lightgcn_data,
        tfidf_matrix=tfidf_matrix,
        tfidf_data=tfidf_data,
        alpha=alpha,
        device=device
    )
    
    return hybrid


if __name__ == "__main__":
    print("Testing HybridRecommender...")
    
    hybrid = load_hybrid_recommender(alpha=0.7)
    
    # Test with a random user
    import random
    user_ids = list(hybrid.lightgcn_data.user2idx.keys())
    test_user_id = random.choice(user_ids)
    test_user_idx = hybrid.lightgcn_data.user2idx[test_user_id]
    user_history = hybrid.lightgcn_data.user_pos_items.get(test_user_idx, set())
    
    print(f"\nTesting with user {test_user_id} ({len(user_history)} interactions)")
    
    recs = hybrid.recommend_with_details(test_user_idx, user_history, top_k=5)
    
    print("\nTop 5 Hybrid Recommendations:")
    for i, rec in enumerate(recs, 1):
        print(f"\n{i}. {rec['name'][:50]}...")
        print(f"   Category: {rec['category']}")
        print(f"   Price: {rec['price']:,.0f} VND")
        print(f"   Score: {rec['score']:.4f}")
    
    print("\nHybridRecommender test passed!")
