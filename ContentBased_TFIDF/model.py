"""
TF-IDF Content-Based Recommender Model
Cosine similarity for item-to-item recommendations
"""

from typing import List, Dict, Optional
import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity


class TFIDFRecommender:
    """
    TF-IDF based content recommender
    
    Algorithm:
        sim(i, j) = cosine(tfidf[i], tfidf[j])
                  = (tfidf[i] · tfidf[j]) / (||tfidf[i]|| × ||tfidf[j]||)
    """
    
    def __init__(
        self,
        tfidf_matrix: sparse.csr_matrix,
        item2idx: Dict[int, int],
        idx2item: Dict[int, int],
        item_info: Dict[int, dict],
    ):
        """
        Initialize recommender with pre-computed TF-IDF matrix
        
        Args:
            tfidf_matrix: Sparse TF-IDF matrix (num_items x vocab_size)
            item2idx: product_id -> index mapping
            idx2item: index -> product_id mapping
            item_info: product_id -> {name, description, category, price}
        """
        self.tfidf_matrix = tfidf_matrix
        self.item2idx = item2idx
        self.idx2item = idx2item
        self.item_info = item_info
        self.num_items = tfidf_matrix.shape[0]
        
        # Pre-normalize for faster cosine similarity
        self._normalize_matrix()
    
    def _normalize_matrix(self):
        """L2 normalize each row for fast cosine similarity"""
        # After normalization: cosine(a, b) = dot(a, b)
        norms = sparse.linalg.norm(self.tfidf_matrix, axis=1)
        norms[norms == 0] = 1  # Avoid division by zero
        # multiply() returns coo_matrix, convert back to csr for indexing
        self.tfidf_normalized = self.tfidf_matrix.multiply(1 / norms.reshape(-1, 1)).tocsr()
    
    def get_similar_items(
        self,
        item_id: int,
        top_k: int = 10,
        exclude_self: bool = True
    ) -> List[Dict]:
        """
        Find top-K similar items by cosine similarity
        
        Args:
            item_id: Query product ID
            top_k: Number of similar items to return
            exclude_self: Whether to exclude the query item
            
        Returns:
            List of dicts with product_id, name, category, price, similarity
        """
        if item_id not in self.item2idx:
            return []
        
        query_idx = self.item2idx[item_id]
        query_vec = self.tfidf_normalized[query_idx]
        
        # Compute similarities (dot product with normalized vectors = cosine)
        similarities = (self.tfidf_normalized @ query_vec.T).toarray().flatten()
        
        # Exclude self
        if exclude_self:
            similarities[query_idx] = -1
        
        # Get top-K
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            sim_id = self.idx2item[idx]
            info = self.item_info.get(sim_id, {})
            results.append({
                'product_id': sim_id,
                'name': info.get('name', ''),
                'category': info.get('category', ''),
                'price': info.get('price', 0),
                'similarity': float(similarities[idx]),
            })
        
        return results
    
    def recommend_for_items(
        self,
        item_ids: List[int],
        top_k: int = 10,
        exclude_input: bool = True
    ) -> List[Dict]:
        """
        Recommend items similar to a set of input items
        Aggregation: Average similarity across all input items
        
        Args:
            item_ids: List of product IDs (e.g., user's liked items)
            top_k: Number of recommendations
            exclude_input: Whether to exclude input items from results
            
        Returns:
            List of recommended items
        """
        # Get valid indices
        valid_indices = [
            self.item2idx[item_id]
            for item_id in item_ids
            if item_id in self.item2idx
        ]
        
        if not valid_indices:
            return []
        
        # Get query vectors
        query_vecs = self.tfidf_normalized[valid_indices]
        
        # Average similarity across all query items
        if len(valid_indices) == 1:
            avg_similarities = (self.tfidf_normalized @ query_vecs.T).toarray().flatten()
        else:
            similarities = (self.tfidf_normalized @ query_vecs.T).toarray()
            avg_similarities = similarities.mean(axis=1)
        
        # Exclude input items
        if exclude_input:
            for idx in valid_indices:
                avg_similarities[idx] = -1
        
        # Get top-K
        top_indices = np.argsort(avg_similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            rec_id = self.idx2item[idx]
            info = self.item_info.get(rec_id, {})
            results.append({
                'product_id': rec_id,
                'name': info.get('name', ''),
                'category': info.get('category', ''),
                'price': info.get('price', 0),
                'score': float(avg_similarities[idx]),
            })
        
        return results
