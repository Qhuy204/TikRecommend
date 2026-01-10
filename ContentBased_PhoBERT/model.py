"""
PhoBERT Content-Based Recommender Model
Cosine similarity using pre-computed PhoBERT embeddings
"""

from typing import List, Dict
import torch
from torch.nn import functional as F


class PhoBERTRecommender:
    """
    PhoBERT-based content recommender
    
    Algorithm:
        sim(i, j) = cosine(emb[i], emb[j])
                  = emb[i] · emb[j]  (if normalized)
    """
    
    def __init__(
        self,
        embeddings: torch.Tensor,
        item2idx: Dict[int, int],
        idx2item: Dict[int, int],
        item_info: Dict[int, dict],
        device: str = "cpu"
    ):
        """
        Initialize recommender with pre-computed embeddings
        
        Args:
            embeddings: Normalized embeddings (num_items x 768)
            item2idx: product_id -> index mapping
            idx2item: index -> product_id mapping
            item_info: product_id -> {name, description, category, price}
            device: Device for computation
        """
        self.embeddings = embeddings.to(device)
        self.item2idx = item2idx
        self.idx2item = idx2item
        self.item_info = item_info
        self.num_items = embeddings.shape[0]
        self.device = device
        
        # Ensure normalized
        norms = self.embeddings.norm(dim=1, keepdim=True)
        if not torch.allclose(norms, torch.ones_like(norms), atol=0.01):
            self.embeddings = F.normalize(self.embeddings, dim=1)
    
    def get_similar_items(
        self,
        item_id: int,
        top_k: int = 10,
        exclude_self: bool = True
    ) -> List[Dict]:
        """
        Find top-K similar items by cosine similarity
        """
        if item_id not in self.item2idx:
            return []
        
        query_idx = self.item2idx[item_id]
        query_vec = self.embeddings[query_idx]
        
        # Cosine similarity (dot product with normalized vectors)
        similarities = torch.matmul(self.embeddings, query_vec)
        
        if exclude_self:
            similarities[query_idx] = -1
        
        # Get top-K
        top_scores, top_indices = torch.topk(similarities, k=min(top_k, self.num_items))
        
        results = []
        for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
            sim_id = self.idx2item[idx]
            info = self.item_info.get(sim_id, {})
            results.append({
                'product_id': sim_id,
                'name': info.get('name', ''),
                'category': info.get('category', ''),
                'price': info.get('price', 0),
                'similarity': score,
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
        """
        valid_indices = [
            self.item2idx[item_id]
            for item_id in item_ids
            if item_id in self.item2idx
        ]
        
        if not valid_indices:
            return []
        
        # Get query vectors and average
        query_vecs = self.embeddings[valid_indices]
        avg_query = query_vecs.mean(dim=0)
        avg_query = F.normalize(avg_query, dim=0)
        
        # Similarities
        similarities = torch.matmul(self.embeddings, avg_query)
        
        if exclude_input:
            for idx in valid_indices:
                similarities[idx] = -1
        
        # Get top-K
        top_scores, top_indices = torch.topk(similarities, k=min(top_k, self.num_items))
        
        results = []
        for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
            rec_id = self.idx2item[idx]
            info = self.item_info.get(rec_id, {})
            results.append({
                'product_id': rec_id,
                'name': info.get('name', ''),
                'category': info.get('category', ''),
                'price': info.get('price', 0),
                'score': score,
            })
        
        return results
