"""
Recommender Interface for SASRec + PhoBERT Fusion
Provides easy-to-use API for generating recommendations
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

import torch

from config import default_config, Config
from models import SASRecModel, SASRecPhoBERTFusion
from data_processor import TikiDataProcessor


class SASRecRecommender:
    """
    Recommender class for serving predictions
    """
    
    def __init__(
        self,
        model: Optional[torch.nn.Module] = None,
        data_processor: Optional[TikiDataProcessor] = None,
        device: str = None
    ):
        self.model = model
        self.data_processor = data_processor
        self.device = device or default_config.training.device
        
        if self.model is not None:
            self.model = self.model.to(self.device)
            self.model.eval()
    
    @classmethod
    def load(
        cls,
        checkpoint_path: str,
        data_processor: TikiDataProcessor,
        model_type: str = 'sasrec'  # 'sasrec' or 'fusion'
    ) -> 'SASRecRecommender':
        """Load a trained recommender from checkpoint"""
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        num_items = data_processor.num_items
        
        if model_type == 'sasrec':
            model = SASRecModel(num_items)
        else:
            model = SASRecPhoBERTFusion(num_items)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        recommender = cls(model, data_processor)
        print(f"Loaded recommender from {checkpoint_path}")
        
        return recommender
    
    def recommend_for_user(
        self,
        user_id: int,
        top_k: int = 10,
        exclude_history: bool = True
    ) -> List[Dict]:
        """
        Generate recommendations for a user
        
        Args:
            user_id: Original user ID
            top_k: Number of recommendations
            exclude_history: Whether to exclude items user already interacted with
            
        Returns:
            List of {product_id, score, name, category}
        """
        if self.data_processor is None:
            raise ValueError("Data processor required for user recommendations")
        
        # Get user index
        user_idx = self.data_processor.user2idx.get(user_id)
        if user_idx is None:
            return []
        
        # Get user sequence
        sequence = self.data_processor.user_sequences.get(user_idx, [])
        if not sequence:
            return []
        
        # Prepare input
        items = [x[0] for x in sequence]
        input_seq = torch.LongTensor([items[-50:]])  # Last 50 items
        attention_mask = torch.ones_like(input_seq)
        
        input_seq = input_seq.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        # Get predictions
        with torch.no_grad():
            scores, indices = self.model.predict(input_seq, attention_mask, top_k=top_k * 2)
        
        # Convert to recommendations
        recommendations = []
        seen_items = set(items) if exclude_history else set()
        
        for score, idx in zip(scores[0].tolist(), indices[0].tolist()):
            if idx in seen_items:
                continue
            
            if len(recommendations) >= top_k:
                break
            
            # Get item info
            item_id = self.data_processor.idx2item.get(idx)
            if item_id is None:
                continue
            
            item_info = self.data_processor.item_info.get(item_id, {})
            
            recommendations.append({
                'product_id': item_id,
                'score': score,
                'name': item_info.get('name', ''),
                'category': item_info.get('category', ''),
                'price': item_info.get('price', 0)
            })
        
        return recommendations
    
    def get_similar_items(
        self,
        item_id: int,
        top_k: int = 10
    ) -> List[Dict]:
        """
        Find similar items based on embedding similarity
        
        Args:
            item_id: Original item ID
            top_k: Number of similar items
            
        Returns:
            List of {product_id, score, name, category}
        """
        if self.data_processor is None:
            raise ValueError("Data processor required")
        
        item_idx = self.data_processor.item2idx.get(item_id)
        if item_idx is None:
            return []
        
        # Get item embedding
        with torch.no_grad():
            item_emb = self.model.encoder.item_embedding.weight[item_idx + 1]
            all_emb = self.model.encoder.item_embedding.weight[1:]  # Exclude padding
            
            # Cosine similarity
            item_emb = item_emb / item_emb.norm()
            all_emb = all_emb / all_emb.norm(dim=1, keepdim=True)
            
            similarities = torch.matmul(all_emb, item_emb)
            
            # Get top-k (excluding self)
            similarities[item_idx] = float('-inf')
            scores, indices = torch.topk(similarities, k=top_k)
        
        # Convert to results
        similar_items = []
        for score, idx in zip(scores.tolist(), indices.tolist()):
            similar_id = self.data_processor.idx2item.get(idx)
            if similar_id is None:
                continue
            
            item_info = self.data_processor.item_info.get(similar_id, {})
            
            similar_items.append({
                'product_id': similar_id,
                'similarity': score,
                'name': item_info.get('name', ''),
                'category': item_info.get('category', ''),
                'price': item_info.get('price', 0)
            })
        
        return similar_items
    
    def recommend_for_sequence(
        self,
        item_ids: List[int],
        top_k: int = 10
    ) -> List[Dict]:
        """
        Generate recommendations based on a sequence of items
        (For cold-start users or anonymous sessions)
        
        Args:
            item_ids: List of item IDs in the sequence
            top_k: Number of recommendations
            
        Returns:
            List of recommendations
        """
        if self.data_processor is None:
            raise ValueError("Data processor required")
        
        # Convert to indices
        item_indices = []
        for item_id in item_ids:
            idx = self.data_processor.item2idx.get(item_id)
            if idx is not None:
                item_indices.append(idx)
        
        if not item_indices:
            return []
        
        # Prepare input
        input_seq = torch.LongTensor([item_indices[-50:]])
        attention_mask = torch.ones_like(input_seq)
        
        input_seq = input_seq.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        # Get predictions
        with torch.no_grad():
            scores, indices = self.model.predict(input_seq, attention_mask, top_k=top_k * 2)
        
        # Convert to recommendations
        recommendations = []
        seen = set(item_indices)
        
        for score, idx in zip(scores[0].tolist(), indices[0].tolist()):
            if idx in seen:
                continue
            
            if len(recommendations) >= top_k:
                break
            
            item_id = self.data_processor.idx2item.get(idx)
            if item_id is None:
                continue
            
            item_info = self.data_processor.item_info.get(item_id, {})
            
            recommendations.append({
                'product_id': item_id,
                'score': score,
                'name': item_info.get('name', ''),
                'category': item_info.get('category', ''),
                'price': item_info.get('price', 0)
            })
        
        return recommendations


def evaluate_recommender(
    recommender: SASRecRecommender,
    test_data: Dict,
    k_values: List[int] = [5, 10, 20]
) -> Dict[str, float]:
    """
    Evaluate recommender on test data
    
    Args:
        recommender: Trained recommender
        test_data: Dict of user_idx -> (input_items, target_item)
        k_values: K values for metrics
        
    Returns:
        Dict of metrics
    """
    metrics = {f'Precision@{k}': 0.0 for k in k_values}
    metrics.update({f'Recall@{k}': 0.0 for k in k_values})
    metrics.update({f'NDCG@{k}': 0.0 for k in k_values})
    metrics.update({f'HR@{k}': 0.0 for k in k_values})
    
    total_users = 0
    
    for user_idx, (input_items, target) in test_data.items():
        # Get original user_id
        user_id = recommender.data_processor.idx2user.get(user_idx)
        if user_id is None:
            continue
        
        # Get recommendations
        recs = recommender.recommend_for_user(user_id, top_k=max(k_values))
        rec_ids = [r['product_id'] for r in recs]
        
        # Get target item id
        target_id = recommender.data_processor.idx2item.get(target)
        if target_id is None:
            continue
        
        total_users += 1
        
        for k in k_values:
            top_k_recs = rec_ids[:k]
            
            # Hit Rate
            hit = 1.0 if target_id in top_k_recs else 0.0
            metrics[f'HR@{k}'] += hit
            
            # Precision (for single target, HR == Precision if hit, else 0)
            metrics[f'Precision@{k}'] += hit / k
            
            # Recall (for single target, HR == Recall)
            metrics[f'Recall@{k}'] += hit
            
            # NDCG
            if target_id in top_k_recs:
                rank = top_k_recs.index(target_id) + 1
                metrics[f'NDCG@{k}'] += 1.0 / np.log2(rank + 1)
    
    # Average
    if total_users > 0:
        for key in metrics:
            metrics[key] /= total_users
    
    return metrics


if __name__ == "__main__":
    print("Recommender module loaded.")
    print("Use SASRecRecommender.load() to load a trained model.")
