#!/usr/bin/env python3
"""
Demo Module for CL4SRec
Load once, run recommendations many times
"""

import random
from pathlib import Path
from typing import List, Dict, Optional

import torch

from config import config
from preprocessing import load_processed_data, ProcessedData
from models import CL4SRecModel


class DemoRecommender:
    """
    CL4SRec Demo - Load data once, use many times
    
    Usage:
        demo = DemoRecommender()   # Load 1 lần
        demo.recommend_user(12345)  # Gọi nhiều lần
        demo.similar_items(277725874)
        demo.recommend_sequence([1, 2, 3])
    """
    
    def __init__(self, checkpoint: str = "best_model.pt"):
        """Initialize - load data + model once"""
        print("=" * 50)
        print("Initializing CL4SRec Demo...")
        print("=" * 50)
        
        # Load data
        print("\nLoading data...")
        self.data = load_processed_data()
        
        # Load model
        checkpoint_path = Path(config.training.checkpoint_dir) / checkpoint
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}\n"
                "   Train first: python train.py"
            )
        
        print("\nLoading model...")
        self.device = config.training.device
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        
        self.model = CL4SRecModel(self.data.num_items, cl_weight=ckpt.get('cl_weight', 0.1))
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"   Loaded: epoch {ckpt['epoch']}, HR@10={ckpt['best_metric']:.4f}")
        print("\nReady to use!")
        print(f"   Items: {self.data.num_items:,}")
        print(f"   Users: {self.data.num_users:,}")
        print("=" * 50)
    
    def _format_item(self, item: dict, index: int = None) -> str:
        """Format item info for display"""
        prefix = f"{index}. " if index else ""
        name = item.get('name', 'Unknown')[:55]
        category = item.get('category', 'N/A')
        price = item.get('price', 0)
        score = item.get('score', item.get('similarity', 0))
        
        lines = [
            f"   {prefix}{name}...",
            f"      {category}",
            f"      {price:,.0f} VND",
        ]
        if score:
            lines.append(f"      Score: {score:.4f}")
        
        return "\n".join(lines)
    
    def recommend_user(self, user_id: int = None, top_k: int = 10) -> List[Dict]:
        """
        Get recommendations for a user
        
        Args:
            user_id: User ID (random if None)
            top_k: Number of recommendations
        """
        if user_id is None:
            user_id = random.choice(list(self.data.user2idx.keys()))
        
        print(f"\nRecommendations for User {user_id}")
        print("-" * 40)
        
        if user_id not in self.data.user2idx:
            print(f"User not found. Sample: {list(self.data.user2idx.keys())[:5]}")
            return []
        
        # Get history
        user_idx = self.data.user2idx[user_id]
        sequence = self.data.user_sequences.get(user_idx, [])
        items = [x[0] for x in sequence]
        
        print(f"History ({len(items)} items):")
        for item_idx, _, rating in sequence:
            item_id = self.data.idx2item.get(item_idx)
            info = self.data.item_info.get(item_id, {})
            print(f"   • {info.get('name', '?')[:50]}... ({rating}⭐)")
        
        # Predict
        input_seq = torch.LongTensor([items[-config.data.max_seq_length:]]).to(self.device)
        mask = torch.ones_like(input_seq)
        
        with torch.no_grad():
            scores, indices = self.model.predict(input_seq, mask, top_k=top_k + len(items))
        
        # Filter seen and format
        seen = set(items)
        recs = []
        
        print(f"\nTop {top_k} Recommendations:")
        for score, idx in zip(scores[0].tolist(), indices[0].tolist()):
            if idx in seen or len(recs) >= top_k:
                continue
            
            item_id = self.data.idx2item.get(idx)
            info = self.data.item_info.get(item_id, {})
            
            rec = {
                'product_id': item_id,
                'name': info.get('name', ''),
                'category': info.get('category', ''),
                'price': info.get('price', 0),
                'score': score
            }
            recs.append(rec)
            print(self._format_item(rec, len(recs)))
            print()
        
        return recs
    
    def similar_items(self, item_id: int = None, top_k: int = 10) -> List[Dict]:
        """
        Find similar items by embedding cosine similarity
        """
        if item_id is None:
            item_id = random.choice(list(self.data.item2idx.keys()))
        
        print(f"\nSimilar to Item {item_id}")
        print("-" * 40)
        
        if item_id not in self.data.item2idx:
            print(f"Item not found. Sample: {list(self.data.item2idx.keys())[:5]}")
            return []
        
        # Show query item
        item_idx = self.data.item2idx[item_id]
        info = self.data.item_info.get(item_id, {})
        print(f"Query: {info.get('name', '?')}")
        print(f"   {info.get('category', 'N/A')} | {info.get('price', 0):,.0f} VND")
        
        # Compute similarity
        with torch.no_grad():
            emb = self.model.encoder.item_embedding.weight[1:]
            query = emb[item_idx]
            
            query_norm = query / query.norm()
            all_norm = emb / emb.norm(dim=1, keepdim=True)
            sim = torch.matmul(all_norm, query_norm)
            sim[item_idx] = -float('inf')
            
            top_scores, top_indices = torch.topk(sim, k=top_k)
        
        # Format results
        similar = []
        print(f"\nTop {top_k} Similar:")
        for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
            sim_id = self.data.idx2item.get(idx)
            sim_info = self.data.item_info.get(sim_id, {})
            
            item = {
                'product_id': sim_id,
                'name': sim_info.get('name', ''),
                'category': sim_info.get('category', ''),
                'price': sim_info.get('price', 0),
                'similarity': score
            }
            similar.append(item)
            print(self._format_item(item, len(similar)))
            print()
        
        return similar
    
    def recommend_sequence(self, item_ids: List[int], top_k: int = 10) -> List[Dict]:
        """
        Recommend based on item sequence (for cold-start/anonymous users)
        """
        print(f"\n🛒 Session-based Recommendation")
        print("-" * 40)
        
        # Convert to indices
        indices = []
        print(f"Input sequence ({len(item_ids)} items):")
        for item_id in item_ids[-5:]:
            idx = self.data.item2idx.get(item_id)
            if idx is not None:
                indices.append(idx)
                info = self.data.item_info.get(item_id, {})
                print(f"   • {info.get('name', '?')[:45]}...")
        
        if not indices:
            print("No valid items in sequence")
            return []
        
        # Predict
        input_seq = torch.LongTensor([indices[-config.data.max_seq_length:]]).to(self.device)
        mask = torch.ones_like(input_seq)
        
        with torch.no_grad():
            scores, top_indices = self.model.predict(input_seq, mask, top_k=top_k + len(indices))
        
        # Filter and format
        seen = set(indices)
        recs = []
        
        print(f"\nRecommended next:")
        for score, idx in zip(scores[0].tolist(), top_indices[0].tolist()):
            if idx in seen or len(recs) >= top_k:
                continue
            
            item_id = self.data.idx2item.get(idx)
            info = self.data.item_info.get(item_id, {})
            
            rec = {
                'product_id': item_id,
                'name': info.get('name', ''),
                'category': info.get('category', ''),
                'price': info.get('price', 0),
                'score': score
            }
            recs.append(rec)
            print(self._format_item(rec, len(recs)))
            print()
        
        return recs
    
    def random_demo(self, top_k: int = 5):
        """Run random demo"""
        print("\n" + "=" * 50)
        print("RANDOM DEMO")
        print("=" * 50)
        
        self.recommend_user(top_k=top_k)
        self.similar_items(top_k=top_k)
    
    def get_sample_users(self, n: int = 10) -> List[int]:
        """Get sample user IDs"""
        return random.sample(list(self.data.user2idx.keys()), min(n, self.data.num_users))
    
    def get_sample_items(self, n: int = 10) -> List[int]:
        """Get sample item IDs"""
        return random.sample(list(self.data.item2idx.keys()), min(n, self.data.num_items))


# ============================================================
# QUICK USAGE
# ============================================================

if __name__ == "__main__":
    demo = DemoRecommender()
    
    print("\n\n" + "=" * 50)
    print("EXAMPLE USAGE")
    print("=" * 50)
    print("""
# Sau khi init, có thể gọi nhiều lần:

demo.recommend_user(12345)       # Recommend cho user
demo.recommend_user()            # Random user

demo.similar_items(277725874)    # Items tương tự
demo.similar_items()             # Random item

demo.recommend_sequence([1,2,3]) # Session-based

demo.get_sample_users(5)         # Lấy sample user IDs
demo.get_sample_items(5)         # Lấy sample item IDs
    """)
    
    # Run random demo
    demo.random_demo(top_k=3)
