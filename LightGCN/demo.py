#!/usr/bin/env python3
"""
Demo for LightGCN
Load once, run many times
"""

import random
from pathlib import Path
from typing import List, Dict

import torch

from config import config
from preprocessing import load_processed_data, ProcessedData
from models import LightGCN


class DemoRecommender:
    """
    LightGCN Demo - Load once, use many times
    
    Usage:
        demo = DemoRecommender()
        demo.recommend_user(12345)
        demo.similar_items(277725)
    """
    
    def __init__(self, checkpoint: str = "best_model.pt"):
        print("=" * 50)
        print("LightGCN Demo")
        print("=" * 50)
        
        print("\nLoading data...")
        self.data = load_processed_data()
        
        print("\nLoading model...")
        checkpoint_path = Path(config.training.checkpoint_dir) / checkpoint
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Not found: {checkpoint_path}")
        
        self.device = config.training.device
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        
        self.model = LightGCN(
            num_users=self.data.num_users,
            num_items=self.data.num_items,
            adj_matrix=self.data.adj_matrix
        )
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.adj_matrix = self.model.adj_matrix.to(self.device)
        self.model.eval()
        
        print(f"   Loaded: epoch {ckpt['epoch']}, HR@10={ckpt['best_metric']:.4f}")
        print(f"\nReady! Users: {self.data.num_users:,} | Items: {self.data.num_items:,}")
        print("=" * 50)
    
    def _format_item(self, item: dict, index: int = None) -> str:
        prefix = f"{index}. " if index else ""
        name = item.get('name', 'Unknown')[:55]
        category = item.get('category', 'N/A')
        price = item.get('price', 0)
        score = item.get('score', item.get('similarity', 0))
        
        lines = [
            f"   {prefix}{name}...",
            f"      Category: {category}",
            f"      Price: {price:,.0f} VND",
        ]
        if score:
            lines.append(f"      Score: {score:.4f}")
        
        return "\n".join(lines)
    
    def recommend_user(self, user_id: int = None, top_k: int = 10) -> List[Dict]:
        if user_id is None:
            user_id = random.choice(list(self.data.user2idx.keys()))
        
        print(f"\nRecommendations for User {user_id}")
        print("-" * 40)
        
        if user_id not in self.data.user2idx:
            print(f"Not found. Sample: {list(self.data.user2idx.keys())[:5]}")
            return []
        
        user_idx = self.data.user2idx[user_id]
        seen_items = self.data.user_pos_items.get(user_idx, set())
        
        print(f"History ({len(seen_items)} items)")
        
        # Predict
        user_tensor = torch.tensor([user_idx], device=self.device)
        with torch.no_grad():
            scores, indices = self.model.predict(user_tensor, top_k=top_k + len(seen_items))
        
        recs = []
        print(f"\nTop {top_k} Recommendations:")
        for score, idx in zip(scores[0].tolist(), indices[0].tolist()):
            if idx in seen_items or len(recs) >= top_k:
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
        if item_id is None:
            item_id = random.choice(list(self.data.item2idx.keys()))
        
        print(f"\nSimilar to Item {item_id}")
        print("-" * 40)
        
        if item_id not in self.data.item2idx:
            print(f"Not found")
            return []
        
        item_idx = self.data.item2idx[item_id]
        info = self.data.item_info.get(item_id, {})
        print(f"Query: {info.get('name', '?')}")
        print(f"   {info.get('category', 'N/A')} | {info.get('price', 0):,.0f} VND")
        
        # Get item embeddings after propagation
        with torch.no_grad():
            _, all_item_emb = self.model.get_all_embeddings()
            query = all_item_emb[item_idx]
            
            query_norm = query / query.norm()
            all_norm = all_item_emb / all_item_emb.norm(dim=1, keepdim=True)
            sim = torch.matmul(all_norm, query_norm)
            sim[item_idx] = -float('inf')
            
            top_scores, top_indices = torch.topk(sim, k=top_k)
        
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
    
    def random_demo(self, top_k: int = 5):
        print("\n" + "=" * 50)
        print("RANDOM DEMO")
        print("=" * 50)
        self.recommend_user(top_k=top_k)
        self.similar_items(top_k=top_k)
    
    def get_sample_users(self, n: int = 10) -> List[int]:
        return random.sample(list(self.data.user2idx.keys()), min(n, self.data.num_users))
    
    def get_sample_items(self, n: int = 10) -> List[int]:
        return random.sample(list(self.data.item2idx.keys()), min(n, self.data.num_items))


if __name__ == "__main__":
    demo = DemoRecommender()
    
    print("\n\n" + "=" * 50)
    print("USAGE")
    print("=" * 50)
    print("""
demo.recommend_user(12345)   # User recommendations
demo.similar_items(277725)   # Similar items
demo.random_demo(top_k=3)    # Random demo
    """)
    
    demo.random_demo(top_k=3)
