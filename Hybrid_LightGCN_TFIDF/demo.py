#!/usr/bin/env python3
"""
Demo for Hybrid LightGCN + TF-IDF Recommender
Load once, run many times
"""

import random
from pathlib import Path
from typing import List, Dict

from config import config
from model import load_hybrid_recommender, HybridRecommender


class DemoRecommender:
    """
    Hybrid Demo - Load once, use many times
    
    Usage:
        demo = DemoRecommender()
        demo.recommend_user(12345)
        demo.compare_methods(12345)
    """
    
    def __init__(self, alpha: float = None, device: str = "cuda"):
        # Use config default if alpha not specified
        if alpha is None:
            alpha = config.hybrid.alpha
        print("=" * 60)
        print("Hybrid LightGCN + TF-IDF Demo")
        print("=" * 60)
        
        self.hybrid = load_hybrid_recommender(alpha=alpha, device=device)
        self.data = self.hybrid.lightgcn_data
        
        print(f"\nReady!")
        print(f"   Users: {self.data.num_users:,}")
        print(f"   Items: {self.data.num_items:,}")
        print(f"   Alpha: {self.hybrid.alpha}")
        print("=" * 60)
    
    def _format_item(self, item: dict, index: int = None) -> str:
        prefix = f"{index}. " if index else ""
        name = item.get('name', 'Unknown')[:55]
        category = item.get('category', 'N/A')
        price = item.get('price', 0)
        score = item.get('score', 0)
        
        lines = [
            f"   {prefix}{name}...",
            f"      Category: {category}",
            f"      Price: {price:,.0f} VND",
        ]
        if score:
            lines.append(f"      Score: {score:.4f}")
        
        return "\n".join(lines)
    
    def recommend_user(
        self, 
        user_id: int = None, 
        top_k: int = 10,
        alpha: float = None
    ) -> List[Dict]:
        """Get hybrid recommendations for a user"""
        if user_id is None:
            user_id = random.choice(list(self.data.user2idx.keys()))
        
        if alpha is not None:
            self.hybrid.set_alpha(alpha)
        
        print(f"\nHybrid Recommendations for User {user_id}")
        print(f"   alpha={self.hybrid.alpha:.2f} (LightGCN), 1-alpha={1-self.hybrid.alpha:.2f} (TF-IDF)")
        print("-" * 50)
        
        if user_id not in self.data.user2idx:
            print(f"User not found. Sample: {list(self.data.user2idx.keys())[:5]}")
            return []
        
        user_idx = self.data.user2idx[user_id]
        user_history = self.data.user_pos_items.get(user_idx, set())
        
        # Display user's interaction history
        print(f"\nInteraction History ({len(user_history)} items):")
        print("-" * 50)
        for i, item_idx in enumerate(list(user_history)[:10], 1):  # Show max 10 items
            item_id = self.data.idx2item.get(item_idx)
            info = self.data.item_info.get(item_id, {})
            name = info.get('name', 'Unknown')[:50]
            category = info.get('category', 'N/A')
            price = info.get('price', 0)
            print(f"   {i}. {name}...")
            print(f"      {category} | {price:,.0f} VND")
        
        if len(user_history) > 10:
            print(f"   ... and {len(user_history) - 10} more items")
        
        # Get recommendations
        recs = self.hybrid.recommend_with_details(user_idx, user_history, top_k)
        
        print(f"\nTop {top_k} Recommendations:")
        print("-" * 50)
        for i, rec in enumerate(recs, 1):
            print(self._format_item(rec, i))
            print()
        
        return recs
    
    def compare_methods(
        self, 
        user_id: int = None, 
        top_k: int = 5
    ) -> Dict[str, List[Dict]]:
        """Compare LightGCN-only, TF-IDF-only, and Hybrid"""
        if user_id is None:
            user_id = random.choice(list(self.data.user2idx.keys()))
        
        print(f"\nCOMPARISON for User {user_id}")
        print("=" * 60)
        
        if user_id not in self.data.user2idx:
            print(f"User not found")
            return {}
        
        user_idx = self.data.user2idx[user_id]
        user_history = self.data.user_pos_items.get(user_idx, set())
        
        print(f"History: {len(user_history)} items\n")
        
        # Save original alpha to restore later
        original_alpha = self.hybrid.alpha
        
        results = {}
        methods = [
            ("LightGCN only (α=1.0)", 1.0),
            ("TF-IDF only (α=0.0)", 0.0),
            (f"Hybrid (α={original_alpha})", original_alpha),
        ]
        
        for method_name, alpha in methods:
            print(f"\n{'-' * 50}")
            print(f"[{method_name}]")
            print(f"{'-' * 50}")
            
            self.hybrid.set_alpha(alpha)
            recs = self.hybrid.recommend_with_details(user_idx, user_history, top_k)
            results[method_name] = recs
            
            for i, rec in enumerate(recs, 1):
                print(f"   {i}. {rec['name'][:45]}...")
                print(f"      Score: {rec['score']:.4f}")
        
        # Reset to original alpha
        self.hybrid.set_alpha(original_alpha)
        
        return results
    
    def random_demo(self, top_k: int = 5):
        """Random user demo with comparison"""
        print("\n" + "=" * 60)
        print("RANDOM DEMO")
        print("=" * 60)
        self.compare_methods(top_k=top_k)
    
    def get_sample_users(self, n: int = 10) -> List[int]:
        return random.sample(list(self.data.user2idx.keys()), min(n, self.data.num_users))
    
    def get_sample_items(self, n: int = 10) -> List[int]:
        return random.sample(list(self.data.item2idx.keys()), min(n, self.data.num_items))


if __name__ == "__main__":
    demo = DemoRecommender()  # Uses config.hybrid.alpha (0.8)
    
    print("\n\n" + "=" * 60)
    print("USAGE")
    print("=" * 60)
    print("""
demo.recommend_user(12345)          # Hybrid recommendations
demo.recommend_user(12345, alpha=0.5)  # Custom alpha
demo.compare_methods(12345)         # Compare all methods
demo.random_demo(top_k=3)           # Random demo
    """)
    
    demo.random_demo(top_k=3)
