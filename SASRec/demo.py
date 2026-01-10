#!/usr/bin/env python3
"""
Demo Recommendation System - Load Once, Run Many
Sử dụng class DemoRecommender để load data 1 lần và gọi nhiều lần
"""

import random
from pathlib import Path
from typing import List, Dict, Optional

import torch

from config import default_config
from data_processor import TikiDataProcessor
from models import SASRecModel
from recommender import SASRecRecommender


class DemoRecommender:
    """
    Demo class - Load data một lần, gọi recommend nhiều lần
    
    Usage:
        demo = DemoRecommender()  # Load 1 lần
        demo.recommend_user(12345)  # Gọi nhiều lần
        demo.similar_items(277725874)
    """
    
    def __init__(self, checkpoint_path: str = "checkpoints/best_model.pt"):
        """Initialize và load data + model một lần"""
        print("=" * 50)
        print("Initializing Demo Recommender...")
        print("=" * 50)
        
        # Load data processor
        print("\nLoading data...")
        self.processor = TikiDataProcessor()
        self.processor.load_raw_data()
        
        # Load model
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}\n"
                "   Train first: python main.py --mode train"
            )
        
        print("\nLoading model...")
        self.recommender = SASRecRecommender.load(
            str(checkpoint_path),
            self.processor,
            model_type='sasrec'
        )
        
        print("\nReady to use!")
        print(f"   Items: {self.processor.num_items:,}")
        print(f"   Users: {self.processor.num_users:,}")
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
            
        Returns:
            List of recommended items
        """
        # Random user if not specified
        if user_id is None:
            user_id = random.choice(list(self.processor.user2idx.keys()))
        
        print(f"\nRecommendations for User {user_id}")
        print("-" * 40)
        
        # Check user exists
        if user_id not in self.processor.user2idx:
            print(f"User not found. Sample users: {list(self.processor.user2idx.keys())[:5]}")
            return []
        
        # Show history
        user_idx = self.processor.user2idx[user_id]
        history = self.processor.user_sequences.get(user_idx, [])
        
        print(f"History ({len(history)} items):")
        for item_idx, _, rating in history[-3:]:
            item_id = self.processor.idx2item.get(item_idx)
            info = self.processor.item_info.get(item_id, {})
            print(f"   • {info.get('name', '?')[:45]}... ({rating}⭐)")
        
        # Get recommendations
        recs = self.recommender.recommend_for_user(user_id, top_k=top_k)
        
        print(f"\nTop {len(recs)} Recommendations:")
        for i, rec in enumerate(recs, 1):
            print(self._format_item(rec, i))
            print()
        
        return recs
    
    def similar_items(self, item_id: int = None, top_k: int = 10) -> List[Dict]:
        """
        Find similar items
        
        Args:
            item_id: Item ID (random if None)
            top_k: Number of similar items
            
        Returns:
            List of similar items
        """
        # Random item if not specified
        if item_id is None:
            item_id = random.choice(list(self.processor.item2idx.keys()))
        
        print(f"\nSimilar to Item {item_id}")
        print("-" * 40)
        
        # Check item exists
        if item_id not in self.processor.item2idx:
            print(f"Item not found. Sample items: {list(self.processor.item2idx.keys())[:5]}")
            return []
        
        # Show query item
        info = self.processor.item_info.get(item_id, {})
        print(f"Query: {info.get('name', '?')}")
        print(f"   {info.get('category', 'N/A')} | {info.get('price', 0):,.0f} VND")
        
        # Get similar
        similar = self.recommender.get_similar_items(item_id, top_k=top_k)
        
        print(f"\nTop {len(similar)} Similar:")
        for i, item in enumerate(similar, 1):
            print(self._format_item(item, i))
            print()
        
        return similar
    
    def recommend_sequence(self, item_ids: List[int], top_k: int = 10) -> List[Dict]:
        """
        Recommend based on a sequence of items (for cold-start/anonymous users)
        
        Args:
            item_ids: List of item IDs representing browsing history
            top_k: Number of recommendations
            
        Returns:
            List of recommended items
        """
        print(f"\n🛒 Session-based Recommendation")
        print("-" * 40)
        
        print(f"Input sequence ({len(item_ids)} items):")
        for item_id in item_ids[-5:]:
            info = self.processor.item_info.get(item_id, {})
            print(f"   • {info.get('name', '?')[:45]}...")
        
        recs = self.recommender.recommend_for_sequence(item_ids, top_k=top_k)
        
        print(f"\nRecommended next:")
        for i, rec in enumerate(recs, 1):
            print(self._format_item(rec, i))
            print()
        
        return recs
    
    def random_demo(self, top_k: int = 5):
        """Run random demo"""
        print("\n" + "=" * 50)
        print("RANDOM DEMO")
        print("=" * 50)
        
        # Random user
        self.recommend_user(top_k=top_k)
        
        # Random similar items
        self.similar_items(top_k=top_k)
    
    def get_sample_users(self, n: int = 10) -> List[int]:
        """Get sample user IDs"""
        return random.sample(list(self.processor.user2idx.keys()), 
                            min(n, len(self.processor.user2idx)))
    
    def get_sample_items(self, n: int = 10) -> List[int]:
        """Get sample item IDs"""
        return random.sample(list(self.processor.item2idx.keys()),
                            min(n, len(self.processor.item2idx)))


# ============================================================
# QUICK USAGE
# ============================================================

if __name__ == "__main__":
    # Load once
    demo = DemoRecommender()
    
    # Use many times
    print("\n\n" + "=" * 50)
    print("EXAMPLE USAGE")
    print("=" * 50)
    print("""
# Sau khi init, có thể gọi nhiều lần:

demo.recommend_user(12345)      # Recommend cho user
demo.recommend_user()           # Random user

demo.similar_items(277725874)   # Items tương tự
demo.similar_items()            # Random item

demo.recommend_sequence([1,2,3]) # Session-based

demo.get_sample_users(5)        # Lấy sample user IDs
demo.get_sample_items(5)        # Lấy sample item IDs
    """)
    
    # Run random demo
    demo.random_demo(top_k=3)
