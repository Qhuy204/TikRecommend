#!/usr/bin/env python3
"""
Demo Module for PhoBERT Content-Based Recommender
Load once, run recommendations many times
"""

import random
from pathlib import Path
from typing import List, Dict

from preprocessing import load_processed_data, ProcessedData
from model import PhoBERTRecommender
from config import config


class DemoRecommender:
    """
    PhoBERT Content-Based Demo - Load once, use many times
    
    Usage:
        demo = DemoRecommender()
        demo.recommend_user(12345)
        demo.similar_items(277725874)
        demo.recommend_sequence([1, 2, 3])
        demo.random_demo()
    """
    
    def __init__(self):
        """Initialize - load data and embeddings once"""
        print("=" * 50)
        print("PhoBERT Content-Based Demo")
        print("=" * 50)
        
        # Load data and embeddings
        print("\nLoading data...")
        self.data, embeddings = load_processed_data()
        
        # Create recommender
        print("\nInitializing recommender...")
        self.recommender = PhoBERTRecommender(
            embeddings=embeddings,
            item2idx=self.data.item2idx,
            idx2item=self.data.idx2item,
            item_info=self.data.item_info,
            device="cpu"  # Use CPU for demo (fast enough)
        )
        
        print(f"\nReady!")
        print(f"   Items: {self.data.num_items:,}")
        print(f"   Users: {self.data.num_users:,}")
        print(f"   Embedding dim: {embeddings.shape[1]}")
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
        """Recommend items for a user based on their liked items"""
        if user_id is None:
            user_id = random.choice(list(self.data.user2idx.keys()))
        
        print(f"\nRecommendations for User {user_id}")
        print("-" * 40)
        
        if user_id not in self.data.user2idx:
            print(f"User not found. Sample: {list(self.data.user2idx.keys())[:5]}")
            return []
        
        user_idx = self.data.user2idx[user_id]
        liked_indices = self.data.user_pos_items.get(user_idx, set())
        liked_item_ids = [self.data.idx2item[idx] for idx in liked_indices]
        
        print(f"Liked items ({len(liked_item_ids)}):")
        for item_id in liked_item_ids[:5]:
            info = self.data.item_info.get(item_id, {})
            print(f"   • {info.get('name', '?')[:50]}...")
        if len(liked_item_ids) > 5:
            print(f"   ... and {len(liked_item_ids) - 5} more")
        
        recs = self.recommender.recommend_for_items(liked_item_ids, top_k=top_k)
        
        print(f"\nTop {top_k} Recommendations:")
        for i, rec in enumerate(recs, 1):
            print(self._format_item(rec, i))
            print()
        
        return recs
    
    def similar_items(self, item_id: int = None, top_k: int = 10) -> List[Dict]:
        """Find items similar to a given item"""
        if item_id is None:
            item_id = random.choice(list(self.data.item2idx.keys()))
        
        print(f"\nSimilar to Item {item_id}")
        print("-" * 40)
        
        if item_id not in self.data.item2idx:
            print(f"Item not found. Sample: {list(self.data.item2idx.keys())[:5]}")
            return []
        
        info = self.data.item_info.get(item_id, {})
        print(f"Query: {info.get('name', '?')}")
        print(f"   {info.get('category', 'N/A')} | {info.get('price', 0):,.0f} VND")
        
        similar = self.recommender.get_similar_items(item_id, top_k=top_k)
        
        print(f"\nTop {top_k} Similar:")
        for i, item in enumerate(similar, 1):
            print(self._format_item(item, i))
            print()
        
        return similar
    
    def recommend_sequence(self, item_ids: List[int], top_k: int = 10) -> List[Dict]:
        """Recommend based on a list of items"""
        print(f"\n🛒 Sequence-based Recommendation")
        print("-" * 40)
        
        valid_ids = [id for id in item_ids if id in self.data.item2idx]
        print(f"Input items ({len(valid_ids)} valid):")
        for item_id in valid_ids[:5]:
            info = self.data.item_info.get(item_id, {})
            print(f"   • {info.get('name', '?')[:45]}...")
        
        if not valid_ids:
            print("No valid items in input")
            return []
        
        recs = self.recommender.recommend_for_items(valid_ids, top_k=top_k)
        
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
        
        self.recommend_user(top_k=top_k)
        self.similar_items(top_k=top_k)
    
    def show_sample_users(self, k: int = 10) -> List[int]:
        """
        Display K random users with their liked items count
        Returns list of user IDs for easy copy-paste
        """
        print(f"\n👥 Random {k} Users:")
        print("-" * 60)
        
        user_ids = random.sample(
            list(self.data.user2idx.keys()),
            min(k, self.data.num_users)
        )
        
        for i, user_id in enumerate(user_ids, 1):
            user_idx = self.data.user2idx[user_id]
            n_liked = len(self.data.user_pos_items.get(user_idx, set()))
            print(f"   {i:2}. ID: {user_id:<12} | Liked items: {n_liked}")
        
        print("-" * 60)
        print(f"💡 Usage: demo.recommend_user({user_ids[0]})")
        return user_ids
    
    def show_sample_items(self, k: int = 10) -> List[int]:
        """
        Display K random items with name, category, price
        Returns list of item IDs for easy copy-paste
        """
        print(f"\nRandom {k} Items:")
        print("-" * 80)
        
        item_ids = random.sample(
            list(self.data.item2idx.keys()),
            min(k, self.data.num_items)
        )
        
        for i, item_id in enumerate(item_ids, 1):
            info = self.data.item_info.get(item_id, {})
            name = info.get('name', 'Unknown')[:40]
            category = info.get('category', 'N/A')[:20]
            price = info.get('price', 0)
            print(f"   {i:2}. ID: {item_id:<12} | {name:<40} | {category}")
        
        print("-" * 80)
        print(f"💡 Usage: demo.similar_items({item_ids[0]})")
        return item_ids
    
    def get_sample_users(self, n: int = 10) -> List[int]:
        """Get sample user IDs (for backwards compatibility)"""
        return self.show_sample_users(n)
    
    def get_sample_items(self, n: int = 10) -> List[int]:
        """Get sample item IDs (for backwards compatibility)"""
        return self.show_sample_items(n)


if __name__ == "__main__":
    demo = DemoRecommender()
    
    print("\n\n" + "=" * 50)
    print("USAGE")
    print("=" * 50)
    print("""
# Show random users/items to choose from:
demo.show_sample_users(10)      # Show 10 random users
demo.show_sample_items(10)      # Show 10 random items

# Recommendations:
demo.recommend_user(12345)      # Recommend for specific user
demo.recommend_user()           # Random user

demo.similar_items(277725874)   # Similar to specific item
demo.similar_items()            # Random item

demo.recommend_sequence([1,2,3]) # From item list

demo.random_demo()              # Quick random test
    """)
    
    demo.random_demo(top_k=3)

