"""
End-to-End Inference Pipeline Demo
Demonstrates complete recommendation flow: User → Retrieval → Ranking → Top-K
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import pickle
from tqdm import tqdm

# Import models
from recommendation_system import TwoTowerModel, MMoEModel
from transformers import AutoTokenizer


class RecommendationEngine:
    """Complete recommendation engine với Two-Stage Funnel"""
    
    def __init__(self, 
                 two_tower_path: str = "models/two_tower_best.pt",
                 mmoe_path: str = "models/mmoe_best.pt",
                 item_features_path: str = "data/processed/item_features.csv",
                 ranking_features_path: str = "data/processed/ranking_features.csv",
                 device: str = "cuda"):
        
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load data
        print("\n[1] Loading data...")
        self.item_df = pd.read_csv(item_features_path)
        self.ranking_df = pd.read_csv(ranking_features_path)
        
        # Prepare item embeddings cache
        self.item_id_to_idx = {pid: idx for idx, pid in enumerate(self.item_df['product_id'])}
        
        print(f"Loaded {len(self.item_df):,} items")
        
        # Load models
        print("\n[2] Loading models...")
        self._load_models(two_tower_path, mmoe_path)
        
        # Initialize tokenizer
        print("\n[3] Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
        
        # Precompute item embeddings
        print("\n[4] Precomputing item embeddings...")
        self.item_embeddings = self._precompute_item_embeddings()
        
        print("\n✓ Engine ready!")
    
    def _load_models(self, two_tower_path, mmoe_path):
        """Load trained models"""
        
        # Two-Tower Model
        num_users = 100000  # Adjust based on your data
        num_categories = self.item_df['category'].nunique()
        num_brands = self.item_df['brand_id'].nunique()
        
        self.two_tower = TwoTowerModel(
            num_users=num_users,
            num_categories=num_categories,
            num_brands=num_brands,
            embedding_dim=128
        )
        
        if Path(two_tower_path).exists():
            self.two_tower.load_state_dict(torch.load(two_tower_path, map_location=self.device))
            print(f"✓ Loaded Two-Tower from {two_tower_path}")
        else:
            print(f"⚠ Two-Tower checkpoint not found: {two_tower_path}")
        
        self.two_tower.to(self.device)
        self.two_tower.eval()
        
        # MMoE Model
        feature_cols = [
            'price', 'list_price', 'discount_rate', 'rating_average',
            'review_count', 'quantity_sold', 'seller_id',
            'is_authentic', 'is_freeship', 'has_return_policy', 'is_available'
        ]
        input_dim = len(feature_cols)
        
        self.mmoe = MMoEModel(
            input_dim=input_dim,
            num_experts=4,
            expert_hidden_dim=128,
            tower_hidden_dim=64
        )
        
        if Path(mmoe_path).exists():
            checkpoint = torch.load(mmoe_path, map_location=self.device)
            self.mmoe.load_state_dict(checkpoint['model_state_dict'])
            self.scaler = checkpoint.get('scaler')
            print(f"✓ Loaded MMoE from {mmoe_path}")
        else:
            print(f"⚠ MMoE checkpoint not found: {mmoe_path}")
        
        self.mmoe.to(self.device)
        self.mmoe.eval()
    
    def _precompute_item_embeddings(self) -> torch.Tensor:
        """Precompute embeddings cho tất cả items"""
        
        embeddings = []
        batch_size = 64
        
        for i in tqdm(range(0, len(self.item_df), batch_size), desc="Computing embeddings"):
            batch = self.item_df.iloc[i:i+batch_size]
            
            # Tokenize text
            texts = batch['text_content'].tolist()
            encodings = self.tokenizer(
                texts,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Category & brand IDs (simplified - use actual encoders in production)
            category_ids = torch.zeros(len(batch), dtype=torch.long)
            brand_ids = torch.zeros(len(batch), dtype=torch.long)
            
            # Move to device
            input_ids = encodings['input_ids'].to(self.device)
            attention_mask = encodings['attention_mask'].to(self.device)
            category_ids = category_ids.to(self.device)
            brand_ids = brand_ids.to(self.device)
            
            # Compute embeddings
            with torch.no_grad():
                item_emb = self.two_tower.item_tower(
                    input_ids, attention_mask, category_ids, brand_ids
                )
            
            embeddings.append(item_emb.cpu())
        
        # Concatenate all
        all_embeddings = torch.cat(embeddings, dim=0)
        
        return all_embeddings
    
    def retrieve_candidates(self, user_id: int, k: int = 100) -> List[int]:
        """
        Stage 1: Retrieval using Two-Tower Model
        Returns: Top-K product IDs
        """
        
        # Get user embedding
        user_tensor = torch.tensor([user_id], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            user_emb = self.two_tower.user_tower(user_tensor)
        
        # Compute similarities with all items
        similarities = torch.matmul(
            user_emb.cpu(), 
            self.item_embeddings.T
        ).squeeze()
        
        # Get top-k indices
        top_k_indices = torch.topk(similarities, k).indices.numpy()
        
        # Convert indices to product IDs
        candidate_ids = self.item_df.iloc[top_k_indices]['product_id'].tolist()
        
        return candidate_ids
    
    def rank_candidates(self, candidate_ids: List[int]) -> List[Tuple[int, float]]:
        """
        Stage 2: Ranking using MMoE Model
        Returns: List of (product_id, score) sorted by score
        """
        
        # Get features for candidates
        candidate_df = self.ranking_df[
            self.ranking_df['product_id'].isin(candidate_ids)
        ].copy()
        
        if len(candidate_df) == 0:
            return []
        
        # Prepare features
        feature_cols = [
            'price', 'list_price', 'discount_rate', 'rating_average',
            'review_count', 'quantity_sold', 'seller_id',
            'is_authentic', 'is_freeship', 'has_return_policy', 'is_available'
        ]
        
        features = candidate_df[feature_cols].fillna(0).values
        
        # Scale features
        if hasattr(self, 'scaler') and self.scaler is not None:
            features = self.scaler.transform(features)
        
        features_tensor = torch.FloatTensor(features).to(self.device)
        
        # Predict
        with torch.no_grad():
            purchase_scores, quality_scores, price_scores = self.mmoe(features_tensor)
        
        # Combine scores (weighted average)
        # Purchase is most important, then quality, then price
        final_scores = (
            0.6 * purchase_scores.cpu().numpy().flatten() +
            0.3 * quality_scores.cpu().numpy().flatten() +
            0.1 * price_scores.cpu().numpy().flatten()
        )
        
        # Create (product_id, score) pairs
        results = list(zip(candidate_df['product_id'].tolist(), final_scores))
        
        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def recommend(self, user_id: int, top_n: int = 10) -> pd.DataFrame:
        """
        Complete recommendation pipeline
        
        Args:
            user_id: User ID
            top_n: Number of recommendations to return
        
        Returns:
            DataFrame with top-N products and their details
        """
        
        print(f"\n{'='*60}")
        print(f"GENERATING RECOMMENDATIONS FOR USER {user_id}")
        print(f"{'='*60}")
        
        # Stage 1: Retrieval
        print("\n[Stage 1: Retrieval]")
        candidates = self.retrieve_candidates(user_id, k=100)
        print(f"Retrieved {len(candidates)} candidates")
        
        # Stage 2: Ranking
        print("\n[Stage 2: Ranking]")
        ranked = self.rank_candidates(candidates)
        print(f"Ranked {len(ranked)} products")
        
        # Get top-N
        top_products = ranked[:top_n]
        top_ids = [pid for pid, _ in top_products]
        scores = [score for _, score in top_products]
        
        # Get product details
        recommendations = self.item_df[
            self.item_df['product_id'].isin(top_ids)
        ].copy()
        
        # Add scores and sort
        score_map = dict(top_products)
        recommendations['score'] = recommendations['product_id'].map(score_map)
        recommendations = recommendations.sort_values('score', ascending=False)
        
        # Merge with ranking features for additional info
        recommendations = recommendations.merge(
            self.ranking_df[['product_id', 'price', 'rating_average', 'discount_rate']],
            on='product_id',
            how='left'
        )
        
        print(f"\n[Results: Top {top_n} Recommendations]")
        print("="*60)
        
        return recommendations
    
    def batch_recommend(self, user_ids: List[int], top_n: int = 10) -> Dict[int, pd.DataFrame]:
        """Batch recommendation cho nhiều users"""
        
        results = {}
        
        for user_id in tqdm(user_ids, desc="Generating recommendations"):
            results[user_id] = self.recommend(user_id, top_n)
        
        return results


# ============================================================================
# DEMO USAGE
# ============================================================================

def display_recommendations(recommendations: pd.DataFrame):
    """Pretty print recommendations"""
    
    print("\n" + "="*80)
    print("TOP RECOMMENDATIONS")
    print("="*80)
    
    for idx, row in recommendations.iterrows():
        print(f"\n#{idx+1} [Score: {row['score']:.3f}]")
        print(f"  Product ID:  {row['product_id']}")
        print(f"  Category:    {row['category']}")
        print(f"  Price:       {row['price']:,.0f} VNĐ")
        
        if pd.notna(row.get('rating_average')):
            print(f"  Rating:      {row['rating_average']:.1f}⭐")
        
        if pd.notna(row.get('discount_rate')) and row['discount_rate'] > 0:
            print(f"  Discount:    {row['discount_rate']:.0f}%")
        
        # Truncate text for display
        text = row.get('text_content', '')
        if len(text) > 100:
            text = text[:100] + "..."
        print(f"  Description: {text}")
        print("-" * 80)


def demo_single_user(engine: RecommendationEngine, user_id: int = 12345):
    """Demo recommendation cho 1 user"""
    
    print("\n" + "🎯 "*30)
    print("DEMO: SINGLE USER RECOMMENDATION")
    print("🎯 "*30)
    
    recommendations = engine.recommend(user_id=user_id, top_n=10)
    
    display_recommendations(recommendations)
    
    # Save results
    output_path = f"results/recommendations_user_{user_id}.csv"
    Path(output_path).parent.mkdir(exist_ok=True)
    recommendations.to_csv(output_path, index=False)
    print(f"\n✓ Saved to: {output_path}")


def demo_batch_users(engine: RecommendationEngine, user_ids: List[int]):
    """Demo batch recommendation"""
    
    print("\n" + "🎯 "*30)
    print("DEMO: BATCH RECOMMENDATION")
    print("🎯 "*30)
    
    results = engine.batch_recommend(user_ids, top_n=10)
    
    print(f"\nGenerated recommendations for {len(results)} users")
    
    # Save all results
    for user_id, recs in results.items():
        output_path = f"results/batch/recommendations_user_{user_id}.csv"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        recs.to_csv(output_path, index=False)
    
    print(f"✓ Saved all results to results/batch/")


def demo_cold_start(engine: RecommendationEngine):
    """Demo recommendation cho sản phẩm mới (cold-start)"""
    
    print("\n" + "🎯 "*30)
    print("DEMO: COLD-START PRODUCT")
    print("🎯 "*30)
    
    # Giả lập sản phẩm mới chưa có rating
    new_products = engine.item_df[engine.item_df['product_id'] > 270000000].head(10)
    
    print(f"\nTesting with {len(new_products)} cold-start products")
    
    # Test xem model có gợi ý được không
    for _, product in new_products.iterrows():
        print(f"\nProduct: {product['product_id']}")
        print(f"Category: {product['category']}")
        
        # Compute similarity với 1 user mẫu
        user_id = 12345
        recommendations = engine.recommend(user_id, top_n=20)
        
        if product['product_id'] in recommendations['product_id'].values:
            rank = recommendations[recommendations['product_id'] == product['product_id']].index[0] + 1
            print(f"✓ Found in recommendations at rank #{rank}")
        else:
            print("✗ Not in top-20 recommendations")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Recommendation Engine Demo')
    parser.add_argument('--mode', type=str, 
                       choices=['single', 'batch', 'cold_start'],
                       default='single',
                       help='Demo mode')
    parser.add_argument('--user_id', type=int, default=12345,
                       help='User ID for single mode')
    parser.add_argument('--user_ids', type=int, nargs='+',
                       help='User IDs for batch mode')
    
    args = parser.parse_args()
    
    # Initialize engine
    print("\n" + "🚀 "*30)
    print("INITIALIZING RECOMMENDATION ENGINE")
    print("🚀 "*30)
    
    engine = RecommendationEngine()
    
    # Run demo
    if args.mode == 'single':
        demo_single_user(engine, user_id=args.user_id)
    
    elif args.mode == 'batch':
        user_ids = args.user_ids or [12345, 67890, 11111, 22222]
        demo_batch_users(engine, user_ids)
    
    elif args.mode == 'cold_start':
        demo_cold_start(engine)
    
    print("\n" + "✅ "*30)
    print("DEMO COMPLETE!")
    print("✅ "*30)