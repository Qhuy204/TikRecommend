#!/usr/bin/env python3
"""
Evaluation script for Hybrid LightGCN + TF-IDF Recommender
Compare performance across different alpha values
"""

import argparse
import random
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
import pickle
import time

import numpy as np
from tqdm import tqdm

from config import config
from model import load_hybrid_recommender, HybridRecommender


def compute_metrics(
    recommendations: List[int],
    ground_truth: int,
    k: int = 10
) -> Tuple[float, float]:
    """
    Compute HR@K and NDCG@K for a single user
    
    Args:
        recommendations: List of recommended item indices
        ground_truth: True item index (leave-one-out evaluation)
        k: Cutoff for metrics
        
    Returns:
        (hit, ndcg) tuple
    """
    recommendations = recommendations[:k]
    
    # Hit Rate
    hit = 1.0 if ground_truth in recommendations else 0.0
    
    # NDCG
    ndcg = 0.0
    if ground_truth in recommendations:
        rank = recommendations.index(ground_truth)
        ndcg = 1.0 / np.log2(rank + 2)  # +2 because rank is 0-indexed
    
    return hit, ndcg


def evaluate_hybrid(
    hybrid: HybridRecommender,
    test_data: List[Tuple[int, set, int]],
    alpha: float = 0.7,
    top_k: int = 10
) -> Dict[str, float]:
    """
    Evaluate hybrid recommender on test data
    
    Args:
        hybrid: HybridRecommender instance
        test_data: List of (user_idx, user_history, ground_truth_item_idx)
        alpha: Fusion weight
        top_k: Cutoff for metrics
        
    Returns:
        Dict with HR@K and NDCG@K
    """
    hybrid.set_alpha(alpha)
    
    hits = []
    ndcgs = []
    
    for user_idx, user_history, gt_item in tqdm(test_data, desc=f"α={alpha:.1f}"):
        # Get recommendations
        recs, _ = hybrid.recommend(user_idx, user_history, top_k=top_k)
        
        # Compute metrics
        hit, ndcg = compute_metrics(recs, gt_item, k=top_k)
        hits.append(hit)
        ndcgs.append(ndcg)
    
    return {
        f'HR@{top_k}': np.mean(hits),
        f'NDCG@{top_k}': np.mean(ndcgs)
    }


def prepare_test_data(
    lightgcn_data,
    sample_users: int = 5000
) -> List[Tuple[int, set, int]]:
    """
    Prepare leave-one-out test data
    
    For each user, hold out one random interaction as ground truth
    
    Args:
        lightgcn_data: LightGCN processed data
        sample_users: Number of users to sample
        
    Returns:
        List of (user_idx, history_without_gt, gt_item_idx)
    """
    test_data = []
    
    all_users = list(lightgcn_data.user_pos_items.keys())
    
    # Filter users with at least 2 interactions
    eligible_users = [
        u for u in all_users 
        if len(lightgcn_data.user_pos_items.get(u, set())) >= 2
    ]
    
    # Sample users
    if sample_users < len(eligible_users):
        sample_users_list = random.sample(eligible_users, sample_users)
    else:
        sample_users_list = eligible_users
    
    for user_idx in sample_users_list:
        user_items = list(lightgcn_data.user_pos_items[user_idx])
        
        # Hold out one random item as ground truth
        gt_item = random.choice(user_items)
        history = set(user_items) - {gt_item}
        
        test_data.append((user_idx, history, gt_item))
    
    return test_data


def run_evaluation(
    sample_users: int = 5000,
    alpha_values: List[float] = None,
    top_k: int = 10,
    device: str = "cuda"
):
    """Run full evaluation with multiple alpha values"""
    
    if alpha_values is None:
        alpha_values = config.eval.alpha_values
    
    print("=" * 60)
    print("HYBRID RECOMMENDER EVALUATION")
    print("=" * 60)
    
    # Load hybrid recommender
    hybrid = load_hybrid_recommender(device=device)
    
    # Prepare test data
    print(f"\nPreparing test data ({sample_users} users)...")
    test_data = prepare_test_data(hybrid.lightgcn_data, sample_users)
    print(f"   Test samples: {len(test_data)}")
    
    # Evaluate for each alpha
    print(f"\nEvaluating with K={top_k}...")
    results = {}
    
    for alpha in alpha_values:
        start_time = time.time()
        metrics = evaluate_hybrid(hybrid, test_data, alpha=alpha, top_k=top_k)
        elapsed = time.time() - start_time
        
        results[alpha] = metrics
        print(f"   α={alpha:.1f}: HR@{top_k}={metrics[f'HR@{top_k}']:.4f}, "
              f"NDCG@{top_k}={metrics[f'NDCG@{top_k}']:.4f} ({elapsed:.1f}s)")
    
    # Print summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n{'Method':<25} {'HR@' + str(top_k):<12} {'NDCG@' + str(top_k):<12}")
    print("-" * 50)
    
    method_names = {
        0.0: "TF-IDF only (α=0.0)",
        0.3: "Hybrid (α=0.3)",
        0.5: "Hybrid (α=0.5)",
        0.7: "Hybrid (α=0.7) *",
        1.0: "LightGCN only (α=1.0)",
    }
    
    for alpha in sorted(results.keys()):
        name = method_names.get(alpha, f"Hybrid (α={alpha:.1f})")
        metrics = results[alpha]
        print(f"{name:<25} {metrics[f'HR@{top_k}']:<12.4f} {metrics[f'NDCG@{top_k}']:<12.4f}")
    
    print("\n* Default setting")
    
    # Find best alpha
    best_alpha = max(results.keys(), key=lambda a: results[a][f'HR@{top_k}'])
    print(f"\nBest α = {best_alpha:.1f} with HR@{top_k} = {results[best_alpha][f'HR@{top_k}']:.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Hybrid Recommender")
    parser.add_argument("--sample-users", type=int, default=5000,
                        help="Number of users to sample for evaluation")
    parser.add_argument("--alpha", type=float, default=None,
                        help="Single alpha value to test (if not set, tests multiple)")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Cutoff K for metrics")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"], help="Device for inference")
    
    args = parser.parse_args()
    
    # Determine alpha values
    if args.alpha is not None:
        alpha_values = [args.alpha]
    else:
        alpha_values = [0.0, 0.3, 0.5, 0.7, 1.0]
    
    # Run evaluation
    results = run_evaluation(
        sample_users=args.sample_users,
        alpha_values=alpha_values,
        top_k=args.top_k,
        device=args.device
    )


if __name__ == "__main__":
    main()
