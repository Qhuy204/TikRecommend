#!/usr/bin/env python3
"""
Alpha Tuning for Hybrid LightGCN + TF-IDF Recommender
Find the optimal alpha weight through grid search
"""

import argparse
import random
from pathlib import Path
from typing import List, Dict, Tuple
import pickle
import time
import json

import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from config import config
from model import load_hybrid_recommender, HybridRecommender


def compute_metrics(
    recommendations: List[int],
    ground_truth: int,
    k: int = 10
) -> Tuple[float, float]:
    """Compute HR@K and NDCG@K for a single user"""
    recommendations = recommendations[:k]
    hit = 1.0 if ground_truth in recommendations else 0.0
    ndcg = 0.0
    if ground_truth in recommendations:
        rank = recommendations.index(ground_truth)
        ndcg = 1.0 / np.log2(rank + 2)
    return hit, ndcg


def prepare_test_data(lightgcn_data, sample_users: int = 5000):
    """Prepare leave-one-out test data"""
    test_data = []
    all_users = list(lightgcn_data.user_pos_items.keys())
    eligible_users = [
        u for u in all_users 
        if len(lightgcn_data.user_pos_items.get(u, set())) >= 2
    ]
    
    if sample_users < len(eligible_users):
        sample_users_list = random.sample(eligible_users, sample_users)
    else:
        sample_users_list = eligible_users
    
    for user_idx in sample_users_list:
        user_items = list(lightgcn_data.user_pos_items[user_idx])
        gt_item = random.choice(user_items)
        history = set(user_items) - {gt_item}
        test_data.append((user_idx, history, gt_item))
    
    return test_data


def evaluate_alpha(
    hybrid: HybridRecommender,
    test_data: List[Tuple[int, set, int]],
    alpha: float,
    top_k: int = 10,
    show_progress: bool = True
) -> Dict[str, float]:
    """Evaluate single alpha value"""
    hybrid.set_alpha(alpha)
    hits, ndcgs = [], []
    
    iterator = test_data
    if show_progress:
        iterator = tqdm(test_data, desc=f"α={alpha:.2f}", leave=False)
    
    for user_idx, user_history, gt_item in iterator:
        recs, _ = hybrid.recommend(user_idx, user_history, top_k=top_k)
        hit, ndcg = compute_metrics(recs, gt_item, k=top_k)
        hits.append(hit)
        ndcgs.append(ndcg)
    
    return {
        f'HR@{top_k}': np.mean(hits),
        f'NDCG@{top_k}': np.mean(ndcgs)
    }


def tune_alpha(
    sample_users: int = 5000,
    alpha_range: Tuple[float, float] = (0.0, 1.0),
    alpha_step: float = 0.1,
    top_k: int = 10,
    device: str = "cuda",
    save_results: bool = True
) -> Dict:
    """
    Grid search for optimal alpha
    
    Args:
        sample_users: Number of users for evaluation
        alpha_range: (min_alpha, max_alpha)
        alpha_step: Step size for alpha grid
        top_k: Cutoff for metrics
        device: Device for inference
        save_results: Whether to save results to file
        
    Returns:
        Dict with all results and best alpha
    """
    print("=" * 60)
    print("ALPHA TUNING - Hybrid Recommender")
    print("=" * 60)
    
    # Generate alpha values
    alpha_values = np.arange(alpha_range[0], alpha_range[1] + alpha_step/2, alpha_step)
    alpha_values = [round(a, 2) for a in alpha_values]
    
    print(f"\nAlpha values to test: {alpha_values}")
    print(f"   Sample users: {sample_users}")
    print(f"   Top-K: {top_k}")
    
    # Load hybrid recommender
    print("\nLoading models...")
    hybrid = load_hybrid_recommender(device=device)
    
    # Prepare test data (same for all alpha values)
    print(f"\nPreparing test data...")
    random.seed(42)  # For reproducibility
    test_data = prepare_test_data(hybrid.lightgcn_data, sample_users)
    print(f"   Test samples: {len(test_data)}")
    
    # Evaluate each alpha
    print(f"\nEvaluating alpha values...")
    results = {}
    
    for alpha in alpha_values:
        start_time = time.time()
        metrics = evaluate_alpha(hybrid, test_data, alpha, top_k)
        elapsed = time.time() - start_time
        
        results[alpha] = {
            'HR': metrics[f'HR@{top_k}'],
            'NDCG': metrics[f'NDCG@{top_k}'],
            'time': elapsed
        }
        
        print(f"   α={alpha:.2f}: HR@{top_k}={metrics[f'HR@{top_k}']:.4f}, "
              f"NDCG@{top_k}={metrics[f'NDCG@{top_k}']:.4f} ({elapsed:.1f}s)")
    
    # Find best alpha
    best_alpha_hr = max(results.keys(), key=lambda a: results[a]['HR'])
    best_alpha_ndcg = max(results.keys(), key=lambda a: results[a]['NDCG'])
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    print(f"\n{'Alpha':<8} {'HR@' + str(top_k):<12} {'NDCG@' + str(top_k):<12} {'Description':<25}")
    print("-" * 60)
    
    for alpha in sorted(results.keys()):
        hr = results[alpha]['HR']
        ndcg = results[alpha]['NDCG']
        
        if alpha == 0.0:
            desc = "TF-IDF only"
        elif alpha == 1.0:
            desc = "LightGCN only"
        elif alpha == best_alpha_hr:
            desc = "* Best HR"
        elif alpha == best_alpha_ndcg:
            desc = "* Best NDCG"
        else:
            desc = ""
        
        print(f"{alpha:<8.2f} {hr:<12.4f} {ndcg:<12.4f} {desc}")
    
    print(f"\nBest alpha for HR@{top_k}: {best_alpha_hr:.2f} "
          f"(HR={results[best_alpha_hr]['HR']:.4f})")
    print(f"Best alpha for NDCG@{top_k}: {best_alpha_ndcg:.2f} "
          f"(NDCG={results[best_alpha_ndcg]['NDCG']:.4f})")
    
    # Calculate improvement over baselines
    baseline_lgcn_hr = results[1.0]['HR']
    baseline_tfidf_hr = results[0.0]['HR']
    best_hr = results[best_alpha_hr]['HR']
    
    improvement_over_lgcn = ((best_hr - baseline_lgcn_hr) / baseline_lgcn_hr) * 100
    improvement_over_tfidf = ((best_hr - baseline_tfidf_hr) / baseline_tfidf_hr) * 100
    
    print(f"\nImprovement over LightGCN: {improvement_over_lgcn:+.2f}%")
    print(f"Improvement over TF-IDF: {improvement_over_tfidf:+.2f}%")
    
    # Prepare output
    output = {
        'alpha_values': alpha_values,
        'results': results,
        'best_alpha_hr': best_alpha_hr,
        'best_alpha_ndcg': best_alpha_ndcg,
        'sample_users': sample_users,
        'top_k': top_k,
        'improvement_over_lgcn': improvement_over_lgcn,
        'improvement_over_tfidf': improvement_over_tfidf
    }
    
    # Save results
    if save_results:
        results_path = Path(__file__).parent / "alpha_tuning_results.json"
        with open(results_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to: {results_path}")
        
        # Generate plot
        plot_alpha_comparison(results, top_k, best_alpha_hr, best_alpha_ndcg)
    
    return output


def plot_alpha_comparison(
    results: Dict, 
    top_k: int,
    best_alpha_hr: float,
    best_alpha_ndcg: float
):
    """Generate comparison plot"""
    alphas = sorted(results.keys())
    hrs = [results[a]['HR'] for a in alphas]
    ndcgs = [results[a]['NDCG'] for a in alphas]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # HR plot
    ax1.plot(alphas, hrs, 'b-o', linewidth=2, markersize=8)
    ax1.axvline(x=best_alpha_hr, color='g', linestyle='--', alpha=0.7, 
                label=f'Best α={best_alpha_hr:.2f}')
    ax1.set_xlabel('α (LightGCN weight)', fontsize=12)
    ax1.set_ylabel(f'HR@{top_k}', fontsize=12)
    ax1.set_title(f'Hit Rate@{top_k} vs Alpha', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlim(-0.05, 1.05)
    
    # NDCG plot
    ax2.plot(alphas, ndcgs, 'r-o', linewidth=2, markersize=8)
    ax2.axvline(x=best_alpha_ndcg, color='g', linestyle='--', alpha=0.7,
                label=f'Best α={best_alpha_ndcg:.2f}')
    ax2.set_xlabel('α (LightGCN weight)', fontsize=12)
    ax2.set_ylabel(f'NDCG@{top_k}', fontsize=12)
    ax2.set_title(f'NDCG@{top_k} vs Alpha', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xlim(-0.05, 1.05)
    
    plt.tight_layout()
    
    plot_path = Path(__file__).parent / "alpha_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Tune alpha for Hybrid Recommender")
    parser.add_argument("--sample-users", type=int, default=5000,
                        help="Number of users for evaluation")
    parser.add_argument("--alpha-min", type=float, default=0.0,
                        help="Minimum alpha value")
    parser.add_argument("--alpha-max", type=float, default=1.0,
                        help="Maximum alpha value")
    parser.add_argument("--alpha-step", type=float, default=0.1,
                        help="Step size for alpha grid")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Cutoff K for metrics")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"], help="Device for inference")
    parser.add_argument("--fine", action="store_true",
                        help="Fine-grained search with 0.05 step")
    
    args = parser.parse_args()
    
    # Fine-grained mode
    if args.fine:
        args.alpha_step = 0.05
    
    # Run tuning
    tune_alpha(
        sample_users=args.sample_users,
        alpha_range=(args.alpha_min, args.alpha_max),
        alpha_step=args.alpha_step,
        top_k=args.top_k,
        device=args.device
    )


if __name__ == "__main__":
    main()
