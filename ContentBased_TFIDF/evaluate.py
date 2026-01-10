"""
Evaluation Script for TF-IDF Content-Based Recommender
Evaluates on user's held-out positive items
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List
import numpy as np
from tqdm import tqdm

from preprocessing import load_processed_data, ProcessedData
from model import TFIDFRecommender
from config import config


def create_train_test_split(data: ProcessedData, test_ratio: float = 0.2):
    """
    Split user's positive items into train/test
    For each user, hold out test_ratio of their liked items for evaluation
    """
    train_items = {}  # user_idx -> set of train item indices
    test_items = {}   # user_idx -> set of test item indices
    
    for user_idx, pos_items in data.user_pos_items.items():
        pos_list = list(pos_items)
        if len(pos_list) < 2:
            train_items[user_idx] = pos_items
            test_items[user_idx] = set()
            continue
        
        # Hold out last test_ratio items
        n_test = max(1, int(len(pos_list) * test_ratio))
        np.random.shuffle(pos_list)
        
        test_items[user_idx] = set(pos_list[:n_test])
        train_items[user_idx] = set(pos_list[n_test:])
    
    return train_items, test_items


def compute_metrics(
    recommender: TFIDFRecommender,
    data: ProcessedData,
    train_items: Dict[int, set],
    test_items: Dict[int, set],
    k_values: List[int] = None,
    sample_users: int = None
) -> Dict[str, float]:
    """
    Compute HR@K, NDCG@K, MRR for recommendations
    
    Args:
        sample_users: If set, only evaluate on this many random users
    """
    k_values = k_values or [5, 10, 20]
    max_k = max(k_values)
    
    metrics = {f'HR@{k}': 0.0 for k in k_values}
    metrics.update({f'NDCG@{k}': 0.0 for k in k_values})
    metrics['MRR'] = 0.0
    total = 0
    
    # Get users to evaluate
    all_users = list(data.user_pos_items.keys())
    if sample_users and sample_users < len(all_users):
        np.random.shuffle(all_users)
        eval_users = all_users[:sample_users]
        print(f"   ⚡ Sampling {sample_users:,} users for faster evaluation")
    else:
        eval_users = all_users
    
    for user_idx in tqdm(eval_users, desc="Evaluating"):
        train = train_items.get(user_idx, set())
        test = test_items.get(user_idx, set())
        
        if len(train) == 0 or len(test) == 0:
            continue
        
        # Get train items as product IDs
        train_item_ids = [data.idx2item[idx] for idx in train]
        
        # Get recommendations
        recs = recommender.recommend_for_items(train_item_ids, top_k=max_k + len(train))
        
        # Get recommended indices, filtering out train items
        rec_indices = []
        for rec in recs:
            item_id = rec['product_id']
            if item_id in data.item2idx:
                idx = data.item2idx[item_id]
                if idx not in train:
                    rec_indices.append(idx)
                    if len(rec_indices) >= max_k:
                        break
        
        # Compute metrics for each test item
        for target in test:
            for k in k_values:
                if target in rec_indices[:k]:
                    metrics[f'HR@{k}'] += 1
                    rank = rec_indices[:k].index(target)
                    metrics[f'NDCG@{k}'] += 1.0 / np.log2(rank + 2)
            
            if target in rec_indices:
                rank = rec_indices.index(target)
                metrics['MRR'] += 1.0 / (rank + 1)
            
            total += 1
    
    for key in metrics:
        if total > 0:
            metrics[key] /= total
    
    return metrics


def compute_category_precision(
    recommender: TFIDFRecommender,
    data: ProcessedData,
    n_samples: int = 1000,
    top_k: int = 10
) -> float:
    """
    Compute category precision: how often similar items are in the same category
    """
    correct = 0
    total = 0
    
    # Sample items
    item_ids = list(data.item2idx.keys())
    np.random.shuffle(item_ids)
    sample_ids = item_ids[:n_samples]
    
    for item_id in tqdm(sample_ids, desc="Category Precision"):
        info = data.item_info.get(item_id, {})
        query_category = info.get('category', '')
        
        if not query_category:
            continue
        
        similar = recommender.get_similar_items(item_id, top_k=top_k)
        
        for rec in similar:
            rec_category = rec.get('category', '')
            if query_category == rec_category:
                correct += 1
            total += 1
    
    return correct / total if total > 0 else 0.0


def print_metrics(metrics: Dict[str, float], title: str = "Results"):
    print("\n" + "=" * 50)
    print(f"{title}")
    print("=" * 50)
    
    print("\nHit Rate:")
    for k, v in sorted([(k, v) for k, v in metrics.items() if k.startswith('HR@')]):
        print(f"   {k}: {v:.4f}")
    
    print("\nNDCG:")
    for k, v in sorted([(k, v) for k, v in metrics.items() if k.startswith('NDCG@')]):
        print(f"   {k}: {v:.4f}")
    
    print(f"\nMRR: {metrics.get('MRR', 0):.4f}")
    
    if 'CategoryPrecision' in metrics:
        print(f"\nCategory Precision@10: {metrics['CategoryPrecision']:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-ratio', type=float, default=0.2,
                        help='Ratio of each user items to hold out for test')
    parser.add_argument('--sample-users', type=int, default=None,
                        help='Sample N users for faster evaluation (default: all)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    print("=" * 50)
    print("TF-IDF Content-Based Evaluation")
    print("=" * 50)
    
    print("\nLoading data...")
    data, tfidf_matrix = load_processed_data()
    
    print("\nCreating recommender...")
    recommender = TFIDFRecommender(
        tfidf_matrix=tfidf_matrix,
        item2idx=data.item2idx,
        idx2item=data.idx2item,
        item_info=data.item_info,
    )
    
    print("\nCreating train/test split...")
    train_items, test_items = create_train_test_split(data, args.test_ratio)
    
    n_test_users = sum(1 for v in test_items.values() if len(v) > 0)
    n_test_items = sum(len(v) for v in test_items.values())
    print(f"   Test users: {n_test_users:,}")
    print(f"   Test items: {n_test_items:,} (held-out {args.test_ratio*100:.0f}% per user)")
    
    print("\nComputing metrics...")
    metrics = compute_metrics(
        recommender, data, train_items, test_items,
        sample_users=args.sample_users
    )
    
    print("\nComputing category precision...")
    cat_prec = compute_category_precision(recommender, data)
    metrics['CategoryPrecision'] = cat_prec
    
    print_metrics(metrics, "TF-IDF Results")
    
    # Save results
    results_path = Path("eval_results.json")
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved: {results_path}")


if __name__ == "__main__":
    main()
