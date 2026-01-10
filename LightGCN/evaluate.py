"""
Evaluation Script for LightGCN
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import torch
from tqdm import tqdm
import numpy as np

from config import config
from preprocessing import load_processed_data
from dataset import create_dataloaders
from models import LightGCN


def compute_metrics(model, eval_loader, user_pos_items, device, k_values=None) -> Dict[str, float]:
    model.eval()
    k_values = k_values or [5, 10, 20]
    max_k = max(k_values)
    
    metrics = {f'HR@{k}': 0.0 for k in k_values}
    metrics.update({f'NDCG@{k}': 0.0 for k in k_values})
    metrics['MRR'] = 0.0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            users, targets = batch
            users = users.to(device)
            targets = targets.to(device)
            
            _, indices = model.predict(users, top_k=max_k + 100)
            
            for i in range(users.size(0)):
                user = users[i].item()
                target = targets[i].item()
                seen = user_pos_items.get(user, set())
                
                # Filter seen items (but keep target for evaluation)
                preds = [idx for idx in indices[i].tolist() if idx not in seen or idx == target][:max_k]
                
                for k in k_values:
                    if target in preds[:k]:
                        metrics[f'HR@{k}'] += 1
                        rank = preds[:k].index(target)
                        metrics[f'NDCG@{k}'] += 1.0 / np.log2(rank + 2)
                
                # MRR
                if target in preds:
                    rank = preds.index(target)
                    metrics['MRR'] += 1.0 / (rank + 1)
                
                total += 1
    
    for key in metrics:
        metrics[key] /= total
    
    return metrics


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='best_model.pt')
    parser.add_argument('--split', type=str, default='test', choices=['val', 'test'])
    args = parser.parse_args()
    
    device = config.training.device
    
    print("Loading data...")
    data = load_processed_data()
    _, val_loader, test_loader = create_dataloaders(data)
    
    eval_loader = test_loader if args.split == 'test' else val_loader
    
    print("Loading model...")
    checkpoint_path = Path(config.training.checkpoint_dir) / args.checkpoint
    
    if not checkpoint_path.exists():
        print(f"Not found: {checkpoint_path}")
        return
    
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    model = LightGCN(
        num_users=data.num_users,
        num_items=data.num_items,
        adj_matrix=data.adj_matrix
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.adj_matrix = model.adj_matrix.to(device)
    
    print(f"   Loaded: epoch {ckpt['epoch']}, HR@10={ckpt['best_metric']:.4f}")
    
    metrics = compute_metrics(model, eval_loader, data.user_pos_items, device)
    print_metrics(metrics, f"LightGCN Results ({args.split})")
    
    results_path = Path(config.training.checkpoint_dir) / f'eval_{args.split}.json'
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved: {results_path}")


if __name__ == "__main__":
    main()
