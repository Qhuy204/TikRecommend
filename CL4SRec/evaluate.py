"""
Evaluation Script for CL4SRec
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from config import config
from preprocessing import load_processed_data
from dataset import create_dataloaders
from models import CL4SRecModel


def compute_metrics(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: str,
    k_values: List[int] = None
) -> Dict[str, float]:
    """Compute ranking metrics"""
    model.eval()
    k_values = k_values or [5, 10, 20]
    max_k = max(k_values)
    
    metrics = {f'HR@{k}': 0.0 for k in k_values}
    metrics.update({f'NDCG@{k}': 0.0 for k in k_values})
    metrics['MRR'] = 0.0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_seq = batch['input_seq'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['target'].to(device)
            
            _, indices = model.predict(input_seq, attention_mask, top_k=max_k)
            
            batch_size = targets.size(0)
            total += batch_size
            
            for k in k_values:
                top_k = indices[:, :k]
                hits = (top_k == targets.unsqueeze(1)).any(dim=1).float()
                metrics[f'HR@{k}'] += hits.sum().item()
                
                for i in range(batch_size):
                    if targets[i].item() in top_k[i].tolist():
                        rank = (top_k[i] == targets[i]).nonzero(as_tuple=True)[0].item() + 1
                        metrics[f'NDCG@{k}'] += 1.0 / np.log2(rank + 1)
            
            for i in range(batch_size):
                if targets[i].item() in indices[i].tolist():
                    rank = (indices[i] == targets[i]).nonzero(as_tuple=True)[0].item() + 1
                    metrics['MRR'] += 1.0 / rank
    
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
    parser = argparse.ArgumentParser(description='Evaluate CL4SRec')
    parser.add_argument('--checkpoint', type=str, default='best_model.pt')
    parser.add_argument('--split', type=str, default='test', choices=['val', 'test'])
    args = parser.parse_args()
    
    device = config.training.device
    
    # Load data
    print("Loading data...")
    data = load_processed_data()
    _, val_loader, test_loader = create_dataloaders(data, use_augmentation=False)
    
    eval_loader = test_loader if args.split == 'test' else val_loader
    
    # Load model
    print("Loading model...")
    checkpoint_path = Path(config.training.checkpoint_dir) / args.checkpoint
    
    if not checkpoint_path.exists():
        print(f"Not found: {checkpoint_path}")
        return
    
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    model = CL4SRecModel(data.num_items, cl_weight=ckpt.get('cl_weight', 0.1))
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    
    print(f"   Loaded: epoch {ckpt['epoch']}, HR@10={ckpt['best_metric']:.4f}")
    
    # Evaluate
    metrics = compute_metrics(model, eval_loader, device)
    print_metrics(metrics, f"Test Results ({args.split})")
    
    # Save
    results_path = Path(config.training.checkpoint_dir) / f'eval_{args.split}.json'
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved: {results_path}")


if __name__ == "__main__":
    main()
