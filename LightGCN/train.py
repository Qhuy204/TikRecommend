"""
Training Script for LightGCN
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict

import torch
import torch.optim as optim
from tqdm import tqdm

from config import config
from preprocessing import load_processed_data, is_cache_valid
from dataset import create_dataloaders
from models import LightGCN


class Trainer:
    def __init__(self, model: LightGCN, device: str = None, lr: float = 1e-3):
        self.model = model
        self.device = device or config.training.device
        self.model = self.model.to(self.device)
        
        # Move graph to device
        if self.model.adj_matrix is not None:
            self.model.adj_matrix = self.model.adj_matrix.to(self.device)
        
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        
        self.epoch = 0
        self.best_metric = 0.0
        self.patience_counter = 0
        self.history = []
        
        Path(config.training.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self, train_loader) -> Dict[str, float]:
        self.model.train()
        
        total_loss = 0.0
        total_bpr = 0.0
        total_samples = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.epoch + 1}")
        
        for batch in pbar:
            users, pos_items, neg_items = batch
            users = users.to(self.device)
            pos_items = pos_items.to(self.device)
            neg_items = neg_items.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(users, pos_items, neg_items)
            
            loss = output['loss']
            loss.backward()
            self.optimizer.step()
            
            batch_size = users.size(0)
            total_loss += loss.item() * batch_size
            total_bpr += output['bpr_loss'].item() * batch_size
            total_samples += batch_size
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'bpr': f'{output["bpr_loss"].item():.4f}'
            })
        
        return {
            'loss': total_loss / total_samples,
            'bpr_loss': total_bpr / total_samples
        }
    
    @torch.no_grad()
    def evaluate(self, eval_loader, user_pos_items: dict, k_values: list = None) -> Dict[str, float]:
        self.model.eval()
        k_values = k_values or [5, 10, 20]
        max_k = max(k_values)
        
        metrics = {f'HR@{k}': 0.0 for k in k_values}
        metrics.update({f'NDCG@{k}': 0.0 for k in k_values})
        total = 0
        
        for batch in tqdm(eval_loader, desc="Evaluating"):
            users, targets = batch
            users = users.to(self.device)
            targets = targets.to(self.device)
            
            _, indices = self.model.predict(users, top_k=max_k + 100)
            
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
                        metrics[f'NDCG@{k}'] += 1.0 / (torch.log2(torch.tensor(rank + 2.0))).item()
                
                total += 1
        
        for key in metrics:
            metrics[key] /= total
        
        return metrics
    
    def train(self, train_loader, val_loader, user_pos_items: dict, epochs: int = 50, patience: int = 10) -> Dict:
        print("\n" + "=" * 60)
        print("LightGCN Training")
        print("=" * 60)
        print(f"   Epochs: {epochs}")
        print(f"   Device: {self.device}")
        print(f"   GCN Layers: {self.model.cfg.num_layers}")
        print(f"   Samples: {len(train_loader.dataset):,}")
        
        start_time = time.time()
        
        for epoch in range(epochs):
            self.epoch = epoch
            
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader, user_pos_items)
            current_metric = val_metrics.get('HR@10', 0.0)
            
            self.history.append({'epoch': epoch + 1, **train_metrics, **val_metrics})
            
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print(f"   Loss: {train_metrics['loss']:.4f} | BPR: {train_metrics['bpr_loss']:.4f}")
            print(f"   HR@10: {val_metrics.get('HR@10', 0):.4f} | NDCG@10: {val_metrics.get('NDCG@10', 0):.4f}")
            
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                self.patience_counter = 0
                self._save_checkpoint('best_model.pt')
                print("   New best!")
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                    break
        
        total_time = time.time() - start_time
        print(f"\nTraining complete in {total_time/60:.1f} min")
        print(f"   Best HR@10: {self.best_metric:.4f}")
        
        self._save_checkpoint('final_model.pt')
        self._save_history()
        
        return {'best_metric': self.best_metric, 'history': self.history}
    
    def _save_checkpoint(self, filename: str):
        path = Path(config.training.checkpoint_dir) / filename
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'num_users': self.model.num_users,
            'num_items': self.model.num_items,
        }, path)
    
    def _save_history(self):
        path = Path(config.training.checkpoint_dir) / 'history.json'
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--layers', type=int, default=3)
    args = parser.parse_args()
    
    if not is_cache_valid():
        print("Run: python preprocessing.py first")
        return
    
    print("Loading data...")
    data = load_processed_data()
    
    train_loader, val_loader, _ = create_dataloaders(data, batch_size=args.batch_size)
    
    print(f"\nCreating LightGCN model...")
    config.model.num_layers = args.layers
    
    model = LightGCN(
        num_users=data.num_users,
        num_items=data.num_items,
        adj_matrix=data.adj_matrix
    )
    print(f"   Users: {data.num_users:,}")
    print(f"   Items: {data.num_items:,}")
    print(f"   GCN Layers: {args.layers}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    trainer = Trainer(model, lr=args.lr)
    trainer.train(train_loader, val_loader, data.user_pos_items, epochs=args.epochs)


if __name__ == "__main__":
    main()
