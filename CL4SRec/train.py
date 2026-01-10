"""
Training Script for CL4SRec
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import config
from preprocessing import load_processed_data, is_cache_valid
from dataset import create_dataloaders
from models import CL4SRecModel


class CL4SRecTrainer:
    """Trainer for CL4SRec"""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5
    ):
        self.model = model
        self.device = device or config.training.device
        self.model = self.model.to(self.device)
        
        self.optimizer = optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        
        self.epoch = 0
        self.best_metric = 0.0
        self.patience_counter = 0
        self.history = []
        
        Path(config.training.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train one epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_bpr = 0.0
        total_cl = 0.0
        total_samples = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.epoch + 1}")
        
        for batch in pbar:
            # Move to device
            input_seq = batch['input_seq'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            positive = batch['positive_item'].to(self.device)
            negative = batch['negative_items'].to(self.device)
            
            # Augmented sequences (if available)
            aug_seq1 = batch.get('aug_seq1')
            aug_mask1 = batch.get('aug_mask1')
            aug_seq2 = batch.get('aug_seq2')
            aug_mask2 = batch.get('aug_mask2')
            
            if aug_seq1 is not None:
                aug_seq1 = aug_seq1.to(self.device)
                aug_mask1 = aug_mask1.to(self.device)
                aug_seq2 = aug_seq2.to(self.device)
                aug_mask2 = aug_mask2.to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            output = self.model(
                input_seq, attention_mask, positive, negative,
                aug_seq1, aug_mask1, aug_seq2, aug_mask2
            )
            
            loss = output['loss']
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Stats
            batch_size = input_seq.size(0)
            total_loss += loss.item() * batch_size
            total_bpr += output['bpr_loss'].item() * batch_size
            total_cl += output['cl_loss'].item() * batch_size
            total_samples += batch_size
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'bpr': f'{output["bpr_loss"].item():.4f}',
                'cl': f'{output["cl_loss"].item():.4f}'
            })
        
        return {
            'loss': total_loss / total_samples,
            'bpr_loss': total_bpr / total_samples,
            'cl_loss': total_cl / total_samples
        }
    
    @torch.no_grad()
    def evaluate(self, eval_loader: DataLoader, k_values: list = None) -> Dict[str, float]:
        """Evaluate with ranking metrics"""
        self.model.eval()
        k_values = k_values or [5, 10, 20]
        
        metrics = {f'HR@{k}': 0.0 for k in k_values}
        metrics.update({f'NDCG@{k}': 0.0 for k in k_values})
        total = 0
        
        for batch in tqdm(eval_loader, desc="Evaluating"):
            input_seq = batch['input_seq'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            targets = batch['target'].to(self.device)
            
            _, indices = self.model.predict(input_seq, attention_mask, top_k=max(k_values))
            
            batch_size = targets.size(0)
            total += batch_size
            
            for k in k_values:
                top_k = indices[:, :k]
                hits = (top_k == targets.unsqueeze(1)).any(dim=1).float()
                metrics[f'HR@{k}'] += hits.sum().item()
                
                for i in range(batch_size):
                    if targets[i].item() in top_k[i].tolist():
                        rank = (top_k[i] == targets[i]).nonzero(as_tuple=True)[0].item() + 1
                        metrics[f'NDCG@{k}'] += 1.0 / (torch.log2(torch.tensor(rank + 1.0))).item()
        
        for key in metrics:
            metrics[key] /= total
        
        return metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 30,
        patience: int = 5
    ) -> Dict:
        """Full training loop"""
        print("\n" + "=" * 60)
        print("CL4SRec Training")
        print("=" * 60)
        print(f"   Epochs: {epochs}")
        print(f"   Device: {self.device}")
        print(f"   CL Weight: {self.model.cl_weight}")
        print(f"   Training samples: {len(train_loader.dataset):,}")
        
        start_time = time.time()
        
        for epoch in range(epochs):
            self.epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.evaluate(val_loader) if val_loader else {}
            current_metric = val_metrics.get('HR@10', 0.0)
            
            # Log
            self.history.append({
                'epoch': epoch + 1,
                **train_metrics,
                **val_metrics
            })
            
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print(f"   Loss: {train_metrics['loss']:.4f} (BPR: {train_metrics['bpr_loss']:.4f}, CL: {train_metrics['cl_loss']:.4f})")
            if val_metrics:
                print(f"   HR@10: {val_metrics.get('HR@10', 0):.4f} | NDCG@10: {val_metrics.get('NDCG@10', 0):.4f}")
            
            # Early stopping
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
            'num_items': self.model.num_items,
            'cl_weight': self.model.cl_weight,
        }, path)
    
    def _save_history(self):
        path = Path(config.training.checkpoint_dir) / 'history.json'
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Train CL4SRec')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--cl_weight', type=float, default=0.1, help='Weight for CL loss')
    parser.add_argument('--no_cl', action='store_true', help='Disable contrastive learning')
    args = parser.parse_args()
    
    if not is_cache_valid():
        print("Run: python preprocessing.py first")
        return
    
    # Load data
    print("Loading data...")
    data = load_processed_data()
    
    train_loader, val_loader, _ = create_dataloaders(
        data, batch_size=args.batch_size,
        use_augmentation=not args.no_cl
    )
    
    # Create model
    print(f"\nCreating CL4SRec model...")
    model = CL4SRecModel(
        num_items=data.num_items,
        cl_weight=args.cl_weight if not args.no_cl else 0.0
    )
    print(f"   Items: {data.num_items:,}")
    print(f"   CL Weight: {model.cl_weight}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    trainer = CL4SRecTrainer(model, learning_rate=args.lr)
    trainer.train(train_loader, val_loader, epochs=args.epochs)


if __name__ == "__main__":
    main()
