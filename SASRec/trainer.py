"""
Training Pipeline for SASRec + PhoBERT Fusion
"""

import os
import time
import json
from pathlib import Path
from typing import Dict, Optional, Tuple
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config import TrainingConfig, default_config
from models import SASRecModel, SASRecPhoBERTFusion


class Trainer:
    """
    Trainer for SASRec models
    Supports both pure SASRec and SASRec+PhoBERT fusion
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig = None,
        device: str = None
    ):
        self.model = model
        self.config = config or default_config.training
        self.device = device or self.config.device
        
        self.model = self.model.to(self.device)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.epochs
        )
        
        # Training state
        self.epoch = 0
        self.best_metric = 0.0
        self.patience_counter = 0
        self.training_history = []
        
        # Create checkpoint directory
        Path(self.config.save_dir).mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_samples = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.epoch + 1}")
        
        for batch in pbar:
            # Move to device
            input_seq = batch['input_seq'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            positive = batch['positive'].to(self.device)
            negative = batch['negatives'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            loss, pos_scores, neg_scores = self.model(
                input_seq, attention_mask, positive, negative
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update stats
            batch_size = input_seq.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / total_samples
        
        return {'loss': avg_loss}
    
    @torch.no_grad()
    def evaluate(
        self, 
        eval_loader: DataLoader,
        k_values: list = [5, 10, 20]
    ) -> Dict[str, float]:
        """Evaluate model with ranking metrics"""
        self.model.eval()
        
        metrics = {f'HR@{k}': 0.0 for k in k_values}
        metrics.update({f'NDCG@{k}': 0.0 for k in k_values})
        total_samples = 0
        
        for batch in tqdm(eval_loader, desc="Evaluating"):
            input_seq = batch['input_seq'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            targets = batch['target'].to(self.device)
            
            # Get predictions
            if hasattr(self.model, 'predict'):
                _, top_indices = self.model.predict(
                    input_seq, attention_mask, top_k=max(k_values)
                )
            else:
                # For fusion model
                scores = self.model.get_item_scores(input_seq, attention_mask)
                _, top_indices = torch.topk(scores, k=max(k_values), dim=-1)
            
            # Calculate metrics
            batch_size = targets.size(0)
            total_samples += batch_size
            
            for k in k_values:
                top_k = top_indices[:, :k]
                
                # Hit Rate
                hits = (top_k == targets.unsqueeze(1)).any(dim=1).float()
                metrics[f'HR@{k}'] += hits.sum().item()
                
                # NDCG
                for i in range(batch_size):
                    target = targets[i].item()
                    if target in top_k[i].tolist():
                        rank = (top_k[i] == target).nonzero(as_tuple=True)[0].item() + 1
                        metrics[f'NDCG@{k}'] += 1.0 / torch.log2(torch.tensor(rank + 1.0)).item()
        
        # Average metrics
        for key in metrics:
            metrics[key] /= total_samples
        
        return metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: Optional[int] = None
    ) -> Dict:
        """Full training loop"""
        epochs = epochs or self.config.epochs
        
        print(f"\nStarting training for {epochs} epochs")
        print(f"   Device: {self.device}")
        print(f"   Batch size: {train_loader.batch_size}")
        print(f"   Training samples: {len(train_loader.dataset):,}")
        
        start_time = time.time()
        
        for epoch in range(epochs):
            self.epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Evaluate
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                current_metric = val_metrics.get('HR@10', 0.0)
            else:
                val_metrics = {}
                current_metric = -train_metrics['loss']  # Use negative loss
            
            # Learning rate step
            self.scheduler.step()
            
            # Log
            log_entry = {
                'epoch': epoch + 1,
                'train_loss': train_metrics['loss'],
                'lr': self.scheduler.get_last_lr()[0],
                **val_metrics
            }
            self.training_history.append(log_entry)
            
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print(f"   Train Loss: {train_metrics['loss']:.4f}")
            if val_metrics:
                print(f"   Val HR@10: {val_metrics.get('HR@10', 0):.4f}")
                print(f"   Val NDCG@10: {val_metrics.get('NDCG@10', 0):.4f}")
            
            # Early stopping / model saving
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                self.patience_counter = 0
                self._save_checkpoint('best_model.pt')
                print(f"   New best model saved!")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.patience:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                    break
        
        # Training complete
        total_time = time.time() - start_time
        print(f"\nTraining complete in {total_time/60:.1f} minutes")
        print(f"   Best metric: {self.best_metric:.4f}")
        
        # Save final model
        self._save_checkpoint('final_model.pt')
        self._save_history()
        
        return {
            'best_metric': self.best_metric,
            'total_time': total_time,
            'history': self.training_history
        }
    
    def _save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        path = Path(self.config.save_dir) / filename
        
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'training_history': self.training_history
        }
        
        torch.save(checkpoint, path)
    
    def _save_history(self):
        """Save training history as JSON"""
        path = Path(self.config.save_dir) / 'training_history.json'
        with open(path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
    
    def load_checkpoint(self, filename: str = 'best_model.pt'):
        """Load model from checkpoint"""
        path = Path(self.config.save_dir) / filename
        
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_metric = checkpoint['best_metric']
        self.training_history = checkpoint.get('training_history', [])
        
        print(f"Loaded checkpoint from {path}")
        print(f"   Epoch: {self.epoch}, Best metric: {self.best_metric:.4f}")


class EarlyStopping:
    """Early stopping helper"""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_value = None
        self.should_stop = False
    
    def __call__(self, value: float) -> bool:
        if self.best_value is None:
            self.best_value = value
            return False
        
        if value > self.best_value + self.min_delta:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


if __name__ == "__main__":
    # Test trainer
    from models import SASRecModel
    
    print("Testing Trainer...")
    
    model = SASRecModel(num_items=1000)
    trainer = Trainer(model)
    
    print(f"Device: {trainer.device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Trainer test passed!")
