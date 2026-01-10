"""
Dataset for LightGCN
BPR training with negative sampling
"""

import random
from typing import List, Tuple, Dict, Set
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from config import config
from preprocessing import ProcessedData


class BPRDataset(Dataset):
    """
    BPR Dataset for LightGCN
    
    Returns (user, positive_item, negative_item) triplets
    """
    
    def __init__(
        self,
        interactions: List[Tuple[int, int]],
        user_pos_items: Dict[int, Set[int]],
        num_items: int,
        num_negatives: int = 1
    ):
        self.interactions = interactions
        self.user_pos_items = user_pos_items
        self.num_items = num_items
        self.num_negatives = num_negatives
    
    def __len__(self) -> int:
        return len(self.interactions)
    
    def __getitem__(self, idx: int) -> Tuple[int, int, int]:
        user, pos_item = self.interactions[idx]
        
        # Sample negative item
        neg_item = random.randint(0, self.num_items - 1)
        while neg_item in self.user_pos_items.get(user, set()):
            neg_item = random.randint(0, self.num_items - 1)
        
        return user, pos_item, neg_item


class EvalDataset(Dataset):
    """Dataset for evaluation - returns (user, single_target_item)"""
    
    def __init__(self, interactions: List[Tuple[int, int]]):
        # Each interaction is a separate sample
        self.interactions = interactions
    
    def __len__(self) -> int:
        return len(self.interactions)
    
    def __getitem__(self, idx: int) -> Tuple[int, int]:
        user, item = self.interactions[idx]
        return user, item


def create_dataloaders(data: ProcessedData, batch_size: int = 2048) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create dataloaders for LightGCN"""
    
    train_ds = BPRDataset(
        data.train_interactions,
        data.user_pos_items,
        data.num_items
    )
    
    val_ds = EvalDataset(data.val_interactions)
    test_ds = EvalDataset(data.test_interactions)
    
    print(f"Dataset: Train {len(train_ds):,} | Val {len(val_ds):,} | Test {len(test_ds):,}")
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                             num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    print("Testing dataset...")
    from preprocessing import load_processed_data
    
    data = load_processed_data()
    train_loader, val_loader, _ = create_dataloaders(data, batch_size=32)
    
    batch = next(iter(train_loader))
    print(f"Train batch: users={batch[0].shape}, pos={batch[1].shape}, neg={batch[2].shape}")
    
    val_batch = next(iter(val_loader))
    print(f"Val batch: users={val_batch[0].shape}, targets={val_batch[1].shape}")
    print("Dataset test passed!")

