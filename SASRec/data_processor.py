"""
Data Processor for SASRec + PhoBERT Fusion
Loads tiki_dataset.jsonl and prepares sequential training data
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader

from config import DataConfig, default_config


class TikiDataProcessor:
    """
    Process tiki_dataset.jsonl into sequential recommendation format
    """
    
    def __init__(self, config: DataConfig = None):
        self.config = config or default_config.data
        
        # Mappings
        self.user2idx: Dict[int, int] = {}
        self.idx2user: Dict[int, int] = {}
        self.item2idx: Dict[int, int] = {}
        self.idx2item: Dict[int, int] = {}
        
        # Data
        self.item_info: Dict[int, dict] = {}  # item_id -> {name, description, category, price}
        self.user_sequences: Dict[int, List[Tuple[int, int, float]]] = defaultdict(list)  # user_id -> [(item_id, timestamp, rating), ...]
        
        self.num_users = 0
        self.num_items = 0
        
    def load_raw_data(self, filepath: str = None) -> None:
        """Load and parse tiki_dataset.jsonl"""
        filepath = filepath or self.config.raw_data_path
        filepath = Path(filepath)
        
        if not filepath.exists():
            # Try relative to Newmethod folder
            filepath = Path(__file__).parent / filepath
        
        print(f"Loading data from {filepath}...")
        
        item_set = set()
        seen_product_ids = set()  # Track duplicates
        user_interactions = defaultdict(list)
        
        skipped_errors = 0
        skipped_duplicates = 0
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Reading JSONL"):
                try:
                    data = json.loads(line.strip())
                    
                    product_id = data['product_id']
                    category = data.get('category', 'Unknown')
                    
                    # Extract product info
                    product_detail = data.get('product_detail', {})
                    
                    # Skip error/redirect entries
                    detail_str = str(product_detail).lower()
                    if 'error' in detail_str or 'redirect' in detail_str:
                        skipped_errors += 1
                        continue
                    
                    # Skip if product has no name (invalid entry)
                    if not product_detail.get('name'):
                        skipped_errors += 1
                        continue
                    
                    # Skip duplicates (keep first occurrence)
                    if product_id in seen_product_ids:
                        skipped_duplicates += 1
                        continue
                    seen_product_ids.add(product_id)
                    self.item_info[product_id] = {
                        'name': product_detail.get('name', ''),
                        'description': product_detail.get('short_description', ''),
                        'category': category,
                        'price': product_detail.get('price', 0),
                    }
                    item_set.add(product_id)
                    
                    # Extract reviews (user interactions)
                    reviews = data.get('reviews', [])
                    for review in reviews:
                        user_id = review.get('customer_id')
                        rating = review.get('rating', 0)
                        timestamp = review.get('created_at', 0)
                        
                        if user_id and rating > 0:
                            user_interactions[user_id].append({
                                'item_id': product_id,
                                'timestamp': timestamp,
                                'rating': rating
                            })
                            
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    continue
        
        # Build mappings
        print("Building ID mappings...")
        
        # Item mapping
        for idx, item_id in enumerate(sorted(item_set)):
            self.item2idx[item_id] = idx
            self.idx2item[idx] = item_id
        self.num_items = len(self.item2idx)
        
        # User mapping (only users with enough interactions)
        valid_users = [
            uid for uid, interactions in user_interactions.items()
            if len(interactions) >= self.config.min_seq_length
        ]
        
        for idx, user_id in enumerate(sorted(valid_users)):
            self.user2idx[user_id] = idx
            self.idx2user[idx] = user_id
            
            # Sort by timestamp and store
            sorted_interactions = sorted(
                user_interactions[user_id], 
                key=lambda x: x['timestamp']
            )
            self.user_sequences[idx] = [
                (self.item2idx[x['item_id']], x['timestamp'], x['rating'])
                for x in sorted_interactions
                if x['item_id'] in self.item2idx
            ]
        
        self.num_users = len(self.user2idx)
        
        print(f"Loaded {self.num_items:,} items, {self.num_users:,} users")
        print(f"   Skipped: {skipped_errors:,} errors/redirect, {skipped_duplicates:,} duplicates")
        print(f"   Total interactions: {sum(len(v) for v in self.user_sequences.values()):,}")
    
    def get_item_text(self, item_idx: int) -> str:
        """Get text content for an item"""
        item_id = self.idx2item.get(item_idx)
        if item_id is None:
            return ""
        
        info = self.item_info.get(item_id, {})
        name = info.get('name', '')
        desc = info.get('description', '')
        category = info.get('category', '')
        
        return f"{category}: {name}. {desc}"
    
    def create_train_val_test_split(self) -> Tuple[dict, dict, dict]:
        """
        Split user sequences into train/val/test
        Strategy: For each user, use last item for test, second-last for val, rest for train
        """
        train_data = {}
        val_data = {}
        test_data = {}
        
        for user_idx, sequence in self.user_sequences.items():
            if len(sequence) < 3:
                continue
                
            items = [x[0] for x in sequence]  # Extract item indices
            
            # Leave-one-out split
            train_items = items[:-2]
            val_item = items[-2]
            test_item = items[-1]
            
            if len(train_items) >= 1:
                train_data[user_idx] = train_items
                val_data[user_idx] = (train_items, val_item)
                test_data[user_idx] = (train_items + [val_item], test_item)
        
        print(f"Split: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test users")
        
        return train_data, val_data, test_data


class SASRecDataset(Dataset):
    """
    Dataset for SASRec training
    Returns: (input_sequence, positive_item, negative_items)
    """
    
    def __init__(
        self,
        user_sequences: Dict[int, List[int]],
        num_items: int,
        max_seq_length: int = 50,
        num_negatives: int = 4,
        mode: str = 'train'  # 'train', 'val', 'test'
    ):
        self.user_sequences = user_sequences
        self.num_items = num_items
        self.max_seq_length = max_seq_length
        self.num_negatives = num_negatives
        self.mode = mode
        
        self.users = list(user_sequences.keys())
        
        # For train mode, create all training samples
        if mode == 'train':
            self.samples = self._create_training_samples()
        else:
            self.samples = self.users
    
    def _create_training_samples(self) -> List[Tuple[int, List[int], int]]:
        """Create (user, input_seq, target) samples for training"""
        samples = []
        
        for user_idx, items in self.user_sequences.items():
            # Create samples for each position (except first)
            for i in range(1, len(items)):
                input_seq = items[max(0, i - self.max_seq_length):i]
                target = items[i]
                samples.append((user_idx, input_seq, target))
        
        return samples
    
    def _sample_negatives(self, positive_item: int, exclude_items: set) -> List[int]:
        """Sample negative items"""
        negatives = []
        while len(negatives) < self.num_negatives:
            neg = np.random.randint(0, self.num_items)
            if neg != positive_item and neg not in exclude_items:
                negatives.append(neg)
        return negatives
    
    def _pad_sequence(self, seq: List[int]) -> Tuple[List[int], List[int]]:
        """Pad sequence to max_seq_length, return (padded_seq, attention_mask)"""
        seq = seq[-self.max_seq_length:]  # Truncate if too long
        
        pad_len = self.max_seq_length - len(seq)
        padded = [0] * pad_len + seq  # Left padding
        mask = [0] * pad_len + [1] * len(seq)
        
        return padded, mask
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int):
        if self.mode == 'train':
            user_idx, input_seq, target = self.samples[idx]
            exclude = set(input_seq)
            negatives = self._sample_negatives(target, exclude)
            
            padded_seq, mask = self._pad_sequence(input_seq)
            
            return {
                'user_idx': user_idx,
                'input_seq': torch.LongTensor(padded_seq),
                'attention_mask': torch.LongTensor(mask),
                'positive': target,
                'negatives': torch.LongTensor(negatives)
            }
        else:
            # Val/Test mode
            user_idx = self.samples[idx]
            input_seq, target = self.user_sequences[user_idx]
            
            padded_seq, mask = self._pad_sequence(input_seq)
            
            return {
                'user_idx': user_idx,
                'input_seq': torch.LongTensor(padded_seq),
                'attention_mask': torch.LongTensor(mask),
                'target': target
            }


def create_dataloaders(
    data_processor: TikiDataProcessor,
    batch_size: int = 128,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders"""
    
    train_data, val_data, test_data = data_processor.create_train_val_test_split()
    
    train_dataset = SASRecDataset(
        train_data, 
        data_processor.num_items,
        mode='train'
    )
    
    val_dataset = SASRecDataset(
        val_data,
        data_processor.num_items,
        mode='val'
    )
    
    test_dataset = SASRecDataset(
        test_data,
        data_processor.num_items,
        mode='test'
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test data loading
    processor = TikiDataProcessor()
    processor.load_raw_data()
    
    print(f"\nSample item info:")
    sample_item = list(processor.item_info.keys())[0]
    print(f"   Item {sample_item}: {processor.item_info[sample_item]}")
    
    print(f"\nSample user sequence:")
    sample_user = list(processor.user_sequences.keys())[0]
    print(f"   User {sample_user}: {processor.user_sequences[sample_user][:5]}...")
