"""
PyTorch Datasets for CL4SRec
Includes sequence augmentation for contrastive learning
"""

import random
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from config import config
from preprocessing import ProcessedData, load_processed_data


class SequenceAugmentor:
    """Sequence augmentation for contrastive learning"""
    
    def __init__(self, crop_ratio: float = 0.6, mask_ratio: float = 0.2):
        self.crop_ratio = crop_ratio
        self.mask_ratio = mask_ratio
    
    def crop(self, seq: List[int]) -> List[int]:
        if len(seq) <= 2:
            return seq.copy()
        crop_len = max(2, int(len(seq) * self.crop_ratio))
        start = random.randint(0, len(seq) - crop_len)
        return seq[start:start + crop_len]
    
    def mask(self, seq: List[int], mask_token: int = 0) -> List[int]:
        if len(seq) <= 1:
            return seq.copy()
        masked = seq.copy()
        num_mask = max(1, int(len(seq) * self.mask_ratio))
        positions = random.sample(range(len(seq)), min(num_mask, len(seq) - 1))
        for pos in positions:
            masked[pos] = mask_token
        return masked
    
    def reorder(self, seq: List[int]) -> List[int]:
        if len(seq) <= 2:
            return seq.copy()
        reordered = seq.copy()
        seg_len = max(2, int(len(seq) * 0.2))
        start = random.randint(0, max(0, len(seq) - seg_len))
        segment = reordered[start:start + seg_len]
        random.shuffle(segment)
        reordered[start:start + seg_len] = segment
        return reordered
    
    def augment(self, seq: List[int]) -> List[int]:
        aug_type = random.choice(['crop', 'mask', 'reorder'])
        if aug_type == 'crop':
            return self.crop(seq)
        elif aug_type == 'mask':
            return self.mask(seq)
        return self.reorder(seq)


class CL4SRecDataset(Dataset):
    """
    Dataset for CL4SRec with sequence augmentation
    
    Returns:
    - input_seq, attention_mask: Original sequence
    - positive_item: Next item to predict
    - negative_items: Random negative samples
    - aug_seq1, aug_mask1: First augmented view
    - aug_seq2, aug_mask2: Second augmented view
    """
    
    def __init__(
        self,
        user_sequences: Dict[int, List[int]],
        num_items: int,
        max_seq_length: int = 50,
        num_negatives: int = 4,
        mode: str = 'train',
        use_augmentation: bool = True
    ):
        self.user_sequences = user_sequences
        self.num_items = num_items
        self.max_seq_length = max_seq_length
        self.num_negatives = num_negatives
        self.mode = mode
        self.use_augmentation = use_augmentation and mode == 'train'
        
        self.augmentor = SequenceAugmentor()
        self.users = list(user_sequences.keys())
        
        if mode == 'train':
            self.samples = self._create_samples()
        else:
            self.samples = self.users
    
    def _create_samples(self) -> List[Tuple[int, List[int], int]]:
        """Create (user, input_seq, target) samples"""
        samples = []
        for user_idx, items in self.user_sequences.items():
            if len(items) < 2:
                continue
            # Each item (except first) can be a target
            for i in range(1, len(items)):
                input_seq = items[:i]
                target = items[i]
                samples.append((user_idx, input_seq, target))
        return samples
    
    def _pad_sequence(self, seq: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Left-pad sequence"""
        seq = seq[-self.max_seq_length:]
        pad_len = self.max_seq_length - len(seq)
        padded = [0] * pad_len + seq
        mask = [0] * pad_len + [1] * len(seq)
        return np.array(padded, dtype=np.int64), np.array(mask, dtype=np.int64)
    
    def _sample_negatives(self, positive: int) -> np.ndarray:
        """Sample random negative items"""
        negatives = []
        while len(negatives) < self.num_negatives:
            neg = random.randint(0, self.num_items - 1)
            if neg != positive:
                negatives.append(neg)
        return np.array(negatives, dtype=np.int64)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.mode == 'train':
            user_idx, input_seq, target = self.samples[idx]
            
            # Pad main sequence
            seq_padded, seq_mask = self._pad_sequence(input_seq)
            
            # Sample negatives
            negatives = self._sample_negatives(target)
            
            result = {
                'user_idx': user_idx,
                'input_seq': torch.LongTensor(seq_padded),
                'attention_mask': torch.LongTensor(seq_mask),
                'positive_item': target,
                'negative_items': torch.LongTensor(negatives),
            }
            
            # Augmented views for contrastive learning
            if self.use_augmentation:
                aug1 = self.augmentor.crop(input_seq)
                aug2 = self.augmentor.augment(input_seq)
                
                aug1_padded, aug1_mask = self._pad_sequence(aug1)
                aug2_padded, aug2_mask = self._pad_sequence(aug2)
                
                result['aug_seq1'] = torch.LongTensor(aug1_padded)
                result['aug_mask1'] = torch.LongTensor(aug1_mask)
                result['aug_seq2'] = torch.LongTensor(aug2_padded)
                result['aug_mask2'] = torch.LongTensor(aug2_mask)
            
            return result
        else:
            # Validation/Test
            user_idx = self.samples[idx]
            items = self.user_sequences[user_idx]
            
            if isinstance(items, tuple):
                input_seq, target = items
            else:
                input_seq = items[:-1]
                target = items[-1]
            
            seq_padded, seq_mask = self._pad_sequence(input_seq)
            
            return {
                'user_idx': user_idx,
                'input_seq': torch.LongTensor(seq_padded),
                'attention_mask': torch.LongTensor(seq_mask),
                'target': target,
            }


def create_data_splits(processed_data: ProcessedData) -> Tuple[Dict, Dict, Dict]:
    """Leave-one-out split"""
    train_data, val_data, test_data = {}, {}, {}
    
    for user_idx, sequence in processed_data.user_sequences.items():
        items = [x[0] for x in sequence]
        
        if len(items) < 3:
            continue
        
        train_data[user_idx] = items[:-1]
        val_data[user_idx] = (items[:-2], items[-2])
        test_data[user_idx] = (items[:-1], items[-1])
    
    print(f"Data splits: Train {len(train_data):,} | Val {len(val_data):,} | Test {len(test_data):,}")
    return train_data, val_data, test_data


def create_dataloaders(
    processed_data: ProcessedData,
    batch_size: int = 128,
    num_workers: int = 4,
    use_augmentation: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create dataloaders for CL4SRec"""
    
    train_data, val_data, test_data = create_data_splits(processed_data)
    
    train_dataset = CL4SRecDataset(
        train_data, processed_data.num_items,
        max_seq_length=config.data.max_seq_length,
        mode='train', use_augmentation=use_augmentation
    )
    val_dataset = CL4SRecDataset(
        val_data, processed_data.num_items,
        max_seq_length=config.data.max_seq_length,
        mode='val'
    )
    test_dataset = CL4SRecDataset(
        test_data, processed_data.num_items,
        max_seq_length=config.data.max_seq_length,
        mode='test'
    )
    
    print(f"   Samples: Train {len(train_dataset):,} | Val {len(val_dataset):,} | Test {len(test_dataset):,}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    print("Testing CL4SRec dataset...")
    data = load_processed_data()
    train_loader, _, _ = create_dataloaders(data, batch_size=4)
    
    batch = next(iter(train_loader))
    print(f"Keys: {batch.keys()}")
    print(f"Input: {batch['input_seq'].shape}")
    if 'aug_seq1' in batch:
        print(f"Aug1: {batch['aug_seq1'].shape}")
    print("Dataset test passed!")
