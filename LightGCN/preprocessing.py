"""
Data Preprocessing for LightGCN
Builds user-item bipartite graph
"""

import json
import pickle
import re
import html
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
import numpy as np
import scipy.sparse as sp
import torch

from config import config


class ProcessedData:
    """Container for LightGCN data"""
    
    def __init__(self, **kwargs):
        # Mappings
        self.item2idx = kwargs.get('item2idx', {})
        self.idx2item = kwargs.get('idx2item', {})
        self.item_info = kwargs.get('item_info', {})
        self.num_items = kwargs.get('num_items', 0)
        
        self.user2idx = kwargs.get('user2idx', {})
        self.idx2user = kwargs.get('idx2user', {})
        self.num_users = kwargs.get('num_users', 0)
        
        # Interactions
        self.train_interactions = kwargs.get('train_interactions', [])
        self.val_interactions = kwargs.get('val_interactions', [])
        self.test_interactions = kwargs.get('test_interactions', [])
        
        # User history (for evaluation)
        self.user_pos_items = kwargs.get('user_pos_items', {})
        
        # Graph
        self.adj_matrix = kwargs.get('adj_matrix', None)
        
        self.stats = kwargs.get('stats', {})
    
    def to_dict(self):
        return {
            'item2idx': self.item2idx,
            'idx2item': self.idx2item,
            'item_info': self.item_info,
            'num_items': self.num_items,
            'user2idx': self.user2idx,
            'idx2user': self.idx2user,
            'num_users': self.num_users,
            'train_interactions': self.train_interactions,
            'val_interactions': self.val_interactions,
            'test_interactions': self.test_interactions,
            'user_pos_items': self.user_pos_items,
            'adj_matrix': self.adj_matrix,
            'stats': self.stats,
        }
    
    @classmethod
    def from_dict(cls, d):
        return cls(**d)


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_and_process_data() -> ProcessedData:
    """Load raw data and build graph"""
    cfg = config.data
    filepath = Path(cfg.raw_data_path)
    
    print("=" * 60)
    print("LightGCN Data Preprocessing")
    print("=" * 60)
    
    item_info_raw = {}
    user_items = defaultdict(list)
    seen_products = set()
    
    skipped_errors = 0
    skipped_duplicates = 0
    
    print("\n📖 Reading JSONL...")
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading"):
            try:
                data = json.loads(line.strip())
                product_id = data.get('product_id')
                if not product_id:
                    continue
                
                product_detail = data.get('product_detail', {})
                detail_str = str(product_detail).lower()
                
                if 'error' in detail_str or 'redirect' in detail_str:
                    skipped_errors += 1
                    continue
                
                if not product_detail.get('name'):
                    skipped_errors += 1
                    continue
                
                if product_id in seen_products:
                    skipped_duplicates += 1
                    continue
                seen_products.add(product_id)
                
                item_info_raw[product_id] = {
                    'name': clean_text(product_detail.get('name', '')),
                    'category': data.get('category', ''),
                    'price': product_detail.get('price', 0),
                }
                
                reviews = data.get('reviews', [])
                for review in reviews:
                    user_id = review.get('customer_id')
                    rating = review.get('rating', 0)
                    timestamp = review.get('created_at', 0)
                    
                    if user_id and rating > 0:
                        user_items[user_id].append({
                            'item_id': product_id,
                            'timestamp': timestamp,
                            'rating': rating
                        })
            except:
                skipped_errors += 1
    
    print(f"   Valid products: {len(seen_products):,}")
    print(f"   Skipped: {skipped_errors:,} errors, {skipped_duplicates:,} duplicates")
    
    # Filter items
    print("\nFiltering items...")
    item_counts = defaultdict(int)
    for interactions in user_items.values():
        for inter in interactions:
            item_counts[inter['item_id']] += 1
    
    valid_items = {
        item_id for item_id, count in item_counts.items()
        if count >= cfg.min_item_count and item_id in item_info_raw
    }
    print(f"   Items >= {cfg.min_item_count} interactions: {len(valid_items):,}")
    
    # Build item mappings
    item2idx, idx2item, item_info = {}, {}, {}
    for idx, item_id in enumerate(sorted(valid_items)):
        item2idx[item_id] = idx
        idx2item[idx] = item_id
        item_info[item_id] = item_info_raw[item_id]
    num_items = len(item2idx)
    
    # Filter users and build mappings
    print("\nBuilding user mappings...")
    user2idx, idx2user = {}, {}
    user_pos_items = {}
    all_interactions = []
    
    user_count = 0
    for user_id, interactions in user_items.items():
        valid = [i for i in interactions if i['item_id'] in item2idx]
        if len(valid) < cfg.min_user_count:
            continue
        
        user2idx[user_id] = user_count
        idx2user[user_count] = user_id
        
        # Sort by timestamp
        valid.sort(key=lambda x: x['timestamp'])
        
        item_indices = [item2idx[i['item_id']] for i in valid]
        user_pos_items[user_count] = set(item_indices)
        
        for item_idx in item_indices:
            all_interactions.append((user_count, item_idx))
        
        user_count += 1
    
    num_users = len(user2idx)
    print(f"   Users: {num_users:,}")
    print(f"   Interactions: {len(all_interactions):,}")
    
    # Train/Val/Test split (80/10/10 random)
    print("\nSplitting data...")
    np.random.seed(42)
    np.random.shuffle(all_interactions)
    
    n = len(all_interactions)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    
    train_interactions = all_interactions[:n_train]
    val_interactions = all_interactions[n_train:n_train+n_val]
    test_interactions = all_interactions[n_train+n_val:]
    
    print(f"   Train: {len(train_interactions):,}")
    print(f"   Val: {len(val_interactions):,}")
    print(f"   Test: {len(test_interactions):,}")
    
    # Build adjacency matrix
    print("\nBuilding adjacency matrix...")
    adj_matrix = build_adjacency_matrix(train_interactions, num_users, num_items)
    
    stats = {
        'num_users': num_users,
        'num_items': num_items,
        'num_interactions': len(all_interactions),
        'density': len(all_interactions) / (num_users * num_items) * 100
    }
    print(f"   Density: {stats['density']:.4f}%")
    
    return ProcessedData(
        item2idx=item2idx, idx2item=idx2item,
        item_info=item_info, num_items=num_items,
        user2idx=user2idx, idx2user=idx2user,
        num_users=num_users,
        train_interactions=train_interactions,
        val_interactions=val_interactions,
        test_interactions=test_interactions,
        user_pos_items=user_pos_items,
        adj_matrix=adj_matrix,
        stats=stats
    )


def build_adjacency_matrix(interactions: List[Tuple], num_users: int, num_items: int) -> sp.csr_matrix:
    """
    Build normalized adjacency matrix for LightGCN
    
    A = [[0, R], [R^T, 0]]  (user-item bipartite)
    Normalized: D^{-0.5} A D^{-0.5}
    """
    n_nodes = num_users + num_items
    
    # Build sparse matrix
    rows, cols = [], []
    for user_idx, item_idx in interactions:
        # User -> Item
        rows.append(user_idx)
        cols.append(num_users + item_idx)
        # Item -> User (symmetric)
        rows.append(num_users + item_idx)
        cols.append(user_idx)
    
    data = np.ones(len(rows))
    adj = sp.coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
    adj = adj.tocsr()
    
    # Normalize: D^{-0.5} A D^{-0.5}
    degrees = np.array(adj.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(degrees + 1e-10, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    norm_adj = D_inv_sqrt @ adj @ D_inv_sqrt
    
    return norm_adj.tocsr()


def sparse_to_torch(adj: sp.csr_matrix) -> torch.sparse.FloatTensor:
    """Convert scipy sparse to torch sparse tensor"""
    coo = adj.tocoo()
    indices = torch.LongTensor([coo.row, coo.col])
    values = torch.FloatTensor(coo.data)
    return torch.sparse_coo_tensor(indices, values, coo.shape)


def save_processed_data(data: ProcessedData, filepath: str = None) -> str:
    filepath = filepath or config.data.cache_file
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving to {filepath}...")
    with open(filepath, 'wb') as f:
        pickle.dump(data.to_dict(), f)
    
    print(f"   Size: {filepath.stat().st_size / (1024*1024):.1f} MB")
    return str(filepath)


def load_processed_data(filepath: str = None) -> ProcessedData:
    filepath = filepath or config.data.cache_file
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Not found: {filepath}\nRun: python preprocessing.py")
    
    print(f"Loading from {filepath}...")
    with open(filepath, 'rb') as f:
        d = pickle.load(f)
    
    data = ProcessedData.from_dict(d) if isinstance(d, dict) else d
    print(f"   Users: {data.num_users:,} | Items: {data.num_items:,}")
    return data


def is_cache_valid(filepath: str = None) -> bool:
    filepath = filepath or config.data.cache_file
    return Path(filepath).exists()


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    
    if is_cache_valid() and not args.force:
        print(f"Cache exists: {config.data.cache_file}")
        data = load_processed_data()
    else:
        data = load_and_process_data()
        save_processed_data(data)
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
