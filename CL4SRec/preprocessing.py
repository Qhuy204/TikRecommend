"""
Data Preprocessing for SASRec-CE
Run this once to clean data and save to pickle cache
"""

import json
import pickle
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import html

from config import config, DataConfig


class ProcessedData:
    """Container for processed data"""
    def __init__(self, **kwargs):
        # Items
        self.item2idx = kwargs.get('item2idx', {})
        self.idx2item = kwargs.get('idx2item', {})
        self.item_info = kwargs.get('item_info', {})
        self.num_items = kwargs.get('num_items', 0)
        
        # Users
        self.user2idx = kwargs.get('user2idx', {})
        self.idx2user = kwargs.get('idx2user', {})
        self.num_users = kwargs.get('num_users', 0)
        
        # Sequences
        self.user_sequences = kwargs.get('user_sequences', {})
        
        # Statistics
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
            'user_sequences': self.user_sequences,
            'stats': self.stats,
        }
    
    @classmethod
    def from_dict(cls, d):
        return cls(**d)


def clean_text(text: str) -> str:
    """Clean text content"""
    if not text:
        return ""
    
    # Decode HTML entities
    text = html.unescape(text)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    
    # Remove special characters but keep Vietnamese
    text = re.sub(r'[^\w\s\u00C0-\u024F\u1E00-\u1EFF]', ' ', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def is_valid_product(product_detail: dict) -> bool:
    """Check if product is valid"""
    if not product_detail:
        return False
    
    # Must have name
    name = product_detail.get('name', '')
    if not name or len(name) < 3:
        return False
    
    # Check for error/redirect
    detail_str = str(product_detail).lower()
    if 'error' in detail_str or 'redirect' in detail_str:
        return False
    
    return True


def load_and_process_data(data_config: DataConfig = None) -> ProcessedData:
    """
    Load raw JSONL and process into structured format
    
    Returns:
        ProcessedData object with all processed data
    """
    cfg = data_config or config.data
    filepath = Path(cfg.raw_data_path)
    
    print("=" * 60)
    print("Data Preprocessing")
    print("=" * 60)
    print(f"   Source: {filepath}")
    
    # Temporary storage
    item_info_raw: Dict[int, dict] = {}
    user_interactions: Dict[int, List[dict]] = defaultdict(list)
    
    # Counters
    total_lines = 0
    skipped_errors = 0
    skipped_duplicates = 0
    seen_product_ids = set()
    
    # Read JSONL
    print("\n📖 Reading JSONL...")
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading"):
            total_lines += 1
            try:
                data = json.loads(line.strip())
                
                product_id = data.get('product_id')
                if not product_id:
                    continue
                
                product_detail = data.get('product_detail', {})
                
                # Validate product
                if not is_valid_product(product_detail):
                    skipped_errors += 1
                    continue
                
                # Skip duplicates
                if product_id in seen_product_ids:
                    skipped_duplicates += 1
                    continue
                seen_product_ids.add(product_id)
                
                # Extract product info
                category = data.get('category', 'Unknown')
                item_info_raw[product_id] = {
                    'name': clean_text(product_detail.get('name', '')),
                    'description': clean_text(product_detail.get('short_description', '')),
                    'category': category,
                    'price': product_detail.get('price', 0),
                }
                
                # Extract reviews
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
                skipped_errors += 1
            except Exception:
                skipped_errors += 1
    
    print(f"   Total lines: {total_lines:,}")
    print(f"   Valid products: {len(seen_product_ids):,}")
    print(f"   Skipped errors: {skipped_errors:,}")
    print(f"   Skipped duplicates: {skipped_duplicates:,}")
    
    # Filter items by minimum count
    print("\nFiltering items...")
    item_counts = defaultdict(int)
    for interactions in user_interactions.values():
        for inter in interactions:
            item_counts[inter['item_id']] += 1
    
    valid_items = {
        item_id for item_id, count in item_counts.items()
        if count >= cfg.min_item_count and item_id in item_info_raw
    }
    print(f"   Items with >= {cfg.min_item_count} interactions: {len(valid_items):,}")
    
    # Build item mapping
    print("\nBuilding ID mappings...")
    item2idx: Dict[int, int] = {}
    idx2item: Dict[int, int] = {}
    item_info: Dict[int, dict] = {}
    
    for idx, item_id in enumerate(sorted(valid_items)):
        item2idx[item_id] = idx
        idx2item[idx] = item_id
        item_info[item_id] = item_info_raw[item_id]
    
    num_items = len(item2idx)
    print(f"   Total items: {num_items:,}")
    
    # Filter and build user sequences
    print("\nBuilding user sequences...")
    user2idx: Dict[int, int] = {}
    idx2user: Dict[int, int] = {}
    user_sequences: Dict[int, List[Tuple[int, int, int]]] = {}
    
    valid_user_count = 0
    total_interactions = 0
    
    for user_id, interactions in user_interactions.items():
        # Filter to valid items
        valid_interactions = [
            inter for inter in interactions
            if inter['item_id'] in item2idx
        ]
        
        # Check minimum length
        if len(valid_interactions) < cfg.min_seq_length:
            continue
        
        # Sort by timestamp
        valid_interactions.sort(key=lambda x: x['timestamp'])
        
        # Truncate to max length
        valid_interactions = valid_interactions[-cfg.max_seq_length:]
        
        # Build sequence
        user2idx[user_id] = valid_user_count
        idx2user[valid_user_count] = user_id
        
        user_sequences[valid_user_count] = [
            (item2idx[inter['item_id']], inter['timestamp'], inter['rating'])
            for inter in valid_interactions
        ]
        
        total_interactions += len(valid_interactions)
        valid_user_count += 1
    
    num_users = len(user2idx)
    
    print(f"   Total users: {num_users:,}")
    print(f"   Total interactions: {total_interactions:,}")
    print(f"   Avg interactions/user: {total_interactions/num_users:.1f}")
    
    # Build statistics
    stats = {
        'num_items': num_items,
        'num_users': num_users,
        'total_interactions': total_interactions,
        'avg_seq_length': total_interactions / num_users if num_users > 0 else 0,
        'skipped_errors': skipped_errors,
        'skipped_duplicates': skipped_duplicates,
    }
    
    # Create ProcessedData
    processed = ProcessedData(
        item2idx=item2idx,
        idx2item=idx2item,
        item_info=item_info,
        num_items=num_items,
        user2idx=user2idx,
        idx2user=idx2user,
        num_users=num_users,
        user_sequences=user_sequences,
        stats=stats,
    )
    
    print("\nPreprocessing complete!")
    
    return processed


def save_processed_data(data: ProcessedData, filepath: str = None) -> str:
    """Save processed data to pickle file (as dict for compatibility)"""
    filepath = filepath or config.data.cache_file
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving to {filepath}...")
    with open(filepath, 'wb') as f:
        # Save as dict to avoid pickle compatibility issues
        pickle.dump(data.to_dict(), f)
    
    file_size = filepath.stat().st_size / (1024 * 1024)
    print(f"   File size: {file_size:.1f} MB")
    
    return str(filepath)


def load_processed_data(filepath: str = None) -> ProcessedData:
    """Load processed data from pickle file"""
    filepath = filepath or config.data.cache_file
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(
            f"Processed data not found at {filepath}\n"
            "Run: python preprocessing.py"
        )
    
    print(f"Loading cached data from {filepath}...")
    with open(filepath, 'rb') as f:
        d = pickle.load(f)
    
    # Convert dict back to ProcessedData
    if isinstance(d, dict):
        data = ProcessedData.from_dict(d)
    else:
        data = d  # Already ProcessedData (old format)
    
    print(f"   Items: {data.num_items:,}")
    print(f"   Users: {data.num_users:,}")
    
    return data


def is_cache_valid(filepath: str = None) -> bool:
    """Check if cache file exists"""
    filepath = filepath or config.data.cache_file
    return Path(filepath).exists()


def main():
    """Main preprocessing script"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess data for SASRec-CE')
    parser.add_argument('--force', action='store_true', help='Force reprocessing')
    parser.add_argument('--output', type=str, default=None, help='Output path')
    args = parser.parse_args()
    
    output_path = args.output or config.data.cache_file
    
    # Check if already processed
    if is_cache_valid(output_path) and not args.force:
        print(f"Cache already exists: {output_path}")
        print("   Use --force to reprocess")
        
        # Load and show stats
        data = load_processed_data(output_path)
        print(f"\nCached data stats:")
        for key, value in data.stats.items():
            print(f"   {key}: {value:,}" if isinstance(value, int) else f"   {key}: {value:.2f}")
        return
    
    # Process data
    data = load_and_process_data()
    
    # Save to cache
    save_processed_data(data, output_path)
    
    print("\n" + "=" * 60)
    print("Done! Cached data ready for training.")
    print("=" * 60)


if __name__ == "__main__":
    main()
