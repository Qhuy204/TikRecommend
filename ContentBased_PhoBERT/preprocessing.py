"""
Data Preprocessing for PhoBERT Content-Based Recommender
Handles: HTML tags, duplicates, errors, text cleaning, PhoBERT encoding
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
import torch
from torch.nn import functional as F

from config import config


class ProcessedData:
    """Container for processed data"""
    def __init__(self, **kwargs):
        # Items
        self.item2idx: Dict[int, int] = kwargs.get('item2idx', {})
        self.idx2item: Dict[int, int] = kwargs.get('idx2item', {})
        self.item_info: Dict[int, dict] = kwargs.get('item_info', {})
        self.item_texts: Dict[int, str] = kwargs.get('item_texts', {})
        self.num_items: int = kwargs.get('num_items', 0)
        
        # Users
        self.user2idx: Dict[int, int] = kwargs.get('user2idx', {})
        self.idx2user: Dict[int, int] = kwargs.get('idx2user', {})
        self.num_users: int = kwargs.get('num_users', 0)
        
        # User-item interactions (for recommend_user)
        self.user_pos_items: Dict[int, set] = kwargs.get('user_pos_items', {})
        
        # Statistics
        self.stats: dict = kwargs.get('stats', {})
    
    def to_dict(self):
        return {
            'item2idx': self.item2idx,
            'idx2item': self.idx2item,
            'item_info': self.item_info,
            'item_texts': self.item_texts,
            'num_items': self.num_items,
            'user2idx': self.user2idx,
            'idx2user': self.idx2user,
            'num_users': self.num_users,
            'user_pos_items': {k: list(v) for k, v in self.user_pos_items.items()},
            'stats': self.stats,
        }
    
    @classmethod
    def from_dict(cls, d):
        if 'user_pos_items' in d:
            d['user_pos_items'] = {k: set(v) for k, v in d['user_pos_items'].items()}
        return cls(**d)


def clean_text(text: str) -> str:
    """Basic text cleaning for name field"""
    if not text:
        return ""
    
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'[^\w\s\u00C0-\u024F\u1E00-\u1EFF]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def clean_description(text: str) -> str:
    """
    Deep clean HTML description field:
    - Remove all HTML tags (img, p, h3, ol, li, etc.)
    - Remove all URLs (http, https, tikicdn, tracking links)
    - Remove style attributes and inline styles
    - Remove image references and data URLs
    - Keep only meaningful Vietnamese text
    - Normalize whitespace
    """
    if not text:
        return ""
    
    # Step 1: Decode HTML entities
    text = html.unescape(text)
    
    # Step 2: Remove script and style blocks entirely
    text = re.sub(r'<script[^>]*>.*?</script>', ' ', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<style[^>]*>.*?</style>', ' ', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Step 3: Remove image tags completely
    text = re.sub(r'<img[^>]*>', ' ', text, flags=re.IGNORECASE)
    
    # Step 4: Remove all URLs
    text = re.sub(r'https?://[^\s<>"\']+', ' ', text)
    text = re.sub(r'data:[^\s<>"\']+', ' ', text)
    text = re.sub(r'www\.[^\s<>"\']+', ' ', text)
    
    # Step 5: Remove all remaining HTML tags but keep content
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Step 6: Remove style artifacts
    text = re.sub(r'font-weight:\s*\d+;?', ' ', text)
    text = re.sub(r'aria-level="\d+"', ' ', text)
    text = re.sub(r'style="[^"]*"', ' ', text)
    
    # Step 7: Remove special characters but keep Vietnamese
    text = re.sub(r'[^\w\s\u00C0-\u024F\u1E00-\u1EFF.,;:!?\-–]', ' ', text)
    
    # Step 8: Remove standalone special chars and asterisks
    text = re.sub(r'\*+', ' ', text)
    text = re.sub(r'\s+[–\-]\s+', ' ', text)
    
    # Step 9: Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Step 10: Remove very short results
    if len(text) < 20:
        return ""
    
    return text


def is_valid_product(product_detail: dict) -> bool:
    """Check if product is valid"""
    if not product_detail:
        return False
    
    name = product_detail.get('name', '')
    if not name or len(name) < 3:
        return False
    
    detail_str = str(product_detail).lower()
    if 'error' in detail_str or 'redirect' in detail_str:
        return False
    
    return True


def load_and_process_data() -> ProcessedData:
    """Load raw JSONL and process data (same as TF-IDF)"""
    filepath = Path(config.data.raw_data_path)
    
    print("=" * 60)
    print("PhoBERT Content-Based Preprocessing")
    print("=" * 60)
    print(f"   Source: {filepath}")
    
    # Storage
    item_info_raw: Dict[int, dict] = {}
    item_texts_raw: Dict[int, str] = {}
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
                    skipped_errors += 1
                    continue
                
                product_detail = data.get('product_detail', {})
                
                if not is_valid_product(product_detail):
                    skipped_errors += 1
                    continue
                
                if product_id in seen_product_ids:
                    skipped_duplicates += 1
                    continue
                seen_product_ids.add(product_id)
                
                name = clean_text(product_detail.get('name', ''))
                description = clean_description(product_detail.get('description', ''))
                category = data.get('category', 'Unknown')
                price = product_detail.get('price', 0)
                
                # Combined text for PhoBERT: name + cleaned description
                combined_text = f"{name} {description}"[:config.data.max_text_length]
                
                item_info_raw[product_id] = {
                    'name': name,
                    'description': description[:500],  # Store truncated for display
                    'category': category,
                    'price': price,
                }
                item_texts_raw[product_id] = combined_text
                
                reviews = data.get('reviews', [])
                for review in reviews:
                    user_id = review.get('customer_id')
                    rating = review.get('rating', 0)
                    
                    if user_id and rating > 0:
                        user_interactions[user_id].append({
                            'item_id': product_id,
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
    
    # Use ALL items (no filtering by interaction count for content-based)
    print("\nBuilding ID mappings (ALL items)...")
    item2idx: Dict[int, int] = {}
    idx2item: Dict[int, int] = {}
    item_info: Dict[int, dict] = {}
    item_texts: Dict[int, str] = {}
    
    for idx, item_id in enumerate(sorted(item_info_raw.keys())):
        item2idx[item_id] = idx
        idx2item[idx] = item_id
        item_info[item_id] = item_info_raw[item_id]
        item_texts[item_id] = item_texts_raw[item_id]
    
    num_items = len(item2idx)
    print(f"   Total items: {num_items:,}")
    
    # Build user data
    print("\nBuilding user data...")
    user2idx: Dict[int, int] = {}
    idx2user: Dict[int, int] = {}
    user_pos_items: Dict[int, set] = {}
    
    valid_user_count = 0
    total_interactions = 0
    
    for user_id, interactions in user_interactions.items():
        pos_items = set()
        for inter in interactions:
            if inter['item_id'] in item2idx and inter['rating'] >= 4:
                pos_items.add(item2idx[inter['item_id']])
        
        if len(pos_items) >= config.data.min_user_count:
            user2idx[user_id] = valid_user_count
            idx2user[valid_user_count] = user_id
            user_pos_items[valid_user_count] = pos_items
            total_interactions += len(pos_items)
            valid_user_count += 1
    
    num_users = len(user2idx)
    print(f"   Total users: {num_users:,}")
    print(f"   Total positive interactions: {total_interactions:,}")
    
    stats = {
        'num_items': num_items,
        'num_users': num_users,
        'total_interactions': total_interactions,
        'skipped_errors': skipped_errors,
        'skipped_duplicates': skipped_duplicates,
    }
    
    return ProcessedData(
        item2idx=item2idx,
        idx2item=idx2item,
        item_info=item_info,
        item_texts=item_texts,
        num_items=num_items,
        user2idx=user2idx,
        idx2user=idx2user,
        num_users=num_users,
        user_pos_items=user_pos_items,
        stats=stats,
    )


def encode_with_phobert(data: ProcessedData) -> torch.Tensor:
    """
    Encode item texts with PhoBERT
    Returns normalized embeddings tensor
    """
    from transformers import AutoTokenizer, AutoModel
    
    print("\n🤖 Loading PhoBERT...")
    tokenizer = AutoTokenizer.from_pretrained(config.phobert.model_name)
    # Use safetensors format to avoid torch.load security issues
    model = AutoModel.from_pretrained(
        config.phobert.model_name,
        use_safetensors=True
    )
    model = model.to(config.phobert.device)
    model.eval()
    
    print(f"   Model: {config.phobert.model_name}")
    print(f"   Device: {config.phobert.device}")
    
    # Prepare texts in order
    texts = []
    for idx in range(data.num_items):
        item_id = data.idx2item[idx]
        texts.append(data.item_texts[item_id])
    
    # Encode in batches
    print(f"\nEncoding {len(texts):,} items...")
    embeddings = []
    batch_size = config.phobert.batch_size
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenize
        inputs = tokenizer(
            batch_texts,
            max_length=config.phobert.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(config.phobert.device)
        
        # Encode
        with torch.no_grad():
            outputs = model(**inputs)
            # Use [CLS] token embedding (first token)
            batch_emb = outputs.last_hidden_state[:, 0, :]
            embeddings.append(batch_emb.cpu())
    
    # Concatenate all embeddings
    all_embeddings = torch.cat(embeddings, dim=0)
    
    # Normalize for fast cosine similarity
    all_embeddings = F.normalize(all_embeddings, dim=1)
    
    print(f"   Embeddings shape: {all_embeddings.shape}")
    print(f"   Normalized: L2 norm = 1.0")
    
    return all_embeddings


def save_processed_data(data: ProcessedData, embeddings: torch.Tensor):
    """Save processed data and embeddings"""
    cache_path = Path(config.data.cache_file)
    emb_path = Path(config.data.embeddings_file)
    
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving to {cache_path}...")
    with open(cache_path, 'wb') as f:
        pickle.dump(data.to_dict(), f)
    
    print(f"Saving embeddings to {emb_path}...")
    # Save as float16 to reduce size
    torch.save(embeddings.half(), emb_path)
    
    print(f"   Data size: {cache_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"   Embeddings size: {emb_path.stat().st_size / 1024 / 1024:.1f} MB")


def load_processed_data() -> Tuple[ProcessedData, torch.Tensor]:
    """Load processed data and embeddings"""
    cache_path = Path(config.data.cache_file)
    emb_path = Path(config.data.embeddings_file)
    
    if not cache_path.exists() or not emb_path.exists():
        raise FileNotFoundError(
            f"Processed data not found.\n"
            "Run: python preprocessing.py"
        )
    
    print(f"Loading from {cache_path}...")
    with open(cache_path, 'rb') as f:
        d = pickle.load(f)
    data = ProcessedData.from_dict(d)
    
    print(f"Loading embeddings from {emb_path}...")
    embeddings = torch.load(emb_path, weights_only=True).float()  # Convert back to float32
    
    print(f"   Items: {data.num_items:,}")
    print(f"   Users: {data.num_users:,}")
    print(f"   Embeddings shape: {embeddings.shape}")
    
    return data, embeddings


def main():
    """Main preprocessing script"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess data for PhoBERT')
    parser.add_argument('--force', action='store_true', help='Force reprocessing')
    args = parser.parse_args()
    
    cache_path = Path(config.data.cache_file)
    emb_path = Path(config.data.embeddings_file)
    
    if cache_path.exists() and emb_path.exists() and not args.force:
        print(f"Cache exists: {cache_path}")
        print("   Use --force to reprocess")
        data, embeddings = load_processed_data()
        print(f"\nStats: {data.stats}")
        return
    
    # Process data
    data = load_and_process_data()
    
    # Encode with PhoBERT
    embeddings = encode_with_phobert(data)
    
    # Save
    save_processed_data(data, embeddings)
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
