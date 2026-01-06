"""
Intelligent E-commerce Product Recommendation System
Implementation of Two-Stage Funnel: Retrieval + Ranking
Dataset: TikDataset (Vietnamese E-commerce)

Complete Pipeline:
1. Data Cleaning (Remove errors, clean HTML)
2. Feature Extraction
3. Two-Tower Model Training (Retrieval)
4. MMoE Model Training (Ranking)
"""

import json
import os
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

# ============================================================================
# PART 1: DATA CLEANING & PREPROCESSING
# ============================================================================

class DataCleaner:
    """Làm sạch JSONL dataset - loại bỏ lỗi redirect và corrupt data"""
    
    @staticmethod
    def clean_jsonl_dataset(input_path: str, output_path: str):
        """
        Loại bỏ các dòng có lỗi redirect hoặc JSON không hợp lệ
        """
        file_size = os.path.getsize(input_path)
        count_total = 0
        count_removed = 0
        count_valid = 0

        print(f"\n{'='*60}")
        print(f"CLEANING DATASET: {input_path}")
        print(f"{'='*60}")
        
        with open(input_path, 'r', encoding='utf-8') as f_in, \
             open(output_path, 'w', encoding='utf-8') as f_out, \
             tqdm(total=file_size, unit='B', unit_scale=True, desc="Cleaning") as pbar:
            
            for line in f_in:
                count_total += 1
                pbar.update(len(line.encode('utf-8')))
                
                try:
                    data = json.loads(line)
                    
                    # Kiểm tra lỗi redirect
                    product_detail = data.get("product_detail", {})
                    if isinstance(product_detail, dict) and product_detail.get("error") == "redirect":
                        count_removed += 1
                        continue
                    
                    # Kiểm tra product_detail có hợp lệ
                    if not product_detail or not isinstance(product_detail, dict):
                        count_removed += 1
                        continue
                    
                    # Ghi dòng hợp lệ
                    f_out.write(line)
                    count_valid += 1
                    
                except json.JSONDecodeError:
                    count_removed += 1
                    continue

        print(f"\n{'='*60}")
        print(f"CLEANING RESULTS:")
        print(f"{'='*60}")
        print(f"Total lines:     {count_total:,}")
        print(f"Valid products:  {count_valid:,} ({count_valid/count_total*100:.2f}%)")
        print(f"Removed errors:  {count_removed:,} ({count_removed/count_total*100:.2f}%)")
        print(f"Output file:     {output_path}")
        print(f"{'='*60}\n")
        
        return count_valid, count_removed


class HTMLCleaner:
    """Làm sạch HTML và rich text từ product descriptions"""
    
    @staticmethod
    def remove_html_tags(text: str) -> str:
        """Sử dụng BeautifulSoup để loại bỏ tất cả các thẻ HTML"""
        if not text or not isinstance(text, str):
            return ""
        
        # Parse HTML
        soup = BeautifulSoup(text, 'html.parser')
        
        # Loại bỏ các thẻ không mong muốn
        for tag in soup(["img", "a", "script", "style", "iframe", "noscript"]):
            tag.decompose()
        
        # Lấy text thuần
        clean_text = soup.get_text(separator=' ', strip=True)
        return clean_text
    
    @staticmethod
    def remove_markdown_and_special_chars(text: str) -> str:
        """Loại bỏ Markdown, URL và ký tự đặc biệt"""
        if not text:
            return ""
        
        # Loại bỏ URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)
        
        # Loại bỏ Markdown
        text = re.sub(r'[*_#]+', ' ', text)
        
        # Loại bỏ ký tự đặc biệt dư thừa
        text = re.sub(r'[\t\n\r]+', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    @staticmethod
    def remove_unwanted_sections(text: str) -> str:
        """Loại bỏ các phần không cần thiết như FAQ, footer"""
        patterns_to_remove = [
            r"Câu hỏi thường gặp.*",
            r"Một số nội dung tìm kiếm.*",
            r"Giá sản phẩm trên Tiki.*",
            r"HSD.*ngày SX.*",
            r"Bên cạnh đó.*phí vận chuyển.*",
        ]
        
        for pattern in patterns_to_remove:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)
        
        return text
    
    @staticmethod
    def clean_description(text: str) -> str:
        """Pipeline đầy đủ để làm sạch description"""
        if not text or not isinstance(text, str):
            return ""
        
        # Step 1: Remove HTML tags
        text = HTMLCleaner.remove_html_tags(text)
        
        # Step 2: Remove unwanted sections
        text = HTMLCleaner.remove_unwanted_sections(text)
        
        # Step 3: Remove markdown and special chars
        text = HTMLCleaner.remove_markdown_and_special_chars(text)
        
        # Step 4: Remove URLs
        text = re.sub(r"http\S+|www\S+", " ", text)
        
        # Step 5: Remove special characters (giữ tiếng Việt)
        text = re.sub(r"[^\w\sÀ-ỹà-ỹ.,!?%€₫]", " ", text)
        text = re.sub(r"(\*{1,}|-+|•+|–+|…+|_+)", " ", text)
        text = re.sub(r"([.!?]){2,}", r"\1", text)
        
        # Step 6: Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        
        return text
    
    @staticmethod
    def batch_clean_descriptions(descriptions: List[str], batch_size: int = 100) -> List[str]:
        """Batch processing cho hiệu suất tốt hơn"""
        cleaned = []
        
        print(f"\nCleaning {len(descriptions)} descriptions...")
        for i in tqdm(range(0, len(descriptions), batch_size), desc="Batch cleaning"):
            batch = descriptions[i:i+batch_size]
            cleaned.extend([HTMLCleaner.clean_description(desc) for desc in batch])
        
        return cleaned


class TikDataPreprocessor:
    """Xử lý dữ liệu từ TikDataset đã được làm sạch"""
    
    def __init__(self, data_dir: str = "TikDataset"):
        self.data_dir = Path(data_dir)
        self.html_cleaner = HTMLCleaner()
    
    def clean_html(self, html_text: str) -> str:
        """Wrapper cho HTMLCleaner"""
        return self.html_cleaner.clean_description(html_text)
    
    def flatten_specifications(self, specs: List[Dict]) -> str:
        """Chuyển specifications thành text"""
        if not specs:
            return ""
        texts = []
        for spec_group in specs:
            if 'attributes' in spec_group:
                for attr in spec_group['attributes']:
                    name = attr.get('name', '')
                    value = self.clean_html(str(attr.get('value', '')))
                    if name and value:
                        texts.append(f"{name} {value}")
        return ". ".join(texts)
    
    def extract_item_text(self, product: Dict) -> str:
        """Trích xuất và kết hợp tất cả text features của sản phẩm"""
        detail = product.get('product_detail', {})
        
        # Kiểm tra lỗi
        if 'error' in detail:
            return ""
        
        # Text fields
        name = detail.get('name', '')
        short_desc = detail.get('short_description', '')
        description = self.clean_html(detail.get('description', ''))
        specs = self.flatten_specifications(detail.get('specifications', []))
        
        # Brand
        brand = detail.get('brand', {})
        brand_name = brand.get('name', '') if isinstance(brand, dict) else ''
        
        # Category
        category = product.get('category', '')
        
        combined = f"{name}. {short_desc}. {description}. {specs}. Thương hiệu {brand_name}. Danh mục {category}"
        
        # Làm sạch
        combined = re.sub(r'\s+', ' ', combined).strip()
        return combined[:512]  # Giới hạn độ dài cho PhoBERT
    
    def extract_item_features(self, product: Dict) -> Dict:
        """Trích xuất features cho Item Tower"""
        detail = product.get('product_detail', {})
        
        if 'error' in detail:
            return None
        
        # Text content
        text_content = self.extract_item_text(product)
        
        # Categorical features
        category = product.get('category', 'unknown')
        brand_id = detail.get('brand', {}).get('id', 0) if isinstance(detail.get('brand'), dict) else 0
        
        # Product metadata
        product_id = product.get('product_id', 0)
        
        return {
            'product_id': product_id,
            'text_content': text_content,
            'category': category,
            'brand_id': brand_id
        }
    
    def extract_ranking_features(self, product: Dict) -> Dict:
        """Trích xuất features cho Ranking Model"""
        detail = product.get('product_detail', {})
        
        if 'error' in detail:
            return None
        
        # Dense features
        price = detail.get('price', 0)
        list_price = detail.get('list_price', 0)
        discount_rate = detail.get('discount_rate', 0)
        rating_avg = detail.get('rating_average', 0)
        review_count = detail.get('review_count', 0)
        
        # Quantity sold
        qty_sold = detail.get('quantity_sold', {})
        if isinstance(qty_sold, dict):
            qty_sold_value = qty_sold.get('value', 0)
        else:
            qty_sold_value = 0
        
        # Seller
        seller = detail.get('current_seller', {})
        seller_id = seller.get('id', 0) if isinstance(seller, dict) else 0
        
        # Badges (rất quan trọng!)
        badges = detail.get('badges_new', [])
        is_authentic = any(b.get('code') == 'is_authentic' for b in badges)
        is_freeship = any(b.get('code') == 'freeship_xtra' for b in badges)
        has_return = any(b.get('code') == 'return_policy' for b in badges)
        
        # Inventory
        inventory_status = detail.get('inventory_status', 'unknown')
        is_available = 1 if inventory_status == 'available' else 0
        
        return {
            'product_id': product.get('product_id', 0),
            'price': price,
            'list_price': list_price,
            'discount_rate': discount_rate,
            'rating_average': rating_avg,
            'review_count': review_count,
            'quantity_sold': qty_sold_value,
            'seller_id': seller_id,
            'is_authentic': int(is_authentic),
            'is_freeship': int(is_freeship),
            'has_return_policy': int(has_return),
            'is_available': is_available
        }
    
    def extract_auxiliary_labels(self, reviews: List[Dict]) -> Tuple[int, int]:
        """
        Trích xuất labels cho auxiliary tasks từ reviews
        Returns: (y_quality, y_price)
        """
        y_quality = 0
        y_price = 0
        
        for review in reviews:
            # Task 2: Quality
            vote_attrs = review.get('vote_attributes', {})
            if isinstance(vote_attrs, dict):
                agree_list = vote_attrs.get('agree', [])
                if isinstance(agree_list, list):
                    quality_keywords = ['chất lượng', 'thấm hút', 'bền', 'đẹp', 'tốt']
                    for keyword in quality_keywords:
                        if any(keyword in str(item).lower() for item in agree_list):
                            y_quality = 1
                            break
            
            # Task 3: Price sensitivity
            content = review.get('content', '').lower()
            if any(kw in content for kw in ['giá rẻ', 'giá tốt', 'rẻ', 'hời']):
                y_price = 1
        
        return y_quality, y_price
    
    def load_and_preprocess(self, jsonl_path: Optional[str] = None, 
                           sample_size: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load và preprocess toàn bộ dataset từ JSONL hoặc từ thư mục JSON
        Returns: (item_features_df, ranking_features_df)
        """
        item_data = []
        ranking_data = []
        
        if jsonl_path and Path(jsonl_path).exists():
            # Load từ JSONL file
            print(f"\nLoading from JSONL: {jsonl_path}")
            
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if sample_size:
                    lines = lines[:sample_size]
                
                for line in tqdm(lines, desc="Processing JSONL"):
                    try:
                        product = json.loads(line)
                        
                        # Extract features
                        item_feat = self.extract_item_features(product)
                        if item_feat:
                            item_data.append(item_feat)
                        
                        rank_feat = self.extract_ranking_features(product)
                        if rank_feat:
                            ranking_data.append(rank_feat)
                    except Exception as e:
                        continue
        else:
            # Load từ thư mục JSON files
            json_files = list(self.data_dir.glob("*.json"))
            
            if sample_size:
                json_files = json_files[:sample_size]
            
            print(f"\nProcessing {len(json_files)} JSON files...")
            
            for json_file in tqdm(json_files, desc="Processing files"):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        product = json.load(f)
                    
                    # Extract item features
                    item_feat = self.extract_item_features(product)
                    if item_feat:
                        item_data.append(item_feat)
                    
                    # Extract ranking features
                    rank_feat = self.extract_ranking_features(product)
                    if rank_feat:
                        ranking_data.append(rank_feat)
                        
                except Exception as e:
                    continue
        
        item_df = pd.DataFrame(item_data)
        ranking_df = pd.DataFrame(ranking_data)
        
        # Clean descriptions in batch
        if len(item_df) > 0:
            print("\nCleaning text content...")
            item_df['text_content'] = self.html_cleaner.batch_clean_descriptions(
                item_df['text_content'].tolist()
            )
        
        print(f"\n{'='*60}")
        print(f"PREPROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Items extracted:    {len(item_df):,}")
        print(f"Ranking samples:    {len(ranking_df):,}")
        print(f"{'='*60}\n")
        
        return item_df, ranking_df


# ============================================================================
# PART 2: METHOD 1 - SEMANTIC-ENHANCED TWO-TOWER MODEL (RETRIEVAL)
# ============================================================================

class TwoTowerDataset(Dataset):
    """Dataset cho Two-Tower Model"""
    
    def __init__(self, interactions: List[Tuple[int, int]], 
                 item_features: Dict, tokenizer, max_len: int = 128):
        """
        interactions: List of (user_id, item_id) tuples
        item_features: Dict mapping item_id to text content
        """
        self.interactions = interactions
        self.item_features = item_features
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.interactions)
    
    def __getitem__(self, idx):
        user_id, item_id = self.interactions[idx]
        
        # Get item text
        item_text = self.item_features.get(item_id, "")
        
        # Tokenize
        encoding = self.tokenizer(
            item_text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'user_id': torch.tensor(user_id, dtype=torch.long),
            'item_id': torch.tensor(item_id, dtype=torch.long),
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }


class ItemTower(nn.Module):
    """Item Tower với PhoBERT để encode text"""
    
    def __init__(self, phobert_model_name: str = 'vinai/phobert-base', 
                 embedding_dim: int = 128, num_categories: int = 100, 
                 num_brands: int = 1000):
        super().__init__()
        
        # PhoBERT cho text encoding
        self.phobert = AutoModel.from_pretrained(phobert_model_name)
        phobert_dim = self.phobert.config.hidden_size  # 768
        
        # Embeddings cho categorical features
        self.category_emb = nn.Embedding(num_categories, 32)
        self.brand_emb = nn.Embedding(num_brands, 32)
        
        # Projection layers
        total_dim = phobert_dim + 32 + 32  # 768 + 32 + 32 = 832
        self.projector = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
    
    def forward(self, input_ids, attention_mask, category_ids, brand_ids):
        # PhoBERT encoding
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        text_emb = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        
        # Categorical embeddings
        cat_emb = self.category_emb(category_ids)
        brand_emb = self.brand_emb(brand_ids)
        
        # Concatenate
        combined = torch.cat([text_emb, cat_emb, brand_emb], dim=1)
        
        # Project to final embedding
        item_embedding = self.projector(combined)
        
        # L2 normalize
        item_embedding = F.normalize(item_embedding, p=2, dim=1)
        
        return item_embedding


class UserTower(nn.Module):
    """User Tower"""
    
    def __init__(self, num_users: int, embedding_dim: int = 128):
        super().__init__()
        
        self.user_emb = nn.Embedding(num_users, 64)
        
        self.projector = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
    
    def forward(self, user_ids):
        user_emb = self.user_emb(user_ids)
        user_embedding = self.projector(user_emb)
        
        # L2 normalize
        user_embedding = F.normalize(user_embedding, p=2, dim=1)
        
        return user_embedding


class TwoTowerModel(nn.Module):
    """Complete Two-Tower Model"""
    
    def __init__(self, num_users: int, num_categories: int, num_brands: int,
                 embedding_dim: int = 128):
        super().__init__()
        
        self.user_tower = UserTower(num_users, embedding_dim)
        self.item_tower = ItemTower(
            embedding_dim=embedding_dim,
            num_categories=num_categories,
            num_brands=num_brands
        )
        
        self.temperature = nn.Parameter(torch.tensor(0.07))
    
    def forward(self, user_ids, input_ids, attention_mask, category_ids, brand_ids):
        user_emb = self.user_tower(user_ids)
        item_emb = self.item_tower(input_ids, attention_mask, category_ids, brand_ids)
        
        return user_emb, item_emb
    
    def compute_similarity(self, user_emb, item_emb):
        """Cosine similarity với temperature scaling"""
        similarity = torch.matmul(user_emb, item_emb.T) / self.temperature
        return similarity


# ============================================================================
# PART 3: METHOD 2 - MULTI-TASK LEARNING MMoE (RANKING)
# ============================================================================

class Expert(nn.Module):
    """Single Expert Network"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
    
    def forward(self, x):
        return self.network(x)


class GatingNetwork(nn.Module):
    """Gating Network cho từng task"""
    
    def __init__(self, input_dim: int, num_experts: int):
        super().__init__()
        
        self.gate = nn.Sequential(
            nn.Linear(input_dim, num_experts),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        return self.gate(x)


class MMoEModel(nn.Module):
    """Multi-gate Mixture-of-Experts Model"""
    
    def __init__(self, input_dim: int, num_experts: int = 4, 
                 expert_hidden_dim: int = 128, tower_hidden_dim: int = 64):
        super().__init__()
        
        self.num_experts = num_experts
        
        # Shared Experts
        self.experts = nn.ModuleList([
            Expert(input_dim, expert_hidden_dim) for _ in range(num_experts)
        ])
        
        # Gating Networks for 3 tasks
        self.gate_purchase = GatingNetwork(input_dim, num_experts)
        self.gate_quality = GatingNetwork(input_dim, num_experts)
        self.gate_price = GatingNetwork(input_dim, num_experts)
        
        # Task-specific Towers
        self.tower_purchase = nn.Sequential(
            nn.Linear(expert_hidden_dim, tower_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(tower_hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.tower_quality = nn.Sequential(
            nn.Linear(expert_hidden_dim, tower_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(tower_hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.tower_price = nn.Sequential(
            nn.Linear(expert_hidden_dim, tower_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(tower_hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Get expert outputs
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        # Shape: [batch_size, num_experts, expert_hidden_dim]
        
        # Get gate weights for each task
        gate_purchase = self.gate_purchase(x).unsqueeze(2)  # [batch, num_experts, 1]
        gate_quality = self.gate_quality(x).unsqueeze(2)
        gate_price = self.gate_price(x).unsqueeze(2)
        
        # Weighted sum of expert outputs
        purchase_input = torch.sum(expert_outputs * gate_purchase, dim=1)
        quality_input = torch.sum(expert_outputs * gate_quality, dim=1)
        price_input = torch.sum(expert_outputs * gate_price, dim=1)
        
        # Task towers
        purchase_pred = self.tower_purchase(purchase_input)
        quality_pred = self.tower_quality(quality_input)
        price_pred = self.tower_price(price_input)
        
        return purchase_pred, quality_pred, price_pred


class RankingDataset(Dataset):
    """Dataset cho Ranking Model"""
    
    def __init__(self, features: pd.DataFrame, labels: Optional[pd.DataFrame] = None):
        self.features = torch.FloatTensor(features.values)
        self.labels = torch.FloatTensor(labels.values) if labels is not None else None
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        if self.labels is not None:
            return self.features[idx], self.labels[idx]
        return self.features[idx]


# ============================================================================
# PART 4: TRAINING UTILITIES
# ============================================================================

def train_two_tower_epoch(model, dataloader, optimizer, device):
    """Train Two-Tower Model for one epoch"""
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training Two-Tower"):
        user_ids = batch['user_id'].to(device)
        item_ids = batch['item_id'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # Forward pass
        user_emb, item_emb = model(
            user_ids, input_ids, attention_mask,
            category_ids=torch.zeros_like(user_ids),  # Placeholder
            brand_ids=torch.zeros_like(user_ids)  # Placeholder
        )
        
        # Compute similarity
        similarity = model.compute_similarity(user_emb, item_emb)
        
        # Contrastive loss (InfoNCE)
        labels = torch.arange(len(user_ids)).to(device)
        loss = F.cross_entropy(similarity, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def train_mmoe_epoch(model, dataloader, optimizer, device):
    """Train MMoE Model for one epoch"""
    model.train()
    total_loss = 0
    
    criterion = nn.BCELoss()
    
    for features, labels in tqdm(dataloader, desc="Training MMoE"):
        features = features.to(device)
        labels = labels.to(device)
        
        # Forward
        purchase_pred, quality_pred, price_pred = model(features)
        
        # Multi-task loss
        loss_purchase = criterion(purchase_pred.squeeze(), labels[:, 0])
        loss_quality = criterion(quality_pred.squeeze(), labels[:, 1])
        loss_price = criterion(price_pred.squeeze(), labels[:, 2])
        
        # Weighted combination
        loss = loss_purchase + 0.5 * loss_quality + 0.5 * loss_price
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


# ============================================================================
# PART 5: COMPLETE PIPELINE
# ============================================================================

class RecommendationPipeline:
    """Complete end-to-end pipeline"""
    
    def __init__(self, data_dir: str = "TikDataset"):
        self.data_dir = data_dir
        self.preprocessor = TikDataPreprocessor(data_dir)
        self.cleaner = DataCleaner()
    
    def run_full_pipeline(self, raw_jsonl: Optional[str] = None, 
                         clean_jsonl: Optional[str] = None,
                         sample_size: Optional[int] = None):
        """
        Chạy toàn bộ pipeline từ A-Z
        """
        print("\n" + "="*80)
        print("INTELLIGENT RECOMMENDATION SYSTEM - FULL PIPELINE")
        print("="*80)
        
        # Step 1: Clean JSONL if provided
        if raw_jsonl and clean_jsonl:
            print("\n[STEP 1] Cleaning raw JSONL data...")
            self.cleaner.clean_jsonl_dataset(raw_jsonl, clean_jsonl)
            data_source = clean_jsonl
        elif clean_jsonl:
            data_source = clean_jsonl
        else:
            data_source = None
        
        # Step 2: Preprocess data
        print("\n[STEP 2] Preprocessing features...")
        item_df, ranking_df = self.preprocessor.load_and_preprocess(
            jsonl_path=data_source,
            sample_size=sample_size
        )
        
        # Step 3: Save processed data
        print("\n[STEP 3] Saving preprocessed data...")
        output_dir = Path("data/processed")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        item_path = output_dir / "item_features.csv"
        ranking_path = output_dir / "ranking_features.csv"
        
        item_df.to_csv(item_path, index=False)
        ranking_df.to_csv(ranking_path, index=False)
        
        print(f"\n✓ Item features saved:    {item_path}")
        print(f"✓ Ranking features saved: {ranking_path}")
        
        # Step 4: Display statistics
        print("\n" + "="*80)
        print("DATA STATISTICS")
        print("="*80)
        
        print("\n[ITEM FEATURES]")
        print(f"Total products:           {len(item_df):,}")
        print(f"Unique categories:        {item_df['category'].nunique():,}")
        print(f"Unique brands:            {item_df['brand_id'].nunique():,}")
        print(f"Avg text length:          {item_df['text_content'].str.len().mean():.0f} chars")
        
        print("\n[RANKING FEATURES]")
        print(f"Total samples:            {len(ranking_df):,}")
        print(f"Avg price:                {ranking_df['price'].mean():,.0f} VNĐ")
        print(f"Avg rating:               {ranking_df['rating_average'].mean():.2f}/5")
        print(f"Authentic products:       {ranking_df['is_authentic'].sum():,} ({ranking_df['is_authentic'].mean()*100:.1f}%)")
        print(f"Freeship products:        {ranking_df['is_freeship'].sum():,} ({ranking_df['is_freeship'].mean()*100:.1f}%)")
        
        print("\n" + "="*80)
        print("PIPELINE COMPLETE!")
        print("="*80)
        
        return item_df, ranking_df


# ============================================================================
# PART 6: MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='TikDataset Recommendation System')
    parser.add_argument('--raw_jsonl', type=str, help='Path to raw JSONL file')
    parser.add_argument('--clean_jsonl', type=str, help='Path to save/load cleaned JSONL')
    parser.add_argument('--data_dir', type=str, default='TikDataset', help='Directory containing JSON files')
    parser.add_argument('--sample_size', type=int, help='Number of samples to process (for testing)')
    parser.add_argument('--mode', type=str, choices=['clean', 'preprocess', 'full'], 
                       default='full', help='Pipeline mode')
    
    args = parser.parse_args()
    
    pipeline = RecommendationPipeline(data_dir=args.data_dir)
    
    if args.mode == 'clean' and args.raw_jsonl and args.clean_jsonl:
        # Only clean JSONL
        print("\n[MODE: CLEAN ONLY]")
        DataCleaner.clean_jsonl_dataset(args.raw_jsonl, args.clean_jsonl)
    
    elif args.mode == 'preprocess':
        # Only preprocess
        print("\n[MODE: PREPROCESS ONLY]")
        item_df, ranking_df = pipeline.preprocessor.load_and_preprocess(
            jsonl_path=args.clean_jsonl,
            sample_size=args.sample_size
        )
    
    else:
        # Full pipeline
        print("\n[MODE: FULL PIPELINE]")
        item_df, ranking_df = pipeline.run_full_pipeline(
            raw_jsonl=args.raw_jsonl,
            clean_jsonl=args.clean_jsonl,
            sample_size=args.sample_size
        )
    
    print("\n✓ Done! Ready for model training.")
    print("\nNext steps:")
    print("1. Train Two-Tower Model: python train_two_tower.py")
    print("2. Train MMoE Model: python train_mmoe.py")
    print("3. Build inference pipeline")