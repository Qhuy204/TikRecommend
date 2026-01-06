"""
Training Scripts for Recommendation Models
- train_two_tower.py: Train retrieval model
- train_mmoe.py: Train ranking model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Import models từ recommendation_system.py
# from recommendation_system import TwoTowerModel, MMoEModel, TwoTowerDataset, RankingDataset


# ============================================================================
# TRAINING TWO-TOWER MODEL
# ============================================================================

class TwoTowerTrainer:
    """Trainer cho Two-Tower Retrieval Model"""
    
    def __init__(self, model, device='cuda', learning_rate=1e-4):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
    def train_epoch(self, dataloader):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(dataloader, desc="Training"):
            # Move to device
            user_ids = batch['user_id'].to(self.device)
            item_ids = batch['item_id'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            category_ids = batch.get('category_id', torch.zeros_like(user_ids)).to(self.device)
            brand_ids = batch.get('brand_id', torch.zeros_like(user_ids)).to(self.device)
            
            # Forward pass
            user_emb, item_emb = self.model(
                user_ids, input_ids, attention_mask, category_ids, brand_ids
            )
            
            # Compute similarity matrix
            similarity = self.model.compute_similarity(user_emb, item_emb)
            
            # In-batch negatives: mỗi user chỉ match với 1 item trong batch
            labels = torch.arange(len(user_ids)).to(self.device)
            loss = nn.CrossEntropyLoss()(similarity, labels)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def evaluate(self, dataloader, k=10):
        """Evaluate Recall@K"""
        self.model.eval()
        total_recall = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                user_ids = batch['user_id'].to(self.device)
                item_ids = batch['item_id'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                category_ids = batch.get('category_id', torch.zeros_like(user_ids)).to(self.device)
                brand_ids = batch.get('brand_id', torch.zeros_like(user_ids)).to(self.device)
                
                # Forward
                user_emb, item_emb = self.model(
                    user_ids, input_ids, attention_mask, category_ids, brand_ids
                )
                
                # Compute similarity
                similarity = self.model.compute_similarity(user_emb, item_emb)
                
                # Get top-k predictions
                _, top_k_indices = torch.topk(similarity, k, dim=1)
                
                # Compute Recall@K (trong batch, true item là diagonal)
                labels = torch.arange(len(user_ids)).to(self.device)
                hits = (top_k_indices == labels.unsqueeze(1)).any(dim=1).float()
                recall = hits.mean().item()
                
                total_recall += recall
                num_batches += 1
        
        return total_recall / num_batches


def prepare_two_tower_data(item_df, interactions_df):
    """
    Chuẩn bị dữ liệu cho Two-Tower Model
    
    Args:
        item_df: DataFrame với columns [product_id, text_content, category, brand_id]
        interactions_df: DataFrame với columns [user_id, product_id, rating]
    
    Returns:
        train_dataset, val_dataset, tokenizer
    """
    from transformers import AutoTokenizer
    from sklearn.preprocessing import LabelEncoder

    user_encoder = LabelEncoder()
    # Chuyển đổi toàn bộ user_id về dải số liên tục bắt đầu từ 0
    interactions_df['user_id'] = user_encoder.fit_transform(interactions_df['user_id'])
        
    # Load PhoBERT tokenizer
    tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
    
    # Encode categories và brands
    cat_encoder = LabelEncoder()
    brand_encoder = LabelEncoder()
    
    item_df['category_encoded'] = cat_encoder.fit_transform(item_df['category'].fillna('unknown'))
    item_df['brand_encoded'] = brand_encoder.fit_transform(item_df['brand_id'].fillna(0).astype(str))
    
    # Create item features dict
    item_features = {}
    for _, row in item_df.iterrows():
        item_features[row['product_id']] = {
            'text': row['text_content'],
            'category': row['category_encoded'],
            'brand': row['brand_encoded']
        }
    
    # Filter positive interactions (rating >= 4)
    positive_interactions = interactions_df[interactions_df['rating'] >= 4]
    
    # Create interaction list
    interactions = list(zip(
        positive_interactions['user_id'],
        positive_interactions['product_id']
    ))
    
    # Train-val split
    train_interactions, val_interactions = train_test_split(
        interactions, test_size=0.2, random_state=42
    )
    
    # Create datasets
    from recommendation_system import TwoTowerDataset
    
    train_dataset = TwoTowerDataset(train_interactions, item_features, tokenizer)
    val_dataset = TwoTowerDataset(val_interactions, item_features, tokenizer)
    
    return train_dataset, val_dataset, tokenizer


def train_two_tower_model(item_csv='data/processed/item_features.csv',
                          interactions_csv='data/processed/interactions.csv',
                          epochs=10,
                          batch_size=64):
    """
    Main training function cho Two-Tower Model
    """
    print("\n" + "="*80)
    print("TRAINING TWO-TOWER MODEL (RETRIEVAL)")
    print("="*80)
    
    # Check if files exist
    from pathlib import Path
    
    if not Path(item_csv).exists():
        print(f"\n❌ Error: {item_csv} not found!")
        print("\nPlease run preprocessing first:")
        print("  make preprocess")
        print("  Or: python recommendation_system.py --mode preprocess")
        return
    
    if not Path(interactions_csv).exists():
        print(f"\n❌ Error: {interactions_csv} not found!")
        print("\nPlease extract interactions first:")
        print("  make interactions")
        print("  Or: python create_interactions.py --input data/clean/tiki_dataset_clean.jsonl")
        return
    
    # Load data
    print("\n[1] Loading data...")
    item_df = pd.read_csv(item_csv)
    interactions_df = pd.read_csv(interactions_csv)
    
    print(f"Items: {len(item_df):,}")
    print(f"Interactions: {len(interactions_df):,}")
    
    # Prepare datasets
    print("\n[2] Preparing datasets...")
    train_dataset, val_dataset, tokenizer = prepare_two_tower_data(item_df, interactions_df)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    print("\n[3] Initializing model...")
    num_users = int(interactions_df['user_id'].max() + 1)
    num_categories = int(item_df['category_encoded'].max() + 1)
    num_brands = int(item_df['brand_encoded'].max() + 1)
        
    from recommendation_system import TwoTowerModel
    model = TwoTowerModel(
        num_users=num_users,
        num_categories=num_categories,
        num_brands=num_brands,
        embedding_dim=128
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Train
    print(f"\n[4] Training for {epochs} epochs...")
    trainer = TwoTowerTrainer(model, device=device)
    
    best_recall = 0
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Train
        train_loss = trainer.train_epoch(train_loader)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Evaluate
        val_recall = trainer.evaluate(val_loader, k=10)
        print(f"Val Recall@10: {val_recall:.4f}")
        
        # Save best model
        if val_recall > best_recall:
            best_recall = val_recall
            torch.save(model.state_dict(), 'models/two_tower_best.pt')
            print(f"✓ Saved best model (Recall@10: {best_recall:.4f})")
    
    print("\n" + "="*80)
    print(f"TRAINING COMPLETE! Best Recall@10: {best_recall:.4f}")
    print("="*80)


# ============================================================================
# TRAINING MMoE MODEL
# ============================================================================

class MMoETrainer:
    """Trainer cho MMoE Ranking Model"""
    
    def __init__(self, model, device='cuda', learning_rate=1e-3):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.BCELoss()
        
    def train_epoch(self, dataloader):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        
        for features, labels in tqdm(dataloader, desc="Training"):
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            # Forward
            purchase_pred, quality_pred, price_pred = self.model(features)
            
            # Multi-task loss
            loss_purchase = self.criterion(purchase_pred.squeeze(), labels[:, 0])
            loss_quality = self.criterion(quality_pred.squeeze(), labels[:, 1])
            loss_price = self.criterion(price_pred.squeeze(), labels[:, 2])
            
            # Weighted sum (có thể tune weights)
            loss = loss_purchase + 0.5 * loss_quality + 0.5 * loss_price
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def evaluate(self, dataloader):
        """Evaluate AUC for each task"""
        from sklearn.metrics import roc_auc_score
        
        self.model.eval()
        all_preds = {'purchase': [], 'quality': [], 'price': []}
        all_labels = {'purchase': [], 'quality': [], 'price': []}
        
        with torch.no_grad():
            for features, labels in tqdm(dataloader, desc="Evaluating"):
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                # Forward
                purchase_pred, quality_pred, price_pred = self.model(features)
                
                # Store predictions
                all_preds['purchase'].extend(purchase_pred.cpu().numpy().flatten())
                all_preds['quality'].extend(quality_pred.cpu().numpy().flatten())
                all_preds['price'].extend(price_pred.cpu().numpy().flatten())
                
                all_labels['purchase'].extend(labels[:, 0].cpu().numpy())
                all_labels['quality'].extend(labels[:, 1].cpu().numpy())
                all_labels['price'].extend(labels[:, 2].cpu().numpy())
        
        # Compute AUC for each task
        auc_scores = {}
        for task in ['purchase', 'quality', 'price']:
            try:
                auc = roc_auc_score(all_labels[task], all_preds[task])
                auc_scores[task] = auc
            except:
                auc_scores[task] = 0.5
        
        return auc_scores


def prepare_mmoe_data(ranking_df):
    """
    Chuẩn bị dữ liệu cho MMoE Model
    
    Args:
        ranking_df: DataFrame với features và labels
    
    Returns:
        train_dataset, val_dataset, scaler
    """
    from sklearn.preprocessing import StandardScaler
    from recommendation_system import RankingDataset
    
    # Features columns
    feature_cols = [
        'price', 'list_price', 'discount_rate', 'rating_average',
        'review_count', 'quantity_sold', 'seller_id',
        'is_authentic', 'is_freeship', 'has_return_policy', 'is_available'
    ]
    
    # Labels columns (cần tạo từ dữ liệu thực tế)
    # Giả sử đã có labels: y_purchase, y_quality, y_price
    label_cols = ['y_purchase', 'y_quality', 'y_price']
    
    # Nếu chưa có labels, tạo synthetic labels cho demo
    if not all(col in ranking_df.columns for col in label_cols):
        print("Warning: Creating synthetic labels for demo")
        ranking_df['y_purchase'] = (ranking_df['rating_average'] >= 4).astype(int)
        ranking_df['y_quality'] = (ranking_df['rating_average'] >= 4.5).astype(int)
        ranking_df['y_price'] = (ranking_df['discount_rate'] > 20).astype(int)
    
    # Normalize features
    scaler = StandardScaler()
    X = ranking_df[feature_cols].fillna(0).values
    X_scaled = scaler.fit_transform(X)
    
    y = ranking_df[label_cols].values
    
    # Train-val split
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # Create datasets
    train_df = pd.DataFrame(X_train)
    val_df = pd.DataFrame(X_val)
    train_labels = pd.DataFrame(y_train)
    val_labels = pd.DataFrame(y_val)
    
    train_dataset = RankingDataset(train_df, train_labels)
    val_dataset = RankingDataset(val_df, val_labels)
    
    return train_dataset, val_dataset, scaler


def train_mmoe_model(ranking_csv='data/processed/ranking_features.csv',
                    epochs=20,
                    batch_size=256):
    """
    Main training function cho MMoE Model
    """
    print("\n" + "="*80)
    print("TRAINING MMoE MODEL (RANKING)")
    print("="*80)
    
    # Check if file exists
    from pathlib import Path
    
    if not Path(ranking_csv).exists():
        print(f"\n❌ Error: {ranking_csv} not found!")
        print("\nPlease run preprocessing first:")
        print("  make preprocess")
        print("  Or: python recommendation_system.py --mode preprocess")
        return
    
    # Load data
    print("\n[1] Loading data...")
    ranking_df = pd.read_csv(ranking_csv)
    print(f"Samples: {len(ranking_df):,}")
    
    # Prepare datasets
    print("\n[2] Preparing datasets...")
    train_dataset, val_dataset, scaler = prepare_mmoe_data(ranking_df)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    print("\n[3] Initializing model...")
    input_dim = len(train_dataset[0][0])
    
    from recommendation_system import MMoEModel
    model = MMoEModel(
        input_dim=input_dim,
        num_experts=4,
        expert_hidden_dim=128,
        tower_hidden_dim=64
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Train
    print(f"\n[4] Training for {epochs} epochs...")
    trainer = MMoETrainer(model, device=device)
    
    best_auc = 0
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Train
        train_loss = trainer.train_epoch(train_loader)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Evaluate
        auc_scores = trainer.evaluate(val_loader)
        avg_auc = np.mean(list(auc_scores.values()))
        
        print(f"Val AUC - Purchase: {auc_scores['purchase']:.4f}")
        print(f"Val AUC - Quality:  {auc_scores['quality']:.4f}")
        print(f"Val AUC - Price:    {auc_scores['price']:.4f}")
        print(f"Val AUC - Average:  {avg_auc:.4f}")
        
        # Save best model
        if avg_auc > best_auc:
            best_auc = avg_auc
            torch.save({
                'model_state_dict': model.state_dict(),
                'scaler': scaler
            }, 'models/mmoe_best.pt')
            print(f"✓ Saved best model (Avg AUC: {best_auc:.4f})")
    
    print("\n" + "="*80)
    print(f"TRAINING COMPLETE! Best Avg AUC: {best_auc:.4f}")
    print("="*80)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Recommendation Models')
    parser.add_argument('--model', type=str, choices=['two_tower', 'mmoe', 'both'],
                       default='both', help='Which model to train')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    
    args = parser.parse_args()
    
    # Create models directory
    Path('models').mkdir(exist_ok=True)
    
    if args.model in ['two_tower', 'both']:
        print("\n" + "🚀 "*20)
        train_two_tower_model(epochs=args.epochs, batch_size=args.batch_size)
    
    if args.model in ['mmoe', 'both']:
        print("\n" + "🚀 "*20)
        train_mmoe_model(epochs=args.epochs, batch_size=args.batch_size)
    
    print("\n✅ All training complete!")