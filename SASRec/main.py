#!/usr/bin/env python3
"""
SASRec + PhoBERT Fusion Recommendation System
Main entry point for training, evaluation, and inference

Usage:
    python main.py --mode train
    python main.py --mode evaluate
    python main.py --mode demo --user_id 12345
"""

import argparse
import json
from pathlib import Path

import torch

from config import default_config, Config
from data_processor import TikiDataProcessor, create_dataloaders, SASRecDataset
from models import SASRecModel, SASRecPhoBERTFusion
from trainer import Trainer
from recommender import SASRecRecommender, evaluate_recommender


def train(args):
    """Train the SASRec model"""
    print("=" * 60)
    print("SASRec + PhoBERT Training Pipeline")
    print("=" * 60)
    
    # Load and process data
    print("\nLoading data...")
    processor = TikiDataProcessor()
    processor.load_raw_data()
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_data, val_data, test_data = processor.create_train_val_test_split()
    
    train_dataset = SASRecDataset(
        train_data,
        processor.num_items,
        max_seq_length=default_config.data.max_seq_length,
        num_negatives=default_config.training.num_negatives,
        mode='train'
    )
    
    val_dataset = SASRecDataset(
        val_data,
        processor.num_items,
        mode='val'
    )
    
    from torch.utils.data import DataLoader
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size or default_config.training.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size or default_config.training.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Create model
    print(f"\nCreating SASRec model...")
    print(f"   Items: {processor.num_items:,}")
    print(f"   Users: {processor.num_users:,}")
    
    model = SASRecModel(
        num_items=processor.num_items,
        config=default_config.sasrec
    )
    
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        config=default_config.training
    )
    
    # Train
    epochs = args.epochs or default_config.training.epochs
    result = trainer.train(train_loader, val_loader, epochs=epochs)
    
    # Save processor for later use
    processor_path = Path(default_config.training.save_dir) / 'data_processor.json'
    with open(processor_path, 'w') as f:
        json.dump({
            'user2idx': processor.user2idx,
            'idx2user': {str(k): v for k, v in processor.idx2user.items()},
            'item2idx': processor.item2idx,
            'idx2item': {str(k): v for k, v in processor.idx2item.items()},
            'num_users': processor.num_users,
            'num_items': processor.num_items
        }, f)
    
    print(f"\nTraining complete!")
    print(f"   Best HR@10: {result['best_metric']:.4f}")
    print(f"   Checkpoints saved to: {default_config.training.save_dir}/")
    
    return result


def evaluate(args):
    """Evaluate trained model"""
    print("=" * 60)
    print("SASRec Model Evaluation")
    print("=" * 60)
    
    # Load data
    processor = TikiDataProcessor()
    processor.load_raw_data()
    
    # Create test data
    _, _, test_data = processor.create_train_val_test_split()
    
    # Load model
    checkpoint_path = args.checkpoint or f"{default_config.training.save_dir}/best_model.pt"
    recommender = SASRecRecommender.load(
        checkpoint_path,
        processor,
        model_type='sasrec'
    )
    
    # Evaluate
    print("\nEvaluating...")
    metrics = evaluate_recommender(recommender, test_data, k_values=[5, 10, 20])
    
    print("\n" + "=" * 40)
    print("EVALUATION RESULTS")
    print("=" * 40)
    
    for metric, value in sorted(metrics.items()):
        print(f"   {metric}: {value:.4f}")
    
    # Save results
    results_path = Path(default_config.training.save_dir) / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    
    return metrics


def demo(args):
    """Interactive demo"""
    print("=" * 60)
    print("SASRec Recommendation Demo")
    print("=" * 60)
    
    # Load data
    processor = TikiDataProcessor()
    processor.load_raw_data()
    
    # Load model
    checkpoint_path = args.checkpoint or f"{default_config.training.save_dir}/best_model.pt"
    
    if not Path(checkpoint_path).exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        print("   Please train the model first: python main.py --mode train")
        return
    
    recommender = SASRecRecommender.load(
        checkpoint_path,
        processor,
        model_type='sasrec'
    )
    
    if args.user_id:
        # Recommend for specific user
        print(f"\nRecommendations for user {args.user_id}:")
        recs = recommender.recommend_for_user(args.user_id, top_k=args.top_k)
        
        if not recs:
            print("   No recommendations available (user not found)")
        else:
            for i, rec in enumerate(recs, 1):
                print(f"\n   {i}. {rec['name'][:60]}...")
                print(f"      Category: {rec['category']}")
                print(f"      Price: {rec['price']:,} VND")
                print(f"      Score: {rec['score']:.4f}")
    
    elif args.item_id:
        # Find similar items
        print(f"\nSimilar items to {args.item_id}:")
        similars = recommender.get_similar_items(args.item_id, top_k=args.top_k)
        
        if not similars:
            print("   No similar items found (item not found)")
        else:
            for i, item in enumerate(similars, 1):
                print(f"\n   {i}. {item['name'][:60]}...")
                print(f"      Category: {item['category']}")
                print(f"      Similarity: {item['similarity']:.4f}")
    
    else:
        # Show sample recommendations
        print("\nSample recommendations:")
        sample_users = list(processor.user2idx.keys())[:3]
        
        for user_id in sample_users:
            print(f"\nUser {user_id}:")
            recs = recommender.recommend_for_user(user_id, top_k=5)
            for i, rec in enumerate(recs, 1):
                print(f"   {i}. {rec['name'][:50]}... (score: {rec['score']:.2f})")


def main():
    parser = argparse.ArgumentParser(
        description='SASRec + PhoBERT Recommendation System'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'evaluate', 'demo'],
        default='train',
        help='Mode: train, evaluate, or demo'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='Batch size'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to model checkpoint'
    )
    
    parser.add_argument(
        '--user_id',
        type=int,
        default=None,
        help='User ID for demo'
    )
    
    parser.add_argument(
        '--item_id',
        type=int,
        default=None,
        help='Item ID for similar items demo'
    )
    
    parser.add_argument(
        '--top_k',
        type=int,
        default=10,
        help='Number of recommendations'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train(args)
    elif args.mode == 'evaluate':
        evaluate(args)
    elif args.mode == 'demo':
        demo(args)


if __name__ == "__main__":
    main()
