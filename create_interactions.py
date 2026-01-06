"""
Script để tạo user-item interactions từ reviews data
"""

import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse


def extract_interactions_from_jsonl(jsonl_path: str, 
                                   output_path: str,
                                   min_rating: int = 4):
    """
    Trích xuất user-item interactions từ JSONL file
    
    Args:
        jsonl_path: Path đến clean JSONL file
        output_path: Path để save interactions CSV
        min_rating: Rating tối thiểu để coi là positive interaction
    """
    print("\n" + "="*80)
    print("EXTRACTING USER-ITEM INTERACTIONS")
    print("="*80)
    print(f"\nInput: {jsonl_path}")
    print(f"Output: {output_path}")
    print(f"Min rating threshold: {min_rating}")
    print("="*80 + "\n")
    
    interactions = []
    num_products = 0
    num_reviews = 0
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Processing"):
            try:
                data = json.loads(line)
                product_id = data.get('product_id')
                
                if not product_id:
                    continue
                
                num_products += 1
                reviews = data.get('reviews', [])
                
                for review in reviews:
                    num_reviews += 1
                    
                    # Extract review data
                    customer_id = review.get('customer_id')
                    rating = review.get('rating', 0)
                    created_time = review.get('created_at', '')
                    content = review.get('content', '')
                    
                    # Skip if missing essential data
                    if not customer_id or rating < min_rating:
                        continue
                    
                    # Extract auxiliary signals
                    vote_attrs = review.get('vote_attributes', {})
                    agree_list = vote_attrs.get('agree', []) if isinstance(vote_attrs, dict) else []
                    
                    # Quality signal
                    quality_keywords = ['chất lượng', 'thấm hút', 'bền', 'đẹp', 'tốt']
                    is_good_quality = any(
                        any(kw in str(item).lower() for kw in quality_keywords)
                        for item in agree_list
                    )
                    
                    # Price signal
                    price_keywords = ['giá rẻ', 'giá tốt', 'rẻ', 'hời']
                    is_good_price = any(kw in content.lower() for kw in price_keywords)
                    
                    interactions.append({
                        'user_id': customer_id,
                        'product_id': product_id,
                        'rating': rating,
                        'timestamp': created_time,
                        'is_good_quality': int(is_good_quality),
                        'is_good_price': int(is_good_price)
                    })
                    
            except json.JSONDecodeError:
                continue
            except Exception as e:
                continue
    
    # Create DataFrame
    df = pd.DataFrame(interactions)
    
    # Remove duplicates (keep latest)
    df = df.sort_values('timestamp').drop_duplicates(
        subset=['user_id', 'product_id'], 
        keep='last'
    )
    
    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print("\n" + "="*80)
    print("EXTRACTION COMPLETE")
    print("="*80)
    print(f"\nProcessed:")
    print(f"  Products:     {num_products:,}")
    print(f"  Reviews:      {num_reviews:,}")
    print(f"  Interactions: {len(df):,}")
    print(f"\nUnique:")
    print(f"  Users:        {df['user_id'].nunique():,}")
    print(f"  Products:     {df['product_id'].nunique():,}")
    print(f"\nQuality:")
    print(f"  Avg rating:   {df['rating'].mean():.2f}")
    print(f"  Good quality: {df['is_good_quality'].sum():,} ({df['is_good_quality'].mean()*100:.1f}%)")
    print(f"  Good price:   {df['is_good_price'].sum():,} ({df['is_good_price'].mean()*100:.1f}%)")
    print(f"\nSaved to: {output_path}")
    print("="*80 + "\n")
    
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract user-item interactions')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to clean JSONL file')
    parser.add_argument('--output', type=str, 
                       default='data/processed/interactions.csv',
                       help='Path to save interactions CSV')
    parser.add_argument('--min_rating', type=int, default=4,
                       help='Minimum rating for positive interaction')
    
    args = parser.parse_args()
    
    extract_interactions_from_jsonl(
        jsonl_path=args.input,
        output_path=args.output,
        min_rating=args.min_rating
    )