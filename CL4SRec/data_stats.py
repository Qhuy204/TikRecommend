"""
Data Visualization & Statistics for CL4SRec
Visualize data distribution before/after processing
"""

import json
from collections import defaultdict, Counter
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11


def load_raw_stats(filepath: str = "../data/raw/tiki_dataset.jsonl"):
    """Load raw data and compute statistics"""
    print("Loading raw data...")
    
    stats = {
        'total_lines': 0,
        'valid_products': 0,
        'skipped_errors': 0,
        'skipped_duplicates': 0,
        'total_reviews': 0,
        'products_with_reviews': 0,
    }
    
    item_review_counts = []  # Number of reviews per product
    user_review_counts = defaultdict(int)  # Reviews per user
    category_counts = defaultdict(int)
    rating_dist = defaultdict(int)
    seen_products = set()
    
    filepath = Path(filepath)
    if not filepath.exists():
        filepath = Path(__file__).parent / filepath
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Reading"):
            stats['total_lines'] += 1
            try:
                data = json.loads(line.strip())
                product_id = data.get('product_id')
                
                if not product_id:
                    continue
                
                product_detail = data.get('product_detail', {})
                
                # Check errors
                detail_str = str(product_detail).lower()
                if 'error' in detail_str or 'redirect' in detail_str:
                    stats['skipped_errors'] += 1
                    continue
                
                if not product_detail.get('name'):
                    stats['skipped_errors'] += 1
                    continue
                
                # Check duplicates
                if product_id in seen_products:
                    stats['skipped_duplicates'] += 1
                    continue
                seen_products.add(product_id)
                
                stats['valid_products'] += 1
                
                # Category
                category = data.get('category', 'Unknown')
                category_counts[category] += 1
                
                # Reviews
                reviews = data.get('reviews', [])
                num_reviews = len(reviews)
                item_review_counts.append(num_reviews)
                
                if num_reviews > 0:
                    stats['products_with_reviews'] += 1
                    stats['total_reviews'] += num_reviews
                    
                    for review in reviews:
                        user_id = review.get('customer_id')
                        rating = review.get('rating', 0)
                        if user_id:
                            user_review_counts[user_id] += 1
                        if rating > 0:
                            rating_dist[rating] += 1
                
            except:
                continue
    
    return {
        'stats': stats,
        'item_review_counts': item_review_counts,
        'user_review_counts': list(user_review_counts.values()),
        'category_counts': dict(category_counts),
        'rating_dist': dict(rating_dist)
    }


def load_processed_stats(filepath: str = "../data/processed/sasrec_processed.pkl"):
    """Load processed data statistics"""
    import pickle
    
    filepath = Path(filepath)
    if not filepath.exists():
        filepath = Path(__file__).parent / filepath
    
    print("Loading processed data...")
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    if isinstance(data, dict):
        num_items = data.get('num_items', 0)
        num_users = data.get('num_users', 0)
        user_sequences = data.get('user_sequences', {})
    else:
        num_items = data.num_items
        num_users = data.num_users
        user_sequences = data.user_sequences
    
    # Sequence lengths
    seq_lengths = [len(seq) for seq in user_sequences.values()]
    
    # Item frequencies
    item_freq = defaultdict(int)
    for seq in user_sequences.values():
        for item in seq:
            if isinstance(item, tuple):
                item_freq[item[0]] += 1
            else:
                item_freq[item] += 1
    
    return {
        'num_items': num_items,
        'num_users': num_users,
        'seq_lengths': seq_lengths,
        'item_frequencies': list(item_freq.values()),
        'total_interactions': sum(seq_lengths)
    }


def create_visualizations(raw_data, processed_data, output_dir: str = "figures"):
    """Create all visualizations"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    fig = plt.figure(figsize=(16, 14))
    
    # 1. Data Pipeline Overview (Top)
    ax1 = fig.add_subplot(3, 2, 1)
    
    stages = ['Raw Lines', 'Valid Products', 'Products w/ Reviews', 'After Filter\n(≥5 interactions)']
    values = [
        raw_data['stats']['total_lines'],
        raw_data['stats']['valid_products'],
        raw_data['stats']['products_with_reviews'],
        processed_data['num_items']
    ]
    colors = ['#3498db', '#2ecc71', '#f1c40f', '#e74c3c']
    
    bars = ax1.bar(stages, values, color=colors, edgecolor='black', linewidth=1.2)
    ax1.set_ylabel('Count')
    ax1.set_title('Data Pipeline: Items at Each Stage', fontsize=13, fontweight='bold')
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000, 
                f'{val:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax1.set_ylim(0, max(values) * 1.15)
    
    # 2. Users Pipeline
    ax2 = fig.add_subplot(3, 2, 2)
    
    user_stages = ['Total Users\n(Raw)', 'After Filter\n(≥3 interactions)']
    user_values = [len(raw_data['user_review_counts']), processed_data['num_users']]
    colors2 = ['#9b59b6', '#1abc9c']
    
    bars2 = ax2.bar(user_stages, user_values, color=colors2, edgecolor='black', linewidth=1.2)
    ax2.set_ylabel('Count')
    ax2.set_title('👥 Data Pipeline: Users', fontsize=13, fontweight='bold')
    for bar, val in zip(bars2, user_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
                f'{val:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax2.set_ylim(0, max(user_values) * 1.15)
    
    # 3. Item Review Distribution (Before)
    ax3 = fig.add_subplot(3, 2, 3)
    
    item_counts = raw_data['item_review_counts']
    ax3.hist(item_counts, bins=50, color='#3498db', edgecolor='white', alpha=0.8)
    ax3.axvline(x=5, color='red', linestyle='--', linewidth=2, label='Threshold=5')
    ax3.set_xlabel('Number of Reviews per Item')
    ax3.set_ylabel('Number of Items')
    ax3.set_title('Items: Review Count Distribution (Raw)', fontsize=12, fontweight='bold')
    ax3.set_xlim(0, 50)
    ax3.legend()
    
    below_5 = sum(1 for x in item_counts if x < 5)
    above_5 = sum(1 for x in item_counts if x >= 5)
    ax3.text(0.95, 0.95, f'< 5: {below_5:,}\n≥ 5: {above_5:,}', 
             transform=ax3.transAxes, ha='right', va='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 4. User Sequence Length (After)
    ax4 = fig.add_subplot(3, 2, 4)
    
    seq_lengths = processed_data['seq_lengths']
    ax4.hist(seq_lengths, bins=30, color='#1abc9c', edgecolor='white', alpha=0.8)
    ax4.axvline(x=np.mean(seq_lengths), color='red', linestyle='--', linewidth=2, 
                label=f'Mean={np.mean(seq_lengths):.1f}')
    ax4.set_xlabel('Sequence Length')
    ax4.set_ylabel('Number of Users')
    ax4.set_title('Users: Sequence Length Distribution (Processed)', fontsize=12, fontweight='bold')
    ax4.legend()
    
    # 5. Rating Distribution
    ax5 = fig.add_subplot(3, 2, 5)
    
    rating_dist = raw_data['rating_dist']
    ratings = sorted(rating_dist.keys())
    counts = [rating_dist[r] for r in ratings]
    colors5 = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(ratings)))
    
    bars5 = ax5.bar([str(r) for r in ratings], counts, color=colors5, edgecolor='black')
    ax5.set_xlabel('Rating')
    ax5.set_ylabel('Count')
    ax5.set_title('Rating Distribution', fontsize=12, fontweight='bold')
    for bar, val in zip(bars5, counts):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                f'{val:,}', ha='center', va='bottom', fontsize=9)
    
    # 6. Summary Table
    ax6 = fig.add_subplot(3, 2, 6)
    ax6.axis('off')
    
    summary_text = f"""
    ╔══════════════════════════════════════════════╗
    ║           DATA SUMMARY                    ║
    ╠══════════════════════════════════════════════╣
    ║  RAW DATA                                    ║
    ║    Total lines:        {raw_data['stats']['total_lines']:>15,}   ║
    ║    Valid products:     {raw_data['stats']['valid_products']:>15,}   ║
    ║    Skipped errors:     {raw_data['stats']['skipped_errors']:>15,}   ║
    ║    Skipped duplicates: {raw_data['stats']['skipped_duplicates']:>15,}   ║
    ║    Total reviews:      {raw_data['stats']['total_reviews']:>15,}   ║
    ╠══════════════════════════════════════════════╣
    ║  PROCESSED DATA (min_item=5, min_user=3)     ║
    ║    Items:              {processed_data['num_items']:>15,}   ║
    ║    Users:              {processed_data['num_users']:>15,}   ║
    ║    Interactions:       {processed_data['total_interactions']:>15,}   ║
    ║    Avg seq length:     {np.mean(seq_lengths):>15.1f}   ║
    ╚══════════════════════════════════════════════╝
    """
    
    ax6.text(0.5, 0.5, summary_text, transform=ax6.transAxes, 
             fontsize=11, fontfamily='monospace',
             ha='center', va='center',
             bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#dee2e6'))
    
    plt.tight_layout()
    
    # Save
    output_path = output_dir / 'data_statistics.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir / 'data_statistics.pdf', bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    
    plt.show()
    
    return output_path


def main():
    print("=" * 60)
    print("Data Statistics & Visualization")
    print("=" * 60)
    
    # Load raw stats
    raw_data = load_raw_stats()
    
    # Load processed stats
    try:
        processed_data = load_processed_stats()
    except FileNotFoundError:
        print("Processed data not found. Run preprocessing.py first.")
        processed_data = {
            'num_items': 0, 'num_users': 0, 
            'seq_lengths': [], 'item_frequencies': [],
            'total_interactions': 0
        }
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nITEMS:")
    print(f"   Raw valid products:    {raw_data['stats']['valid_products']:,}")
    print(f"   After filter (≥5):     {processed_data['num_items']:,}")
    print(f"   Removed:               {raw_data['stats']['valid_products'] - processed_data['num_items']:,}")
    
    print(f"\n👥 USERS:")
    print(f"   Raw unique users:      {len(raw_data['user_review_counts']):,}")
    print(f"   After filter (≥3):     {processed_data['num_users']:,}")
    
    print(f"\nINTERACTIONS:")
    print(f"   Raw total reviews:     {raw_data['stats']['total_reviews']:,}")
    print(f"   After processing:      {processed_data['total_interactions']:,}")
    
    # Create visualizations
    output_path = create_visualizations(raw_data, processed_data)
    
    print("\nDone!")
    return output_path


if __name__ == "__main__":
    main()
