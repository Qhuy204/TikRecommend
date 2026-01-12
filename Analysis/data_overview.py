# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Dataset Overview
# Phân tích tổng quan dataset Tiki E-commerce

# %%
import sys
sys.path.insert(0, '..')

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12

# %% [markdown]
# ## 1. Load Data

# %%
# Load LightGCN processed data
with open('../data/processed/lightgcn_processed.pkl', 'rb') as f:
    data = pickle.load(f)

print("=== Dataset Statistics ===")
print(f"Users: {data['num_users']:,}")
print(f"Items: {data['num_items']:,}")
print(f"Interactions: {sum(len(v) for v in data['user_pos_items'].values()):,}")

# %% [markdown]
# ## 2. User Interaction Distribution

# %%
# Count interactions per user
user_interactions = [len(items) for items in data['user_pos_items'].values()]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
axes[0].hist(user_interactions, bins=50, edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Number of Interactions')
axes[0].set_ylabel('Number of Users')
axes[0].set_title('User Interaction Distribution')
axes[0].axvline(np.mean(user_interactions), color='r', linestyle='--', label=f'Mean: {np.mean(user_interactions):.1f}')
axes[0].axvline(np.median(user_interactions), color='g', linestyle='--', label=f'Median: {np.median(user_interactions):.1f}')
axes[0].legend()

# Box plot
axes[1].boxplot(user_interactions, vert=True)
axes[1].set_ylabel('Number of Interactions')
axes[1].set_title('User Interaction Box Plot')

plt.tight_layout()
plt.savefig('user_interaction_dist.png', dpi=150, bbox_inches='tight')
plt.show()

# Statistics
print("\n=== User Interaction Statistics ===")
print(f"Mean: {np.mean(user_interactions):.2f}")
print(f"Median: {np.median(user_interactions):.2f}")
print(f"Std: {np.std(user_interactions):.2f}")
print(f"Min: {min(user_interactions)}")
print(f"Max: {max(user_interactions)}")
print(f"25th percentile: {np.percentile(user_interactions, 25):.0f}")
print(f"75th percentile: {np.percentile(user_interactions, 75):.0f}")

# %% [markdown]
# ## 3. Item Popularity Distribution

# %%
# Count how many users interacted with each item
item_popularity = Counter()
for items in data['user_pos_items'].values():
    for item in items:
        item_popularity[item] += 1

popularity_counts = list(item_popularity.values())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram (log scale)
axes[0].hist(popularity_counts, bins=50, edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Number of Users')
axes[0].set_ylabel('Number of Items')
axes[0].set_title('Item Popularity Distribution')
axes[0].set_yscale('log')

# Top items
top_items = item_popularity.most_common(20)
items_idx = [str(x[0]) for x in top_items]
items_count = [x[1] for x in top_items]

axes[1].barh(range(len(items_idx)), items_count, color='steelblue')
axes[1].set_yticks(range(len(items_idx)))
axes[1].set_yticklabels([f"Item {i}" for i in items_idx])
axes[1].set_xlabel('Number of Interactions')
axes[1].set_title('Top 20 Most Popular Items')
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig('item_popularity_dist.png', dpi=150, bbox_inches='tight')
plt.show()

# Statistics
print("\n=== Item Popularity Statistics ===")
print(f"Mean: {np.mean(popularity_counts):.2f}")
print(f"Median: {np.median(popularity_counts):.2f}")
print(f"Items with 1 interaction: {sum(1 for c in popularity_counts if c == 1):,}")
print(f"Items with >100 interactions: {sum(1 for c in popularity_counts if c > 100):,}")

# %% [markdown]
# ## 4. Category Analysis

# %%
# Load item info
item_info = data.get('item_info', {})

if item_info:
    categories = [info.get('category', 'Unknown') for info in item_info.values()]
    category_counts = Counter(categories)
    
    # Top categories
    top_cats = category_counts.most_common(15)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    cats = [x[0][:30] for x in top_cats]
    counts = [x[1] for x in top_cats]
    
    bars = ax.barh(range(len(cats)), counts, color='coral')
    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels(cats)
    ax.set_xlabel('Number of Items')
    ax.set_title('Top 15 Categories by Item Count')
    ax.invert_yaxis()
    
    # Add value labels
    for i, (bar, count) in enumerate(zip(bars, counts)):
        ax.text(count + 50, i, f'{count:,}', va='center')
    
    plt.tight_layout()
    plt.savefig('category_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nTotal categories: {len(category_counts)}")
else:
    print("Item info not available")

# %% [markdown]
# ## 5. Price Distribution

# %%
if item_info:
    prices = [info.get('price', 0) for info in item_info.values() if info.get('price', 0) > 0]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram (log scale for better visualization)
    axes[0].hist(prices, bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Price (VND)')
    axes[0].set_ylabel('Number of Items')
    axes[0].set_title('Price Distribution')
    axes[0].set_xscale('log')
    
    # Price ranges
    price_ranges = [
        ('< 100K', sum(1 for p in prices if p < 100000)),
        ('100K-500K', sum(1 for p in prices if 100000 <= p < 500000)),
        ('500K-1M', sum(1 for p in prices if 500000 <= p < 1000000)),
        ('1M-5M', sum(1 for p in prices if 1000000 <= p < 5000000)),
        ('> 5M', sum(1 for p in prices if p >= 5000000)),
    ]
    
    labels = [x[0] for x in price_ranges]
    values = [x[1] for x in price_ranges]
    
    axes[1].pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
    axes[1].set_title('Price Range Distribution')
    
    plt.tight_layout()
    plt.savefig('price_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n=== Price Statistics ===")
    print(f"Mean: {np.mean(prices):,.0f} VND")
    print(f"Median: {np.median(prices):,.0f} VND")
    print(f"Min: {min(prices):,.0f} VND")
    print(f"Max: {max(prices):,.0f} VND")

# %% [markdown]
# ## 6. Summary Table

# %%
summary_data = {
    'Metric': ['Total Users', 'Total Items', 'Total Interactions', 
               'Avg Interactions/User', 'Avg Interactions/Item',
               'Sparsity (%)'],
    'Value': [
        f"{data['num_users']:,}",
        f"{data['num_items']:,}",
        f"{sum(len(v) for v in data['user_pos_items'].values()):,}",
        f"{np.mean(user_interactions):.2f}",
        f"{np.mean(popularity_counts):.2f}",
        f"{(1 - sum(len(v) for v in data['user_pos_items'].values()) / (data['num_users'] * data['num_items'])) * 100:.4f}"
    ]
}

summary_df = pd.DataFrame(summary_data)
print("\n=== Dataset Summary ===")
print(summary_df.to_string(index=False))

# Save summary
summary_df.to_csv('dataset_summary.csv', index=False)
print("\nSaved to dataset_summary.csv")
