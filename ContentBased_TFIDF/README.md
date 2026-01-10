# ContentBased_TFIDF

**TF-IDF Content-Based Recommender** - Classical content-based filtering using text similarity.

## 📚 Paper References

| Paper | Details |
|-------|---------|
| **IDF Original** | Spärck Jones, K. (1972). *"A statistical interpretation of term specificity and its application in retrieval"*. Journal of Documentation. |
| **Content-Based Filtering** | Lops, P., et al. (2011). *"Content-based Recommender Systems: State of the Art and Trends"*. Recommender Systems Handbook. |

## 📊 Method Overview

| Aspect | Value |
|--------|-------|
| **Type** | Content-based (Text) |
| **Input** | Item name + short_description |
| **Algorithm** | TF-IDF + Cosine Similarity |
| **Training** | No training, just fit() |
| **Cold-start** | ✅ Text only |

## 📈 Data Statistics (ALL items)

| Metric | Value |
|--------|-------|
| Items | 123,016 |
| Users | 104,796 |
| Interactions | 581,357 |
| TF-IDF Matrix | 123K x 15K |
| File size | ~58 MB |

## 🚀 Quick Start

```bash
# 1. Preprocessing (creates TF-IDF matrix)
python preprocessing.py

# 2. Run demo
python demo.py

# 3. Evaluate
python evaluate.py
```

## 📖 Usage

```python
from demo import DemoRecommender

demo = DemoRecommender()

# Recommend for user (based on liked items)
demo.recommend_user(12345)

# Find similar items
demo.similar_items(277725874)

# Recommend from item list (cold-start)
demo.recommend_sequence([item1, item2, item3])

# Random demo
demo.random_demo()
```

## 📊 Evaluation

```bash
python evaluate.py --test-ratio 0.2 --seed 42
```

**Metrics:**
- **HR@K** - Hit Rate at K
- **NDCG@K** - Normalized Discounted Cumulative Gain
- **MRR** - Mean Reciprocal Rank
- **CategoryPrecision** - % similar items in same category

## 📁 Files

| File | Description |
|------|-------------|
| `config.py` | Configuration (TF-IDF params, data paths) |
| `preprocessing.py` | Data loading, cleaning, TF-IDF fitting |
| `model.py` | TFIDFRecommender class |
| `demo.py` | DemoRecommender interface |
| `evaluate.py` | Evaluation with HR@K, NDCG@K, MRR |

## ⚙️ Configuration

```python
TFIDFConfig:
    max_features: 15000   # Vocabulary size
    ngram_range: (1, 2)   # Unigrams + bigrams
    max_text_length: 500  # Max chars
```
