# ContentBased_PhoBERT

**PhoBERT Content-Based Recommender** - Vietnamese BERT for semantic text understanding.

## 📚 Paper References

| Paper | Details |
|-------|---------|
| **PhoBERT** | Nguyen & Nguyen (2020). *"PhoBERT: Pre-trained language models for Vietnamese"*. Findings of EMNLP 2020. [arXiv:2003.00744](https://arxiv.org/abs/2003.00744) |
| **BERT** | Devlin et al. (2019). *"BERT: Pre-training of Deep Bidirectional Transformers"*. NAACL 2019. |

## 📊 Method Overview

| Aspect | Value |
|--------|-------|
| **Type** | Content-based (Text) |
| **Input** | Item name + short_description |
| **Algorithm** | PhoBERT embeddings + Cosine Similarity |
| **Embedding** | 768-dimensional vectors |
| **Training** | No training, pre-trained encoder |
| **Cold-start** | ✅ Text only |

## 📈 Data Statistics (ALL items)

| Metric | Value |
|--------|-------|
| Items | 123,016 |
| Users | 104,796 |
| Interactions | 581,357 |
| Embeddings | 123K x 768 |
| File size | ~180 MB |

## 🚀 Quick Start

```bash
# 1. Preprocessing (encodes all items with PhoBERT)
# ⚠️ Takes ~10-15 minutes on GPU
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

demo.recommend_user(12345)           # Based on liked items
demo.similar_items(277725874)        # Find similar items
demo.recommend_sequence([1, 2, 3])   # Cold-start
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
| `config.py` | PhoBERT settings, data paths |
| `preprocessing.py` | Data loading, PhoBERT encoding |
| `model.py` | PhoBERTRecommender class |
| `demo.py` | DemoRecommender interface |
| `evaluate.py` | Evaluation with HR@K, NDCG@K, MRR |

## ⚙️ Configuration

```python
PhoBERTConfig:
    model_name: "vinai/phobert-base"
    max_length: 256         # Max tokens
    embedding_dim: 768      # Hidden size
    batch_size: 32
```

## ⚠️ Requirements

- GPU recommended for preprocessing (~10-15 min)
- `transformers` library
- ~180MB for embeddings file (123K x 768 x float16)
