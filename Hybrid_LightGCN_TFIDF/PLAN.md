# Hybrid LightGCN + TF-IDF Recommendation System

## 1. Brief (Tổng quan)

**Mục tiêu**: Xây dựng hệ thống gợi ý **Hybrid** kết hợp 2 phương pháp:
- **LightGCN** (Collaborative Filtering) - Học từ pattern tương tác user-item  
- **TF-IDF** (Content-Based) - Tính similarity dựa trên nội dung sản phẩm

**Công thức fusion**:
```
hybrid_score = α × LightGCN_score + (1-α) × TF-IDF_score
```
Với `α = 0.7` (default) ưu tiên LightGCN.

**Lý do kết hợp**:

| Method | Ưu điểm | Nhược điểm |
|--------|---------|------------|
| LightGCN | Personalized, capture user preference | Cold-start problem |
| TF-IDF | Không cold-start, nhanh | Không personalization |
| **Hybrid** | Giải quyết cả 2 vấn đề | - |

---

## 2. Method (Phương pháp chi tiết)

### 2.1 LightGCN Component

**Công thức Graph Convolution**:
```
E^(k+1) = D^(-1/2) A D^(-1/2) E^(k)
```

**Layer Combination**:
```
Final_embedding = mean(E^0, E^1, ..., E^K)
```

**Prediction**:
```
CF_score = dot(user_emb, item_emb)
```

- **Input**: User-Item interaction graph (bipartite graph)
- **Output**: User/Item embeddings để tính compatibility score
- **Loss**: BPR (Bayesian Personalized Ranking)

### 2.2 TF-IDF Component

**Công thức TF-IDF**:
```
TF-IDF(t, d) = TF(t, d) × IDF(t)

Where:
  TF(t, d) = frequency of term t in document d
  IDF(t) = log(N / df(t))
  N = total documents
  df(t) = documents containing term t
```

**Similarity Score**:
```
CB_score = cosine_similarity(user_profile, item_vector)
```

- **Input**: Item text (name + description)
- **Output**: Item vectors → Cosine similarity
- **User profile**: Average TF-IDF của items user đã interact

### 2.3 Hybrid Fusion Strategy

**Weighted Score Fusion**:
```python
def hybrid_recommend(user_idx, user_history, alpha=0.7):
    # 1. Get CF scores from LightGCN
    cf_scores = lightgcn.get_all_scores(user_idx)
    
    # 2. Get CB scores from TF-IDF (based on user history)
    cb_scores = tfidf.get_similarity_scores(user_history)
    
    # 3. Min-Max Normalize to [0, 1]
    cf_norm = (cf_scores - min) / (max - min)
    cb_norm = (cb_scores - min) / (max - min)
    
    # 4. Weighted fusion
    hybrid_scores = alpha * cf_norm + (1-alpha) * cb_norm
    
    return top_k(hybrid_scores)
```

---

## 3. Reference Models/Papers

### 3.1 LightGCN

| Attribute | Value |
|-----------|-------|
| **Paper** | *LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation* |
| **Authors** | Xiangnan He, Kuan Deng, Xiang Wang, Yan Li, Yongdong Zhang, Meng Wang |
| **Venue** | SIGIR 2020 |
| **ArXiv** | https://arxiv.org/abs/2002.02126 |
| **GitHub** | https://github.com/gusye1234/LightGCN-PyTorch |

**BibTeX**:
```bibtex
@inproceedings{he2020lightgcn,
  title={LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation},
  author={He, Xiangnan and Deng, Kuan and Wang, Xiang and Li, Yan and Zhang, Yongdong and Wang, Meng},
  booktitle={Proceedings of the 43rd International ACM SIGIR Conference},
  pages={639--648},
  year={2020}
}
```

### 3.2 TF-IDF

| Attribute | Value |
|-----------|-------|
| **Method** | Term Frequency-Inverse Document Frequency |
| **Origin** | Classic Information Retrieval (Salton & McGill, 1983) |
| **Implementation** | scikit-learn `TfidfVectorizer` |
| **Documentation** | https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html |

### 3.3 Hybrid Recommender Systems

| Attribute | Value |
|-----------|-------|
| **Paper** | *Hybrid Recommender Systems: Survey and Experiments* |
| **Authors** | Robin Burke |
| **Venue** | User Modeling and User-Adapted Interaction (2002) |
| **Link** | https://link.springer.com/article/10.1023/A:1021240730564 |

---

## 4. Data Requirements (Dữ liệu cần dùng)

### 4.1 Pre-processed Data (Hiện có)

| File | Size | Mô tả | Component |
|------|------|-------|-----------|
| `data/processed/lightgcn_processed.pkl` | ~25MB | User-item mappings, adj matrix | LightGCN |
| `data/processed/tfidf_processed.pkl` | ~326MB | Item info, mappings | TF-IDF |
| `data/processed/tfidf_matrix.npz` | ~329MB | Sparse TF-IDF matrix | TF-IDF |
| `LightGCN/checkpoints/best_model.pt` | - | Trained LightGCN weights | LightGCN |

### 4.2 Data Structure

**LightGCN Data**:
```python
{
    'user2idx': Dict[int, int],     # user_id -> index
    'idx2user': Dict[int, int],     # index -> user_id
    'item2idx': Dict[int, int],     # item_id -> index
    'idx2item': Dict[int, int],     # index -> item_id
    'adj_matrix': scipy.sparse.csr_matrix,  # Normalized adjacency
    'train_interactions': List[Tuple[int, int]],  # (user_idx, item_idx)
    'num_users': int,
    'num_items': int,
}
```

**TF-IDF Data**:
```python
{
    'tfidf_matrix': scipy.sparse.csr_matrix,  # (num_items, vocab_size)
    'item2idx': Dict[int, int],   # item_id -> index
    'idx2item': Dict[int, int],   # index -> item_id
    'item_info': Dict[int, dict], # item_id -> {name, category, price, ...}
}
```

---

## 5. Implementation Plan

### 5.1 Folder Structure

```
Hybrid_LightGCN_TFIDF/
├── __init__.py           # Package init
├── config.py             # HybridConfig (paths, alpha)
├── model.py              # HybridRecommender class
├── demo.py               # DemoRecommender interface
├── evaluate.py           # Evaluation metrics
├── pipeline.ipynb        # Demo notebook
└── README.md             # Documentation
```

### 5.2 Implementation Steps

#### Step 1: Config (`config.py`)
```python
@dataclass
class HybridConfig:
    # Paths to pre-trained models
    lightgcn_checkpoint: str = "../LightGCN/checkpoints/best_model.pt"
    lightgcn_data: str = "../data/processed/lightgcn_processed.pkl"
    tfidf_cache: str = "../data/processed/tfidf_processed.pkl"
    tfidf_matrix: str = "../data/processed/tfidf_matrix.npz"
    
    # Fusion weights
    alpha: float = 0.7  # LightGCN weight (1-alpha for TF-IDF)
    
    # Top-K
    top_k: int = 10
```

#### Step 2: HybridRecommender (`model.py`)
```python
class HybridRecommender:
    def __init__(self, lightgcn_model, tfidf_recommender, config):
        self.lightgcn = lightgcn_model
        self.tfidf = tfidf_recommender
        self.alpha = config.alpha
    
    def _normalize(self, scores):
        """Min-max normalization to [0, 1]"""
        min_s, max_s = scores.min(), scores.max()
        if max_s == min_s:
            return np.zeros_like(scores)
        return (scores - min_s) / (max_s - min_s)
    
    def recommend(self, user_idx, user_history, top_k=10):
        # Get LightGCN scores for all items
        cf_scores = self.lightgcn.get_all_item_scores(user_idx)
        cf_norm = self._normalize(cf_scores)
        
        # Get TF-IDF scores based on user history
        cb_scores = self.tfidf.get_user_profile_scores(user_history)
        cb_norm = self._normalize(cb_scores)
        
        # Weighted fusion
        hybrid_scores = self.alpha * cf_norm + (1 - self.alpha) * cb_norm
        
        # Exclude already interacted items
        for item_idx in user_history:
            hybrid_scores[item_idx] = -1
        
        # Get top-K
        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
        
        return top_indices, hybrid_scores[top_indices]
```

#### Step 3: Evaluation (`evaluate.py`)
```python
def evaluate_hybrid(hybrid_model, test_data, alpha_values=[0.0, 0.3, 0.5, 0.7, 1.0]):
    results = {}
    
    for alpha in alpha_values:
        hybrid_model.set_alpha(alpha)
        
        hits, ndcgs = [], []
        for user_idx, user_history, ground_truth in test_data:
            recommendations = hybrid_model.recommend(user_idx, user_history, top_k=10)
            
            # Calculate HR@10
            hit = 1 if ground_truth in recommendations else 0
            hits.append(hit)
            
            # Calculate NDCG@10
            if ground_truth in recommendations:
                rank = list(recommendations).index(ground_truth)
                ndcgs.append(1 / np.log2(rank + 2))
            else:
                ndcgs.append(0)
        
        results[alpha] = {
            'HR@10': np.mean(hits),
            'NDCG@10': np.mean(ndcgs)
        }
    
    return results
```

---

## 6. Verification Plan

### 6.1 Automated Tests

```bash
# Test hybrid model with different alpha values
cd Hybrid_LightGCN_TFIDF
python evaluate.py --sample-users 5000 --alpha 0.7

# Compare with baselines
python evaluate.py --alpha 1.0   # Pure LightGCN
python evaluate.py --alpha 0.0   # Pure TF-IDF
```

### 6.2 Expected Metrics

| Method | Alpha | HR@10 | NDCG@10 |
|--------|-------|-------|---------|
| TF-IDF only | 0.0 | ~6% | ~3.6% |
| LightGCN only | 1.0 | ~8-10% | ~4-5% |
| **Hybrid** | **0.7** | **~10-12%** | **~5-6%** |

---

## 7. Implementation Checklist

- [ ] Create folder `Hybrid_LightGCN_TFIDF/`
- [ ] Implement `__init__.py`
- [ ] Implement `config.py` with HybridConfig
- [ ] Implement `model.py` with HybridRecommender
- [ ] Implement `demo.py` with DemoRecommender
- [ ] Implement `evaluate.py`
- [ ] Create `README.md`
- [ ] Create `pipeline.ipynb` notebook demo
- [ ] Test and verify metrics
