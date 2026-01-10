# CL4SRec: Contrastive Learning for Sequential Recommendation

**Paper**: "Contrastive Learning for Sequential Recommendation" (WWW 2022)

---

## 🚀 Quick Start

```bash
cd Newmethod

# Step 1: Preprocess (run once, ~2 min)
python preprocessing.py

# Step 2: Train (~10 min for 30 epochs)
python train.py --epochs 30 --cl_weight 0.1

# Step 3: Evaluate
python evaluate.py --split test

# Step 4: Demo
python demo.py
```

---

## 📊 Demo Usage

```python
from demo import DemoRecommender

# Load once
demo = DemoRecommender()

# === Recommend cho user ===
demo.recommend_user(21614396)  # User cụ thể
demo.recommend_user()          # Random user

# === Items tương tự ===
demo.similar_items(277725874)  # Item cụ thể  
demo.similar_items()           # Random item

# === Session-based (cold-start) ===
items = demo.get_sample_items(3)
demo.recommend_sequence(items)

# === Random demo ===
demo.random_demo(top_k=5)

# === Lấy sample IDs ===
demo.get_sample_users(10)
demo.get_sample_items(10)
```

---

## 📁 Files

| File | Mô tả |
|------|-------|
| `preprocessing.py` | Xử lý data, lưu cache pickle |
| `train.py` | Training với BPR + Contrastive loss |
| `evaluate.py` | Tính HR@K, NDCG@K, MRR |
| `demo.py` | Interactive demo |
| `data_stats.py` | Visualize thống kê data |

---

## 📈 Results

| Metric | @5 | @10 | @20 |
|--------|-----|-----|-----|
| HR | 0.046 | 0.076 | 0.121 |
| NDCG | 0.029 | 0.038 | 0.050 |

---

## 🔬 Method

```
Loss = L_bpr + λ × L_contrastive

Augmentation: Crop (60%), Mask (20%), Reorder (20%)
λ = 0.1 (default)
```

---

## ⚙️ Config

Sửa trong `config.py`:

```python
min_item_count = 5      # Items có ≥5 reviews
min_seq_length = 3      # Users có ≥3 interactions
max_seq_length = 50     # Max sequence length
```
