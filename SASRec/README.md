# SASRec + PhoBERT Fusion Recommendation System

**Self-Attentive Sequential Recommendation với Vietnamese NLP Enhancement**

> Modern Deep Learning approach (2018, updated 2023+) cho Vietnamese E-commerce Recommendation

---

## � Table of Contents

1. [Giới thiệu](#-giới-thiệu)
2. [Lý thuyết SASRec](#-lý-thuyết-sasrec)
3. [Kiến trúc Model](#-kiến-trúc-model)
4. [Loss Function](#-loss-function-bpr-loss)
5. [PhoBERT Integration](#-phobert-integration)
6. [Hướng dẫn sử dụng](#-hướng-dẫn-sử-dụng)
7. [References](#-references)

---

## 🎯 Giới thiệu

### Problem Statement

Trong E-commerce, việc recommend sản phẩm dựa trên **lịch sử hành vi theo thứ tự thời gian** (sequential behavior) rất quan trọng:

```
User clicks: Item_1 → Item_2 → Item_3 → ?
                                         ↓
                              Predict: Item_4
```

### Why SASRec?

| Method | Year | Approach | Limitation |
|--------|------|----------|------------|
| Matrix Factorization | 2009 | Static latent factors | No temporal patterns |
| GRU4Rec | 2016 | RNN-based | Limited long-range |
| Caser | 2018 | CNN-based | Fixed window size |
| **SASRec** | **2018** | **Self-Attention** | **State-of-the-art** |
| BERT4Rec | 2019 | Bidirectional | More complex |

**SASRec advantages:**
- ✅ Captures **long-range dependencies** với O(1) path length
- ✅ **Parallelizable** - không sequential như RNN
- ✅ **Lightweight** hơn BERT4Rec
- ✅ Proven performance trên nhiều benchmarks

---

## � Lý thuyết SASRec

### Core Idea

**Self-Attentive Sequential Recommendation** sử dụng Transformer self-attention để model user behavior sequences:

```
Traditional: User history → RNN/LSTM → Hidden state → Predict next
SASRec:      User history → Self-Attention → Context-aware repr → Predict next
```

### Key Components

#### 1. Item Embedding

Mỗi item được biểu diễn bởi một trainable embedding vector:

$$\mathbf{E} \in \mathbb{R}^{|V| \times d}$$

Trong đó:
- $|V|$: Số lượng items (vocabulary size)
- $d$: Embedding dimension

#### 2. Positional Encoding

Vì self-attention không có notion về order, cần thêm positional information:

$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d})$$
$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d})$$

Sinusoidal encoding cho phép model học relative positions.

#### 3. Self-Attention Mechanism

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Trong đó:
- $Q = XW^Q$ (Query)
- $K = XW^K$ (Key)  
- $V = XW^V$ (Value)
- $\sqrt{d_k}$: Scaling factor để prevent gradient vanishing

#### 4. Causal Masking

**Critical!** Để prevent **information leakage** từ future items:

```
Sequence: [A, B, C, D, E]

Without mask:  A can see B, C, D, E  ❌ (cheating!)
With mask:     A can only see A      ✅
               B can see A, B        ✅
               C can see A, B, C     ✅
```

Mask được implement bằng:
```python
causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
scores = scores.masked_fill(causal_mask.bool(), -1e9)
```

#### 5. Multi-Head Attention

Cho phép model attend to different representation subspaces:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

Trong đó: $\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$

#### 6. Feed-Forward Network

Position-wise FFN sau attention:

$$\text{FFN}(x) = \text{GELU}(xW_1 + b_1)W_2 + b_2$$

#### 7. Residual Connections & Layer Norm

```python
x = x + dropout(attention(layer_norm(x)))
x = x + dropout(ffn(layer_norm(x)))
```

---

## 🏗 Kiến trúc Model

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    USER BEHAVIOR SEQUENCE                    │
│              [Item_1, Item_2, Item_3, ..., Item_n]          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    ITEM EMBEDDING LAYER                      │
│                    E ∈ ℝ^(|V|+1 × d)                        │
│                    (+1 for padding token)                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  POSITIONAL ENCODING                         │
│                  PE ∈ ℝ^(max_len × d)                       │
│                  Sinusoidal encoding                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              TRANSFORMER BLOCK × L (default: 2)              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Layer Norm → Multi-Head Self-Attention → Dropout      │ │
│  │              (with causal mask)                         │ │
│  │                      ↓                                  │ │
│  │  Residual Connection (+)                                │ │
│  │                      ↓                                  │ │
│  │  Layer Norm → Feed-Forward Network → Dropout            │ │
│  │                      ↓                                  │ │
│  │  Residual Connection (+)                                │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    FINAL LAYER NORM                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              GET LAST HIDDEN STATE                           │
│       h_n = hidden[last_valid_position]                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   PREDICTION LAYER                           │
│           score(item) = h_n · e_item                        │
│           (dot product with item embeddings)                │
└─────────────────────────────────────────────────────────────┘
```

### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `embedding_dim` | 64 | Item embedding dimension |
| `num_attention_heads` | 2 | Number of attention heads |
| `num_transformer_blocks` | 2 | Number of transformer layers |
| `hidden_dim` | 128 | FFN hidden dimension |
| `max_seq_length` | 50 | Maximum sequence length |
| `dropout` | 0.2 | Dropout rate |

---

## 📐 Loss Function: BPR Loss

### Bayesian Personalized Ranking (BPR)

SASRec sử dụng **BPR Loss** cho implicit feedback learning:

$$\mathcal{L}_{BPR} = -\sum_{(u,i,j) \in D_S} \ln \sigma(\hat{x}_{ui} - \hat{x}_{uj})$$

Trong đó:
- $(u, i, j)$: User $u$, positive item $i$, negative item $j$
- $\hat{x}_{ui}$: Predicted score cho positive item
- $\hat{x}_{uj}$: Predicted score cho negative item  
- $\sigma$: Sigmoid function

### Intuition

BPR loss tối ưu hóa **pairwise ranking**:
- Positive item (user đã interact) nên có score **cao hơn** negative item
- Margin: $\hat{x}_{ui} - \hat{x}_{uj} > 0$

```
Positive: Item user clicked     → score = 0.8
Negative: Random sampled item   → score = 0.3
                                        ↓
BPR: maximize sigmoid(0.8 - 0.3) = sigmoid(0.5) ≈ 0.62
Loss = -log(0.62) ≈ 0.48
```

### Implementation

```python
def bpr_loss(pos_scores, neg_scores):
    """
    pos_scores: (batch_size,)
    neg_scores: (batch_size, num_negatives)
    """
    # Difference between positive and negative scores
    diff = pos_scores.unsqueeze(1) - neg_scores
    
    # Clamp for numerical stability
    diff = diff.clamp(-80, 80)
    
    # BPR loss
    loss = -torch.log(torch.sigmoid(diff) + 1e-10).mean()
    
    return loss
```

### Negative Sampling

Mỗi positive sample có $k$ negative samples (default: $k=4$):

```python
def sample_negatives(positive_item, all_items, k=4):
    negatives = []
    while len(negatives) < k:
        neg = random.choice(all_items)
        if neg != positive_item:
            negatives.append(neg)
    return negatives
```

---

## 🇻🇳 PhoBERT Integration

### Why PhoBERT?

- Pre-trained **Vietnamese language model** từ VinAI
- Hiểu semantic meaning của Vietnamese product descriptions
- Cold-start handling: New items có thể được represent qua text

### Architecture (Optional Enhancement)

```
                    ┌─────────────────┐
User Sequence  ──→  │    SASRec       │ ──→ Sequence Repr
                    └─────────────────┘
                              │
                              ▼ (Fusion)
                    ┌─────────────────┐
Item Text     ──→   │    PhoBERT      │ ──→ Content Repr
(Vietnamese)        │   (Frozen)      │
                    └─────────────────┘
```

### Fusion Strategies

1. **Concat**: `fused = Linear(concat(seq, content))`
2. **Gate**: `fused = gate * seq + (1-gate) * content`
3. **Addition**: `fused = proj(seq) + proj(content)`

---

## 🚀 Hướng dẫn sử dụng

### Quick Start

```bash
cd Newmethod

# Train (30 epochs, ~20-30 min on RTX 3060)
python main.py --mode train --epochs 30

# Quick test (2 epochs)
python main.py --mode train --epochs 2

# Evaluate
python main.py --mode evaluate

# Demo recommendations
python main.py --mode demo --user_id 28013
```

### API Usage

```python
from recommender import SASRecRecommender
from data_processor import TikiDataProcessor

# Load
processor = TikiDataProcessor()
processor.load_raw_data()

recommender = SASRecRecommender.load(
    'checkpoints/best_model.pt',
    processor
)

# Get recommendations for user
recs = recommender.recommend_for_user(user_id=12345, top_k=10)
for r in recs:
    print(f"{r['name']} - {r['price']} VND")

# Find similar items
similar = recommender.get_similar_items(item_id=277725874, top_k=5)
```

### Project Structure

```
Newmethod/
├── main.py           # Entry point (train/eval/demo)
├── config.py         # Configuration dataclasses
├── data_processor.py # Load tiki_dataset.jsonl
├── models.py         # SASRec + PhoBERT architecture
├── trainer.py        # Training pipeline
├── recommender.py    # Inference interface
└── checkpoints/      # Saved models
```

---

## 📊 Expected Results

| Metric | 10 epochs | 30 epochs |
|--------|-----------|-----------|
| HR@10 | ~0.08-0.12 | ~0.15-0.25 |
| NDCG@10 | ~0.04-0.08 | ~0.10-0.18 |
| Training Time | ~8 min | ~25 min |

---

## 📖 References

### Original Paper

1. **SASRec: Self-Attentive Sequential Recommendation**
   - Authors: Wang-Cheng Kang, Julian McAuley
   - Conference: ICDM 2018
   - Link: [https://arxiv.org/abs/1808.09781](https://arxiv.org/abs/1808.09781)
   - GitHub: [https://github.com/kang205/SASRec](https://github.com/kang205/SASRec)

### Related Papers

2. **Attention Is All You Need** (Transformer)
   - Authors: Vaswani et al.
   - Conference: NeurIPS 2017
   - Link: [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

3. **BPR: Bayesian Personalized Ranking from Implicit Feedback**
   - Authors: Rendle et al.
   - Conference: UAI 2009
   - Link: [https://arxiv.org/abs/1205.2618](https://arxiv.org/abs/1205.2618)

4. **BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer**
   - Authors: Sun et al.
   - Conference: CIKM 2019
   - Link: [https://arxiv.org/abs/1904.06690](https://arxiv.org/abs/1904.06690)

5. **PhoBERT: Pre-trained language models for Vietnamese**
   - Authors: Nguyen & Nguyen (VinAI)
   - Conference: EMNLP 2020
   - Link: [https://arxiv.org/abs/2003.00744](https://arxiv.org/abs/2003.00744)

### Implementations

- RecBole (PyTorch): [https://recbole.io/](https://recbole.io/)
- SASRec PyTorch: [https://github.com/pmixer/SASRec.pytorch](https://github.com/pmixer/SASRec.pytorch)

---

## 📝 Citation

```bibtex
@inproceedings{kang2018self,
  title={Self-Attentive Sequential Recommendation},
  author={Kang, Wang-Cheng and McAuley, Julian},
  booktitle={2018 IEEE International Conference on Data Mining (ICDM)},
  pages={197--206},
  year={2018},
  organization={IEEE}
}
```

---

## ⚙️ Requirements

- Python >= 3.8
- PyTorch >= 2.0
- transformers >= 4.30 (for PhoBERT)
- CUDA GPU recommended (tested on RTX 3060 12GB)

---

*Developed for Vietnamese E-commerce Recommendation System*
