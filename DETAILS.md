# Technical Details

Detailed explanations of model architectures, ideas, and loss functions.

---

## Table of Contents

1. [LightGCN](#1-lightgcn)
2. [SASRec](#2-sasrec)
3. [CL4SRec](#3-cl4srec)
4. [TF-IDF](#4-tf-idf)
5. [PhoBERT](#5-phobert)
6. [Hybrid](#6-hybrid-lightgcn--tf-idf)

---

## 1. LightGCN

**Paper**: He et al., "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation", SIGIR 2020

### Idea

LightGCN simplifies Graph Convolutional Networks for recommendation by removing:
- Feature transformation matrices (W)
- Nonlinear activation functions (ReLU)

Only keeping **neighborhood aggregation** for learning user/item embeddings from the user-item interaction graph.

### Architecture

```
User-Item Bipartite Graph
         ↓
[User Embeddings] [Item Embeddings]
         ↓
   GCN Layer 1: E^(1) = A · E^(0)
         ↓
   GCN Layer 2: E^(2) = A · E^(1)
         ↓
   GCN Layer 3: E^(3) = A · E^(2)
         ↓
   Layer Combination: E = (E^(0) + E^(1) + E^(2) + E^(3)) / 4
         ↓
   Prediction: score(u,i) = e_u · e_i
```

Where:
- `A`: Normalized adjacency matrix of user-item graph
- `E^(k)`: Embeddings after k-th layer propagation
- Final embedding is the mean of all layers

### Loss Function: BPR Loss

**Bayesian Personalized Ranking (BPR)** optimizes pairwise ranking:

```
L_BPR = -Σ log(σ(score(u,i+) - score(u,i-)))
```

Where:
- `(u, i+, i-)`: User u, positive item i+, negative item i-
- `σ`: Sigmoid function
- Goal: Positive items should score higher than negative items

**Intuition**: For each user, the model learns to rank items the user has interacted with higher than random items.

---

## 2. SASRec

**Paper**: Kang & McAuley, "Self-Attentive Sequential Recommendation", ICDM 2018

### Idea

Model user behavior as a **sequence** of items and use **self-attention** (Transformer) to capture:
- Long-range dependencies between items
- User's evolving preferences over time

### Architecture

```
User Sequence: [item_1, item_2, ..., item_n]
                      ↓
            Item Embedding + Position Encoding
                      ↓
         ┌─────────────────────────────┐
         │  Transformer Block × L      │
         │  - Multi-Head Self-Attention│
         │  - Causal Mask (no future)  │
         │  - Feed-Forward Network     │
         │  - Layer Norm + Residual    │
         └─────────────────────────────┘
                      ↓
            Last Hidden State h_n
                      ↓
         Prediction: score(i) = h_n · e_i
```

Key components:
- **Causal Masking**: Prevents attending to future items
- **Positional Encoding**: Sinusoidal encoding for sequence order
- **Multi-Head Attention**: Multiple attention heads for different patterns

### Loss Function: BPR Loss

Same as LightGCN, but applied to sequential prediction:

```
L_BPR = -Σ log(σ(score(next_item) - score(random_item)))
```

For each position in the sequence, predict the next item.

---

## 3. CL4SRec

**Paper**: Xie et al., "Contrastive Learning for Sequential Recommendation", WWW 2022

### Idea

Enhance SASRec with **Contrastive Learning** to:
- Learn more robust sequence representations
- Improve generalization via data augmentation
- Create self-supervised signals

### Architecture

```
Original Sequence: [A, B, C, D, E]
           ↓
   Data Augmentation
           ↓
   ┌───────────────┬───────────────┐
   │ View 1        │ View 2        │
   │ [A, _, C, D]  │ [B, C, _, E]  │
   │ (Mask)        │ (Mask)        │
   └───────────────┴───────────────┘
           ↓               ↓
      SASRec Encoder   SASRec Encoder
           ↓               ↓
         z_1             z_2
           ↓
   Contrastive Loss: Maximize similarity(z_1, z_2)
```

Augmentation strategies:
- **Crop**: Random subsequence
- **Mask**: Random item masking
- **Reorder**: Shuffle a portion

### Loss Function: Combined Loss

```
L_total = L_BPR + λ × L_CL
```

**Contrastive Loss (InfoNCE)**:
```
L_CL = -log(exp(sim(z_i, z_i') / τ) / Σ exp(sim(z_i, z_j) / τ))
```

Where:
- `z_i, z_i'`: Two augmented views of the same sequence
- `z_j`: Negative samples (other sequences in batch)
- `τ`: Temperature parameter
- `λ`: Contrastive weight (default: 0.1)

---

## 4. TF-IDF

**Classical Information Retrieval Method**

### Idea

Represent items as **text vectors** using TF-IDF (Term Frequency - Inverse Document Frequency), then find similar items via **cosine similarity**.

### Architecture

```
Item Text: "Sữa bột Frisolac Gold cho trẻ em"
                    ↓
           Text Preprocessing
           (Tokenize, Clean, N-grams)
                    ↓
           TF-IDF Vectorization
                    ↓
         [0.2, 0.0, 0.5, 0.1, ...]  (sparse vector)
                    ↓
           Cosine Similarity
                    ↓
         Similar Items Ranking
```

### TF-IDF Formula

```
TF-IDF(t, d) = TF(t, d) × IDF(t)

TF(t, d) = count(t in d) / total_terms(d)
IDF(t) = log(N / df(t))
```

Where:
- `TF`: How often term t appears in document d
- `IDF`: How rare term t is across all documents
- `N`: Total number of documents
- `df(t)`: Number of documents containing term t

### Similarity (No Training)

```
cosine_sim(A, B) = (A · B) / (||A|| × ||B||)
```

User profile = average of TF-IDF vectors of liked items.

---

## 5. PhoBERT

**Paper**: Nguyen & Nguyen, "PhoBERT: Pre-trained language models for Vietnamese", EMNLP 2020

### Idea

Use **pre-trained Vietnamese BERT** to create semantic embeddings for item descriptions, capturing meaning beyond keyword matching.

### Architecture

```
Item Text: "Sữa bột Frisolac Gold cho trẻ em"
                    ↓
          PhoBERT Tokenizer
                    ↓
          [CLS] Sữa bột Frisolac ... [SEP]
                    ↓
          PhoBERT Encoder (12 layers)
                    ↓
          [CLS] Token Embedding (768-dim)
                    ↓
          Cosine Similarity
                    ↓
          Similar Items Ranking
```

### Why PhoBERT?

- Pre-trained on 20GB Vietnamese text
- Understands Vietnamese semantics
- Word segmentation for Vietnamese
- No training required (use pre-trained embeddings)

### Similarity

Same as TF-IDF: cosine similarity on embeddings.

---

## 6. Hybrid (LightGCN + TF-IDF)

### Idea

Combine **Collaborative Filtering** (LightGCN) with **Content-Based** (TF-IDF) using weighted score fusion:

- CF captures user behavior patterns
- CB handles cold-start and content similarity
- Weighted fusion balances both signals

### Architecture

```
                    User + History
                         ↓
         ┌───────────────┴───────────────┐
         │                               │
    LightGCN                         TF-IDF
    (CF Score)                    (CB Score)
         │                               │
    Normalize                       Normalize
    [0, 1]                          [0, 1]
         │                               │
         └───────────────┬───────────────┘
                         ↓
              Weighted Fusion
         hybrid = α × CF + (1-α) × CB
                         ↓
                Top-K Recommendations
```

### Score Normalization

Min-Max normalization to [0, 1]:
```
normalized = (score - min) / (max - min)
```

### Fusion Formula

```
hybrid_score(i) = α × cf_score(i) + (1 - α) × cb_score(i)
```

Where:
- `α = 0.8` (optimal from tuning)
- 80% weight on LightGCN, 20% on TF-IDF

### Why α = 0.8?

Grid search results show:
- α = 1.0 (LightGCN only): HR@10 = 29.10%
- α = 0.8 (Hybrid): HR@10 = **29.80%** (+2.4%)
- α = 0.0 (TF-IDF only): HR@10 = 9.37%

The small TF-IDF contribution helps with items that have weak collaborative signals.

---

## Evaluation Metrics

### HR@K (Hit Rate)

Proportion of test cases where the ground truth item appears in top-K recommendations.

```
HR@K = (# hits in top-K) / (# test cases)
```

### NDCG@K (Normalized DCG)

Measures ranking quality, giving higher scores to hits at top positions.

```
DCG@K = Σ (rel_i / log2(i + 1))
NDCG@K = DCG@K / IDCG@K
```

### MRR (Mean Reciprocal Rank)

Average of 1/rank for the first relevant item.

```
MRR = (1/N) × Σ (1 / rank_i)
```

---

## References

1. He et al., "LightGCN", SIGIR 2020 - [arXiv:2002.02126](https://arxiv.org/abs/2002.02126)
2. Kang & McAuley, "SASRec", ICDM 2018 - [arXiv:1808.09781](https://arxiv.org/abs/1808.09781)
3. Xie et al., "CL4SRec", WWW 2022 - [arXiv:2010.14395](https://arxiv.org/abs/2010.14395)
4. Nguyen & Nguyen, "PhoBERT", EMNLP 2020 - [arXiv:2003.00744](https://arxiv.org/abs/2003.00744)
5. Rendle et al., "BPR", UAI 2009 - [arXiv:1205.2618](https://arxiv.org/abs/1205.2618)
