# 🛒 Recommendation System - Vietnamese E-commerce

Multi-model recommendation system for Vietnamese E-commerce (Tiki dataset).

---

## 📊 Performance Comparison

| Model | Type | HR@10 | NDCG@10 | MRR | Paper |
|-------|------|-------|---------|-----|-------|
| **Hybrid (α=0.8)** | CF + CB | **29.80%** | **17.70%** | - | LightGCN + TF-IDF |
| LightGCN | Collaborative | 13.50% | 7.85% | 6.50% | [SIGIR 2020](https://arxiv.org/abs/2002.02126) |
| SASRec | Sequential | 9.74% | 5.10% | - | [ICDM 2018](https://arxiv.org/abs/1808.09781) |
| CL4SRec | Sequential + CL | 9.85% | 5.16% | - | [WWW 2022](https://arxiv.org/abs/2010.14395) |
| TF-IDF | Content-Based | 7.50% | 4.35% | 3.58% | Classical IR |
| PhoBERT | Content-Based | 2.55% | 1.54% | 1.30% | [EMNLP 2020](https://arxiv.org/abs/2003.00744) |

> **Best Model**: Hybrid (LightGCN + TF-IDF) với α=0.80 đạt **+121%** so với LightGCN thuần

---

## 📈 Detailed Metrics

### Hybrid LightGCN + TF-IDF ⭐ (Best)

| Alpha | HR@10 | NDCG@10 |
|-------|-------|---------|
| 0.00 (TF-IDF) | 9.37% | 5.41% |
| 0.50 | 24.13% | 14.29% |
| **0.80** | **29.80%** | **17.70%** |
| 1.00 (LightGCN) | 29.10% | 16.91% |

### LightGCN

| Metric | @5 | @10 | @20 |
|--------|-----|-----|-----|
| HR | 9.28% | 13.50% | 18.76% |
| NDCG | 6.50% | 7.85% | 9.18% |

### SASRec (50 epochs)

| Metric | @10 |
|--------|-----|
| HR | 9.74% |
| NDCG | 5.10% |

### CL4SRec (50 epochs, λ=0.1)

| Metric | @10 |
|--------|-----|
| HR | 9.85% |
| NDCG | 5.16% |

### Content-Based

| Model | HR@10 | NDCG@10 | Category Precision |
|-------|-------|---------|-------------------|
| TF-IDF | 7.50% | 4.35% | 74.72% |
| PhoBERT | 2.55% | 1.54% | 50.90% |

---

## 📁 Project Structure

```
recommendation-system/
├── data/
│   ├── raw/                    # Raw Tiki dataset
│   └── processed/              # Preprocessed data
├── demo/                       # Gradio Web UI
├── LightGCN/                   # Graph Convolution Network
├── SASRec/                     # Self-Attentive Sequential
├── CL4SRec/                    # Contrastive Learning for SR
├── ContentBased_TFIDF/         # TF-IDF text similarity
├── ContentBased_PhoBERT/       # Vietnamese BERT embeddings
└── Hybrid_LightGCN_TFIDF/      # Hybrid CF + CB
```

---

## 🚀 Web Demo

Interactive Gradio web interface:

```bash
# Install Gradio
pip install gradio

# Run demo
cd demo
python app.py
```

Open http://localhost:7860 in browser.

**Features:**
- User Recommendations (with model selection)
- Similar Items (content-based)
- Compare Methods (LightGCN vs TF-IDF vs Hybrid)

---

## 🔬 Models

### 1. Hybrid LightGCN + TF-IDF ⭐

```bash
cd Hybrid_LightGCN_TFIDF
python demo.py
python tune_alpha.py --sample-users 3000
```

### 2. LightGCN

```bash
cd LightGCN
python train.py --epochs 50
python evaluate.py
```

### 3. SASRec

```bash
cd SASRec
python main.py --mode train --epochs 50
```

### 4. CL4SRec

```bash
cd CL4SRec
python train.py --epochs 50 --cl_weight 0.1
```

### 5. ContentBased_TFIDF

```bash
cd ContentBased_TFIDF
python preprocessing.py
python evaluate.py
```

### 6. ContentBased_PhoBERT

```bash
cd ContentBased_PhoBERT
python preprocessing.py   # ~10-15 min GPU
python evaluate.py
```

---

## 📈 Dataset

| Metric | Value |
|--------|-------|
| Users | 109,567 |
| Items (filtered) | 19,855 |
| Total Items | 123,016 |
| Interactions | 581,357 |

**Source**: Crawled from [tiki.vn](https://tiki.vn)

**Dataset**: [Qhuy204/TikDataset](https://huggingface.co/datasets/Qhuy204/TikDataset) on Hugging Face

---

## 📖 References

| Model | Paper | Link |
|-------|-------|------|
| LightGCN | He et al., SIGIR 2020 | [arXiv:2002.02126](https://arxiv.org/abs/2002.02126) |
| SASRec | Kang & McAuley, ICDM 2018 | [arXiv:1808.09781](https://arxiv.org/abs/1808.09781) |
| CL4SRec | Xie et al., WWW 2022 | [arXiv:2010.14395](https://arxiv.org/abs/2010.14395) |
| PhoBERT | Nguyen & Nguyen, EMNLP 2020 | [arXiv:2003.00744](https://arxiv.org/abs/2003.00744) |
| Hybrid RS | Burke, UMUAI 2002 | [Springer](https://link.springer.com/article/10.1023/A:1021240730564) |
| BPR Loss | Rendle et al., UAI 2009 | [arXiv:1205.2618](https://arxiv.org/abs/1205.2618) |

---

## ⚙️ Requirements

```bash
pip install torch numpy scipy scikit-learn tqdm transformers
```

- Python >= 3.8
- PyTorch >= 2.0
- CUDA GPU recommended

---

*Vietnamese E-commerce Recommendation System*
