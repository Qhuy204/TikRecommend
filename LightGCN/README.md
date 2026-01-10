# LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

**Paper**: SIGIR 2020 - He et al.

---

## 🚀 Quick Start

```bash
cd LightGCN

# Step 1: Preprocess (build user-item graph)
python preprocessing.py

# Step 2: Train
python train.py --epochs 50

# Step 3: Evaluate
python evaluate.py

# Step 4: Demo
python demo.py
```

---

## 🔬 Method

```
LightGCN removes:
- Feature transformation (W matrices)  
- Nonlinear activation (ReLU)

Only keeps:
- Neighborhood aggregation
- Layer combination (weighted sum)

Final = (E^0 + E^1 + E^2 + ... + E^K) / (K+1)
```

---

## 📁 Files

| File | Description |
|------|-------------|
| `preprocessing.py` | Build user-item interaction graph |
| `models.py` | LightGCN model |
| `train.py` | BPR training |
| `evaluate.py` | Metrics |
| `demo.py` | Interactive demo |

---

## 📖 Reference

```bibtex
@inproceedings{lightgcn2020,
  title={LightGCN: Simplifying and Powering Graph Convolution Network},
  author={He, Xiangnan and Deng, Kuan and Wang, Xiang and Li, Yan and Zhang, Yongdong and Wang, Meng},
  booktitle={SIGIR},
  year={2020}
}
```
