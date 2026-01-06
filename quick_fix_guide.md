# 🔧 Quick Fix Guide - Common Issues

## 🚨 Your Current Issue

### Error: `FileNotFoundError: data/processed/item_features.csv`

**Cause:** Bạn đã chạy `make interactions` nhưng chưa chạy `make preprocess` trước đó.

**Solution:**

```bash
# Option 1: Chạy đúng thứ tự
make preprocess      # Tạo item_features.csv và ranking_features.csv
make interactions    # Tạo interactions.csv
make train-all       # Train models

# Option 2: Chạy một lần luôn (recommended)
make pipeline        # Chạy cả 3 bước trên
make train-all       # Sau đó train
```

---

## 📋 Correct Pipeline Order

```
Step 1: Download
  ↓
Step 2: Clean Data (remove errors)
  ↓
Step 3: Preprocess (extract features)  ← BẠN THIẾU BƯỚC NÀY!
  ↓
Step 4: Extract Interactions
  ↓
Step 5: Train Models
```

### Commands theo thứ tự:

```bash
# 1. Download dataset (nếu chưa có)
make download

# 2. Clean data
make clean-data
# Output: data/clean/tiki_dataset_clean.jsonl

# 3. Preprocess features (IMPORTANT!)
make preprocess
# Output: 
#   - data/processed/item_features.csv
#   - data/processed/ranking_features.csv

# 4. Extract interactions
make interactions
# Output: data/processed/interactions.csv

# 5. Train models
make train-all

# 6. Demo
make demo-single
```

---

## ✅ Quick Check - Verify Files

```bash
# Check what files you have
python check_files.py

# Or manually check
ls -lh data/processed/
```

**Expected output:**
```
item_features.csv       # ← YOU NEED THIS
ranking_features.csv    # ← YOU NEED THIS
interactions.csv        # ✓ YOU HAVE THIS
```

---

## 🎯 Fast Fix - One Command

Nếu bạn muốn chạy lại từ đầu:

```bash
# Clean everything and start fresh
make clean
make pipeline    # This runs: clean-data → preprocess → interactions
make train-all   # Train both models
```

---

## 🔍 Debugging Steps

### 1. Check if clean data exists

```bash
ls -lh data/clean/tiki_dataset_clean.jsonl
```

**If missing:**
```bash
make clean-data
```

### 2. Check if preprocessed features exist

```bash
ls -lh data/processed/item_features.csv
ls -lh data/processed/ranking_features.csv
```

**If missing:**
```bash
make preprocess
```

### 3. Check if interactions exist

```bash
ls -lh data/processed/interactions.csv
```

**If missing:**
```bash
make interactions
```

### 4. Now train

```bash
make train-all
```

---

## 📊 Understanding the Files

### `item_features.csv` (for Two-Tower Model)
Columns:
- `product_id`: ID sản phẩm
- `text_content`: Text đã clean (name + description + specs)
- `category`: Danh mục
- `brand_id`: ID thương hiệu

**Created by:** `make preprocess`

### `ranking_features.csv` (for MMoE Model)
Columns:
- `product_id`, `price`, `list_price`, `discount_rate`
- `rating_average`, `review_count`, `quantity_sold`
- `seller_id`, `is_authentic`, `is_freeship`, `has_return_policy`, `is_available`

**Created by:** `make preprocess`

### `interactions.csv` (for Training)
Columns:
- `user_id`: Customer ID
- `product_id`: Product ID
- `rating`: Rating score
- `timestamp`: Review time
- `is_good_quality`: Quality signal (0/1)
- `is_good_price`: Price signal (0/1)

**Created by:** `make interactions`

---

## 🚀 Full Workflow - Start to Finish

```bash
# Step 1: Setup
git clone <repo>
cd recommendation-system
make install

# Step 2: Get data
make download

# Step 3: Process data (IMPORTANT - ALL 3 FILES)
make pipeline
# This creates:
#   ✓ data/clean/tiki_dataset_clean.jsonl
#   ✓ data/processed/item_features.csv
#   ✓ data/processed/ranking_features.csv  
#   ✓ data/processed/interactions.csv

# Step 4: Verify files
python check_files.py
# Should show all ✅

# Step 5: Train
make train-all
# This trains:
#   ✓ models/two_tower_best.pt
#   ✓ models/mmoe_best.pt

# Step 6: Test
make demo-single
```

---

## ⚡ Time Estimates

With full dataset (~125k products, 1.3M reviews):

| Step | Time | Output |
|------|------|--------|
| `make download` | 10-30 min | Raw JSONL |
| `make clean-data` | 2 min | Clean JSONL |
| `make preprocess` | 10-15 min | Features CSV |
| `make interactions` | 1-2 min | Interactions CSV |
| `make train-two-tower` | 30-60 min | Two-Tower model |
| `make train-mmoe` | 10-20 min | MMoE model |
| **TOTAL** | **~1-2 hours** | Full system |

With sample (1000 products):

```bash
make preprocess-sample  # ~1 min
make train-quick        # ~5 min
make demo-single        # <1 min
```

---

## 🐛 Other Common Errors

### Error: "CUDA out of memory"

**Solution:**
```bash
# Reduce batch size
python training_scripts.py --batch_size 32

# Or use CPU
export CUDA_VISIBLE_DEVICES=-1
python training_scripts.py --batch_size 64
```

### Error: "PhoBERT not found"

**Solution:**
```bash
# Download manually first
python -c "from transformers import AutoModel, AutoTokenizer; \
           AutoModel.from_pretrained('vinai/phobert-base'); \
           AutoTokenizer.from_pretrained('vinai/phobert-base')"
```

### Error: "No module named 'recommendation_system'"

**Solution:**
```bash
# Make sure you're in the right directory
cd recommendation-system

# Check if file exists
ls recommendation_system.py
```

---

## 📞 Still Having Issues?

1. **Check files:**
   ```bash
   python check_files.py
   ```

2. **Clean and restart:**
   ```bash
   make clean
   make pipeline
   ```

3. **Check logs:**
   ```bash
   make debug
   ```

4. **Verify dataset:**
   ```bash
   head -n 1 data/clean/tiki_dataset_clean.jsonl | python -m json.tool
   ```

---

## ✅ Success Checklist

Before training:
- [ ] ✅ `data/clean/tiki_dataset_clean.jsonl` exists
- [ ] ✅ `data/processed/item_features.csv` exists
- [ ] ✅ `data/processed/ranking_features.csv` exists
- [ ] ✅ `data/processed/interactions.csv` exists

After training:
- [ ] ✅ `models/two_tower_best.pt` exists
- [ ] ✅ `models/mmoe_best.pt` exists

Ready to demo:
- [ ] ✅ All above files exist
- [ ] ✅ `python check_files.py` shows all green

---

**Now you can run:**
```bash
make demo-single
```

Good luck! 🚀
