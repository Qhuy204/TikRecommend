# Makefile for Recommendation System Project
# Quick commands for common tasks

.PHONY: help install download clean preprocess train demo all

# Default target
help:
	@echo "=================================="
	@echo "RECOMMENDATION SYSTEM - COMMANDS"
	@echo "=================================="
	@echo ""
	@echo "Setup:"
	@echo "  make install      - Install dependencies"
	@echo "  make download     - Download TikDataset"
	@echo "  make setup        - Install + Download"
	@echo ""
	@echo "Data Pipeline:"
	@echo "  make clean-data   - Clean raw JSONL data"
	@echo "  make preprocess   - Extract features"
	@echo "  make interactions - Extract user-item interactions"
	@echo "  make pipeline     - Run full data pipeline"
	@echo ""
	@echo "Training:"
	@echo "  make train-two-tower  - Train retrieval model"
	@echo "  make train-mmoe       - Train ranking model"
	@echo "  make train-all        - Train both models"
	@echo ""
	@echo "Demo:"
	@echo "  make demo-single     - Demo single user"
	@echo "  make demo-batch      - Demo batch users"
	@echo "  make demo-coldstart  - Demo cold-start"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean       - Clean generated files"
	@echo "  make test        - Run tests"
	@echo "  make all         - Full pipeline + training"
	@echo ""

# ============================================================================
# SETUP
# ============================================================================

install:
	@echo "Installing dependencies..."
	pip install --upgrade pip
	pip install -r requirements.txt
	@echo "✓ Dependencies installed!"

download:
	@echo "Downloading TikDataset..."
	# Tạo thư mục trước để đảm bảo script có chỗ lưu
	@mkdir -p TikDataset
	# Gọi script với đúng tham số nó yêu cầu
	python download_dataset.py --output_dir TikDataset --repo_id Qhuy204/TikDataset
	@echo "✓ Dataset downloaded!"

setup: install download
	@echo "✓ Setup complete!"

# ============================================================================
# DATA PIPELINE
# ============================================================================

clean-data:
	@echo "Cleaning raw JSONL data..."
	# Tạo thư mục đích nếu chưa có
	@mkdir -p data/raw data/clean  
	python recommendation_system.py \
		--mode clean \
		--raw_jsonl TikDataset/tiki_dataset.jsonl \
		--clean_jsonl data/clean/tiki_dataset_clean.jsonl
	@echo "✓ Data cleaned!"

preprocess:
	@echo "Extracting features..."
	python recommendation_system.py \
		--mode preprocess \
		--clean_jsonl data/clean/tiki_dataset_clean.jsonl
	@echo "✓ Features extracted!"

interactions:
	@echo "Extracting user-item interactions..."
	python create_interactions.py \
		--input data/clean/tiki_dataset_clean.jsonl \
		--output data/processed/interactions.csv
	@echo "✓ Interactions extracted!"

pipeline: clean-data preprocess interactions
	@echo "✓ Full data pipeline complete!"

# Quick preprocessing with sample (for testing)
preprocess-sample:
	@echo "Processing sample data (1000 items)..."
	python recommendation_system.py \
		--mode full \
		--clean_jsonl data/clean/tiki_dataset_clean.jsonl \
		--sample_size 1000

# ============================================================================
# TRAINING
# ============================================================================

train-two-tower:
	@echo "Training Two-Tower Model..."
	python training_scripts.py \
		--model two_tower \
		--epochs 10 \
		--batch_size 64
	@echo "✓ Two-Tower training complete!"

train-mmoe:
	@echo "Training MMoE Model..."
	python training_scripts.py \
		--model mmoe \
		--epochs 20 \
		--batch_size 256
	@echo "✓ MMoE training complete!"

train-all: train-two-tower train-mmoe
	@echo "✓ All models trained!"

# Quick training with smaller epochs (for testing)
train-quick:
	@echo "Quick training (5 epochs each)..."
	python training_scripts.py \
		--model both \
		--epochs 5 \
		--batch_size 32

# ============================================================================
# DEMO & INFERENCE
# ============================================================================

demo-single:
	@echo "Running single user demo..."
	python demo_inference.py \
		--mode single \
		--user_id 12345

demo-batch:
	@echo "Running batch demo..."
	python demo_inference.py \
		--mode batch \
		--user_ids 12345 67890 11111 22222 33333

demo-coldstart:
	@echo "Running cold-start demo..."
	python demo_inference.py \
		--mode cold_start

# ============================================================================
# UTILITIES
# ============================================================================

clean:
	@echo "Cleaning generated files..."
	rm -rf data/processed/*.csv
	rm -rf models/*.pt
	rm -rf results/*
	rm -rf logs/*
	rm -rf cache/*
	rm -rf __pycache__
	rm -rf *.pyc
	@echo "✓ Cleaned!"

clean-all: clean
	@echo "Cleaning all data (including raw and clean)..."
	rm -rf data/raw/*
	rm -rf data/clean/*
	rm -rf TikDataset/*
	@echo "✓ All data cleaned!"

test:
	@echo "Running tests..."
	python -m pytest tests/ -v

lint:
	@echo "Running linters..."
	flake8 *.py
	black --check *.py

format:
	@echo "Formatting code..."
	black *.py
	isort *.py

# ============================================================================
# FULL PIPELINE
# ============================================================================

all: setup pipeline train-all
	@echo ""
	@echo "=================================="
	@echo "✓ FULL PIPELINE COMPLETE!"
	@echo "=================================="
	@echo ""
	@echo "Models saved:"
	@echo "  - models/two_tower_best.pt"
	@echo "  - models/mmoe_best.pt"
	@echo ""
	@echo "Next: make demo-single"

# Development workflow
dev: preprocess-sample train-quick demo-single
	@echo "✓ Development workflow complete!"

# ============================================================================
# MONITORING & ANALYSIS
# ============================================================================

stats:
	@echo "Dataset statistics:"
	@python -c "import pandas as pd; \
		df = pd.read_csv('data/processed/item_features.csv'); \
		print(f'Items: {len(df):,}'); \
		print(f'Categories: {df[\"category\"].nunique():,}'); \
		print(f'Brands: {df[\"brand_id\"].nunique():,}')"

check-models:
	@echo "Checking saved models..."
	@ls -lh models/*.pt 2>/dev/null || echo "No models found. Run 'make train-all' first."

tensorboard:
	@echo "Starting TensorBoard..."
	tensorboard --logdir=logs/

# ============================================================================
# DOCKER (Optional)
# ============================================================================

docker-build:
	@echo "Building Docker image..."
	docker build -t recommendation-system .

docker-run:
	@echo "Running in Docker..."
	docker run -it --gpus all -v $(PWD):/app recommendation-system

# ============================================================================
# JUPYTER NOTEBOOK
# ============================================================================

notebook:
	@echo "Starting Jupyter Notebook..."
	jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser

# ============================================================================
# BACKUP & EXPORT
# ============================================================================

backup:
	@echo "Creating backup..."
	@mkdir -p backups
	@timestamp=$$(date +%Y%m%d_%H%M%S); \
	tar -czf backups/backup_$$timestamp.tar.gz \
		data/processed/ models/ config.yaml
	@echo "✓ Backup created in backups/"

export-models:
	@echo "Exporting models..."
	@mkdir -p exports
	@cp models/*.pt exports/
	@cp config.yaml exports/
	@echo "✓ Models exported to exports/"

# ============================================================================
# DOCUMENTATION
# ============================================================================

docs:
	@echo "Generating documentation..."
	@mkdir -p docs
	@pdoc --html --output-dir docs recommendation_system.py training_scripts.py
	@echo "✓ Documentation generated in docs/"

# ============================================================================
# PROFILING & DEBUGGING
# ============================================================================

profile-preprocess:
	@echo "Profiling preprocessing..."
	python -m cProfile -o profile_preprocess.prof recommendation_system.py \
		--mode preprocess --sample_size 100
	@python -c "import pstats; \
		p = pstats.Stats('profile_preprocess.prof'); \
		p.sort_stats('cumulative').print_stats(20)"

profile-inference:
	@echo "Profiling inference..."
	python -m cProfile -o profile_inference.prof demo_inference.py \
		--mode single --user_id 12345
	@python -c "import pstats; \
		p = pstats.Stats('profile_inference.prof'); \
		p.sort_stats('cumulative').print_stats(20)"

debug:
	@echo "Running with debug logging..."
	python -u recommendation_system.py \
		--mode preprocess \
		--sample_size 10 \
		2>&1 | tee logs/debug.log