"""
Script để kiểm tra xem tất cả files cần thiết đã có chưa
Run: python check_files.py
"""

from pathlib import Path
import pandas as pd
import sys

def check_file(filepath: str, file_type: str = "file") -> bool:
    """Kiểm tra file/folder có tồn tại không"""
    path = Path(filepath)
    
    if file_type == "file":
        exists = path.is_file()
        size = path.stat().st_size if exists else 0
        size_mb = size / (1024 * 1024)
        
        status = "✅" if exists else "❌"
        size_str = f"({size_mb:.2f} MB)" if exists else ""
        
        print(f"{status} {filepath} {size_str}")
        return exists
    else:
        exists = path.is_dir()
        status = "✅" if exists else "❌"
        print(f"{status} {filepath}/")
        return exists

def check_csv_content(filepath: str, name: str):
    """Kiểm tra nội dung CSV file"""
    try:
        df = pd.read_csv(filepath)
        print(f"   └─ {name}: {len(df):,} rows, {len(df.columns)} columns")
        print(f"      Columns: {', '.join(df.columns[:5])}" + 
              (f", ... (+{len(df.columns)-5} more)" if len(df.columns) > 5 else ""))
        return True
    except Exception as e:
        print(f"   └─ ❌ Error reading file: {e}")
        return False

def main():
    print("\n" + "="*80)
    print("FILE VERIFICATION - Recommendation System")
    print("="*80)
    
    all_ok = True
    
    # Check directories
    print("\n[1] CHECKING DIRECTORIES")
    print("-" * 80)
    required_dirs = [
        "data",
        "data/raw",
        "data/clean",
        "data/processed",
        "models",
        "results"
    ]
    
    for dir_path in required_dirs:
        if not check_file(dir_path, "folder"):
            all_ok = False
            print(f"    → Run: mkdir -p {dir_path}")
    
    # Check raw data
    print("\n[2] CHECKING RAW DATA")
    print("-" * 80)
    raw_files = [
        "data/raw/tiki_dataset.jsonl",
    ]
    
    has_raw = False
    for file_path in raw_files:
        if check_file(file_path, "file"):
            has_raw = True
    
    if not has_raw:
        print("    → Run: python download_dataset.py")
        print("    → Or: make download")
        all_ok = False
    
    # Check cleaned data
    print("\n[3] CHECKING CLEANED DATA")
    print("-" * 80)
    clean_file = "data/clean/tiki_dataset_clean.jsonl"
    
    if check_file(clean_file, "file"):
        print("    ✓ Data cleaning completed")
    else:
        print("    → Run: make clean-data")
        print("    → Or: python recommendation_system.py --mode clean \\")
        print("              --raw_jsonl data/raw/tiki_dataset.jsonl \\")
        print("              --clean_jsonl data/clean/tiki_dataset_clean.jsonl")
        all_ok = False
    
    # Check processed features
    print("\n[4] CHECKING PROCESSED FEATURES")
    print("-" * 80)
    
    processed_files = {
        "data/processed/item_features.csv": "Item Features",
        "data/processed/ranking_features.csv": "Ranking Features",
        "data/processed/interactions.csv": "User-Item Interactions"
    }
    
    missing_processed = []
    for file_path, name in processed_files.items():
        if check_file(file_path, "file"):
            check_csv_content(file_path, name)
        else:
            missing_processed.append(name)
            all_ok = False
    
    if missing_processed:
        print(f"\n    ❌ Missing: {', '.join(missing_processed)}")
        print("    → Run: make preprocess")
        print("    → Or: python recommendation_system.py --mode preprocess \\")
        print("              --clean_jsonl data/clean/tiki_dataset_clean.jsonl")
        
        if "User-Item Interactions" in missing_processed:
            print("\n    For interactions specifically:")
            print("    → Run: make interactions")
            print("    → Or: python create_interactions.py \\")
            print("              --input data/clean/tiki_dataset_clean.jsonl \\")
            print("              --output data/processed/interactions.csv")
    
    # Check trained models
    print("\n[5] CHECKING TRAINED MODELS")
    print("-" * 80)
    
    model_files = {
        "models/two_tower_best.pt": "Two-Tower Model",
        "models/mmoe_best.pt": "MMoE Model"
    }
    
    missing_models = []
    for file_path, name in model_files.items():
        if check_file(file_path, "file"):
            print(f"    ✓ {name} trained")
        else:
            missing_models.append(name)
    
    if missing_models:
        print(f"\n    ⚠️  Models not trained yet: {', '.join(missing_models)}")
        print("    → Run: make train-all")
        print("    → Or: python training_scripts.py --model both")
    
    # Summary
    print("\n" + "="*80)
    if all_ok:
        print("✅ ALL CHECKS PASSED!")
        print("="*80)
        print("\nYour pipeline is ready!")
        print("\nNext steps:")
        if missing_models:
            print("  1. Train models: make train-all")
            print("  2. Run demo: make demo-single")
        else:
            print("  1. Run demo: make demo-single")
            print("  2. Generate recommendations: python demo_inference.py")
    else:
        print("❌ SOME FILES ARE MISSING")
        print("="*80)
        print("\nQuick fix - Run complete pipeline:")
        print("  make pipeline     # Clean + Preprocess + Extract interactions")
        print("  make train-all    # Train both models")
        print("  make demo-single  # Test recommendations")
        
        print("\nOr step by step:")
        print("  1. make clean-data      # Clean raw JSONL")
        print("  2. make preprocess      # Extract features")
        print("  3. make interactions    # Extract user-item data")
        print("  4. make train-all       # Train models")
        print("  5. make demo-single     # Demo")
    
    print("\n" + "="*80)
    
    # Return exit code
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())