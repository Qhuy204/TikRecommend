"""
Utility functions cho recommendation system
"""

import yaml
import json
import pickle
from pathlib import Path
from typing import Any, Dict


def load_config(config_path: str = "config.yaml") -> Dict:
    """Load configuration từ YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_pickle(obj: Any, filepath: str):
    """Save object as pickle"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
    print(f"✓ Saved: {filepath}")


def load_pickle(filepath: str) -> Any:
    """Load pickle object"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_json(obj: Any, filepath: str, indent: int = 2):
    """Save object as JSON"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)
    print(f"✓ Saved: {filepath}")


def load_json(filepath: str) -> Any:
    """Load JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_dir_structure():
    """Tạo cấu trúc thư mục project"""
    dirs = [
        "data/raw",
        "data/clean", 
        "data/processed",
        "models",
        "logs",
        "cache"
    ]
    
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    
    print("✓ Created directory structure")


if __name__ == "__main__":
    print("Creating project structure...")
    create_dir_structure()
    print("Done!")