import os
import json
import requests
from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

# Config
BASE_DIR = os.path.dirname(__file__) 
input_file = os.path.join(BASE_DIR, "../data/processed/tiki_dataset_clean.jsonl")
output_dir = os.path.join(BASE_DIR, "../data/images")
os.makedirs(output_dir, exist_ok=True)  

target_size = (224, 224)
max_workers = 16  # Thread

os.makedirs(output_dir, exist_ok=True)

def download_and_resize(product_id, url, idx, size=(224,224)):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")
        img = img.resize(size, Image.LANCZOS)
        filename = os.path.join(output_dir, f"{product_id}_{idx}.jpg")
        img.save(filename, format="JPEG", quality=85)
        return filename
    except Exception as e:
        return f"Failed {url}: {e}"

# read JSONL
tasks = []
with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        record = json.loads(line)
        product_id = record["product_id"]
        images = record["product_detail"].get("images", [])
        for i, img in enumerate(images):
            url = img.get("base_url")
            if url:
                tasks.append((product_id, url, i))

print(f"Number of images: {len(tasks)}")

# Parallel image downloading
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = [executor.submit(download_and_resize, pid, url, idx, target_size) 
               for pid, url, idx in tasks]

    for i, future in enumerate(as_completed(futures), 1):
        result = future.result()
        if i % 500 == 0:
            print(f"Processed {i}/{len(tasks)} image")

print("Complete download + resize image!")
