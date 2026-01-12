import os
import json
import time
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

# CONFIG
BASE_DIR = os.path.dirname(__file__) 
CATEGORIES_FILE = os.path.join(BASE_DIR, "tiki_leaf_categories.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "../data/raw/full_reviews")
os.makedirs(OUTPUT_DIR, exist_ok=True)  
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "tiki_dataset.jsonl")
MAX_WORKERS = 8
MAX_PAGES = 40

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "application/json",
}

# API HELPER
def fetch_json(url, params=None, retries=3):
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, params=params,
                             timeout=120, allow_redirects=False)
            if r.is_redirect or r.status_code in (301, 302, 303, 307, 308):
                print(f"Redirect at {url}")
                return {"error": "redirect"}

            if r.status_code == 429:
                print(f"Rate limited (429). Sleeping before retry... {url}")
                time.sleep(3)
                continue

            r.raise_for_status()
            if not r.text.strip():
                return {"error": "empty"}
            return r.json()
        except Exception as e:
            print(f"Attempt {attempt+1} failed for {url}: {e}")
            time.sleep(1)
    return {"error": "failed"}

def get_listings(cat_id, url_key, page=1, limit=40):
    url = "https://tiki.vn/api/personalish/v1/blocks/listings"
    params = {"limit": limit, "page": page, "urlKey": url_key, "category": cat_id}
    return fetch_json(url, params)

def get_product_detail(pid, spid):
    url = f"https://tiki.vn/api/v2/products/{pid}"
    return fetch_json(url, {"platform": "web", "spid": spid, "version": 3})

def get_all_reviews(pid, spid, sid, limit=20, max_pages=200):
    all_reviews = []
    current_page = 1

    while True:
        url = "https://tiki.vn/api/v2/reviews"
        params = {
            "limit": limit,
            "include": "comments,contribute_info,attribute_vote_summary",
            "sort": "score|desc,id|desc,stars|all",
            "page": current_page,
            "spid": spid,
            "product_id": pid,
            "seller_id": sid,
        }

        data = fetch_json(url, params)

        # Error or redirect
        if not data or "data" not in data:
            break

        batch = data.get("data", [])
        if not batch:
            break

        all_reviews.extend(batch)

        # Lấy last_page từ API
        paging = data.get("paging", {})
        last_page = paging.get("last_page", current_page)

        print(f"  → Reviews PID={pid} page={current_page}/{last_page}, got={len(batch)}")

        if current_page >= last_page:
            break

        current_page += 1
        time.sleep(0.3)  # tránh 429

    return all_reviews

def get_reviews(pid, spid, sid):
    url = "https://tiki.vn/api/v2/reviews"
    params = {
        "limit": 10,
        "include": "comments,contribute_info,attribute_vote_summary",
        "sort": "score|desc,id|desc,stars|all",
        "page": 1,
        "spid": spid,
        "product_id": pid,
        "seller_id": sid,
    }
    return fetch_json(url, params)

def get_top_reviews(pid, spid, sid):
    url = "https://tiki.vn/api/v2/reviews"
    params = {
        "product_id": pid,
        "include": "comments",
        "page": 1,
        "limit": -1,
        "top": "true",
        "spid": spid,
        "seller_id": sid,
    }
    return fetch_json(url, params)

# WORKER
def crawl_product(item, category_name):
    pid = item.get("id")
    spid = item.get("seller_product_id") or item.get("spid")
    sid = item.get("seller_id") or (item.get("seller") or {}).get("id")

    if not pid or not spid:
        return {"product_id": pid, "error": "missing spid"}

    product = get_product_detail(pid, spid)
    # reviews = get_reviews(pid, spid, sid)
    reviews = get_all_reviews(pid, spid, sid)
    top_reviews = get_top_reviews(pid, spid, sid)

    entry = {
        "product_id": pid,
        "category": category_name,
        "product_detail": product,
        "reviews": reviews,
        "top_reviews": top_reviews,
    }
    print(f"Done {pid}")
    return entry

# SAVE NDJSON
def save_batch(batch):
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        for entry in batch:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"💾 Saved {len(batch)} products")

# MAIN CRAWL
def crawl_all():
    df = pd.read_csv(CATEGORIES_FILE)

    for _, row in df.iterrows():
        cat_id, url_key, cat_name = row["id"], row["url_key"], row["name"]
        page = 1
        while page <= MAX_PAGES:
            data = get_listings(cat_id, url_key, page)
            if not data or "data" not in data:
                break
            items = data.get("data", [])
            if not items:
                break

            batch, futures = [], []
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                for item in items:
                    futures.append(executor.submit(crawl_product, item, cat_name))

                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        batch.append(result)
                        if len(batch) >= 10:
                            save_batch(batch)
                            batch = []

            if batch:
                save_batch(batch)

            print(f"Done category {cat_id}, page {page}, items={len(items)}")
            page += 1

if __name__ == "__main__":
    while True:
        crawl_all()
        print("One full crawl done, sleep 2s...")
        time.sleep(2)
