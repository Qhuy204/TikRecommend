import json
import pandas as pd

def extract_features_from_jsonl(file_path, output_csv="products_features.csv", limit=None):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            try:
                record = json.loads(line.strip())
            except:
                continue

            # 1. ID & Category
            product_id = record.get("product_id")
            category = record.get("category")

            detail = record.get("product_detail")
            if not isinstance(detail, dict): 
                detail = {}

            breadcrumbs = " > ".join(
                [b.get("name", "") for b in detail.get("breadcrumbs", [])]
            )

            # 2. Product information
            name = detail.get("name")
            short_desc = detail.get("short_description")
            long_desc = detail.get("description")

            # highlight
            highlight_items = None
            if isinstance(detail.get("highlight"), dict):
                highlight_items = " ".join(detail["highlight"].get("items", []))

            # specifications
            specs = []
            for spec in detail.get("specifications", []) or []:
                for attr in spec.get("attributes", []) or []:
                    if isinstance(attr, dict) and "value" in attr:
                        specs.append(attr["value"])
            specs_text = " ".join(specs) if specs else None

            brand = (detail.get("brand") or {}).get("name")

            # 3. product image
            images = []
            for img in detail.get("images", []) or []:
                if isinstance(img, dict) and "base_url" in img:
                    images.append(img["base_url"])
            thumbnail = detail.get("thumbnail_url")

            # 4. Metadata numeric/categorical
            price = detail.get("price")
            list_price = detail.get("list_price")
            discount_rate = detail.get("discount_rate")
            rating_avg = detail.get("rating_average")
            review_count = detail.get("review_count")
            qty_sold = detail.get("all_time_quantity_sold")
            fav_count = detail.get("favourite_count")
            seller_name = (detail.get("current_seller") or {}).get("name")

            # 5. Review features
            reviews = record.get("reviews") or {}
            review_avg = reviews.get("rating_average")
            review_texts = []
            if isinstance(reviews.get("data"), list):
                for rv in reviews["data"]:
                    if isinstance(rv, dict) and "content" in rv:
                        review_texts.append(rv["content"])
            review_texts = " ".join(review_texts) if review_texts else None

            stars = reviews.get("stars", {})

            # 6. Optional
            badges = [b.get("code") for b in detail.get("badges_v3", []) or [] if isinstance(b, dict)]
            benefits = [b.get("text") for b in detail.get("benefits", []) or [] if isinstance(b, dict)]
            installment = [i.get("title") for i in detail.get("installment_info_v3", []) or [] if isinstance(i, dict)]

            data.append({
                "product_id": product_id,
                "category": category,
                "breadcrumbs": breadcrumbs,
                "name": name,
                "short_description": short_desc,
                "description": long_desc,
                "highlight": highlight_items,
                "specifications": specs_text,
                "brand": brand,
                "images": images,
                "thumbnail": thumbnail,
                "price": price,
                "list_price": list_price,
                "discount_rate": discount_rate,
                "rating_avg": rating_avg,
                "review_count": review_count,
                "all_time_quantity_sold": qty_sold,
                "favourite_count": fav_count,
                "seller_name": seller_name,
                "review_avg": review_avg,
                "review_texts": review_texts,
                "stars": stars,
                "badges": badges,
                "benefits": benefits,
                "installment_info": installment
            })

            if limit and i + 1 >= limit:
                break

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    return df


if __name__ == "__main__":
    df = extract_features_from_jsonl("tiki_dataset_clean.jsonl", limit=100000)
    print(df.head(3))
    print(df.info())
