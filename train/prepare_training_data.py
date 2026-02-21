import tarfile
import gzip
import json
import io
import os
import pandas as pd
import random
from collections import defaultdict

def get_english_attributes(item):
    """Extracts specified fields for ANY English locale (en_IN, en_US, etc.)."""
    target_fields = {
        'brand', 'bullet_point', 'color', 'item_id', 
        'item_keywords', 'item_name', 'main_image_id', 
        'other_image_id', 'product_type'
    }
    extracted = {}
    
    for key in target_fields:
        value = item.get(key)
        if value is None:
            continue
            
        if isinstance(value, list):
            if key == 'product_type':
                if len(value) > 0 and isinstance(value[0], dict):
                    extracted[key] = value[0].get('value', 'UNKNOWN')
                else:
                    extracted[key] = 'UNKNOWN'
            elif key == 'other_image_id':
                extracted[key] = value
            elif key in ['bullet_point', 'item_keywords']:
                vals = [v['value'] for v in value if isinstance(v, dict) and str(v.get('language_tag', '')).startswith('en')]
                if vals: extracted[key] = vals
            else:
                # Prioritize en_IN, fallback to any 'en'
                en_val = next((v['value'] for v in value if isinstance(v, dict) and v.get('language_tag') == 'en_IN'), None)
                if not en_val:
                    en_val = next((v['value'] for v in value if isinstance(v, dict) and str(v.get('language_tag', '')).startswith('en')), None)
                if en_val: extracted[key] = en_val
        else:
            extracted[key] = value
            
    return extracted

def prepare_strict_100_dataset(listings_tar, images_tar, csv_path, output_dir="training_data", count_per_type=100):
    img_output_dir = os.path.join(output_dir, "images")
    os.makedirs(img_output_dir, exist_ok=True)
    jsonl_path = os.path.join(output_dir, "full_metadata.jsonl")

    # 1. Load Technical Metadata (CSV)
    df_map = pd.read_csv(csv_path)
    tech_map = df_map.set_index('image_id').to_dict('index')

    # 2. Group all valid English items
    category_map = defaultdict(list)
    print("--- Scanning listings ---")
    
    with tarfile.open(listings_tar, "r") as l_tar:
        members = [m for m in l_tar.getmembers() if m.name.endswith(".json.gz")]
        for m in members:
            f = l_tar.extractfile(m)
            if f:
                with gzip.open(io.BytesIO(f.read()), 'rt', encoding='utf-8') as gz:
                    for line in gz:
                        product = json.loads(line)
                        img_id = product.get("main_image_id")
                        meta = get_english_attributes(product)
                        
                        if img_id in tech_map and meta.get("item_name"):
                            meta['height'] = tech_map[img_id]['height']
                            meta['width'] = tech_map[img_id]['width']
                            meta['csv_path'] = tech_map[img_id]['path']
                            
                            p_type = meta.get('product_type', 'UNKNOWN')
                            category_map[p_type].append({
                                "item_id": meta.get('item_id'),
                                "text": meta.get('item_name'),
                                "metadata": meta,
                                "tar_path": f"images/small/{tech_map[img_id]['path']}"
                            })

    # 3. Apply Strict Balancing (Downsample and Upsample)
    final_selection = []
    print("\n--- APPLYING STRICT 100-COUNT BALANCE ---")
    print(f"{'PRODUCT_TYPE':<40} | {'ORIGINAL':<10} | {'ACTION'}")
    print("-" * 65)

    for p_type, items in sorted(category_map.items(), key=lambda x: len(x[1]), reverse=True):
        original_count = len(items)
        if original_count >= count_per_type:
            # DOWNSAMPLE
            selected = random.sample(items, count_per_type)
            action = f"Downsampled to {count_per_type}"
        else:
            # UPSAMPLE (Repeat images)
            repeats = (count_per_type // original_count) + 1
            selected = (items * repeats)[:count_per_type]
            action = f"Upsampled (Repeated) to {count_per_type}"
            
        final_selection.extend(selected)
        print(f"{str(p_type)[:39]:<40} | {original_count:<10} | {action}")

    # 4. Extract Images & Save
    unique_paths = {info["tar_path"] for info in final_selection}
    print(f"\n--- Extracting {len(unique_paths)} unique files ---")
    
    extracted_files = 0
    with tarfile.open(images_tar, "r") as i_tar:
        for member in i_tar:
            if member.name in unique_paths:
                f_img = i_tar.extractfile(member)
                if f_img:
                    filename = os.path.basename(member.name)
                    hash_prefix = filename[:2]
                    target_subdir = os.path.join(img_output_dir, hash_prefix)
                    os.makedirs(target_subdir, exist_ok=True)
                    with open(os.path.join(target_subdir, filename), "wb") as f_out:
                        f_out.write(f_img.read())
                    extracted_files += 1

    # Save the JSONL (with 100 records per category)
    with open(jsonl_path, "w", encoding="utf-8") as out:
        for info in final_selection:
            filename = os.path.basename(info['tar_path'])
            hash_prefix = filename[:2]
            record = {
                "item_id": info["item_id"],
                "text": info["text"],
                "image_path": f"images/{hash_prefix}/{filename}",
                "metadata": info["metadata"]
            }
            out.write(json.dumps(record) + "\n")

    print(f"\n🏁 Finished! Every category now has exactly {count_per_type} entries.")
    print(f"Total entries in JSONL: {len(final_selection)}")

if __name__ == "__main__":
    prepare_strict_100_dataset(
        listings_tar=r"data\abo-listings.tar",
        images_tar=r"data\abo-images-small.tar",
        csv_path=r"data\images.csv",
        count_per_type=100
    )

