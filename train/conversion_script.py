import webdataset as wds
import json
import os
from PIL import Image
import io


# Configuration
input_jsonl = "training_data/refined_sagemaker_metadata.jsonl"
output_pattern = "mobileclip_data_%05d.tar"
max_count = 10000  # Number of images per tar shard
# Set the base directory for images so relative paths are resolved correctly
base_image_dir = os.path.join(os.path.dirname(__file__), "..", "training_data", "images")

sink = wds.ShardWriter(output_pattern, maxcount=max_count)

with open(input_jsonl, "r") as f:
    for i, line in enumerate(f):
        data = json.loads(line)
        

        img_path = data['image_path']
        caption = data['refined_caption']

        # If img_path starts with 'images/', remove it before joining
        img_path_norm = img_path.replace("\\", "/")
        if base_image_dir and not os.path.isabs(img_path):
            if img_path_norm.startswith("images/"):
                img_path_trimmed = img_path_norm[len("images/"):]
                img_path_full = os.path.join(base_image_dir, img_path_trimmed)
            else:
                img_path_full = os.path.join(base_image_dir, img_path)
        else:
            img_path_full = img_path

        if os.path.exists(img_path_full):
            # --- Resize Block ---
            try:
                with Image.open(img_path_full) as img:
                    img = img.convert("RGB")  # Ensure 3-channel RGB
                    img = img.resize((256, 256), Image.BICUBIC)

                    buffer = io.BytesIO()
                    img.save(buffer, format="JPEG", quality=95)
                    image_data = buffer.getvalue()
            except Exception as e:
                print(f"Error processing {img_path_full} at line {i+1}: {e}")
                continue
            # ---------------------

            # Write the pair to the current shard
            sink.write({
                "__key__": f"sample_{i:06d}",
                "jpg": image_data,
                "txt": caption
            })
        else:
            print(f"Warning: {img_path_full} not found (line {i+1}).")

sink.close()
