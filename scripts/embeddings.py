import torch, faiss, pickle, open_clip, io, os, tarfile
import numpy as np
from PIL import Image
from peft import PeftModel
from timm.utils import reparameterize_model

# --- 1. CONFIGURATION ---
BASE = "/mnt/custom-file-systems/efs/fs-0e27656366552e874_fsap-0caa5a9a3db4b6b7d/train"
MODEL_PATH = f"{BASE}/output/mobileclip2_lora"
SHARD_DIR = f"{BASE}/embedding/shards"
DEVICE = "cuda"
BATCH_SIZE = 256

# --- 2. MODEL PREP ---
print("--- Loading & Merging Model ---")
# Use the dfndr2b weights as requested
model, _, preprocess = open_clip.create_model_and_transforms(
    "MobileCLIP2-S2", 
    pretrained='dfndr2b'
)

if os.path.exists(MODEL_PATH):
    print(f"Applying LoRA from {MODEL_PATH}...")
    model = PeftModel.from_pretrained(model, MODEL_PATH).merge_and_unload()
else:
    print("WARNING: LoRA path not found. Using base model.")

print("Optimizing for A10G inference...")
model = reparameterize_model(model).to(DEVICE).eval()

# --- 3. RAW TAR SCANNER ---
def stream_images_from_tars(directory):
    """
    Directly opens TAR files and hunts for image extensions.
    Bypasses all folder/grouping logic.
    """
    tar_files = sorted([f for f in os.listdir(directory) if f.endswith('.tar')])
    if not tar_files:
        print(f"No TAR files found in {directory}")
        return

    for tar_name in tar_files:
        tar_path = os.path.join(directory, tar_name)
        print(f"Scanning {tar_name}...")
        with tarfile.open(tar_path, "r") as tar:
            for member in tar.getmembers():
                # Only process if it's a file ending in an image extension
                if member.isfile() and member.name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    try:
                        f = tar.extractfile(member)
                        if f:
                            # Load and ensure RGB
                            img = Image.open(io.BytesIO(f.read())).convert("RGB")
                            # Use basename (e.g., 00a6.jpg) as the key
                            yield img, os.path.basename(member.name)
                    except Exception as e:
                        print(f"Error reading {member.name}: {e}")
                        continue

# --- 4. INFERENCE LOOP ---
print("--- Starting Inference ---")
embeddings, metadata = [], []
batch_imgs, batch_keys = [], []

with torch.no_grad():
    # Loop through the deep-scan generator
    for img, key in stream_images_from_tars(SHARD_DIR):
        # preprocess() resizes the image to 256x256 (or model native size)
        batch_imgs.append(preprocess(img))
        batch_keys.append(key)
        
        # When batch is full, push to GPU
        if len(batch_imgs) == BATCH_SIZE:
            input_tensor = torch.stack(batch_imgs).to(DEVICE)
            feat = model.encode_image(input_tensor)
            
            # Unit normalization for Cosine Similarity (FAISS IndexFlatIP)
            feat /= feat.norm(dim=-1, keepdim=True)
            
            embeddings.append(feat.cpu().numpy())
            metadata.extend([{"key": k} for k in batch_keys])
            
            batch_imgs, batch_keys = [], []
            if len(metadata) % 1024 == 0:
                print(f"Processed {len(metadata)} images...")

    # Final cleanup for the last partial batch
    if batch_imgs:
        input_tensor = torch.stack(batch_imgs).to(DEVICE)
        feat = model.encode_image(input_tensor)
        feat /= feat.norm(dim=-1, keepdim=True)
        embeddings.append(feat.cpu().numpy())
        metadata.extend([{"key": k} for k in batch_keys])

# --- 5. SAVE ---
if not embeddings:
    print("FATAL: No images were found in the shards.")
else:
    print(f"Success! Building Index for {len(metadata)} vectors...")
    emb_np = np.vstack(embeddings).astype('float32')
    
    # Flat Inner Product index = Cosine Similarity on normalized vectors
    index = faiss.IndexFlatIP(emb_np.shape[1])
    index.add(emb_np)
    
    faiss.write_index(index, "shoptalk_index.faiss")
    pickle.dump(metadata, open("metadata.pkl", "wb"))
    print(f"DONE. Files saved: shoptalk_index.faiss, metadata.pkl")