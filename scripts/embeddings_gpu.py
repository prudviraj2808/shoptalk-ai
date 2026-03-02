import torch, faiss, pickle, open_clip, io, os, tarfile
import numpy as np
from PIL import Image
from peft import PeftModel
from timm.utils import reparameterize_model

# ----------------------------
# 1. CONFIGURATION
# ----------------------------
BASE = "/mnt/custom-file-systems/efs/fs-0e27656366552e874_fsap-0caa5a9a3db4b6b7d/train"
MODEL_PATH = f"{BASE}/output/mobileclip2_lora"
SHARD_DIR = f"{BASE}/embedding/shards"

DEVICE = "cuda"
BATCH_SIZE = 512          # Optimal for A10G
NLIST = 1024              # IVF clusters (good for 300K)
NPROBE = 16               # Search speed vs recall
TRAINING_SIZE = 50000     # IVF training subset

# ----------------------------
# 2. MODEL PREP
# ----------------------------
print("--- Loading & Merging Model ---")

model, _, preprocess = open_clip.create_model_and_transforms(
    "MobileCLIP2-S2",
    pretrained="dfndr2b"
)

if os.path.exists(MODEL_PATH):
    print(f"Applying LoRA from {MODEL_PATH}...")
    model = PeftModel.from_pretrained(model, MODEL_PATH).merge_and_unload()
else:
    print("WARNING: LoRA path not found. Using base model.")

model = reparameterize_model(model).to(DEVICE).eval()
torch.backends.cudnn.benchmark = True

# ----------------------------
# 3. TAR IMAGE STREAMER
# ----------------------------
def stream_images_from_tars(directory):
    tar_files = sorted([f for f in os.listdir(directory) if f.endswith('.tar')])
    if not tar_files:
        print(f"No TAR files found in {directory}")
        return

    for tar_name in tar_files:
        tar_path = os.path.join(directory, tar_name)
        print(f"Scanning {tar_name}...")
        with tarfile.open(tar_path, "r") as tar:
            for member in tar.getmembers():
                if member.isfile() and member.name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    try:
                        f = tar.extractfile(member)
                        if f:
                            img = Image.open(io.BytesIO(f.read())).convert("RGB")
                            yield img, os.path.basename(member.name)
                    except Exception as e:
                        print(f"Error reading {member.name}: {e}")
                        continue

# ----------------------------
# 4. INFERENCE LOOP
# ----------------------------
print("--- Starting Inference ---")

embeddings = []
metadata = []

batch_imgs = []
batch_keys = []

with torch.no_grad():
    for img, key in stream_images_from_tars(SHARD_DIR):
        batch_imgs.append(preprocess(img))
        batch_keys.append(key)

        if len(batch_imgs) == BATCH_SIZE:
            input_tensor = torch.stack(batch_imgs).to(DEVICE, non_blocking=True)

            feat = model.encode_image(input_tensor)
            feat /= feat.norm(dim=-1, keepdim=True)

            embeddings.append(feat.cpu().numpy())
            metadata.extend([{"key": k} for k in batch_keys])

            batch_imgs.clear()
            batch_keys.clear()

            if len(metadata) % 5000 == 0:
                print(f"Processed {len(metadata)} images...")

    # Final partial batch
    if batch_imgs:
        input_tensor = torch.stack(batch_imgs).to(DEVICE, non_blocking=True)
        feat = model.encode_image(input_tensor)
        feat /= feat.norm(dim=-1, keepdim=True)

        embeddings.append(feat.cpu().numpy())
        metadata.extend([{"key": k} for k in batch_keys])

# ----------------------------
# 5. BUILD ULTRA-FAST IVF INDEX
# ----------------------------
if not embeddings:
    print("FATAL: No images found.")
    exit()

print(f"\nSuccess! Building IVF index for {len(metadata)} vectors...")

emb_np = np.vstack(embeddings).astype("float32")
dim = emb_np.shape[1]

# Normalize for cosine similarity
faiss.normalize_L2(emb_np)

# Quantizer
quantizer = faiss.IndexFlatIP(dim)

# IVF Index
index = faiss.IndexIVFFlat(
    quantizer,
    dim,
    NLIST,
    faiss.METRIC_INNER_PRODUCT
)

# Train on subset for speed
train_samples = emb_np[np.random.choice(
    emb_np.shape[0],
    min(TRAINING_SIZE, emb_np.shape[0]),
    replace=False
)]

print("Training IVF clusters...")
index.train(train_samples)

print("Adding vectors to index...")
index.add(emb_np)

index.nprobe = NPROBE

# ----------------------------
# 6. SAVE (SAME FINAL NAMES)
# ----------------------------
faiss.write_index(index, "shoptalk_index.faiss")
pickle.dump(metadata, open("metadata.pkl", "wb"))

print("\n✅ DONE.")
print("Saved:")
print(" - shoptalk_index.faiss")
print(" - metadata.pkl")