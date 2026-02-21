import os
import glob
import logging
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
import open_clip
import webdataset as wds

from peft import LoraConfig, get_peft_model
from tqdm import tqdm


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# ============================================================
# Configuration (Environment Driven)
# ============================================================

MODEL_NAME = os.environ.get("MODEL_NAME", "MobileCLIP2-S2")
PRETRAINED = os.environ.get("PRETRAINED", "dfndr2b")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = DEVICE == "cuda"

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 16))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", 5e-5))
EPOCHS = int(os.environ.get("EPOCHS", 3))
NUM_WORKERS = 0 if DEVICE == "cpu" else int(os.environ.get("NUM_WORKERS", 4))

TAR_PATTERN = os.environ.get(
    "DATA_PATTERN",
    os.path.join(os.getcwd(), "mobileclip_data_*.tar")
)

OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR",
    os.path.join(os.getcwd(), "output/mobileclip2_lora")
)


# ============================================================
# Dataset Preprocessing
# ============================================================

def preprocess_sample(sample, preprocess, tokenizer):
    img, caption = sample
    img = preprocess(img)
    # We squeeze(0) because tokenizer(caption) often returns [1, SeqLen]
    # and wds.batched() expects a flat list of samples to stack.
    txt = tokenizer(caption).squeeze(0)
    return img, txt


# ============================================================
# Training Function
# ============================================================

def train():

    logging.info("========== CONFIG ==========")
    logging.info(f"MODEL_NAME={MODEL_NAME}")
    logging.info(f"PRETRAINED={PRETRAINED}")
    logging.info(f"DEVICE={DEVICE}")
    logging.info(f"AMP_ENABLED={USE_AMP}")
    logging.info(f"BATCH_SIZE={BATCH_SIZE}")
    logging.info(f"LR={LEARNING_RATE}")
    logging.info(f"EPOCHS={EPOCHS}")
    logging.info(f"NUM_WORKERS={NUM_WORKERS}")
    logging.info(f"TAR_PATTERN={TAR_PATTERN}")
    logging.info("============================")

    # --------------------------------------------------------
    # 1. Load Model
    # --------------------------------------------------------

    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=PRETRAINED
    )
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)

    model = model.to(DEVICE)
    logging.info("Base model loaded.")

    # --------------------------------------------------------
    # 2. Apply LoRA
    # --------------------------------------------------------

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "c_proj", "fc1", "fc2"],
        lora_dropout=0.05,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    logging.info("LoRA adapters attached.")

    # --------------------------------------------------------
    # 3. Dataset Setup
    # --------------------------------------------------------

    tar_files = glob.glob(TAR_PATTERN)

    if len(tar_files) == 0:
        raise RuntimeError(f"No tar files found matching {TAR_PATTERN}")

    fixed_paths = []
    for f in tar_files:
        p = Path(f).resolve()
        if os.name == "nt":
            fixed_paths.append("file:" + p.as_posix())
        else:
            fixed_paths.append(p.as_posix())

    tar_files = fixed_paths

    logging.info(f"Found {len(tar_files)} tar files.")

    preprocess_fn = partial(
        preprocess_sample,
        preprocess=preprocess,
        tokenizer=tokenizer
    )

    # Dataset pipeline: Decode -> Tuple -> Map -> Batch
    dataset = (
        wds.WebDataset(tar_files, shardshuffle=False)
        .shuffle(1000)
        .decode("pil")
        .to_tuple("jpg", "txt")
        .map(preprocess_fn)
        .batched(BATCH_SIZE, partial=False) # Ensure we always get BATCH_SIZE
    )

    # THE CRITICAL FIX: Set batch_size=None because .batched() 
    # already handled the batching in the pipeline above.
    loader = wds.WebLoader(
        dataset,
        batch_size=None, 
        num_workers=NUM_WORKERS,
        persistent_workers=(NUM_WORKERS > 0)
    )

    logging.info("Dataset initialized.")

    # --------------------------------------------------------
    # 4. Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE
    )

    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    # --------------------------------------------------------
    # 5. Training Loop
    # --------------------------------------------------------

    model.train()

    for epoch in range(EPOCHS):
        logging.info(f"Starting epoch {epoch+1}/{EPOCHS}")
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}")

        for batch_idx, (imgs, txts) in enumerate(pbar):
            # Move to device
            imgs = imgs.to(DEVICE, non_blocking=True)
            txts = txts.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=USE_AMP):
                # CLIP Forward Pass
                image_features = model.encode_image(imgs)
                text_features = model.encode_text(txts)

                # Normalize
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                # Scaled pairwise cosine similarities
                logit_scale = model.logit_scale.exp().clamp(max=100)
                logits_per_image = image_features @ text_features.T * logit_scale
                logits_per_text = logits_per_image.T

                # Symmetric Cross Entropy Loss
                ground_truth = torch.arange(len(imgs), device=DEVICE)
                loss_i = F.cross_entropy(logits_per_image, ground_truth)
                loss_t = F.cross_entropy(logits_per_text, ground_truth)
                loss = (loss_i + loss_t) / 2

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            pbar.set_postfix(loss=f"{loss.item():.4f}")

            if batch_idx % 50 == 0:
                logging.info(f"Epoch {epoch+1} Batch {batch_idx}: Loss={loss.item():.4f}")

        logging.info(f"Epoch {epoch+1} completed.")

    # --------------------------------------------------------
    # 6. Save LoRA Weights
    # --------------------------------------------------------

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    logging.info(f"LoRA weights saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    train()