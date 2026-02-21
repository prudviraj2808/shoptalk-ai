import os
import glob
import math
import logging
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
import open_clip
import webdataset as wds

from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from torchvision import transforms


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# ============================================================
# Configuration
# ============================================================ 

MODEL_NAME = os.environ.get("MODEL_NAME", "MobileCLIP2-S2")
PRETRAINED = os.environ.get("PRETRAINED", "dfndr2b")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = DEVICE == "cuda"

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 64))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", 3e-5))
EPOCHS = int(os.environ.get("EPOCHS", 8))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 8))

DATASET_SIZE = int(os.environ.get("DATASET_SIZE", 50000))

WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
LABEL_SMOOTHING = 0.1
GRAD_CLIP = 1.0

TAR_PATTERN = os.environ.get(
    "DATA_PATTERN",
    os.path.join(os.getcwd(), "mobileclip_data_*.tar")
)

OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR",
    os.path.join(os.getcwd(), "output/mobileclip2_lora")
)


# ============================================================
# Augmentation (Fixed Properly)
# ============================================================

def build_transforms(base_preprocess):

    normalize = None
    for t in base_preprocess.transforms:
        if isinstance(t, transforms.Normalize):
            normalize = t

    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        normalize,
    ])


def preprocess_sample(sample, preprocess, tokenizer):
    img, caption = sample
    img = preprocess(img)
    txt = tokenizer(caption).squeeze(0)
    return img, txt


# ============================================================
# Training
# ============================================================

def train():

    logging.info("Loading model...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=PRETRAINED
    )

    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    preprocess = build_transforms(preprocess)

    model = model.to(DEVICE)

    if DEVICE == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    # --------------------------------------------------------
    # LoRA (Reduced Rank to Avoid Overfit)
    # --------------------------------------------------------

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "c_proj", "fc1", "fc2"],
        lora_dropout=0.1,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    tar_files = glob.glob(TAR_PATTERN)
    if len(tar_files) == 0:
        raise RuntimeError(f"No tar files found matching {TAR_PATTERN}")

    tar_files = [Path(f).resolve().as_posix() for f in tar_files]

    preprocess_fn = partial(
        preprocess_sample,
        preprocess=preprocess,
        tokenizer=tokenizer
    )

    dataset = (
        wds.WebDataset(tar_files, shardshuffle=True, empty_check=False)
        .shuffle(5000)
        .decode("pil")
        .to_tuple("jpg", "txt")
        .map(preprocess_fn)
        .batched(BATCH_SIZE)
    )

    loader = wds.WebLoader(
        dataset,
        batch_size=None,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
    )

    logging.info("Dataset ready.")

    # --------------------------------------------------------
    # Optimizer + Cosine Scheduler
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        fused=(DEVICE == "cuda")
    )

    steps_per_epoch = DATASET_SIZE // BATCH_SIZE
    total_steps = steps_per_epoch * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    # --------------------------------------------------------
    # Training Loop
    # --------------------------------------------------------

    global_step = 0
    model.train()

    for epoch in range(EPOCHS):
        logging.info(f"Starting epoch {epoch+1}/{EPOCHS}")
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}")

        for imgs, txts in pbar:

            imgs = imgs.to(DEVICE, non_blocking=True)
            txts = txts.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=USE_AMP):

                image_features = model.encode_image(imgs)
                text_features = model.encode_text(txts)

                image_features = F.normalize(image_features, dim=-1)
                text_features = F.normalize(text_features, dim=-1)

                logit_scale = model.logit_scale.exp().clamp(max=100)
                logits = image_features @ text_features.T * logit_scale

                labels = torch.arange(len(imgs), device=DEVICE)

                loss_i = F.cross_entropy(logits, labels, label_smoothing=LABEL_SMOOTHING)
                loss_t = F.cross_entropy(logits.T, labels, label_smoothing=LABEL_SMOOTHING)
                loss = (loss_i + loss_t) / 2

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            global_step += 1

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}"
            )

        logging.info(f"Epoch {epoch+1} completed.")

    # --------------------------------------------------------
    # Save LoRA Weights
    # --------------------------------------------------------

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)

    logging.info("Training complete.")
    logging.info(f"LoRA adapters saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    train()