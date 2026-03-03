import logging
import os
import pickle

import faiss
import open_clip
import torch
from PIL import Image
from peft import PeftModel
from timm.utils import reparameterize_model

logger = logging.getLogger(__name__)


class ProductSearchTool:
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if ProductSearchTool._initialized:
            return

        logger.info("Initializing ProductSearchTool...")

        BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        # Paths — overridable via env vars per environment (local, Docker, S3)
        self.index_path = os.getenv(
            "FAISS_INDEX_PATH",
            os.path.join(BASE_DIR, "vector-index", "shoptalk_index.faiss"),
        )
        self.meta_path = os.getenv(
            "METADATA_PATH",
            os.path.join(BASE_DIR, "vector-index", "metadata.pkl"),
        )
        self.lora_dir = os.getenv(
            "LORA_ADAPTER_PATH",
            os.path.join(BASE_DIR, "model", "mobileclip2_lora"),
        )

        # Device — reads DEVICE env var first, then auto-detects
        # CI/CD sets DEVICE=cuda on g5.2xlarge; local dev defaults to cpu
        requested_device = os.getenv("DEVICE", "auto")

        if requested_device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif requested_device == "cuda" and not torch.cuda.is_available():
            logger.warning(
                "DEVICE=cuda requested but no CUDA GPU found. "
                "Falling back to cpu. Check NVIDIA drivers and nvidia-container-toolkit."
            )
            self.device = "cpu"
        else:
            self.device = requested_device

        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        logger.info("Loading on device=%s dtype=%s", self.device, self.dtype)

        # Validate asset paths — fail fast with a clear message
        for label, path in [("FAISS index", self.index_path), ("metadata", self.meta_path)]:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"{label} not found at '{path}'. "
                    "Run 'aws s3 sync' to pull assets, or check your volume mounts."
                )

        # Load FAISS index
        logger.info("Loading FAISS index from %s", self.index_path)
        self.index = faiss.read_index(self.index_path)

        if isinstance(self.index, faiss.IndexIVF):
            # nprobe tunable via env: raise for better recall, lower for speed
            self.index.nprobe = int(os.getenv("FAISS_NPROBE", "16"))
            logger.info("IVF index — nprobe=%d", self.index.nprobe)

        # Load metadata
        logger.info("Loading metadata from %s", self.meta_path)
        with open(self.meta_path, "rb") as f:
            self.metadata = pickle.load(f)
        logger.info("Metadata loaded — %d items", len(self.metadata))

        # Load MobileCLIP2 base model
        model_name = os.getenv("MODEL_NAME", "MobileCLIP2-S2")
        pretrained = os.getenv("PRETRAINED", "dfndr2b")
        logger.info("Loading base model %s pretrained=%s", model_name, pretrained)

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
        )

        # Apply LoRA adapter
        if os.path.exists(self.lora_dir):
            logger.info("Applying LoRA adapter from %s", self.lora_dir)
            self.model = PeftModel.from_pretrained(self.model, self.lora_dir)
            self.model = self.model.merge_and_unload()
            logger.info("LoRA merged successfully")
        else:
            logger.warning(
                "LoRA adapter not found at '%s' — using base model weights. "
                "Check LORA_ADAPTER_PATH or S3 sync.",
                self.lora_dir,
            )

        self.model = reparameterize_model(self.model)
        self.model = self.model.to(self.device, dtype=self.dtype).eval()

        # cuDNN autotuner — speeds up fixed-size inference on A10G GPU
        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True
            logger.info("cuDNN benchmark mode enabled")

        self.tokenizer = open_clip.get_tokenizer(model_name)

        ProductSearchTool._initialized = True
        logger.info(
            "ProductSearchTool ready — %d items indexed, device=%s",
            len(self.metadata),
            self.device,
        )

    # Search methods

    def search_text(self, query: str, top_k: int = 3):
        tokens = self.tokenizer([query]).to(self.device)

        with torch.no_grad(), torch.autocast(device_type=self.device, enabled=self.device == "cuda"):
            features = self.model.encode_text(tokens)
            features /= features.norm(dim=-1, keepdim=True)

        return self._faiss_query(features, top_k)

    def search_image(self, image_path: str, top_k: int = 3):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Query image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        img_tensor = (
            self.preprocess(image)
            .unsqueeze(0)
            .to(self.device, dtype=self.dtype)
        )

        with torch.no_grad(), torch.autocast(device_type=self.device, enabled=self.device == "cuda"):
            features = self.model.encode_image(img_tensor)
            features /= features.norm(dim=-1, keepdim=True)

        return self._faiss_query(features, top_k)

    # Internal FAISS query

    def _faiss_query(self, embedding, top_k: int):
        query_np = embedding.detach().cpu().float().numpy()
        distances, indices = self.index.search(query_np, top_k)

        results = []
        for i in range(min(top_k, len(indices[0]))):
            idx = indices[0][i]
            if idx != -1 and idx < len(self.metadata):
                filename = self.metadata[idx].get("key")
                if filename:
                    subfolder = filename[:2]
                    results.append(f"small/{subfolder}/{filename}")

        logger.debug("FAISS query returned %d results", len(results))
        return results


# Singleton getter

_visual_search_instance = None


def get_visual_search_tool() -> ProductSearchTool:
    global _visual_search_instance
    if _visual_search_instance is None:
        _visual_search_instance = ProductSearchTool()
    return _visual_search_instance