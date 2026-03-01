import os
import faiss
import pickle
import torch
import open_clip
from PIL import Image
from timm.utils import reparameterize_model
from peft import PeftModel


class ProductSearchTool:
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ProductSearchTool, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if ProductSearchTool._initialized:
            return

        print("🔥 Initializing ProductSearchTool...")
        print("PID:", os.getpid())

        # -------- Paths --------
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.BASE_DIR = os.path.abspath(os.path.join(current_dir, ".."))

        self.index_path = os.path.join(self.BASE_DIR, "shoptalk_index.faiss")
        self.meta_path = os.path.join(self.BASE_DIR, "metadata.pkl")
        self.lora_dir = os.path.join(self.BASE_DIR, "model", "mobileclip2_lora")

        # Image root folder
        self.image_root = os.path.join(
            self.BASE_DIR,
            "data",
            "abo-images-small",
            "images",
            "small"
        )

        # -------- Device --------
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        print(f"📦 Loading on {self.device}")

        # -------- Load FAISS --------
        self.index = faiss.read_index(self.index_path)

        # -------- Load Metadata --------
        with open(self.meta_path, "rb") as f:
            self.metadata = pickle.load(f)

        # -------- Load Model --------
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "MobileCLIP2-S2",
            pretrained="dfndr2b"
        )

        if os.path.exists(self.lora_dir):
            self.model = PeftModel.from_pretrained(self.model, self.lora_dir)
            self.model = self.model.merge_and_unload()
            print("✅ LoRA merged")

        self.model = reparameterize_model(self.model)
        self.model = self.model.to(self.device, dtype=self.dtype).eval()
        self.tokenizer = open_clip.get_tokenizer("MobileCLIP2-S2")

        ProductSearchTool._initialized = True
        print(f"✅ ProductSearchTool Ready ({len(self.metadata)} items)")

    # -------- SEARCH METHODS --------

    def search_text(self, query: str, top_k: int = 3):
        tokens = self.tokenizer([query]).to(self.device)

        with torch.no_grad():
            features = self.model.encode_text(tokens)
            features /= features.norm(dim=-1, keepdim=True)

        return self._faiss_query(features, top_k)

    def search_image(self, image_path: str, top_k: int = 3):
        image = Image.open(image_path).convert("RGB")
        img_tensor = (
            self.preprocess(image)
            .unsqueeze(0)
            .to(self.device, dtype=self.dtype)
        )

        with torch.no_grad():
            features = self.model.encode_image(img_tensor)
            features /= features.norm(dim=-1, keepdim=True)

        return self._faiss_query(features, top_k)

    # -------- INTERNAL METHODS --------

    def _build_image_path(self, filename: str):
        """
        Builds full ABO image path:
        small/<first_two_chars>/<filename>
        """
        if not filename:
            return None

        subfolder = filename[:2]
        return os.path.join(self.image_root, subfolder, filename)

    def _faiss_query(self, embedding, top_k):
        query_np = embedding.cpu().numpy().astype("float32")
        _, indices = self.index.search(query_np, top_k)

        image_paths = []

        for i in range(min(top_k, len(indices[0]))):
            idx = indices[0][i]

            if idx != -1 and idx < len(self.metadata):

                meta = self.metadata[idx]
                filename = meta.get("key")

                full_path = self._build_image_path(filename)

                if full_path and os.path.exists(full_path):
                    image_paths.append(full_path)

        return image_paths


# -------- Singleton Getter --------

_visual_search_instance = None


def get_visual_search_tool():
    global _visual_search_instance
    if _visual_search_instance is None:
        _visual_search_instance = ProductSearchTool()
    return _visual_search_instance