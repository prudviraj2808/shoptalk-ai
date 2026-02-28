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
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        BASE_DIR = os.path.abspath(os.path.join(current_dir, ".."))

        self.index_path = os.path.join(BASE_DIR, "shoptalk_index.faiss")
        self.meta_path = os.path.join(BASE_DIR, "metadata.pkl")
        self.lora_dir = os.path.join(BASE_DIR, "model", "mobileclip2_lora")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        print(f"📦 Pre-loading Search Engine on {self.device}...")

        self.index = faiss.read_index(self.index_path)
        with open(self.meta_path, "rb") as f:
            self.metadata = pickle.load(f)

        try:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                "MobileCLIP2-S2", pretrained="dfndr2b"
            )
            if os.path.exists(self.lora_dir):
                self.model = PeftModel.from_pretrained(self.model, self.lora_dir)
                self.model = self.model.merge_and_unload()

            self.model = reparameterize_model(self.model)
            self.model = self.model.to(device=self.device, dtype=self.dtype).eval()
            self.tokenizer = open_clip.get_tokenizer("MobileCLIP2-S2")
            
            ProductSearchTool._initialized = True
            print(f"✅ ProductSearchTool Ready.")
        except Exception as e:
            print(f"❌ Initialization Error: {e}")
            raise

    def search_text(self, query: str, top_k: int = 3):
        tokens = self.tokenizer([query]).to(self.device)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(self.device == "cuda")):
            features = self.model.encode_text(tokens)
            features /= features.norm(dim=-1, keepdim=True)
        return self._faiss_query(features, top_k)

    def _faiss_query(self, embedding, top_k):
        query_np = embedding.cpu().numpy().astype("float32")
        distances, indices = self.index.search(query_np, top_k)
        results = []
        for i in range(min(top_k, len(indices[0]))):
            idx = indices[0][i]
            if idx != -1 and idx < len(self.metadata):
                results.append({"score": round(float(distances[0][i]), 4), "data": self.metadata[idx]})
        return results

# This line ensures it loads on import
visual_search_tool = ProductSearchTool()