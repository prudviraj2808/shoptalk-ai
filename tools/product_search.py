import os
import faiss
import pickle
import torch
import open_clip
from PIL import Image
from timm.utils import reparameterize_model

class ProductSearchTool:
    def __init__(self):
        # 1. Setup Paths
        BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
        self.index_path = os.path.join(BASE_DIR, "shoptalk_index.faiss")
        self.meta_path = os.path.join(BASE_DIR, "metadata.pkl")
        
        # Local model checkpoint path
        self.model_checkpoint = os.path.join(BASE_DIR, "models", "mobileclip2_lora", "open_clip_pytorch_model.bin")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 2. Load FAISS Index
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"FAISS index not found: {self.index_path}")
        self.index = faiss.read_index(self.index_path)

        # 3. Load Metadata
        with open(self.meta_path, "rb") as f:
            self.metadata = pickle.load(f)

        # 4. Initialize MobileCLIP2-S2
        # We load the architecture 'MobileCLIP2-S2' and point to your local .bin
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "MobileCLIP2-S2",
            pretrained=self.model_checkpoint
        )
        
        # Optimize for inference (Reparameterization)
        self.model = reparameterize_model(self.model)
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer("MobileCLIP2-S2")

        print(f"✅ ProductSearchTool initialized on {self.device}")

    def search_text(self, query: str, top_k: int = 3):
        """Processes text query and returns best matches from FAISS."""
        tokens = self.tokenizer([query]).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.encode_text(tokens)
            # Normalize for Cosine Similarity
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
        return self._faiss_query(text_features, top_k)

    def search_image(self, image_path: str, top_k: int = 3):
        """Processes image file and returns best matches from FAISS."""
        if not os.path.exists(image_path):
            return {"error": f"Image file not found at {image_path}"}

        image = Image.open(image_path).convert("RGB")
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            # Normalize for Cosine Similarity
            image_features /= image_features.norm(dim=-1, keepdim=True)

        return self._faiss_query(image_features, top_k)

    def _faiss_query(self, embedding, top_k):
        """Internal helper to execute the FAISS search."""
        query_np = embedding.cpu().numpy().astype("float32")
        distances, indices = self.index.search(query_np, top_k)

        results = []
        for i in range(top_k):
            idx = indices[0][i]
            if idx != -1 and idx < len(self.metadata):
                item = self.metadata[idx]
                results.append({
                    "key": item.get("key", "unknown"),
                    "similarity_score": round(float(distances[0][i]), 4),
                    "metadata": item  # Includes category, price, etc. if in your pkl
                })
        return results