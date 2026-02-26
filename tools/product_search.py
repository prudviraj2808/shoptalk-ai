import os
import faiss
import pickle
import torch
import open_clip
from PIL import Image
from timm.utils import reparameterize_model
from peft import PeftModel  # Required for LoRA loading

class ProductSearchTool:
    def __init__(self):
        # 1. Setup Paths
        # Using /app/model/ as requested
        if os.path.exists("/app/shoptalk_index.faiss"):
            BASE_DIR = "/app"
        else:
            # Local Windows Logic fallback
            current_dir = os.path.dirname(os.path.abspath(__file__))
            BASE_DIR = os.path.abspath(os.path.join(current_dir, ".."))

        self.index_path = os.path.join(BASE_DIR, "shoptalk_index.faiss")
        self.meta_path = os.path.join(BASE_DIR, "metadata.pkl")
        
        # Path to the directory containing adapter_config.json
        self.lora_dir = os.path.join(BASE_DIR, "model", "mobileclip2_lora")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"🔍 Path Debugging:")
        print(f"   - BASE_DIR: {BASE_DIR}")
        print(f"   - LoRA Dir: {self.lora_dir}")

        # 2. Load FAISS Index
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"❌ FAISS index not found! {self.index_path}")
        self.index = faiss.read_index(self.index_path)
        print(f"✅ FAISS Index loaded.")

        # 3. Load Metadata
        if not os.path.exists(self.meta_path):
            raise FileNotFoundError(f"❌ Metadata PKL not found! {self.meta_path}")
        with open(self.meta_path, "rb") as f:
            self.metadata = pickle.load(f)
        print(f"✅ Metadata loaded ({len(self.metadata)} items).")

        # 4. Initialize MobileCLIP2-S2 with LoRA
        print(f"🧠 Loading MobileCLIP2-S2 + LoRA on {self.device}...")
        try:
            # Step A: Load the base architecture
            # We use 'apple_s2' as the base weights for MobileCLIP2-S2
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                "MobileCLIP2-S2",
                pretrained="dfndr2b" 
            )

            # Step B: Apply LoRA adapters
            if os.path.exists(self.lora_dir):
                # PeftModel wraps the open_clip model and injects the adapter layers
                self.model = PeftModel.from_pretrained(self.model, self.lora_dir)
                # Merge LoRA weights into the base model for faster inference
                self.model = self.model.merge_and_unload()
                print(f"✅ LoRA adapters merged successfully.")
            else:
                print(f"⚠️ LoRA directory not found. Running with base model only.")

            # Step C: Final Optimization
            self.model = reparameterize_model(self.model)
            self.model = self.model.to(self.device).eval()
            self.tokenizer = open_clip.get_tokenizer("MobileCLIP2-S2")
            print(f"🚀 ProductSearchTool ready.")

        except Exception as e:
            print(f"❌ Failed to initialize Model: {e}")
            raise

    def search_text(self, query: str, top_k: int = 3):
        tokens = self.tokenizer([query]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return self._faiss_query(text_features, top_k)

    def search_image(self, image_path: str, top_k: int = 3):
        if not os.path.exists(image_path):
            return {"error": f"Image file not found at {image_path}"}
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return self._faiss_query(image_features, top_k)

    def _faiss_query(self, embedding, top_k):
        query_np = embedding.cpu().numpy().astype("float32")
        distances, indices = self.index.search(query_np, top_k)
        results = []
        for i in range(min(top_k, len(indices[0]))):
            idx = indices[0][i]
            if idx != -1 and idx < len(self.metadata):
                item = self.metadata[idx]
                results.append({
                    "key": item.get("key", "unknown"),
                    "similarity_score": round(float(distances[0][i]), 4),
                    "metadata": item 
                })
        return results