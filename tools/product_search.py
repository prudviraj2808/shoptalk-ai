import faiss
import pickle
import numpy as np
import torch
import mobileclip
from google_adk.types import Tool # Adjust based on your ADK version

class ProductSearchTool:
    def __init__(self, index_path="shoptalk_index.faiss", meta_path="metadata.pkl"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.index = faiss.read_index(index_path)
        
        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)
            
        # Load your MobileCLIP model
        # Note: Ensure the model path matches your local unzipped folder
        self.model, _, _ = mobileclip.create_model_and_transforms(
            'mobileclip_s0', 
            pretrained='./mobileclip2_lora'
        )
        self.model.to(self.device)
        self.model.eval()

    def search(self, query: str, top_k: int = 3):
        """
        Search for clothing and fashion products based on a natural language description.
        Args:
            query: A descriptive string (e.g., 'blue denim jacket' or 'floral summer dress').
            top_k: Number of products to return.
        Returns:
            A list of matching products with their captions.
        """
        # Encode text to embedding
        text_tokens = mobileclip.tokenize([query]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        
        # Query FAISS
        query_np = text_features.cpu().numpy().astype('float32')
        distances, indices = self.index.search(query_np, top_k)
        
        results = []
        for i in range(top_k):
            idx = indices[0][i]
            if idx != -1: # Ensure a match was found
                results.append({
                    "product_id": self.metadata[idx]['key'],
                    "description": self.metadata[idx]['txt'],
                    "similarity_score": float(distances[0][i])
                })
        
        return results