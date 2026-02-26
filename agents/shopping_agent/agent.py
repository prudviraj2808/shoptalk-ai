import os
from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

# 1. Import your tool class 
# Ensure your file is named product_search.py or match the filename
from tools.product_search import ProductSearchTool

load_dotenv()

# SageMaker Llama endpoint configuration
ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "your-llama-endpoint-name")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

# Initialize the LiteLlm model pointing to your SageMaker endpoint
model = LiteLlm(
    model=f"sagemaker/{ENDPOINT_NAME}",
    aws_region_name=AWS_REGION,
)

# 2. Initialize the search engine (Loads FAISS & MobileCLIP once)
search_engine = ProductSearchTool()

# 3. Define the Tool Functions for the ADK
# We separate them so the LLM knows which one to call based on input type

def search_by_text(query: str):
    """
    Search the product catalog using a text description (e.g., 'red summer dress'). 
    Use this when the user describes what they want in words.
    """
    return search_engine.search_text(query, top_k=3)

def search_by_image(image_path: str):
    """
    Search for products that look like an uploaded image. 
    The 'image_path' should be the local file path to the image.
    Use this when a user provides or refers to an image.
    """
    return search_engine.search_image(image_path, top_k=3)

# 4. ShopTalk Agent Definition
# 
root_agent = LlmAgent(
    name="ShopTalk",
    model=model,
    instruction="""
    You are 'ShopTalk', a personalized shopping assistant. 
    - If a user describes an item, use 'search_by_text'.
    - If a user mentions an image or provides a path, use 'search_by_image'.
    - When you get results, display the 'key' (Product ID) and the 'similarity_score'.
    - Be conversational: explain why these items match their style.
    - If the similarity scores are very low, suggest that the match might not be perfect.
    """,
    tools=[search_by_text, search_by_image],
)
