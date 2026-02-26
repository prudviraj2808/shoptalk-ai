import os
import litellm  # Import the base library to set global flags
from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from tools.product_search import ProductSearchTool

load_dotenv()

# --- FIX 1: Set global LiteLLM behavior ---
# This tells LiteLLM to strip the 'tools' parameter before sending it to SageMaker
litellm.drop_params = True 

ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "your-llama-endpoint-name")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

# --- FIX 2: Pass allowed params to the Model Instance ---
model = LiteLlm(
    model=f"sagemaker/{ENDPOINT_NAME}",
    aws_region_name=AWS_REGION,
    # Some ADK versions prefer the param passed here as well
    drop_params=True 
)

search_engine = ProductSearchTool()

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