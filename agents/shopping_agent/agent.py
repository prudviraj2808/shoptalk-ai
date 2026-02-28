import os
import litellm
from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

# --- FIX 3: Import the PRE-LOADED instance ---
from tools.product_search import visual_search_tool

load_dotenv()

litellm.drop_params = True 

ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "your-llama-endpoint-name")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

model = LiteLlm(
    model=f"sagemaker/{ENDPOINT_NAME}",
    aws_region_name=AWS_REGION,
    drop_params=True 
)

# Use the hot instance we created in main.py
def search_by_text(query: str):
    """
    Search the product catalog using a text description (e.g., 'red summer dress'). 
    """
    return visual_search_tool.search_text(query, top_k=3)

def search_by_image(image_path: str):
    """
    Search for products that look like an uploaded image. 
    """
    return visual_search_tool.search_image(image_path, top_k=3)

root_agent = LlmAgent(
    name="ShopTalk",
    model=model,
    instruction="""
    You are 'ShopTalk', a personalized shopping assistant. 
    
    - If a user describes an item, use 'search_by_text'.
    - If a user mentions an image, use 'search_by_image'.
    
    STRICT RESPONSE RULES:
    1. For every product found, you MUST display the Product ID (key).
    2. You MUST provide the full image path found in the 'data' field of the tool results.
    3. IMPORTANT: To show the image in the UI, use the format: ![Product](path_to_image)
    4. Mention the similarity score so the user knows how confident we are.
    5. If no clear path is found, describe the item based on the metadata.
    """,
    tools=[search_by_text, search_by_image],
)