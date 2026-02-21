import os
from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

# 1. Import your tool class
from product_search import ProductSearchTool 

load_dotenv()

# SageMaker Llama endpoint
ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "your-llama-endpoint-name")

model = LiteLlm(
    model=f"sagemaker/{ENDPOINT_NAME}",
    aws_region_name="us-east-1",
)

# 2. Initialize the search engine
# This loads the FAISS index and MobileCLIP model into memory once
search_engine = ProductSearchTool()

# 3. Define the tool function for the ADK
# The ADK uses this docstring to tell Llama WHEN to call the search
def product_retrieval_tool(query: str):
    """
    Search the local product catalog for fashion items, clothing, and accessories 
    using a natural language description. Use this whenever the user asks for 
    recommendations or specific items.
    """
    return search_engine.search(query, top_k=3)

# 4. ShopTalk Agent Definition
root_agent = LlmAgent(
    name="ShopTalk",
    model=model,
    instruction="""
    You are a personalized shopping assistant. 
    When a user asks for a product, use the 'product_retrieval_tool' to find items.
    Always describe the items found and explain why they match the user's request.
    If no items are found, suggest a different search term.
    """,
    tools=[product_retrieval_tool], # Add your tool here
)