import os
from dotenv import load_dotenv

from google.adk.agents import LlmAgent
from google.adk.models import Gemini
from tools.product_search import get_visual_search_tool

load_dotenv()

# -------- MODEL --------

model_config = Gemini(
    model="gemini-2.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY")
)


# -------- TOOL FUNCTIONS --------

def search_by_text(query: str):
    tool = get_visual_search_tool()
    return tool.search_text(query, top_k=3)


def search_by_image(image_path: str):
    tool = get_visual_search_tool()
    return tool.search_image(image_path, top_k=3)


# -------- AGENT --------

root_agent = LlmAgent(
    name="ShopTalk",
    model=model_config,
    description="Searches for products using text or visual inputs.",
instruction = """
You are 'ShopTalk', a visual shopping assistant.
RULES:
- If user describes a product → call search_by_text
- If user provides an image  → call search_by_image

RESPONSE FORMAT:
For each matched product, return:
• Product ID: <product_id>
• Image:
  ![Product](/images/small/<subfolder>/<filename>.jpg)

IMPORTANT:
- Show 3 best matches unless user specifies otherwise.
""",
    tools=[search_by_text, search_by_image],
)