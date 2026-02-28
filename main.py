import os
from dotenv import load_dotenv

# This import triggers the model load BEFORE the app starts serving
from tools.product_search import visual_search_tool 

from google.adk.cli.fast_api import get_fast_api_app
from google.adk.sessions import DatabaseSessionService
from utils.database import init_db

load_dotenv()

session_service = DatabaseSessionService(db_url=os.getenv("DATABASE_URL"))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AGENT_DIR = os.path.join(BASE_DIR, "agents")

app = get_fast_api_app(
    agents_dir=AGENT_DIR,
    web=True,
    allow_origins=["*"],
)

@app.on_event("startup")
async def startup_event():
    await init_db()
    print("🚀 ShopTalk AI: Warm and ready for Google ADK UI sessions.")