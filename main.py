import os
from dotenv import load_dotenv

from fastapi.staticfiles import StaticFiles
from google.adk.cli.fast_api import get_fast_api_app
from google.adk.sessions import DatabaseSessionService

from utils.database import init_db
from tools.product_search import get_visual_search_tool

load_dotenv()

# -------- SESSION SERVICE -------- #

session_service = DatabaseSessionService(
    db_url=os.getenv("DATABASE_URL")
)

# -------- AGENT DIRECTORY -------- #

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AGENT_DIR = os.path.join(BASE_DIR, "agents")

# 🔥 FORCE MODEL + FAISS LOAD AT IMPORT TIME
print("🔥 Loading FAISS + Model...")
get_visual_search_tool()
print("🚀 Model + FAISS fully loaded.")

# -------- FASTAPI APP -------- #

app = get_fast_api_app(
    agents_dir=AGENT_DIR,
    web=True,
    allow_origins=["*"],
)

# -------- SERVE IMAGE DATASET -------- #

IMAGE_DIRECTORY = "/data/abo-images-small/images"

if os.path.exists(IMAGE_DIRECTORY):
    app.mount(
        "/images",
        StaticFiles(directory=IMAGE_DIRECTORY),
        name="images",
    )
    print(f"✅ Serving images from {IMAGE_DIRECTORY}")
else:
    print("⚠️ Image directory not found. Check Docker volume mount.")

