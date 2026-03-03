import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from google.adk.cli.fast_api import get_fast_api_app
from google.adk.sessions import DatabaseSessionService

from utils.database import init_db
from tools.product_search import get_visual_search_tool

load_dotenv()

# Structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

# Session service
session_service = DatabaseSessionService(
    db_url=os.getenv("DATABASE_URL")
)

# Agent directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AGENT_DIR = os.path.join(BASE_DIR, "agents")

# CORS — set ALLOWED_ORIGINS=https://yourfrontend.com in .env for production
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

# FastAPI app via Google ADK
IS_PRODUCTION = os.getenv("ENV", "production") == "production"

app = get_fast_api_app(
    agents_dir=AGENT_DIR,
    web=True,
    allow_origins=ALLOWED_ORIGINS,
)

# Disable Swagger/ReDoc/OpenAPI schema in production
if IS_PRODUCTION:
    app.docs_url = None
    app.redoc_url = None
    app.openapi_url = None

# Startup: load model + FAISS + init DB
@app.on_event("startup")
async def startup():
    logger.info("Starting ShopTalk... ENV=%s DEVICE=%s",
                os.getenv("ENV", "production"),
                os.getenv("DEVICE", "cpu"))

    logger.info("Loading FAISS index + MobileCLIP2 model...")
    get_visual_search_tool()
    logger.info("Model + FAISS fully loaded.")

    logger.info("Initialising database...")
    await init_db()
    logger.info("Database ready.")

    IMAGE_DIRECTORY = "/app/data/abo-images-small/images"
    if os.path.exists(IMAGE_DIRECTORY):
        app.mount(
            "/images",
            StaticFiles(directory=IMAGE_DIRECTORY),
            name="images",
        )
        logger.info("Serving images from %s", IMAGE_DIRECTORY)
    else:
        logger.warning(
            "Image directory not found at %s — check your Docker volume mount.",
            IMAGE_DIRECTORY,
        )

# Health check — required by CI/CD pipeline and load balancers
@app.get("/health", tags=["ops"], include_in_schema=False)
async def health():
    return {"status": "ok"}