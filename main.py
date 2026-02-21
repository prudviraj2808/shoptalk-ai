import os
from dotenv import load_dotenv

from google.adk.cli.fast_api import get_fast_api_app
from google.adk.sessions import DatabaseSessionService
from utils.database import init_db

load_dotenv()

# 1️⃣ Setup the Persistent Session Service
# This connects the ADK UI and API to your Postgres DB
session_service = DatabaseSessionService(
    db_url=os.getenv("DATABASE_URL")
)

# 2️⃣ Define the STRICT directory for agents
# This prevents 'src' and other folders from appearing in the UI dropdown
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AGENT_DIR = os.path.join(BASE_DIR, "agents")

# 3️⃣ Create the FastAPI app with the restricted discovery
app = get_fast_api_app(
    agents_dir=AGENT_DIR,            # Only look inside /agents
    web=True,                         # Enable the /dev-ui/ interface
    allow_origins=["*"],
)

# 4️⃣ Database Startup
@app.on_event("startup")
async def startup_event():
    await init_db()