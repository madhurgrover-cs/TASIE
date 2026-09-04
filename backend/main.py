from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from backend.database import engine, Base
from backend.api.routes import router

# Project root (…/sast-iq-main), independent of the process working directory.
BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

# Auto-create all tables on startup
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Incremental Learning SAST Platform",
    description="Self-improving SAST that learns from developer feedback.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    # A wildcard origin cannot be combined with credentialed requests per the
    # CORS spec; the dashboard is same-origin and sends no credentials anyway.
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

# Serve the frontend dashboard at /dashboard
if FRONTEND_DIR.is_dir():
    app.mount("/dashboard", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


@app.get("/")
def root():
    return {
        "status": "running",
        "docs": "/docs",
        "dashboard": "/dashboard",
    }
