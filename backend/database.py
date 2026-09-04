import os
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Default the SQLite file to the project root so it doesn't depend on the
# process working directory. Override with the DATABASE_URL env var.
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DB_URL = f"sqlite:///{(BASE_DIR / 'sast_learning.db').as_posix()}"
DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_DB_URL)

# check_same_thread is a SQLite-only connect arg.
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
