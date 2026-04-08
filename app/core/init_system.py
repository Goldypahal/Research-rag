import os
import logging
from .settings import settings

logger = logging.getLogger(__name__)

def ensure_directories():
    directories = [
        "data/chroma_db",
        "data/bm25_index",
        "data/feedback",
        "data/traces",
        "data/golden_eval",
        "data/uploads"
    ]
    for d in directories:
        if not os.path.exists(d):
            os.makedirs(d, exist_ok=True)
            logger.info(f"Created directory: {d}")
        else:
            logger.debug(f"Directory already exists: {d}")

if __name__ == "__main__":
    ensure_directories()
