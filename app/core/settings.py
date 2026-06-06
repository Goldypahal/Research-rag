from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    CHROMA_DB_PATH: str = "data/chroma_db"
    BM25_INDEX_PATH: str = "data/bm25_index.pkl"

    # Cloud API keys — leave blank for fully local/private operation
    COHERE_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    GOOGLE_API_KEY: Optional[str] = None

    # Local Ollama settings
    OLLAMA_MODEL: str = "gemma2:2b"
    OLLAMA_EMBED_MODEL: str = "nomic-embed-text"
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # Mode toggles — both True = fully offline/private
    USE_LOCAL_LLM: bool = True        # Use Ollama for generation + citation enforcement
    USE_LOCAL_RERANKER: bool = True   # Use local SBERT instead of Cohere for reranking

    LOG_LEVEL: str = "INFO"

    # RAG chunking settings
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()

