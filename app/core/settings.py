from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    CHROMA_DB_PATH: str = "data/chroma_db"
    BM25_INDEX_PATH: str = "data/bm25_index.pkl"
    COHERE_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    GOOGLE_API_KEY: Optional[str] = None
    OLLAMA_MODEL: str = "qwen3:14b"
    OLLAMA_EMBED_MODEL: str = "nomic-embed-text"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    LOG_LEVEL: str = "INFO"
    
    # RAG Settings
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100
    
    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()
