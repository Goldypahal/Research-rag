from __future__ import annotations
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from typing import List, Optional
from ..models.chunk import Chunk
from ..core.settings import settings
from ..core.retry_utils import (
    APIRateLimitError,
    APIServerError,
    APITimeoutError,
    retry_api_call,
)

class ChromaIndex:
    def __init__(self):
        self.embeddings = OllamaEmbeddings(
            model=settings.OLLAMA_EMBED_MODEL,
            base_url=settings.OLLAMA_BASE_URL
        )
        self.vector_store = Chroma(
            persist_directory=settings.CHROMA_DB_PATH,
            embedding_function=self.embeddings,
            collection_name="research_papers"
        )

    @retry_api_call(max_attempts=4, min_wait=1, max_wait=10)
    def _add_texts_with_retry(self, texts: List[str], metadatas: List[dict], ids: List[str]):
        try:
            self.vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        except Exception as exc:
            msg = str(exc).lower()
            if "timeout" in msg:
                raise APITimeoutError(str(exc)) from exc
            if "429" in msg or "rate" in msg:
                raise APIRateLimitError(str(exc)) from exc
            if any(x in msg for x in ["500", "502", "503", "504", "server"]):
                raise APIServerError(str(exc)) from exc
            raise

    def add_chunks(self, chunks: List[Chunk]):
        texts = [c.text for c in chunks]
        metadatas = [c.flat_metadata() for c in chunks]
        ids = [c.chunk_id for c in chunks]
        self._add_texts_with_retry(texts, metadatas, ids)

    @retry_api_call(max_attempts=5, min_wait=1, max_wait=10)
    def query(self, query: str, k: int = 20, filters: Optional[dict] = None):
        try:
            return self.vector_store.similarity_search_with_score(query, k=k, filter=filters)
        except Exception as exc:
            msg = str(exc).lower()
            if "timeout" in msg:
                raise APITimeoutError(str(exc)) from exc
            if "429" in msg or "rate" in msg:
                raise APIRateLimitError(str(exc)) from exc
            if any(x in msg for x in ["500", "502", "503", "504", "server"]):
                raise APIServerError(str(exc)) from exc
            raise
