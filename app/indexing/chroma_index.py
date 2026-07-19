from typing import List, Optional
from ..models.chunk import Chunk
from ..core.settings import settings
from ..core.connectivity import mode_manager, LLMMode
from ..core.retry_utils import (
    APIRateLimitError,
    APIServerError,
    APITimeoutError,
    retry_api_call,
)

class ChromaIndex:
    def __init__(self):
        self._local_vector_store = None
        self._cloud_vector_store = None

    def _make_cloud_embeddings(self):
        """Build Google GenAI embeddings (langchain-google-genai >= 4.x)."""
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        # Use gemini-embedding-2 which is fully supported on the Developer API
        return GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-2",
            google_api_key=settings.GOOGLE_API_KEY,
        )

    def _make_local_embeddings(self):
        """Build local HuggingFace/SentenceTransformers embeddings using BAAI/bge-m3."""
        import logging
        logger = logging.getLogger(__name__)
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            logger.info("Initializing local HuggingFaceEmbeddings with BAAI/bge-m3...")
            return HuggingFaceEmbeddings(
                model_name="BAAI/bge-m3",
                model_kwargs={'device': 'cpu'}
            )
        except Exception as e:
            logger.warning(f"Failed to load local BAAI/bge-m3 embeddings: {e}. Falling back to Ollama nomic-embed-text.")
            from langchain_ollama import OllamaEmbeddings
            return OllamaEmbeddings(
                model=settings.OLLAMA_EMBED_MODEL,
                base_url=settings.OLLAMA_BASE_URL
            )

    @property
    def vector_store(self):
        from langchain_chroma import Chroma

        # Switch vector store & embedding model dynamically
        if mode_manager.mode == LLMMode.CLOUD and settings.GOOGLE_API_KEY:
            if self._cloud_vector_store is None:
                embeddings = self._make_cloud_embeddings()
                self._cloud_vector_store = Chroma(
                    persist_directory=settings.CHROMA_DB_PATH + "_cloud",
                    embedding_function=embeddings,
                    collection_name="research_papers_cloud"
                )
            return self._cloud_vector_store
        else:
            if self._local_vector_store is None:
                embeddings = self._make_local_embeddings()
                self._local_vector_store = Chroma(
                    persist_directory=settings.CHROMA_DB_PATH,
                    embedding_function=embeddings,
                    collection_name="research_papers"
                )
            return self._local_vector_store

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
        chroma_filter = None
        if filters:
            chroma_filter = {}
            for key, val in filters.items():
                if isinstance(val, list):
                    if len(val) == 1:
                        chroma_filter[key] = val[0]
                    else:
                        chroma_filter[key] = {"$in": val}
                else:
                    chroma_filter[key] = val
        try:
            return self.vector_store.similarity_search_with_score(query, k=k, filter=chroma_filter)
        except Exception as exc:
            msg = str(exc).lower()
            if "timeout" in msg:
                raise APITimeoutError(str(exc)) from exc
            if "429" in msg or "rate" in msg:
                raise APIRateLimitError(str(exc)) from exc
            if any(x in msg for x in ["500", "502", "503", "504", "server"]):
                raise APIServerError(str(exc)) from exc
            raise
