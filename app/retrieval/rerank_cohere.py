from __future__ import annotations
from typing import List
import cohere
from ..core.retry_utils import (
    APIRateLimitError,
    APIServerError,
    APITimeoutError,
    retry_api_call,
)
from ..models.chunk import Chunk
from ..core.settings import settings

class CohereReranker:
    def __init__(self, api_key: str = None, model: str = "rerank-v3.5", top_n: int = 8):
        self.api_key = api_key or settings.COHERE_API_KEY
        self.client = cohere.ClientV2(api_key=self.api_key) if self.api_key else None
        self.model = model
        self.top_n = top_n

    @retry_api_call(max_attempts=5, min_wait=1, max_wait=16)
    def _rerank(self, query: str, documents: List[str]):
        if not self.client:
            raise ValueError("Cohere API key not configured.")
        try:
            return self.client.rerank(
                model=self.model,
                query=query,
                documents=documents,
                top_n=min(self.top_n, len(documents)),
            )
        except Exception as exc:
            msg = str(exc).lower()
            if "timeout" in msg:
                raise APITimeoutError(str(exc)) from exc
            if "429" in msg or "rate" in msg:
                raise APIRateLimitError(str(exc)) from exc
            if any(x in msg for x in ["500", "502", "503", "504", "server"]):
                raise APIServerError(str(exc)) from exc
            raise

    def rerank(self, query: str, chunks: List[Chunk]) -> List[Chunk]:
        if not chunks or not self.client:
            return chunks[:self.top_n]

        docs = [c.text for c in chunks]
        response = self._rerank(query, docs)

        ranked = []
        for item in response.results:
            chunk = chunks[item.index]
            chunk.metadata["rerank_score_cohere"] = item.relevance_score
            ranked.append(chunk)
        return ranked
