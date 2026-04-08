from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from .chunk import Chunk

class QueryTrace(BaseModel):
    query: str
    dense_hits: List[Dict[str, Any]] = []
    lexical_hits: List[Dict[str, Any]] = []
    retrieved_chunks: List[Chunk]
    reranker_scores: List[float]
    prompt_version: str
    final_citations: List[Dict[str, Any]]
    model_output: str
    latency: float
    metadata: Dict[str, Any] = {}
