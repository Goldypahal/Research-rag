from __future__ import annotations
from typing import List
from sentence_transformers import CrossEncoder
from ..models.chunk import Chunk

class SBERTReranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_n: int = 8,
    ):
        self.model = CrossEncoder(model_name)
        self.top_n = top_n

    def rerank(self, query: str, chunks: List[Chunk]) -> List[Chunk]:
        if not chunks:
            return []

        pairs = [[query, c.text] for c in chunks]
        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(chunks, scores),
            key=lambda x: float(x[1]),
            reverse=True,
        )[: self.top_n]

        out: List[Chunk] = []
        for chunk, score in ranked:
            chunk.metadata["rerank_score_sbert"] = float(score)
            out.append(chunk)
        return out
