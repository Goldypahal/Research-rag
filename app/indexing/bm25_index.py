from rank_bm25 import BM25Okapi
import pickle
import os
from typing import List, Optional, Dict
from ..models.chunk import Chunk
from ..core.settings import settings

class BM25Index:
    def __init__(self):
        self.chunks: List[Chunk] = []
        self.bm25 = None
        if os.path.exists(settings.BM25_INDEX_PATH):
            self.load()

    def add_chunks(self, chunks: List[Chunk]):
        self.chunks.extend(chunks)
        tokenized_corpus = [c.text.split() for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.save()

    def query(self, query: str, k: int = 20, filters: Optional[dict] = None):
        if not self.bm25:
            return []
        tokenized_query = query.split()
        doc_scores = self.bm25.get_scores(tokenized_query)
        
        # Apply filters if provided (basic implementation for paper_id)
        candidate_indices = range(len(self.chunks))
        if filters and "paper_id" in filters:
            allowed_ids = filters["paper_id"]
            if isinstance(allowed_ids, str): allowed_ids = [allowed_ids]
            candidate_indices = [i for i in candidate_indices if self.chunks[i].paper_id in allowed_ids]

        if not candidate_indices:
            return []

        top_indices = sorted(candidate_indices, key=lambda i: doc_scores[i], reverse=True)[:k]
        return [(self.chunks[i], doc_scores[i]) for i in top_indices]

    def save(self):
        with open(settings.BM25_INDEX_PATH, 'wb') as f:
            pickle.dump((self.chunks, self.bm25), f)

    def load(self):
        with open(settings.BM25_INDEX_PATH, 'rb') as f:
            self.chunks, self.bm25 = pickle.load(f)
