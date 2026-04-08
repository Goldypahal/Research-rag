from __future__ import annotations
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Any
from ..models.chunk import Chunk
from ..indexing.chroma_index import ChromaIndex
from ..indexing.bm25_index import BM25Index

def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[str]],
    k: int = 60,
) -> Dict[str, float]:
    scores: Dict[str, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, chunk_id in enumerate(ranked, start=1):
            scores[chunk_id] += 1.0 / (k + rank)
    return dict(scores)

def expand_parent_context(
    top_chunks: List[Chunk],
    all_context_chunks: List[Chunk],
    include_same_subsection: bool = True,
    include_same_section: bool = False,
    max_extra_chunks: int = 4,
) -> List[Chunk]:
    selected_ids = {c.chunk_id for c in top_chunks}
    extra: List[Chunk] = []

    for top in top_chunks:
        for chunk in all_context_chunks:
            if chunk.chunk_id in selected_ids:
                continue

            same_subsection = (
                include_same_subsection
                and chunk.section_path == top.section_path
            )

            same_section = (
                include_same_section
                and chunk.section == top.section
            )

            if same_subsection or same_section:
                extra.append(chunk)
                selected_ids.add(chunk.chunk_id)
                if len(extra) >= max_extra_chunks:
                    break

        if len(extra) >= max_extra_chunks:
            break

    return top_chunks + extra

class HybridRetriever:
    def __init__(self, chroma_index: ChromaIndex, bm25_index: BM25Index, reranker=None, final_top_k: int = 8):
        self.chroma_index = chroma_index
        self.bm25_index = bm25_index
        self.reranker = reranker
        self.final_top_k = final_top_k

    @staticmethod
    def _metadata_bonus(query: str, chunk: Chunk) -> float:
        q = query.lower()
        bonus = 0.0

        if chunk.heading_number and chunk.heading_number.lower() in q:
            bonus += 0.30

        if chunk.heading_title and chunk.heading_title.lower() in q:
            bonus += 0.25

        if chunk.subsubsection and chunk.subsubsection.lower() in q:
            bonus += 0.25

        if chunk.subsection and chunk.subsection.lower() in q:
            bonus += 0.20

        if chunk.section and chunk.section.lower() in q:
            bonus += 0.10

        joined_path = " > ".join(chunk.section_path).lower()
        if joined_path and any(part.strip() and part.strip() in q for part in joined_path.split(">")):
            bonus += 0.15

        if "figure" in q and chunk.chunk_type in {"figure_caption", "figure"}:
            bonus += 0.2
        if "table" in q and chunk.chunk_type in {"table", "table_summary"}:
            bonus += 0.2
        
        # Topic specific boosts
        for topic in ["hyperparameter", "dataset", "result", "ablation", "method"]:
            if topic in q and chunk.heading_title and topic in chunk.heading_title.lower():
                bonus += 0.20

        return bonus

    def retrieve(
        self,
        query: str,
        dense_top_k: int = 20,
        bm25_top_k: int = 20,
        merged_top_k: int = 15,
        filters: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        from concurrent.futures import ThreadPoolExecutor
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_chroma = executor.submit(self.chroma_index.query, query, k=dense_top_k, filters=filters)
            future_bm25 = executor.submit(self.bm25_index.query, query, k=bm25_top_k, filters=filters)
            
            chroma_results = future_chroma.result()
            bm25_results = future_bm25.result()

        dense_hits = []
        for doc, score in chroma_results:
            # Reconstruct Chunk from LangChain document
            metadata = doc.metadata.copy()
            metadata["text"] = doc.page_content
            
            # Restore lists that were flattened to strings for storage
            if isinstance(metadata.get("authors"), str):
                metadata["authors"] = [a.strip() for a in metadata["authors"].split(",") if a.strip()]
            if isinstance(metadata.get("section_path"), str):
                metadata["section_path"] = [s.strip() for s in metadata["section_path"].split(">") if s.strip()]
            
            # Map top-level keys back to Chunk fields, extras go to metadata dict
            from ..models.chunk import Chunk
            field_names = Chunk.model_fields.keys()
            chunk_params = {}
            extra_metadata = {}
            
            for k, v in metadata.items():
                if k in field_names and k != "metadata":
                    chunk_params[k] = v
                else:
                    extra_metadata[k] = v
            
            chunk_params["metadata"] = extra_metadata
            dense_hits.append(Chunk(**chunk_params))
            
        lexical_hits = [c for c, score in bm25_results]

        dense_ids = [c.chunk_id for c in dense_hits]
        lexical_ids = [c.chunk_id for c in lexical_hits]

        rrf_scores = reciprocal_rank_fusion([dense_ids, lexical_ids])

        lookup: Dict[str, Chunk] = {}
        for c in dense_hits + lexical_hits:
            lookup[c.chunk_id] = c

        merged = []
        for chunk_id, base_score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
            chunk = lookup[chunk_id]
            score = base_score + self._metadata_bonus(query, chunk)
            chunk.metadata["hybrid_score"] = score
            merged.append(chunk)

        merged = sorted(
            merged,
            key=lambda c: float(c.metadata.get("hybrid_score", 0.0)),
            reverse=True,
        )[:merged_top_k]

        final_chunks = merged
        if self.reranker:
            try:
                final_chunks = self.reranker.rerank(query, merged)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"Primary reranker failed after retries: {e}. Falling back to SBERT.")
                from .rerank_sbert import SBERTReranker
                fallback = SBERTReranker()
                final_chunks = fallback.rerank(query, merged)

        return {
            "top_chunks": final_chunks[:self.final_top_k],
            "merged_candidates": merged, # Full pool for expansion
            "dense_hits": [{"chunk_id": c.chunk_id, "score": 0} for c in dense_hits],
            "lexical_hits": [{"chunk_id": c.chunk_id, "score": 0} for c in lexical_hits]
        }