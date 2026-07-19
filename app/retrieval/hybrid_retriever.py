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
        
        # Initialize query orchestration sub-systems
        from .query_decomposer import QueryDecomposer
        from .adaptive_retriever import AdaptiveRetriever
        self.decomposer = QueryDecomposer()
        self.adaptive_retriever = AdaptiveRetriever()

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
        decompose: bool = True,
        adaptive: bool = True,
    ) -> Dict[str, Any]:
        # 1. Parse Temporal Filters
        from .temporal_parser import extract_temporal_filter
        temp_filter = extract_temporal_filter(query)
        if temp_filter:
            if filters is None:
                filters = temp_filter
            else:
                filters = {**filters, **temp_filter}
            import logging
            logging.getLogger(__name__).info(f"Extracted temporal filter: {temp_filter}. Updated filters: {filters}")

        # Default to paragraph level chunks to avoid text duplication of multiple resolutions
        if filters is None:
            filters = {"chunk_level": "paragraph"}
        elif "chunk_level" not in filters:
            filters["chunk_level"] = "paragraph"

        # 2. Adaptive Retrieval
        active_final_top_k = self.final_top_k
        query_intent = "unknown"
        if adaptive:
            try:
                limits = self.adaptive_retriever.get_retrieval_limits(query)
                query_intent = limits["intent"]
                dense_top_k = limits["dense_top_k"]
                bm25_top_k = limits["bm25_top_k"]
                merged_top_k = limits["merged_top_k"]
                active_final_top_k = limits["final_top_k"]
                import logging
                logging.getLogger(__name__).info(
                    f"Adaptive Retrieval enabled. Intent: {query_intent}. "
                    f"Limits -> dense_top_k: {dense_top_k}, bm25_top_k: {bm25_top_k}, "
                    f"merged_top_k: {merged_top_k}, final_top_k: {active_final_top_k}"
                )
            except Exception as e:
                import logging
                logging.getLogger(__name__).error(f"Failed to run adaptive retrieval: {e}")

        # Helper to query index for a single query string
        def _run_single_query_retrieval(q: str):
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_chroma = executor.submit(self.chroma_index.query, q, k=dense_top_k, filters=filters)
                future_bm25 = executor.submit(self.bm25_index.query, q, k=bm25_top_k, filters=filters)
                
                chroma_results = future_chroma.result()
                bm25_results = future_bm25.result()
            return chroma_results, bm25_results

        # Helper to parse Chroma documents into Chunk objects
        def _parse_chroma_results(chroma_results):
            dense_hits = []
            for doc, score in chroma_results:
                metadata = doc.metadata.copy()
                metadata["text"] = doc.page_content
                if isinstance(metadata.get("authors"), str):
                    metadata["authors"] = [a.strip() for a in metadata["authors"].split(",") if a.strip()]
                if isinstance(metadata.get("section_path"), str):
                    metadata["section_path"] = [s.strip() for s in metadata["section_path"].split(">") if s.strip()]
                
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
            return dense_hits

        # 3. Query Decomposition
        sub_queries = [query]
        if decompose:
            try:
                sub_queries = self.decomposer.decompose(query)
            except Exception as e:
                import logging
                logging.getLogger(__name__).error(f"Failed to decompose query: {e}")

        # Retrieve for all sub-queries
        all_dense_hits = []
        all_lexical_hits = []
        
        # Run sub-query lookups in parallel
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(4, len(sub_queries))) as executor:
            futures = [executor.submit(_run_single_query_retrieval, sq) for sq in sub_queries]
            for sq, future in zip(sub_queries, futures):
                try:
                    chroma_res, bm25_res = future.result()
                    dense_hits = _parse_chroma_results(chroma_res)
                    lexical_hits = [c for c, score in bm25_res]
                    
                    all_dense_hits.append((sq, dense_hits))
                    all_lexical_hits.append((sq, lexical_hits))
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).error(f"Retrieval for sub-query '{sq}' failed: {e}")

        # Reciprocal Rank Fusion & Merging across all sub-queries
        # Sum RRF scores for chunks retrieved by multiple sub-queries
        combined_scores: Dict[str, float] = defaultdict(float)
        lookup: Dict[str, Chunk] = {}

        for sq in sub_queries:
            # Find the hits for this sub-query
            dense_hits = next((hits for q, hits in all_dense_hits if q == sq), [])
            lexical_hits = next((hits for q, hits in all_lexical_hits if q == sq), [])

            dense_ids = [c.chunk_id for c in dense_hits]
            lexical_ids = [c.chunk_id for c in lexical_hits]

            # Compute RRF for this sub-query's list
            rrf_scores = reciprocal_rank_fusion([dense_ids, lexical_ids])
            
            # Store in lookup
            for c in dense_hits + lexical_hits:
                lookup[c.chunk_id] = c
                
            # Add to combined scores
            for chunk_id, base_score in rrf_scores.items():
                combined_scores[chunk_id] += base_score

        # Apply metadata bonus based on the original query
        merged = []
        for chunk_id, total_rrf_score in combined_scores.items():
            chunk = lookup[chunk_id]
            # Use original query for metadata bonus
            score = total_rrf_score + self._metadata_bonus(query, chunk)
            chunk.metadata["hybrid_score"] = score
            merged.append(chunk)

        # Sort by total hybrid score
        merged = sorted(
            merged,
            key=lambda c: float(c.metadata.get("hybrid_score", 0.0)),
            reverse=True,
        )[:merged_top_k]

        # Rerank against original query
        final_chunks = merged
        if self.reranker:
            try:
                final_chunks = self.reranker.rerank(query, merged)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"Primary reranker failed: {e}. Falling back to SBERT.")
                from .rerank_sbert import SBERTReranker
                fallback = SBERTReranker()
                final_chunks = fallback.rerank(query, merged)

        # Build list of dense_hits and lexical_hits representation for the trace
        flat_dense_hits = []
        for _, hits in all_dense_hits:
            flat_dense_hits.extend([{"chunk_id": c.chunk_id, "score": 0} for c in hits])
        flat_lexical_hits = []
        for _, hits in all_lexical_hits:
            flat_lexical_hits.extend([{"chunk_id": c.chunk_id, "score": 0} for c in hits])

        return {
            "top_chunks": final_chunks[:active_final_top_k],
            "merged_candidates": merged, # Full pool for expansion
            "dense_hits": flat_dense_hits,
            "lexical_hits": flat_lexical_hits,
            "decomposed_queries": sub_queries,
            "query_intent": query_intent
        }