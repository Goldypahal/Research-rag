from __future__ import annotations
from typing import Dict, List, Optional, Any
import time
import os
import json
import logging
from ..export.literature_notes import CitationItem, LiteratureNote, LiteratureNotesExporter
from ..models.chunk import Chunk
from ..retrieval.hybrid_retriever import HybridRetriever
from ..multimodal.query_classifier import is_figure_question
from ..multimodal.figure_analyzer import FigureAnalyzer
from ..core.settings import settings

logger = logging.getLogger(__name__)

class QueryService:
    def __init__(self, retriever: HybridRetriever, answer_chain, citation_enforcer):
        self.retriever = retriever
        self.answer_chain = answer_chain
        self.citation_enforcer = citation_enforcer
        self.figure_analyzer = FigureAnalyzer(api_key=settings.GOOGLE_API_KEY) if settings.GOOGLE_API_KEY else None
        
        # Initialize query orchestration planner & multihop retrieval
        from .agent_planner import AgentPlanner
        from ..retrieval.multihop_retriever import MultiHopRetriever
        self.planner = AgentPlanner()
        self.multihop_retriever = MultiHopRetriever(self.retriever)

    def ask(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        expand_context: bool = False,
        prompt_version: str = "v1",
        export_markdown_path: Optional[str] = None,
        export_docx_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        start_time = time.time()
        
        # 1. Plan Route
        route = self.planner.plan_route(query, prompt_version=prompt_version)
        logger.info(f"AgentPlanner routed query to: '{route}'")
        
        # 2. Retrieve
        ret_start = time.time()
        
        # Check if it's a figure-based question and we have relevant figures
        if (route == "figure" or is_figure_question(query)) and self.figure_analyzer:
            retrieval_res = self.retriever.retrieve(query=query, filters=filters)
            top_chunks = retrieval_res["top_chunks"]
            figure_chunks = [c for c in top_chunks if c.image_path and os.path.exists(c.image_path)]
            if figure_chunks:
                logger.info(f"Detected figure-related question. Analyzing: {figure_chunks[0].image_path}")
                best_fig = figure_chunks[0]
                
                # Attempt to find a citation/caption context
                caption = best_fig.text if "caption" in best_fig.chunk_type else None
                if not caption:
                    # Look for caption in siblings or nearby chunks
                    all_c = retrieval_res.get("merged_candidates", [])
                    sibling_captions = [c.text for c in all_c if "caption" in c.chunk_type and c.page_start == best_fig.page_start]
                    caption = sibling_captions[0] if sibling_captions else best_fig.text

                fig_analysis = self.figure_analyzer.analyze_figure(
                    image_path=best_fig.image_path,
                    question=query,
                    caption=caption
                )
                
                fig_title = "Figure"
                if best_fig.title and best_fig.title.strip().lower() not in {"none", "undefined", "null"}:
                    fig_title = best_fig.title[:20]
                elif best_fig.paper_id and best_fig.paper_id.strip().lower() not in {"none", "undefined", "null"}:
                    fig_title = best_fig.paper_id

                return {
                    "answer": fig_analysis,
                    "chunks": [best_fig.model_dump()],
                    "citations": [{"label": f"{fig_title} | Section: {best_fig.section or 'Visual'} | p.{best_fig.page_start or '?'}", "quote": caption or "Visual content", "page": best_fig.page_start}],
                    "latency": time.time() - start_time,
                    "is_multimodal": True
                }

        # Otherwise, retrieve using standard hybrid or multi-hop
        if route == "multihop":
            retrieval_res = self.multihop_retriever.retrieve_multi_hop(query=query, filters=filters)
        else:
            retrieval_res = self.retriever.retrieve(query=query, filters=filters)
            
        top_chunks = retrieval_res["top_chunks"]
        logger.info(f"Retrieval took {time.time() - ret_start:.2f}s")

        chunks = top_chunks
        if expand_context and chunks:
            from ..retrieval.hybrid_retriever import expand_parent_context
            chunks = expand_parent_context(
                top_chunks=chunks,
                all_context_chunks=retrieval_res.get("merged_candidates", []),
                max_extra_chunks=4
            )
        
        if not chunks:
            return {
                "answer": "Insufficient evidence in retrieved sources.",
                "chunks": [],
                "citations": [],
                "latency": time.time() - start_time
            }

        # 3. Generate
        gen_start = time.time()
        raw_answer = self.answer_chain.generate(query=query, chunks=chunks, prompt_version=prompt_version)
        logger.info(f"Generation took {time.time() - gen_start:.2f}s")
        
        # 4. Enforce Citations
        enf_start = time.time()
        verified = self.citation_enforcer.enforce(answer=raw_answer, chunks=chunks)
        logger.info(f"Citation enforcement took {time.time() - enf_start:.2f}s")

        latency = time.time() - start_time
        logger.info(f"Total ask() latency: {latency:.2f}s")

        result = {
            "answer": verified["answer"],
            "chunks": [c.model_dump() for c in chunks],
            "citations": verified["citations"],
            "latency": latency,
            "is_multimodal": False
        }

        # 5. Export if requested
        if export_markdown_path or export_docx_path:
            note = LiteratureNote(
                title="Research Summary",
                question=query,
                answer=verified["answer"],
                citations=[
                    CitationItem(
                        label=c["label"],
                        quote=c["quote"],
                        page_start=c.get("page_start"),
                        page_end=c.get("page_end"),
                    )
                    for c in verified["citations"]
                ],
                limitations=verified.get("limitations", []),
            )
            if export_markdown_path:
                LiteratureNotesExporter.save_markdown(note, export_markdown_path)
            if export_docx_path:
                LiteratureNotesExporter.save_docx(note, export_docx_path)

        # 6. Persist Trace
        self._log_trace(query, retrieval_res, chunks, verified, latency)

        return result

    def _log_trace(self, query, retrieval_res, chunks, verified, latency):
        trace = {
            "query": query,
            "dense_hits": retrieval_res.get("dense_hits", []),
            "lexical_hits": retrieval_res.get("lexical_hits", []),
            "final_chunks": [c.chunk_id for c in chunks],
            "answer": verified.get("answer", "n/a"),
            "latency": latency,
            "timestamp": time.time(),
            "decomposed_queries": retrieval_res.get("decomposed_queries", [query]),
            "query_intent": retrieval_res.get("query_intent", "unknown")
        }
        os.makedirs("data/traces", exist_ok=True)
        with open("data/traces/query_logs.jsonl", "a") as f:
            f.write(json.dumps(trace) + "\n")
