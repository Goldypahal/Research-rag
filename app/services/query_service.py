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
        
        # 1. Retrieve
        ret_start = time.time()
        retrieval_res = self.retriever.retrieve(query=query, filters=filters)
        top_chunks = retrieval_res["top_chunks"]
        logger.info(f"Retrieval took {time.time() - ret_start:.2f}s")
        
        # Check if it's a figure-based question and we have relevant figures
        if is_figure_question(query) and self.figure_analyzer:
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
                
                return {
                    "answer": fig_analysis,
                    "chunks": [best_fig.model_dump()],
                    "citations": [{"label": best_fig.title or "Figure", "quote": caption or "Visual content", "page": best_fig.page_start}],
                    "latency": time.time() - start_time,
                    "is_multimodal": True
                }

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

        # 2. Generate
        gen_start = time.time()
        raw_answer = self.answer_chain.generate(query=query, chunks=chunks, prompt_version=prompt_version)
        logger.info(f"Generation took {time.time() - gen_start:.2f}s")
        
        # 3. Enforce Citations
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

        # 4. Export if requested
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

        # 5. Persist Trace
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
            "timestamp": time.time()
        }
        os.makedirs("data/traces", exist_ok=True)
        with open("data/traces/query_logs.jsonl", "a") as f:
            f.write(json.dumps(trace) + "\n")
