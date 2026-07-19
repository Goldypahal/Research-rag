from __future__ import annotations
import logging
from typing import Dict, List, Any, Optional
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ..core.settings import settings
from ..core.retry_utils import retry_api_call
from ..generation.prompt_loader import PromptLoader
from ..indexing.graph_index import SQLiteGraphIndex
from ..models.chunk import Chunk
from .hybrid_retriever import HybridRetriever

logger = logging.getLogger(__name__)

class MultiHopRetriever:
    def __init__(self, hybrid_retriever: HybridRetriever, db_path: str = "data/knowledge_graph.db"):
        self.retriever = hybrid_retriever
        self.graph_db = SQLiteGraphIndex(db_path=db_path)
        
        if settings.USE_LOCAL_LLM:
            logger.info(f"Initializing MultiHopRetriever using local ChatOllama ({settings.OLLAMA_MODEL})...")
            self.llm = ChatOllama(
                model=settings.OLLAMA_MODEL,
                base_url=settings.OLLAMA_BASE_URL,
                temperature=0
            )
        elif settings.GOOGLE_API_KEY:
            logger.info("Initializing MultiHopRetriever using ChatGoogleGenerativeAI (gemini-2.5-flash)...")
            from langchain_google_genai import ChatGoogleGenerativeAI
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=settings.GOOGLE_API_KEY,
                temperature=0
            )
        else:
            logger.info(f"Initializing MultiHopRetriever using fallback ChatOllama ({settings.OLLAMA_MODEL})...")
            self.llm = ChatOllama(
                model=settings.OLLAMA_MODEL,
                base_url=settings.OLLAMA_BASE_URL,
                temperature=0
            )
        self.prompt_loader = PromptLoader()

    def _get_chain(self, version: str = "v1"):
        template = self.prompt_loader.load_prompt("generate_follow_up", version=version)
        prompt = ChatPromptTemplate.from_template(template)
        return prompt | self.llm | StrOutputParser()

    @retry_api_call(max_attempts=3, min_wait=1, max_wait=10)
    def _call_llm(self, chain, question: str, context: str) -> str:
        return chain.invoke({"question": question, "context": context})

    def retrieve_multi_hop(
        self,
        query: str,
        filters: Optional[Dict] = None,
        max_hops: int = 2
    ) -> Dict[str, Any]:
        """
        Executes a multi-hop retrieval loop.
        Fuses standard hybrid search with knowledge graph entity retrieval.
        """
        logger.info(f"Starting multi-hop retrieval for: '{query}'")
        
        # 1. Retrieve Graph RAG context (fusing Graph facts with text chunks)
        graph_context = self.graph_db.get_graph_context_for_query(query)
        
        # 2. Run initial retrieval
        ret_res = self.retriever.retrieve(query, filters=filters)
        current_chunks = ret_res["top_chunks"]
        
        # Keep track of all retrieved chunk IDs to avoid duplicates
        retrieved_ids = {c.chunk_id for c in current_chunks}
        all_merged_candidates = list(ret_res.get("merged_candidates", []))
        
        hops_completed = 1
        follow_up_queries = []
        
        while hops_completed < max_hops:
            # Build context summary to pass to the planner
            context_summary = "\n".join([f"- Chunk (Page {c.page_start}): {c.text[:200]}..." for c in current_chunks])
            if graph_context:
                context_summary += f"\n\n{graph_context}"

            try:
                chain = self._get_chain()
                follow_up = self._call_llm(chain, query, context_summary).strip()
                
                if not follow_up or follow_up.upper() == "NONE":
                    logger.info("Multi-hop planner decided no further hops are needed.")
                    break
                    
                logger.info(f"Multi-hop follow-up query generated (Hop {hops_completed + 1}): '{follow_up}'")
                follow_up_queries.append(follow_up)
                
                # Perform the secondary retrieval
                follow_up_res = self.retriever.retrieve(follow_up, filters=filters)
                follow_up_chunks = follow_up_res["top_chunks"]
                
                # Merge new unique chunks
                new_chunks_added = 0
                for c in follow_up_chunks:
                    if c.chunk_id not in retrieved_ids:
                        retrieved_ids.add(c.chunk_id)
                        current_chunks.append(c)
                        new_chunks_added += 1
                
                # Append candidates
                for c in follow_up_res.get("merged_candidates", []):
                    if not any(x.chunk_id == c.chunk_id for x in all_merged_candidates):
                        all_merged_candidates.append(c)
                        
                logger.info(f"Secondary retrieval added {new_chunks_added} new chunks.")
                hops_completed += 1
                
            except Exception as e:
                logger.error(f"Failed during multi-hop lookup: {e}")
                break

        # If graph context was found, we append it as a virtual "fact" chunk so that the LLM uses it
        if graph_context:
            import uuid
            graph_chunk = Chunk(
                chunk_id=f"graph-fact-{uuid.uuid4()}",
                paper_id="knowledge-graph",
                text=graph_context,
                title="Local Knowledge Graph",
                chunk_type="text",
                chunk_level="paragraph"
            )
            # Prepend graph facts to context
            current_chunks.insert(0, graph_chunk)

        return {
            "top_chunks": current_chunks[:self.retriever.final_top_k],
            "merged_candidates": all_merged_candidates,
            "dense_hits": ret_res.get("dense_hits", []),
            "lexical_hits": ret_res.get("lexical_hits", []),
            "follow_up_queries": follow_up_queries,
            "graph_context": graph_context
        }
