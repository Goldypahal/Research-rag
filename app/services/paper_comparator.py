from __future__ import annotations
import logging
import sqlite3
from typing import List, Dict, Any, Optional
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ..core.settings import settings
from ..core.retry_utils import retry_api_call
from ..generation.prompt_loader import PromptLoader
from ..indexing.graph_index import SQLiteGraphIndex
from ..indexing.bm25_index import BM25Index

logger = logging.getLogger(__name__)

class PaperComparator:
    def __init__(self, db_path: str = "data/knowledge_graph.db", bm25: Optional[BM25Index] = None):
        self.db_path = db_path
        self.bm25 = bm25
        
        if settings.USE_LOCAL_LLM:
            logger.info(f"Initializing PaperComparator using local ChatOllama ({settings.OLLAMA_MODEL})...")
            self.llm = ChatOllama(
                model=settings.OLLAMA_MODEL,
                base_url=settings.OLLAMA_BASE_URL,
                temperature=0
            )
        elif settings.GOOGLE_API_KEY:
            logger.info("Initializing PaperComparator using ChatGoogleGenerativeAI (gemini-2.5-flash)...")
            from langchain_google_genai import ChatGoogleGenerativeAI
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=settings.GOOGLE_API_KEY,
                temperature=0
            )
        else:
            logger.info(f"Initializing PaperComparator using fallback ChatOllama ({settings.OLLAMA_MODEL})...")
            self.llm = ChatOllama(
                model=settings.OLLAMA_MODEL,
                base_url=settings.OLLAMA_BASE_URL,
                temperature=0
            )
        self.prompt_loader = PromptLoader()

    def _get_chain(self, version: str = "v1"):
        template = self.prompt_loader.load_prompt("compare_features", version=version)
        prompt = ChatPromptTemplate.from_template(template)
        return prompt | self.llm | StrOutputParser()

    @retry_api_call(max_attempts=3, min_wait=1, max_wait=10)
    def _call_llm(self, chain, facts: str) -> str:
        return chain.invoke({"facts": facts})

    def _get_paper_facts_from_db(self, paper_id: str) -> str:
        """Query entities and relationships from the SQLite graph index."""
        facts_lines = []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Fetch entities
            cursor.execute("SELECT name, type FROM entities WHERE paper_id = ?", (paper_id,))
            entities = cursor.fetchall()
            if entities:
                facts_lines.append(f"Paper '{paper_id}' contains entities:")
                for ent in entities:
                    facts_lines.append(f"  * {ent['name']} ({ent['type']})")
                    
            # Fetch relations
            cursor.execute("""
                SELECT e1.name AS src, e1.type AS src_type, e2.name AS target, e2.type AS target_type, r.relation_type, r.description
                FROM relations r
                JOIN entities e1 ON r.source_id = e1.entity_id
                JOIN entities e2 ON r.target_id = e2.entity_id
                WHERE e1.paper_id = ?
            """, (paper_id,))
            relations = cursor.fetchall()
            if relations:
                facts_lines.append(f"Paper '{paper_id}' relationships:")
                for rel in relations:
                    desc = f" ({rel['description']})" if rel['description'] else ""
                    facts_lines.append(f"  * {rel['src']} ({rel['src_type']}) - {rel['relation_type']} -> {rel['target']} ({rel['target_type']}){desc}")
                    
        return "\n".join(facts_lines)

    def _get_paper_facts_from_chunks(self, paper_id: str) -> str:
        """Fallback: get facts from first 5 chunks of the paper if Graph is empty."""
        if not self.bm25:
            return ""
        # Find chunks belonging to this paper
        paper_chunks = [c for c in self.bm25.chunks if c.paper_id == paper_id][:5]
        if not paper_chunks:
            return ""
        
        text = "\n\n".join([f"Chunk {i}: {c.text}" for i, c in enumerate(paper_chunks)])
        return f"Paper '{paper_id}' Text Content:\n{text}"

    def compare_papers(self, paper_ids: List[str], prompt_version: str = "v1") -> str:
        """Generates a markdown comparison matrix table for a list of papers."""
        logger.info(f"Comparing papers: {paper_ids}")
        if not paper_ids:
            return "No papers selected for comparison."

        all_facts = []
        for pid in paper_ids:
            facts = self._get_paper_facts_from_db(pid)
            if not facts:
                # Try fallback
                logger.info(f"No graph facts found for paper {pid}. Attempting chunk fallback.")
                facts = self._get_paper_facts_from_chunks(pid)
            if facts:
                all_facts.append(facts)
            else:
                all_facts.append(f"No data available for Paper '{pid}'.")

        combined_facts = "\n\n===\n\n".join(all_facts)
        try:
            chain = self._get_chain(version=prompt_version)
            markdown_table = self._call_llm(chain, combined_facts)
            return markdown_table
        except Exception as e:
            logger.error(f"Failed to generate paper comparison: {e}")
            return "Failed to generate comparison matrix due to an LLM error."
