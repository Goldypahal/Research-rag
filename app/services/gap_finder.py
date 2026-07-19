from __future__ import annotations
import logging
import sqlite3
from typing import List, Optional
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ..core.settings import settings
from ..core.retry_utils import retry_api_call
from ..generation.prompt_loader import PromptLoader

logger = logging.getLogger(__name__)

class ResearchGapFinder:
    def __init__(self, db_path: str = "data/knowledge_graph.db"):
        self.db_path = db_path
        
        if settings.USE_LOCAL_LLM:
            logger.info(f"Initializing ResearchGapFinder using local ChatOllama ({settings.OLLAMA_MODEL})...")
            self.llm = ChatOllama(
                model=settings.OLLAMA_MODEL,
                base_url=settings.OLLAMA_BASE_URL,
                temperature=0
            )
        elif settings.GOOGLE_API_KEY:
            logger.info("Initializing ResearchGapFinder using ChatGoogleGenerativeAI (gemini-2.5-flash)...")
            from langchain_google_genai import ChatGoogleGenerativeAI
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=settings.GOOGLE_API_KEY,
                temperature=0
            )
        else:
            logger.info(f"Initializing ResearchGapFinder using fallback ChatOllama ({settings.OLLAMA_MODEL})...")
            self.llm = ChatOllama(
                model=settings.OLLAMA_MODEL,
                base_url=settings.OLLAMA_BASE_URL,
                temperature=0
            )
        self.prompt_loader = PromptLoader()

    def _get_chain(self, version: str = "v1"):
        template = self.prompt_loader.load_prompt("find_gaps", version=version)
        prompt = ChatPromptTemplate.from_template(template)
        return prompt | self.llm | StrOutputParser()

    @retry_api_call(max_attempts=3, min_wait=1, max_wait=10)
    def _call_llm(self, chain, facts: str) -> str:
        return chain.invoke({"facts": facts})

    def _get_corpus_inventory(self, paper_ids: Optional[List[str]] = None) -> str:
        """Fetch all methods and datasets from SQLite to create a cross-comparison matrix."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Fetch all methods
            if paper_ids:
                placeholders = ",".join("?" for _ in paper_ids)
                cursor.execute(f"SELECT DISTINCT name, paper_id FROM entities WHERE type = 'Method' AND paper_id IN ({placeholders})", paper_ids)
            else:
                cursor.execute("SELECT DISTINCT name, paper_id FROM entities WHERE type = 'Method'")
            methods = cursor.fetchall()

            # Fetch all datasets
            if paper_ids:
                placeholders = ",".join("?" for _ in paper_ids)
                cursor.execute(f"SELECT DISTINCT name, paper_id FROM entities WHERE type = 'Dataset' AND paper_id IN ({placeholders})", paper_ids)
            else:
                cursor.execute("SELECT DISTINCT name, paper_id FROM entities WHERE type = 'Dataset'")
            datasets = cursor.fetchall()
            
            # Fetch evaluated_on relations to see what combinations exist
            if paper_ids:
                placeholders = ",".join("?" for _ in paper_ids)
                cursor.execute(f"""
                    SELECT DISTINCT e1.name AS method, e2.name AS dataset
                    FROM relations r
                    JOIN entities e1 ON r.source_id = e1.entity_id
                    JOIN entities e2 ON r.target_id = e2.entity_id
                    WHERE r.relation_type = 'evaluated_on' AND e1.paper_id IN ({placeholders})
                """, paper_ids)
            else:
                cursor.execute("""
                    SELECT DISTINCT e1.name AS method, e2.name AS dataset
                    FROM relations r
                    JOIN entities e1 ON r.source_id = e1.entity_id
                    JOIN entities e2 ON r.target_id = e2.entity_id
                    WHERE r.relation_type = 'evaluated_on'
                """)
            evals = cursor.fetchall()

        inventory_lines = ["Corpus Inventory:"]
        inventory_lines.append("\nMethods Extracted:")
        for m in methods:
            inventory_lines.append(f"- {m['name']} (from {m['paper_id']})")
            
        inventory_lines.append("\nDatasets Extracted:")
        for d in datasets:
            inventory_lines.append(f"- {d['name']} (from {d['paper_id']})")
            
        inventory_lines.append("\nExisting Evaluated Combinations:")
        for ev in evals:
            inventory_lines.append(f"- Method '{ev['method']}' evaluated on Dataset '{ev['dataset']}'")
            
        return "\n".join(inventory_lines)

    def find_gaps(self, paper_ids: Optional[List[str]] = None, prompt_version: str = "v1") -> str:
        """Finds research gaps by analyzing methods, datasets, and existing evaluations."""
        logger.info("Spotting research gaps...")
        facts = self._get_corpus_inventory(paper_ids)
        try:
            chain = self._get_chain(version=prompt_version)
            report = self._call_llm(chain, facts)
            return report
        except Exception as e:
            logger.error(f"Failed to find research gaps: {e}")
            return "Failed to perform research gap analysis due to an LLM error."
