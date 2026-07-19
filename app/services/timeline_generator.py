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

class ResearchTimelineGenerator:
    def __init__(self, db_path: str = "data/knowledge_graph.db"):
        self.db_path = db_path
        
        if settings.USE_LOCAL_LLM:
            logger.info(f"Initializing ResearchTimelineGenerator using local ChatOllama ({settings.OLLAMA_MODEL})...")
            self.llm = ChatOllama(
                model=settings.OLLAMA_MODEL,
                base_url=settings.OLLAMA_BASE_URL,
                temperature=0
            )
        elif settings.GOOGLE_API_KEY:
            logger.info("Initializing ResearchTimelineGenerator using ChatGoogleGenerativeAI (gemini-2.5-flash)...")
            from langchain_google_genai import ChatGoogleGenerativeAI
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=settings.GOOGLE_API_KEY,
                temperature=0
            )
        else:
            logger.info(f"Initializing ResearchTimelineGenerator using fallback ChatOllama ({settings.OLLAMA_MODEL})...")
            self.llm = ChatOllama(
                model=settings.OLLAMA_MODEL,
                base_url=settings.OLLAMA_BASE_URL,
                temperature=0
            )
        self.prompt_loader = PromptLoader()

    def _get_chain(self, version: str = "v1"):
        template = self.prompt_loader.load_prompt("timeline", version=version)
        prompt = ChatPromptTemplate.from_template(template)
        return prompt | self.llm | StrOutputParser()

    @retry_api_call(max_attempts=3, min_wait=1, max_wait=10)
    def _call_llm(self, chain, facts: str) -> str:
        return chain.invoke({"facts": facts})

    def _get_sorted_paper_metadata(self, paper_ids: Optional[List[str]] = None) -> str:
        """Fetch papers, titles, publication years, and methods, sorted chronologically."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Fetch paper titles and years
            if paper_ids:
                placeholders = ",".join("?" for _ in paper_ids)
                cursor.execute(f"""
                    SELECT DISTINCT paper_id, name AS title, 
                           (SELECT DISTINCT year FROM entities WHERE paper_id = e.paper_id AND year IS NOT NULL LIMIT 1) AS pub_year
                    FROM entities e
                    WHERE type = 'Paper' AND paper_id IN ({placeholders})
                """, paper_ids)
            else:
                cursor.execute("""
                    SELECT DISTINCT paper_id, name AS title, 
                           (SELECT DISTINCT year FROM entities WHERE paper_id = e.paper_id AND year IS NOT NULL LIMIT 1) AS pub_year
                    FROM entities e
                    WHERE type = 'Paper'
                """)
            papers = cursor.fetchall()
            
            # Convert to list and clean None years
            papers_list = []
            for p in papers:
                year = p["pub_year"]
                # Try fallback chunk query if year not in entities table
                if year is None:
                    cursor.execute("SELECT DISTINCT year FROM entities WHERE paper_id = ? AND year IS NOT NULL LIMIT 1", (p["paper_id"],))
                    fallback_year_row = cursor.fetchone()
                    year = fallback_year_row["year"] if fallback_year_row else None
                
                # Default to 2026 if not found
                year_val = int(year) if year is not None else 2026
                papers_list.append({
                    "paper_id": p["paper_id"],
                    "title": p["title"],
                    "year": year_val
                })
                
            # Sort papers by year ascending
            papers_list.sort(key=lambda x: x["year"])
            
            fact_lines = []
            for p in papers_list:
                # Fetch methods/algorithms linked to this paper
                cursor.execute("SELECT DISTINCT name FROM entities WHERE paper_id = ? AND type IN ('Method', 'Algorithm')", (p["paper_id"],))
                methods = [r["name"] for r in cursor.fetchall()]
                methods_str = ", ".join(methods) if methods else "None documented"
                
                fact_lines.append(f"Year {p['year']}:")
                fact_lines.append(f"- Title: {p['title']}")
                fact_lines.append(f"- ID: {p['paper_id']}")
                fact_lines.append(f"- Focus Methods: {methods_str}")
                fact_lines.append("")
                
        return "\n".join(fact_lines)

    def generate_timeline(self, paper_ids: Optional[List[str]] = None, prompt_version: str = "v1") -> str:
        """Generates a chronological evolution timeline of the selected papers."""
        logger.info("Generating chronological research timeline...")
        facts = self._get_sorted_paper_metadata(paper_ids)
        if not facts.strip():
            return "No paper metadata available in graph index to construct timeline."
        try:
            chain = self._get_chain(version=prompt_version)
            timeline = self._call_llm(chain, facts)
            return timeline
        except Exception as e:
            logger.error(f"Failed to generate timeline: {e}")
            return "Failed to compile chronological research timeline due to an LLM error."
