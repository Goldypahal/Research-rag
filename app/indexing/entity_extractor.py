from __future__ import annotations
import json
import logging
from typing import List, Dict, Any
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ..core.settings import settings
from ..core.retry_utils import retry_api_call
from ..generation.prompt_loader import PromptLoader
from .graph_index import SQLiteGraphIndex
from ..models.paper import Paper

logger = logging.getLogger(__name__)

class EntityExtractor:
    def __init__(self, db_path: str = "data/knowledge_graph.db"):
        self.db = SQLiteGraphIndex(db_path=db_path)
        if settings.USE_LOCAL_LLM:
            logger.info(f"Initializing EntityExtractor using local ChatOllama ({settings.OLLAMA_MODEL})...")
            self.llm = ChatOllama(
                model=settings.OLLAMA_MODEL,
                base_url=settings.OLLAMA_BASE_URL,
                temperature=0
            )
        elif settings.GOOGLE_API_KEY:
            logger.info("Initializing EntityExtractor using ChatGoogleGenerativeAI (gemini-2.5-flash)...")
            from langchain_google_genai import ChatGoogleGenerativeAI
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=settings.GOOGLE_API_KEY,
                temperature=0
            )
        else:
            logger.info(f"Initializing EntityExtractor using fallback ChatOllama ({settings.OLLAMA_MODEL})...")
            self.llm = ChatOllama(
                model=settings.OLLAMA_MODEL,
                base_url=settings.OLLAMA_BASE_URL,
                temperature=0
            )
        self.prompt_loader = PromptLoader()

    def _get_chain(self, version: str = "v1"):
        template = self.prompt_loader.load_prompt("extract_entities", version=version)
        prompt = ChatPromptTemplate.from_template(template)
        return prompt | self.llm | StrOutputParser()

    @retry_api_call(max_attempts=3, min_wait=1, max_wait=10)
    def _call_llm(self, chain, text: str) -> str:
        return chain.invoke({"text": text})

    def extract_and_index_paper(self, paper: Paper, prompt_version: str = "v1") -> Dict[str, Any]:
        """
        Runs entity/relation extraction on the paper summary/intro chunks and indexes them.
        To avoid massive LLM calls on every chunk, we select the top 3 chunks (e.g. abstract/intro)
        to extract the core methodology, authors, algorithms, and datasets.
        """
        logger.info(f"Extracting entities and relations for paper: {paper.title or paper.paper_id}")
        
        # Collect top text chunks (up to first 3 elements of the parsed paper)
        texts_to_extract = []
        element_count = 0
        for el in paper.elements:
            if el.element_type in {"Text", "NarrativeText", "Title", "Heading"} and len(el.text.split()) > 5:
                texts_to_extract.append(el.text)
                element_count += 1
                if element_count >= 3:
                    break

        combined_text = "\n\n".join(texts_to_extract)
        if not combined_text:
            logger.warning("No narrative text elements found to extract entities from.")
            return {"entities": 0, "relations": 0}

        try:
            chain = self._get_chain(version=prompt_version)
            raw_response = self._call_llm(chain, combined_text).strip()
            
            # Clean up markdown output blocks
            if raw_response.startswith("```json"):
                raw_response = raw_response[7:]
            if raw_response.startswith("```"):
                raw_response = raw_response[3:]
            if raw_response.endswith("```"):
                raw_response = raw_response[:-3]
            
            raw_response = raw_response.strip()
            data = json.loads(raw_response)
            
            entities_added = 0
            relations_added = 0

            # 1. Add paper itself as a primary node
            paper_node_id = self.db.add_entity(
                name=paper.title or paper.paper_id,
                type="Paper",
                paper_id=paper.paper_id
            )
            entities_added += 1

            # 2. Add authors if present
            if paper.authors:
                for author in paper.authors:
                    author_id = self.db.add_entity(name=author, type="Author", paper_id=paper.paper_id)
                    self.db.add_relation(
                        source_id=author_id,
                        target_id=paper_node_id,
                        relation_type="wrote",
                        description=f"Author of paper '{paper.title}'"
                    )
                    entities_added += 1
                    relations_added += 1

            # 3. Add extracted entities
            name_to_db_id = {}
            for ent in data.get("entities", []):
                ent_name = ent.get("name")
                ent_type = ent.get("type")
                if ent_name and ent_type:
                    db_id = self.db.add_entity(name=ent_name, type=ent_type, paper_id=paper.paper_id)
                    name_to_db_id[(ent_name.lower(), ent_type.lower())] = db_id
                    
                    # Link to source paper
                    self.db.add_relation(
                        source_id=paper_node_id,
                        target_id=db_id,
                        relation_type="contains",
                        description=f"Paper discusses the {ent_type} '{ent_name}'"
                    )
                    entities_added += 1
                    relations_added += 1

            # 4. Add extracted relationships
            for rel in data.get("relationships", []):
                src_name = rel.get("source_name")
                src_type = rel.get("source_type")
                target_name = rel.get("target_name")
                target_type = rel.get("target_type")
                rel_type = rel.get("relation_type")
                desc = rel.get("description")

                if src_name and src_type and target_name and target_type and rel_type:
                    # Find or add source node
                    src_id = name_to_db_id.get((src_name.lower(), src_type.lower()))
                    if not src_id:
                        src_id = self.db.add_entity(name=src_name, type=src_type, paper_id=paper.paper_id)
                        name_to_db_id[(src_name.lower(), src_type.lower())] = src_id
                        entities_added += 1

                    # Find or add target node
                    target_id = name_to_db_id.get((target_name.lower(), target_type.lower()))
                    if not target_id:
                        target_id = self.db.add_entity(name=target_name, type=target_type, paper_id=paper.paper_id)
                        name_to_db_id[(target_name.lower(), target_type.lower())] = target_id
                        entities_added += 1

                    self.db.add_relation(
                        source_id=src_id,
                        target_id=target_id,
                        relation_type=rel_type,
                        description=desc
                    )
                    relations_added += 1

            logger.info(f"Successfully indexed {entities_added} entities and {relations_added} relations for paper {paper.paper_id}")
            return {"entities": entities_added, "relations": relations_added}
        except Exception as e:
            logger.error(f"Failed to extract and index paper entities: {e}")
            return {"entities": 0, "relations": 0}
