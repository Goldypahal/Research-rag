from __future__ import annotations
import json
import logging
from typing import List
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ..core.settings import settings
from ..core.retry_utils import retry_api_call
from ..generation.prompt_loader import PromptLoader

logger = logging.getLogger(__name__)

class QueryDecomposer:
    def __init__(self):
        if settings.USE_LOCAL_LLM:
            logger.info(f"Initializing QueryDecomposer using local ChatOllama ({settings.OLLAMA_MODEL})...")
            self.llm = ChatOllama(
                model=settings.OLLAMA_MODEL,
                base_url=settings.OLLAMA_BASE_URL,
                temperature=0
            )
        elif settings.GOOGLE_API_KEY:
            logger.info("Initializing QueryDecomposer using ChatGoogleGenerativeAI (gemini-2.5-flash)...")
            from langchain_google_genai import ChatGoogleGenerativeAI
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=settings.GOOGLE_API_KEY,
                temperature=0
            )
        else:
            logger.info(f"Initializing QueryDecomposer using fallback ChatOllama ({settings.OLLAMA_MODEL})...")
            self.llm = ChatOllama(
                model=settings.OLLAMA_MODEL,
                base_url=settings.OLLAMA_BASE_URL,
                temperature=0
            )
        self.prompt_loader = PromptLoader()

    def _get_chain(self, version: str = "v1"):
        template = self.prompt_loader.load_prompt("query_decompose", version=version)
        prompt = ChatPromptTemplate.from_template(template)
        return prompt | self.llm | StrOutputParser()

    @retry_api_call(max_attempts=3, min_wait=1, max_wait=10)
    def _call_llm(self, chain, query: str) -> str:
        return chain.invoke({"query": query})

    def decompose(self, query: str, prompt_version: str = "v1") -> List[str]:
        logger.info(f"Decomposing query: '{query}'")
        try:
            chain = self._get_chain(version=prompt_version)
            raw_response = self._call_llm(chain, query).strip()
            
            # Clean up potential markdown code formatting
            if raw_response.startswith("```json"):
                raw_response = raw_response[7:]
            if raw_response.startswith("```"):
                raw_response = raw_response[3:]
            if raw_response.endswith("```"):
                raw_response = raw_response[:-3]
            
            raw_response = raw_response.strip()
            sub_queries = json.loads(raw_response)
            
            if isinstance(sub_queries, list) and all(isinstance(q, str) for q in sub_queries):
                logger.info(f"Decomposed queries: {sub_queries}")
                return sub_queries
            else:
                logger.warning(f"Response was JSON but not a list of strings: {sub_queries}")
        except Exception as e:
            logger.error(f"Failed to decompose query via LLM: {e}. Falling back to original query.")
            
        return [query]
