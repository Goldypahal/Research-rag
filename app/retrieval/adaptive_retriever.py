from __future__ import annotations
import logging
from typing import Dict
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ..core.settings import settings
from ..core.retry_utils import retry_api_call
from ..generation.prompt_loader import PromptLoader

logger = logging.getLogger(__name__)

class AdaptiveRetrieverConfig:
    # Maps query intent to specific retrieval limits:
    # {intent: (final_top_k, dense_top_k, bm25_top_k, merged_top_k)}
    INTENT_MAPS = {
        "definition": (3, 10, 10, 8),
        "equation": (8, 15, 15, 12),
        "methodology": (8, 15, 15, 12),
        "figure/table": (8, 15, 15, 12),
        "summarization": (10, 20, 20, 15),
        "comparison": (15, 30, 30, 20),
        "multi-paper analysis": (15, 30, 30, 20),
        "literature review": (25, 40, 40, 30),
        "research gap": (35, 50, 50, 40),
    }

    @classmethod
    def get_limits(cls, intent: str) -> tuple[int, int, int, int]:
        return cls.INTENT_MAPS.get(intent, (8, 20, 20, 15))  # Default fallback values

class AdaptiveRetriever:
    def __init__(self):
        if settings.USE_LOCAL_LLM:
            logger.info(f"Initializing AdaptiveRetriever using local ChatOllama ({settings.OLLAMA_MODEL})...")
            self.llm = ChatOllama(
                model=settings.OLLAMA_MODEL,
                base_url=settings.OLLAMA_BASE_URL,
                temperature=0
            )
        elif settings.GOOGLE_API_KEY:
            logger.info("Initializing AdaptiveRetriever using ChatGoogleGenerativeAI (gemini-2.5-flash)...")
            from langchain_google_genai import ChatGoogleGenerativeAI
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=settings.GOOGLE_API_KEY,
                temperature=0
            )
        else:
            logger.info(f"Initializing AdaptiveRetriever using fallback ChatOllama ({settings.OLLAMA_MODEL})...")
            self.llm = ChatOllama(
                model=settings.OLLAMA_MODEL,
                base_url=settings.OLLAMA_BASE_URL,
                temperature=0
            )
        self.prompt_loader = PromptLoader()

    def _get_chain(self, version: str = "v1"):
        template = self.prompt_loader.load_prompt("query_classify_intent", version=version)
        prompt = ChatPromptTemplate.from_template(template)
        return prompt | self.llm | StrOutputParser()

    @retry_api_call(max_attempts=3, min_wait=1, max_wait=10)
    def _call_llm(self, chain, query: str) -> str:
        return chain.invoke({"query": query})

    def classify_intent(self, query: str, prompt_version: str = "v1") -> str:
        logger.info(f"Classifying query intent: '{query}'")
        try:
            chain = self._get_chain(version=prompt_version)
            raw_response = self._call_llm(chain, query).strip().lower()
            
            # Match 10 distinct intents
            intents = [
                "definition", "comparison", "summarization", "methodology", 
                "dataset", "equation", "figure/table", "literature review", 
                "research gap", "multi-paper analysis"
            ]
            for intent in intents:
                if intent in raw_response:
                    logger.info(f"Classified query intent: {intent}")
                    return intent
            
            logger.warning(f"Unexpected classification response: '{raw_response}'. Defaulting to 'methodology'.")
        except Exception as e:
            logger.error(f"Failed to classify query intent via LLM: {e}. Defaulting to 'methodology'.")
            
        return "methodology"

    def get_retrieval_limits(self, query: str, prompt_version: str = "v1") -> Dict[str, int]:
        intent = self.classify_intent(query, prompt_version=prompt_version)
        final_top_k, dense_top_k, bm25_top_k, merged_top_k = AdaptiveRetrieverConfig.get_limits(intent)
        return {
            "intent": intent,
            "final_top_k": final_top_k,
            "dense_top_k": dense_top_k,
            "bm25_top_k": bm25_top_k,
            "merged_top_k": merged_top_k
        }
