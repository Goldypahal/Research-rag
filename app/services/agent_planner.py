from __future__ import annotations
import logging
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ..core.settings import settings
from ..core.retry_utils import retry_api_call
from ..generation.prompt_loader import PromptLoader

logger = logging.getLogger(__name__)

class AgentPlanner:
    def __init__(self):
        if settings.USE_LOCAL_LLM:
            logger.info(f"Initializing AgentPlanner using local ChatOllama ({settings.OLLAMA_MODEL})...")
            self.llm = ChatOllama(
                model=settings.OLLAMA_MODEL,
                base_url=settings.OLLAMA_BASE_URL,
                temperature=0
            )
        elif settings.GOOGLE_API_KEY:
            logger.info("Initializing AgentPlanner using ChatGoogleGenerativeAI (gemini-2.5-flash)...")
            from langchain_google_genai import ChatGoogleGenerativeAI
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=settings.GOOGLE_API_KEY,
                temperature=0
            )
        else:
            logger.info(f"Initializing AgentPlanner using fallback ChatOllama ({settings.OLLAMA_MODEL})...")
            self.llm = ChatOllama(
                model=settings.OLLAMA_MODEL,
                base_url=settings.OLLAMA_BASE_URL,
                temperature=0
            )
        self.prompt_loader = PromptLoader()

    def _get_chain(self, version: str = "v1"):
        template = self.prompt_loader.load_prompt("agent_planner", version=version)
        prompt = ChatPromptTemplate.from_template(template)
        return prompt | self.llm | StrOutputParser()

    @retry_api_call(max_attempts=3, min_wait=1, max_wait=10)
    def _call_llm(self, chain, query: str) -> str:
        return chain.invoke({"query": query})

    def plan_route(self, query: str, prompt_version: str = "v1") -> str:
        """
        Determines the routing path for the query.
        Returns one of: 'figure', 'compare', 'multihop', 'standard'.
        """
        logger.info(f"Planning route for query: '{query}'")
        try:
            chain = self._get_chain(version=prompt_version)
            raw_response = self._call_llm(chain, query).strip().lower()
            
            for path in ["figure", "compare", "multihop", "standard"]:
                if path in raw_response:
                    logger.info(f"Routed query to path: {path}")
                    return path
            
            logger.warning(f"Unexpected planner response: '{raw_response}'. Defaulting to 'standard'.")
        except Exception as e:
            logger.error(f"Agent planner failed to route: {e}. Defaulting to 'standard'.")
            
        return "standard"
