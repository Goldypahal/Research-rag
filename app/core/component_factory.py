"""
component_factory.py
---------------------
Builds (and rebuilds) the RAG pipeline components based on the
current LLMMode — LOCAL (Ollama + SBERT) or CLOUD (Gemini + Cohere).

Called once at startup and again whenever the user switches modes
via the /api/v1/mode endpoint.
"""
from __future__ import annotations
import logging
from .settings import settings
from .connectivity import mode_manager, LLMMode

logger = logging.getLogger(__name__)


def build_components():
    """
    Returns a fresh (generator, enforcer, reranker) tuple based on
    the current mode_manager.mode value.
    """
    from ..retrieval.rerank_sbert import SBERTReranker
    from ..generation.answer_chain import AnswerChain
    from ..retrieval.citation_enforcer import CitationEnforcer

    current_mode = mode_manager.mode

    if current_mode == LLMMode.CLOUD and settings.COHERE_API_KEY:
        logger.info("ComponentFactory: building CLOUD components (Gemini + Cohere).")
        from ..retrieval.rerank_cohere import CohereReranker
        reranker = CohereReranker()
    else:
        logger.info("ComponentFactory: building LOCAL components (Ollama + SBERT).")
        reranker = SBERTReranker()

    # AnswerChain and CitationEnforcer already read settings.USE_LOCAL_LLM
    # We override that flag here at runtime based on current mode
    _patch_settings(current_mode)

    generator = AnswerChain()
    enforcer = CitationEnforcer()

    return generator, enforcer, reranker


def _patch_settings(mode: LLMMode) -> None:
    """
    Temporarily patches the settings singleton so AnswerChain and
    CitationEnforcer constructors pick up the correct mode.
    """
    if mode == LLMMode.LOCAL:
        settings.USE_LOCAL_LLM = True
        logger.debug("Settings patched: USE_LOCAL_LLM=True")
    else:
        # Cloud mode — only effective if keys are present
        has_google = bool(settings.GOOGLE_API_KEY)
        settings.USE_LOCAL_LLM = not has_google
        if not has_google:
            logger.warning(
                "CLOUD mode requested but GOOGLE_API_KEY is missing. "
                "Falling back to local Ollama for generation."
            )
        logger.debug(f"Settings patched: USE_LOCAL_LLM={settings.USE_LOCAL_LLM}")
