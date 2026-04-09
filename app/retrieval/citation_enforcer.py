from __future__ import annotations
import re
from typing import Dict, List, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from ..models.chunk import Chunk
from ..core.settings import settings
from ..generation.prompt_loader import PromptLoader

class CitationEnforcer:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=settings.GOOGLE_API_KEY) if settings.GOOGLE_API_KEY else None
        self.prompt_loader = PromptLoader()
        try:
            template = self.prompt_loader.load_prompt("citation_check", version="v1")
        except:
            template = "Verify if the claim: '{claim}' is supported by this context: '{context}'. Answer 'Yes' or 'No'."
        self.prompt = ChatPromptTemplate.from_template(template)

    def _normalize(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    def enforce(self, answer: str, chunks: List[Chunk]) -> Dict[str, Any]:
        # Split into sentences considering common abbreviations
        sentence_endings = r'(?<![A-Z])(?<!et al)(?<!e\.g)(?<!i\.e)(?<!Fig)\.\s+'
        sentences = [s.strip() for s in re.split(sentence_endings, answer) if s.strip()]
        
        supported_sentences: List[str] = []
        citations: List[Dict[str, Any]] = []

        for i, sent in enumerate(sentences):
            # 1. Fast overlap filter to find candidates
            candidates = self._find_candidate_chunks(sent, chunks)
            
    def _verify_with_llm(self, claim: str, context: str) -> bool:
        """Isolated LLM check for easier mocking."""
        if not self.llm: return False
        res = self.llm.predict(self.prompt.format(claim=claim, context=context))
        return "yes" in res.lower()

    def enforce(self, answer: str, chunks: List[Chunk]) -> Dict[str, Any]:
        # Split into sentences considering common abbreviations
        sentence_endings = r'(?<![A-Z])(?<!et al)(?<!e\.g)(?<!i\.e)(?<!Fig)\.\s+'
        sentences = [s.strip() for s in re.split(sentence_endings, answer) if s.strip()]
        
        supported_sentences: List[str] = []
        citations: List[Dict[str, Any]] = []

        for i, sent in enumerate(sentences):
            # 1. Fast overlap filter to find candidates
            candidates = self._find_candidate_chunks(sent, chunks)
            
            # 2. LLM verification
            best_chunk = None
            if candidates:
                for chunk in candidates:
                    if self._verify_with_llm(sent, chunk.text):
                        best_chunk = chunk
                        break
            
            if best_chunk:
                label = (
                    f"{best_chunk.title[:20] if best_chunk.title else best_chunk.paper_id}"
                    f" | {best_chunk.section or 'Unknown Section'}"
                    f" | p.{best_chunk.page_start or '?'}"
                )
                supported_sentences.append(f"{sent} [{label}]")
                citations.append({
                    "sentence_index": i,
                    "label": label,
                    "quote": best_chunk.text[:300],
                    "page_start": best_chunk.page_start,
                    "page_end": best_chunk.page_end,
                    "chunk_id": best_chunk.chunk_id,
                })
            else:
                # Optionally keep unsupported sentences but flag them, 
                # or drop them as per strict requirement. 
                # Dropping unsupported claims to ensure reliability.
                pass

        final_answer = ". ".join(supported_sentences).strip()
        if supported_sentences:
            final_answer += "."

        return {
            "answer": final_answer or "Insufficient evidence in retrieved sources.",
            "citations": citations,
            "limitations": [] if supported_sentences else ["The generated answer did not have enough chunk-level support."]
        }

    def _find_candidate_chunks(self, sentence: str, chunks: List[Chunk]) -> List[Chunk]:
        s_norm = self._normalize(sentence)
        sentence_terms = set(re.findall(r"\w+", s_norm))
        if len(sentence_terms) < 5: return []
        
        candidates = []
        for chunk in chunks:
            chunk_terms = set(re.findall(r"\w+", self._normalize(chunk.text)))
            overlap = len(sentence_terms & chunk_terms) / len(sentence_terms)
            if overlap > 0.2:
                candidates.append(chunk)
        return candidates
