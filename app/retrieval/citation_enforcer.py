from __future__ import annotations
import re
import json
import logging
from typing import Dict, List, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from ..models.chunk import Chunk
from ..core.settings import settings
from ..generation.prompt_loader import PromptLoader

logger = logging.getLogger(__name__)

BATCH_VERIFICATION_PROMPT = """You are an expert citation and fact-checking assistant.
Your task is to verify which sentences in the "Generated Answer" are directly supported by the provided "Source Chunks" of scientific papers.

[Source Chunks]
{source_chunks_text}

[Generated Answer]
{generated_answer_text}

For each sentence in the "Generated Answer", determine if it is directly supported by any of the "Source Chunks".
Respond strictly with a JSON object in this format:
{{
  "verifications": [
     {{
        "sentence_index": <int, the 0-based index of the sentence>,
        "sentence": "<str, the exact sentence text>",
        "is_supported": <bool, true if supported, false otherwise>,
        "supported_by_chunk_index": <int, the 0-based index of the supporting chunk, or null>,
        "quote": "<str, the exact quote from the chunk supporting the claim, or empty string>"
     }}
  ]
}}
Ensure the response is valid, parsable JSON. Do not include markdown code block formatting like ```json or any trailing characters.
"""

class CitationEnforcer:
    def __init__(self):
        if settings.USE_LOCAL_LLM:
            from langchain_ollama import ChatOllama
            logger.info(f"CitationEnforcer: using local Ollama ({settings.OLLAMA_MODEL}) — data stays on device.")
            self.llm = ChatOllama(
                model=settings.OLLAMA_MODEL,
                base_url=settings.OLLAMA_BASE_URL,
                temperature=0
            )
        elif settings.GOOGLE_API_KEY:
            # Only reached when USE_LOCAL_LLM is explicitly False AND key exists
            logger.info("CitationEnforcer: using Gemini API (cloud mode).")
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=settings.GOOGLE_API_KEY,
                temperature=0
            )
        else:
            logger.warning("CitationEnforcer: no LLM configured. Falling back to overlap heuristic only.")
            self.llm = None
        self.prompt_loader = PromptLoader()

    def _normalize(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    def enforce(self, answer: str, chunks: List[Chunk]) -> Dict[str, Any]:
        if not answer or not chunks:
            return {
                "answer": answer or "Insufficient evidence in retrieved sources.",
                "citations": [],
                "limitations": ["No content or chunks provided for verification."]
            }

        # Split into sentences considering common abbreviations
        sentence_endings = r'(?<![A-Z])(?<!et al)(?<!e\.g)(?<!i\.e)(?<!Fig)\.\s+'
        sentences = [s.strip() for s in re.split(sentence_endings, answer) if s.strip()]
        
        supported_sentences: List[str] = []
        citations: List[Dict[str, Any]] = []
        verification_map = {}

        # 1. Format inputs for Batch LLM verification
        chunks_formatted = []
        for idx, chunk in enumerate(chunks):
            chunks_formatted.append(
                f"Chunk Index {idx}:\n"
                f"Title: {chunk.title or 'Unknown'}\n"
                f"Section: {chunk.section or 'Unknown'}\n"
                f"Content: {chunk.text}\n"
            )
        source_chunks_text = "\n---\n".join(chunks_formatted)

        sentences_formatted = []
        for idx, sent in enumerate(sentences):
            sentences_formatted.append(f"Sentence Index {idx}: {sent}")
        generated_answer_text = "\n".join(sentences_formatted)

        # 2. Try Batch LLM verification first
        if self.llm:
            try:
                prompt_content = BATCH_VERIFICATION_PROMPT.format(
                    source_chunks_text=source_chunks_text,
                    generated_answer_text=generated_answer_text
                )
                logger.info(f"Performing batch citation check for {len(sentences)} claims against {len(chunks)} chunks...")
                response = self.llm.invoke(prompt_content)
                res_text = response.content if hasattr(response, 'content') else str(response)
                
                # Strip markdown code blocks if present
                res_text = res_text.strip()
                if res_text.startswith("```"):
                    res_text = re.sub(r"^```(?:json)?\s*", "", res_text)
                    res_text = re.sub(r"\s*```$", "", res_text)
                
                # Extract first { and last } to isolate JSON
                start_idx = res_text.find("{")
                end_idx = res_text.rfind("}")
                if start_idx != -1 and end_idx != -1:
                    res_text = res_text[start_idx:end_idx+1]
                
                data = json.loads(res_text)
                verifications = data.get("verifications", [])
                
                for item in verifications:
                    s_idx = item.get("sentence_index")
                    is_supp = item.get("is_supported", False)
                    c_idx = item.get("supported_by_chunk_index")
                    if is_supp and s_idx is not None and c_idx is not None:
                        if 0 <= c_idx < len(chunks):
                            verification_map[s_idx] = (chunks[c_idx], item.get("quote", ""))
                logger.info(f"Batch citation check completed successfully. Verified {len(verification_map)} of {len(sentences)} claims.")
            except Exception as e:
                logger.warning(f"Batch citation verification failed ({e}). Falling back to overlap heuristic.")

        # 3. Apply results & run fallback for missing verifications
        for i, sent in enumerate(sentences):
            best_chunk = None
            quote_text = ""
            
            if i in verification_map:
                best_chunk, quote_text = verification_map[i]
            else:
                # Overlap heuristic fallback
                candidates = self._find_candidate_chunks(sent, chunks)
                if candidates:
                    best_chunk = candidates[0]
                    quote_text = best_chunk.text[:300]
            
            if best_chunk:
                title_seg = "Source"
                if best_chunk.title and best_chunk.title.strip() and best_chunk.title.lower() not in {"none", "undefined", "null"}:
                    title_seg = best_chunk.title[:20]
                elif best_chunk.paper_id and best_chunk.paper_id.strip() and best_chunk.paper_id.lower() not in {"none", "undefined", "null"}:
                    title_seg = best_chunk.paper_id
                
                section_seg = best_chunk.section or "Unknown Section"
                if isinstance(section_seg, str) and section_seg.strip().lower() in {"none", "undefined", "null"}:
                    section_seg = "Unknown Section"
                
                page_seg = f"p.{best_chunk.page_start}" if best_chunk.page_start is not None else "p.?"
                label = f"{title_seg} | {section_seg} | {page_seg}"
                supported_sentences.append(f"{sent} [{label}]")
                citations.append({
                    "sentence_index": i,
                    "label": label,
                    "quote": quote_text or best_chunk.text[:300],
                    "page_start": best_chunk.page_start,
                    "page_end": best_chunk.page_end,
                    "chunk_id": best_chunk.chunk_id,
                })
            else:
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
