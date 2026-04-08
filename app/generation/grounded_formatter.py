from typing import List, Dict, Any
from ..models.chunk import Chunk

class GroundedFormatter:
    def format_context(self, chunks: List[Chunk]) -> str:
        formatted = []
        for i, chunk in enumerate(chunks):
            formatted.append(f"--- SOURCE {i} (Paper: {chunk.title}, Section: {chunk.section}) ---\n{chunk.text}")
        return "\n\n".join(formatted)

    def attach_metadata(self, answer: str, citations: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "answer": answer,
            "citations": citations
        }
