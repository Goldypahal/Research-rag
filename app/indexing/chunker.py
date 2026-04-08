from __future__ import annotations
import re
import uuid
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Dict
from ..models.chunk import Chunk
from ..models.paper import ParsedElement, Paper

HEADING_RE = re.compile(r"^(\d+(?:\.\d+)*)\s+(.+?)\s*$")

class HeadingTracker:
    def __init__(self) -> None:
        self.stack: List[str] = []
        self.current_heading_number: Optional[str] = None
        self.current_heading_title: Optional[str] = None
        self.current_heading_level: Optional[int] = None

    def update_if_heading(self, text: str) -> Optional[Dict]:
        m = HEADING_RE.match(text.strip())
        if not m:
            return None

        heading_number = m.group(1)
        heading_title = m.group(2).strip()
        level = heading_number.count(".") + 1
        full_heading = f"{heading_number} {heading_title}"

        self.stack = self.stack[: level - 1]
        self.stack.append(full_heading)

        self.current_heading_number = heading_number
        self.current_heading_title = heading_title
        self.current_heading_level = level

        return self.current_metadata()

    def current_metadata(self) -> Dict:
        section = self.stack[0] if len(self.stack) > 0 else None
        subsection = self.stack[1] if len(self.stack) > 1 else None
        subsubsection = self.stack[2] if len(self.stack) > 2 else None

        return {
            "section": section,
            "subsection": subsection,
            "subsubsection": subsubsection,
            "section_path": self.stack.copy(),
            "heading_number": self.current_heading_number,
            "heading_title": self.current_heading_title,
            "heading_level": self.current_heading_level,
        }

class SectionAwareChunker:
    def __init__(
        self,
        chunk_size_tokens: int = 800,
        chunk_overlap_tokens: int = 100,
        min_chunk_tokens: int = 500,
    ):
        self.chunk_size_tokens = chunk_size_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens
        self.min_chunk_tokens = min_chunk_tokens

    def _split_text_with_overlap(self, text: str) -> List[str]:
        words = text.split()
        if not words:
            return []

        max_words = int(self.chunk_size_tokens * 0.75)
        overlap_words = int(self.chunk_overlap_tokens * 0.75)

        chunks = []
        start = 0
        while start < len(words):
            end = min(len(words), start + max_words)
            chunks.append(" ".join(words[start:end]).strip())
            if end == len(words):
                break
            start = max(start + 1, end - overlap_words)
        return chunks

    def split(self, paper: Paper) -> List[Chunk]:
        tracker = HeadingTracker()
        chunks: List[Chunk] = []
        current_text_bucket: Optional[Dict] = None

        def flush_bucket(bucket: Dict):
            if not bucket or not bucket["lines"]:
                return
            combined_text = "\n".join(bucket["lines"]).strip()
            if not combined_text:
                return

            text_parts = self._split_text_with_overlap(combined_text)
            page_start = min(bucket["pages"]) if bucket["pages"] else None
            page_end = max(bucket["pages"]) if bucket["pages"] else None

            for part in text_parts:
                chunks.append(
                    Chunk(
                        chunk_id=str(uuid.uuid4()),
                        paper_id=paper.paper_id,
                        text=part,
                        title=paper.title,
                        doi=paper.doi,
                        authors=paper.authors,
                        year=paper.year,
                        section=bucket["section"],
                        subsection=bucket["subsection"],
                        subsubsection=bucket["subsubsection"],
                        heading_number=bucket["heading_number"],
                        heading_title=bucket["heading_title"],
                        heading_level=bucket["heading_level"],
                        section_path=bucket["section_path"],
                        page_start=page_start,
                        page_end=page_end,
                        chunk_type=bucket["chunk_type"],
                        metadata={"source_path": paper.source_path}
                    )
                )

        for el in paper.elements:
            text = el.text.strip()
            if not text:
                continue

            heading_meta = tracker.update_if_heading(text)
            if heading_meta:
                flush_bucket(current_text_bucket)
                current_text_bucket = None
                continue

            current_meta = tracker.current_metadata()
            chunk_type = el.metadata.get("chunk_type", "text")
            
            # If it's a table or figure, it should be its own chunk (flush preceding text)
            if chunk_type in {"table", "table_summary", "figure", "figure_caption"}:
                flush_bucket(current_text_bucket)
                current_text_bucket = None
                
                # Create dedicated chunk for this element
                chunks.append(
                    Chunk(
                        chunk_id=str(uuid.uuid4()),
                        paper_id=paper.paper_id,
                        text=text,
                        title=paper.title,
                        doi=paper.doi,
                        authors=paper.authors,
                        year=paper.year,
                        section=current_meta["section"],
                        subsection=current_meta["subsection"],
                        subsubsection=current_meta["subsubsection"],
                        heading_number=current_meta["heading_number"],
                        heading_title=current_meta["heading_title"],
                        heading_level=current_meta["heading_level"],
                        section_path=current_meta["section_path"],
                        page_start=el.page,
                        page_end=el.page,
                        chunk_type=chunk_type,
                        image_path=el.image_path,
                        metadata={"source_path": paper.source_path}
                    )
                )
                continue

            # Standard narrative text handling
            key = tuple(current_meta["section_path"])
            if current_text_bucket is None or current_text_bucket["key"] != key:
                flush_bucket(current_text_bucket)
                current_text_bucket = {
                    "key": key,
                    "section": current_meta["section"],
                    "subsection": current_meta["subsection"],
                    "subsubsection": current_meta["subsubsection"],
                    "heading_number": current_meta["heading_number"],
                    "heading_title": current_meta["heading_title"],
                    "heading_level": current_meta["heading_level"],
                    "section_path": current_meta["section_path"],
                    "lines": [],
                    "pages": [],
                    "chunk_type": "text"
                }

            current_text_bucket["lines"].append(text)
            if el.page is not None:
                current_text_bucket["pages"].append(el.page)

        # Final flush
        flush_bucket(current_text_bucket)

        return chunks
