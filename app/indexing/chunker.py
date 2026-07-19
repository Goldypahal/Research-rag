from __future__ import annotations
import re
import uuid
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Dict, Any
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
        use_semantic_splits: bool = True,
    ):
        self.chunk_size_tokens = chunk_size_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens
        self.min_chunk_tokens = min_chunk_tokens
        self.use_semantic_splits = use_semantic_splits

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

    def _split_text_semantically(self, text: str) -> List[str]:
        # Split text into sentences using abbreviation-aware regex
        sentence_endings = r'(?<![A-Z])(?<!et al)(?<!e\.g)(?<!i\.e)(?<!Fig)\.\s+'
        sentences = [s.strip() for s in re.split(sentence_endings, text) if s.strip()]
        if not sentences:
            return []
        if len(sentences) == 1:
            return sentences

        try:
            # Lazy import to avoid startup delays
            from sentence_transformers import SentenceTransformer
            import numpy as np

            # Use local CPU-friendly lightweight model
            model = SentenceTransformer("all-MiniLM-L6-v2")
            embeddings = model.encode(sentences, convert_to_numpy=True)
            
            similarities = []
            for i in range(len(embeddings) - 1):
                vec1 = embeddings[i]
                vec2 = embeddings[i+1]
                norm1 = np.linalg.norm(vec1)
                norm2 = np.linalg.norm(vec2)
                if norm1 > 0 and norm2 > 0:
                    sim = np.dot(vec1, vec2) / (norm1 * norm2)
                else:
                    sim = 0.0
                similarities.append(sim)
                
            if similarities:
                # Use a similarity percentile threshold (e.g. 20th percentile)
                threshold = np.percentile(similarities, 20)
            else:
                threshold = 0.5

            chunks = []
            current_chunk_sentences = [sentences[0]]
            for i, sim in enumerate(similarities):
                current_len = sum(len(s.split()) for s in current_chunk_sentences)
                # Split if similarity is below threshold, or if chunk size is getting too large
                if sim < threshold or current_len > (self.chunk_size_tokens * 0.75):
                    chunks.append(". ".join(current_chunk_sentences) + ".")
                    current_chunk_sentences = [sentences[i+1]]
                else:
                    current_chunk_sentences.append(sentences[i+1])
            if current_chunk_sentences:
                chunks.append(". ".join(current_chunk_sentences) + ".")
            return chunks
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Semantic chunking failed: {e}. Falling back to token overlap split.")
            return self._split_text_with_overlap(text)

    def split(self, paper: Paper) -> List[Chunk]:
        tracker = HeadingTracker()
        
        # 1. Parse into initial paragraph-level chunks (using semantic or token-based splitting)
        paragraph_chunks: List[Chunk] = []
        current_text_bucket: Optional[Dict] = None

        def flush_bucket(bucket: Dict):
            if not bucket or not bucket["lines"]:
                return
            combined_text = "\n".join(bucket["lines"]).strip()
            if not combined_text:
                return

            if self.use_semantic_splits:
                text_parts = self._split_text_semantically(combined_text)
            else:
                text_parts = self._split_text_with_overlap(combined_text)

            page_start = min(bucket["pages"]) if bucket["pages"] else None
            page_end = max(bucket["pages"]) if bucket["pages"] else None

            for part in text_parts:
                paragraph_chunks.append(
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
                        chunk_level="paragraph",
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
                paragraph_chunks.append(
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
                        chunk_level="paragraph",
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

        # 2. Generate Sentence Chunks bottom-up from paragraph chunks
        sentence_chunks: List[Chunk] = []
        sentence_endings = r'(?<![A-Z])(?<!et al)(?<!e\.g)(?<!i\.e)(?<!Fig)\.\s+'
        
        for p_chunk in paragraph_chunks:
            # Skip tables/figures for sentence extraction
            if p_chunk.chunk_type in {"table", "figure"}:
                continue
                
            sentences = [s.strip() for s in re.split(sentence_endings, p_chunk.text) if s.strip()]
            for sent in sentences:
                if len(sent.split()) < 3: # Skip very short snippets
                    continue
                s_chunk = Chunk(
                    chunk_id=str(uuid.uuid4()),
                    paper_id=p_chunk.paper_id,
                    text=sent + ".",
                    title=p_chunk.title,
                    doi=p_chunk.doi,
                    authors=p_chunk.authors,
                    year=p_chunk.year,
                    section=p_chunk.section,
                    subsection=p_chunk.subsection,
                    subsubsection=p_chunk.subsubsection,
                    heading_number=p_chunk.heading_number,
                    heading_title=p_chunk.heading_title,
                    heading_level=p_chunk.heading_level,
                    section_path=p_chunk.section_path,
                    page_start=p_chunk.page_start,
                    page_end=p_chunk.page_end,
                    chunk_type="sentence",
                    chunk_level="sentence",
                    parent_id=p_chunk.chunk_id,
                    metadata=p_chunk.metadata.copy()
                )
                sentence_chunks.append(s_chunk)
                p_chunk.children_ids.append(s_chunk.chunk_id)

        # 3. Generate Subsection Chunks
        subsection_chunks: List[Chunk] = []
        grouped_by_sub: Dict[str, List[Chunk]] = {}
        for p_chunk in paragraph_chunks:
            if p_chunk.subsection:
                key = f"{p_chunk.paper_id}::{p_chunk.section}::{p_chunk.subsection}"
                grouped_by_sub.setdefault(key, []).append(p_chunk)

        for key, p_list in grouped_by_sub.items():
            first = p_list[0]
            sub_text = "\n\n".join([c.text for c in p_list])
            sub_chunk = Chunk(
                chunk_id=str(uuid.uuid4()),
                paper_id=first.paper_id,
                text=sub_text,
                title=first.title,
                doi=first.doi,
                authors=first.authors,
                year=first.year,
                section=first.section,
                subsection=first.subsection,
                subsubsection=first.subsubsection,
                heading_number=first.heading_number,
                heading_title=first.heading_title,
                heading_level=first.heading_level,
                section_path=first.section_path[:2],
                page_start=min([c.page_start for c in p_list if c.page_start is not None], default=first.page_start),
                page_end=max([c.page_end for c in p_list if c.page_end is not None], default=first.page_end),
                chunk_type="text",
                chunk_level="subsection",
                children_ids=[c.chunk_id for c in p_list],
                metadata=first.metadata.copy()
            )
            subsection_chunks.append(sub_chunk)
            for c in p_list:
                c.parent_id = sub_chunk.chunk_id

        # 4. Generate Section Chunks
        section_chunks: List[Chunk] = []
        grouped_by_sec: Dict[str, List[Chunk]] = {}
        # We group subsections if they exist, or standard paragraph chunks if they belong directly to a section
        for sub_chunk in subsection_chunks:
            key = f"{sub_chunk.paper_id}::{sub_chunk.section}"
            grouped_by_sec.setdefault(key, []).append(sub_chunk)
            
        for p_chunk in paragraph_chunks:
            if p_chunk.section and not p_chunk.subsection:
                key = f"{p_chunk.paper_id}::{p_chunk.section}"
                grouped_by_sec.setdefault(key, []).append(p_chunk)

        for key, child_list in grouped_by_sec.items():
            if not child_list:
                continue
            first = child_list[0]
            sec_text = "\n\n".join([c.text for c in child_list])
            sec_chunk = Chunk(
                chunk_id=str(uuid.uuid4()),
                paper_id=first.paper_id,
                text=sec_text,
                title=first.title,
                doi=first.doi,
                authors=first.authors,
                year=first.year,
                section=first.section,
                subsection=first.subsection,
                subsubsection=first.subsubsection,
                heading_number=first.heading_number,
                heading_title=first.heading_title,
                heading_level=first.heading_level,
                section_path=first.section_path[:1],
                page_start=min([c.page_start for c in child_list if c.page_start is not None], default=first.page_start),
                page_end=max([c.page_end for c in child_list if c.page_end is not None], default=first.page_end),
                chunk_type="text",
                chunk_level="section",
                children_ids=[c.chunk_id for c in child_list],
                metadata=first.metadata.copy()
            )
            section_chunks.append(sec_chunk)
            for c in child_list:
                c.parent_id = sec_chunk.chunk_id

        # Return all levels of chunks
        return paragraph_chunks + sentence_chunks + subsection_chunks + section_chunks
