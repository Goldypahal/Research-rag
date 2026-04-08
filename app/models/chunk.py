from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class Chunk(BaseModel):
    chunk_id: str
    paper_id: str
    text: str

    title: Optional[str] = None
    doi: Optional[str] = None
    authors: List[str] = Field(default_factory=list)
    year: Optional[int] = None

    section: Optional[str] = None
    subsection: Optional[str] = None
    subsubsection: Optional[str] = None

    heading_number: Optional[str] = None
    heading_title: Optional[str] = None
    heading_level: Optional[int] = None
    section_path: List[str] = Field(default_factory=list)

    page_start: Optional[int] = None
    page_end: Optional[int] = None
    chunk_type: str = "text"
    image_path: Optional[str] = None

    metadata: Dict[str, Any] = Field(default_factory=dict)

    def flat_metadata(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "paper_id": self.paper_id,
            "title": self.title,
            "doi": self.doi,
            "authors": ", ".join(self.authors) if self.authors else None,
            "year": self.year,
            "section": self.section,
            "subsection": self.subsection,
            "subsubsection": self.subsubsection,
            "heading_number": self.heading_number,
            "heading_title": self.heading_title,
            "heading_level": self.heading_level,
            "section_path": " > ".join(self.section_path) if self.section_path else None,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "chunk_type": self.chunk_type,
            "image_path": self.image_path,
            **self.metadata,
        }