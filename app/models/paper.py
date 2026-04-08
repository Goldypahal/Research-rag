from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class ParsedElement(BaseModel):
    element_type: str
    text: str
    page: Optional[int] = None
    section: Optional[str] = None
    subsection: Optional[str] = None
    subsubsection: Optional[str] = None
    section_path: List[str] = Field(default_factory=list)
    heading_number: Optional[str] = None
    heading_level: Optional[int] = None
    image_path: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class Paper(BaseModel):
    paper_id: str
    source_path: str
    title: Optional[str] = None
    doi: Optional[str] = None
    authors: List[str] = Field(default_factory=list)
    year: Optional[int] = None
    abstract: Optional[str] = None
    elements: List[ParsedElement] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)