from __future__ import annotations
import os
import re
import hashlib
from typing import List, Optional
from unstructured.partition.pdf import partition_pdf
from .base_parser import BasePaperParser
from .figure_extractor import FigureExtractor
from ..models.paper import ParsedElement, Paper
from ..core.logging import logger

DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)
HEADING_RE = re.compile(r"^(\d+(?:\.\d+){0,5})\s+(.+?)\s*$")

class UnstructuredPaperParser(BasePaperParser):
    def __init__(
        self,
        strategy: str = "hi_res",
        infer_table_structure: bool = True,
        extract_image_block_types: Optional[List[str]] = None,
    ):
        self.strategy = strategy
        self.infer_table_structure = infer_table_structure
        self.extract_image_block_types = extract_image_block_types or ["Image", "Table"]
        self.figure_extractor = FigureExtractor()

    @staticmethod
    def _extract_doi(text: str) -> Optional[str]:
        m = DOI_RE.search(text)
        return m.group(0) if m else None

    @staticmethod
    def _detect_title(elements) -> Optional[str]:
        for el in elements[:15]:
            if getattr(el, "category", "").lower() == "title":
                return str(el).strip()
        for el in elements[:10]:
            text = str(el).strip()
            if len(text) > 30 and not text.lower().startswith("abstract"):
                return text
        return "Unknown Title"

    def parse(self, file_path: str, paper_id: str) -> Paper:
        with open(file_path, "rb") as f:
            checksum = hashlib.sha256(f.read()).hexdigest()

        # Extract textual elements
        elements = partition_pdf(
            filename=file_path,
            strategy=self.strategy,
            infer_table_structure=self.infer_table_structure,
            extract_image_block_types=self.extract_image_block_types,
        )

        # Extract figures using PyMuPDF helper
        extracted_figures = self.figure_extractor.extract_figures(file_path, paper_id)

        parsed_elements: List[ParsedElement] = []
        current_path: List[str] = []
        current_section: Optional[str] = None
        current_subsection: Optional[str] = None
        current_subsubsection: Optional[str] = None
        current_heading_number: Optional[str] = None
        current_heading_level: Optional[int] = None
        all_text = []

        # Keep track of figures used to avoid duplicates
        used_figs = set()

        for el in elements:
            text = str(el).strip()
            if not text: continue
            all_text.append(text)

            category = getattr(el, "category", "Text")
            metadata = getattr(el, "metadata", None)
            page = getattr(metadata, "page_number", None) if metadata else None

            # Detect heading
            m = HEADING_RE.match(text)
            if m:
                heading_number = m.group(1)
                heading_title = m.group(2).strip()
                level = heading_number.count(".") + 1
                full_heading = f"{heading_number} {heading_title}"
                
                current_path = current_path[: level - 1]
                current_path.append(full_heading)
                
                current_heading_number = heading_number
                current_heading_level = level
                current_section = current_path[0] if len(current_path) > 0 else None
                current_subsection = current_path[1] if len(current_path) > 1 else heading_title
                current_subsubsection = current_path[2] if len(current_path) > 2 else None
                
                parsed_elements.append(
                    ParsedElement(
                        element_type="Title",
                        text=text,
                        page=page,
                        section=current_section,
                        subsection=current_subsection,
                        subsubsection=current_subsubsection,
                        section_path=current_path.copy(),
                        heading_number=current_heading_number,
                        heading_level=current_heading_level,
                        metadata={"chunk_type": "heading"},
                    )
                )
                continue

            # Determine chunk type
            chunk_type = "text"
            category_lower = category.lower()
            image_path = None

            if category_lower == "table":
                chunk_type = "table"
            elif category_lower == "image":
                chunk_type = "figure"
                # Naive matching: take the first unused figure from this page
                for fig in extracted_figures:
                    if fig["page"] == page and fig["image_path"] not in used_figs:
                        image_path = fig["image_path"]
                        used_figs.add(image_path)
                        break
            elif text.lower().startswith("table"):
                chunk_type = "table_caption"
            elif text.lower().startswith(("figure", "fig.")):
                chunk_type = "figure_caption"

            parsed_elements.append(
                ParsedElement(
                    element_type=category,
                    text=text,
                    page=page,
                    section=current_section,
                    subsection=current_subsection,
                    subsubsection=current_subsubsection,
                    section_path=current_path.copy(),
                    heading_number=current_heading_number,
                    heading_level=current_heading_level,
                    image_path=image_path,
                    metadata={"chunk_type": chunk_type, "category": category},
                )
            )

        full_text = "\n".join(all_text)
        doi = self._extract_doi(full_text)
        title = self._detect_title(elements)
        abstract = next((t for t in all_text[:50] if t.lower().startswith("abstract")), None)

        return Paper(
            paper_id=paper_id,
            source_path=file_path,
            title=title,
            doi=doi,
            abstract=abstract,
            elements=parsed_elements,
            metadata={"checksum": checksum}
        )
