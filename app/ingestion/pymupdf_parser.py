from __future__ import annotations
import re
from typing import List, Optional
import fitz
import os
import hashlib
from .base_parser import BasePaperParser
from ..models.paper import ParsedElement, Paper
from ..core.logging import logger

DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)
HEADING_RE = re.compile(r"^(\d+(?:\.\d+){0,5})\s+(.+?)\s*$")

class PyMuPDFPaperParser(BasePaperParser):
    @staticmethod
    def _extract_doi(text: str) -> Optional[str]:
        m = DOI_RE.search(text)
        return m.group(0) if m else None

    def parse(self, file_path: str, paper_id: str) -> Paper:
        doc = fitz.open(file_path)
        elements: List[ParsedElement] = []

        # Generate checksum
        with open(file_path, "rb") as f:
            checksum = hashlib.sha256(f.read()).hexdigest()

        title = doc.metadata.get("title")
        authors_raw = doc.metadata.get("author") or ""
        authors = [a.strip() for a in authors_raw.split(",")] if authors_raw else []
        year = None
        if doc.metadata.get("creationDate"):
            try:
                year = int(doc.metadata.get("creationDate")[2:6])
            except: pass

        current_path: List[str] = []
        current_section = None
        current_subsection = None
        current_subsubsection = None
        current_heading_number = None
        current_heading_level = None

        all_text = []

        for page_idx, page in enumerate(doc):
            # 1. Extract and format tables
            table_markdowns = []
            try:
                if hasattr(page, "find_tables"):
                    tables = page.find_tables()
                    for table in tables:
                        table_data = table.extract()
                        if table_data:
                            md_lines = []
                            headers = [str(x or "").replace("\n", " ").strip() for x in table_data[0]]
                            md_lines.append("| " + " | ".join(headers) + " |")
                            md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
                            for row in table_data[1:]:
                                cols = [str(x or "").replace("\n", " ").strip() for x in row]
                                if len(cols) < len(headers):
                                    cols.extend([""] * (len(headers) - len(cols)))
                                md_lines.append("| " + " | ".join(cols[:len(headers)]) + " |")
                            table_markdowns.append("\n".join(md_lines))
            except Exception as e:
                logger.warning(f"Failed to find tables on page {page_idx + 1}: {e}")

            # 2. Extract standard text lines
            text = page.get_text("text")
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            all_text.extend(lines)

            if page_idx == 0 and not title and lines:
                title = lines[0]

            # 3. Add Table elements
            for md_tbl in table_markdowns:
                elements.append(
                    ParsedElement(
                        element_type="Table",
                        text=md_tbl,
                        page=page_idx + 1,
                        section=current_section,
                        subsection=current_subsection,
                        subsubsection=current_subsubsection,
                        section_path=current_path.copy(),
                        heading_number=current_heading_number,
                        heading_level=current_heading_level,
                        metadata={"chunk_type": "table"},
                    )
                )

            # 4. Process lines
            for line in lines:
                m = HEADING_RE.match(line)
                if m:
                    heading_number = m.group(1)
                    heading_title = m.group(2).strip()
                    level = heading_number.count(".") + 1
                    
                    full_heading = f"{heading_number} {heading_title}"
                    current_path = current_path[: level - 1]
                    current_path.append(full_heading)
                    
                    current_section = current_path[0] if len(current_path) > 0 else None
                    current_subsection = current_path[1] if len(current_path) > 1 else None
                    current_subsubsection = current_path[2] if len(current_path) > 2 else None
                    current_heading_number = heading_number
                    current_heading_level = level

                    elements.append(
                        ParsedElement(
                            element_type="Title",
                            text=line,
                            page=page_idx + 1,
                            section=current_section,
                            subsection=current_subsection,
                            subsubsection=current_subsubsection,
                            section_path=current_path.copy(),
                            heading_number=current_heading_number,
                            heading_level=current_heading_level,
                            metadata={"chunk_type": "heading"},
                        )
                    )
                else:
                    chunk_type = "text"
                    element_type = "NarrativeText"
                    lower = line.lower()
                    if lower.startswith("figure"):
                        chunk_type = "figure_caption"
                    elif lower.startswith("table"):
                        chunk_type = "table_summary"
                    
                    # LaTeX equation check
                    if (re.search(r"\(\d+\)\s*$", line) or 
                        re.search(r"\\(?:alpha|beta|gamma|delta|epsilon|theta|lambda|mu|pi|sigma|phi|omega|sum|int|partial|nabla|times|div|approx|neq|leq|geq|infty|rightarrow)", line) or 
                        "$" in line):
                        chunk_type = "equation"
                        element_type = "Equation"

                    elements.append(
                        ParsedElement(
                            element_type=element_type,
                            text=line,
                            page=page_idx + 1,
                            section=current_section,
                            subsection=current_subsection,
                            subsubsection=current_subsubsection,
                            section_path=current_path.copy(),
                            heading_number=current_heading_number,
                            heading_level=current_heading_level,
                            metadata={"chunk_type": chunk_type},
                        )
                    )

        full_text = "\n".join(all_text)
        doi = self._extract_doi(full_text)
        abstract = next((line for line in all_text if line.lower().startswith("abstract")), None)

        return Paper(
            paper_id=paper_id,
            source_path=file_path,
            title=title or "Unknown Title",
            doi=doi,
            authors=authors,
            year=year,
            abstract=abstract,
            elements=elements,
            metadata={"checksum": checksum}
        )
