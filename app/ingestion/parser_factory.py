from __future__ import annotations
from .pymupdf_parser import PyMuPDFPaperParser
from .unstructured_parser import UnstructuredPaperParser

def get_parser(parser_name: str = "pymupdf"):
    parser_name = parser_name.lower().strip()
    if parser_name == "unstructured":
        return UnstructuredPaperParser()
    if parser_name == "pymupdf":
        return PyMuPDFPaperParser()
    raise ValueError(f"Unsupported parser: {parser_name}")
