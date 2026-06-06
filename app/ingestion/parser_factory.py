from __future__ import annotations

def get_parser(parser_name: str = "pymupdf"):
    parser_name = parser_name.lower().strip()
    if parser_name == "unstructured":
        from .unstructured_parser import UnstructuredPaperParser
        return UnstructuredPaperParser()
    if parser_name == "pymupdf":
        from .pymupdf_parser import PyMuPDFPaperParser
        return PyMuPDFPaperParser()
    raise ValueError(f"Unsupported parser: {parser_name}")
