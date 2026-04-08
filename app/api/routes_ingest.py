from fastapi import APIRouter, UploadFile, File, HTTPException
import os
from typing import List
from ..models.paper import Paper
from ..ingestion.parser_factory import get_parser
from ..indexing.chunker import SectionAwareChunker
from ..indexing.chroma_index import ChromaIndex
from ..indexing.bm25_index import BM25Index

router = APIRouter()
chroma = ChromaIndex()
bm25 = BM25Index()

async def _process_single_file(file: UploadFile, parser_name: str) -> Paper:
    # 1. Save file
    temp_path = f"data/raw_papers/{file.filename}"
    os.makedirs("data/raw_papers", exist_ok=True)
    with open(temp_path, "wb") as f:
        f.write(await file.read())
    
    # 2. Parse
    parser = get_parser(parser_name)
    paper_id = file.filename.replace(".pdf", "")
    paper = parser.parse(temp_path, paper_id)
    
    # 3. Chunk
    chunker = SectionAwareChunker()
    chunks = chunker.split(paper)
    
    # 4. Index
    chroma.add_chunks(chunks)
    bm25.add_chunks(chunks)
    
    return paper

@router.post("/ingest", response_model=Paper)
async def ingest_paper(file: UploadFile = File(...), parser_name: str = "pymupdf"):
    return await _process_single_file(file, parser_name)

@router.post("/ingest-batch", response_model=List[Paper])
async def ingest_batch_papers(files: List[UploadFile] = File(...), parser_name: str = "pymupdf"):
    results = []
    for file in files:
        paper = await _process_single_file(file, parser_name)
        results.append(paper)
    return results
