import os
import logging
from typing import List
from ..indexing.chroma_index import ChromaIndex
from ..indexing.bm25_index import BM25Index
from ..models.chunk import Chunk

logger = logging.getLogger(__name__)

def run_auto_ingest_background(chroma: ChromaIndex, bm25: BM25Index):
    import time
    logger.info("Auto-ingestion background thread started.")
    
    # Wait 2 seconds for the FastAPI server startup to fully complete
    time.sleep(2)
    
    try:
        # Check count of the active vector store (local or cloud depending on mode)
        count = chroma.vector_store._collection.count()
        logger.info(f"Current collection count: {count}")
    except Exception as e:
        logger.error(f"Failed to check Chroma collection count: {e}")
        count = 0

    if count > 0:
        logger.info(f"Database already contains {count} chunks. Skipping auto-ingestion.")
        return

    raw_dir = "data/raw_papers"
    if not os.path.exists(raw_dir):
        logger.info("No raw papers directory found. Skipping auto-ingestion.")
        return

    pdf_files = [f for f in os.listdir(raw_dir) if f.endswith(".pdf")]
    if not pdf_files:
        logger.info("No PDF papers found in raw directory. Skipping auto-ingestion.")
        return

    logger.info(f"Database is empty. Starting background auto-ingestion of {len(pdf_files)} papers...")
    
    all_new_chunks: List[Chunk] = []
    
    for filename in pdf_files:
        paper_id = filename.replace(".pdf", "")
        file_path = os.path.join(raw_dir, filename)
        logger.info(f"Auto-ingesting: {filename}")
        try:
            # Import parsers lazily to prevent OOM
            from ..ingestion.parser_factory import get_parser
            from ..indexing.chunker import SectionAwareChunker
            
            parser = get_parser("pymupdf")
            paper = parser.parse(file_path, paper_id)
            
            chunker = SectionAwareChunker()
            chunks = chunker.split(paper)
            
            chroma.add_chunks(chunks)
            all_new_chunks.extend(chunks)
            logger.info(f"Successfully auto-ingested {filename} ({len(chunks)} chunks)")
        except Exception as e:
            logger.error(f"Failed to auto-ingest {filename}: {e}", exc_info=True)

    if all_new_chunks:
        try:
            bm25.add_chunks(all_new_chunks)
            logger.info(f"Successfully synchronized {len(all_new_chunks)} chunks in BM25 index.")
        except Exception as e:
            logger.error(f"Failed to update BM25 index: {e}")

    logger.info("Background auto-ingestion process completed.")
