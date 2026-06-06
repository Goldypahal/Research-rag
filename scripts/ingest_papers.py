import os
import sys
# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ingestion.parser_factory import get_parser
from app.indexing.chunker import SectionAwareChunker
from app.indexing.chroma_index import ChromaIndex
from app.indexing.bm25_index import BM25Index
from app.core.logging import logger

def main(input_dir="data/raw_papers", parser_name="pymupdf"):
    parser = get_parser(parser_name)
    chunker = SectionAwareChunker()
    chroma = ChromaIndex()
    bm25 = BM25Index()
    
    if not os.path.exists(input_dir):
        logger.error(f"Directory {input_dir} does not exist.")
        return

    for filename in os.listdir(input_dir):
        if filename.endswith(".pdf"):
            path = os.path.join(input_dir, filename)
            logger.info(f"Ingesting {filename}...")
            
            paper_id = filename.replace(".pdf", "")
            paper = parser.parse(path, paper_id)
            chunks = chunker.split(paper)
            
            logger.info(f"Split into {len(chunks)} chunks.")
            
            chroma.add_chunks(chunks)
            bm25.add_chunks(chunks)
            
            logger.info(f"Indexing complete for {filename}.")

if __name__ == "__main__":
    # Support specifying a custom directory or parser name if needed
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="data/raw_papers", help="Directory containing raw PDFs")
    parser.add_argument("--parser", default="pymupdf", choices=["pymupdf", "unstructured"], help="Parser strategy to use")
    args = parser.parse_args()
    
    main(input_dir=args.dir, parser_name=args.parser)
