import time
import logging
from typing import Set
from .s3_watcher import S3Watcher
from ..ingestion.parser_factory import get_parser
from ..indexing.chunker import SectionAwareChunker
from ..indexing.chroma_index import ChromaIndex
from ..indexing.bm25_index import BM25Index

logger = logging.getLogger(__name__)

class S3IngestionWorker:
    def __init__(self, bucket_name: str, poll_interval: int = 60):
        self.watcher = S3Watcher(bucket_name)
        self.seen_files: Set[str] = set()
        self.poll_interval = poll_interval
        
        # Initialize pipeline components
        self.parser = get_parser("unstructured")
        self.chunker = SectionAwareChunker()
        self.chroma_index = ChromaIndex()
        self.bm25_index = BM25Index()

    def run(self):
        logger.info(f"S3 Ingestion Worker started for bucket: {self.watcher.bucket}")
        while True:
            try:
                files = self.watcher.list_pdfs()
                for key in files:
                    if key not in self.seen_files:
                        logger.info(f"New paper detected in S3: {key}")
                        local_path = self.watcher.download_file(key)
                        
                        # Trigger Ingestion Pipeline
                        paper_id = key.split("/")[-1].replace(".pdf", "")
                        self._ingest_paper(local_path, paper_id)
                        
                        self.seen_files.add(key)
                
                time.sleep(self.poll_interval)
            except Exception as e:
                logger.error(f"Error in S3 worker loop: {e}")
                time.sleep(10) # Wait a bit before retrying

    def _ingest_paper(self, pdf_path: str, paper_id: str):
        logger.info(f"Starting ingestion for {paper_id}...")
        try:
            # 1. Parse
            paper = self.parser.parse(pdf_path, paper_id)
            
            # 2. Chunk
            chunks = self.chunker.split(paper)
            
            # 3. Index
            self.chroma_index.add_chunks(chunks)
            self.bm25_index.add_chunks(chunks)
            
            logger.info(f"Successfully indexed paper: {paper_id} ({len(chunks)} chunks)")
        except Exception as e:
            logger.error(f"Failed to ingest paper {paper_id}: {e}")
