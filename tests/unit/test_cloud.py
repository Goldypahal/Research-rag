import pytest
from unittest.mock import MagicMock, patch
from app.cloud.s3_ingestion_worker import S3IngestionWorker

def test_s3_worker_logic():
    """Verify that the worker detects new files and calls ingestion."""
    with patch("app.cloud.s3_ingestion_worker.S3Watcher") as mock_watcher_cls:
        mock_watcher = mock_watcher_cls.return_value
        mock_watcher.list_pdfs.side_effect = [
            ["paper1.pdf"], # First poll
            ["paper1.pdf", "paper2.pdf"] # Second poll
        ]
        mock_watcher.download_file.return_value = "/tmp/paper.pdf"
        
        with patch("app.cloud.s3_ingestion_worker.ChromaIndex"), \
             patch("app.cloud.s3_ingestion_worker.BM25Index"), \
             patch("app.cloud.s3_ingestion_worker.get_parser"), \
             patch("app.cloud.s3_ingestion_worker.SectionAwareChunker"):
             
            worker = S3IngestionWorker(bucket_name="test-bucket")
            # Clear mocks and set them explicitly for our logic check
            worker.parser = MagicMock()
            worker.chunker = MagicMock()
            worker.chroma_index = MagicMock()
            worker.bm25_index = MagicMock()
            
            # We manually trigger the loop logic to avoid infinite sleep
            # 1. First poll
            files = mock_watcher.list_pdfs()
            for f in files:
                if f not in worker.seen_files:
                    worker._ingest_paper(mock_watcher.download_file(f), f)
                    worker.seen_files.add(f)
            
            assert "paper1.pdf" in worker.seen_files
            assert worker.parser.parse.call_count == 1
            
            # 2. Second poll
            files = mock_watcher.list_pdfs()
            for f in files:
                if f not in worker.seen_files:
                    worker._ingest_paper(mock_watcher.download_file(f), f)
                    worker.seen_files.add(f)
                    
            assert "paper2.pdf" in worker.seen_files
            assert worker.parser.parse.call_count == 2
