import boto3
import os
import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

class S3Watcher:
    def __init__(self, bucket_name: str, download_dir: str = "data/raw_papers"):
        self.bucket = bucket_name
        self.download_dir = download_dir
        os.makedirs(download_dir, exist_ok=True)
        # Assumes AWS credentials are in env or config
        self.s3 = boto3.client("s3")

    def list_pdfs(self) -> List[str]:
        try:
            objects = self.s3.list_objects_v2(Bucket=self.bucket)
            pdf_files = []
            if "Contents" in objects:
                for obj in objects["Contents"]:
                    key = obj["Key"]
                    if key.lower().endswith(".pdf"):
                        pdf_files.append(key)
            return pdf_files
        except Exception as e:
            logger.error(f"Failed to list S3 objects: {e}")
            return []

    def download_file(self, key: str) -> str:
        local_path = os.path.join(self.download_dir, os.path.basename(key))
        try:
            logger.info(f"Downloading {key} from S3...")
            self.s3.download_file(self.bucket, key, local_path)
            return local_path
        except Exception as e:
            logger.error(f"Failed to download {key} from S3: {e}")
            raise
