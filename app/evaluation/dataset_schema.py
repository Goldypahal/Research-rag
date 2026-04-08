from pydantic import BaseModel
from typing import List, Optional

class GoldenQA(BaseModel):
    question: str
    gold_answer: str
    paper_id: str
    gold_chunk_ids: List[str]
    answerable: bool

class GoldenDataset(BaseModel):
    pairs: List[GoldenQA]
