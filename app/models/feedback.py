from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class Feedback(BaseModel):
    query: str
    answer: str
    rating: int  # +1 = 👍 , -1 = 👎
    comment: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
