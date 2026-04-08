from __future__ import annotations
from abc import ABC, abstractmethod
from ..models.paper import Paper

class BasePaperParser(ABC):
    @abstractmethod
    def parse(self, file_path: str, paper_id: str) -> Paper:
        raise NotImplementedError
