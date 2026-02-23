"""Data models for charter party clause extraction."""

from dataclasses import dataclass
from typing import List


@dataclass
class Clause:
    """Represents a single legal clause from the charter party document."""

    id: str
    title: str
    text: str

    def to_dict(self) -> dict:
        return {"id": self.id, "title": self.title, "text": self.text}


@dataclass
class ExtractionResult:
    """Result of clause extraction from the document."""

    clauses: List[Clause]
    total_pages_processed: int

    def to_dict(self) -> dict:
        return {
            "total_clauses": len(self.clauses),
            "clauses": [c.to_dict() for c in self.clauses],
        }
