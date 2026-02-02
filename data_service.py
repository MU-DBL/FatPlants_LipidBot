from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Any, List, Tuple

class LLMProvider(Enum):
    GEMINI = "gemini"
    OLLAMA = "ollama"

@dataclass
class EntityMention:
    text: str
    start: int
    end: int


@dataclass
class Citation:
    citation_id: str
    title: str
    abstract: str


@dataclass
class Chunk:
    citation_id: str
    chunk_id: int
    text: str
    title: str


@dataclass
class Hit:
    score: float
    citation_id: str
    chunk_id: int
    text: str
    title: str
