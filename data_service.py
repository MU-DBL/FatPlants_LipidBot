from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Any, List, Tuple
from typing import List, Literal, Optional
from pydantic import BaseModel

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

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    fuse: Literal["rrf", "vote", "max"] = "rrf"
    per: Literal["chunk"] = "chunk"
    rrf_k: int = 60
    model_names: Optional[List[str]] = None

class CypherQueryRequest(BaseModel):
    query: str

class LipidBotRequest(BaseModel):
    llm_type:str # llama, gptoss
    query: str
    top_k: int = 5
    fuse: Literal["rrf", "vote", "max"] = "rrf"
    per: Literal["chunk"] = "chunk"
    rrf_k: int = 60
    model_names: Optional[List[str]] = None
