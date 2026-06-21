"""Shared data models for all Graphify Knowledge Onboarding Agent pipeline stages.

These dataclasses are the *only* contracts between pipeline stages.
Stage modules import from here; they never import from sibling stage modules.

Pipeline data flow:
    FileEvent  → [Ingestion]     → Chunk
    Chunk      → [Embeddings]    → EmbeddedChunk
    EmbeddedChunk → [Storage]
    query str  → [Retrieval]     → RetrievedChunk
    RetrievedChunk → [Orchestration] → Response
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class FileEvent:
    """A filesystem event emitted by the FileWatcher."""

    path: Path
    event_type: str  # "created" | "modified" | "deleted"
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Section:
    """A section within a parsed document, delimited by a heading."""

    heading: str    # heading text; empty string for preamble or headingless content
    level: int      # 1–6 for H1–H6; 0 for content with no heading
    content: str    # plain text content of this section (markdown stripped)


@dataclass
class ParsedDocument:
    """A markdown document after parsing, before chunking.

    Produced by MarkdownParser; consumed by ChunkingStrategy.
    """

    source_path: Path
    title: str
    content: str            # full plain-text body (all sections joined)
    sections: list[Section]
    front_matter: dict      # YAML front matter key-value pairs (may be empty)
    modified_at: datetime
    word_count: int


@dataclass
class Chunk:
    """A text chunk ready to be embedded.

    Produced by ChunkingStrategy; consumed by EmbeddingProvider and VectorStore.
    """

    id: str             # stable identifier: "<source_stem>:<chunk_index>"
    source_path: Path
    content: str
    chunk_index: int    # 0-based position within the source document
    metadata: dict      # heading context, word count, source file path, etc.
    content_hash: str   # SHA-256 hex digest of content — used for change detection


@dataclass
class EmbeddedChunk:
    """A Chunk paired with its vector embedding.

    Produced by EmbeddingProvider; consumed by VectorStore.
    """

    chunk: Chunk
    vector: list[float]


@dataclass
class RetrievedChunk:
    """A Chunk returned from a similarity query, with a relevance score.

    Produced by SemanticSearch; consumed by the Orchestration stage.
    """

    chunk: Chunk
    score: float    # similarity score; higher = more relevant (cosine: 0.0–1.0)


@dataclass
class Response:
    """The final answer produced by the Orchestration stage.

    Produced by QueryEngine; the terminal output of the pipeline.
    """

    answer: str                     # LLM-generated answer text
    sources: list[RetrievedChunk]   # chunks used as context for the answer
    query: str                      # the original question


# ---------------------------------------------------------------------------
# Knowledge graph data models (Phase 6)
# ---------------------------------------------------------------------------

@dataclass
class Entity:
    """A single entity extracted from a Chunk by the EntityExtractor.

    Produced by EntityExtractor; consumed by GraphStore.
    """

    name: str           # original-casing display name (e.g. "FastAPI")
    entity_type: str    # e.g. "Framework", "Concept", "Technology"
    source_path: Path   # the document this entity was found in
    chunk_id: str       # the Chunk.id that contained this entity


@dataclass
class Relationship:
    """A directed relationship between two entities extracted from a Chunk.

    Produced by EntityExtractor; consumed by GraphStore.
    """

    source_entity: str      # normalized (lowercase) name of the source entity
    target_entity: str      # normalized (lowercase) name of the target entity
    relationship_type: str  # e.g. "Uses", "DependsOn", "Implements"
    source_path: Path       # the document this relationship was found in
    chunk_id: str           # the Chunk.id that evidenced this relationship


@dataclass
class GraphQueryResult:
    """Result of a knowledge graph traversal query.

    Produced by GraphStore.query_context(); consumed by GraphRetriever.
    """

    entity_names: list[str]         # normalized node names that matched the query
    chunk_ids: list[str]            # all Chunk.id values referenced by matched nodes
    source_paths: list[Path]        # all source documents referenced by matched nodes
