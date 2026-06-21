"""Protocol definitions for all inter-stage contracts in Graphify Knowledge Onboarding Agent.

All concrete implementations must conform to these Protocols.
Stages must only import from this module — never from a sibling stage's module.
The module graph is a strict DAG: no circular imports are permitted.

Pipeline order (data flows top-to-bottom):
    Ingestion → Embeddings → Storage → Retrieval → Orchestration
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from knowledge_onboarding_agent.models import (
        Chunk,
        Entity,
        GraphQueryResult,
        ParsedDocument,
        Relationship,
        RetrievedChunk,
    )


class EmbeddingProvider(Protocol):
    """Converts text into vector embeddings using a local model."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns one float vector per input text."""
        ...


class VectorStore(Protocol):
    """Persists and retrieves embeddings with associated metadata."""

    def upsert(
        self,
        ids: list[str],
        vectors: list[list[float]],
        metadatas: list[dict],
    ) -> None:
        """Insert or update records. Identified by id."""
        ...

    def query(self, vector: list[float], top_k: int) -> list[dict]:
        """Return the top_k most similar records to the given vector."""
        ...

    def delete(self, ids: list[str]) -> None:
        """Remove records by id."""
        ...

    def count(self) -> int:
        """Return the total number of stored records."""
        ...


class ChunkingStrategy(Protocol):
    """Splits a ParsedDocument into a list of Chunk objects."""

    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        """Split *document* into chunks. Returns List[Chunk]."""
        ...


class Retriever(Protocol):
    """Searches stored embeddings for chunks relevant to a query string."""

    def search(self, query: str) -> list[RetrievedChunk]:
        """Return the most relevant chunks for *query*, ordered by descending score."""
        ...


# ---------------------------------------------------------------------------
# Knowledge graph Protocols (Phase 6)
# ---------------------------------------------------------------------------

class EntityExtractor(Protocol):
    """Extracts entities and relationships from a Chunk using a local LLM."""

    def extract(self, chunk: Chunk) -> tuple[list[Entity], list[Relationship]]:
        """Return all entities and relationships found in *chunk*.

        Returns two parallel lists: entities first, then relationships.
        Both lists may be empty if no meaningful entities are found.
        """
        ...


class GraphStore(Protocol):
    """Persists and queries a knowledge graph of entities and relationships."""

    def upsert(
        self,
        entities: list[Entity],
        relationships: list[Relationship],
    ) -> None:
        """Insert or update entities and relationships.

        Nodes and edges are identified by their normalized names and types;
        duplicate inserts are merged rather than duplicated.
        """
        ...

    def query_context(
        self,
        entity_names: list[str],
        hops: int = 1,
    ) -> GraphQueryResult:
        """Traverse the graph starting from *entity_names* up to *hops* away.

        Returns a ``GraphQueryResult`` with all matched nodes' chunk references.
        Returns an empty result when no nodes match.
        """
        ...

    def delete_by_source(self, source_path: Path) -> None:
        """Remove all nodes and edges that originated from *source_path*.

        Nodes that also appear in other source documents are not deleted;
        only their reference to *source_path* is removed.
        """
        ...

    def get_processed_chunk_ids(self) -> set[str]:
        """Return the set of Chunk IDs for which extraction has been completed.

        Used by the ingestion pipeline to skip already-processed chunks.
        """
        ...
