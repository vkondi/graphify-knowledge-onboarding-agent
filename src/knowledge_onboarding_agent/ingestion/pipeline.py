"""IngestionPipeline: end-to-end orchestration of FileEvent → List[Chunk].

Optionally drives entity extraction and graph storage alongside the vector
pipeline when ``knowledge_graph.enabled`` is true in settings (Phase 6).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from knowledge_onboarding_agent.config import Settings
from knowledge_onboarding_agent.ingestion.chunker import SentenceWindowChunker
from knowledge_onboarding_agent.ingestion.parser import MarkdownParser
from knowledge_onboarding_agent.models import Chunk, FileEvent

if TYPE_CHECKING:
    from knowledge_onboarding_agent.interfaces import EntityExtractor, GraphStore

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Processes a FileEvent (or a direct file path) into a list of Chunks.

    Wires together MarkdownParser → SentenceWindowChunker.

    When *entity_extractor* and *graph_store* are supplied, entity extraction
    runs after chunking for each chunk that has not already been processed
    (determined via ``graph_store.get_processed_chunk_ids()``).

    Deletion events return an empty list; the caller is responsible for
    removing the corresponding chunks from storage by source path.
    """

    def __init__(
        self,
        parser: MarkdownParser,
        chunker: SentenceWindowChunker,
        entity_extractor: EntityExtractor | None = None,
        graph_store: GraphStore | None = None,
    ) -> None:
        self._parser = parser
        self._chunker = chunker
        self._entity_extractor = entity_extractor
        self._graph_store = graph_store

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        entity_extractor: EntityExtractor | None = None,
        graph_store: GraphStore | None = None,
    ) -> IngestionPipeline:
        """Construct an IngestionPipeline from a Settings object."""
        chunker = SentenceWindowChunker(
            chunk_size=settings.ingestion.chunking.chunk_size,
            chunk_overlap=settings.ingestion.chunking.chunk_overlap,
        )
        return cls(
            parser=MarkdownParser(),
            chunker=chunker,
            entity_extractor=entity_extractor,
            graph_store=graph_store,
        )

    def process_event(self, event: FileEvent) -> list[Chunk]:
        """Process a FileEvent into chunks.

        Returns an empty list for deletion events and for files that no
        longer exist at event-processing time (race condition on delete).
        """
        if event.event_type == "deleted":
            return []
        path = event.path
        if not path.exists():
            return []
        return self._ingest(path)

    def ingest_file(self, path: Path) -> list[Chunk]:
        """Directly ingest a single file without wrapping it in a FileEvent."""
        return self._ingest(path)

    def _ingest(self, path: Path) -> list[Chunk]:
        document = self._parser.parse(path)
        chunks = self._chunker.chunk(document)
        if self._entity_extractor is not None and self._graph_store is not None:
            self._run_extraction(chunks)
        return chunks

    def _run_extraction(self, chunks: list[Chunk]) -> None:
        """Extract entities from *chunks* that have not been processed yet."""
        processed_ids = self._graph_store.get_processed_chunk_ids()  # type: ignore[union-attr]
        new_chunks = [c for c in chunks if c.id not in processed_ids]
        if not new_chunks:
            return
        total = len(new_chunks)
        for idx, chunk in enumerate(new_chunks, start=1):
            print(f"    [graph {idx}/{total}] extracting entities from chunk {chunk.id}...", flush=True)
            try:
                entities, relationships = self._entity_extractor.extract(chunk)  # type: ignore[union-attr]
                if entities or relationships:
                    self._graph_store.upsert(entities, relationships)  # type: ignore[union-attr]
                self._graph_store.mark_chunk_processed(chunk.id)  # type: ignore[union-attr]
            except Exception:  # noqa: BLE001
                logger.warning(
                    "Entity extraction failed for chunk %s; skipping.", chunk.id, exc_info=True
                )
