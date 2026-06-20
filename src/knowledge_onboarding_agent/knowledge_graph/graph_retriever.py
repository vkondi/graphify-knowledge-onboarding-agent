"""GraphRetriever: retrieves chunks by traversing the knowledge graph.

Conforms to the ``Retriever`` Protocol defined in ``interfaces.py``, so it
can be used as a drop-in replacement for (or alongside) ``SemanticSearch``
without any changes to ``QueryEngine``.

Retrieval flow:
    1. A small Ollama LLM call extracts entity names from the query string.
    2. ``GraphStore.query_context()`` traverses the graph starting from matched nodes.
    3. Matched chunk IDs are fetched from ``ChromaDBStore`` to reconstruct ``Chunk`` objects.
    4. Proximity-based scores are assigned: 1.0 for direct entity match, 0.7 for neighbours.
    5. Results are returned as ``List[RetrievedChunk]``, ordered by descending score.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import ollama

from knowledge_onboarding_agent.models import Chunk, RetrievedChunk

if TYPE_CHECKING:
    from knowledge_onboarding_agent.config import Settings
    from knowledge_onboarding_agent.knowledge_graph.graph_store import GraphStore
    from knowledge_onboarding_agent.storage.chroma_store import ChromaDBStore

logger = logging.getLogger(__name__)

# Scores assigned based on how closely the chunk relates to the query entities.
_SCORE_DIRECT_MATCH = 1.0   # entity found directly in the graph
_SCORE_NEIGHBOUR = 0.7      # entity is a 1-hop neighbour of a direct match

_QUERY_ENTITY_PROMPT = """\
Extract the key entities and concepts from the following question as a JSON array of strings.
Return ONLY a JSON array — no explanation, no markdown fences.
Example: ["Python", "FastAPI", "REST"]

Question: {query}"""


class GraphRetriever:
    """Retrieves chunks by traversing the knowledge graph.

    Parameters
    ----------
    graph_store:
        A ``GraphStore`` instance to traverse.
    chroma_store:
        A ``ChromaDBStore`` instance used to reconstruct ``Chunk`` objects from
        chunk IDs returned by the graph traversal.
    llm_model:
        Ollama model name for query entity extraction.
    llm_base_url:
        Base URL of the local Ollama daemon.
    top_k:
        Maximum number of results to return.
    _llm_client:
        Injectable ``ollama.Client``-compatible object for testing.
    """

    def __init__(
        self,
        graph_store: GraphStore,
        chroma_store: ChromaDBStore,
        llm_model: str,
        llm_base_url: str,
        top_k: int = 10,
        _llm_client: Any | None = None,
    ) -> None:
        self._graph_store = graph_store
        self._chroma_store = chroma_store
        self._model = llm_model
        self._top_k = top_k
        self._client: Any = _llm_client or ollama.Client(host=llm_base_url)

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        graph_store: GraphStore,
        chroma_store: ChromaDBStore,
        *,
        _llm_client: Any | None = None,
    ) -> GraphRetriever:
        """Construct a ``GraphRetriever`` from a ``Settings`` object."""
        return cls(
            graph_store=graph_store,
            chroma_store=chroma_store,
            llm_model=settings.llm.model,
            llm_base_url=settings.llm.ollama_base_url,
            top_k=settings.retrieval.top_k,
            _llm_client=_llm_client,
        )

    # ------------------------------------------------------------------
    # Retriever Protocol
    # ------------------------------------------------------------------

    def search(self, query: str) -> list[RetrievedChunk]:
        """Return chunks related to *query* by graph traversal.

        Returns an empty list when the graph is empty, the query is blank,
        or no matching entities are found.
        """
        if not query.strip():
            return []

        # Step 1: extract query entities via LLM
        query_entities = self._extract_query_entities(query)
        if not query_entities:
            return []

        # Step 2: traverse graph — direct matches first (hops=1)
        direct_result = self._graph_store.query_context(query_entities, hops=0)
        neighbour_result = self._graph_store.query_context(query_entities, hops=1)

        # Assign scores: direct match = 1.0, neighbour-only = 0.7
        direct_chunk_ids = set(direct_result.chunk_ids)
        neighbour_only_ids = [
            cid for cid in neighbour_result.chunk_ids if cid not in direct_chunk_ids
        ]

        scored: list[tuple[str, float]] = (
            [(cid, _SCORE_DIRECT_MATCH) for cid in direct_result.chunk_ids]
            + [(cid, _SCORE_NEIGHBOUR) for cid in neighbour_only_ids]
        )

        if not scored:
            return []

        # Truncate before fetching from ChromaDB
        scored = scored[: self._top_k]

        # Step 3: reconstruct Chunk objects from ChromaDB
        chunk_id_to_score = {cid: score for cid, score in scored}
        chunks = self._fetch_chunks(list(chunk_id_to_score.keys()))

        results = [
            RetrievedChunk(chunk=chunk, score=chunk_id_to_score[chunk.id])
            for chunk in chunks
        ]
        results.sort(key=lambda r: r.score, reverse=True)
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_query_entities(self, query: str) -> list[str]:
        """Call the LLM to extract entity names from *query*.

        Returns an empty list on failure (LLM error or unparseable output).
        """
        prompt = _QUERY_ENTITY_PROMPT.format(query=query)
        try:
            response = self._client.chat(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0},
            )
            raw: str = response["message"]["content"].strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("GraphRetriever: entity extraction LLM call failed: %s", exc)
            return []

        return self._parse_entity_list(raw)

    @staticmethod
    def _parse_entity_list(raw: str) -> list[str]:
        """Parse a JSON array of entity name strings from *raw*."""
        # Strip markdown fences if present
        fence_match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", raw, re.DOTALL)
        if fence_match:
            raw = fence_match.group(1)
        # Find the first [...] block
        array_match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not array_match:
            return []
        try:
            items = json.loads(array_match.group(0))
            return [str(item).strip() for item in items if str(item).strip()]
        except (json.JSONDecodeError, TypeError):
            return []

    def _fetch_chunks(self, chunk_ids: list[str]) -> list[Chunk]:
        """Fetch ``Chunk`` objects for *chunk_ids* from ChromaDB.

        Chunk IDs that no longer exist in the store are silently skipped
        (can happen if ChromaDB was wiped but the graph was not).
        """
        if not chunk_ids:
            return []
        try:
            return self._chroma_store.get_chunks_by_ids(chunk_ids)
        except Exception as exc:  # noqa: BLE001
            logger.warning("GraphRetriever: ChromaDB fetch failed: %s", exc)
            return []
