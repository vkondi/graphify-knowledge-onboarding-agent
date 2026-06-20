"""HybridRetrieval: merge SemanticSearch and GraphRetriever results.

Conforms to the ``Retriever`` Protocol defined in ``interfaces.py``.
Because it satisfies the same interface, ``QueryEngine`` requires no changes
when hybrid retrieval is enabled.

Retrieval modes (sourced from ``settings.knowledge_graph.retrieval_mode``):

    vector  — delegate entirely to ``SemanticSearch``. Identical to the
              pre-Phase-6 behaviour; the graph is not consulted.
    graph   — delegate entirely to ``GraphRetriever``. Only graph-traversal
              results are returned.
    hybrid  — both retrievers run concurrently; results are merged and
              deduplicated.  When the same chunk appears in both result sets
              the score is blended::

                  merged = (1 - w) * vector_score + w * graph_score

              where ``w = settings.knowledge_graph.graph_weight``.
              Chunks unique to one retriever keep their original score.
              Final list is sorted by descending score and truncated to ``top_k``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from knowledge_onboarding_agent.models import RetrievedChunk

if TYPE_CHECKING:
    from knowledge_onboarding_agent.config import Settings
    from knowledge_onboarding_agent.interfaces import Retriever
    from knowledge_onboarding_agent.knowledge_graph.graph_retriever import GraphRetriever
    from knowledge_onboarding_agent.retrieval.semantic_search import SemanticSearch

_VALID_MODES = frozenset({"vector", "graph", "hybrid"})


class HybridRetrieval:
    """Merges ``SemanticSearch`` and ``GraphRetriever`` results.

    Parameters
    ----------
    semantic_search:
        The vector-similarity retriever.
    graph_retriever:
        The graph-traversal retriever.
    mode:
        One of ``"vector"``, ``"graph"``, or ``"hybrid"``.
    graph_weight:
        Blend weight for graph scores in hybrid mode (0.0–1.0).
        0.0 = full vector, 1.0 = full graph.
    top_k:
        Maximum results to return after merging.
    """

    def __init__(
        self,
        semantic_search: SemanticSearch,
        graph_retriever: GraphRetriever,
        mode: str = "hybrid",
        graph_weight: float = 0.3,
        top_k: int = 10,
    ) -> None:
        if mode not in _VALID_MODES:
            raise ValueError(f"retrieval mode must be one of {sorted(_VALID_MODES)}, got {mode!r}")
        if not 0.0 <= graph_weight <= 1.0:
            raise ValueError(f"graph_weight must be in [0.0, 1.0], got {graph_weight}")
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        self._semantic = semantic_search
        self._graph = graph_retriever
        self._mode = mode
        self._graph_weight = graph_weight
        self._top_k = top_k

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        semantic_search: SemanticSearch,
        graph_retriever: GraphRetriever,
    ) -> HybridRetrieval:
        """Construct a ``HybridRetrieval`` from a ``Settings`` object."""
        return cls(
            semantic_search=semantic_search,
            graph_retriever=graph_retriever,
            mode=settings.knowledge_graph.retrieval_mode,
            graph_weight=settings.knowledge_graph.graph_weight,
            top_k=settings.retrieval.top_k,
        )

    # ------------------------------------------------------------------
    # Retriever Protocol
    # ------------------------------------------------------------------

    def search(self, query: str) -> list[RetrievedChunk]:
        """Return the most relevant chunks for *query* using the configured mode.

        Returns an empty list when the query is blank.
        """
        if not query.strip():
            return []

        if self._mode == "vector":
            return self._semantic.search(query)

        if self._mode == "graph":
            return self._graph.search(query)

        # mode == "hybrid"
        return self._merge(
            vector_results=self._semantic.search(query),
            graph_results=self._graph.search(query),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _merge(
        self,
        vector_results: list[RetrievedChunk],
        graph_results: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """Merge and score vector and graph results.

        Chunks that appear in both result sets receive a blended score.
        Chunks unique to one set keep their original score.
        """
        w = self._graph_weight
        vector_by_id: dict[str, RetrievedChunk] = {r.chunk.id: r for r in vector_results}
        graph_by_id: dict[str, RetrievedChunk] = {r.chunk.id: r for r in graph_results}

        merged: dict[str, RetrievedChunk] = {}

        # Chunks in vector results
        for chunk_id, vr in vector_by_id.items():
            if chunk_id in graph_by_id:
                gr = graph_by_id[chunk_id]
                blended_score = (1.0 - w) * vr.score + w * gr.score
                merged[chunk_id] = RetrievedChunk(chunk=vr.chunk, score=blended_score)
            else:
                # Scale vector-only score by the vector weight so it is
                # comparable to blended scores when graph_weight > 0.
                merged[chunk_id] = RetrievedChunk(chunk=vr.chunk, score=(1.0 - w) * vr.score)

        # Chunks only in graph results
        for chunk_id, gr in graph_by_id.items():
            if chunk_id not in merged:
                merged[chunk_id] = RetrievedChunk(chunk=gr.chunk, score=w * gr.score)

        results = sorted(merged.values(), key=lambda r: r.score, reverse=True)
        return results[: self._top_k]
