"""Unit tests for HybridRetrieval."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from knowledge_onboarding_agent.models import Chunk, RetrievedChunk
from knowledge_onboarding_agent.retrieval.hybrid_retrieval import HybridRetrieval


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_retrieved(chunk_id: str, score: float) -> RetrievedChunk:
    chunk = Chunk(
        id=chunk_id,
        source_path=Path("doc.md"),
        content=f"Content of {chunk_id}",
        chunk_index=0,
        metadata={},
        content_hash=chunk_id,
    )
    return RetrievedChunk(chunk=chunk, score=score)


def _make_retriever(results: list[RetrievedChunk]) -> MagicMock:
    m = MagicMock()
    m.search.return_value = results
    return m


# ---------------------------------------------------------------------------
# Construction / validation
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_invalid_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="retrieval mode"):
            HybridRetrieval(
                semantic_search=MagicMock(),
                graph_retriever=MagicMock(),
                mode="invalid",
            )

    def test_invalid_graph_weight_raises(self) -> None:
        with pytest.raises(ValueError, match="graph_weight"):
            HybridRetrieval(
                semantic_search=MagicMock(),
                graph_retriever=MagicMock(),
                graph_weight=1.5,
            )

    def test_invalid_top_k_raises(self) -> None:
        with pytest.raises(ValueError, match="top_k"):
            HybridRetrieval(
                semantic_search=MagicMock(),
                graph_retriever=MagicMock(),
                top_k=0,
            )


# ---------------------------------------------------------------------------
# Vector-only mode
# ---------------------------------------------------------------------------

class TestVectorMode:
    def test_delegates_to_semantic_search(self) -> None:
        expected = [_make_retrieved("a", 0.9)]
        semantic = _make_retriever(expected)
        graph = _make_retriever([])
        hr = HybridRetrieval(semantic, graph, mode="vector")
        result = hr.search("test query")
        semantic.search.assert_called_once_with("test query")
        graph.search.assert_not_called()
        assert result == expected

    def test_blank_query_returns_empty(self) -> None:
        hr = HybridRetrieval(_make_retriever([]), _make_retriever([]), mode="vector")
        assert hr.search("") == []


# ---------------------------------------------------------------------------
# Graph-only mode
# ---------------------------------------------------------------------------

class TestGraphMode:
    def test_delegates_to_graph_retriever(self) -> None:
        expected = [_make_retrieved("b", 0.8)]
        semantic = _make_retriever([])
        graph = _make_retriever(expected)
        hr = HybridRetrieval(semantic, graph, mode="graph")
        result = hr.search("test query")
        graph.search.assert_called_once_with("test query")
        semantic.search.assert_not_called()
        assert result == expected


# ---------------------------------------------------------------------------
# Hybrid mode — merging
# ---------------------------------------------------------------------------

class TestHybridMode:
    def test_unique_chunks_combined(self) -> None:
        v_results = [_make_retrieved("v1", 0.9)]
        g_results = [_make_retrieved("g1", 1.0)]
        hr = HybridRetrieval(
            _make_retriever(v_results),
            _make_retriever(g_results),
            mode="hybrid",
            graph_weight=0.3,
        )
        result = hr.search("query")
        ids = {r.chunk.id for r in result}
        assert "v1" in ids
        assert "g1" in ids

    def test_shared_chunk_gets_blended_score(self) -> None:
        # chunk "shared" appears in both with different scores
        v_results = [_make_retrieved("shared", 0.8)]
        g_results = [_make_retrieved("shared", 1.0)]
        w = 0.3
        hr = HybridRetrieval(
            _make_retriever(v_results),
            _make_retriever(g_results),
            mode="hybrid",
            graph_weight=w,
        )
        result = hr.search("query")
        assert len(result) == 1
        expected_score = (1 - w) * 0.8 + w * 1.0
        assert abs(result[0].score - expected_score) < 1e-9

    def test_vector_only_chunk_scaled_by_vector_weight(self) -> None:
        v_results = [_make_retrieved("v_only", 1.0)]
        w = 0.3
        hr = HybridRetrieval(
            _make_retriever(v_results),
            _make_retriever([]),
            mode="hybrid",
            graph_weight=w,
        )
        result = hr.search("query")
        assert len(result) == 1
        assert abs(result[0].score - (1 - w) * 1.0) < 1e-9

    def test_graph_only_chunk_scaled_by_graph_weight(self) -> None:
        g_results = [_make_retrieved("g_only", 1.0)]
        w = 0.3
        hr = HybridRetrieval(
            _make_retriever([]),
            _make_retriever(g_results),
            mode="hybrid",
            graph_weight=w,
        )
        result = hr.search("query")
        assert len(result) == 1
        assert abs(result[0].score - w * 1.0) < 1e-9

    def test_results_sorted_descending(self) -> None:
        v_results = [_make_retrieved("a", 0.5), _make_retrieved("b", 0.9)]
        hr = HybridRetrieval(
            _make_retriever(v_results),
            _make_retriever([]),
            mode="hybrid",
            graph_weight=0.0,
        )
        result = hr.search("query")
        scores = [r.score for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_truncates_results(self) -> None:
        v_results = [_make_retrieved(f"v{i}", float(i) / 10) for i in range(8)]
        g_results = [_make_retrieved(f"g{i}", float(i) / 10) for i in range(8)]
        hr = HybridRetrieval(
            _make_retriever(v_results),
            _make_retriever(g_results),
            mode="hybrid",
            top_k=5,
        )
        result = hr.search("query")
        assert len(result) <= 5

    def test_both_empty_returns_empty(self) -> None:
        hr = HybridRetrieval(
            _make_retriever([]),
            _make_retriever([]),
            mode="hybrid",
        )
        assert hr.search("query") == []

    def test_graph_weight_zero_equivalent_to_vector_mode(self) -> None:
        v_results = [_make_retrieved("a", 0.9)]
        g_results = [_make_retrieved("b", 1.0)]
        hr = HybridRetrieval(
            _make_retriever(v_results),
            _make_retriever(g_results),
            mode="hybrid",
            graph_weight=0.0,
        )
        result = hr.search("query")
        # graph-only chunk "b" gets score 0.0 * 1.0 = 0.0 and is still returned
        ids = {r.chunk.id for r in result if r.score > 0}
        assert "a" in ids
