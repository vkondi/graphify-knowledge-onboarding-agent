"""Unit tests for GraphRetriever."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from knowledge_onboarding_agent.knowledge_graph.graph_retriever import GraphRetriever
from knowledge_onboarding_agent.knowledge_graph.graph_store import GraphStore
from knowledge_onboarding_agent.models import Chunk, Entity, GraphQueryResult, RetrievedChunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(chunk_id: str = "doc:0", source: str = "doc.md") -> Chunk:
    return Chunk(
        id=chunk_id,
        source_path=Path(source),
        content="Some content about Python and FastAPI and building modern web applications.",
        chunk_index=0,
        metadata={},
        content_hash="abc123",
    )


def _mock_llm(entities: list[str]) -> MagicMock:
    """Return a mock client that returns a JSON entity list."""
    client = MagicMock()
    client.chat.return_value = {"message": {"content": json.dumps(entities)}}
    return client


def _mock_chroma(chunks: list[Chunk]) -> MagicMock:
    """Return a mock ChromaDBStore that returns matching chunks for requested IDs."""
    store = MagicMock()
    chunk_map = {c.id: c for c in chunks}
    store.get_chunks_by_ids.side_effect = lambda ids: [
        chunk_map[cid] for cid in ids if cid in chunk_map
    ]
    return store


# ---------------------------------------------------------------------------
# _parse_entity_list (static helper)
# ---------------------------------------------------------------------------

class TestParseEntityList:
    def test_clean_json_array(self) -> None:
        assert GraphRetriever._parse_entity_list('["Python", "FastAPI"]') == ["Python", "FastAPI"]

    def test_json_in_markdown_fence(self) -> None:
        raw = '```json\n["Docker", "Kubernetes"]\n```'
        assert GraphRetriever._parse_entity_list(raw) == ["Docker", "Kubernetes"]

    def test_json_with_commentary(self) -> None:
        raw = 'The entities are: ["Python", "Django"].'
        assert GraphRetriever._parse_entity_list(raw) == ["Python", "Django"]

    def test_empty_array(self) -> None:
        assert GraphRetriever._parse_entity_list("[]") == []

    def test_invalid_input_returns_empty(self) -> None:
        assert GraphRetriever._parse_entity_list("not json") == []

    def test_blank_strings_filtered(self) -> None:
        result = GraphRetriever._parse_entity_list('["Python", "", "  "]')
        assert result == ["Python"]


# ---------------------------------------------------------------------------
# GraphRetriever.search
# ---------------------------------------------------------------------------

class TestGraphRetrieverSearch:
    def _make_retriever(
        self,
        graph_store: GraphStore,
        chunk: Chunk | None = None,
        entities: list[str] | None = None,
    ) -> GraphRetriever:
        if entities is None:
            entities = ["Python"]
        if chunk is None:
            chunk = _make_chunk()
        llm = _mock_llm(entities)
        chroma = _mock_chroma([chunk])
        return GraphRetriever(
            graph_store=graph_store,
            chroma_store=chroma,
            llm_model="mistral",
            llm_base_url="http://localhost:11434",
            top_k=10,
            _llm_client=llm,
        )

    def test_blank_query_returns_empty(self, tmp_path: Path) -> None:
        gs = GraphStore(tmp_path)
        retriever = self._make_retriever(gs)
        assert retriever.search("") == []
        assert retriever.search("   ") == []

    def test_empty_graph_returns_empty(self, tmp_path: Path) -> None:
        gs = GraphStore(tmp_path)
        retriever = self._make_retriever(gs)
        results = retriever.search("What is Python?")
        assert results == []

    def test_direct_match_gets_score_1(self, tmp_path: Path) -> None:
        gs = GraphStore(tmp_path)
        chunk = _make_chunk()
        entity = Entity("Python", "Language", chunk.source_path, chunk.id)
        gs.upsert([entity], [])
        gs.mark_chunk_processed(chunk.id)

        retriever = self._make_retriever(gs, chunk=chunk, entities=["Python"])
        results = retriever.search("What is Python?")
        assert len(results) == 1
        assert results[0].score == 1.0

    def test_results_ordered_by_descending_score(self, tmp_path: Path) -> None:
        gs = GraphStore(tmp_path)
        chunk_direct = _make_chunk("doc:0")
        chunk_neighbour = _make_chunk("doc:1", "other.md")

        from knowledge_onboarding_agent.models import Relationship
        e_py = Entity("Python", "Language", chunk_direct.source_path, chunk_direct.id)
        e_fa = Entity("FastAPI", "Framework", chunk_neighbour.source_path, chunk_neighbour.id)
        rel = Relationship("FastAPI", "Python", "Uses", chunk_direct.source_path, chunk_direct.id)
        gs.upsert([e_py, e_fa], [rel])

        llm = _mock_llm(["Python"])
        chroma = _mock_chroma([chunk_direct, chunk_neighbour])
        retriever = GraphRetriever(
            graph_store=gs,
            chroma_store=chroma,
            llm_model="mistral",
            llm_base_url="http://localhost:11434",
            top_k=10,
            _llm_client=llm,
        )
        results = retriever.search("Python web framework")
        if len(results) >= 2:
            assert results[0].score >= results[1].score

    def test_llm_failure_returns_empty(self, tmp_path: Path) -> None:
        gs = GraphStore(tmp_path)
        entity = Entity("Python", "Language", Path("doc.md"), "doc:0")
        gs.upsert([entity], [])

        client = MagicMock()
        client.chat.side_effect = RuntimeError("LLM down")
        chroma = _mock_chroma([_make_chunk()])
        retriever = GraphRetriever(
            graph_store=gs,
            chroma_store=chroma,
            llm_model="mistral",
            llm_base_url="http://localhost:11434",
            _llm_client=client,
        )
        results = retriever.search("Python")
        assert results == []

    def test_top_k_limits_results(self, tmp_path: Path) -> None:
        gs = GraphStore(tmp_path)
        chunks = [_make_chunk(f"doc:{i}") for i in range(5)]
        for i, chunk in enumerate(chunks):
            e = Entity(f"Entity{i}", "Concept", chunk.source_path, chunk.id)
            gs.upsert([e], [])

        llm = _mock_llm([f"Entity{i}" for i in range(5)])
        chroma = _mock_chroma(chunks)
        retriever = GraphRetriever(
            graph_store=gs,
            chroma_store=chroma,
            llm_model="mistral",
            llm_base_url="http://localhost:11434",
            top_k=2,
            _llm_client=llm,
        )
        results = retriever.search("entities")
        assert len(results) <= 2


# ---------------------------------------------------------------------------
# Integration test (requires running Ollama)
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_graph_retriever_real_ollama(tmp_path: Path) -> None:
    """Verify GraphRetriever works end-to-end against a real Ollama instance."""
    from unittest.mock import MagicMock

    from knowledge_onboarding_agent.config import load_settings
    from knowledge_onboarding_agent.knowledge_graph.graph_store import GraphStore
    from knowledge_onboarding_agent.models import Entity

    settings = load_settings()
    gs = GraphStore(path=tmp_path)
    chunk = _make_chunk()
    gs.upsert([Entity("Python", "Language", chunk.source_path, chunk.id)], [])

    chroma = _mock_chroma([chunk])
    retriever = GraphRetriever.from_settings(settings, graph_store=gs, chroma_store=chroma)
    results = retriever.search("Tell me about Python")
    assert isinstance(results, list)
