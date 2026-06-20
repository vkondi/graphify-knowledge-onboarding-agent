"""Unit tests for EntityExtractor."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from knowledge_onboarding_agent.knowledge_graph.entity_extractor import EntityExtractor
from knowledge_onboarding_agent.models import Chunk


def _make_chunk(
    content: str = "FastAPI is a modern Python web framework for building REST APIs with automatic docs.",
    chunk_id: str = "doc:0",
    source: str = "doc.md",
) -> Chunk:
    return Chunk(
        id=chunk_id,
        source_path=Path(source),
        content=content,
        chunk_index=0,
        metadata={},
        content_hash="abc123",
    )


def _mock_client(response_json: dict) -> MagicMock:
    """Return a mock ollama.Client that returns the given JSON as message content."""
    client = MagicMock()
    client.chat.return_value = {
        "message": {"content": json.dumps(response_json)}
    }
    return client


def _extractor(client: MagicMock) -> EntityExtractor:
    return EntityExtractor(
        llm_model="mistral",
        llm_base_url="http://localhost:11434",
        _llm_client=client,
    )


# ---------------------------------------------------------------------------
# Successful extraction
# ---------------------------------------------------------------------------

class TestSuccessfulExtraction:
    def test_returns_entities_from_llm_json(self) -> None:
        client = _mock_client({
            "entities": [{"name": "FastAPI", "type": "Framework"}],
            "relationships": [],
        })
        extractor = _extractor(client)
        entities, relationships = extractor.extract(_make_chunk())
        assert len(entities) == 1
        assert entities[0].name == "FastAPI"
        assert entities[0].entity_type == "Framework"

    def test_returns_relationships_from_llm_json(self) -> None:
        client = _mock_client({
            "entities": [
                {"name": "FastAPI", "type": "Framework"},
                {"name": "Python", "type": "Language"},
            ],
            "relationships": [
                {"source": "FastAPI", "type": "Uses", "target": "Python"}
            ],
        })
        extractor = _extractor(client)
        entities, relationships = extractor.extract(_make_chunk())
        assert len(relationships) == 1
        rel = relationships[0]
        assert rel.source_entity == "FastAPI"
        assert rel.target_entity == "Python"
        assert rel.relationship_type == "Uses"

    def test_entity_source_path_matches_chunk(self) -> None:
        client = _mock_client({
            "entities": [{"name": "FastAPI", "type": "Framework"}],
            "relationships": [],
        })
        extractor = _extractor(client)
        chunk = _make_chunk(source="notes/api.md")
        entities, _ = extractor.extract(chunk)
        assert entities[0].source_path == Path("notes/api.md")

    def test_entity_chunk_id_matches_chunk(self) -> None:
        client = _mock_client({
            "entities": [{"name": "FastAPI", "type": "Framework"}],
            "relationships": [],
        })
        extractor = _extractor(client)
        chunk = _make_chunk(chunk_id="api:3")
        entities, _ = extractor.extract(chunk)
        assert entities[0].chunk_id == "api:3"

    def test_empty_entities_list_is_valid(self) -> None:
        client = _mock_client({"entities": [], "relationships": []})
        extractor = _extractor(client)
        entities, relationships = extractor.extract(_make_chunk())
        assert entities == []
        assert relationships == []


# ---------------------------------------------------------------------------
# Deduplication and limits
# ---------------------------------------------------------------------------

class TestLimits:
    def test_max_entities_respected(self) -> None:
        many = [{"name": f"Entity{i}", "type": "Concept"} for i in range(20)]
        client = _mock_client({"entities": many, "relationships": []})
        extractor = EntityExtractor(
            llm_model="mistral",
            llm_base_url="http://localhost:11434",
            max_entities=5,
            _llm_client=client,
        )
        entities, _ = extractor.extract(_make_chunk())
        assert len(entities) <= 5

    def test_max_relationships_respected(self) -> None:
        entities = [{"name": f"E{i}", "type": "Concept"} for i in range(5)]
        rels = [{"source": f"E{i}", "type": "RelatedTo", "target": f"E{i+1}"} for i in range(4)]
        client = _mock_client({"entities": entities, "relationships": rels})
        extractor = EntityExtractor(
            llm_model="mistral",
            llm_base_url="http://localhost:11434",
            max_relationships=2,
            _llm_client=client,
        )
        _, relationships = extractor.extract(_make_chunk())
        assert len(relationships) <= 2

    def test_self_referencing_relationship_skipped(self) -> None:
        client = _mock_client({
            "entities": [{"name": "Python", "type": "Language"}],
            "relationships": [{"source": "Python", "type": "IsA", "target": "Python"}],
        })
        extractor = _extractor(client)
        _, relationships = extractor.extract(_make_chunk())
        assert relationships == []


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_short_content_returns_empty(self) -> None:
        client = _mock_client({"entities": [], "relationships": []})
        extractor = _extractor(client)
        chunk = _make_chunk(content="Too short here.")  # 3 words < 10
        entities, relationships = extractor.extract(chunk)
        assert entities == []
        assert relationships == []
        # LLM should not be called for very short content
        client.chat.assert_not_called()

    def test_llm_failure_returns_empty(self) -> None:
        client = MagicMock()
        client.chat.side_effect = RuntimeError("Ollama is down")
        extractor = _extractor(client)
        entities, relationships = extractor.extract(_make_chunk())
        assert entities == []
        assert relationships == []

    def test_invalid_json_response_returns_empty(self) -> None:
        client = MagicMock()
        client.chat.return_value = {"message": {"content": "not json at all"}}
        extractor = _extractor(client)
        entities, relationships = extractor.extract(_make_chunk())
        assert entities == []
        assert relationships == []

    def test_json_in_markdown_fence_is_parsed(self) -> None:
        response = "```json\n" + json.dumps({
            "entities": [{"name": "Docker", "type": "Tool"}],
            "relationships": [],
        }) + "\n```"
        client = MagicMock()
        client.chat.return_value = {"message": {"content": response}}
        extractor = _extractor(client)
        entities, _ = extractor.extract(_make_chunk())
        assert len(entities) == 1
        assert entities[0].name == "Docker"

    def test_json_with_commentary_is_parsed(self) -> None:
        payload = json.dumps({
            "entities": [{"name": "Kubernetes", "type": "Tool"}],
            "relationships": [],
        })
        response = f"Sure, here is the JSON:\n{payload}\nEnd."
        client = MagicMock()
        client.chat.return_value = {"message": {"content": response}}
        extractor = _extractor(client)
        entities, _ = extractor.extract(_make_chunk())
        assert len(entities) == 1

    def test_missing_name_field_skipped(self) -> None:
        client = _mock_client({
            "entities": [{"type": "Framework"}],  # no "name"
            "relationships": [],
        })
        extractor = _extractor(client)
        entities, _ = extractor.extract(_make_chunk())
        assert entities == []


# ---------------------------------------------------------------------------
# Integration test (requires running Ollama)
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_extract_real_ollama() -> None:
    """Verify EntityExtractor works against a real Ollama instance."""
    from knowledge_onboarding_agent.config import load_settings
    settings = load_settings()
    extractor = EntityExtractor.from_settings(settings)
    chunk = _make_chunk(
        content=(
            "Docker is a containerization platform. "
            "Kubernetes orchestrates Docker containers and manages their lifecycle. "
            "Docker Compose is used for local multi-container development."
        ),
        chunk_id="test:0",
    )
    entities, relationships = extractor.extract(chunk)
    assert isinstance(entities, list)
    assert isinstance(relationships, list)
