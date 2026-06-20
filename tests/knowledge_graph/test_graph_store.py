"""Unit tests for GraphStore."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from knowledge_onboarding_agent.knowledge_graph.graph_store import GraphStore
from knowledge_onboarding_agent.models import Entity, Relationship


@pytest.fixture
def store(tmp_path: Path) -> GraphStore:
    return GraphStore(path=tmp_path)


@pytest.fixture
def entity_python(tmp_path: Path) -> Entity:
    return Entity(
        name="Python",
        entity_type="Language",
        source_path=tmp_path / "doc.md",
        chunk_id="doc:0",
    )


@pytest.fixture
def entity_fastapi(tmp_path: Path) -> Entity:
    return Entity(
        name="FastAPI",
        entity_type="Framework",
        source_path=tmp_path / "doc.md",
        chunk_id="doc:0",
    )


@pytest.fixture
def rel_uses(tmp_path: Path) -> Relationship:
    return Relationship(
        source_entity="FastAPI",
        target_entity="Python",
        relationship_type="Uses",
        source_path=tmp_path / "doc.md",
        chunk_id="doc:0",
    )


# ---------------------------------------------------------------------------
# Node insertion and retrieval
# ---------------------------------------------------------------------------

class TestUpsertNodes:
    def test_insert_single_entity_creates_node(
        self, store: GraphStore, entity_python: Entity
    ) -> None:
        store.upsert([entity_python], [])
        assert store.node_count() == 1

    def test_node_key_is_lowercased(
        self, store: GraphStore, entity_python: Entity
    ) -> None:
        store.upsert([entity_python], [])
        node = store.get_node("python")
        assert node is not None

    def test_node_display_name_preserves_case(
        self, store: GraphStore, entity_python: Entity
    ) -> None:
        store.upsert([entity_python], [])
        node = store.get_node("python")
        assert node["display_name"] == "Python"

    def test_node_entity_type_stored(
        self, store: GraphStore, entity_python: Entity
    ) -> None:
        store.upsert([entity_python], [])
        node = store.get_node("python")
        assert node["entity_type"] == "Language"

    def test_node_source_path_stored(
        self, store: GraphStore, entity_python: Entity
    ) -> None:
        store.upsert([entity_python], [])
        node = store.get_node("python")
        assert str(entity_python.source_path) in node["source_paths"]

    def test_node_chunk_id_stored(
        self, store: GraphStore, entity_python: Entity
    ) -> None:
        store.upsert([entity_python], [])
        node = store.get_node("python")
        assert "doc:0" in node["chunk_ids"]

    def test_duplicate_insert_does_not_duplicate_source_path(
        self, store: GraphStore, entity_python: Entity
    ) -> None:
        store.upsert([entity_python], [])
        store.upsert([entity_python], [])
        node = store.get_node("python")
        assert node["source_paths"].count(str(entity_python.source_path)) == 1

    def test_duplicate_insert_does_not_duplicate_chunk_id(
        self, store: GraphStore, entity_python: Entity
    ) -> None:
        store.upsert([entity_python], [])
        store.upsert([entity_python], [])
        node = store.get_node("python")
        assert node["chunk_ids"].count("doc:0") == 1

    def test_same_entity_from_different_source(self, store: GraphStore, tmp_path: Path) -> None:
        e1 = Entity("Python", "Language", tmp_path / "a.md", "a:0")
        e2 = Entity("Python", "Language", tmp_path / "b.md", "b:0")
        store.upsert([e1], [])
        store.upsert([e2], [])
        node = store.get_node("python")
        assert len(node["source_paths"]) == 2
        assert len(node["chunk_ids"]) == 2


# ---------------------------------------------------------------------------
# Edge insertion
# ---------------------------------------------------------------------------

class TestUpsertEdges:
    def test_insert_relationship_creates_edge(
        self,
        store: GraphStore,
        entity_python: Entity,
        entity_fastapi: Entity,
        rel_uses: Relationship,
    ) -> None:
        store.upsert([entity_python, entity_fastapi], [rel_uses])
        assert store.edge_count() == 1

    def test_edge_creates_missing_nodes(
        self, store: GraphStore, rel_uses: Relationship
    ) -> None:
        # No entities provided — nodes should be auto-created from the relationship
        store.upsert([], [rel_uses])
        assert store.node_count() == 2

    def test_duplicate_edge_not_inserted(
        self,
        store: GraphStore,
        entity_python: Entity,
        entity_fastapi: Entity,
        rel_uses: Relationship,
    ) -> None:
        store.upsert([entity_python, entity_fastapi], [rel_uses])
        store.upsert([entity_python, entity_fastapi], [rel_uses])
        assert store.edge_count() == 1


# ---------------------------------------------------------------------------
# Query context
# ---------------------------------------------------------------------------

class TestQueryContext:
    def test_empty_graph_returns_empty_result(self, store: GraphStore) -> None:
        result = store.query_context(["Python"])
        assert result.chunk_ids == []
        assert result.entity_names == []

    def test_matching_entity_returns_chunk_id(
        self, store: GraphStore, entity_python: Entity
    ) -> None:
        store.upsert([entity_python], [])
        result = store.query_context(["Python"])
        assert "doc:0" in result.chunk_ids

    def test_case_insensitive_match(
        self, store: GraphStore, entity_python: Entity
    ) -> None:
        store.upsert([entity_python], [])
        result = store.query_context(["python"])
        assert "doc:0" in result.chunk_ids

    def test_hops_includes_neighbour(
        self,
        store: GraphStore,
        entity_python: Entity,
        entity_fastapi: Entity,
        rel_uses: Relationship,
    ) -> None:
        store.upsert([entity_python, entity_fastapi], [rel_uses])
        # query for Python, hop to FastAPI
        result = store.query_context(["python"], hops=1)
        assert "doc:0" in result.chunk_ids

    def test_no_match_returns_empty(self, store: GraphStore, entity_python: Entity) -> None:
        store.upsert([entity_python], [])
        result = store.query_context(["Rust"])
        assert result.chunk_ids == []

    def test_multiple_entities_merged(
        self, store: GraphStore, entity_python: Entity, entity_fastapi: Entity, tmp_path: Path
    ) -> None:
        e_rust = Entity("Rust", "Language", tmp_path / "other.md", "other:0")
        store.upsert([entity_python, entity_fastapi, e_rust], [])
        result = store.query_context(["Python", "Rust"])
        assert len(result.entity_names) >= 2


# ---------------------------------------------------------------------------
# Delete by source
# ---------------------------------------------------------------------------

class TestDeleteBySource:
    def test_delete_removes_node_with_single_source(
        self, store: GraphStore, entity_python: Entity
    ) -> None:
        store.upsert([entity_python], [])
        store.delete_by_source(entity_python.source_path)
        assert store.node_count() == 0

    def test_delete_preserves_node_with_another_source(
        self, store: GraphStore, tmp_path: Path
    ) -> None:
        e1 = Entity("Python", "Language", tmp_path / "a.md", "a:0")
        e2 = Entity("Python", "Language", tmp_path / "b.md", "b:0")
        store.upsert([e1, e2], [])
        store.delete_by_source(tmp_path / "a.md")
        # Node should still exist because b.md still references it
        assert store.node_count() == 1

    def test_delete_removes_processed_ids_for_source(
        self, store: GraphStore, entity_python: Entity
    ) -> None:
        store.upsert([entity_python], [])
        store.mark_chunk_processed("doc:0")
        store.delete_by_source(entity_python.source_path)
        assert "doc:0" not in store.get_processed_chunk_ids()


# ---------------------------------------------------------------------------
# Processed chunk ID tracking
# ---------------------------------------------------------------------------

class TestProcessedChunkIds:
    def test_mark_and_get_processed(self, store: GraphStore) -> None:
        store.mark_chunk_processed("chunk:1")
        assert "chunk:1" in store.get_processed_chunk_ids()

    def test_initially_empty(self, store: GraphStore) -> None:
        assert store.get_processed_chunk_ids() == set()


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_graph_survives_reload(
        self, tmp_path: Path, entity_python: Entity
    ) -> None:
        s1 = GraphStore(path=tmp_path)
        s1.upsert([entity_python], [])

        s2 = GraphStore(path=tmp_path)
        assert s2.node_count() == 1
        assert s2.get_node("python") is not None

    def test_processed_ids_survive_reload(self, tmp_path: Path) -> None:
        s1 = GraphStore(path=tmp_path)
        s1.mark_chunk_processed("a:1")
        s1._save_processed_ids()

        s2 = GraphStore(path=tmp_path)
        assert "a:1" in s2.get_processed_chunk_ids()

    def test_corrupted_graph_file_loads_empty_graph(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "graph.json"
        bad_file.write_text("not valid json", encoding="utf-8")
        s = GraphStore(path=tmp_path)
        assert s.node_count() == 0
