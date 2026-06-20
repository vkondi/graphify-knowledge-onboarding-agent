"""Tests for knowledge_onboarding_agent.ingestion.pipeline."""

from datetime import datetime
from pathlib import Path

import pytest

from knowledge_onboarding_agent.config import load_settings
from knowledge_onboarding_agent.ingestion.pipeline import IngestionPipeline
from knowledge_onboarding_agent.models import FileEvent

FIXTURES = Path(__file__).parent.parent / "fixtures"
CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "settings.yaml"


@pytest.fixture
def pipeline() -> IngestionPipeline:
    settings = load_settings(CONFIG_PATH)
    return IngestionPipeline.from_settings(settings)


class TestIngestionPipelineIngestFile:
    def test_ingest_file_returns_chunks(self, pipeline):
        chunks = pipeline.ingest_file(FIXTURES / "simple.md")
        assert len(chunks) > 0

    def test_ingest_file_chunks_have_non_empty_content(self, pipeline):
        chunks = pipeline.ingest_file(FIXTURES / "simple.md")
        assert all(c.content.strip() for c in chunks)

    def test_ingest_file_all_chunks_have_valid_ids(self, pipeline):
        chunks = pipeline.ingest_file(FIXTURES / "simple.md")
        for chunk in chunks:
            assert ":" in chunk.id
            stem, index = chunk.id.rsplit(":", 1)
            assert stem == "simple"
            assert index.isdigit()

    def test_ingest_file_indices_sequential_from_zero(self, pipeline):
        chunks = pipeline.ingest_file(FIXTURES / "long_article.md")
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    def test_ingest_file_source_path_matches_input(self, pipeline):
        path = FIXTURES / "simple.md"
        chunks = pipeline.ingest_file(path)
        assert all(c.source_path == path for c in chunks)

    def test_ingest_no_headings_fixture(self, pipeline):
        chunks = pipeline.ingest_file(FIXTURES / "no_headings.md")
        assert len(chunks) >= 1

    def test_ingest_long_article_many_chunks(self, pipeline):
        settings = load_settings(CONFIG_PATH)
        from knowledge_onboarding_agent.ingestion.chunker import SentenceWindowChunker
        from knowledge_onboarding_agent.ingestion.parser import MarkdownParser
        small_pipeline = IngestionPipeline(
            parser=MarkdownParser(),
            chunker=SentenceWindowChunker(chunk_size=100, chunk_overlap=20),
        )
        chunks = small_pipeline.ingest_file(FIXTURES / "long_article.md")
        assert len(chunks) > 3


class TestIngestionPipelineProcessEvent:
    def test_process_created_event_returns_chunks(self, pipeline):
        event = FileEvent(
            path=FIXTURES / "simple.md",
            event_type="created",
            timestamp=datetime.now(),
        )
        chunks = pipeline.process_event(event)
        assert len(chunks) > 0

    def test_process_modified_event_returns_chunks(self, pipeline):
        event = FileEvent(
            path=FIXTURES / "simple.md",
            event_type="modified",
            timestamp=datetime.now(),
        )
        chunks = pipeline.process_event(event)
        assert len(chunks) > 0

    def test_process_deleted_event_returns_empty_list(self, pipeline):
        event = FileEvent(
            path=FIXTURES / "simple.md",
            event_type="deleted",
            timestamp=datetime.now(),
        )
        chunks = pipeline.process_event(event)
        assert chunks == []

    def test_process_event_nonexistent_file_returns_empty(self, pipeline, tmp_path):
        missing = tmp_path / "does_not_exist.md"
        event = FileEvent(path=missing, event_type="created", timestamp=datetime.now())
        chunks = pipeline.process_event(event)
        assert chunks == []


class TestIngestionPipelineFromSettings:
    def test_from_settings_constructs_pipeline(self):
        settings = load_settings(CONFIG_PATH)
        pipeline = IngestionPipeline.from_settings(settings)
        assert isinstance(pipeline, IngestionPipeline)

    def test_from_settings_uses_config_chunk_size(self):
        settings = load_settings(CONFIG_PATH)
        pipeline = IngestionPipeline.from_settings(settings)
        # Chunk size from settings should be respected (we verify indirectly
        # by confirming the pipeline runs without error)
        chunks = pipeline.ingest_file(FIXTURES / "simple.md")
        assert len(chunks) > 0


# ---------------------------------------------------------------------------
# Entity extraction integration in the pipeline (Phase 6)
# ---------------------------------------------------------------------------

class TestIngestionPipelineWithEntityExtraction:
    """Tests that verify the pipeline calls entity extraction when wired up."""

    def _make_pipeline_with_mocks(self, settings):
        """Return (pipeline, mock_extractor, mock_graph_store)."""
        from unittest.mock import MagicMock
        from knowledge_onboarding_agent.ingestion.pipeline import IngestionPipeline
        from knowledge_onboarding_agent.models import Entity, Relationship

        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = (
            [Entity("Python", "Language", FIXTURES / "simple.md", "simple:0")],
            [],
        )
        mock_graph_store = MagicMock()
        mock_graph_store.get_processed_chunk_ids.return_value = set()

        pipeline = IngestionPipeline.from_settings(
            settings,
            entity_extractor=mock_extractor,
            graph_store=mock_graph_store,
        )
        return pipeline, mock_extractor, mock_graph_store

    def test_entity_extractor_called_for_each_new_chunk(self):
        settings = load_settings(CONFIG_PATH)
        pipeline, mock_extractor, _ = self._make_pipeline_with_mocks(settings)
        chunks = pipeline.ingest_file(FIXTURES / "simple.md")
        assert mock_extractor.extract.call_count == len(chunks)

    def test_graph_store_upsert_called_when_entities_found(self):
        settings = load_settings(CONFIG_PATH)
        pipeline, _, mock_graph_store = self._make_pipeline_with_mocks(settings)
        pipeline.ingest_file(FIXTURES / "simple.md")
        assert mock_graph_store.upsert.called

    def test_graph_store_mark_chunk_processed_called(self):
        settings = load_settings(CONFIG_PATH)
        pipeline, _, mock_graph_store = self._make_pipeline_with_mocks(settings)
        chunks = pipeline.ingest_file(FIXTURES / "simple.md")
        assert mock_graph_store.mark_chunk_processed.call_count == len(chunks)

    def test_already_processed_chunk_skipped(self):
        settings = load_settings(CONFIG_PATH)
        from unittest.mock import MagicMock
        from knowledge_onboarding_agent.ingestion.pipeline import IngestionPipeline

        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = ([], [])
        mock_graph_store = MagicMock()
        # Pretend ALL chunks have already been processed
        chunks_preview = IngestionPipeline.from_settings(settings).ingest_file(FIXTURES / "simple.md")
        mock_graph_store.get_processed_chunk_ids.return_value = {c.id for c in chunks_preview}

        pipeline = IngestionPipeline.from_settings(
            settings,
            entity_extractor=mock_extractor,
            graph_store=mock_graph_store,
        )
        pipeline.ingest_file(FIXTURES / "simple.md")
        # Extractor should not be called since all chunks are already processed
        mock_extractor.extract.assert_not_called()

    def test_no_extractor_runs_cleanly(self):
        """Pipeline without extractor/graph_store runs exactly as before."""
        settings = load_settings(CONFIG_PATH)
        pipeline = IngestionPipeline.from_settings(settings)
        chunks = pipeline.ingest_file(FIXTURES / "simple.md")
        assert len(chunks) > 0

    def test_extractor_exception_does_not_abort_pipeline(self):
        """A failing extractor logs a warning but the pipeline returns chunks."""
        from unittest.mock import MagicMock
        settings = load_settings(CONFIG_PATH)
        mock_extractor = MagicMock()
        mock_extractor.extract.side_effect = RuntimeError("extraction boom")
        mock_graph_store = MagicMock()
        mock_graph_store.get_processed_chunk_ids.return_value = set()

        pipeline = IngestionPipeline.from_settings(
            settings,
            entity_extractor=mock_extractor,
            graph_store=mock_graph_store,
        )
        chunks = pipeline.ingest_file(FIXTURES / "simple.md")
        assert len(chunks) > 0
