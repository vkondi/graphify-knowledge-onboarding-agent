# Session Log

> Append a new entry at the end of each development session.
> Keep entries brief. Purpose: allow any future AI session to understand what happened last time.

---

## Format

```
## YYYY-MM-DD - [Brief Title]

**Session goal**: what we set out to do
**Completed**: what was actually finished
**Decisions made**: any architectural or implementation choices locked in
**Deferred**: what was intentionally left for next time
**Next session should start with**: specific first action
```

---

## 2026-05-18 - Project Bootstrap

**Session goal**: Establish AI-native project memory, documentation structure, and Copilot collaboration workflow. No implementation code.

**Completed**:
- Designed and documented repository folder structure
- Created `context/CONTEXT.md` (master AI memory file)
- Created `context/implementation-tracker.md` (progress tracker)
- Created `context/session-log.md` (this file)
- Created `docs/project-overview.md`
- Created `docs/architecture/system-design.md`
- Created `docs/constraints/runtime-constraints.md`
- Created `docs/roadmap/roadmap.md`
- Created `docs/decisions/ADR-template.md`
- Created `docs/decisions/ADR-001-model-selection.md` (draft)
- Created `docs/workflows/development-workflow.md`
- Created `.github/copilot-instructions.md`
- Created `.github/prompts/` reusable prompt templates

**Decisions made**:
- Python 3.11+, Ollama, LlamaIndex, ChromaDB confirmed as primary stack
- Architecture is a 5-stage pipeline (Ingestion → Embeddings → Storage → Retrieval → Orchestration)
- Each stage is a separate module with no direct cross-imports
- Protocols (not ABCs) for interfaces

**Deferred**:
- ADR-001 model selection needs finalization (benchmark needed)
- No implementation code written yet - by design

**Next session should start with**:
1. Load `.github/context/CONTEXT.md` and `.github/context/implementation-tracker.md`
2. Finalize ADR-001 (choose embedding model after reviewing Ollama model library)
3. Scaffold `pyproject.toml` and `src/knowledge_onboarding_agent/` package stubs (Phase 0 completion)

---

## 2026-05-19 - Phase 0 Completion + Phase 1 Ingestion Pipeline

**Session goal**: Complete Phase 0 scaffolding and implement the full ingestion pipeline.

**Completed**:
- Finalized ADR-001: `nomic-embed-text` (embedding) + `mistral` (LLM) - both confirmed on target hardware
- Created `pyproject.toml` (setuptools backend, dev extras, `koa` CLI entry point)
- Created `config/settings.yaml` with all defaults
- Scaffolded `src/knowledge_onboarding_agent/` package with stage sub-packages
- Defined all data models in `models.py`: `FileEvent`, `Section`, `ParsedDocument`, `Chunk`, `EmbeddedChunk`, `RetrievedChunk`, `Response`
- Implemented `FileWatcher` (`ingestion/watcher.py`) with Watchdog, `seed_from_existing`, create/modify/delete/rename handling
- Implemented `MarkdownParser` (`ingestion/parser.py`) with front matter, heading hierarchy, plain-text stripping
- Implemented `SentenceWindowChunker` (`ingestion/chunker.py`) and `ChunkingStrategy` Protocol
- Implemented `IngestionPipeline` (`ingestion/pipeline.py`) with `from_settings()`, `process_event()`, `ingest_file()`
- 69 ingestion tests passing (81 total unit tests)

**Decisions made**:
- `SentenceWindowChunker` chosen as default chunking strategy; fixed-size and recursive strategies deferred
- `mistune` used for markdown parsing

**Deferred**: Embedding pipeline (Phase 2)

**Next session should start with**: Phase 2 - implement `OllamaEmbedder` and `ChunkEmbedder`

---

## 2026-05-20 - Phase 2 Embedding Pipeline + Phase 3 Storage Layer

**Session goal**: Implement embedding pipeline and storage layer.

**Completed**:
- Implemented `OllamaEmbedder` (`embeddings/ollama_embedder.py`) - batched calls to `ollama.Client.embed`
- Implemented `ChunkEmbedder` (`embeddings/chunk_embedder.py`) - deduplication by `content_hash`, `known_hashes` set
- Implemented `EmbeddingProvider` Protocol in `interfaces.py`
- 25 unit tests for embedding components (4 integration tests marked)
- Implemented `ChromaDBStore` (`storage/chroma_store.py`) - `upsert`/`query`/`delete`/`count`, `get_stored_hashes`, `delete_by_source`
- Implemented `FAISSStore` (`storage/faiss_store.py`) - deferred import, raises `ImportError` with install hint
- Implemented `VectorStore` Protocol in `interfaces.py`
- 26 ChromaDB tests passing; FAISS tests skip gracefully when `faiss-cpu` not installed
- 156 total unit tests passing

**Decisions made**:
- ChromaDB confirmed as primary vector store; FAISS as optional fallback

**Next session should start with**: Phase 4 - implement `SemanticSearch`

---

## 2026-05-21 - Phase 4 Retrieval + Phase 5 Orchestration and CLI

**Session goal**: Implement retrieval and the full query interface with CLI.

**Completed**:
- Implemented `SemanticSearch` (`retrieval/semantic_search.py`) - `SemanticSearch(embedder, store, top_k)` + `from_settings`
- Added `Retriever` Protocol to `interfaces.py`
- 25 retrieval unit tests; 1 integration test
- Implemented `QueryEngine` (`orchestration/query_engine.py`) - `ask(question)`, `detect_conflicts(topic)`, `generate_learning_path(topic)`; uses `ollama` client directly
- Implemented CLI (`orchestration/__init__.py`) - sub-commands: `ingest`, `reingest`, `ask`, `conflicts`, `path`, `watch`
- Added getting-started docs: `configuration.md`, `indexing.md`, `querying.md`, `development.md`, `troubleshooting.md`
- 231 total unit tests passing; 1 skipped (FAISS); 8 deselected (integration)

**Decisions made**:
- Reranking and hybrid search deferred (`reranking_enabled: false` in config)
- Web UI deferred to future consideration

**Next session should start with**: Phase 6 - implement `TopicSynthesizer` and `koa summarize` command

---

## 2026-06-21 - Phase 6 Knowledge Graph Layer

**Session goal**: Enhance the existing RAG agent with a knowledge graph layer for relationship-aware retrieval, hybrid retrieval, and graph-backed query answering.

**Completed**:
- Wrote ADR-002 (`docs/decisions/ADR-002-knowledge-graph-selection.md`) — NetworkX selected over Kuzu/Neo4j/SQLite
- Wrote knowledge graph architecture document (`docs/architecture/knowledge-graph.md`) — entity schema, extraction strategy, traversal, hybrid retrieval, config reference
- Added new data models: `Entity`, `Relationship`, `GraphQueryResult` to `models.py`
- Added new Protocols: `EntityExtractor`, `GraphStore` to `interfaces.py`
- Added `GraphExtractionConfig`, `KnowledgeGraphConfig` to `config.py`; added `knowledge_graph:` section to `settings.yaml`
- Added `networkx>=3.0` to `pyproject.toml`
- Implemented `knowledge_graph/graph_store.py` — NetworkX `MultiDiGraph`, JSON persistence, node/edge upsert+dedup, `query_context()`, `delete_by_source()`, processed chunk ID tracking
- Implemented `knowledge_graph/entity_extractor.py` — Ollama LLM structured JSON extraction, robust response parsing (fences, commentary), content length guard
- Implemented `knowledge_graph/graph_retriever.py` — query entity extraction, graph traversal (direct + 1-hop), Chunk reconstruction from ChromaDB, proximity scoring (1.0 direct / 0.7 neighbour)
- Added `get_chunks_by_ids()` to `ChromaDBStore` for `GraphRetriever`
- Implemented `retrieval/hybrid_retrieval.py` — conforms to `Retriever` Protocol; `vector`/`graph`/`hybrid` modes; blended scoring with configurable weight; `top_k` truncation
- Extended `IngestionPipeline` — optional `entity_extractor` + `graph_store` params; incremental extraction via processed chunk IDs; failures logged, never raised
- Wired knowledge graph into `orchestration/__init__.py` — `_build_ingester` creates graph components when enabled; `_build_engine` routes through `HybridRetrieval`; `reingest` resets graph; `watch` handles graph deletion on file events; `koa graph-stats` command
- Wrote 109 new unit tests across: `tests/knowledge_graph/test_graph_store.py`, `test_entity_extractor.py`, `test_graph_retriever.py`; `tests/retrieval/test_hybrid_retrieval.py`; extended `test_pipeline.py` and `test_chroma_store.py`
- Updated `system-design.md`, `CONTEXT.md`, `implementation-tracker.md`, `session-log.md`
- **302 unit tests passing, 0 failures** (pre-existing `test_parse_simple_fixture` failure unrelated to Phase 6)

**Decisions made**:
- NetworkX for graph store (see ADR-002): no server, pure Python, schema-free, transparent JSON, rich algorithms
- Entity extraction uses `mistral` (same model as QA) at `temperature=0.0` — no additional model required
- Hybrid blend weight default `0.3` (graph contributes 30%, vector 70%) — tunable via `knowledge_graph.graph_weight`
- `HybridRetrieval` satisfies the existing `Retriever` Protocol — `QueryEngine` required zero changes
- Graphify (`graphifyy` on PyPI) was evaluated and rejected: it targets codebases via Tree-sitter AST analysis and requires cloud LLM APIs, both incompatible with this project's constraints

**Deferred**:
- Integration tests against running Ollama (marked `@pytest.mark.integration`)
- Graph visualisation / export to HTML
- Multi-hop traversal depth > 1 as a configurable parameter

**Next session should start with**:
1. Load `CONTEXT.md` and `implementation-tracker.md`
2. Run `koa reingest` against `sample-knowledge/` to build a live graph
3. Run `koa graph-stats` to inspect the populated graph
4. Run `koa ask "What frameworks use Python?"` to exercise hybrid retrieval end-to-end
