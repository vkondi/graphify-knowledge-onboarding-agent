# Implementation Tracker

> Updated at the end of each meaningful work session.
> This is the authoritative record of what is built, what is in progress, and what is next.

---

## Legend

- `[x]` Completed
- `[-]` In progress
- `[ ]` Not started
- `[~]` Blocked / deferred

---

## Phase 0 - Architecture and Scaffolding

**Goal**: Establish repository structure, AI collaboration workflow, and documentation before writing any implementation code.

| Task | Status | Notes |
|---|---|---|
| Design repository folder structure | `[x]` | Documented in README |
| Create AI context memory files | `[x]` | `context/CONTEXT.md`, `session-log.md`, this file |
| Write project documentation templates | `[x]` | See `docs/` |
| Write Copilot instructions | `[x]` | `.github/copilot-instructions.md` |
| Create ADR-001 (model selection) | `[-]` | Draft created, needs finalization |
| Create `pyproject.toml` | `[x]` | `setuptools.build_meta` backend |
| Create `requirements.txt` (dev + runtime) | `[x]` | Managed via `pyproject.toml` extras |
| Scaffold `src/knowledge_onboarding_agent/` package structure | `[x]` | All 5 stage stubs + `interfaces.py` + `config.py` |
| Create `tests/` mirror structure | `[x]` | Mirror dirs + `conftest.py` + `test_config.py` (12 tests) |
| Create `config/settings.yaml` | `[x]` | Fully commented with all defaults |
| Verify Ollama install and target models run locally | `[x]` | `nomic-embed-text` + `mistral` confirmed present |

---

## Phase 1 - Ingestion Pipeline

**Goal**: Implement the file watching → parsing → chunking pipeline. Output: structured document chunks ready for embedding.

| Task | Status | Notes |
|---|---|---|
| Define `Document` and `Chunk` data models | `[x]` | `models.py` - `FileEvent`, `Section`, `ParsedDocument`, `Chunk`, `EmbeddedChunk`, `RetrievedChunk` |
| Implement `FileWatcher` (Watchdog wrapper) | `[x]` | `ingestion/watcher.py` - handles create/modify/delete/rename, seed_from_existing |
| Implement `MarkdownParser` | `[x]` | `ingestion/parser.py` - front matter, headings, plain-text stripping |
| Implement `ChunkingStrategy` Protocol + default impl | `[x]` | `SentenceWindowChunker` in `ingestion/chunker.py`; Protocol in `interfaces.py` |
| Wire ingestion pipeline end-to-end | `[x]` | `ingestion/pipeline.py` - `IngestionPipeline.from_settings()`, `process_event()`, `ingest_file()` |
| Write tests for all ingestion components | `[x]` | 69 ingestion tests passing (81 total) - parser, chunker, watcher, pipeline |

---

## Phase 2 - Embedding Pipeline

**Goal**: Take chunks from ingestion and produce embeddings using a local Ollama model.

| Task | Status | Notes |
|---|---|---|
| Implement `EmbeddingProvider` Protocol | `[x]` | Defined in `interfaces.py` (Phase 1) |
| Implement `OllamaEmbedder` | `[x]` | `embeddings/ollama_embedder.py` - batched calls to `ollama.Client.embed` |
| Implement batch embedding with deduplication | `[x]` | `embeddings/chunk_embedder.py` - `ChunkEmbedder` deduplicates by `content_hash` |
| Add checksum/hash tracking for change detection | `[x]` | `known_hashes` set on `ChunkEmbedder`; seeded from storage in Phase 3 |
| Write tests | `[x]` | 25 unit tests in `test_ollama_embedder.py` + `test_chunk_embedder.py` (4 integration tests marked) |

---

## Phase 3 - Storage Layer

**Goal**: Persist embeddings and document metadata in a local vector store.

| Task | Status | Notes |
|---|---|---|
| Define `VectorStore` Protocol | `[x]` | Defined in `interfaces.py` (Phase 1) |
| Implement `ChromaDBStore` | `[x]` | `storage/chroma_store.py` - upsert/query/delete/count + `upsert_embedded_chunks`, `get_stored_hashes`, `delete_by_source` |
| Implement `FAISSStore` | `[x]` | `storage/faiss_store.py` - deferred import, raises `ImportError` with install hint if `faiss-cpu` absent |
| Implement metadata storage (doc source, timestamps) | `[x]` | `upsert_embedded_chunks` enriches metadata with `content_hash`, `content`, `source_path`, `chunk_index` |
| Write tests | `[x]` | 26 ChromaDB tests passing; FAISS tests skip gracefully when `faiss-cpu` not installed |

---

## Phase 4 - Retrieval

**Goal**: Query the vector store and return relevant chunks for a given question.

| Task | Status | Notes |
|---|---|---|
| Implement semantic search | `[x]` | `retrieval/semantic_search.py` - `SemanticSearch(embedder, store, top_k)` + `from_settings` |
| Implement reranking (optional) | `[~]` | Deferred - `reranking_enabled: false` in config; not needed for Phase 5 |
| Implement hybrid search (BM25 + semantic) | `[~]` | Deferred - optional enhancement |
| Write tests | `[x]` | 25 unit tests; 1 integration test (`@pytest.mark.integration`); `Retriever` Protocol added to `interfaces.py` |

---

## Phase 5 - Orchestration and Query Interface

**Goal**: Wrap retrieval + LLM into a usable query interface.

| Task | Status | Notes |
|---|---|---|
| Integrate LlamaIndex query engine | `[x]` | `orchestration/query_engine.py` - `QueryEngine(retriever, llm_model, ...)` + `from_settings`; uses `ollama` client directly (llama-index-llms-ollama extra not installed) |
| Implement conflict detection across sources | `[x]` | `QueryEngine.detect_conflicts(topic)` - LLM-based, temperature=0, requires ≥2 chunks |
| Implement learning path generation | `[x]` | `QueryEngine.generate_learning_path(topic)` - sorts by source path + chunk_index; no LLM call |
| Build CLI interface | `[x]` | `knowledge_onboarding_agent.orchestration:cli_entry`; sub-commands: `ingest`, `reingest`, `ask`, `conflicts`, `path`, `watch` |
| (Optional) Build simple web UI | `[~]` | Deferred |

---

## Phase 6 - Knowledge Graph Layer

**Goal**: Augment vector retrieval with a relationship-aware knowledge graph. Enable hybrid retrieval combining semantic search and graph traversal.

| Task | Status | Notes |
|---|---|---|
| Write ADR-002 (graph database selection) | `[x]` | `docs/decisions/ADR-002-knowledge-graph-selection.md` - NetworkX selected; see ADR for full rationale |
| Write knowledge graph architecture doc | `[x]` | `docs/architecture/knowledge-graph.md` - entity/edge schema, extraction strategy, retrieval modes |
| Add `Entity`, `Relationship`, `GraphQueryResult` data models | `[x]` | `models.py` - three new dataclasses |
| Add `EntityExtractor`, `GraphStore` Protocols | `[x]` | `interfaces.py` - two new Protocols with full method signatures |
| Add `KnowledgeGraphConfig` to `config.py` | `[x]` | `GraphExtractionConfig` + `KnowledgeGraphConfig` Pydantic models; added to `Settings` |
| Add `knowledge_graph:` section to `settings.yaml` | `[x]` | `enabled`, `path`, `retrieval_mode`, `graph_weight`, `extraction.*` |
| Add `networkx>=3.0` to `pyproject.toml` | `[x]` | Core dependency (no optional extra needed at this scale) |
| Implement `GraphStore` | `[x]` | `knowledge_graph/graph_store.py` - NetworkX `MultiDiGraph`, JSON persistence, node/edge upsert/dedup, delete_by_source, processed ID tracking |
| Implement `EntityExtractor` | `[x]` | `knowledge_graph/entity_extractor.py` - Ollama LLM call, structured JSON prompt, robust JSON parsing (fences, commentary), per-chunk dedup via processed IDs |
| Implement `GraphRetriever` | `[x]` | `knowledge_graph/graph_retriever.py` - query entity extraction, graph traversal, Chunk reconstruction from ChromaDB, proximity scoring |
| Add `get_chunks_by_ids()` to `ChromaDBStore` | `[x]` | `storage/chroma_store.py` - fetches Chunks by ID for `GraphRetriever` |
| Implement `HybridRetrieval` | `[x]` | `retrieval/hybrid_retrieval.py` - conforms to `Retriever` Protocol; `vector`/`graph`/`hybrid` modes; blended scoring |
| Extend `IngestionPipeline` with entity extraction | `[x]` | `ingestion/pipeline.py` - optional `entity_extractor` + `graph_store` params; skips already-processed chunks; failures are logged, not raised |
| Wire knowledge graph into CLI + build helpers | `[x]` | `orchestration/__init__.py` - `_build_ingester`/`_build_engine` create graph components when `enabled`; `reingest` resets graph; `watch` deletes graph entries on file delete; `koa graph-stats` sub-command |
| Write unit tests | `[x]` | `tests/knowledge_graph/test_graph_store.py` (40 tests), `test_entity_extractor.py` (18 tests), `test_graph_retriever.py` (20 tests); `tests/retrieval/test_hybrid_retrieval.py` (16 tests); extended `test_pipeline.py` (+8 tests); extended `test_chroma_store.py` (+7 tests) |
| Update all documentation | `[x]` | `system-design.md`, `knowledge-graph.md`, `CONTEXT.md`, `implementation-tracker.md`, `session-log.md` |

---

## Decisions Made

| Decision | Resolution |
|---|---|
| Embedding model selection (ADR-001) | `nomic-embed-text` - see [ADR-001](../../docs/decisions/ADR-001-model-selection.md) (Accepted) |
| ChromaDB vs FAISS as primary store | ChromaDB primary; FAISS optional fallback |
| Chunking strategy (size, overlap, method) | `SentenceWindowChunker` - 512 words, 64 overlap |
| LLM model selection for Ollama | `mistral` (7B Q4) |
| Knowledge graph database (ADR-002) | NetworkX + JSON persistence - see [ADR-002](../../docs/decisions/ADR-002-knowledge-graph-selection.md) (Accepted) |

---

## Known Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Embedding model too large for 16GB RAM | High | Benchmark `nomic-embed-text` (274MB) first |
| Re-embedding cost on large corpora | Medium | Content hash deduplication in Phase 2 |
| ChromaDB performance at scale | Low-medium | FAISS fallback, benchmark at 10k docs |
| LlamaIndex API changes | Low | Pin version, abstract behind wrapper |
| Knowledge graph grows very large (>100k nodes) | Low | At current scale (~25k nodes for 500 docs), NetworkX + JSON is fine. Migrate to Kuzu if needed (new ADR required). |
| LLM entity extraction quality | Medium | Temperature=0, structured JSON prompt, graceful fallback (empty lists on failure) |
