# System Design - Graphify Knowledge Onboarding Agent

> **Purpose**: Describe the architecture of the system: components, data flow, interfaces, and design principles.
> This is the primary reference when working on any component. Keep it updated as the design evolves.

---

## Architectural Style

Graphify Knowledge Onboarding Agent is a **pipeline architecture** with six stages.

Each stage is:
- A separate Python module under `src/knowledge_onboarding_agent/`
- Connected to adjacent stages via well-defined data contracts (dataclasses/Pydantic models)
- Independently testable without other stages
- Replaceable without cascading changes (all inter-stage communication is through Protocols)

---

## High-Level Data Flow

```
┌─────────────────────────────────────────────────────────┐
│                    Watched Folders                       │
│         ~/notes/, ~/articles/, ~/docs/                  │
└───────────────────────┬─────────────────────────────────┘
                        │  filesystem events (create/modify/delete)
                        ▼
┌─────────────────────────────────────────────────────────┐
│                   1. INGESTION                          │
│  FileWatcher → MarkdownParser → ChunkingStrategy        │
│  Output: List[Chunk]                                    │
└──────────────┬────────────────────────┬─────────────────┘
               │  List[Chunk]            │  List[Chunk]
               ▼                         ▼
┌──────────────────────────┐  ┌──────────────────────────────┐
│     2. EMBEDDINGS        │  │  2b. ENTITY EXTRACTION       │
│  OllamaEmbedder          │  │  EntityExtractor (Ollama LLM) │
│  → List[EmbeddedChunk]   │  │  → entities + relationships   │
└──────────┬───────────────┘  └────────────┬─────────────────┘
           │                               │
           ▼                               ▼
┌──────────────────────────┐  ┌──────────────────────────────┐
│     3. VECTOR STORAGE    │  │  3b. GRAPH STORAGE            │
│  ChromaDB / FAISS        │  │  GraphStore (NetworkX + JSON) │
│  (unchanged)             │  │  Nodes: entities              │
│                          │  │  Edges: relationships         │
└──────────┬───────────────┘  └────────────┬─────────────────┘
           │                               │
           └──────────────┬────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│             4. RETRIEVAL (hybrid)                       │
│  mode=vector  → SemanticSearch only                     │
│  mode=graph   → GraphRetriever only                     │
│  mode=hybrid  → HybridRetrieval (merged + scored)       │
│  Output: List[RetrievedChunk] (with scores)             │
└───────────────────────┬─────────────────────────────────┘
                        │  List[RetrievedChunk] + query
                        ▼
┌─────────────────────────────────────────────────────────┐
│                  5. ORCHESTRATION                       │
│  QueryEngine + Ollama LLM (via ollama.Client)           │
│  Features: QA, conflict detection, learning paths       │
│  CLI: koa ask | ingest | reingest | watch |             │
│       conflicts | path | graph-stats                    │
│  Output: Response (answer + sources + metadata)         │
└─────────────────────────────────────────────────────────┘
```

> See `docs/architecture/knowledge-graph.md` for the detailed knowledge graph design.

---

## Component Specifications

### 1. Ingestion (`src/knowledge_onboarding_agent/ingestion/`)

**Responsibility**: Convert files on disk into structured, tokenizable chunks.

| Sub-component | Responsibility |
|---|---|
| `FileWatcher` | Monitors directories using Watchdog; emits `FileEvent` objects |
| `MarkdownParser` | Parses markdown into sections with headings, code blocks, metadata |
| `ChunkingStrategy` | Splits parsed content into `Chunk` objects with overlap |

**Key data model**:
```python
@dataclass
class Chunk:
    id: str             # stable identifier: "<source_stem>:<chunk_index>"
    source_path: Path   # absolute path to source file
    content: str        # raw text content
    chunk_index: int    # 0-based position within the source document
    metadata: dict      # heading context, word count, source file path, etc.
    content_hash: str   # SHA-256 hex digest of content for change detection
```

**Design note**: The `ChunkingStrategy` is a Protocol - multiple strategies can be swapped at config time (sentence-window, fixed-size, recursive).

---

### 2. Embeddings (`src/knowledge_onboarding_agent/embeddings/`)

**Responsibility**: Convert `Chunk` text into vector representations.

| Sub-component | Responsibility |
|---|---|
| `EmbeddingProvider` | Protocol defining `.embed(texts: list[str]) -> list[list[float]]` |
| `OllamaEmbedder` | Calls Ollama `/api/embeddings` endpoint |
| `ChunkEmbedder` | Batches chunks, skips already-embedded (by content hash) |

**Design note**: Embedding is the most expensive operation. The `ChunkEmbedder` must check the `content_hash` of each chunk against storage before calling Ollama. Only new or changed chunks are re-embedded.

---

### 3. Storage (`src/knowledge_onboarding_agent/storage/`)

**Responsibility**: Persist and retrieve embeddings with metadata.

| Sub-component | Responsibility |
|---|---|
| `VectorStore` | Protocol defining `upsert`, `query`, `delete`, `count` |
| `ChromaDBStore` | Primary implementation using ChromaDB |
| `FAISSStore` | Alternative implementation using FAISS |

**Design note**: Storage is the only component that has mutable persistent state. All other components are stateless. This makes the storage layer the one to invest in correctness and testing.

---

### 4. Retrieval (`src/knowledge_onboarding_agent/retrieval/`)

**Responsibility**: Given a query, return the most relevant chunks from storage.

| Sub-component | Responsibility |
|---|---|
| `SemanticSearch` | Embeds query, runs nearest-neighbor search; supports dynamic `top_k` scaling |
| `Reranker` | _(deferred)_ Cross-encoder reranking of top-k results |
| `HybridSearch` | _(deferred)_ BM25 + semantic fusion |

**Design note**: Retrieval is read-only. It never writes to storage. The `Reranker` is optional and disabled by default (adds latency).

---

### 5. Orchestration (`src/knowledge_onboarding_agent/orchestration/`)

**Responsibility**: Coordinate retrieval and LLM to produce responses.

| Sub-component | Responsibility |
|---|---|
| `QueryEngine` | Coordinates retrieval + Ollama LLM via `ollama.Client` directly; exposes `ask()`, `detect_conflicts()`, and `generate_learning_path()` methods |

**Design note**: This is the thinnest layer. It should not contain business logic - only coordination. Heavy logic belongs in retrieval or dedicated utilities. LLM calls are made directly via `ollama.Client`.

---

## Interface Contracts

All cross-stage communication uses these data models. These are the **only** contracts between stages.

```
FileEvent → [Ingestion] → Chunk → [Embeddings] → EmbeddedChunk → [Storage]
Query → [Retrieval] → RetrievedChunk → [Orchestration] → Response
```

Protocols defined in `src/knowledge_onboarding_agent/interfaces.py`:

| Protocol | Used by |
|---|---|
| `EmbeddingProvider` | `ChunkEmbedder`, `SemanticSearch` |
| `VectorStore` | `ChunkEmbedder`, `SemanticSearch` |
| `ChunkingStrategy` | `IngestionPipeline` |
| `Retriever` | `QueryEngine` |
| `EntityExtractor` | `IngestionPipeline` |
| `GraphStore` | `IngestionPipeline`, `GraphRetriever` |

Concrete implementations import interfaces; nothing imports concrete implementations across stages.

---

## Configuration

All runtime configuration lives in `config/settings.yaml`. Components read config at startup. No hardcoded values in source code.

Key config sections:
- `ingestion.watch_paths` - list of folders to monitor
- `ingestion.chunking.strategy` - chunking method name
- `embeddings.model` - Ollama model name
- `storage.backend` - `chromadb` or `faiss`
- `storage.path` - local persistence path
- `retrieval.top_k` - number of results to retrieve
- `llm.model` - Ollama LLM model name
- `knowledge_graph.enabled` - enable/disable the knowledge graph layer
- `knowledge_graph.path` - directory for graph persistence
- `knowledge_graph.retrieval_mode` - `vector`, `graph`, or `hybrid`
- `knowledge_graph.graph_weight` - blend weight for hybrid mode (0.0–1.0)

---

## Scalability Considerations

| Concern | Current approach | Upgrade path |
|---|---|---|
| Large corpora (>10k docs) | ChromaDB in-memory+persist | FAISS HNSW index, async embedding |
| Slow embedding | Synchronous batch | Async queue worker |
| Memory pressure | Small models (`nomic-embed-text`) | Quantized models, streaming |
| Multiple knowledge bases | Single ChromaDB collection | Named collections per workspace |

---

## What This Design Defers

The following are explicitly out of scope for the initial architecture and will require new ADRs when the time comes:

- Fine-tuning or LoRA adaptation
- Multi-agent coordination
- Real-time streaming responses
- Web UI / API layer
- Multi-user / access control

---

## Related Documents

- [Project Overview](../project-overview.md)
- [Runtime Constraints](../constraints/runtime-constraints.md)
- [ADR-001: Model Selection](../decisions/ADR-001-model-selection.md)
- [ADR-002: Knowledge Graph Database Selection](../decisions/ADR-002-knowledge-graph-selection.md)
- [Knowledge Graph Architecture](knowledge-graph.md)
- [Roadmap](../roadmap/roadmap.md)
