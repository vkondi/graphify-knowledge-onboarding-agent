# Knowledge Graph Architecture

> **Purpose**: Describe the knowledge graph layer introduced in Phase 6.
> This document is the primary reference for the entity extraction, graph storage,
> graph retrieval, and hybrid retrieval components.
> Read alongside `system-design.md` which covers the full pipeline.

---

## Overview

The knowledge graph layer augments the existing vector-based retrieval with a
relationship-aware retrieval path. It runs alongside ChromaDB and does not replace it.

When a document is ingested:
1. The existing vector pipeline (embedding → ChromaDB) runs unchanged.
2. An LLM extracts entities and relationships from each chunk.
3. Entities become graph nodes; relationships become directed edges.
4. Nodes and edges carry references back to their originating chunks.

At query time, the caller can choose one of three retrieval modes (controlled by
`knowledge_graph.retrieval_mode` in `settings.yaml`):

| Mode | Behaviour |
|---|---|
| `vector` | Existing `SemanticSearch` only. Identical to pre-Phase-6 behaviour. |
| `graph` | `GraphRetriever` only. Traverses the entity graph to find related chunks. |
| `hybrid` | Both retrievers run; results are merged and deduplicated. |

---

## Component Diagram

```
┌────────────────────────────────────────────────────────────────────────┐
│                         INGESTION PIPELINE                              │
│  MarkdownParser → SentenceWindowChunker → List[Chunk]                  │
└────────────────────────────────┬───────────────────────────────────────┘
                                 │
              ┌──────────────────┴──────────────────┐
              ▼                                       ▼
┌─────────────────────────┐           ┌──────────────────────────────────┐
│  EMBEDDINGS              │           │  ENTITY EXTRACTION               │
│  OllamaEmbedder          │           │  EntityExtractor                 │
│  ChunkEmbedder           │           │  (Ollama LLM, JSON output)       │
│                          │           │  Deduped by content_hash         │
│  → List[EmbeddedChunk]   │           │  → list[Entity], list[Relationship]│
└────────────┬────────────┘           └────────────────┬─────────────────┘
             │                                          │
             ▼                                          ▼
┌─────────────────────────┐           ┌──────────────────────────────────┐
│  CHROMADB STORE          │           │  GRAPH STORE                     │
│  Vectors + metadata      │           │  NetworkX MultiDiGraph           │
│  (unchanged)             │           │  Nodes = entities                │
│                          │           │  Edges = relationships           │
│                          │           │  Persists: graph.json            │
└────────────┬────────────┘           └────────────────┬─────────────────┘
             │                                          │
             └──────────────────┬───────────────────────┘
                                ▼
             ┌────────────────────────────────────────────┐
             │           HYBRID RETRIEVAL                  │
             │  mode=vector  →  SemanticSearch only        │
             │  mode=graph   →  GraphRetriever only        │
             │  mode=hybrid  →  both, merged               │
             │  Conforms to Retriever Protocol             │
             └────────────────────────────────────────────┘
                                │
                                ▼
             ┌────────────────────────────────────────────┐
             │           ORCHESTRATION                     │
             │  QueryEngine (unchanged)                    │
             └────────────────────────────────────────────┘
```

---

## Entity Extraction

### Strategy

Entity extraction uses the local Ollama LLM (`mistral` by default, same as the QA model).
A structured prompt requests a JSON object containing entities and relationships.
Temperature is set to `0.0` for deterministic, reproducible output.

### Extraction Prompt

```
Extract all meaningful entities and their relationships from the following text.
Return ONLY valid JSON with this structure:
{
  "entities": [{"name": "...", "type": "..."}],
  "relationships": [{"source": "...", "type": "...", "target": "..."}]
}

Entity type examples (not exhaustive): Technology, Framework, Library, Concept,
Project, Organization, Person, Tool, Language, Protocol, Algorithm, Pattern

Relationship type examples (not exhaustive): Uses, Implements, Extends,
DependsOn, RelatedTo, BelongsTo, IsA, Replaces, ComplementsWith, PartOf

Text:
{content}
```

### Deduplication

Extraction is tracked by `content_hash` (same mechanism as `ChunkEmbedder`).
`GraphStore` stores the set of processed chunk IDs. A chunk whose `content_hash`
is already in the store is skipped, enabling incremental indexing.

### Normalization

Entity names are stored in their original casing for display, but compared in
lowercase for deduplication. The node ID in the graph is the lowercase entity name.
Display name is preserved as a `display_name` node attribute.

---

## Graph Schema

### Node Attributes

Each node in the `MultiDiGraph` represents a unique entity (keyed by lowercase name).

| Attribute | Type | Description |
|---|---|---|
| `display_name` | `str` | Original casing of the entity name |
| `entity_type` | `str` | e.g. `"Technology"`, `"Concept"` |
| `source_paths` | `list[str]` | Source file paths where this entity appears |
| `chunk_ids` | `list[str]` | Chunk IDs (e.g. `"python-basics:3"`) referencing this entity |

### Edge Attributes

Each directed edge represents a relationship between two entity nodes.

| Attribute | Type | Description |
|---|---|---|
| `relationship_type` | `str` | e.g. `"Uses"`, `"DependsOn"` |
| `source_path` | `str` | Source file path where this relationship was observed |
| `chunk_id` | `str` | The chunk ID that evidenced this relationship |

### Persistence Format

The graph is serialized using `networkx.node_link_data()` to `<knowledge_graph.path>/graph.json`.
The file is re-written on every `upsert` call. A companion file
`<knowledge_graph.path>/processed_chunk_ids.json` stores the set of chunk IDs
for which extraction has been completed.

---

## Graph Retrieval

### Approach

1. **Query entity extraction**: A small LLM call extracts entity names from the query string.
2. **Node lookup**: Each extracted entity is matched against graph nodes using case-insensitive substring matching.
3. **Neighbourhood traversal**: For each matched node, immediate neighbours (depth-1) are included. If the initial node set is empty, depth-2 traversal is attempted.
4. **Chunk ID collection**: `chunk_ids` from all matched and neighbouring nodes are collected.
5. **Chunk reconstruction**: `ChromaDBStore.get_by_ids()` fetches stored metadata; `Chunk` objects are reconstructed identically to `SemanticSearch`.
6. **Scoring**: Graph-retrieved chunks are assigned a proximity score: `1.0` for direct entity match, `0.7` for 1-hop neighbour.

### Query Entity Extraction Prompt

```
Extract the key entities and concepts from the following question as a JSON array of strings.
Return ONLY a JSON array. Example: ["Python", "FastAPI", "REST"]

Question: {query}
```

---

## Hybrid Retrieval

`HybridRetrieval` implements the `Retriever` Protocol and is the component that
`QueryEngine` receives in hybrid and graph modes.

### Merge Strategy

When `mode=hybrid`:
1. Both `SemanticSearch` and `GraphRetriever` run independently.
2. Results are merged into a single list, deduplicated by `chunk.id`.
3. When the same chunk appears in both result sets, the score is:
   `merged_score = (1 - graph_weight) * vector_score + graph_weight * graph_score`
4. Chunks that appear only in one result set keep their original score.
5. Final list is sorted by descending score and truncated to `top_k`.

`graph_weight` is sourced from `knowledge_graph.graph_weight` in `settings.yaml` (default `0.3`).

---

## Setup Requirements

No new services are required. NetworkX is a pure-Python library added to
`[project.dependencies]`. The graph directory is created automatically on first use.

To rebuild the graph from scratch:
```bash
koa reingest          # clears ChromaDB AND graph store
koa graph-stats       # inspect current graph state
```

---

## Configuration Reference

```yaml
knowledge_graph:
  enabled: true
  path: ./.knowledge-onboarding-agent/graph
  retrieval_mode: hybrid          # vector | graph | hybrid
  graph_weight: 0.3               # hybrid blend weight (0.0 = vector only, 1.0 = graph only)
  extraction:
    max_entities_per_chunk: 10
    max_relationships_per_chunk: 10
    extraction_temperature: 0.0
```
