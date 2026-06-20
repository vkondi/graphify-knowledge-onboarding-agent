# ADR-002 - Knowledge Graph Database Selection

**Date**: 2026-06-20
**Status**: Accepted
**Deciders**: Project architect
**Supersedes**: N/A

---

## Context

The Phase 6 enhancement introduces a knowledge graph layer alongside the existing ChromaDB vector store. The graph stores entities and relationships extracted from ingested markdown documents, enabling relationship-aware retrieval in addition to semantic similarity search.

A graph database must be selected that:
- Runs entirely locally (no cloud services, no separate server process)
- Fits within the existing 16GB RAM memory budget (currently ~10–12GB allocated)
- Supports dynamic node and edge attributes without a fixed schema
- Persists to disk and survives process restarts
- Is available on public PyPI
- Integrates cleanly with the existing Python 3.11+ codebase

---

## Decision Drivers

- **No additional server processes** — the system already runs Ollama as a daemon; adding a database server would increase operational complexity for a personal tool
- **Memory budget** — the personal knowledge base scale (hundreds of markdown files, ~25k entities at saturation) does not require a dedicated graph database engine
- **Schema flexibility** — entity types and relationship types are extracted dynamically by an LLM; the data model cannot be fully specified upfront
- **Transparency** — persistence should be inspectable without special tooling
- **Simplicity** — the system should remain operable by a single developer without database administration knowledge

---

## Options Considered

### Option A: NetworkX with JSON persistence

NetworkX is a pure-Python graph library. The graph is held in process memory as a `MultiDiGraph`. It is serialised to `graph.json` via `networkx.node_link_data` on every write.

**Pros:**
- Zero additional server processes; library-only
- Pure Python — no native extension, no build dependencies
- Fully schema-free: nodes and edges carry arbitrary attribute dictionaries
- Rich built-in graph algorithms (shortest path, connected components, centrality)
- JSON persistence is human-readable and inspectable with any text editor
- ~5–20MB RAM for personal knowledge base scale
- Well-known, stable library with comprehensive documentation

**Cons:**
- Full graph must fit in process memory (acceptable at this scale)
- JSON serialisation on every write is slower than write-ahead logging (acceptable at this write volume)
- Not suitable for graphs with millions of nodes

### Option B: Kuzu (embedded graph database)

Kuzu is an embedded, columnar graph database with a Cypher-like query language.

**Pros:**
- Disk-native storage; true incremental persistence
- Declarative query language
- Better performance for large graphs

**Cons:**
- Requires a fixed schema defined upfront (incompatible with dynamic entity extraction)
- Less familiar than NetworkX
- Younger project with a smaller ecosystem
- Binary on-disk format is not human-inspectable

### Option C: Neo4j

Neo4j is a production-grade graph database with a Java-based server.

**Pros:**
- Industry standard; mature ecosystem
- Excellent tooling and visualisation (Neo4j Browser)
- Cypher query language is expressive and readable

**Cons:**
- Requires a separate Java server process (violates the no-additional-server constraint)
- Heavy memory footprint (~500MB minimum)
- Requires installation outside pip
- Significant operational overhead for a personal tool

### Option D: SQLite with adjacency table

Model the graph as an adjacency list in a SQLite database (two tables: `nodes`, `edges`).

**Pros:**
- SQLite is in the Python standard library
- Familiar SQL query language

**Cons:**
- Graph traversal (multi-hop) requires recursive CTEs or multiple round-trips, reducing code clarity
- No built-in graph algorithms
- Accumulating attributes on a node across multiple ingestion runs is more complex than a dict-based approach

---

## Decision

**Selected: Option A — NetworkX with JSON persistence.**

NetworkX satisfies all constraints for the scale of this project. The full knowledge graph for hundreds of markdown documents fits comfortably in RAM, JSON persistence is transparent and portable, and the rich algorithm library supports future enhancements (learning paths, community detection). The schema-free model aligns directly with dynamic LLM-based entity extraction.

---

## Consequences

### Positive
- No new server processes or database daemons required
- Graph data is human-readable and version-controllable if desired
- Full traversal and algorithm access within a single Python process
- Graph file can be deleted and rebuilt from scratch via `koa reingest`

### Negative / Mitigations
- **In-memory only during runtime**: If the process crashes between write operations, the last write is lost. Mitigation: `GraphStore.save()` is called after every upsert, not batched.
- **Not suitable for very large graphs**: If the knowledge base grows to tens of thousands of documents, a migration to Kuzu may be warranted. This will be addressed in a future ADR.
- **No concurrent access**: NetworkX is not thread-safe for writes. The existing single-threaded ingestion pipeline already ensures writes are serialised.

---

## Dependency Addition

`networkx>=3.0` is added to `[project.dependencies]` in `pyproject.toml`.
