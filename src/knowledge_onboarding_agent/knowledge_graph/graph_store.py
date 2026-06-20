"""GraphStore: persist and query a knowledge graph of entities and relationships.

Backed by a NetworkX ``MultiDiGraph`` held in process memory and serialised to
``graph.json`` on every write (see ADR-002 for selection rationale).

Conforms to the ``GraphStore`` Protocol defined in ``interfaces.py``.

File layout on disk (rooted at ``settings.knowledge_graph.path``):
    graph.json                — NetworkX node-link serialisation of the full graph
    processed_chunk_ids.json  — JSON array of Chunk IDs already processed by
                                 EntityExtractor; used to skip unchanged chunks on
                                 incremental re-ingestion
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import networkx as nx

from knowledge_onboarding_agent.models import Entity, GraphQueryResult, Relationship

if TYPE_CHECKING:
    from knowledge_onboarding_agent.config import Settings

logger = logging.getLogger(__name__)

_GRAPH_FILENAME = "graph.json"
_PROCESSED_IDS_FILENAME = "processed_chunk_ids.json"


class GraphStore:
    """NetworkX-backed knowledge graph with JSON persistence.

    Nodes represent entities (keyed by lowercase name).
    Directed edges represent relationships between entities.

    Parameters
    ----------
    path:
        Directory where ``graph.json`` and ``processed_chunk_ids.json`` are stored.
        Created automatically if it does not exist.
    """

    def __init__(self, path: str | Path) -> None:
        self._dir = Path(path)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._graph_path = self._dir / _GRAPH_FILENAME
        self._ids_path = self._dir / _PROCESSED_IDS_FILENAME
        self._graph: nx.MultiDiGraph = self._load_graph()
        self._processed_ids: set[str] = self._load_processed_ids()

    @classmethod
    def from_settings(cls, settings: Settings) -> GraphStore:
        """Construct a ``GraphStore`` from a ``Settings`` object."""
        return cls(path=settings.knowledge_graph.path)

    # ------------------------------------------------------------------
    # GraphStore Protocol
    # ------------------------------------------------------------------

    def upsert(
        self,
        entities: list[Entity],
        relationships: list[Relationship],
    ) -> None:
        """Insert or update entities and relationships.

        Nodes are merged by lowercase name; duplicate source paths and
        chunk IDs are deduplicated. Edges are deduplicated by
        ``(source, target, relationship_type, chunk_id)``.
        Persists both ``graph.json`` and ``processed_chunk_ids.json``
        after every call.
        """
        for entity in entities:
            self._upsert_node(entity)
        for rel in relationships:
            self._upsert_edge(rel)
        self._save_graph()
        self._save_processed_ids()

    def query_context(
        self,
        entity_names: list[str],
        hops: int = 1,
    ) -> GraphQueryResult:
        """Traverse up to *hops* edges from matched nodes and return chunk references.

        Matching is case-insensitive. Substring matching is used so that
        "Python" matches nodes named "python 3", "python library", etc.

        Returns an empty ``GraphQueryResult`` when no nodes match.
        """
        seed_nodes = self._find_nodes(entity_names)
        if not seed_nodes and hops < 2:
            # Try a one-extra-hop fallback when no exact/substring match
            seed_nodes = self._find_nodes(entity_names, fuzzy=True)

        if not seed_nodes:
            return GraphQueryResult(entity_names=[], chunk_ids=[], source_paths=[])

        # Gather all nodes within `hops` of the seed set
        reachable: set[str] = set(seed_nodes)
        frontier = set(seed_nodes)
        for _ in range(hops):
            next_frontier: set[str] = set()
            for node in frontier:
                next_frontier.update(self._graph.successors(node))
                next_frontier.update(self._graph.predecessors(node))
            reachable.update(next_frontier)
            frontier = next_frontier - reachable  # avoid revisiting

        chunk_ids: list[str] = []
        source_paths: list[Path] = []
        seen_chunk_ids: set[str] = set()
        seen_paths: set[str] = set()

        for node_id in reachable:
            attrs = self._graph.nodes[node_id]
            for cid in attrs.get("chunk_ids", []):
                if cid not in seen_chunk_ids:
                    chunk_ids.append(cid)
                    seen_chunk_ids.add(cid)
            for sp in attrs.get("source_paths", []):
                if sp not in seen_paths:
                    source_paths.append(Path(sp))
                    seen_paths.add(sp)

        return GraphQueryResult(
            entity_names=list(reachable),
            chunk_ids=chunk_ids,
            source_paths=source_paths,
        )

    def delete_by_source(self, source_path: Path) -> None:
        """Remove all contributions from *source_path* in the graph.

        Nodes that only came from this source are removed entirely.
        Nodes that also appear in other sources have *source_path*
        stripped from their attribute lists.
        Edges whose ``source_path`` matches are removed.
        """
        sp_str = str(source_path)

        # Remove matching edges
        edges_to_remove = [
            (u, v, k)
            for u, v, k, attrs in self._graph.edges(keys=True, data=True)
            if attrs.get("source_path") == sp_str
        ]
        self._graph.remove_edges_from(edges_to_remove)

        # Update or remove nodes
        nodes_to_remove = []
        for node_id in list(self._graph.nodes):
            attrs = self._graph.nodes[node_id]
            attrs["source_paths"] = [p for p in attrs.get("source_paths", []) if p != sp_str]
            attrs["chunk_ids"] = [
                cid for cid in attrs.get("chunk_ids", [])
                if not cid.startswith(source_path.stem + ":")
            ]
            if not attrs["source_paths"]:
                nodes_to_remove.append(node_id)
        self._graph.remove_nodes_from(nodes_to_remove)

        # Remove processed IDs for this source
        prefix = source_path.stem + ":"
        self._processed_ids = {cid for cid in self._processed_ids if not cid.startswith(prefix)}

        self._save_graph()
        self._save_processed_ids()

    def get_processed_chunk_ids(self) -> set[str]:
        """Return chunk IDs for which entity extraction has already been done."""
        return set(self._processed_ids)

    def mark_chunk_processed(self, chunk_id: str) -> None:
        """Record that *chunk_id* has been processed by the EntityExtractor."""
        self._processed_ids.add(chunk_id)

    # ------------------------------------------------------------------
    # Read-only helpers for CLI / tests
    # ------------------------------------------------------------------

    def node_count(self) -> int:
        """Return the number of entity nodes in the graph."""
        return self._graph.number_of_nodes()

    def edge_count(self) -> int:
        """Return the number of relationship edges in the graph."""
        return self._graph.number_of_edges()

    def get_node(self, name: str) -> dict[str, Any] | None:
        """Return the attribute dict for a node, or None if not found."""
        key = name.lower()
        if key in self._graph:
            return dict(self._graph.nodes[key])
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _upsert_node(self, entity: Entity) -> None:
        """Insert a new entity node or merge into the existing one."""
        key = entity.name.lower()
        sp_str = str(entity.source_path)

        if key not in self._graph:
            self._graph.add_node(
                key,
                display_name=entity.name,
                entity_type=entity.entity_type,
                source_paths=[sp_str],
                chunk_ids=[entity.chunk_id],
            )
        else:
            attrs = self._graph.nodes[key]
            if sp_str not in attrs["source_paths"]:
                attrs["source_paths"].append(sp_str)
            if entity.chunk_id not in attrs["chunk_ids"]:
                attrs["chunk_ids"].append(entity.chunk_id)
            # Keep the most recently seen entity_type (later documents win)
            attrs["entity_type"] = entity.entity_type

    def _upsert_edge(self, rel: Relationship) -> None:
        """Insert a relationship edge, skipping exact duplicates."""
        src = rel.source_entity.lower()
        tgt = rel.target_entity.lower()

        # Ensure both nodes exist (may have been omitted from entities list)
        if src not in self._graph:
            self._graph.add_node(
                src,
                display_name=rel.source_entity,
                entity_type="Unknown",
                source_paths=[str(rel.source_path)],
                chunk_ids=[rel.chunk_id],
            )
        if tgt not in self._graph:
            self._graph.add_node(
                tgt,
                display_name=rel.target_entity,
                entity_type="Unknown",
                source_paths=[str(rel.source_path)],
                chunk_ids=[rel.chunk_id],
            )

        # Dedup: skip if an identical edge already exists for this chunk
        for _, _, attrs in self._graph.edges(src, data=True):
            if (
                attrs.get("relationship_type") == rel.relationship_type
                and attrs.get("chunk_id") == rel.chunk_id
            ):
                return

        self._graph.add_edge(
            src,
            tgt,
            relationship_type=rel.relationship_type,
            source_path=str(rel.source_path),
            chunk_id=rel.chunk_id,
        )

    def _find_nodes(self, entity_names: list[str], fuzzy: bool = False) -> list[str]:
        """Return graph node IDs that match any name in *entity_names*.

        Exact match: lowercase entity name == node key.
        Substring match (fuzzy=True): node key contains the entity name as substring.
        """
        found: list[str] = []
        for name in entity_names:
            norm = name.lower().strip()
            if not norm:
                continue
            if norm in self._graph:
                found.append(norm)
            elif fuzzy:
                for node_id in self._graph.nodes:
                    if norm in node_id or node_id in norm:
                        if node_id not in found:
                            found.append(node_id)
        return found

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_graph(self) -> nx.MultiDiGraph:
        if self._graph_path.exists():
            try:
                with self._graph_path.open(encoding="utf-8") as fh:
                    data = json.load(fh)
                return nx.node_link_graph(data, directed=True, multigraph=True)
            except Exception:  # noqa: BLE001
                logger.warning(
                    "graph.json could not be loaded; starting with an empty graph. "
                    "Run 'koa reingest' to rebuild.",
                )
        return nx.MultiDiGraph()

    def _save_graph(self) -> None:
        data = nx.node_link_data(self._graph)
        with self._graph_path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)

    def _load_processed_ids(self) -> set[str]:
        if self._ids_path.exists():
            try:
                with self._ids_path.open(encoding="utf-8") as fh:
                    return set(json.load(fh))
            except Exception:  # noqa: BLE001
                logger.warning("processed_chunk_ids.json could not be loaded; starting fresh.")
        return set()

    def _save_processed_ids(self) -> None:
        with self._ids_path.open("w", encoding="utf-8") as fh:
            json.dump(sorted(self._processed_ids), fh, indent=2)
