"""EntityExtractor: extracts entities and relationships from a Chunk via Ollama LLM.

Conforms to the ``EntityExtractor`` Protocol defined in ``interfaces.py``.

The extraction prompt requests structured JSON output.  Temperature is set to 0.0
(or ``settings.knowledge_graph.extraction.extraction_temperature``) for
deterministic, reproducible results.

Deduplication is the caller's responsibility: the pipeline checks
``GraphStore.get_processed_chunk_ids()`` before calling ``extract()``.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import ollama

from knowledge_onboarding_agent.models import Chunk, Entity, Relationship

if TYPE_CHECKING:
    from knowledge_onboarding_agent.config import Settings

logger = logging.getLogger(__name__)

_EXTRACTION_PROMPT = """\
Extract all meaningful entities and their relationships from the following text.
Return ONLY valid JSON with exactly this structure — no explanation, no markdown fences:
{{
  "entities": [{{"name": "...", "type": "..."}}],
  "relationships": [{{"source": "...", "type": "...", "target": "..."}}]
}}

Entity type examples (not exhaustive): Technology, Framework, Library, Concept, \
Project, Organization, Person, Tool, Language, Protocol, Algorithm, Pattern

Relationship type examples (not exhaustive): Uses, Implements, Extends, DependsOn, \
RelatedTo, BelongsTo, IsA, Replaces, ComplementsWith, PartOf

Extract at most {max_entities} entities and {max_relationships} relationships.
If no meaningful entities are present, return {{"entities": [], "relationships": []}}.

Text:
{content}"""


class EntityExtractor:
    """Calls the local Ollama LLM to extract entities and relationships from a chunk.

    Parameters
    ----------
    llm_model:
        Ollama model name (e.g. ``"mistral"``).
    llm_base_url:
        Base URL of the local Ollama daemon.
    max_entities:
        Maximum number of entities to extract per chunk.
    max_relationships:
        Maximum number of relationships to extract per chunk.
    temperature:
        LLM temperature for extraction.  Use ``0.0`` for deterministic output.
    _llm_client:
        Injectable ``ollama.Client``-compatible object for testing.
    """

    def __init__(
        self,
        llm_model: str,
        llm_base_url: str,
        max_entities: int = 10,
        max_relationships: int = 10,
        temperature: float = 0.0,
        _llm_client: Any | None = None,
    ) -> None:
        self._model = llm_model
        self._max_entities = max_entities
        self._max_relationships = max_relationships
        self._temperature = temperature
        self._client: Any = _llm_client or ollama.Client(host=llm_base_url)

    @classmethod
    def from_settings(cls, settings: Settings, *, _llm_client: Any | None = None) -> EntityExtractor:
        """Construct an ``EntityExtractor`` from a ``Settings`` object."""
        return cls(
            llm_model=settings.llm.model,
            llm_base_url=settings.llm.ollama_base_url,
            max_entities=settings.knowledge_graph.extraction.max_entities_per_chunk,
            max_relationships=settings.knowledge_graph.extraction.max_relationships_per_chunk,
            temperature=settings.knowledge_graph.extraction.extraction_temperature,
            _llm_client=_llm_client,
        )

    # ------------------------------------------------------------------
    # EntityExtractor Protocol
    # ------------------------------------------------------------------

    def extract(self, chunk: Chunk) -> tuple[list[Entity], list[Relationship]]:
        """Extract entities and relationships from *chunk*.

        Returns ``(entities, relationships)``.  Both lists are empty when the
        LLM produces no valid output or when the chunk content is too short
        to contain meaningful entities.

        LLM call failures are logged as warnings and return empty lists rather
        than propagating exceptions, ensuring the ingestion pipeline continues.
        """
        if len(chunk.content.split()) < 10:
            return [], []

        prompt = _EXTRACTION_PROMPT.format(
            max_entities=self._max_entities,
            max_relationships=self._max_relationships,
            content=chunk.content,
        )

        try:
            response = self._client.chat(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": self._temperature},
            )
            raw_text: str = response["message"]["content"]
        except Exception as exc:  # noqa: BLE001
            logger.warning("EntityExtractor: LLM call failed for chunk %s: %s", chunk.id, exc)
            return [], []

        return self._parse_response(raw_text, chunk)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_response(
        self, raw_text: str, chunk: Chunk
    ) -> tuple[list[Entity], list[Relationship]]:
        """Parse the LLM JSON response into Entity and Relationship objects."""
        cleaned = self._extract_json(raw_text)
        if not cleaned:
            logger.warning(
                "EntityExtractor: no JSON found in LLM response for chunk %s", chunk.id
            )
            return [], []

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.warning(
                "EntityExtractor: JSON parse error for chunk %s: %s", chunk.id, exc
            )
            return [], []

        entities: list[Entity] = []
        for raw_entity in (data.get("entities") or [])[: self._max_entities]:
            name = str(raw_entity.get("name", "")).strip()
            entity_type = str(raw_entity.get("type", "Unknown")).strip()
            if not name:
                continue
            entities.append(
                Entity(
                    name=name,
                    entity_type=entity_type,
                    source_path=chunk.source_path,
                    chunk_id=chunk.id,
                )
            )

        # Build a set of known entity names for relationship validation
        known_names = {e.name.lower() for e in entities}

        relationships: list[Relationship] = []
        for raw_rel in (data.get("relationships") or [])[: self._max_relationships]:
            src = str(raw_rel.get("source", "")).strip()
            tgt = str(raw_rel.get("target", "")).strip()
            rel_type = str(raw_rel.get("type", "RelatedTo")).strip()
            if not src or not tgt or src.lower() == tgt.lower():
                continue
            relationships.append(
                Relationship(
                    source_entity=src,
                    target_entity=tgt,
                    relationship_type=rel_type,
                    source_path=chunk.source_path,
                    chunk_id=chunk.id,
                )
            )

        return entities, relationships

    @staticmethod
    def _extract_json(text: str) -> str:
        """Extract the first JSON object from *text*.

        Handles cases where the LLM wraps the JSON in markdown fences or
        adds surrounding commentary.
        """
        # Try direct parse first (ideal case: LLM returned clean JSON)
        stripped = text.strip()
        if stripped.startswith("{"):
            return stripped

        # Strip markdown fences: ```json ... ``` or ``` ... ```
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fence_match:
            return fence_match.group(1)

        # Find the first { ... } block
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            return brace_match.group(0)

        return ""
