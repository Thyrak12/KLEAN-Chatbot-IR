"""Document conversion utility for KLEAN Chatbot.

Converts structured restaurant records into searchable textual documents
suitable for inverted indexing, TF-IDF, BM25, and vector embeddings, while
maintaining explicit mapping back to the original restaurant records.
"""

from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


@dataclass
class SearchDocument:
    """Represents a searchable document representation of a restaurant."""

    doc_id: int
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    original_record: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert SearchDocument to a dictionary."""
        return {
            "doc_id": self.doc_id,
            "text": self.text,
            "metadata": self.metadata,
        }


class DocumentConverter:
    """Converts structured restaurant dictionaries into searchable documents."""

    DEFAULT_FIELDS: Sequence[str] = ("name", "category", "sub_category", "address", "postal_code")

    def __init__(self, include_fields: Optional[Sequence[str]] = None):
        """Initialize DocumentConverter.

        Args:
            include_fields: Sequence of field names to include in generated text.
                           Defaults to ('name', 'category', 'sub_category', 'address', 'postal_code').
        """
        self.include_fields = list(include_fields) if include_fields else list(self.DEFAULT_FIELDS)
        self._doc_id_to_record: Dict[int, Dict[str, Any]] = {}
        self._doc_id_to_document: Dict[int, SearchDocument] = {}

    def convert_record(self, record: Dict[str, Any]) -> SearchDocument:
        """Convert a single restaurant record into a SearchDocument.

        Empty and null fields are omitted. Document ID matches restaurant ID.

        Args:
            record: Raw restaurant dictionary conforming to project schema.

        Returns:
            SearchDocument instance with formatted text and mappings.
        """
        doc_id = record.get("id")
        if doc_id is None:
            raise ValueError("Restaurant record must contain an 'id' field")

        parts: List[str] = []

        for field_name in self.include_fields:
            val = record.get(field_name)
            if val is None:
                continue

            # Format field values cleanly
            if isinstance(val, list):
                # E.g., sub_category: ["Italian", "Pizza"] -> "Italian, Pizza"
                cleaned_items = [str(item).strip() for item in val if item and str(item).strip()]
                if cleaned_items:
                    label = field_name.replace("_", " ").title()
                    parts.append(f"{label}: {', '.join(cleaned_items)}")
            elif isinstance(val, str):
                cleaned_val = val.strip()
                if cleaned_val:
                    label = field_name.replace("_", " ").title()
                    parts.append(f"{label}: {cleaned_val}")
            elif isinstance(val, (int, float)):
                label = field_name.replace("_", " ").title()
                parts.append(f"{label}: {val}")

        # Combine into cohesive textual document
        text = " | ".join(parts)

        # Store metadata for filtering and citation
        metadata = {
            "name": record.get("name"),
            "category": record.get("category"),
            "sub_category": record.get("sub_category", []),
            "address": record.get("address"),
            "postal_code": record.get("postal_code"),
            "latitude": record.get("latitude"),
            "longitude": record.get("longitude"),
            "phone": record.get("phone"),
            "website": record.get("website"),
        }

        doc = SearchDocument(
            doc_id=doc_id,
            text=text,
            metadata=metadata,
            original_record=record,
        )

        return doc

    def convert_all(self, records: List[Dict[str, Any]]) -> List[SearchDocument]:
        """Convert a list of restaurant records into a list of SearchDocuments.

        Also populates the internal bidirectional mapping indices.

        Args:
            records: List of restaurant dictionaries.

        Returns:
            List of SearchDocument instances.
        """
        documents: List[SearchDocument] = []
        self._doc_id_to_record.clear()
        self._doc_id_to_document.clear()

        for rec in records:
            doc = self.convert_record(rec)
            documents.append(doc)
            self._doc_id_to_record[doc.doc_id] = rec
            self._doc_id_to_document[doc.doc_id] = doc

        logger.info("Converted %d restaurant records into searchable documents", len(documents))
        return documents

    def get_record(self, doc_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve original restaurant record by document ID."""
        return self._doc_id_to_record.get(doc_id)

    def get_document(self, doc_id: int) -> Optional[SearchDocument]:
        """Retrieve SearchDocument by document ID."""
        return self._doc_id_to_document.get(doc_id)

    def get_mapping(self) -> Dict[int, Dict[str, Any]]:
        """Return a mapping of all document IDs to their original records."""
        return dict(self._doc_id_to_record)

    def get_corpus(self, documents: Optional[List[SearchDocument]] = None) -> Dict[int, str]:
        """Return a dictionary mapping doc_id -> text.

        Convenient for IR indexers (TF-IDF, BM25, Inverted Index).
        """
        if documents is not None:
            return {doc.doc_id: doc.text for doc in documents}
        return {doc_id: doc.text for doc_id, doc in self._doc_id_to_document.items()}
