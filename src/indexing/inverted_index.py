"""Inverted Index module for Information Retrieval.

Constructs and manages an inverted index mapping terms to document IDs and
term frequencies, supporting document additions, updates, deletions, and lookups.
"""

from collections import defaultdict
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

logger = logging.getLogger(__name__)


class InvertedIndex:
    """Inverted Index supporting term lookups, postings lists, and frequency metrics."""

    def __init__(self):
        """Initialize empty InvertedIndex."""
        # term -> {doc_id: term_frequency}
        self._index: Dict[str, Dict[int, int]] = defaultdict(dict)
        # doc_id -> total token count
        self._doc_lengths: Dict[int, int] = {}
        # doc_id -> {term: frequency} (forward index for fast removal / lookup)
        self._doc_to_terms: Dict[int, Dict[str, int]] = defaultdict(dict)

    def add_document(self, doc_id: int, tokens: List[str]) -> None:
        """Add or replace a single document in the inverted index.

        Args:
            doc_id: Unique integer identifier for the document.
            tokens: Preprocessed tokens for the document.
        """
        # If document already exists, remove it first to handle clean replacement
        if doc_id in self._doc_lengths:
            self.remove_document(doc_id)

        self._doc_lengths[doc_id] = len(tokens)

        # Calculate term frequencies for this document
        tf_counts: Dict[str, int] = defaultdict(int)
        for token in tokens:
            tf_counts[token] += 1

        # Populate inverted index and forward mapping
        for term, freq in tf_counts.items():
            self._index[term][doc_id] = freq
            self._doc_to_terms[doc_id][term] = freq

    def update_document(self, doc_id: int, tokens: List[str]) -> None:
        """Update an existing document in the index."""
        self.add_document(doc_id, tokens)

    def remove_document(self, doc_id: int) -> bool:
        """Remove a document from the inverted index.

        Args:
            doc_id: Integer identifier of the document to remove.

        Returns:
            True if document existed and was removed, False otherwise.
        """
        if doc_id not in self._doc_lengths:
            return False

        # Remove terms from inverted index
        terms = list(self._doc_to_terms.get(doc_id, {}).keys())
        for term in terms:
            if term in self._index and doc_id in self._index[term]:
                del self._index[term][doc_id]
                # If term has no more documents, clean up the term entry
                if not self._index[term]:
                    del self._index[term]

        del self._doc_lengths[doc_id]
        if doc_id in self._doc_to_terms:
            del self._doc_to_terms[doc_id]

        return True

    def build(self, tokenized_corpus: Dict[int, List[str]]) -> None:
        """Build the inverted index from a preprocessed tokenized corpus.

        Args:
            tokenized_corpus: Dictionary mapping doc_id -> list of tokens.
        """
        self.clear()
        for doc_id, tokens in tokenized_corpus.items():
            self.add_document(doc_id, tokens)
        logger.info(
            "Built InvertedIndex with %d unique terms across %d documents",
            len(self._index),
            len(self._doc_lengths),
        )

    def clear(self) -> None:
        """Clear the entire index."""
        self._index.clear()
        self._doc_lengths.clear()
        self._doc_to_terms.clear()

    def get_postings(self, term: str) -> List[int]:
        """Retrieve list of document IDs containing the specified term.

        Args:
            term: Term string.

        Returns:
            Sorted list of document IDs (postings list).
        """
        key = term.lower() if term.isascii() else term
        term_postings = self._index.get(key, {})
        return sorted(list(term_postings.keys()))

    def get_term_frequency(self, term: str, doc_id: int) -> int:
        """Get term frequency (TF) of a term inside a specific document.

        Args:
            term: Term string.
            doc_id: Document ID.

        Returns:
            Integer term frequency (0 if not present).
        """
        key = term.lower() if term.isascii() else term
        return self._index.get(key, {}).get(doc_id, 0)

    def get_document_frequency(self, term: str) -> int:
        """Get document frequency (DF) — number of documents containing term.

        Args:
            term: Term string.

        Returns:
            Count of documents containing the term.
        """
        key = term.lower() if term.isascii() else term
        return len(self._index.get(key, {}))

    def get_document_terms(self, doc_id: int) -> Dict[str, int]:
        """Retrieve all terms and their frequencies for a specific document."""
        return dict(self._doc_to_terms.get(doc_id, {}))

    @property
    def vocabulary(self) -> List[str]:
        """Return all unique terms in the vocabulary sorted alphabetically."""
        return sorted(list(self._index.keys()))

    @property
    def vocabulary_size(self) -> int:
        """Return total number of unique terms indexed."""
        return len(self._index)

    @property
    def num_documents(self) -> int:
        """Return total number of indexed documents."""
        return len(self._doc_lengths)

    @property
    def doc_lengths(self) -> Dict[int, int]:
        """Return dictionary mapping doc_id -> token length."""
        return dict(self._doc_lengths)

    @property
    def avg_doc_length(self) -> float:
        """Calculate average document token length across the corpus."""
        if not self._doc_lengths:
            return 0.0
        return sum(self._doc_lengths.values()) / len(self._doc_lengths)

    def to_dict(self) -> Dict[str, Any]:
        """Export inverted index structure to a serializable dictionary."""
        return {
            "index": {term: dict(postings) for term, postings in self._index.items()},
            "doc_lengths": {str(doc_id): length for doc_id, length in self._doc_lengths.items()},
        }

    def save(self, filepath: Union[str, Path]) -> None:
        """Save inverted index to a JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info("Saved InvertedIndex to %s", path)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "InvertedIndex":
        """Load an InvertedIndex from a saved JSON file."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Index file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        idx = cls()
        raw_index = data.get("index", {})
        raw_lengths = data.get("doc_lengths", {})

        for term, postings in raw_index.items():
            for doc_id_str, freq in postings.items():
                doc_id = int(doc_id_str)
                idx._index[term][doc_id] = int(freq)
                idx._doc_to_terms[doc_id][term] = int(freq)

        for doc_id_str, length in raw_lengths.items():
            idx._doc_lengths[int(doc_id_str)] = int(length)

        logger.info("Loaded InvertedIndex from %s with %d terms", path, len(idx._index))
        return idx
