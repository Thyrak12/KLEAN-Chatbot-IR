"""Context Builder for KLEAN Chatbot RAG.

Extracts and formats verified restaurant metadata into compact,
grounded context strings for LLM injection.
"""

from typing import Any, Dict, List, Optional


class ContextBuilder:
    """Builds grounded prompt context from retrieved restaurant records."""

    def __init__(self, records_by_id: Optional[Dict[int, Dict[str, Any]]] = None):
        """Initialize ContextBuilder.

        Args:
            records_by_id: Optional mapping of doc_id to full restaurant record dict.
        """
        self.records_by_id = records_by_id or {}

    def set_records(self, records_by_id: Dict[int, Dict[str, Any]]) -> None:
        """Update restaurant records lookup dictionary."""
        self.records_by_id = records_by_id

    def build_context(
        self,
        retrieval_results: List[Dict[str, Any]],
        max_restaurants: int = 5,
    ) -> str:
        """Format top retrieved restaurant records into a structured context string.

        Args:
            retrieval_results: List of result dicts containing at least 'doc_id'.
            max_restaurants: Maximum number of restaurants to include.

        Returns:
            Formatted string of restaurant facts.
        """
        if not retrieval_results:
            return ""

        context_blocks: List[str] = []
        candidates = retrieval_results[:max_restaurants]

        for i, item in enumerate(candidates, start=1):
            doc_id = item["doc_id"]
            record = self.records_by_id.get(doc_id)

            if not record:
                # If full record not in lookup, use what info is in item
                block = f"[{i}] Restaurant ID: {doc_id} (Score: {item.get('score', 0):.4f})"
                context_blocks.append(block)
                continue

            lines = [f"[{i}] {record.get('name', 'Unknown Name')} (ID: {doc_id})"]

            # Category & Cuisines
            cat = record.get("category")
            sub_cats = record.get("sub_category")
            if cat:
                lines.append(f"    Primary Category: {cat}")
            if sub_cats and isinstance(sub_cats, list) and len(sub_cats) > 0:
                lines.append(f"    Cuisines / Subcategories: {', '.join(sub_cats)}")

            # Location & Address
            address = record.get("address")
            if address:
                lines.append(f"    Address: {address}")

            # Contact Information
            phone = record.get("phone")
            if phone:
                lines.append(f"    Phone: {phone}")

            website = record.get("website")
            if website:
                lines.append(f"    Website: {website}")

            email = record.get("email")
            if email:
                lines.append(f"    Email: {email}")

            # Retrieval ranking diagnostic
            score = item.get("score")
            rank = item.get("rank")
            bm25_r = item.get("bm25_rank")
            vec_r = item.get("vector_rank")
            diag_parts = []
            if rank is not None:
                diag_parts.append(f"Rank: #{rank}")
            if score is not None:
                diag_parts.append(f"Score: {score:.4f}")
            if bm25_r is not None:
                diag_parts.append(f"BM25 Rank: #{bm25_r}")
            if vec_r is not None:
                diag_parts.append(f"Vector Rank: #{vec_r}")

            if diag_parts:
                lines.append(f"    Relevance: {', '.join(diag_parts)}")

            context_blocks.append("\n".join(lines))

        return "\n\n".join(context_blocks)
