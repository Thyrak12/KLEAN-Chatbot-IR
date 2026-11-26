"""Boolean Retrieval module for KLEAN Chatbot.

Implements exact set-based boolean retrieval supporting AND, OR, NOT operators
and parenthetical grouping over an Inverted Index.
"""

from collections import deque
import logging
import re
from typing import Any, Dict, List, Optional, Set, Union

from src.indexing.inverted_index import InvertedIndex
from src.preprocessing.text_preprocessing import TextPreprocessor

logger = logging.getLogger(__name__)


class BooleanRetriever:
    """Evaluates Boolean queries (AND, OR, NOT) using an Inverted Index."""

    def __init__(
        self,
        inverted_index: InvertedIndex,
        preprocessor: Optional[TextPreprocessor] = None,
    ):
        """Initialize BooleanRetriever.

        Args:
            inverted_index: Built InvertedIndex instance.
            preprocessor: Preprocessor instance. If None, default TextPreprocessor is used.
        """
        self.index = inverted_index
        self.preprocessor = preprocessor or TextPreprocessor(remove_stopwords=False)
        self._all_doc_ids: Set[int] = set(self.index.doc_lengths.keys())

    @property
    def all_doc_ids(self) -> Set[int]:
        """Set of all indexed document IDs in the universe."""
        return set(self.index.doc_lengths.keys())

    # -------------------------------------------------------------
    # Low-level Set Operations on Postings Lists
    # -------------------------------------------------------------
    def _and_op(self, set_a: Set[int], set_b: Set[int]) -> Set[int]:
        """Intersection of two document sets (AND)."""
        return set_a.intersection(set_b)

    def _or_op(self, set_a: Set[int], set_b: Set[int]) -> Set[int]:
        """Union of two document sets (OR)."""
        return set_a.union(set_b)

    def _not_op(self, set_a: Set[int]) -> Set[int]:
        """Complement of a document set with respect to the universe (NOT)."""
        return self.all_doc_ids.difference(set_a)

    def _get_term_postings(self, term: str) -> Set[int]:
        """Get postings set for a single raw token."""
        clean_tokens = self.preprocessor.preprocess(term)
        if not clean_tokens:
            return set()
        # If term tokenizes into multiple tokens (e.g. Khmer word or phrase), intersect them
        result_set: Optional[Set[int]] = None
        for token in clean_tokens:
            postings = set(self.index.get_postings(token))
            if result_set is None:
                result_set = postings
            else:
                result_set = result_set.intersection(postings)
        return result_set if result_set is not None else set()

    # -------------------------------------------------------------
    # Query Tokenization & Shunting-Yard Parser
    # -------------------------------------------------------------
    def _tokenize_query(self, query: str) -> List[str]:
        """Split a raw boolean query string into tokens, operators, and parentheses."""
        pattern = re.compile(r"\(|\)|\bAND\b|\bOR\b|\bNOT\b|[^\s()]+")
        matches = pattern.findall(query)
        return [m.strip() for m in matches if m.strip()]

    def _to_postfix(self, tokens: List[str]) -> List[str]:
        """Convert infix Boolean expression to Reverse Polish Notation (Postfix) via Shunting-Yard algorithm.

        Precedence: NOT (highest) > AND > OR (lowest).
        """
        precedence = {"NOT": 3, "AND": 2, "OR": 1}
        output_queue: List[str] = []
        operator_stack: List[str] = []

        i = 0
        while i < len(tokens):
            token = tokens[i]

            if token.upper() in precedence:
                op = token.upper()
                while (
                    operator_stack
                    and operator_stack[-1] != "("
                    and precedence.get(operator_stack[-1], 0) >= precedence[op]
                ):
                    output_queue.append(operator_stack.pop())
                operator_stack.append(op)
            elif token == "(":
                operator_stack.append(token)
            elif token == ")":
                while operator_stack and operator_stack[-1] != "(":
                    output_queue.append(operator_stack.pop())
                if operator_stack and operator_stack[-1] == "(":
                    operator_stack.pop()
            else:
                # Operand (search term)
                output_queue.append(token)
            i += 1

        while operator_stack:
            output_queue.append(operator_stack.pop())

        return output_queue

    def search(self, query: str) -> List[int]:
        """Execute a Boolean query and return matching document IDs.

        Supports queries such as:
          - 'pizza AND italian'
          - 'noodles OR soup'
          - 'khmer AND NOT seafood'
          - 'soup AND (noodles OR beef)'
          - 'italian pizza' (implicitly treated as AND)

        Args:
            query: Boolean query string.

        Returns:
            Sorted list of matching document IDs.
        """
        raw_tokens = self._tokenize_query(query)
        if not raw_tokens:
            return []

        # Insert implicit AND between consecutive operands if user typed 'pizza italian'
        processed_tokens: List[str] = []
        operators = {"AND", "OR", "NOT"}

        for i, token in enumerate(raw_tokens):
            processed_tokens.append(token)
            if i < len(raw_tokens) - 1:
                next_token = raw_tokens[i + 1]
                curr_is_operand = token.upper() not in operators and token not in ("(", ")")
                next_is_operand = next_token.upper() not in operators and next_token not in ("(", ")")
                if curr_is_operand and next_is_operand:
                    processed_tokens.append("AND")
                elif curr_is_operand and next_token == "(":
                    processed_tokens.append("AND")
                elif token == ")" and next_is_operand:
                    processed_tokens.append("AND")

        # Convert to Postfix (RPN)
        postfix = self._to_postfix(processed_tokens)

        # Evaluate Postfix stack
        eval_stack: List[Set[int]] = []

        for token in postfix:
            upper_token = token.upper()
            if upper_token == "NOT":
                operand = eval_stack.pop() if eval_stack else set()
                eval_stack.append(self._not_op(operand))
            elif upper_token == "AND":
                right = eval_stack.pop() if eval_stack else set()
                left = eval_stack.pop() if eval_stack else set()
                eval_stack.append(self._and_op(left, right))
            elif upper_token == "OR":
                right = eval_stack.pop() if eval_stack else set()
                left = eval_stack.pop() if eval_stack else set()
                eval_stack.append(self._or_op(left, right))
            else:
                # Operand -> lookup postings list
                postings = self._get_term_postings(token)
                eval_stack.append(postings)

        result_set = eval_stack.pop() if eval_stack else set()
        return sorted(list(result_set))

    def search_documents(
        self, query: str, converter: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        """Execute Boolean query and return full restaurant records if converter is provided."""
        matching_ids = self.search(query)
        if not converter:
            return [{"doc_id": doc_id} for doc_id in matching_ids]

        results = []
        for doc_id in matching_ids:
            rec = converter.get_record(doc_id)
            if rec:
                results.append(rec)
            else:
                results.append({"doc_id": doc_id})
        return results
