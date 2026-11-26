"""Text preprocessing module for KLEAN Chatbot.

Provides configurable normalization, tokenization, and stopword removal
specifically designed for bilingual (English and Khmer) Information Retrieval.
"""

import logging
import re
import string
from typing import Dict, Iterable, List, Optional, Set, Union

import nltk
from nltk.corpus import stopwords

# Optional Khmer NLP toolkit
try:
    import khmernltk
    HAS_KHMER_NLTK = True
except ImportError:
    HAS_KHMER_NLTK = False

logger = logging.getLogger(__name__)

# Unicode ranges for Khmer script
KHMER_CHAR_REGEX = re.compile(r"[\u1780-\u17FF\u19E0-\u19FF]")
# Token boundary pattern for mixed alphabets and numbers
WORDS_REGEX = re.compile(r"[a-zA-Z0-9]+|[\u1780-\u17FF\u19E0-\u19FF]+")


class TextPreprocessor:
    """Configurable text normalizer and tokenizer for English and Khmer text."""

    def __init__(
        self,
        lowercase: bool = True,
        remove_punctuation: bool = True,
        remove_stopwords: bool = True,
        enable_khmer: bool = True,
        custom_stopwords: Optional[Iterable[str]] = None,
        min_token_length: int = 1,
    ):
        """Initialize TextPreprocessor.

        Args:
            lowercase: Whether to convert Latin characters to lowercase.
            remove_punctuation: Whether to strip punctuation symbols.
            remove_stopwords: Whether to filter English stopwords.
                             (Note: English stopwords are NEVER applied to Khmer text).
            enable_khmer: Whether to use specialized Khmer word segmentation.
            custom_stopwords: Additional terms to treat as stopwords.
            min_token_length: Minimum character length for tokens to be kept.
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.enable_khmer = enable_khmer and HAS_KHMER_NLTK
        self.min_token_length = min_token_length

        # Initialize English stopwords
        try:
            self._stopwords: Set[str] = set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            self._stopwords = set(stopwords.words("english"))

        if custom_stopwords:
            self._stopwords.update(s.lower() for s in custom_stopwords)

        # Common punctuation set
        self._punctuation_table = str.maketrans("", "", string.punctuation + "“”‘’…—–«»")

        if enable_khmer and not HAS_KHMER_NLTK:
            logger.warning(
                "khmer-nltk is not installed. Khmer segmentation will fallback to character/regex extraction."
            )

    @staticmethod
    def contains_khmer(text: str) -> bool:
        """Check if string contains any Khmer script characters."""
        return bool(KHMER_CHAR_REGEX.search(text))

    def _tokenize_khmer_segment(self, segment: str) -> List[str]:
        """Segment a Khmer text segment into discrete words."""
        if not segment.strip():
            return []

        if self.enable_khmer:
            try:
                # khmernltk returns words and spaces
                raw_tokens = khmernltk.word_tokenize(segment)
                return [t.strip() for t in raw_tokens if t.strip()]
            except Exception as e:
                logger.debug("khmernltk segmentation failed on '%s': %s", segment, e)

        # Regex fallback: extract sequences of Khmer characters
        return [match.group() for match in KHMER_CHAR_REGEX.finditer(segment)]

    def preprocess(self, text: Optional[str]) -> List[str]:
        """Normalize and tokenize text into a list of clean tokens.

        Handles English, Khmer, and bilingual text gracefully.

        Args:
            text: Input string.

        Returns:
            List of normalized tokens.
        """
        if not text or not isinstance(text, str):
            return []

        # 1. Lowercase Latin characters
        normalized_text = text.lower() if self.lowercase else text

        # 2. Punctuation removal / space replacement
        if self.remove_punctuation:
            # Replace common IR delimiters (pipe, slash, dash) with spaces to preserve words
            normalized_text = re.sub(r"[|\-_/,\.;:()\[\]{}]", " ", normalized_text)
            normalized_text = normalized_text.translate(self._punctuation_table)

        tokens: List[str] = []

        # 3. Tokenization
        # If text contains Khmer characters and specialized Khmer tokenizer is enabled
        if self.enable_khmer and self.contains_khmer(normalized_text):
            # Split text into mixed tokens/chunks
            chunks = normalized_text.split()
            for chunk in chunks:
                if self.contains_khmer(chunk):
                    # Segment Khmer words without applying English stopwords
                    khmer_words = self._tokenize_khmer_segment(chunk)
                    tokens.extend(khmer_words)
                else:
                    # English / numeric token
                    clean_token = chunk.strip()
                    if clean_token:
                        tokens.append(clean_token)
        else:
            # Standard regex word boundary tokenization (fast path)
            tokens = [m.group() for m in WORDS_REGEX.finditer(normalized_text)]

        # 4. Stopword filtering & length check
        filtered_tokens: List[str] = []
        for token in tokens:
            token = token.strip()
            if not token or len(token) < self.min_token_length:
                continue

            # Only filter against English stopwords if token is pure Latin/ASCII
            if self.remove_stopwords and token.isascii():
                if token.lower() in self._stopwords:
                    continue

            filtered_tokens.append(token)

        return filtered_tokens

    def preprocess_documents(self, corpus: Dict[int, str]) -> Dict[int, List[str]]:
        """Preprocess an entire corpus dictionary {doc_id: text}.

        Args:
            corpus: Mapping from document ID to raw document text.

        Returns:
            Mapping from document ID to list of clean tokens.
        """
        tokenized_corpus: Dict[int, List[str]] = {}
        for doc_id, text in corpus.items():
            tokenized_corpus[doc_id] = self.preprocess(text)
        return tokenized_corpus
