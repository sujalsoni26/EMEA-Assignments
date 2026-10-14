"""BM25 keyword search, used alongside vector search (hybrid retrieval).

Embeddings capture meaning but are weak at exact tokens such as error codes
("E2"), part numbers ("AP-RO2") or rare names. BM25 scores passages by the
query words they contain, weighting rare words more, so it finds those exact
matches. The index is rebuilt in memory whenever the documents change, which
is fast for the few thousand passages a local knowledge base holds.
"""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_STOPWORDS = set(
    """a an and are as at be by can do does for from has have how i in is it its my of on or our should so that the
    their them then there these they this to was we what when where which who why will with you your about into
    than also any all if not no may must""".split()
)


def tokenize(text: str) -> list[str]:
    tokens = []
    for token in _TOKEN.findall(text.lower()):
        if token in _STOPWORDS:
            continue
        tokens.append(token)
        if "-" in token:  # "ap-ro2" also matches "ro2" and "ap"
            tokens.extend(part for part in token.split("-") if part and part not in _STOPWORDS)
    return tokens


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.ids: list[int] = []
        self.term_freqs: list[Counter[str]] = []
        self.lengths: list[int] = []
        self.doc_freq: Counter[str] = Counter()
        self.avg_length = 0.0

    def build(self, passages: dict[int, str]) -> None:
        self.ids = list(passages)
        self.term_freqs = [Counter(tokenize(passages[i])) for i in self.ids]
        self.lengths = [sum(tf.values()) for tf in self.term_freqs]
        self.doc_freq = Counter(term for tf in self.term_freqs for term in tf)
        self.avg_length = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0

    def idf(self, term: str) -> float:
        n = len(self.ids)
        df = self.doc_freq.get(term, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def is_rare(self, term: str) -> bool:
        """A term that occurs in very few passages (a code, part number or specific name)."""
        df = self.doc_freq.get(term, 0)
        return 0 < df <= max(2, int(0.02 * len(self.ids)))

    def search(self, query: str, k: int) -> list[tuple[int, float, set[str]]]:
        """Return up to ``k`` (passage id, score, matched rare terms), best first."""
        terms = set(tokenize(query))
        if not terms or not self.ids:
            return []
        scored = []
        for chunk_id, tf, length in zip(self.ids, self.term_freqs, self.lengths):
            score = 0.0
            for term in terms:
                f = tf.get(term, 0)
                if f:
                    norm = self.k1 * (1 - self.b + self.b * length / (self.avg_length or 1))
                    score += self.idf(term) * f * (self.k1 + 1) / (f + norm)
            if score > 0:
                scored.append((chunk_id, score, {t for t in terms if tf.get(t) and self.is_rare(t)}))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:k]
