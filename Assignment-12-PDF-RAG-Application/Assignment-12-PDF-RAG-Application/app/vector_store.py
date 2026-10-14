"""A persistent vector index built on FAISS.

* Vectors are L2-normalised and stored in an inner-product index, so the score
  of a match is its cosine similarity (1.0 = identical meaning).
* ``IndexIDMap2`` lets us delete all chunks of a document by id.
* Chunk text and document metadata live in a JSON file next to the index.
  Both files are written atomically after every change.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import faiss
import numpy as np

from .lexical import BM25Index

logger = logging.getLogger(__name__)


@dataclass
class DocumentRecord:
    id: str
    filename: str
    sha256: str
    pages: int
    chunks: int
    words: int
    size_bytes: int
    uploaded_at: str


@dataclass
class ChunkRecord:
    id: int
    document_id: str
    filename: str
    page: int
    text: str


@dataclass
class SearchHit:
    chunk: ChunkRecord
    score: float  # cosine similarity between the question and the passage
    keyword_terms: tuple[str, ...] = ()  # rare query terms found in the passage by BM25

# Reciprocal Rank Fusion constant (Cormack et al., 2009).
RRF_K = 60


def _normalise(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype="float32")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


class VectorStore:
    INDEX_FILE = "index.faiss"
    META_FILE = "store.json"

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.index: faiss.IndexIDMap2 | None = None
        self.dimension: int | None = None
        self.documents: dict[str, DocumentRecord] = {}
        self.chunks: dict[int, ChunkRecord] = {}
        self.next_id = 0
        self.bm25 = BM25Index()

    def _rebuild_keyword_index(self) -> None:
        self.bm25.build({cid: chunk.text for cid, chunk in self.chunks.items()})

    # ------------------------------------------------------------------ state
    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    def find_by_hash(self, sha256: str) -> DocumentRecord | None:
        return next((d for d in self.documents.values() if d.sha256 == sha256), None)

    # ------------------------------------------------------------------ mutation
    def add_document(self, document: DocumentRecord, chunks: list[tuple[int, str]], vectors: list[list[float]]) -> None:
        """Add a document. ``chunks`` is a list of (page, text) pairs aligned with ``vectors``."""
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        matrix = _normalise(np.array(vectors))
        if self.index is None:
            self.dimension = matrix.shape[1]
            self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(self.dimension))
        elif matrix.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding size {matrix.shape[1]} does not match the index ({self.dimension}). "
                "Did the embedding model change? Clear the knowledge base and upload the documents again."
            )
        ids = np.arange(self.next_id, self.next_id + len(chunks), dtype="int64")
        self.index.add_with_ids(matrix, ids)
        for chunk_id, (page, text) in zip(ids.tolist(), chunks):
            self.chunks[chunk_id] = ChunkRecord(chunk_id, document.id, document.filename, page, text)
        self.next_id += len(chunks)
        self.documents[document.id] = document
        self.save()

    def delete_document(self, document_id: str) -> bool:
        if document_id not in self.documents:
            return False
        ids = [cid for cid, chunk in self.chunks.items() if chunk.document_id == document_id]
        if ids and self.index is not None:
            self.index.remove_ids(np.array(ids, dtype="int64"))
        for cid in ids:
            del self.chunks[cid]
        del self.documents[document_id]
        if not self.documents:
            self.index, self.dimension = None, None
        self.save()
        return True

    def clear(self) -> None:
        self.index, self.dimension = None, None
        self.documents.clear()
        self.chunks.clear()
        self.save()

    # ------------------------------------------------------------------ search
    def search(self, query_vector: list[float], k: int) -> list[SearchHit]:
        if self.index is None or self.index.ntotal == 0:
            return []
        query = _normalise(np.array([query_vector]))
        if query.shape[1] != self.dimension:
            raise ValueError("The query embedding size does not match the index.")
        scores, ids = self.index.search(query, min(k, self.index.ntotal))
        hits = []
        for score, chunk_id in zip(scores[0].tolist(), ids[0].tolist()):
            if chunk_id != -1 and chunk_id in self.chunks:
                hits.append(SearchHit(self.chunks[chunk_id], float(score)))
        return hits

    def hybrid_search(self, query: str, query_vector: list[float], k: int) -> list[SearchHit]:
        """Combine vector search and BM25 keyword search with Reciprocal Rank Fusion.

        Each candidate gets 1 / (RRF_K + rank) from every ranking it appears in;
        the ``k`` best fused candidates are returned with their cosine similarity
        (computed for keyword-only hits too) and any rare query terms they contain.
        """
        if self.index is None or self.index.ntotal == 0:
            return []
        pool = max(k * 3, 10)
        dense = self.search(query_vector, pool)
        lexical = self.bm25.search(query, pool)
        fused: dict[int, float] = {}
        for rank, hit in enumerate(dense):
            fused[hit.chunk.id] = fused.get(hit.chunk.id, 0.0) + 1 / (RRF_K + rank + 1)
        for rank, (chunk_id, _, _) in enumerate(lexical):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1 / (RRF_K + rank + 1)

        dense_scores = {hit.chunk.id: hit.score for hit in dense}
        rare_terms = {chunk_id: terms for chunk_id, _, terms in lexical}
        query_unit = _normalise(np.array([query_vector]))[0]
        hits = []
        for chunk_id in sorted(fused, key=fused.get, reverse=True)[:k]:
            score = dense_scores.get(chunk_id)
            if score is None:  # found only by keywords: compute its cosine similarity
                score = float(np.dot(self.index.reconstruct(int(chunk_id)), query_unit))
            hits.append(SearchHit(self.chunks[chunk_id], score, tuple(sorted(rare_terms.get(chunk_id, ())))))
        return hits

    # ------------------------------------------------------------------ persistence
    def save(self) -> None:
        self._rebuild_keyword_index()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "dimension": self.dimension,
            "next_id": self.next_id,
            "documents": [asdict(d) for d in self.documents.values()],
            "chunks": [asdict(c) for c in self.chunks.values()],
        }
        self._atomic_write(self.data_dir / self.META_FILE, json.dumps(meta, ensure_ascii=False).encode())
        index_path = self.data_dir / self.INDEX_FILE
        if self.index is None:
            index_path.unlink(missing_ok=True)
        else:
            self._atomic_write(index_path, faiss.serialize_index(self.index).tobytes())

    def load(self) -> None:
        meta_path = self.data_dir / self.META_FILE
        index_path = self.data_dir / self.INDEX_FILE
        if not meta_path.exists():
            return
        try:
            meta = json.loads(meta_path.read_text())
            self.dimension = meta.get("dimension")
            self.next_id = meta.get("next_id", 0)
            self.documents = {d["id"]: DocumentRecord(**d) for d in meta.get("documents", [])}
            self.chunks = {c["id"]: ChunkRecord(**c) for c in meta.get("chunks", [])}
            if index_path.exists():
                self.index = faiss.deserialize_index(np.frombuffer(index_path.read_bytes(), dtype="uint8"))
            if self.index is not None and self.index.ntotal != len(self.chunks):
                raise ValueError("index and metadata are out of sync")
            self._rebuild_keyword_index()
            logger.info("Loaded knowledge base: %d documents, %d chunks", len(self.documents), len(self.chunks))
        except Exception:
            logger.exception("Could not load the saved knowledge base; starting with an empty one")
            self.index, self.dimension, self.documents, self.chunks, self.next_id = None, None, {}, {}, 0

    @staticmethod
    def _atomic_write(path: Path, data: bytes) -> None:
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
