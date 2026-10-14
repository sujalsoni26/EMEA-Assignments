"""PDF validation and text extraction using PyMuPDF.

The extractor works on text *blocks* (roughly paragraphs) rather than raw lines,
which keeps paragraph structure intact. It also removes running headers/footers
and page numbers, and re-joins words hyphenated across line breaks.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

import pymupdf

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class PDFError(Exception):
    status_code = 422
    code = "pdf_error"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class NoFileError(PDFError):
    status_code = 400
    code = "no_file"


class InvalidFileTypeError(PDFError):
    status_code = 415
    code = "invalid_file_type"


class EmptyFileError(PDFError):
    status_code = 400
    code = "empty_file"


class FileTooLargeError(PDFError):
    status_code = 413
    code = "file_too_large"


class CorruptPDFError(PDFError):
    code = "corrupt_pdf"


class EncryptedPDFError(PDFError):
    code = "encrypted_pdf"


class TooManyPagesError(PDFError):
    code = "too_many_pages"


class NoTextError(PDFError):
    code = "no_extractable_text"


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #
@dataclass
class PageText:
    number: int  # 1-based page number
    text: str  # paragraphs separated by blank lines


@dataclass
class ExtractedDocument:
    filename: str
    page_count: int
    pages: list[PageText]
    title: str | None = None
    skipped_pages: list[int] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(page.text for page in self.pages if page.text)

    @property
    def word_count(self) -> int:
        return sum(len(page.text.split()) for page in self.pages)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def _format_size(num_bytes: int) -> str:
    if num_bytes >= 1024 * 1024:
        return f"{num_bytes / (1024 * 1024):.1f} MB"
    return f"{max(1, num_bytes // 1024)} KB"


def validate_upload(filename: str | None, data: bytes, max_bytes: int) -> None:
    """Cheap checks performed before the PDF is parsed."""
    if not filename:
        raise NoFileError("No file was uploaded. Please choose a PDF document.")
    if not filename.lower().endswith(".pdf"):
        raise InvalidFileTypeError(f"'{filename}' is not a PDF. Please upload a file with the .pdf extension.")
    if len(data) == 0:
        raise EmptyFileError(f"'{filename}' is empty.")
    if len(data) > max_bytes:
        raise FileTooLargeError(
            f"'{filename}' is too large ({_format_size(len(data))}). The maximum size is {_format_size(max_bytes)}."
        )
    # The PDF header must appear within the first 1024 bytes (PDF 1.7 spec, 7.5.2).
    if b"%PDF-" not in data[:1024]:
        raise InvalidFileTypeError(
            f"'{filename}' has a .pdf extension but is not a valid PDF file (missing PDF header)."
        )


# --------------------------------------------------------------------------- #
# Text clean-up helpers
# --------------------------------------------------------------------------- #
_PAGE_NUMBER = re.compile(r"^(page\s*)?#(\s*(of|/)\s*#)?$|^[-–—]\s*#\s*[-–—]$", re.IGNORECASE)


def _normalise_for_repeat(text: str) -> str:
    return re.sub(r"\d+", "#", text.lower()).strip()


def _join_block_lines(text: str) -> str:
    # NFKC turns typographic ligatures such as "\ufb01" into plain letters ("fi").
    text = unicodedata.normalize("NFKC", text)
    # Re-join words hyphenated across a line break ("infor-\nmation" -> "information").
    text = re.sub(r"(\w)-\n([a-z])", r"\1\2", text)
    text = text.replace("­", "")  # soft hyphens
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"[ \t ]+", " ", text)
    return text.strip()


def _remove_repeated_headers(pages_blocks: list[list[str]]) -> list[list[str]]:
    """Drop blocks that repeat at the top or bottom of most pages (running
    headers/footers) and blocks that are only page numbers."""
    page_total = len(pages_blocks)
    repeated: set[str] = set()
    if page_total >= 3:
        edge_counts: Counter[str] = Counter()
        for blocks in pages_blocks:
            edges = {_normalise_for_repeat(b) for b in blocks[:2] + blocks[-2:]}
            edge_counts.update(e for e in edges if e and len(e) <= 120)
        threshold = max(3, int(0.5 * page_total))
        repeated = {text for text, count in edge_counts.items() if count >= threshold}

    cleaned: list[list[str]] = []
    for blocks in pages_blocks:
        kept = []
        for block in blocks:
            norm = _normalise_for_repeat(block)
            if norm in repeated or _PAGE_NUMBER.match(norm):
                continue
            kept.append(block)
        cleaned.append(kept)
    return cleaned


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
def extract_pdf(data: bytes, filename: str, *, max_pages: int, min_words: int) -> ExtractedDocument:
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:  # PyMuPDF raises several exception types for bad files
        raise CorruptPDFError(f"'{filename}' could not be opened. The file may be corrupted.") from exc

    with doc:
        if doc.needs_pass:
            raise EncryptedPDFError(
                f"'{filename}' is password-protected. Please remove the password and upload it again."
            )
        if doc.page_count == 0:
            raise CorruptPDFError(f"'{filename}' does not contain any pages.")
        if doc.page_count > max_pages:
            raise TooManyPagesError(
                f"'{filename}' has {doc.page_count} pages. The maximum supported is {max_pages} pages."
            )

        pages_blocks: list[list[str]] = []
        skipped: list[int] = []
        for page in doc:
            try:
                raw_blocks = page.get_text("blocks", sort=True)
            except Exception:  # a single damaged page should not fail the whole document
                logger.warning("Could not read page %d of %s", page.number + 1, filename)
                skipped.append(page.number + 1)
                raw_blocks = []
            blocks = [_join_block_lines(b[4]) for b in raw_blocks if b[6] == 0]  # 0 = text block
            pages_blocks.append([b for b in blocks if b])

        metadata_title = (doc.metadata or {}).get("title", "").strip()
        page_count = doc.page_count

    pages_blocks = _remove_repeated_headers(pages_blocks)
    pages = [PageText(number=i + 1, text="\n\n".join(blocks)) for i, blocks in enumerate(pages_blocks)]
    document = ExtractedDocument(
        filename=filename,
        page_count=page_count,
        pages=pages,
        title=metadata_title if _looks_like_title(metadata_title) else None,
        skipped_pages=skipped,
    )

    if document.word_count < min_words:
        raise NoTextError(
            f"'{filename}' does not contain enough readable text (found {document.word_count} words). "
            "It may be a scanned document or contain only images; OCR is not supported."
        )
    return document


def _looks_like_title(title: str) -> bool:
    """Filter out useless metadata titles such as 'Microsoft Word - doc1.docx'."""
    if not title or len(title) < 4 or len(title) > 200:
        return False
    lowered = title.lower()
    return not (lowered.startswith("microsoft word") or lowered.endswith((".doc", ".docx", ".pdf", ".tex")) or lowered == "untitled")
