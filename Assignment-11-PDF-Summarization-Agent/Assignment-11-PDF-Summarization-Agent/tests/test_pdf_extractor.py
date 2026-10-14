"""Tests for upload validation and PDF text extraction."""

from __future__ import annotations

import pymupdf
import pytest

from app.pdf_extractor import (
    CorruptPDFError,
    EmptyFileError,
    EncryptedPDFError,
    FileTooLargeError,
    InvalidFileTypeError,
    NoFileError,
    NoTextError,
    TooManyPagesError,
    _join_block_lines,
    extract_pdf,
    validate_upload,
)
from tests.conftest import LOREM

MB = 1024 * 1024


@pytest.mark.parametrize(
    "filename, data, error",
    [
        (None, b"%PDF-1.7", NoFileError),
        ("", b"%PDF-1.7", NoFileError),
        ("notes.txt", b"%PDF-1.7", InvalidFileTypeError),
        ("empty.pdf", b"", EmptyFileError),
        ("big.pdf", b"%PDF-" + b"0" * (2 * MB), FileTooLargeError),
        ("fake.pdf", b"just some text pretending to be a pdf", InvalidFileTypeError),
    ],
)
def test_validate_upload_rejects_bad_files(filename, data, error):
    with pytest.raises(error):
        validate_upload(filename, data, max_bytes=1 * MB)


def test_validate_upload_accepts_pdf_with_uppercase_extension():
    validate_upload("REPORT.PDF", b"%PDF-1.7\n...", max_bytes=MB)


def test_extracts_text_and_removes_headers_and_page_numbers(make_pdf):
    data = make_pdf([LOREM * 3, LOREM * 3, LOREM * 3], header="ACME Quarterly Report", footer=True)
    doc = extract_pdf(data, "report.pdf", max_pages=10, min_words=20)
    assert doc.page_count == 3
    assert "Solar panels convert sunlight" in doc.text
    assert "ACME Quarterly Report" not in doc.text
    assert "Page 2 of 3" not in doc.text
    assert doc.word_count == 3 * 3 * len(LOREM.split())


def test_hyphenation_across_lines_is_joined(make_pdf):
    data = make_pdf(["This was described in the infor-\nmation sheet. " + LOREM])
    text = extract_pdf(data, "x.pdf", max_pages=5, min_words=5).text
    assert "information sheet" in text


def test_ligatures_and_whitespace_are_normalised():
    # The PDF base fonts cannot draw ligature glyphs, so test the clean-up function directly.
    assert _join_block_lines("The eﬃcient system was\nﬁrst  used") == "The efficient system was first used"


def test_encrypted_pdf_is_rejected():
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "secret " * 50)
    data = doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="pw", owner_pw="owner")
    with pytest.raises(EncryptedPDFError):
        extract_pdf(data, "locked.pdf", max_pages=5, min_words=5)


def test_image_only_pdf_has_no_text():
    doc = pymupdf.open()
    page = doc.new_page()
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 50, 50), False)
    pix.set_rect(pix.irect, (200, 200, 200))
    page.insert_image(page.rect, pixmap=pix)
    with pytest.raises(NoTextError, match="scanned"):
        extract_pdf(doc.tobytes(), "scan.pdf", max_pages=5, min_words=5)


def test_corrupt_pdf_is_rejected():
    with pytest.raises(CorruptPDFError):
        extract_pdf(b"%PDF-1.7\nthis is not really a pdf body", "broken.pdf", max_pages=5, min_words=5)


def test_too_many_pages(make_pdf):
    with pytest.raises(TooManyPagesError):
        extract_pdf(make_pdf([LOREM] * 4), "long.pdf", max_pages=3, min_words=5)


def test_metadata_title_is_used_when_meaningful(make_pdf):
    doc = pymupdf.open(stream=make_pdf([LOREM]), filetype="pdf")
    doc.set_metadata({"title": "Solar Market Review"})
    extracted = extract_pdf(doc.tobytes(), "x.pdf", max_pages=5, min_words=5)
    assert extracted.title == "Solar Market Review"

    doc.set_metadata({"title": "Microsoft Word - draft3.docx"})
    assert extract_pdf(doc.tobytes(), "x.pdf", max_pages=5, min_words=5).title is None
