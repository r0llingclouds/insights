from __future__ import annotations

import logging


def extract_markdown_with_docling(input_path_or_url: str) -> str:
    """
    Convert a local document or URL to Markdown using Docling.
    """
    # Docling can be very chatty by default.
    logging.getLogger("docling").setLevel(logging.WARNING)
    try:
        from docling.document_converter import DocumentConverter  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Docling is required for this operation. Install dependencies with `uv sync`."
        ) from e

    converter = DocumentConverter()
    result = converter.convert(input_path_or_url)
    return result.document.export_to_markdown()


