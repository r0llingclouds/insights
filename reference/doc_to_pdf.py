"""CLI for converting documents and webpages to text using Docling."""

import argparse
import logging
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

from docling.document_converter import DocumentConverter

# Only show warnings from docling, not info spam
logging.getLogger("docling").setLevel(logging.WARNING)


def get_default_output(input_path: str) -> Path:
    """Generate default output filename from input (same directory as input)."""
    if input_path.startswith(("http://", "https://")):
        # For URLs, output to ~/Downloads
        parsed = urlparse(input_path)
        name = parsed.path.rstrip("/").split("/")[-1]
        name = name.split("?")[0]  # Remove query params
        if not name or name == "/":
            # Use domain name if no path
            name = parsed.netloc.replace(".", "_")
        # Remove existing extension and add .txt
        if "." in name:
            name = name.rsplit(".", 1)[0]
        return Path.home() / "Downloads" / f"{name}.txt"
    else:
        # For local files, output next to input
        input_file = Path(input_path)
        return input_file.with_suffix(".txt")


def convert_url_with_firecrawl(url: str) -> str:
    """Convert URL to text using Firecrawl."""
    api_key = os.environ.get("FIRECRAWL_API_KEY")
    if not api_key:
        raise ValueError(
            "FIRECRAWL_API_KEY environment variable is required for firecrawl backend"
        )

    from firecrawl import Firecrawl

    client = Firecrawl(api_key=api_key)
    doc = client.scrape(url, formats=["markdown"])
    return doc.markdown


def main() -> None:
    """Main entry point for the doc2txt CLI."""
    parser = argparse.ArgumentParser(
        prog="doc2txt",
        description="Convert documents (PDF, DOCX, PPTX, HTML) and webpages to text using Docling",
    )
    parser.add_argument(
        "input",
        help="Path to document file or URL (supports PDF, DOCX, PPTX, HTML, images, and webpages)",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output file path (default: <input>.txt)",
        type=Path,
    )
    parser.add_argument(
        "-u",
        "--url-backend",
        choices=["docling", "firecrawl"],
        default="docling",
        help="Backend for URL conversion: docling (default) or firecrawl",
    )

    args = parser.parse_args()
    output_path = args.output or get_default_output(args.input)

    try:
        print(f"Converting: {args.input}")
        is_url = args.input.startswith(("http://", "https://"))

        if is_url and args.url_backend == "firecrawl":
            text = convert_url_with_firecrawl(args.input)
        else:
            converter = DocumentConverter()
            result = converter.convert(args.input)
            print("Extracting text...")
            text = result.document.export_to_markdown()

        output_path.write_text(text, encoding="utf-8")
        print(f"Done: {output_path}")

    except FileNotFoundError:
        print(f"Error: File not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
