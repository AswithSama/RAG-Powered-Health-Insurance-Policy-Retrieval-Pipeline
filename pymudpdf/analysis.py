import json
import re
from pathlib import Path

import pymupdf as fitz  # PyMuPDF


PDF_PATH = Path("aetna-2500-4000-full-plan.pdf")
OUTPUT_PATH = Path("headings_by_chapter.json")


# Page numbers below are the printed page numbers shown in the table
# of contents, not zero-based PDF indexes.
#
# For this PDF:
#     PDF index = printed page number + 1
#
# Example:
#     printed page 3 -> PDF index 4 -> fifth physical PDF page
CHAPTERS = [
    {
        "name": "Coverage and exclusions",
        "start_printed_page": 3,
        "end_printed_page": 29,
    },
    {
        "name": "General plan exclusions",
        "start_printed_page": 30,
        "end_printed_page": 34,
    },
    {
        "name": "How your plan works",
        "start_printed_page": 35,
        "end_printed_page": 48,
    },
    {
        "name": "Complaints, claim decisions and appeals procedures",
        "start_printed_page": 49,
        "end_printed_page": 52,
    },
    {
        "name": "Eligibility, starting and stopping coverage",
        "start_printed_page": 53,
        "end_printed_page": 56,
    },
    {
        "name": "General provisions – other things you should know",
        "start_printed_page": 57,
        "end_printed_page": 61,
    },
]


def printed_page_to_pdf_index(printed_page: int) -> int:
    """
    Convert the booklet's printed page number to a zero-based PDF index.

    In this document:
        printed page 1 is physical PDF page 3, which has index 2.
        Therefore PDF index = printed page + 1.
    """
    return printed_page + 1


def clean_text(text: str) -> str:
    """Normalize spacing and remove stray bullet characters."""
    text = text.replace("\u00ad", "")
    text = text.replace("\uf0b7", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_bold(span: dict) -> bool:
    """
    Determine whether a PDF text span is bold.

    PyMuPDF flag 16 normally indicates bold text. Checking the font
    name provides an additional fallback.
    """
    font_name = span.get("font", "").lower()
    flags = span.get("flags", 0)

    return bool(flags & 16) or "bold" in font_name


def extract_large_bold_lines(
    page: fitz.Page,
    minimum_font_size: float = 13.5,
    maximum_font_size: float = 15.0,
) -> list[dict]:
    """
    Extract probable main headings from one page.

    This document primarily uses:
        16 pt bold -> chapter title
        14 pt bold -> main heading
        12 pt bold -> subheading
        11 pt      -> body text

    Restricting the range to approximately 14 pt retrieves the
    main headings rather than every bold phrase.
    """
    page_data = page.get_text("dict")
    candidates = []

    for block in page_data.get("blocks", []):
        if "lines" not in block:
            continue

        for line in block["lines"]:
            heading_spans = []

            for span in line.get("spans", []):
                text = clean_text(span.get("text", ""))
                font_size = float(span.get("size", 0))

                if not text:
                    continue

                if (
                    minimum_font_size <= font_size <= maximum_font_size
                    and is_bold(span)
                ):
                    heading_spans.append(span)

            if not heading_spans:
                continue

            # Only join the spans that have heading-sized text.
            # This avoids accidentally including nearby 11-point body text.
            heading_text = clean_text(
                " ".join(span["text"] for span in heading_spans)
            )

            if not heading_text:
                continue

            # Ignore isolated numbers, such as page numbers.
            if re.fullmatch(r"\d+", heading_text):
                continue

            candidates.append(
                {
                    "text": heading_text,
                    "font_size": round(
                        max(float(span["size"]) for span in heading_spans),
                        2,
                    ),
                    "y_position": round(line["bbox"][1], 2),
                }
            )

    return candidates


def extract_headings_by_chapter(pdf_path: Path) -> dict[str, list[dict]]:
    document = fitz.open(pdf_path)
    results = {}

    try:
        for chapter in CHAPTERS:
            chapter_name = chapter["name"]

            start_index = printed_page_to_pdf_index(
                chapter["start_printed_page"]
            )
            end_index = printed_page_to_pdf_index(
                chapter["end_printed_page"]
            )

            chapter_headings = []
            seen = set()

            for page_index in range(start_index, end_index + 1):
                if page_index >= len(document):
                    break

                page = document[page_index]

                for candidate in extract_large_bold_lines(page):
                    heading = candidate["text"]

                    # Avoid including the chapter title itself.
                    if heading.casefold() == chapter_name.casefold():
                        continue

                    # Remove duplicated headings, if any.
                    normalized = heading.casefold()

                    if normalized in seen:
                        continue

                    seen.add(normalized)

                    chapter_headings.append(
                        {
                            "heading": heading,
                            "printed_page": page_index - 1,
                            "pdf_page": page_index + 1,
                            "font_size": candidate["font_size"],
                            "y_position": candidate["y_position"],

                            # Useful later when combining with Docling
                            "chapter_start_pdf_page": start_index + 1,
                            "chapter_end_pdf_page": end_index + 1,
                        }
                    )

            results[chapter_name] = chapter_headings

    finally:
        document.close()

    return results


def main() -> None:
    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"PDF file was not found: {PDF_PATH.resolve()}"
        )

    headings = extract_headings_by_chapter(PDF_PATH)

    OUTPUT_PATH.write_text(
        json.dumps(headings, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    for chapter, chapter_headings in headings.items():
        print(f"\n{'=' * 70}")
        print(chapter)
        print("=" * 70)

        for item in chapter_headings:
            print(
                f"[Printed page {item['printed_page']:>2}] "
                f"{item['heading']}"
            )

    total_headings = sum(
        len(chapter_headings)
        for chapter_headings in headings.values()
    )

    print(f"\nTotal main headings detected: {total_headings}")
    print(f"\nSaved structured output to: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()