from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from docling_core.types.doc import (
    DoclingDocument,
    TableItem,
)

MANUAL_HEADING_ALIASES = {
    "Special coverage options after your coverage ends":
        "Special coverage options after your coverage ends "
        "When coverage may continue under the plan"
}

HEADINGS_PATH = Path("pymudpdf/headings_by_chapter.json")
DOCLING_PATH = Path("docling/aetna_docling.json")
OUTPUT_PATH = Path("structured_sections.json")


def normalize_text(text: str) -> str:
    """Normalize text so PyMuPDF and Docling headings can be matched."""

    text = text.replace("\u00ad", "")

    # Normalize different dash characters.
    text = text.replace("–", "-")
    text = text.replace("—", "-")
    text = text.replace("−", "-")

    text = re.sub(r"\s+", " ", text)

    return text.strip().casefold()


def get_page_number(item) -> int | None:
    """Return the physical PDF page number from Docling provenance."""
    prov = getattr(item, "prov", None)

    if not prov:
        return None

    return prov[0].page_no


def get_label(item) -> str:
    """Return Docling label as a plain lowercase string."""
    label = getattr(item, "label", "")

    if hasattr(label, "value"):
        return str(label.value).lower()

    return str(label).lower()


def is_body_item(item) -> bool:
    """
    Ignore page headers/footers and other document furniture.

    Docling distinguishes body content from furniture.
    """
    layer = getattr(item, "content_layer", None)

    if layer is None:
        return True

    value = getattr(layer, "value", layer)

    return str(value).lower() == "body"


def clean_list_text(text: str) -> str:
    """Remove PDF bullet artifacts because Markdown supplies its own bullet."""
    text = text.replace("", "")
    text = text.replace("\uf0b7", "")
    text = text.strip()

    return text


def render_item(item, doc: DoclingDocument) -> str | None:
    """
    Convert one Docling element into embedding-friendly Markdown.

    Main-heading boundaries are handled elsewhere.
    This function only represents content inside a section.
    """

    # Preserve real tables as Markdown tables.
    if isinstance(item, TableItem):
        return item.export_to_markdown(doc=doc).strip()

    text = getattr(item, "text", None)

    if not text:
        return None

    text = text.strip()

    if not text:
        return None

    label = get_label(item)

    # Preserve internal/sub headings, but they DO NOT create sections.
    if label == "section_header":
        return f"### {text}"

    # Preserve list semantics.
    if label == "list_item":
        text = clean_list_text(text)
        return f"- {text}"

    # Everything else is narrative text.
    return text


def load_heading_anchors() -> list[dict[str, Any]]:
    """
    Flatten headings_by_chapter.json into one ordered list of anchors.
    """

    data = json.loads(
        HEADINGS_PATH.read_text(encoding="utf-8")
    )

    anchors = []
    anchor_id = 0

    for chapter, headings in data.items():
        for heading_index, heading in enumerate(headings, start=1):
            anchor_id+=1
            anchors.append(
                {
                    "anchor_id": anchor_id,
                    "chapter": chapter,
                    "heading_index": heading_index,
                    **heading,
                }
            )

    return anchors


def build_anchor_lookup(
    anchors: list[dict[str, Any]],
) -> dict[tuple[int, str], dict[str, Any]]:

    lookup = {}

    for anchor in anchors:
        heading = anchor["heading"]

        # Use manual Docling representation when necessary.
        match_text = MANUAL_HEADING_ALIASES.get(heading,heading,)

        key = (anchor["pdf_page"],normalize_text(match_text), )

        lookup[key] = anchor

    return lookup

def build_sections(
    doc: DoclingDocument,
    anchors: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], set[int]]:

    anchor_lookup = build_anchor_lookup(anchors)

    sections = []
    matched_anchor_ids = set()

    current_section = None

    for item, _level in doc.iterate_items():

        if not is_body_item(item):
            continue

        page_no = get_page_number(item)

        if page_no is None:
            continue

        text = getattr(item, "text", None)

        matched_anchor = None

        if text:
            key = (
                page_no,
                normalize_text(text),
            )

            matched_anchor = anchor_lookup.get(key)

        if matched_anchor is not None:

            matched_anchor_ids.add(
                matched_anchor["anchor_id"]
            )

            if current_section is not None:
                sections.append(current_section)

            current_section = {
                "chapter": matched_anchor["chapter"],
                "heading_index": matched_anchor["heading_index"],
                "heading": matched_anchor["heading"],
                "start_pdf_page": matched_anchor["pdf_page"],
                "end_pdf_page": matched_anchor["pdf_page"],
                "chapter_end_pdf_page":
                    matched_anchor["chapter_end_pdf_page"],
                "elements": [],
            }

            continue

        if current_section is None:
            continue

        if page_no > current_section["chapter_end_pdf_page"]:
            sections.append(current_section)
            current_section = None
            continue

        rendered = render_item(item, doc)

        if rendered is None:
            continue

        current_section["elements"].append(rendered)
        current_section["end_pdf_page"] = page_no

    if current_section is not None:
        sections.append(current_section)

    return sections, matched_anchor_ids

def finalize_sections(
    sections: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Turn the collected Docling elements into one clean text block
    per main heading.
    """

    final_sections = []

    for section_index, section in enumerate(sections, start=1):

        body = "\n\n".join(section["elements"]).strip()

        content = (
            f"# {section['heading']}\n\n"
            f"{body}"
        ).strip()

        final_sections.append(
            {
                "section_id": f"section_{section_index:03d}",
                "chapter": section["chapter"],
                "heading_index": section["heading_index"],
                "heading": section["heading"],
                "start_pdf_page": section["start_pdf_page"],
                "end_pdf_page": section["end_pdf_page"],
                "content": content,
            }
        )

    return final_sections


def main():

    if not HEADINGS_PATH.exists():
        raise FileNotFoundError(f"Missing heading file: {HEADINGS_PATH}")

    if not DOCLING_PATH.exists():
        raise FileNotFoundError(f"Missing Docling file: {DOCLING_PATH}")

    anchors = load_heading_anchors()

    print(f"Loaded {len(anchors)} PyMuPDF heading anchors.")
    doc = DoclingDocument.load_from_json(DOCLING_PATH)
    sections,matched_anchor_ids = build_sections(doc, anchors)

    final_sections = finalize_sections(sections)

    OUTPUT_PATH.write_text(
        json.dumps(
            {
                "total_heading_anchors": len(anchors),
                "total_sections_created": len(final_sections),
                "sections": final_sections,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"Sections created: {len(final_sections)}")

    if len(final_sections) != len(anchors):
        print(
            "WARNING: Number of sections does not equal "
            "number of heading anchors."
        )

    print(
        f"Saved structured sections to: "
        f"{OUTPUT_PATH.resolve()}"
    )

    unmatched_anchors = [anchor for anchor in anchors if anchor["anchor_id"] not in matched_anchor_ids]

    print("\nUnmatched PyMuPDF headings")
    print("=" * 70)

    for anchor in unmatched_anchors:
        print(
            f"{anchor['anchor_id']:>3} | "
            f"{anchor['chapter']} | "
            f"PDF page {anchor['pdf_page']} | "
            f"{anchor['heading']}"
        )

    print(f"\nMatched:   {len(matched_anchor_ids)}")
    print(f"Unmatched: {len(unmatched_anchors)}")


if __name__ == "__main__":
    main()