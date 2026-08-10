from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import tiktoken



STRUCTURED_PATH = Path("../data/processed/structured_sections.json")
OUTPUT_PATH = Path("../data/processed/policy_chunks.json")

MAX_CHUNK_TOKENS = 750


def count_tokens(text: str) -> int:

    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))



def load_sections() -> list[dict[str, Any]]:
    """
    Load the 105 structured sections created by the
    hybrid PyMuPDF + Docling parsing pipeline.
    """
    if not STRUCTURED_PATH.exists():
        raise FileNotFoundError(
            f"Could not find: {STRUCTURED_PATH.resolve()}"
        )

    data = json.loads(STRUCTURED_PATH.read_text(encoding="utf-8"))

    return data["sections"]


def render_chunk_content(
    chapter: str,
    sections: list[dict[str, Any]],
) -> str:
    """
    Create the actual text that will later be embedded.

    Every section already contains its own main heading
    and structured Docling content.
    """
    section_content = "\n\n".join(section["content"].strip() for section in sections if section.get("content"))

    return (f"Chapter: {chapter}\n\n"f"{section_content}").strip()


def create_chunk(
    chapter: str,
    sections: list[dict[str, Any]],
    chunk_index: int,
) -> dict[str, Any]:
    """
    Convert one or more complete sections into a final chunk.
    """

    content = render_chunk_content(chapter,sections,)

    token_count = count_tokens(content)

    return {
        "chunk_id": f"{chapter_slug(chapter)}_chunk_{chunk_index:03d}",
        "chapter": chapter,
        "chunk_index_in_chapter": chunk_index,
        "headings": [section["heading"] for section in sections],
        "section_ids": [section["section_id"] for section in sections],
        "section_count": len(sections),
        "start_pdf_page": min(section["start_pdf_page"] for section in sections),
        "end_pdf_page": max(section["end_pdf_page"] for section in sections),
        "token_count": token_count,
        # This is allowed when one full section itself
        # is larger than MAX_CHUNK_TOKENS.
        "exceeds_max_tokens": token_count > MAX_CHUNK_TOKENS,
        "content": content,
    }


def chapter_slug(chapter: str) -> str:
    """
    Convert chapter name into a simple chunk-id-safe string.
    """
    slug = chapter.casefold()

    replacements = {
        " ": "_",
        "-": "_",
        "–": "_",
        "—": "_",
        ",": "",
        ":": "",
        "/": "_",
    }

    for old, new in replacements.items():
        slug = slug.replace(old, new)

    while "__" in slug:
        slug = slug.replace("__", "_")

    return slug.strip("_")


def group_sections_by_chapter(
    sections: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """
    Preserve original section order while grouping by chapter.
    """

    chapters: dict[str, list[dict[str, Any]]] = {}

    for section in sections:
        chapter = section["chapter"]

        if chapter not in chapters:
            chapters[chapter] = []

        chapters[chapter].append(section)

    return chapters


def chunk_chapter(chapter: str, sections: list[dict[str, Any]],) -> list[dict[str, Any]]:
    """
    Greedily combine COMPLETE sections.

    Rules:

    1. Never split a section.
    2. Never cross chapter boundaries.
    3. Every chunk has at least one full section.
    4. Add sections while the resulting chunk stays <= 750 tokens.
    5. If one section itself exceeds 750 tokens, keep the entire section as one oversized chunk.
    """

    chunks: list[dict[str, Any]] = []

    current_sections: list[dict[str, Any]] = []

    for section in sections:

        # -------------------------------------------------
        # Start a new chunk if nothing is currently grouped.
        # -------------------------------------------------

        if not current_sections:
            current_sections = [section]

            # Check whether this single section is already
            # larger than our maximum.
            single_content = render_chunk_content(chapter,current_sections,)

            single_tokens = count_tokens(single_content)

            # If the section itself is oversized,
            # immediately save it as its own chunk.
            if single_tokens > MAX_CHUNK_TOKENS:
                chunks.append(create_chunk(chapter,current_sections,len(chunks) + 1,))
                current_sections = []

            continue

        # -------------------------------------------------
        # Try adding the next COMPLETE section.
        # -------------------------------------------------

        candidate_sections = (current_sections + [section])

        candidate_content = render_chunk_content(chapter,candidate_sections)

        candidate_tokens = count_tokens(candidate_content)

        # -------------------------------------------------
        # If the new complete section still fits,
        # keep growing the current chunk.
        # -------------------------------------------------

        if candidate_tokens <= MAX_CHUNK_TOKENS:
            current_sections.append(section)
            continue

        # -------------------------------------------------
        # Otherwise finish the current chunk.
        # -------------------------------------------------

        chunks.append(create_chunk(chapter,current_sections,len(chunks) + 1,))

        # Start the next chunk with the new section.
        current_sections = [section]

        # -------------------------------------------------
        # If this new section alone exceeds 750,
        # keep it whole as its own oversized chunk.
        # -------------------------------------------------

        single_content = render_chunk_content(chapter, current_sections)

        single_tokens = count_tokens(single_content)

        if single_tokens > MAX_CHUNK_TOKENS:
            chunks.append(create_chunk(chapter,current_sections,len(chunks) + 1,))
            current_sections = []

    # -----------------------------------------------------
    # Save any final sections remaining in the buffer.
    # -----------------------------------------------------

    if current_sections:
        chunks.append(create_chunk(chapter,current_sections,len(chunks) + 1,))

    return chunks


def build_chunks(sections: list[dict[str, Any]],) -> dict[str, Any]:
    """
    Build all chunks chapter by chapter.
    """

    chapters = group_sections_by_chapter(sections)

    all_chunks = []
    chapter_summaries = []

    for chapter, chapter_sections in chapters.items():

        chapter_chunks = chunk_chapter(chapter,chapter_sections,)

        all_chunks.extend(chapter_chunks)

        chapter_summaries.append(
            {
                "chapter": chapter,
                "section_count": len(chapter_sections),
                "chunk_count": len(chapter_chunks),
                "oversized_chunks": sum(
                    1
                    for chunk in chapter_chunks
                    if chunk["exceeds_max_tokens"]
                ),
                "total_tokens": sum(
                    chunk["token_count"]
                    for chunk in chapter_chunks
                ),
            }
        )

    return {
        "chunking_config": {
            "max_chunk_tokens": MAX_CHUNK_TOKENS,
            "section_splitting_allowed": False,
            "cross_chapter_chunks_allowed": False,
            "oversized_sections_allowed": True,
            "atomic_unit": ("one complete 14-point main-heading section"),
        },

        "total_sections": len(sections),
        "total_chunks": len(all_chunks),
        "chapter_summaries": chapter_summaries,
        "chunks": all_chunks,
    }


def print_summary(result: dict[str, Any]) -> None:
    """
    Display an easy-to-read chunking report.
    """

    print("\nChunking summary")
    print("=" * 90)

    for chapter in result["chapter_summaries"]:

        print(f"\n{chapter['chapter']}")
        print(f"  Sections:          "f"{chapter['section_count']}")
        print(f"  Chunks:            "f"{chapter['chunk_count']}")
        print(f"  Oversized chunks:  "f"{chapter['oversized_chunks']}")
        print(f"  Total tokens:      "f"{chapter['total_tokens']}")

    print("\n" + "-" * 90)
    print(f"Total sections: {result['total_sections']}")
    print(f"Total chunks:   {result['total_chunks']}")

    oversized = [
        chunk
        for chunk in result["chunks"]
        if chunk["exceeds_max_tokens"]
    ]

    print(f"Oversized chunks: {len(oversized)}")

    if oversized:
        print("\nOversized sections/chunks:")
        for chunk in oversized:
            print(
                f"  {chunk['chunk_id']} | "
                f"{chunk['token_count']} tokens | "
                f"{', '.join(chunk['headings'])}"
            )


def main():
    sections = load_sections()
    print(f"Loaded {len(sections)} structured sections.")

    result = build_chunks(sections)

    OUTPUT_PATH.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print_summary(result)
    print(f"\nSaved chunks to: "f"{OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()