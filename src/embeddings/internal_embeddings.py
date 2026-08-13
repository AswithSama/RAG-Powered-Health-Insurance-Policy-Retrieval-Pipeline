from __future__ import annotations

import json

from sentence_transformers import SentenceTransformer

from src.config import (
    POLICY_CHUNKS_PATH,
    STRUCTURED_SECTIONS_PATH,
    INTERNAL_EMBEDDINGS_PATH,
    EMBEDDING_MODEL,
)


def load_chunks() -> list[dict]:
    if not POLICY_CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"Could not find: {POLICY_CHUNKS_PATH.resolve()}"
        )

    data = json.loads(POLICY_CHUNKS_PATH.read_text(encoding="utf-8"))
    return data["chunks"]


def load_sections() -> list[dict]:
    if not STRUCTURED_SECTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Could not find: {STRUCTURED_SECTIONS_PATH.resolve()}"
        )

    data = json.loads(STRUCTURED_SECTIONS_PATH.read_text(encoding="utf-8"))
    return data["sections"]


def main():
    chunks = load_chunks()
    sections = load_sections()

    print(f"Loaded {len(chunks)} chunks.")
    print(f"Loaded {len(sections)} sections.")

    section_lookup = {
        section["section_id"]: section
        for section in sections
    }

    internal_units = []

    # Create internal narrative units
    for chunk in chunks:
        chunk_id = chunk["chunk_id"]

        for ie_number, section_id in enumerate(chunk["section_ids"], start=1):
            section = section_lookup[section_id]

            internal_id = f"{chunk_id}_ie_{ie_number:03d}"

            internal_units.append(
                {
                    "internal_id": internal_id,
                    "chunk_id": chunk_id,
                    "ie_heading": section["heading"],
                    "ie_content": section["content"],
                }
            )

    print(f"Created {len(internal_units)} internal narrative units.")

    # Load embedding model
    print(f"Loading embedding model: {EMBEDDING_MODEL}")

    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [unit["ie_content"] for unit in internal_units]

    print("Creating internal embeddings...")

    embeddings = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    records = []

    for unit, embedding in zip(internal_units, embeddings):
        records.append(
            {
                "internal_id": unit["internal_id"],
                "chunk_id": unit["chunk_id"],
                "ie_heading": unit["ie_heading"],
                "ie_content": unit["ie_content"],
                "embedding": embedding.tolist(),
            }
        )

    output = {
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimension": len(embeddings[0]),
        "total_internal_embeddings": len(records),
        "records": records,
    }

    INTERNAL_EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)

    INTERNAL_EMBEDDINGS_PATH.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nInternal embedding creation complete.")
    print(f"Total internal embeddings: {len(records)}")
    print(f"Embedding dimension: {len(embeddings[0])}")
    print(f"Saved embeddings to: {INTERNAL_EMBEDDINGS_PATH.resolve()}")


if __name__ == "__main__":
    main()