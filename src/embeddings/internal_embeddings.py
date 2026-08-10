from __future__ import annotations

import json
from pathlib import Path

from sentence_transformers import SentenceTransformer


CHUNKS_PATH = Path("./data/processed/policy_chunks.json")
SECTIONS_PATH = Path("./data/processed/structured_sections.json")
OUTPUT_PATH = Path("./data/processed/policy_internal_embeddings.json")

MODEL_NAME = "BAAI/bge-base-en-v1.5"


def load_chunks() -> list[dict]:
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"Could not find: {CHUNKS_PATH.resolve()}")

    data = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    return data["chunks"]


def load_sections() -> list[dict]:
    if not SECTIONS_PATH.exists():
        raise FileNotFoundError(f"Could not find: {SECTIONS_PATH.resolve()}")

    data = json.loads(SECTIONS_PATH.read_text(encoding="utf-8"))
    return data["sections"]


def main():
    chunks = load_chunks()
    sections = load_sections()

    print(f"Loaded {len(chunks)} chunks.")
    print(f"Loaded {len(sections)} sections.")

    # Fast lookup:
    # section_001 -> section data
    section_lookup = {
        section["section_id"]: section
        for section in sections
    }

    internal_units = []

    # --------------------------------------------------
    # Create internal narrative units
    # --------------------------------------------------

    for chunk_number, chunk in enumerate(chunks, start=1):
        # Temporary simple chunk ID:
        # 001, 002, 003, ...
        chunk_id = f"{chunk_number:03d}"

        section_ids = chunk["section_ids"]

        for ie_number, section_id in enumerate(section_ids, start=1):
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

    # --------------------------------------------------
    # Load embedding model
    # --------------------------------------------------

    print(f"Loading embedding model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    # Only narrative content is embedded.
    texts = [unit["ie_content"] for unit in internal_units]

    print("Creating internal embeddings...")

    embeddings = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    # --------------------------------------------------
    # Attach embeddings
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    output = {
        "embedding_model": MODEL_NAME,
        "embedding_dimension": len(embeddings[0]),
        "total_internal_embeddings": len(records),
        "records": records,
    }

    OUTPUT_PATH.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nInternal embedding creation complete.")
    print(f"Total internal embeddings: {len(records)}")
    print(f"Embedding dimension: {len(embeddings[0])}")
    print(f"Saved embeddings to: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()