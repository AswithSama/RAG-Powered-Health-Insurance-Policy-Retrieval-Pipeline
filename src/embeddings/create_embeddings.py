from __future__ import annotations

import json
from pathlib import Path

from sentence_transformers import SentenceTransformer


CHUNKS_PATH = Path("../data/processed/policy_chunks.json")
OUTPUT_PATH = Path("../data/processed/policy_embeddings.json")

MODEL_NAME = "BAAI/bge-base-en-v1.5"


def load_chunks() -> list[dict]:
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"Could not find: {CHUNKS_PATH.resolve()}")

    data = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))

    return data["chunks"]


def main():
    chunks = load_chunks()

    print(f"Loaded {len(chunks)} chunks.")
    print(f"Loading embedding model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    # Only the meaningful chunk content is embedded.
    texts = [chunk["content"]for chunk in chunks]

    print("Creating embeddings...")

    embeddings = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    records = []

    for chunk, embedding in zip(chunks, embeddings):

        records.append(
            {
                "chunk_id": chunk["chunk_id"],
                "chapter": chunk["chapter"],
                "headings": chunk["headings"],
                "start_pdf_page": chunk["start_pdf_page"],
                "end_pdf_page": chunk["end_pdf_page"],
                # Keep original text for retrieval/debugging.
                "content": chunk["content"],
                # Convert NumPy array into normal Python list
                # so it can be serialized to JSON.
                "embedding": embedding.tolist(),
            }
        )

    output = {
        "embedding_model": MODEL_NAME,
        "embedding_dimension": len(embeddings[0]),
        "total_chunks": len(records),
        "records": records,
    }

    OUTPUT_PATH.write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\nEmbedding creation complete.")
    print(f"Total embeddings: {len(records)}")
    print(f"Embedding dimension: {len(embeddings[0])}")
    print(
        f"Saved embeddings to: "
        f"{OUTPUT_PATH.resolve()}"
    )


if __name__ == "__main__":
    main()